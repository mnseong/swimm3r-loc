#!/usr/bin/env python3
"""
Underwat3R inference with COLMAP-format output.

Accepts a video file OR a directory of images, runs MASt3R-SfM (with optional
LoRA — Dark3R, Underwat3R v1, etc.), and exports the result as a standard
COLMAP sparse reconstruction (cameras/images/points3D in both .txt and .bin)
plus the input frames under `images/`. The output layout is directly
loadable by the COLMAP GUI, NeRF Studio, viser, hloc, etc.

Reuses inference.reconstruction.{load_model, prepare_images,
create_image_pairs, run_reconstruction} so it stays in lock-step with
forward_dark3r.py — same matcher, same global alignment, same checkpoints.

Examples
--------
    # Underwater video (default — sliding window), Underwat3R v7 LoRA
    python forward_dark3r_colmap.py \\
        --video /data/.../dive.mp4 --fps 5 \\
        --outdir results/colmap_v7_dive \\
        --finetuned_model results/underwat3r_v7_physics/weights/0009_lora.pth \\
        --winsize 5

    # Image directory of an underwater video sequence
    python forward_dark3r_colmap.py \\
        --imagedir /data/.../video_frames \\
        --outdir results/colmap_v7_dive \\
        --finetuned_model results/underwat3r_v7_physics/weights/0009_lora.pth \\
        --winsize 5

    # Loop-trajectory (back to start) video — add --win_cyclic
    python forward_dark3r_colmap.py \\
        --imagedir /data/.../loop_dive \\
        --outdir results/colmap_v7_loop \\
        --finetuned_model results/underwat3r_v7_physics/weights/0009_lora.pth \\
        --winsize 5 --win_cyclic

    # Unordered photo collection / lawn-mower survey w/ revisits — use retrieval
    python forward_dark3r_colmap.py \\
        --imagedir /data/.../survey \\
        --outdir results/colmap_survey \\
        --finetuned_model results/underwat3r_v7_physics/weights/0009_lora.pth \\
        --scenegraph_type retrieval --winsize 15 --refid 10

    # Re-export COLMAP from cached scene.pkl
    python forward_dark3r_colmap.py \\
        --imagedir /data/.../video_frames \\
        --outdir results/colmap_v7_dive \\
        --reuse_scene_pkl results/colmap_v7_dive/scene.pkl
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pickle
import shutil
import struct
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from inference.reconstruction import (
    load_model,
    prepare_images,
    create_image_pairs,
    run_reconstruction,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# -----------------------------------------------------------------------------
# Input handling: video or image directory
# -----------------------------------------------------------------------------

def extract_frames(video_path: str, outdir: str, fps: float) -> list[str]:
    """Sample frames from a video at a given fps and save under outdir/images/."""
    images_dir = os.path.join(outdir, "images")
    os.makedirs(images_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / video_fps if video_fps > 0 else 0.0
    interval = video_fps / fps if fps > 0 else 1.0

    print(f"[video] {video_path}")
    print(f"        {video_fps:.1f} fps, {total} frames, {duration:.1f}s")
    print(f"        sampling at {fps} fps (every {interval:.1f} input frames)")

    saved: list[str] = []
    idx = 0
    next_sample = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= next_sample:
            fname = f"{len(saved):06d}.jpg"
            cv2.imwrite(os.path.join(images_dir, fname), frame)
            saved.append(os.path.join(images_dir, fname))
            next_sample += interval
        idx += 1
    cap.release()
    print(f"        extracted {len(saved)} frames -> {images_dir}")
    return saved


def collect_images(imagedir: str, outdir: str) -> list[str]:
    """Copy all images from imagedir into outdir/images/ and return their paths."""
    images_dir = os.path.join(outdir, "images")
    os.makedirs(images_dir, exist_ok=True)
    saved: list[str] = []
    for name in sorted(os.listdir(imagedir)):
        if Path(name).suffix.lower() not in IMAGE_EXTS:
            continue
        src = os.path.join(imagedir, name)
        dst = os.path.join(images_dir, name)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        saved.append(dst)
    if not saved:
        raise FileNotFoundError(f"No images found in {imagedir}")
    print(f"[imagedir] copied {len(saved)} images -> {images_dir}")
    return saved


# -----------------------------------------------------------------------------
# COLMAP export
# -----------------------------------------------------------------------------

def _rotmat_to_qvec_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> COLMAP quaternion order (qw, qx, qy, qz)."""
    q_xyzw = Rotation.from_matrix(R).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def _farthest_point_sampling(points: np.ndarray, n_samples: int) -> np.ndarray:
    """GPU-accelerated FPS via torch (falls back to CPU). Returns selected indices."""
    n_total = len(points)
    if n_samples >= n_total:
        return np.arange(n_total)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pts = torch.from_numpy(points).float().to(device)
    distances = torch.full((n_total,), float("inf"), device=device)
    indices = torch.zeros(n_samples, dtype=torch.long, device=device)
    idx = torch.randint(n_total, (1,), device=device).item()
    for i in range(n_samples):
        indices[i] = idx
        d = torch.norm(pts - pts[idx].unsqueeze(0), dim=1)
        distances = torch.minimum(distances, d)
        idx = torch.argmax(distances).item()
    return indices.cpu().numpy()


def _get_original_dimensions(images_dir: str, image_names: list[str]) -> list[tuple[int, int]]:
    """Read the (W, H) of each saved image in <outdir>/images/."""
    from PIL import Image as PILImage
    out: list[tuple[int, int]] = []
    for name in image_names:
        with PILImage.open(os.path.join(images_dir, name)) as im:
            out.append((im.width, im.height))
    return out


def export_colmap(
    scene,
    outdir: str,
    image_names: list[str],
    min_conf_thr: float = 1.5,
    fps_ratio: float | None = None,
    keep_original_resolution: bool = True,
    *,
    colors_per_frame: list[np.ndarray] | None = None,
    points3d_filename: str = "points3D",
    write_cameras_images: bool = True,
    fps_indices: np.ndarray | None = None,
    exclude_indices: list[int] | set[int] | None = None,
    fps_max_points: int | None = None,
):
    """Write outdir/sparse/0/ in COLMAP text + binary format.

    Camera poses come from scene.get_im_poses() (c2w in OpenGL convention) and
    are converted to COLMAP's w2c-OpenCV convention by inversion. 3D points
    come from scene.get_dense_pts3d(clean_depth=True) filtered by
    confidence > min_conf_thr.

    Returns:
        (all_pts, all_colors, fps_indices_used, mask_per_frame)
        - all_pts:           [N_kept, 3]   final per-point XYZ
        - all_colors:        [N_kept, 3]   final per-point RGB (uint8)
        - fps_indices_used:  None or array of indices applied at the FPS step
        - mask_per_frame:    list of bool arrays (one per frame, len H*W);
                             callers can re-use the same mask to compute aux
                             per-point arrays (β, B^∞, …) without redoing the
                             confidence threshold.

    Optional kwargs (keyword-only — backward-compatible):
        colors_per_frame:    list of [H, W, 3] uint8 arrays, one per frame.
                             Overrides scene.imgs sampling. Used by Swimm3R
                             to write the canonical points3D with descattered
                             colors.
        points3d_filename:   filename stem for the points3D files (default
                             "points3D"; pass "raw_points3D" for the auxiliary
                             raw-color copy).
        write_cameras_images: set False to skip cameras/images.{txt,bin} —
                             only points3D files are written. Used for the
                             second call when writing both raw + descattered.
        fps_indices:         when given, applied as the FPS subsample. When
                             None and fps_ratio<1, indices are computed and
                             returned for re-use across calls.

    keep_original_resolution: if True (default), the COLMAP cameras file
        records each image's ORIGINAL (W, H) (matching the JPGs that were
        copied/extracted into outdir/images/) and the MASt3R intrinsics
        (focal, principal point) are linearly rescaled from the processed
        size to that original size.
    """
    sparse = os.path.join(outdir, "sparse", "0")
    os.makedirs(sparse, exist_ok=True)

    c2w = scene.get_im_poses().detach().cpu().numpy()  # (N, 4, 4)
    focals = scene.get_focals().detach().cpu().numpy().reshape(-1)
    pp = scene.get_principal_points().detach().cpu().numpy()  # (N, 2)
    imgs = scene.imgs

    N = len(image_names)
    proc_H, proc_W = imgs[0].shape[:2]  # MASt3R processed shape

    if keep_original_resolution:
        orig_dims = _get_original_dimensions(os.path.join(outdir, "images"), image_names)
    else:
        orig_dims = [(proc_W, proc_H)] * N

    if write_cameras_images:
        # ---- cameras.txt ----
        with open(os.path.join(sparse, "cameras.txt"), "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"# Number of cameras: {N}\n")
            for i in range(N):
                ow, oh = orig_dims[i]
                sx = ow / float(proc_W)
                sy = oh / float(proc_H)
                fx = float(focals[i]) * sx
                fy = float(focals[i]) * sy
                cx = float(pp[i, 0]) * sx
                cy = float(pp[i, 1]) * sy
                f.write(f"{i + 1} PINHOLE {ow} {oh} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

        # ---- images.txt ----
        with open(os.path.join(sparse, "images.txt"), "w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write(f"# Number of images: {N}\n")
            for i in range(N):
                R_c2w = c2w[i, :3, :3]
                t_c2w = c2w[i, :3, 3]
                R_w2c = R_c2w.T
                t_w2c = -R_w2c @ t_c2w
                qvec = _rotmat_to_qvec_wxyz(R_w2c)
                f.write(
                    f"{i + 1} "
                    f"{qvec[0]:.10f} {qvec[1]:.10f} {qvec[2]:.10f} {qvec[3]:.10f} "
                    f"{t_w2c[0]:.10f} {t_w2c[1]:.10f} {t_w2c[2]:.10f} "
                    f"{i + 1} {image_names[i]}\n\n"
                )

    # ---- points3D ----
    pts3d_list, _, confs = scene.get_dense_pts3d(clean_depth=True)
    all_pts: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    n_before = 0
    n_kept = 0
    for i in range(N):
        pts = pts3d_list[i]
        conf = confs[i]
        if hasattr(pts, "cpu"):
            pts = pts.cpu().numpy()
        if hasattr(conf, "cpu"):
            conf = conf.cpu().numpy()
        pts = pts.reshape(-1, 3)
        conf = conf.reshape(-1)

        if colors_per_frame is not None:
            img = colors_per_frame[i]
        else:
            img = imgs[i]
            if img.max() <= 1.0:
                img = (img * 255).clip(0, 255).astype(np.uint8)
        colors = img.reshape(-1, 3)

        n_before += len(conf)
        mask = conf > min_conf_thr
        # InstantSplat fair-evaluation: cameras listed in `exclude_indices`
        # still get their poses written to COLMAP (test view registration),
        # but their per-pixel points are *not* contributed to the merged
        # point cloud — so the point cloud is train-only while the camera
        # set is train + test.
        if exclude_indices is not None and i in exclude_indices:
            mask = np.zeros_like(mask, dtype=bool)
        n_kept += int(mask.sum())
        all_pts.append(pts[mask])
        all_colors.append(colors[mask])
        masks.append(mask)

    all_pts = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3))
    all_colors = np.concatenate(all_colors, axis=0) if all_colors else np.zeros((0, 3), dtype=np.uint8)
    if write_cameras_images:
        print(f"[points] confidence > {min_conf_thr}: {n_kept:,} / {n_before:,}")

    # Determine target point count from ratio or absolute cap
    n_total = len(all_pts)
    n_target = None
    if fps_ratio is not None and 0 < fps_ratio < 1.0 and n_total > 0:
        n_target = int(n_total * fps_ratio)
    if fps_max_points is not None and n_total > fps_max_points:
        n_target = min(n_target, fps_max_points) if n_target else fps_max_points

    if n_target is not None and n_target < n_total:
        if fps_indices is None:
            print(f"[points] FPS downsample: {n_total:,} -> {n_target:,}")
            fps_indices = _farthest_point_sampling(all_pts, n_target)
        all_pts = all_pts[fps_indices]
        all_colors = all_colors[fps_indices]

    # ---- points3D.txt with custom filename ----
    with open(os.path.join(sparse, f"{points3d_filename}.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(all_pts)}\n")
        for j in range(len(all_pts)):
            x, y, z = all_pts[j]
            r, g, b = (int(all_colors[j, 0]), int(all_colors[j, 1]), int(all_colors[j, 2]))
            f.write(f"{j + 1} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0.0\n")

    # ---- binary mirror ----
    _write_colmap_binary(
        sparse, N, proc_W, proc_H, orig_dims, focals, pp, c2w,
        image_names, all_pts, all_colors,
        write_cameras_images=write_cameras_images,
        points3d_filename=points3d_filename,
    )

    if write_cameras_images:
        print(f"[colmap] wrote {sparse}/{{cameras,images,{points3d_filename}}}.{{txt,bin}}")
    else:
        print(f"[colmap] wrote {sparse}/{points3d_filename}.{{txt,bin}}")

    return all_pts, all_colors, fps_indices, masks


def _write_colmap_binary(
    sparse_dir: str,
    N: int,
    proc_W: int,
    proc_H: int,
    orig_dims: list[tuple[int, int]],
    focals: np.ndarray,
    pp: np.ndarray,
    c2w: np.ndarray,
    image_names: list[str],
    all_pts: np.ndarray,
    all_colors: np.ndarray,
    *,
    write_cameras_images: bool = True,
    points3d_filename: str = "points3D",
) -> None:
    """Write COLMAP binary format alongside the .txt files.

    When write_cameras_images=False, only points3D.bin is written (used by
    swimm3r_colmap.py for the second 'descattered' export call after the
    first 'raw_points3D' call already produced cameras + images bin files).
    """
    if write_cameras_images:
        # cameras.bin
        with open(os.path.join(sparse_dir, "cameras.bin"), "wb") as f:
            f.write(struct.pack("<Q", N))
            for i in range(N):
                ow, oh = orig_dims[i]
                sx = ow / float(proc_W)
                sy = oh / float(proc_H)
                fx = float(focals[i]) * sx
                fy = float(focals[i]) * sy
                cx = float(pp[i, 0]) * sx
                cy = float(pp[i, 1]) * sy
                f.write(struct.pack("<iiQQ", i + 1, 1, ow, oh))  # model_id 1 = PINHOLE
                f.write(struct.pack("<4d", fx, fy, cx, cy))

        # images.bin
        with open(os.path.join(sparse_dir, "images.bin"), "wb") as f:
            f.write(struct.pack("<Q", N))
            for i in range(N):
                R_c2w = c2w[i, :3, :3]
                t_c2w = c2w[i, :3, 3]
                R_w2c = R_c2w.T
                t_w2c = -R_w2c @ t_c2w
                qvec = _rotmat_to_qvec_wxyz(R_w2c)
                f.write(struct.pack("<i", i + 1))
                f.write(struct.pack("<4d", *qvec))
                f.write(struct.pack("<3d", *t_w2c))
                f.write(struct.pack("<i", i + 1))
                name_bytes = image_names[i].encode("utf-8") + b"\x00"
                f.write(name_bytes)
                f.write(struct.pack("<Q", 0))  # zero 2D-3D track entries

    # points3D.bin (always written, with custom filename)
    n_pts = len(all_pts)
    with open(os.path.join(sparse_dir, f"{points3d_filename}.bin"), "wb") as f:
        f.write(struct.pack("<Q", n_pts))
        for j in range(n_pts):
            x, y, z = all_pts[j]
            r, g, b = (int(all_colors[j, 0]), int(all_colors[j, 1]), int(all_colors[j, 2]))
            f.write(struct.pack("<Q", j + 1))
            f.write(struct.pack("<3d", x, y, z))
            f.write(struct.pack("<3B", r, g, b))
            f.write(struct.pack("<d", 0.0))   # error
            f.write(struct.pack("<Q", 0))     # track length


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=str, help="Path to input video")
    src.add_argument("--imagedir", type=str, help="Directory of input images")
    p.add_argument("--fps", type=float, default=5.0,
                   help="Sampling fps when --video is given (default: 5)")

    p.add_argument("--outdir", type=str, required=True)

    # model
    p.add_argument("--model", type=str,
                   default="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    p.add_argument("--retrieval_model", type=str,
                   default="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth")
    p.add_argument("--finetuned_model", type=str, default=None,
                   help="Optional LoRA checkpoint (Dark3R or Underwat3R v*).")
    p.add_argument("--device", type=str, default="cuda")

    # reconstruction
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--optim_level", type=str, default="refine+depth")
    p.add_argument("--lr1", type=float, default=0.07)
    p.add_argument("--niter1", type=int, default=1000)
    p.add_argument("--lr2", type=float, default=0.01)
    p.add_argument("--niter2", type=int, default=1000)
    p.add_argument("--min_conf_thr", type=float, default=1.5)
    p.add_argument("--matching_conf_thr", type=float, default=0)
    # default: 'swin' — sliding window — appropriate for continuous-trajectory
    # underwater videos (the typical input here). For unordered photo sets or
    # surveys with revisits, switch to 'retrieval' (and provide --retrieval_model).
    p.add_argument("--scenegraph_type", type=str, default="swin",
                   choices=["swin", "logwin", "retrieval", "oneref"])
    # winsize semantics differ by scenegraph type:
    #   swin / logwin : pair each frame with ±winsize neighbors (radius)
    #   retrieval     : number of key images (Na in MASt3R-SfM paper)
    p.add_argument("--winsize", type=int, default=5)
    p.add_argument("--win_cyclic", action="store_true",
                   help="Treat first/last frames as adjacent (loop trajectory).")
    # refid only used by 'retrieval' (k nearest neighbors) and 'oneref' (the
    # single reference frame index). Ignored under swin / logwin.
    p.add_argument("--refid", type=int, default=10)
    p.add_argument("--subsample", type=int, default=1)
    p.add_argument("--images_start", type=int, default=-1)
    p.add_argument("--images_end", type=int, default=-1)
    p.add_argument("--init_focal", type=float, default=None)
    p.add_argument("--focal_reg_w", type=float, default=10.0)
    p.add_argument("--optimizer_dtype", type=str, default="float32",
                   choices=["float32", "float16"])

    # colmap export
    p.add_argument("--fps_ratio", type=float, default=None,
                   help="Downsample point cloud via FPS (e.g. 0.1 keeps 10%%).")
    p.add_argument("--no_keep_original_resolution", action="store_true",
                   help="Record MASt3R's processed (downsized) image resolution in "
                        "cameras.{txt,bin} instead of the original-image resolution. "
                        "Default behavior rescales intrinsics to match the JPGs that "
                        "were copied into images/, which is what NeRF / 3DGS / viser "
                        "expect.")

    # cache / reuse
    p.add_argument("--reuse_scene_pkl", type=str, default=None,
                   help="Skip inference and re-export a previously saved scene.pkl.")

    # misc
    p.add_argument("--silent", action="store_true")
    p.add_argument("--pairs_pkl_path", type=str, default=None)
    p.add_argument("--sim_matrix", type=str, default=None)
    p.add_argument("--plot_matches", action="store_true")
    p.add_argument("--subset_list", type=str, default=None)

    args = p.parse_args()

    # Defaults inference.reconstruction expects on the args namespace.
    args.filelist = None
    args.current_scene_state = None
    args.gradio_delete_cache = None
    args.as_pointcloud = True
    args.mask_sky = False
    args.clean_depth = False
    args.transparent_cams = False
    args.cam_size = 0.2
    args.TSDF_thresh = 0
    args.debug = False
    args.verbose = False
    args.subsample_viz = False
    args.subsample_pc = 10
    args.start_server = False
    return args


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    print(f"[args] {args}")

    # 1) collect / extract input frames into <outdir>/images/
    if args.video is not None:
        filelist = extract_frames(args.video, args.outdir, args.fps)
    else:
        filelist = collect_images(args.imagedir, args.outdir)

    # apply optional subsetting (matches forward_dark3r.py semantics)
    if args.subset_list:
        with open(args.subset_list) as f:
            wanted = {line.strip() for line in f if line.strip()}
        filelist = [p for p in filelist if os.path.basename(p) in wanted]
    if args.images_start >= 0 and args.images_end > 0:
        filelist = filelist[args.images_start : args.images_end]
    elif args.images_end > 0 and len(filelist) > args.images_end:
        filelist = filelist[: args.images_end]
    if args.subsample > 1:
        filelist = filelist[args.subsample // 2 :: args.subsample]

    print(f"[reconstruct] using {len(filelist)} frames")
    args.filelist = list(filelist)
    image_names = [os.path.basename(p) for p in filelist]

    # 2) save args.json for reproducibility
    with open(os.path.join(args.outdir, "args.json"), "w") as f:
        json.dump(
            {k: v for k, v in vars(args).items() if not k.startswith("_")},
            f, indent=4, default=str,
        )

    # 3) reconstruction (or reuse cached scene.pkl)
    started = datetime.datetime.now()
    if args.reuse_scene_pkl and os.path.isfile(args.reuse_scene_pkl):
        print(f"[reuse] loading scene from {args.reuse_scene_pkl}")
        with open(args.reuse_scene_pkl, "rb") as f:
            scene = pickle.load(f)
    else:
        model = load_model(args)
        cache_dir = os.path.join(args.outdir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        args.cache_dir = cache_dir

        imgs, filelist_loaded = prepare_images(args)
        pairs = create_image_pairs(args, imgs, filelist_loaded, model)
        scene = run_reconstruction(args, filelist_loaded, pairs, model)

        with open(os.path.join(args.outdir, "scene.pkl"), "wb") as f:
            pickle.dump(scene, f)

    elapsed = (datetime.datetime.now() - started).total_seconds()
    print(f"[reconstruct] done in {elapsed:.1f}s")

    # 4) export COLMAP
    export_colmap(
        scene,
        args.outdir,
        image_names,
        min_conf_thr=args.min_conf_thr,
        fps_ratio=args.fps_ratio,
        keep_original_resolution=not args.no_keep_original_resolution,
    )  # tuple return is ignored — geometry-only path needs no per-point aux

    print()
    print(f"[done] COLMAP workspace: {args.outdir}")
    print(f"       images:    {os.path.join(args.outdir, 'images')}")
    print(f"       sparse/0:  {os.path.join(args.outdir, 'sparse', '0')}")


if __name__ == "__main__":
    main()
