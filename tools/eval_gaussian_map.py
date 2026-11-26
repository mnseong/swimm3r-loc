#!/usr/bin/env python3
"""Feed forward Gaussian map of a Stage 1 scene, scored on held out frames.

The map is built without any optimisation: every frame of the Stage 1 reconstruction gives one
Gaussian per pixel from the Gaussian map head, placed with the pose, the depth and the
intrinsics of our SfM, and all frames are fused in voxels. The held out frames are scored with
the rules of ubs/seg_eval.py: the test pose starts from the pose interpolated from the training
poses and is refined against the fixed map, PSNR is the plain mean squared error, SSIM and
LPIPS (alex) come from torchmetrics.

    python tools/eval_gaussian_map.py --scene_dir STAGE1_DIR --data_dir SEGMENT_DIR \\
        --lora RUN/weights/0019_lora.pth --gauss RUN/weights/0019_gauss.pth --out_dir OUT

Without --gauss the head is replaced by the plain pixel splat, the starting point of the head.

With --kernel beta every render and the pose refinement use the Beta kernel of the gsplat fork
(render_view_beta), the Gaussian sigmas times K_BETA4 of training/primitives4d.py. A 14 channel
head draws with beta 4, a 21 channel head with its own beta. With --fourd (21 channel head) every
map frame has the time fnum / --fps (or the time of its name with --times_from_names), the map is
split into a fused static set and dynamic rows by split_static_dynamic, and every frame is drawn at
its own time. --static_only draws the static set alone. --export_ubs writes the map as the 7D
parameters of ubs/scene/beta_model.py, in the units of the SfM.
"""
import argparse
import json
import os
import pickle
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mast3r.utils.path_to_dust3r  # noqa: E402,F401
from model.gaussian_head import N_OUT, N_OUT_4D, OPACITY_BIAS, SCALE_BASE, GaussianMapHead, head_out_channels  # noqa: E402
from training.gaussian_branch import (  # noqa: E402
    NEAR_UNITS, quat_mul, render_view, render_view_beta, rotmat_to_quat, voxel_fuse,
)
from training.primitives4d import (  # noqa: E402
    DYN_MIN_OPACITY, EPS2D_BETA4, K_BETA4, export_ubs7d, frame_gaps, primitives_at_time, rescale,
    split_static_dynamic, world_velocity, write_ubs_ply,
)


def fnum(name):
    return int(re.search(r"(\d+)", os.path.basename(str(name))).group(1))


def frame_seconds(names, fps, from_names=False):
    """Time in seconds of every name, fnum / fps, or with from_names the time stamp or index of the
    name as frame_times of dataset/window_dataset.py reads it (names '<int>.jpg')."""
    if not from_names:
        return {n: fnum(n) / fps for n in names}
    from dataset.window_dataset import frame_times
    uniq = sorted(set(names), key=lambda n: int(os.path.basename(n)[:-4]))
    return dict(zip(uniq, frame_times([os.path.basename(n) for n in uniq], fps).tolist()))


def interpolate_poses(poses, test_nums, train_nums):
    """Cubic spline of the translation and slerp of the rotation over the frame number."""
    from scipy.interpolate import CubicSpline
    from scipy.spatial.transform import Rotation, Slerp
    train_nums = np.asarray(train_nums, dtype=np.float64)
    order = np.argsort(train_nums)
    nums, poses = train_nums[order], poses[order]
    cs = CubicSpline(nums, poses[:, :3, 3], bc_type="clamped")
    slerp = Slerp(nums, Rotation.from_matrix(poses[:, :3, :3]))
    out = []
    for n in test_nums:
        t = float(np.clip(float(n), nums[0], nums[-1]))
        pose = np.eye(4)
        pose[:3, :3] = slerp(t).as_matrix()
        pose[:3, 3] = cs(t)
        out.append(pose)
    return np.asarray(out)


def load_scene(scene_dir, dev, clean_depth=True):
    """Frames, depth, confidence, intrinsics and poses of a Stage 1 output. clean_depth sets the confidence of the
    points that lie in front of the depth of another view to 0 (MASt3R's cleaning), without it every point keeps its own."""
    with open(os.path.join(scene_dir, "scene.pkl"), "rb") as f:
        scene = pickle.load(f)
    names = [os.path.basename(p) for p in scene.img_paths]
    _, depthmaps, confs = scene.get_dense_pts3d(clean_depth=bool(clean_depth))
    imgs = torch.stack([torch.from_numpy(np.asarray(im, dtype=np.float32)).permute(2, 0, 1) for im in scene.imgs]).to(dev)
    n, _, h, w = imgs.shape
    depth = torch.stack([d.detach().float().reshape(h, w) for d in depthmaps]).to(dev)
    conf = torch.stack([c.detach().float().reshape(h, w) for c in confs]).to(dev)
    intr = torch.stack([torch.as_tensor(k).float() for k in scene.intrinsics]).to(dev)
    c2w = scene.cam2w.detach().float().to(dev)
    return dict(names=names, imgs=imgs, depth=depth, conf=conf, intr=intr, c2w=c2w)


def axis_angle_to_matrix(r):
    zero = torch.zeros((), device=r.device)
    skew = torch.stack([torch.stack([zero, -r[2], r[1]]), torch.stack([r[2], zero, -r[0]]),
                        torch.stack([-r[1], r[0], zero])])
    return torch.linalg.matrix_exp(skew)


def load_image(path, long_edge=None):
    """Full size image, or the image resized and cropped like the input of the network."""
    from PIL import Image
    from dataset.window_dataset import resize_crop_box
    img = Image.open(path).convert("RGB")
    if long_edge is not None:
        size, box = resize_crop_box(*img.size, long_edge=long_edge)
        img = img.resize(size, Image.LANCZOS if max(img.size) > long_edge else Image.BICUBIC).crop(box)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


@torch.no_grad()
def frame_attributes(net, head, imgs, chunk=6, partners=None):
    """Head output of every frame, with the next frame as partner of the pair pass.

    partners replaces the next frame: (N,) indices into imgs, or the partner images (N, 3, H, W).
    """
    n, _, h, w = imgs.shape
    dev = imgs.device
    if head is None:
        rgb = imgs.permute(0, 2, 3, 1)
        return dict(
            opacity=torch.full((n, h, w), 1.0 / (1.0 + np.exp(-OPACITY_BIAS)), device=dev),
            log_scale=torch.zeros(n, h, w, 3, device=dev),
            quat=rgb.new_tensor([1.0, 0.0, 0.0, 0.0]).expand(n, h, w, 4).contiguous(),
            color_raw=rgb, color_clean=rgb)
    shape1 = torch.tensor([[h, w]], device=dev)

    def encode(x):
        feats, poss = [], []
        for s in range(0, n, chunk):
            f, p, _ = net._encode_image(x[s:s + chunk] * 2.0 - 1.0, shape1.repeat(min(chunk, n - s), 1))
            feats.append(f)
            poss.append(p)
        return torch.cat(feats), torch.cat(poss)

    feat, pos = encode(imgs)
    pfeat, ppos = feat, pos
    if partners is None:
        partner = torch.arange(n, device=dev) + 1
        partner[-1] = n - 2
    elif partners.dim() == 1:
        partner = partners.to(device=dev, dtype=torch.long)
    else:
        assert tuple(partners.shape) == tuple(imgs.shape), (partners.shape, imgs.shape)
        pfeat, ppos = encode(partners.to(device=dev, dtype=imgs.dtype))
        partner = torch.arange(n, device=dev)
    out = {}
    for s in range(0, n, chunk):
        i = torch.arange(s, min(s + chunk, n), device=dev)
        dec1, _ = net._decoder(feat[i], pos[i], pfeat[partner[i]], ppos[partner[i]])
        raw = head([t.float() for t in dec1], imgs[i] * 2.0 - 1.0, (h, w))
        act = GaussianMapHead.activate(raw, imgs[i])
        for k, v in act.items():
            out.setdefault(k, []).append(v)
    return {k: torch.cat(v) for k, v in out.items()}


def build_map(act, depth, conf, intr, c2w, args, times=None):
    """World Gaussians of all frames, fused. depth, conf (N, H, W), intr (N, 3, 3), c2w (N, 4, 4).

    zsrc is the depth at which a Gaussian was observed, in the units of the SfM. With a beta in act
    (21 channel head), args.fourd or args.export_ubs the map is built by build_split.
    """
    if "beta" in act or getattr(args, "fourd", False) or getattr(args, "export_ubs", None):
        return build_split(act, depth, conf, intr, c2w, args, times)
    n, h, w = depth.shape
    dev = depth.device
    v, u = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                          torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
    parts = {k: [] for k in ("xyz", "fp", "conf", "zsrc", "opacity", "scales", "quat", "color_raw", "color_clean")}
    for i in range(n):
        f, cx, cy = intr[i, 0, 0], intr[i, 0, 2], intr[i, 1, 2]
        z = depth[i]
        cam = torch.stack([(u - cx) / f * z, (v - cy) / f * z, z], dim=-1)
        xyz = cam @ c2w[i, :3, :3].T + c2w[i, :3, 3]
        fp = z / f
        ok = torch.isfinite(xyz).all(-1) & (z > 1e-4) & (conf[i] >= args.min_conf)
        q = quat_mul(rotmat_to_quat(c2w[i, :3, :3])[None, None].expand(h, w, 4), act["quat"][i])
        parts["xyz"].append(xyz[ok])
        parts["fp"].append(fp[ok])
        parts["conf"].append(conf[i][ok])
        parts["zsrc"].append(z[ok])
        parts["opacity"].append(act["opacity"][i][ok])
        parts["scales"].append((fp[..., None] * SCALE_BASE * torch.exp(act["log_scale"][i]))[ok])
        parts["quat"].append(q[ok])
        parts["color_raw"].append(act["color_raw"][i][ok])
        parts["color_clean"].append(act["color_clean"][i][ok])
    g = {k: torch.cat(vv) for k, vv in parts.items()}
    n_in = g["xyz"].shape[0]
    if args.voxel_px > 0:
        xyz, m, p = voxel_fuse(g["xyz"], g["conf"], g["fp"],
                               dict(opacity=g["opacity"], color_raw=g["color_raw"], color_clean=g["color_clean"], zsrc=g["zsrc"]),
                               dict(scales=g["scales"], quat=g["quat"]), args.voxel_px)
        g = dict(xyz=xyz, opacity=m["opacity"], color_raw=m["color_raw"], color_clean=m["color_clean"], zsrc=m["zsrc"],
                 scales=p["scales"], quat=p["quat"])
    zs = depth[depth > 1e-4]
    unit = NEAR_UNITS / torch.quantile(zs[:: max(1, zs.numel() // 200000)], 0.01).clamp_min(1e-4)
    g["means"], g["scales"], g["unit"] = g["xyz"] * unit, g["scales"] * unit, unit
    g["n_in"], g["n_out"] = n_in, g["means"].shape[0]
    return g


def build_split(act, depth, conf, intr, c2w, args, times=None):
    """build_map for the Beta primitives of training/primitives4d.py.

    The per pixel primitives also carry beta (4 without it in act), and with args.fourd the gate,
    the time of their frame (times (N,) seconds), the temporal width (frame gap times tw_rel), shape
    and the velocity in the world. split_static_dynamic fuses the static copies as build_map fuses
    the Gaussians. g is the static set in the layout of build_map plus beta, 'split_sfm' the split
    in the units of the SfM, 'split' in the units of the render, n_dyn and gate_mean.
    """
    n, h, w = depth.shape
    dev = depth.device
    four_d = bool(getattr(args, "fourd", False))
    if four_d and "gate" not in act:
        raise ValueError("a 4D map needs the 21 channel head")
    v, u = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                          torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
    keys = ["xyz", "fp", "conf", "zsrc", "opacity", "scales", "quat", "color_raw", "color_clean"]
    keys += ["beta"] if "beta" in act else []
    keys += ["gate", "t_src", "tw", "tb", "vel"] if four_d else []
    parts = {k: [] for k in keys}
    if four_d:
        times = torch.as_tensor(times, dtype=torch.float32, device=dev).reshape(n)
        gaps = frame_gaps(times)
    for i in range(n):
        f, cx, cy = intr[i, 0, 0], intr[i, 0, 2], intr[i, 1, 2]
        z = depth[i]
        cam = torch.stack([(u - cx) / f * z, (v - cy) / f * z, z], dim=-1)
        xyz = cam @ c2w[i, :3, :3].T + c2w[i, :3, 3]
        fp = z / f
        ok = torch.isfinite(xyz).all(-1) & (z > 1e-4) & (conf[i] >= args.min_conf)
        q = quat_mul(rotmat_to_quat(c2w[i, :3, :3])[None, None].expand(h, w, 4), act["quat"][i])
        parts["xyz"].append(xyz[ok])
        parts["fp"].append(fp[ok])
        parts["conf"].append(conf[i][ok])
        parts["zsrc"].append(z[ok])
        parts["opacity"].append(act["opacity"][i][ok])
        parts["scales"].append((fp[..., None] * SCALE_BASE * torch.exp(act["log_scale"][i]))[ok])
        parts["quat"].append(q[ok])
        parts["color_raw"].append(act["color_raw"][i][ok])
        parts["color_clean"].append(act["color_clean"][i][ok])
        if "beta" in act:
            parts["beta"].append(act["beta"][i][ok])
        if four_d:
            parts["gate"].append(act["gate"][i][ok])
            parts["t_src"].append(times[i].expand(h, w)[ok])
            parts["tw"].append((gaps[i] * act["tw_rel"][i])[ok])
            parts["tb"].append(act["tb"][i][ok])
            parts["vel"].append(world_velocity(act["vel"][i], fp, c2w[i, :3, :3][None, None])[ok])
    prim = {k: torch.cat(vv) for k, vv in parts.items()}
    n_in = prim["xyz"].shape[0]
    split = split_static_dynamic(prim, voxel_fuse, args.voxel_px, four_d=four_d, extra_mean=("zsrc",))
    zs = depth[depth > 1e-4]
    unit = NEAR_UNITS / torch.quantile(zs[:: max(1, zs.numel() // 200000)], 0.01).clamp_min(1e-4)
    g = dict(split["static"])
    g["means"], g["scales"], g["unit"] = g["xyz"] * unit, g["scales"] * unit, unit
    g["n_in"], g["n_out"] = n_in, g["means"].shape[0]
    g.update(split_sfm=split, split=rescale(split, unit), n_dyn=split["n_dyn"], gate_mean=split["gate_mean"])
    return g


def render(g, c2w, focal, pp, h, w, bg, key="color_raw", t=None):
    c = c2w.clone()
    c[:3, 3] = c[:3, 3] * g["unit"]
    if g.get("kernel") == "beta":
        # the static set (build_map, --static_only) or the whole map at time t
        p, means = g, g["means"]
        if "split" in g and not g.get("static_only"):
            p = primitives_at_time(g["split"], t, min_opacity=DYN_MIN_OPACITY)
            means = p["xyz"]
        img, _, _ = render_view_beta(means, p["opacity"], p["scales"], p["quat"], p[key], c, focal, h, w, bg, pp=pp,
                                     betas=p.get("beta", 4.0), scale_factor=K_BETA4, eps2d=EPS2D_BETA4)
        return img
    img, _, _ = render_view(g["means"], g["opacity"], g["scales"], g["quat"], g[key], c, focal, h, w, bg, pp=pp)
    return img


def refine_pose(g, c2w0, focal, pp, h, w, bg, gt, iters, lr, t=None):
    """Photometric refinement of one camera against the fixed map, L1 as in the renderer."""
    if iters <= 0:
        return c2w0
    dr = torch.zeros(3, device=gt.device, requires_grad=True)
    dt = torch.zeros(3, device=gt.device, requires_grad=True)
    opt = torch.optim.Adam([dr, dt], lr=lr)
    unit = float(g["unit"])

    def pose():
        d = torch.eye(4, device=gt.device)
        d[:3, :3] = axis_angle_to_matrix(dr)
        d[:3, 3] = dt / unit
        return c2w0 @ d

    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        loss = F.l1_loss(render(g, pose(), focal, pp, h, w, bg, t=t), gt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return pose().detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True, help="Stage 1 output with scene.pkl and images/")
    ap.add_argument("--data_dir", default=None, help="folder whose split.json lists the test frames")
    ap.add_argument("--lora", default=None, help="LoRA weights of the network, needed with --gauss")
    ap.add_argument("--gauss", default=None, help="weights of the Gaussian map head")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mast3r", default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    ap.add_argument("--voxel_px", type=float, default=1.0)
    ap.add_argument("--min_conf", type=float, default=0.0)
    ap.add_argument("--pose_iters", type=int, default=300)
    ap.add_argument("--pose_lr", type=float, default=8e-4)
    ap.add_argument("--save_images", type=int, default=6)
    ap.add_argument("--kernel", choices=("gauss", "beta"), default="gauss",
                    help="Gaussian kernel of diff_gauss_pose or Beta kernel of the gsplat fork in ubs/submodules")
    ap.add_argument("--fourd", action="store_true",
                    help="4D map of a 21 channel head, every frame drawn at its own time, needs --kernel beta")
    ap.add_argument("--static_only", action="store_true", help="with --fourd, draw the static set alone")
    ap.add_argument("--fps", type=float, default=5.0, help="frame rate of the frame numbers, the time of a frame is fnum / fps")
    ap.add_argument("--times_from_names", action="store_true",
                    help="times from the names, capture time stamps or indices at --fps "
                         "(frame_times of dataset/window_dataset.py)")
    ap.add_argument("--region", type=int, nargs=2, default=None, metavar=("A", "B"),
                    help="also aggregate the test frames with A <= frame <= B")
    ap.add_argument("--export_ubs", default=None,
                    help="PLY of the map as the 7D parameters of ubs/scene/beta_model.py (SfM units, clean colour), "
                         "needs --kernel beta")
    args = ap.parse_args()
    if args.fourd and args.kernel != "beta":
        ap.error("--fourd needs --kernel beta")
    if args.static_only and not args.fourd:
        ap.error("--static_only needs --fourd")
    if args.export_ubs and args.kernel != "beta":
        ap.error("--export_ubs needs --kernel beta")
    state, n_out = None, N_OUT
    if args.gauss:
        state = torch.load(args.gauss, map_location="cpu")["head"]
        n_out = head_out_channels(state)
    if args.fourd and n_out != N_OUT_4D:
        ap.error(f"--fourd needs a head of {N_OUT_4D} channels, --gauss has {n_out if args.gauss else 'no head'}")
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device("cuda")
    t0 = time.time()

    sc = load_scene(args.scene_dir, dev)
    names, imgs, depth, conf, intr, c2w = sc["names"], sc["imgs"], sc["depth"], sc["conf"], sc["intr"], sc["c2w"]
    n, _, h, w = imgs.shape
    print(f"[scene] {n} frames at {h}x{w}, focal {float(intr[:, 0, 0].median()):.1f} px "
          f"(min {float(intr[:, 0, 0].min()):.1f}, max {float(intr[:, 0, 0].max()):.1f}), "
          f"depth median {float(depth[depth > 0].median()):.3f}")

    net = head = None
    if args.gauss:
        from model.lora_model import LoRAMast3r
        net = LoRAMast3r.from_pretrained(args.mast3r)
        if args.lora:
            net.load_pretrained_lora(args.lora)
        net = net.to(dev).eval()
        head = (GaussianMapHead(net) if n_out == N_OUT else GaussianMapHead(net, n_out=n_out)).to(dev).eval()
        head.load_state_dict(state)
    sec = tmap = None
    if args.fourd or args.export_ubs:
        test_names = json.load(open(os.path.join(args.data_dir, "split.json")))["test_files"] if args.data_dir else []
        sec = frame_seconds(list(names) + list(test_names), args.fps, args.times_from_names)
        t_first = min(sec[x] for x in names)
        sec = {k: v - t_first for k, v in sec.items()}
        tmap = torch.tensor([sec[x] for x in names], dtype=torch.float32, device=dev)
    with torch.no_grad():
        act = frame_attributes(net, head, imgs)
        g = build_map(act, depth, conf, intr, c2w, args, times=tmap)
    t_map = time.time() - t0
    print(f"[map] {g['n_in'] / 1e6:.2f}M Gaussians from {n} frames, {g['n_out'] / 1e6:.2f}M after fusion, "
          f"{t_map:.1f}s after the SfM")
    if args.kernel == "beta":
        g["kernel"], g["static_only"] = "beta", args.static_only
        print(f"[kernel] Beta, scales times {K_BETA4} and pixel filter {EPS2D_BETA4:.3f} px^2 | head {n_out if args.gauss else 0} "
              f"channels | {g['n_out']} static and {g.get('n_dyn', 0)} dynamic rows, gate mean {g.get('gate_mean', 0.0):.4f}, "
              f"beta mean {float(g['beta'].mean()) if 'beta' in g else 4.0:.3f}"
              + (f" | times {float(tmap.min()):.2f} to {float(tmap.max()):.2f} s" if tmap is not None else "")
              + (" | drawn: static set only" if args.static_only else ""), flush=True)
    t_view = tmap if args.fourd else None

    bg = torch.zeros(3, device=dev)
    focal = float(intr[:, 0, 0].median())
    pp = (float(intr[:, 0, 2].median()) + 0.5, float(intr[:, 1, 2].median()) + 0.5)

    from torchmetrics.functional.image import structural_similarity_index_measure as tm_ssim
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lp = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(dev)

    def score(img, gt):
        img = img.clamp(0.0, 1.0)
        return [float(-10.0 * torch.log10(((img - gt) ** 2).mean() + 1e-12)),
                float(tm_ssim(img[None], gt[None], data_range=1.0)),
                float(lp(img[None], gt[None]))]

    def render_at(pose, hw, key="color_raw", f=None, c=None, t=None):
        f = focal if f is None else f
        c = pp if c is None else c
        s = hw[1] / w
        return render(g, pose, f * s, (c[0] * s, c[1] * hw[0] / h), hw[0], hw[1], bg, key, t=t)

    out = {"scene": args.scene_dir, "gauss": args.gauss, "lora": args.lora, "frames_in_map": n,
           "gaussians_before_fusion": g["n_in"], "gaussians": g["n_out"], "seconds_map": t_map,
           "voxel_px": args.voxel_px, "min_conf": args.min_conf, "pose_iters": args.pose_iters}
    if args.kernel == "beta":
        out.update(kernel="beta", fourd=args.fourd, static_only=args.static_only, head_channels=n_out if args.gauss else None,
                   fps=args.fps, times_from_names=args.times_from_names, scale_factor=K_BETA4, eps2d=EPS2D_BETA4,
                   gaussians_static=g["n_out"], gaussians_dynamic=g.get("n_dyn", 0), gate_mean=g.get("gate_mean", 0.0),
                   beta_mean=float(g["beta"].mean()) if "beta" in g else 4.0)
    if args.export_ubs:
        te = time.time()
        params = export_ubs7d(g["split_sfm"], (0.0, float(tmap.max())))
        write_ubs_ply(args.export_ubs, params)
        fe = params["fit_err"][torch.isfinite(params["fit_err"])].double().cpu()
        out["export_ubs"] = {"path": args.export_ubs, "rows": int(params["xyz"].shape[0]), "static": g["n_out"],
                             "dynamic": g.get("n_dyn", 0), "t_range_s": [0.0, float(tmap.max())],
                             "fit_err_median": float(fe.median()) if fe.numel() else None,
                             "fit_err_p99": float(torch.quantile(fe[:: max(1, fe.numel() // 1000000)], 0.99)) if fe.numel() else None,
                             "fit_err_max": float(fe.max()) if fe.numel() else None,
                             "velocities_clamped": int(params["clamped"].sum()), "seconds": time.time() - te}
        print(f"[export] {out['export_ubs']['rows']} rows to {args.export_ubs} in {out['export_ubs']['seconds']:.0f}s", flush=True)
        del params

    # Frames of the map itself, a check of the map and of the conventions.
    with torch.no_grad():
        own = [score(render_at(c2w[i], (h, w), f=float(intr[i, 0, 0]),
                               c=(float(intr[i, 0, 2]) + 0.5, float(intr[i, 1, 2]) + 0.5),
                               t=None if t_view is None else t_view[i]), imgs[i])
               for i in range(0, n, max(1, n // 12))]
    out["map_frames"] = np.mean(own, 0).tolist()
    print("[map frames] PSNR %.2f SSIM %.4f LPIPS %.4f" % tuple(out["map_frames"]))

    if args.data_dir:
        split = json.load(open(os.path.join(args.data_dir, "split.json")))
        centre = set(fnum(x) for x in split.get("centre_files", []))
        train_nums = [fnum(x) for x in names]
        tests = sorted(split["test_files"], key=fnum)
        start = interpolate_poses(c2w.cpu().numpy().astype(np.float64), [fnum(x) for x in tests], train_nums)
        rows = []
        for j, name in enumerate(tests):
            path = os.path.join(args.scene_dir, "images", name)
            gt_full = load_image(path).to(dev)
            hw_full = tuple(gt_full.shape[-2:])
            gt = load_image(path, max(h, w)).to(dev)
            assert tuple(gt.shape[-2:]) == (h, w), (gt.shape, h, w)
            same_view = abs(hw_full[1] / hw_full[0] - w / h) < 1e-3
            p0 = torch.tensor(start[j], dtype=torch.float32, device=dev)
            tj = None if t_view is None else sec[name]
            with torch.no_grad():
                before = score(render_at(p0, (h, w), t=tj), gt)
            p1 = refine_pose(g, p0, focal, pp, h, w, bg, gt, args.pose_iters, args.pose_lr, t=tj)
            with torch.no_grad():
                pred = render_at(p1, (h, w), t=tj)
                after = score(pred, gt)
                full = score(render_at(p1, hw_full, t=tj), gt_full) if same_view else [float("nan")] * 3
                if j < args.save_images or (fnum(name) in centre and j < 4 * args.save_images):
                    from PIL import Image
                    row = torch.cat([gt, pred.clamp(0, 1), render_at(p1, (h, w), "color_clean", t=tj).clamp(0, 1)], dim=2)
                    Image.fromarray((row.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(
                        os.path.join(args.out_dir, "%s_gt_raw_clean.jpg" % os.path.splitext(name)[0]), quality=92)
            num = fnum(name)
            rows.append({"frame": num, "centre": num in centre, "dist": min(abs(num - k) for k in train_nums),
                         "before": before, "after": after, "after_full": full})
            if tj is not None:
                rows[-1]["time"] = tj
            print("%06d %s before %.2f / %.4f / %.4f after %.2f / %.4f / %.4f full %.2f / %.4f / %.4f" % (
                num, "centre" if num in centre else "side  ", *before, *after, *full), flush=True)

        def agg(sel):
            r = [x for x in rows if sel(x)]
            if not r:
                return None
            return {"n": len(r), **{k: np.mean([x[k] for x in r], 0).tolist() for k in ("before", "after", "after_full")}}

        out.update({"all": agg(lambda r: True), "centre": agg(lambda r: r["centre"]),
                    "side": agg(lambda r: not r["centre"]), "frames": rows, "eval_hw": [h, w]})
        if args.region:
            out["region"] = agg(lambda r: args.region[0] <= r["frame"] <= args.region[1])
            out["region_frames"] = list(args.region)
        for k in ("all", "centre", "side") + (("region",) if args.region else ()):
            a = out[k]
            if a:
                print("[%s] n %d | %dx%d before %.3f / %.4f / %.4f after %.3f / %.4f / %.4f | full size after %.3f / %.4f / %.4f" % (
                    k, a["n"], h, w, *a["before"], *a["after"], *a["after_full"]))
    out["seconds_total"] = time.time() - t0
    json.dump(out, open(os.path.join(args.out_dir, "scores.json"), "w"), indent=1)
    print(f"[done] {args.out_dir}/scores.json in {out['seconds_total']:.0f}s")


if __name__ == "__main__":
    main()
