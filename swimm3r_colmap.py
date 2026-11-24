#!/usr/bin/env python3
"""
Swimm3R end-to-end inference: video / image-dir → COLMAP + descattered output.

This is the one-shot entry point — it bundles the geometry pipeline of
forward_dark3r_colmap.py with v7's physics inverse from descatter_colmap.py
in a single process.

What it produces (under --outdir):

    images/                          ← raw input frames (copied / extracted)
    sparse/0/{cameras,images,points3D}.{txt,bin}  ← COLMAP geometry
    transforms.json                  ← NeRF / 3DGS pose format
    scene.pkl                        ← cached scene (resumable)
    descattered/<name>.jpg           ← per-frame J preview (Akkaynak inverse)
    descattered_colors.npy           ← per-point [N, 3] uint8 (J colors)
    physics_params.npz               ← per-frame (β^D, β^B, B^∞) means
    args.json                        ← reproducibility

The viser viewer auto-detects descattered_colors.npy and defaults to the
descattered ("restored") cloud:

    python utils/viewer_viser.py --model swimm3r:<outdir>

Usage
-----
    python swimm3r_colmap.py \\
        --video /data/.../dive.mp4 --fps 5 \\
        --outdir /data/.../swimm3r_phys_video1

    python swimm3r_colmap.py \\
        --imagedir /data/.../video_frames \\
        --outdir   /data/.../swimm3r_phys_video1 \\
        --winsize 5
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch

from inference.reconstruction import (
    load_model,
    prepare_images,
    create_image_pairs,
    run_reconstruction,
)

# Reuse helpers from the geometry-only entrypoint to avoid drift.
from forward_dark3r_colmap import (
    extract_frames,
    collect_images,
    export_colmap,
)


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
    p.add_argument("--finetuned_model", type=str,
                   default="results/underwat3r_v9p3_epsilon/weights/0009_lora.pth",
                   help="LoRA checkpoint.")
    p.add_argument("--physics_model", type=str,
                   default="results/underwat3r_v9p3_epsilon/weights/0009_physics.pth",
                   help="Physics-head checkpoint (e.g. ..._physics.pth). "
                        "Set to empty string for geometry-only export.")
    # Pure MASt3R-SfM ablation baseline — disables ALL Dark3R / Swimm3R
    # extensions. Equivalent to manually setting --finetuned_model None +
    # --physics_model "". Use this for fair comparison runs where ONLY the
    # base MASt3R + sparse_global_alignment pipeline should be active.
    p.add_argument("--pure_mast3r", action="store_true",
                   help="Pure MASt3R-SfM ablation: disable LoRA finetune "
                        "AND physics descattering. base MASt3R only.")
    p.add_argument("--device", type=str, default="cuda")

    # reconstruction (mirrors forward_dark3r_colmap.py defaults — swin for video)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--optim_level", type=str, default="refine+depth")
    p.add_argument("--lr1", type=float, default=0.07)
    p.add_argument("--niter1", type=int, default=1000)
    p.add_argument("--lr2", type=float, default=0.01)
    p.add_argument("--niter2", type=int, default=1000)
    p.add_argument("--min_conf_thr", type=float, default=1.5)
    p.add_argument("--matching_conf_thr", type=float, default=0)
    p.add_argument("--scenegraph_type", type=str, default="swin",
                   choices=["swin", "logwin", "retrieval", "oneref", "swinret"],
                   help="swinret: the swin window of --winsize united with the retrieval graph of --ret_na key "
                        "images and --ret_k neighbours (mast3r/image_pairs.py swinret_pair_ids).")
    p.add_argument("--winsize", type=int, default=5)
    p.add_argument("--ret_na", type=int, default=20, help="Key images of the retrieval part of swinret.")
    p.add_argument("--ret_k", type=int, default=10, help="Nearest images per image of the retrieval part of swinret.")
    p.add_argument("--win_cyclic", action="store_true",
                   help="Treat first/last frames as adjacent (loop trajectory).")
    p.add_argument("--refid", type=int, default=10)
    p.add_argument("--subsample", type=int, default=1)
    p.add_argument("--images_start", type=int, default=-1)
    p.add_argument("--images_end", type=int, default=-1)
    # InstantSplat-style train/test split — prevents test leakage in NVS eval.
    # When --train_test_split=llff: split the sorted image list with every
    # `llffhold`-th frame held out as test; only the train subset is fed to
    # MASt3R-SfM and the descatter pipeline. The test image list is written
    # to <outdir>/split.json for downstream test-pose registration via
    # scripts/init_test_pose_underwater.py. Pose-only entries for the test
    # views are appended to sparse/0/ later (no contribution to the point
    # cloud or descatter, matching InstantSplat fair-evaluation protocol).
    p.add_argument("--train_test_split", choices=["none", "llff"], default="llff",
                   help="Train/test split mode. 'llff' = sort alphabetically, "
                        "test = every llffhold-th frame, train = the rest. "
                        "Only train views are reconstructed/descattered here; "
                        "use scripts/init_test_pose_underwater.py afterwards to "
                        "register test camera poses into the same sparse/0/.")
    p.add_argument("--llffhold", type=int, default=8,
                   help="Hold-out interval for --train_test_split=llff "
                        "(default 8 = 12.5%% test, 87.5%% train).")
    p.add_argument("--n_train_views", type=int, default=-1,
                   help="Optional sparse subset of train views, evenly "
                        "spaced. -1 (default) = use all train views.")
    p.add_argument("--keep_test_cameras", action="store_true",
                   help="(--train_test_split=llff only) Single-pass mode: "
                        "run MASt3R-SfM on train+test together so the test "
                        "camera poses are recovered jointly (no auxiliary "
                        "pass), but mask out test views from the point-cloud "
                        "aggregation so the saved sparse/0/points3D.{txt,bin} "
                        "still represents the train-only descattered geometry. "
                        "Skips the need for scripts/init_test_pose_underwater.py.")
    p.add_argument("--init_focal", type=float, default=None)
    p.add_argument("--focal_reg_w", type=float, default=10.0)
    p.add_argument("--optimizer_dtype", type=str, default="float32",
                   choices=["float32", "float16"])

    # colmap export
    p.add_argument("--fps_ratio", type=float, default=None,
                   help="Downsample point cloud via FPS (e.g. 0.1 keeps 10%%).")
    p.add_argument("--fps_max_points", type=int, default=0,
                   help="Upper limit on point count after FPS downsample "
                        "(0 = no limit, default). Stage 1 saves the full "
                        "predicted point cloud and the downstream splatting "
                        "stage handles per-experiment pruning via "
                        "--init_target_points / --init_voxel_downsample. "
                        "Set a positive value (e.g. 500000) only to bound "
                        "disk / RAM.")
    p.add_argument("--no_keep_original_resolution", action="store_true")

    # cache / reuse
    p.add_argument("--reuse_scene_pkl", type=str, default=None,
                   help="Skip inference and re-export from a saved scene.pkl.")

    # physics-head hyperparams (must match training to be loadable)
    p.add_argument("--phys_head_kind", choices=["depth_dep", "scalar"],
                   default="depth_dep")
    p.add_argument("--phys_max_backscatter", type=float, default=0.7)
    p.add_argument("--phys_b_max", type=float, default=0.2)
    p.add_argument("--phys_beta_clamp_max", type=float, default=10.0)
    p.add_argument("--phys_head_hidden", type=int, default=256)
    # v9: must match the training-time architecture flag of the physics
    # checkpoint, otherwise the weights are loaded into the wrong arithmetic
    # branch (β_B saturates at beta_clamp_max because raw signed coefficients
    # are passed through softplus instead of sigmoid).
    p.add_argument("--phys_constrain_beta_B",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Pass through to PhysicsRegressor — REQUIRED when "
                        "loading a v9 ckpt (trained with --phys_constrain_beta_B). "
                        "Use --no-phys_constrain_beta_B to disable.")
    # v11+ Jerlov unified-β: load v7_jerlov / v15.x ckpts whose physics MLP
    # final Linear has 12 outputs (1 branch, β^B ≡ β^D) instead of 24. Without
    # this flag the state_dict load fails with a shape mismatch. Combine with
    # --no-phys_constrain_beta_B (the two are mutually exclusive Jerlov modes).
    p.add_argument("--phys_jerlov_unified_beta", action="store_true",
                   help="Required when loading v7_jerlov / v15.x ckpts "
                        "(unified-β, MLP 12 outputs, β^B := β^D). "
                        "Combine with --no-phys_constrain_beta_B.")
    # B2-norm — MUST match training. If the model was trained with
    # --phys_depth_norm=median, the physics head + IFM inversion must receive
    # the SAME per-image median-normalised depth (β is in normalised units).
    # Geometry/point positions stay raw; only the descatter inversion uses ẑ.
    # The Sea-Thru-true (out_dim 15) / Jerlov (12) arch is auto-detected from
    # the ckpt, so no extra arch flag is needed beyond this.
    p.add_argument("--phys_depth_norm", choices=["none", "median"], default="none")
    p.add_argument("--phys_seathru_betaB_le_betaD", action="store_true",
                   help="MUST match training. Architectural beta_B <= beta_D cap "
                        "(beta_B = min(beta_B, beta_D) in forward). If the ckpt "
                        "was trained with --phys_seathru_betaB_le_betaD, set it "
                        "here too — it is a forward-time op, not a learned weight, "
                        "so omitting it gives the wrong (uncapped) beta_B.")
    p.add_argument("--phys_z_min", type=float, default=0.1)
    p.add_argument("--phys_z_max", type=float, default=50.0)
    # KNOWN ISSUE #1 guard (docs/SWIMM3R_ARCH_ABSORPTION.md): by default, refuse
    # to run if any DPT-head LoRA tensor in the checkpoint was silently dropped
    # at load (would mean depth runs FROZEN). Set this to downgrade the hard
    # failure to a warning (NOT recommended; only for deliberate base-depth runs).
    p.add_argument("--allow_lora_drop", action="store_true",
                   help="Permit running even if DPT-head LoRA tensors in the "
                        "checkpoint were silently dropped at load (depth FROZEN). "
                        "Default: hard-fail. See docs/SWIMM3R_ARCH_ABSORPTION.md #1.")
    # Inference-time Jerlov R >= G >= B ordering projection on per-point
    # β arrays. Applied just before np.save of beta_D_per_point.npy and
    # beta_B_per_point.npy. Top-down clamp keeps R, projects G ≤ R, B ≤ G.
    # No effect on the trained head's outputs at compute_descatter_per_frame
    # time (so the descattered J preview is unchanged) — only the exported
    # per-point arrays are projected. Use when downstream tools assume the
    # physical ordering.
    p.add_argument("--phys_enforce_rgb_order", action="store_true",
                   help="Project exported per-point β^D, β^B to satisfy "
                        "R >= G >= B (Jerlov ordering) via top-down clamp.")
    # v10+ ratio cap faithfulness — when constrain_beta_B is on, the σ→ratio
    # mapping inside PhysicsRegressor is `ratio = min + (max-min)·σ`. v10+
    # ckpts were TRAINED with bounded ranges (v10-v17: [0.4, 0.95]; v18-v23:
    # [0.4, 0.9]) but inference defaulted to [0.0, 1.0] — same σ yields a
    # different ratio at inference, so β_B at inference ≠ β_B the ckpt was
    # optimised for. Pass the training-time bounds to faithfully reproduce
    # the policy. Default [0.0, 1.0] preserves v9.x inference behaviour
    # (those ckpts were also trained with defaults).
    p.add_argument("--phys_betaB_ratio_min", type=float, default=0.0,
                   help="Lower bound of β_B/β_D ratio at inference. Set to "
                        "match training (v10+: 0.4). Default 0.0 = v9.x.")
    p.add_argument("--phys_betaB_ratio_max", type=float, default=1.0,
                   help="Upper bound of β_B/β_D ratio at inference. Set to "
                        "match training (v10-v17: 0.95; v18+: 0.9). "
                        "Default 1.0 = v9.x.")
    # v9.2 deployment hack — Barbados-class out-of-distribution turbid
    # scenes have β_D under-predicted by the FLSea-trained model, so the
    # un-attenuation factor 1/trans_D is too small and J_R cannot reach
    # natural-colour magnitude even with cap-deflated B∞. This scalar
    # multiplies β_D at inference only; β_B and B∞ are left untouched, so
    # the architectural constraint β_B ≤ β_D becomes strictly tighter (the
    # β_B/β_D ratio shrinks). Use 2.0 for Barbados, 1.0 (default) for
    # FLSea-class.
    p.add_argument("--phys_beta_D_scale", type=float, nargs="+", default=[1.0],
                   help="Inference-only multiplier on predicted β_D. "
                        "One value (uniform) or three values (R G B) for "
                        "per-channel scaling. For OOD turbid water try "
                        "uniform 1.5–2.0 or asymmetric 2.5 1.5 1.0 (R G B).")
    p.add_argument("--phys_max_amp", type=float, nargs="+", default=[0.0],
                   help="Per-pixel cap on the un-attenuation amplification "
                        "factor 1/trans_D. One float (uniform) or three "
                        "(R G B). 0.0 = disabled (use inverse_eps only). "
                        "Recommended ≈ 5.0 for OOD turbid water — prevents "
                        "deep pixels from saturating to pure-red while "
                        "letting shallow pixels recover natural colour.")
    p.add_argument("--phys_max_amp_R_depth_slope", type=float, nargs="+", default=[],
                   help="Depth-aware R-channel amp cap: cap(z) = BASE + "
                        "SLOPE·z (in metres). Two floats. Solves the "
                        "shallow-saturation vs deep-cyan trade-off — shallow "
                        "pixels get tight cap (anti-saturation), deep pixels "
                        "get loose cap (R recovery). Example '5.0 1.5' → "
                        "cap=5 at z=0, cap=12.5 at z=5m, cap=20 at z=10m. "
                        "Overrides --phys_max_amp[0] for R channel only.")
    p.add_argument("--phys_adaptive_R_boost", type=float, default=0.0,
                   help="Per-pixel adaptive β_D_R extra scale. The boost is "
                        "proportional to the raw R-deficit max(0, (G+B)/2 − R), "
                        "so pixels whose R is already comparable to G/B (already "
                        "natural-looking after default inversion) are NOT touched, "
                        "while pixels where R was heavily absorbed get an extra "
                        "β_D_R boost. The argument is the sensitivity factor k: "
                        "extra_scale = clamp(1 + k · R_deficit, max=3). "
                        "Default 0.0 = disabled. Try 5–10 for Barbados.")
    p.add_argument("--phys_adaptive_R_max", type=float, default=3.0,
                   help="Upper cap on the adaptive extra scale "
                        "(clamp(1 + k·deficit, max=phys_adaptive_R_max)).")
    p.add_argument("--phys_anchor_color_balance", type=float, default=0.0,
                   help="Anchor-based per-frame color propagation. Identifies "
                        "pixels that already have natural-looking J (anchor "
                        "set: 0.15 ≤ J_R ≤ 0.7 AND J_R ≥ 0.5·mean(J_G,J_B)) "
                        "and computes per-channel target means. Applies a "
                        "per-channel gain to ALL pixels so the global mean "
                        "matches the anchor mean, blending the cyan-collapsed "
                        "pixels toward the recovered ones. The argument is "
                        "the strength α ∈ [0, 1]: 0 = disabled, 1 = full "
                        "anchor target. Try 0.5–0.8 for Barbados.")
    p.add_argument("--phys_anchor_min_count", type=int, default=500,
                   help="Minimum number of anchor pixels per frame for the "
                        "color balance to apply. If fewer, the frame is left "
                        "unchanged. Prevents noisy propagation when no "
                        "anchors exist.")
    # ----- Inference-time R-saturation soft cap (paper "scene-aware") -----
    # ε baseline preserves cyan-leaning scenes (Barbados v1 R/(GB)=0.383)
    # but produces R-saturation on warm scenes (Barbados v2 J==1 R%=20.4%,
    # DRUVA A12 57.1%). This soft cap reduces J==1 R% on warm scenes WITHOUT
    # touching cyan-leaning scenes (which have ~0% raw J_R > 1).
    #
    # Mechanism: per-frame, compute fraction of pixels with raw J_R > 1.
    # If that fraction exceeds threshold, scale the R channel so the
    # specified high quantile of J_R maps to exactly 1.0. v1-style scenes
    # have ~0% over-saturation → no scaling fires → identical to ε output.
    # Warm scenes have many over-1 pixels → scaling pulls them under 1
    # smoothly (preserves spatial structure, just rescales R).
    p.add_argument("--phys_R_sat_soft_cap", type=float, default=0.05,
                   help="Inference-time R-saturation soft cap threshold. "
                        "When the fraction of pixels with J_R > 1.0 exceeds "
                        "this value, the R channel is scaled down so the "
                        "--phys_R_sat_target_quantile of J_R maps to 1.0. "
                        "0 = disabled. Try 0.05 (= fire when >5%% of pixels "
                        "are R-saturated). Scene-aware: cyan-leaning frames "
                        "(0%% saturation) are left untouched.")
    p.add_argument("--phys_R_sat_target_quantile", type=float, default=0.95,
                   help="When R-sat soft cap fires, scale R channel so this "
                        "quantile of J_R = 1.0. Default 0.95 = 95th "
                        "percentile mapped to 1.0 (top 5%% can still clip "
                        "but rest of distribution preserved).")
    p.add_argument("--phys_per_frame_normalize", action="store_true",
                   help="Two-pass per-frame J normalisation. After all "
                        "J_maps are computed, each frame is rescaled so its "
                        "channel means match the global (across-frames) "
                        "channel means. Removes per-frame inversion bias "
                        "that surfaces as inter-voxel colour variance in the "
                        "merged cloud — adjacent voxels contributed by "
                        "different frames inherit the same colour distribution.")
    p.add_argument("--inverse_eps", type=float, default=0.01,
                   help="Min value for exp(-β^D·z) clamp; prevents J blow-up "
                        "at large depth.")
    p.add_argument("--no_save_descatter_previews", action="store_true")
    p.add_argument("--no_save_depth", action="store_true",
                   help="Skip per-frame depth map export. Default: save "
                        "metric depth (float32, meters) as "
                        "<outdir>/depth/<stem>.npy at the SAME resolution "
                        "as <outdir>/images/<stem>.* (bilinear-upsampled "
                        "from MASt3R's processed resolution). Ready to feed "
                        "into 3DGS depth-supervised training directly.")
    p.add_argument("--save_medium_maps", action="store_true",
                   help="Save per-pixel medium parameters for medium-MLP "
                        "supervision in 3DGS. Writes one npz per frame to "
                        "<outdir>/medium/<stem>.npz containing beta_D (H, W, 3) "
                        "and beta_B (H, W, 3) at the SAME resolution as "
                        "<outdir>/images/<stem>.* (bilinear upsample), plus "
                        "B_inf (3,) as the frame-level veiling colour. Default "
                        "OFF — opt in only when you actually train with non-zero "
                        "lambda_medium_* losses (file size is ~10-15 MB/frame).")

    # 3R-GS correspondence export
    p.add_argument("--save_correspondences", action="store_true",
                   help="Export MASt3R correspondences for 3R-GS epipolar "
                        "loss. Writes <outdir>/correspondences/*.npy "
                        "(required for gaussian_splatting.train).")

    # misc
    p.add_argument("--silent", action="store_true")
    p.add_argument("--pairs_pkl_path", type=str, default=None)
    p.add_argument("--sim_matrix", type=str, default=None)
    p.add_argument("--plot_matches", action="store_true")
    p.add_argument("--subset_list", type=str, default=None)
    p.add_argument("--init_slam_run", type=str, default=None,
                   help="run_dir of tools/slam/run_slam.py: the sparse alignment starts from its camera poses")
    p.add_argument("--init_slam_names", type=str, default=None,
                   help="json map from SLAM frame names to the image names used here (for example name_map_5fps.json)")
    p.add_argument("--init_slam_raw_scale", action="store_true",
                   help="use the SLAM positions as they are instead of removing the Sim3 scale of every frame")

    args = p.parse_args()

    # inference.reconstruction expects these on the namespace
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
# Descatter (in-process, reusing the loaded model)
# -----------------------------------------------------------------------------

def _to_torch(arr, device, dtype=torch.float32):
    if isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=dtype)
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=dtype)


@torch.no_grad()
def compute_descatter_per_frame(args, scene):
    """Run v7 physics head on each cached frame, recover J via Akkaynak inverse,
    and return per-frame uint8 J maps + per-pixel β maps + per-frame B^∞ summary.

    Returns:
        J_maps_per_frame:  list of [H, W, 3] uint8       — descattered RGB previews
        beta_D_per_frame:  list of [H, W, 3] float32     — per-pixel β^D maps
        beta_B_per_frame:  list of [H, W, 3] float32     — per-pixel β^B maps
        B_inf_per_frame:   [N, 3] float32                — per-frame veiling light
        physics_summary:   dict of [N, 3] arrays for diagnostics

    No file writes happen here — caller chooses what to persist.
    """
    from losses.physics_regression import PhysicsRegressor, median_normalize_depth
    from model.lora_model import LoRAMast3r

    device = torch.device(args.device)
    print(f"[descatter] loading encoder for J-recovery ...")
    encoder = LoRAMast3r.from_pretrained(
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    )
    if args.finetuned_model and os.path.isfile(args.finetuned_model):
        # KNOWN ISSUE #1 (docs/SWIMM3R_ARCH_ABSORPTION.md): the DPT-head
        # LoRA-landed guard + the enc/dec-only upstream-rank-inheritance fix now
        # live INSIDE load_pretrained_lora (model/lora_model.py), so they run on
        # every load path uniformly. The fix rebuilds the rank-4 DPT LoRA from
        # this ckpt's own metadata (not the upstream -1), and the guard hard-
        # fails (unless --allow_lora_drop) if any DPT tensor is silently dropped.
        # Expect: "[lora-verify] OK — all N LoRA tensor(s) landed (incl. DPT-head)".
        encoder.load_pretrained_lora(
            args.finetuned_model,
            allow_drop=bool(getattr(args, "allow_lora_drop", False)),
        )
    encoder = encoder.to(device).eval()

    # ── Physics head architecture auto-detection ──────────────────────────
    # The physics MLP final Linear's out_features unambiguously records which
    # β^B architecture produced the checkpoint:
    #   12 → Jerlov unified-β   (β^B ≡ β^D, single double-exp branch)
    #   15 → Sea-Thru-true      (z-independent β^B constant: β^D 12 + β^B 3)
    #   24 → free / constrain   (two double-exp branches)
    # We READ the dim directly and reconstruct the matching head, so the user
    # does NOT have to re-pass the training β^B flag at inference; the file
    # wins over any CLI hint and the silent size-mismatch footgun is gone.
    _peek = torch.load(args.physics_model, map_location="cpu", weights_only=True)
    if any(k.startswith("module.") for k in _peek):
        _peek = {k[len("module."):]: v for k, v in _peek.items()}
    _jerlov_unified = getattr(args, "phys_jerlov_unified_beta", False)
    _seathru_const = False
    _unified_medium = False
    _binf_spatial = False
    if args.phys_head_kind == "depth_dep":
        _mlp_w = [k for k in _peek
                  if k.startswith("physics.mlp.") and k.endswith(".weight")]
        if _mlp_w:
            _last = max(_mlp_w, key=lambda k: int(k.split(".")[2]))
            _out_dim = _peek[_last].shape[0]
            _jerlov_unified = (_out_dim == 12)
            _seathru_const = (_out_dim in (15, 18, 33))
            # 18 = unified medium head: 12 β^D + 3 β^B const + 3 B^∞ — B^∞ comes
            # from the physics MLP itself (no separate BackscatterHead). The
            # checkpoint therefore has NO backscatter.* keys; PhysicsRegressor
            # built with unified_medium_head=True sets self.backscatter=None.
            _unified_medium = (_out_dim in (18, 33))
            _binf_spatial = (_out_dim == 33)
            if _out_dim not in (12, 15, 18, 24, 33):
                raise ValueError(
                    f"[phys-arch-detect] unexpected physics MLP out_features="
                    f"{_out_dim} in {args.physics_model} (expected 12/15/18/24).")
            print(f"[phys-arch-detect] physics head out_features={_out_dim} → "
                  f"jerlov_unified_beta={_jerlov_unified}, "
                  f"seathru_betaB_const={_seathru_const}, "
                  f"unified_medium_head={_unified_medium}")
    del _peek
    args.phys_jerlov_unified_beta = _jerlov_unified
    # constrain_beta_B (default True) is mutually exclusive with the tied /
    # Sea-Thru β^B modes — force it OFF when the ckpt fixes/ties β^B.
    _constrain_beta_B = args.phys_constrain_beta_B and not (_jerlov_unified or _seathru_const)
    if args.phys_constrain_beta_B and (_jerlov_unified or _seathru_const):
        print("[phys-arch-detect] --phys_constrain_beta_B ignored "
              "(checkpoint arch ties / fixes β^B; constrain mode is "
              "incompatible).")
    args.phys_constrain_beta_B = _constrain_beta_B

    physics = PhysicsRegressor(
        in_dim=1024,
        hidden=args.phys_head_hidden,
        max_b=args.phys_max_backscatter,
        kind=args.phys_head_kind,
        b_max=args.phys_b_max,
        beta_max=args.phys_beta_clamp_max,
        constrain_beta_B=_constrain_beta_B,
        betaB_ratio_min=args.phys_betaB_ratio_min,
        betaB_ratio_max=args.phys_betaB_ratio_max,
        jerlov_unified_beta=_jerlov_unified,
        seathru_betaB_const=_seathru_const,
        seathru_betaB_le_betaD=bool(getattr(args, "phys_seathru_betaB_le_betaD", False)),
        unified_medium_head=_unified_medium,
        binf_spatial=_binf_spatial,
    ).to(device).eval()
    if bool(getattr(args, "phys_seathru_betaB_le_betaD", False)):
        print("[seathru] β^B ≤ β^D cap ACTIVE (forward-time β^B = min(β^B, β^D)) "
              "— must match training; omitting it gives uncapped β^B.")
    if _seathru_const and _unified_medium:
        print(f"[seathru] PhysicsRegressor: Sea-Thru-true β^B-const + unified "
              f"B^∞ head (z-dependent β^D + z-independent β^B + B^∞ from MLP; "
              f"MLP outputs 18 scalars, no separate BackscatterHead)")
    elif _seathru_const:
        print(f"[seathru] PhysicsRegressor: Sea-Thru-true β^B-const "
              f"(z-dependent β^D + z-independent β^B; MLP outputs 15 scalars)")
    elif getattr(args, "phys_jerlov_unified_beta", False):
        print(f"[v11] PhysicsRegressor: Jerlov unified-β (β^B := β^D, "
              f"MLP outputs 12 scalars, single branch)")
    elif args.phys_constrain_beta_B:
        print(f"[v9] PhysicsRegressor: β_B = β_D · sigmoid(g(z)) "
              f"(constrain_beta_B=True, ratio ∈ "
              f"[{args.phys_betaB_ratio_min}, {args.phys_betaB_ratio_max}])")
    bd_scale_list = list(args.phys_beta_D_scale)
    if len(bd_scale_list) not in (1, 3):
        raise ValueError(f"--phys_beta_D_scale must take 1 or 3 floats, got {bd_scale_list}")
    if any(s != 1.0 for s in bd_scale_list):
        if len(bd_scale_list) == 1:
            print(f"[v9.2-hack] β_D scaled by {bd_scale_list[0]} at inference "
                  f"(OOD un-attenuation boost; β_B unchanged → β_B ≤ β_D still holds)")
        else:
            print(f"[v9.2-hack] β_D per-channel scale R={bd_scale_list[0]} "
                  f"G={bd_scale_list[1]} B={bd_scale_list[2]} at inference")
    max_amp_list = list(args.phys_max_amp)
    if len(max_amp_list) not in (1, 3):
        raise ValueError(f"--phys_max_amp must take 1 or 3 floats, got {max_amp_list}")
    if any(a > 0 for a in max_amp_list):
        if len(max_amp_list) == 1:
            print(f"[v9.2-hack] amplification 1/trans_D capped at {max_amp_list[0]} "
                  f"per pixel (prevents deep-pixel saturation)")
        else:
            print(f"[v9.2-hack] per-channel amplification cap R={max_amp_list[0]} "
                  f"G={max_amp_list[1]} B={max_amp_list[2]}")
    R_depth_slope = list(args.phys_max_amp_R_depth_slope)
    if len(R_depth_slope) >= 2:
        b, s = R_depth_slope[0], R_depth_slope[1]
        print(f"[v9.2-hack] depth-aware R amp cap: BASE={b} SLOPE={s}/m")
        print(f"             cap(z=0)={b:.1f}, cap(z=5m)={b+5*s:.1f}, "
              f"cap(z=10m)={b+10*s:.1f}")
    if args.phys_adaptive_R_boost > 0:
        print(f"[v9.2-hack] adaptive R boost ON  k={args.phys_adaptive_R_boost} "
              f"(extra β_D_R scale = clamp(1 + k·R_deficit, max={args.phys_adaptive_R_max}))")
        print( "             — only pixels with raw R << raw (G+B)/2 get extra boost,")
        print( "               natural-R pixels are NOT touched.")
    if args.phys_anchor_color_balance > 0:
        print(f"[v9.2-hack] anchor color balance ON  α={args.phys_anchor_color_balance}")
        print( "             — natural-J pixels (J_R∈[0.15,0.7], R≥0.5·mean(G,B))")
        print( "               define per-frame target; per-channel gain blended at α")
        print( "               propagates that colour distribution to the whole frame.")
    if args.phys_per_frame_normalize:
        print("[v9.2] per-frame J normalisation ON  (Pass 1: collect means,")
        print("       Pass 2: rescale each frame to global target — removes per-frame")
        print("       inversion bias that surfaces as inter-voxel colour variance.)")
    state = torch.load(args.physics_model, map_location=device, weights_only=True)
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    physics.load_state_dict(state)

    _, depthmaps_list, _ = scene.get_dense_pts3d(clean_depth=True)
    N = scene.n_imgs
    proc_H, proc_W = scene.imgs[0].shape[:2]
    print(f"[descatter] N={N}, processed shape={proc_H}x{proc_W}")

    J_maps_per_frame: list[np.ndarray] = []
    beta_D_per_frame: list[np.ndarray] = []
    beta_B_per_frame: list[np.ndarray] = []
    B_inf_per_frame  = np.zeros((N, 3), dtype=np.float32)
    physics_summary = {
        "beta_D_mean":   np.zeros((N, 3), dtype=np.float32),
        "beta_B_mean":   np.zeros((N, 3), dtype=np.float32),
        "B_inf":         np.zeros((N, 3), dtype=np.float32),
    }

    for i in range(N):
        img_np = scene.imgs[i]
        if img_np.dtype != np.float32:
            img_np = img_np.astype(np.float32)
        if img_np.max() > 1.5:
            img_np = img_np / 255.0
        I = _to_torch(img_np, device).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0

        z = depthmaps_list[i]
        if not isinstance(z, torch.Tensor):
            z = _to_torch(z, device)
        else:
            z = z.to(device=device, dtype=torch.float32)
        if z.dim() == 1:
            z = z.view(proc_H, proc_W)

        true_shape = torch.tensor([[proc_H, proc_W]], device=device, dtype=torch.long)
        feat1, _, _, _ = encoder._encode_image_pairs(I, I, true_shape, true_shape)

        # B2-norm — feed the physics head the SAME median-normalised depth used
        # in training (β in normalised units). Geometry stays raw; only the
        # descatter (physics + IFM inversion below) uses z_phys.
        z_phys = z.unsqueeze(0)                        # [1, H, W]
        if getattr(args, "phys_depth_norm", "none") == "median":
            z_phys = median_normalize_depth(z_phys, args.phys_z_min, args.phys_z_max)

        out = physics(feat1, z=z_phys)
        beta_D = out["beta_D"]
        beta_B = out["beta_B"]
        B_inf  = out["B_inf"]

        # v9.2 inference-only β_D scaling (deployment hack for OOD turbid
        # water). β_B is NOT scaled — it was generated by the architectural
        # constraint β_B = β_D_orig · σ(g(z)); leaving β_B alone after the
        # β_D scale-up makes β_B/β_D shrink (still ≤ β_D, just more so).
        if any(s != 1.0 for s in bd_scale_list):
            if len(bd_scale_list) == 1:
                beta_D = beta_D * bd_scale_list[0]
            else:  # per-channel R, G, B
                scale = torch.tensor(
                    bd_scale_list, dtype=beta_D.dtype, device=beta_D.device,
                ).view(1, 3, *([1] * (beta_D.dim() - 2)))
                beta_D = beta_D * scale

        z_ = z.unsqueeze(0).unsqueeze(0)
        if beta_D.dim() == 2:
            beta_D = beta_D[:, :, None, None].expand(-1, -1, proc_H, proc_W)
        if beta_B.dim() == 2:
            beta_B = beta_B[:, :, None, None].expand(-1, -1, proc_H, proc_W)

        # v9.2 deployment — per-pixel adaptive R boost. Identifies pixels
        # whose raw R is anomalously small relative to G/B (i.e. R was
        # heavily absorbed by the water column) and scales β_D_R locally
        # so they get a stronger un-attenuation. Pixels whose R is already
        # comparable to G/B (= already producing natural-looking J) are
        # left untouched, so the artefact-free regions stay artefact-free
        # while cyan-collapsed regions are pulled toward natural colour.
        if args.phys_adaptive_R_boost > 0:
            I01_tmp = (I + 1.0) * 0.5      # [B, 3, H, W] in [0, 1]
            GB_mean = 0.5 * (I01_tmp[:, 1:2] + I01_tmp[:, 2:3])
            R_deficit = (GB_mean - I01_tmp[:, 0:1]).clamp(min=0)
            adaptive_extra = (1.0 + args.phys_adaptive_R_boost * R_deficit
                             ).clamp(max=args.phys_adaptive_R_max)
            # broadcast extra ([B,1,H,W]) only onto the R channel of β_D
            # ([B,3,H,W]); G and B unchanged.
            beta_D = beta_D.clone()
            beta_D[:, 0:1] = beta_D[:, 0:1] * adaptive_extra

        # direction dependent veiling light of a 33 output head, else one B^∞ per view
        Binf_ = out["B_inf_map"] if "B_inf_map" in out else B_inf[:, :, None, None]
        trans_D = torch.exp(-beta_D * z_)
        trans_B = torch.exp(-beta_B * z_)
        I01 = (I + 1.0) * 0.5
        # v9.2 deployment: per-pixel amplification cap. The default path
        # 1/trans_D.clamp(min=inverse_eps) gives a uniform max amplification
        # of 1/inverse_eps (≈100x at default 0.01) — fine for shallow pixels
        # but pushes deep ones (or already-bright pixels under R-only β_D
        # boost) past 1.0 → hard clamp → "pure red" saturation artefact.
        use_amp_path = (any(a > 0 for a in max_amp_list)
                        or len(R_depth_slope) >= 2)
        if use_amp_path:
            amp = 1.0 / trans_D.clamp(min=1e-6)
            # Per-channel uniform cap
            if any(a > 0 for a in max_amp_list):
                if len(max_amp_list) == 1:
                    amp = amp.clamp(max=max_amp_list[0])
                else:
                    cap = torch.tensor(
                        max_amp_list, dtype=amp.dtype, device=amp.device,
                    ).view(1, 3, *([1] * (amp.dim() - 2)))
                    amp = torch.minimum(amp, cap)
            # Depth-aware R cap: tight on shallow (anti-saturation),
            # loose on deep (R recovery). cap_R(z) = base + slope·z.
            if len(R_depth_slope) >= 2:
                base, slope = R_depth_slope[0], R_depth_slope[1]
                # z is [H, W] at proc resolution; broadcast to [1, 1, H, W]
                z_4d = z.unsqueeze(0).unsqueeze(0)
                cap_R_pp = (base + slope * z_4d).clamp(min=0.1)
                amp_R = torch.minimum(amp[:, 0:1], cap_R_pp)
                amp = torch.cat([amp_R, amp[:, 1:2], amp[:, 2:3]], dim=1)
            J01 = (I01 - Binf_ * (1.0 - trans_B)) * amp
        else:
            J01 = (I01 - Binf_ * (1.0 - trans_B)) / trans_D.clamp(min=args.inverse_eps)

        # Scene-aware R-saturation soft cap. Detects per-frame whether
        # raw J_R has many over-saturated pixels (warm scenes); if so,
        # scales the R channel down so a target high quantile of J_R
        # maps to 1.0. Cyan-leaning frames have ~0% over-saturation so
        # the cap does not fire — identical output to ε baseline.
        if args.phys_R_sat_soft_cap > 0:
            J_R_raw = J01[:, 0:1]
            sat_frac = (J_R_raw > 1.0).float().mean()
            if float(sat_frac.item()) > args.phys_R_sat_soft_cap:
                p_target = float(args.phys_R_sat_target_quantile)
                J_R_p = torch.quantile(J_R_raw.flatten(), p_target)
                if float(J_R_p.item()) > 1.0:
                    scale = 1.0 / J_R_p
                    J01 = torch.cat(
                        [J_R_raw * scale, J01[:, 1:2], J01[:, 2:3]], dim=1,
                    )

        J01 = J01.clamp(0.0, 1.0)

        # v9.2 deployment — anchor-based color propagation. Find the small
        # subset of pixels that already exhibit natural in-air colour
        # (decent J_R AND J_R not too small relative to mean(J_G, J_B)),
        # take their per-channel means as the per-frame target, then apply
        # a per-channel multiplicative gain to the whole image so the
        # global mean matches the anchor mean. This pulls cyan-collapsed
        # pixels toward the recovered ones without re-running physics.
        if args.phys_anchor_color_balance > 0:
            J_R = J01[:, 0:1]; J_G = J01[:, 1:2]; J_B = J01[:, 2:3]
            GB_mean_pix = 0.5 * (J_G + J_B)
            anchor = (J_R > 0.15) & (J_R < 0.7) & (J_R > 0.5 * GB_mean_pix)
            n_anchor = int(anchor.sum().item())
            if n_anchor >= args.phys_anchor_min_count:
                tgt_R = J_R[anchor].mean()
                tgt_G = J_G[anchor].mean()
                tgt_B = J_B[anchor].mean()
                cur_R = J_R.mean().clamp(min=1e-3)
                cur_G = J_G.mean().clamp(min=1e-3)
                cur_B = J_B.mean().clamp(min=1e-3)
                # Strength α blends: gain = 1 + α·(target/current − 1)
                alpha = float(args.phys_anchor_color_balance)
                gain_R = 1.0 + alpha * (tgt_R / cur_R - 1.0)
                gain_G = 1.0 + alpha * (tgt_G / cur_G - 1.0)
                gain_B = 1.0 + alpha * (tgt_B / cur_B - 1.0)
                J01 = torch.cat([J_R * gain_R, J_G * gain_G, J_B * gain_B], dim=1)
                J01 = J01.clamp(0.0, 1.0)

        J_np = J01.squeeze(0).permute(1, 2, 0).cpu().numpy()
        J_uint8 = (J_np * 255).clip(0, 255).astype(np.uint8)
        bD_np = beta_D.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
        bB_np = beta_B.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
        Binf_np = B_inf.squeeze(0).cpu().numpy().astype(np.float32)

        J_maps_per_frame.append(J_uint8)
        beta_D_per_frame.append(bD_np)
        beta_B_per_frame.append(bB_np)
        B_inf_per_frame[i] = Binf_np

        physics_summary["beta_D_mean"][i] = bD_np.reshape(-1, 3).mean(0)
        physics_summary["beta_B_mean"][i] = bB_np.reshape(-1, 3).mean(0)
        physics_summary["B_inf"][i]       = Binf_np

        if (i + 1) % 25 == 0 or i == N - 1:
            print(f"  [{i+1}/{N}] β^D={physics_summary['beta_D_mean'][i].round(3)}  "
                  f"β^B={physics_summary['beta_B_mean'][i].round(3)}  "
                  f"B^∞={physics_summary['B_inf'][i].round(3)}")

    # v9.2 deployment: per-frame J normalisation. Per-frame Akkaynak-Treibitz
    # inversion produces frame-dependent colour bias (different (β, B∞)
    # estimates ⇒ different J distributions). When the cloud is merged from
    # all frames, adjacent voxels coming from different frames inherit those
    # different distributions, surfacing as inter-voxel colour variance.
    # Two-pass fix:
    #   Pass 1: compute per-frame channel means (in float32 [0, 1])
    #   Global target = mean of per-frame means
    #   Pass 2: gain = global_target / frame_mean; J ← J · gain (then clamp)
    # Result: every frame contributes the SAME mean colour distribution,
    # so the merged cloud has consistent inter-voxel colour without losing
    # per-pixel detail (only the per-frame mean is shifted).
    if args.phys_per_frame_normalize:
        per_frame_means = np.stack([
            (J.astype(np.float32) / 255.0).reshape(-1, 3).mean(axis=0)
            for J in J_maps_per_frame
        ])                                           # [N, 3]
        global_target = per_frame_means.mean(axis=0)  # [3]
        for i, J in enumerate(J_maps_per_frame):
            gain = global_target / np.maximum(per_frame_means[i], 1e-3)
            J01 = (J.astype(np.float32) / 255.0) * gain[None, None, :]
            J01 = np.clip(J01, 0.0, 1.0)
            J_maps_per_frame[i] = (J01 * 255.0).astype(np.uint8)
        print(f"[v9.2] per-frame J normalised to global target "
              f"(R={global_target[0]:.3f}, G={global_target[1]:.3f}, "
              f"B={global_target[2]:.3f})")

    return J_maps_per_frame, beta_D_per_frame, beta_B_per_frame, B_inf_per_frame, physics_summary


def aggregate_per_point_aux(masks, fps_indices, per_frame_arr_list, default_dtype=np.float32):
    """Apply the same conf mask + FPS indices used by export_colmap to a per-frame
    [H, W, 3] array list, producing a [N_kept_points, 3] aligned-row aux array.

    `masks[i]` is a flat boolean mask of length H*W; it MUST come from the same
    export_colmap call that wrote the canonical points3D, so row order is identical.
    """
    pieces = []
    for arr_hw3, mask in zip(per_frame_arr_list, masks):
        flat = arr_hw3.reshape(-1, 3)
        pieces.append(flat[mask])
    out = np.concatenate(pieces, axis=0) if pieces else np.zeros((0, 3), dtype=default_dtype)
    if fps_indices is not None:
        out = out[fps_indices]
    return out.astype(default_dtype, copy=False)


def enforce_rgb_order_topdown(beta: np.ndarray) -> np.ndarray:
    """Enforce Jerlov R >= G >= B ordering on a [..., 3] β array (top-down clamp).

    Underwater attenuation physics: red light decays fastest, then green, then
    blue. β^D and β^B should therefore satisfy β_R >= β_G >= β_B everywhere.
    This top-down version preserves R and clamps G/B downward:
        out[..., 1] = min(G, R)
        out[..., 2] = min(B, G_clamped)

    Used as an inference-time projection on per-point β arrays so the export
    respects the ordering even when the trained head occasionally outputs
    a violating triple (rare under Jerlov unified-β architecture but possible
    under sigmoid-constrained ratio modes).
    """
    if beta.shape[-1] != 3:
        raise ValueError(f"enforce_rgb_order_topdown expects [..., 3], got {beta.shape}")
    out = beta.copy()
    out[..., 1] = np.minimum(out[..., 1], out[..., 0])
    out[..., 2] = np.minimum(out[..., 2], out[..., 1])
    return out


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------

def slam_init_poses(run_dir, names_json, filelist, unscale=True):
    """Camera to world poses of a run_slam.py run for the images of filelist, as init of the sparse alignment.

    T_WC of every frame is a Sim3 [tx ty tz qx qy qz qw s] whose scale s maps the frame's own pointmap
    units to the SLAM world. With unscale the positions are rebuilt along the stream from the steps
    between consecutive posed frames divided by their mean scale, so they are in the units of the
    network's pointmaps like the depths the alignment starts from. Rotations are used as they are.
    """
    from scipy.spatial.transform import Rotation

    frames = json.load(open(os.path.join(run_dir, "frames.json")))
    rename = json.load(open(names_json)) if names_json else {}
    posed = [r for r in frames if r["T_WC"] is not None]
    T = np.array([r["T_WC"] for r in posed], dtype=np.float64)
    pos, scale = T[:, :3], T[:, 7]
    if unscale:
        steps = np.diff(pos, axis=0) / (0.5 * (scale[1:] + scale[:-1]))[:, None]
        pos = np.concatenate([pos[:1] / scale[0], pos[:1] / scale[0] + np.cumsum(steps, axis=0)])
    by_name = {}
    for r, p, t in zip(posed, pos, T):
        c2w = np.eye(4)
        c2w[:3, :3] = Rotation.from_quat(t[3:7]).as_matrix()
        c2w[:3, 3] = p
        by_name[rename.get(r["name"], r["name"])] = c2w
    init, missing = {}, []
    for path in filelist:
        c2w = by_name.get(os.path.basename(path))
        if c2w is None:
            missing.append(os.path.basename(path))
        else:
            init[path] = {"cam2w": torch.tensor(c2w, dtype=torch.float32)}
    if missing:
        raise SystemExit(f"[init] no SLAM pose for {len(missing)} images, for example {missing[:5]}")
    steps = np.linalg.norm(np.diff(np.stack([v["cam2w"][:3, 3].numpy() for v in init.values()]), axis=0), axis=1)
    print(f"[init] {len(init)} SLAM poses from {run_dir} ({'Sim3 scale removed' if unscale else 'raw positions'}), "
          f"median step between images {np.median(steps):.4f}")
    return init


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Pure MASt3R-SfM ablation: hard-override both extension paths so no
    # Dark3R LoRA encoder and no Swimm3R physics descattering can ever
    # load, regardless of other flags / defaults.
    if getattr(args, "pure_mast3r", False):
        args.finetuned_model = None    # reconstruction.load_model → base MASt3R
        args.physics_model = ""         # do_descatter = False
        args.save_medium_maps = False   # nothing to save without physics
        print(
            "[pure_mast3r] BASELINE MODE — Dark3R LoRA + Swimm3R physics "
            "DISABLED. Using base MASt3R-SfM only."
        )

    print(f"[args] {args}")

    if args.physics_model and not os.path.isfile(args.physics_model):
        raise FileNotFoundError(args.physics_model)
    do_descatter = bool(args.physics_model)

    # 1) collect / extract input frames into <outdir>/images/
    if args.video is not None:
        filelist = extract_frames(args.video, args.outdir, args.fps)
    else:
        filelist = collect_images(args.imagedir, args.outdir)

    # apply optional subsetting (mirrors forward_dark3r_colmap.py)
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

    # ─── InstantSplat-style train/test split ──────────────────────────────
    # Run AFTER subset/subsample so the user-controlled selection is what we
    # split. Records the full list + train/test partition in split.json so
    # init_test_pose_underwater.py can re-derive the same partition.
    split_info = None
    args._test_indices_in_filelist: set[int] = set()
    if args.train_test_split == "llff":
        full_sorted = sorted(filelist, key=lambda p: os.path.basename(p))
        test_files  = [f for i, f in enumerate(full_sorted) if i % args.llffhold == 0]
        train_files = [f for i, f in enumerate(full_sorted) if i % args.llffhold != 0]

        if args.n_train_views > 0 and args.n_train_views < len(train_files):
            import numpy as _np
            idx = _np.linspace(0, len(train_files) - 1, args.n_train_views).astype(int)
            train_files = [train_files[int(i)] for i in idx]

        if args.keep_test_cameras:
            # 1-pass mode: keep BOTH train and test in the filelist (sorted
            # order), so sparse_ga jointly registers all cameras. Mark the
            # test indices so the point-cloud export can mask them out.
            keep_set = set(train_files) | set(test_files)
            filelist = [f for f in full_sorted if f in keep_set]
            args._test_indices_in_filelist = {
                i for i, f in enumerate(filelist) if f in set(test_files)
            }
            note = "test masked from points but kept as cameras"
        else:
            filelist = train_files
            note = "test held for init_test_pose_underwater.py"

        split_info = {
            "split_mode": "llff",
            "llffhold": args.llffhold,
            "n_train_views": len(train_files),
            "n_test_views":  len(test_files),
            "keep_test_cameras": bool(args.keep_test_cameras),
            "all_files":     [os.path.basename(p) for p in full_sorted],
            "train_files":   [os.path.basename(p) for p in train_files],
            "test_files":    [os.path.basename(p) for p in test_files],
            "all_files_abs":   [str(p) for p in full_sorted],
            "train_files_abs": [str(p) for p in train_files],
            "test_files_abs":  [str(p) for p in test_files],
        }
        with open(os.path.join(args.outdir, "split.json"), "w") as f:
            json.dump(split_info, f, indent=2)
        print(f"[split] llff hold={args.llffhold} → train={len(train_files)}, "
              f"test={len(test_files)} ({note})")

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
    model = None
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
        if args.init_slam_run:
            args.init_poses = slam_init_poses(args.init_slam_run, args.init_slam_names, filelist_loaded,
                                              not args.init_slam_raw_scale)
        pairs = create_image_pairs(args, imgs, filelist_loaded, model)
        scene = run_reconstruction(args, filelist_loaded, pairs, model)

        with open(os.path.join(args.outdir, "scene.pkl"), "wb") as f:
            pickle.dump(scene, f)

    elapsed = (datetime.datetime.now() - started).total_seconds()
    print(f"[reconstruct] done in {elapsed:.1f}s")

    # 3.4) export MASt3R correspondences for 3R-GS epipolar loss
    if getattr(args, "save_correspondences", False):
        from gaussian_splatting.export_correspondences import (
            export_correspondences_from_scene,
        )
        try:
            export_correspondences_from_scene(args.outdir)
            print("[correspondences] exported for 3R-GS epipolar loss")
        except Exception as e:
            print(f"[correspondences] WARNING: export failed: {e}")

    # 3.5) per-frame depth export at the same resolution as <outdir>/images/.
    if not args.no_save_depth:
        from PIL import Image as _PILImage
        depth_dir = os.path.join(args.outdir, "depth")
        os.makedirs(depth_dir, exist_ok=True)
        images_dir = os.path.join(args.outdir, "images")
        _, depthmaps_list, _ = scene.get_dense_pts3d(clean_depth=True)
        n_saved = 0
        for i in range(len(image_names)):
            z = depthmaps_list[i]
            if hasattr(z, "cpu"):
                z = z.cpu().numpy()
            z = np.asarray(z, dtype=np.float32)
            if z.ndim == 1:
                # Some scenes return flat (H*W,) — reshape using processed imgs[i]
                H, W = scene.imgs[i].shape[:2]
                z = z.reshape(H, W)

            img_path = os.path.join(images_dir, image_names[i])
            try:
                with _PILImage.open(img_path) as im:
                    orig_W, orig_H = im.size
            except FileNotFoundError:
                # Fall back to processed shape if the copy step is skipped.
                orig_H, orig_W = z.shape

            # Bilinear upsample to original input resolution.
            if (z.shape[1], z.shape[0]) != (orig_W, orig_H):
                z_full = cv2.resize(z, (orig_W, orig_H),
                                    interpolation=cv2.INTER_LINEAR)
            else:
                z_full = z
            z_full = z_full.astype(np.float32)

            stem = os.path.splitext(image_names[i])[0]
            np.save(os.path.join(depth_dir, f"{stem}.npy"), z_full)
            n_saved += 1
        print(f"[depth] {n_saved} per-frame depth maps "
              f"→ {depth_dir}/ "
              f"(metric meters, float32, original-resolution bilinear upsample)")

    # 4) descatter (in-memory) — must run BEFORE export_colmap so we can
    #    write descattered colors as the canonical points3D.
    J_maps = bD_maps = bB_maps = None
    Binf_per_frame = None
    physics_summary = None
    if do_descatter:
        (
            J_maps, bD_maps, bB_maps, Binf_per_frame, physics_summary
        ) = compute_descatter_per_frame(args, scene)

    # 4.5) per-pixel medium maps for 3DGS medium-MLP supervision.
    if (args.save_medium_maps and do_descatter
            and bD_maps is not None and bB_maps is not None
            and Binf_per_frame is not None):
        from PIL import Image as _PILImage_m
        medium_dir = os.path.join(args.outdir, "medium")
        os.makedirs(medium_dir, exist_ok=True)
        images_dir_m = os.path.join(args.outdir, "images")
        n_saved_m = 0
        for i in range(len(image_names)):
            img_path = os.path.join(images_dir_m, image_names[i])
            try:
                with _PILImage_m.open(img_path) as im:
                    orig_W, orig_H = im.size
            except FileNotFoundError:
                orig_H, orig_W = bD_maps[i].shape[:2]

            def _resize_hw3(arr, W, H):
                a = np.asarray(arr, dtype=np.float32)
                if a.ndim == 2:
                    a = a[..., None].repeat(3, axis=-1)
                if (a.shape[1], a.shape[0]) != (W, H):
                    a = cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
                return a.astype(np.float32)

            bD_full = _resize_hw3(bD_maps[i], orig_W, orig_H)
            bB_full = _resize_hw3(bB_maps[i], orig_W, orig_H)
            Binf_vec = np.asarray(Binf_per_frame[i], dtype=np.float32).reshape(3)

            stem = os.path.splitext(image_names[i])[0]
            np.savez_compressed(
                os.path.join(medium_dir, f"{stem}.npz"),
                beta_D=bD_full,           # (H, W, 3) per-pixel direct attenuation
                beta_B=bB_full,           # (H, W, 3) per-pixel backscatter
                B_inf=Binf_vec,           # (3,) frame-level veiling colour
            )
            n_saved_m += 1
        print(f"[medium] {n_saved_m} per-pixel medium maps "
              f"(beta_D, beta_B, B_inf) → {medium_dir}/<stem>.npz "
              f"(original-resolution, savez_compressed)")

    # 5) export COLMAP geometry — TWO points3D files when descatter is on:
    #     points3D.{txt,bin}      ← canonical = DESCATTERED colors
    #     raw_points3D.{txt,bin}  ← auxiliary = RAW underwater colors
    keep_orig_res = not args.no_keep_original_resolution
    if do_descatter:
        # In --keep_test_cameras mode, test views get their poses written
        # but their per-pixel points are masked out of the merged cloud.
        _exclude = (
            args._test_indices_in_filelist
            if getattr(args, "keep_test_cameras", False)
            and args._test_indices_in_filelist
            else None
        )

        # First: write cameras/images + canonical points3D (descattered)
        _fps_max = args.fps_max_points or None
        _, _, fps_idx, masks = export_colmap(
            scene, args.outdir, image_names,
            min_conf_thr=args.min_conf_thr,
            fps_ratio=args.fps_ratio,
            keep_original_resolution=keep_orig_res,
            colors_per_frame=J_maps,                    # ← descattered
            points3d_filename="points3D",
            write_cameras_images=True,
            exclude_indices=_exclude,
            fps_max_points=_fps_max,
        )
        # Second: write raw_points3D using the SAME fps_idx so rows align
        export_colmap(
            scene, args.outdir, image_names,
            min_conf_thr=args.min_conf_thr,
            fps_ratio=args.fps_ratio,
            keep_original_resolution=keep_orig_res,
            colors_per_frame=None,                       # ← raw (default sampling)
            points3d_filename="raw_points3D",
            write_cameras_images=False,
            fps_indices=fps_idx,
            exclude_indices=_exclude,
            fps_max_points=_fps_max,
        )

        # Per-point physics aux arrays — row-aligned with points3D.{txt,bin}.
        bD_per_point   = aggregate_per_point_aux(masks, fps_idx, bD_maps,        np.float32)
        bB_per_point   = aggregate_per_point_aux(masks, fps_idx, bB_maps,        np.float32)
        # B^∞ is per-frame; broadcast each frame's value to a [H, W, 3] array
        Binf_maps_for_aux = [
            np.broadcast_to(Binf_per_frame[i].reshape(1, 1, 3),
                            (J_maps[i].shape[0], J_maps[i].shape[1], 3))
            for i in range(len(J_maps))
        ]
        Binf_per_point = aggregate_per_point_aux(masks, fps_idx, Binf_maps_for_aux, np.float32)

        if getattr(args, "phys_enforce_rgb_order", False):
            n_violate_D = int(((bD_per_point[..., 0] < bD_per_point[..., 1])
                               | (bD_per_point[..., 1] < bD_per_point[..., 2])).sum())
            n_violate_B = int(((bB_per_point[..., 0] < bB_per_point[..., 1])
                               | (bB_per_point[..., 1] < bB_per_point[..., 2])).sum())
            bD_per_point = enforce_rgb_order_topdown(bD_per_point)
            bB_per_point = enforce_rgb_order_topdown(bB_per_point)
            print(f"[rgb-order] β^D R>=G>=B clamp applied: "
                  f"{n_violate_D:,} / {len(bD_per_point):,} pts violated pre-clamp")
            print(f"[rgb-order] β^B R>=G>=B clamp applied: "
                  f"{n_violate_B:,} / {len(bB_per_point):,} pts violated pre-clamp")

        np.save(os.path.join(args.outdir, "beta_D_per_point.npy"), bD_per_point)
        np.save(os.path.join(args.outdir, "beta_B_per_point.npy"), bB_per_point)
        np.save(os.path.join(args.outdir, "B_inf_per_point.npy"),  Binf_per_point)
        print(f"[points] beta_D_per_point.npy  {bD_per_point.shape}")
        print(f"[points] beta_B_per_point.npy  {bB_per_point.shape}")
        print(f"[points] B_inf_per_point.npy   {Binf_per_point.shape}")

        # Per-point confidence — same conf mask + FPS idx → row-aligned with
        # points3D.{txt,bin}. Used by downstream Gaussian-Splatting init
        # (e.g. confidence-driven opacity init in ubs-uw / InstantSplat-style).
        _, _, confs_list = scene.get_dense_pts3d(clean_depth=True)
        conf_maps_for_aux = []
        for c in confs_list:
            if hasattr(c, "cpu"):
                c = c.cpu().numpy()
            c = np.asarray(c, dtype=np.float32).reshape(scene.imgs[0].shape[0],
                                                         scene.imgs[0].shape[1], 1)
            conf_maps_for_aux.append(np.repeat(c, 3, axis=-1))   # aggregate_per_point_aux expects [H, W, 3]
        conf_per_point = aggregate_per_point_aux(
            masks, fps_idx, conf_maps_for_aux, np.float32,
        )[:, 0:1]    # collapse [N, 3] back to [N, 1] (all 3 channels were duplicates)
        # Filename matches ubs-uw's expected key "confidence_dsp"
        # (scene/dataset_readers.py:_load_swimm3r_aux) and InstantSplatPP's
        # post-FPS-downsample convention.
        np.save(os.path.join(args.outdir, "confidence_dsp.npy"), conf_per_point.reshape(-1))
        print(f"[points] confidence_dsp.npy  {conf_per_point.reshape(-1).shape}  "
              f"(mean={conf_per_point.mean():.2f}, "
              f"min={conf_per_point.min():.2f}, max={conf_per_point.max():.2f})")

        # Per-point descattered colors (legacy file kept for backward compat)
        desc_colors_per_point = aggregate_per_point_aux(
            masks, fps_idx,
            [J_maps[i].astype(np.uint8) for i in range(len(J_maps))],
            default_dtype=np.uint8,
        )
        np.save(os.path.join(args.outdir, "descattered_colors.npy"), desc_colors_per_point)

        # Per-frame physics summary
        np.savez(os.path.join(args.outdir, "physics_params.npz"), **physics_summary)
        print(f"[descatter] physics_params.npz saved (per-frame β/B^∞ means)")

        # Per-frame J preview JPEGs
        if not args.no_save_descatter_previews:
            preview_dir = Path(args.outdir) / "descattered"
            preview_dir.mkdir(exist_ok=True)
            for i, J_uint8 in enumerate(J_maps):
                name = image_names[i] if i < len(image_names) else f"{i:06d}.jpg"
                cv2.imwrite(str(preview_dir / name), J_uint8[..., ::-1])
            print(f"[descatter] {len(J_maps)} J previews → {preview_dir}/")

        # Sequence-mean physics print-out
        bD_seq = physics_summary["beta_D_mean"].mean(axis=0)
        bB_seq = physics_summary["beta_B_mean"].mean(axis=0)
        Binf_seq = physics_summary["B_inf"].mean(axis=0)
        print()
        print("=" * 60)
        print(f"  Sequence-mean physics:")
        print(f"    β^D (R, G, B) = {bD_seq.round(3)}     [1/m]")
        print(f"    β^B (R, G, B) = {bB_seq.round(3)}     [1/m]")
        print(f"    B^∞ (R, G, B) = {Binf_seq.round(3)}")
        print("=" * 60)
    else:
        # Geometry-only: single points3D with raw colors (legacy behaviour)
        _exclude = (
            args._test_indices_in_filelist
            if getattr(args, "keep_test_cameras", False)
            and args._test_indices_in_filelist
            else None
        )
        _fps_max = args.fps_max_points or None
        export_colmap(
            scene, args.outdir, image_names,
            min_conf_thr=args.min_conf_thr,
            fps_ratio=args.fps_ratio,
            keep_original_resolution=keep_orig_res,
            exclude_indices=_exclude,
            fps_max_points=_fps_max,
        )
        print("[descatter] skipped (no --physics_model provided)")

    print()
    print(f"[done] Swimm3R workspace: {args.outdir}")
    print(f"       images:                 {os.path.join(args.outdir, 'images')}")
    print(f"       sparse/0:                {os.path.join(args.outdir, 'sparse', '0')}")
    if do_descatter:
        print(f"         points3D.{{txt,bin}}      = DESCATTERED colors (canonical)")
        print(f"         raw_points3D.{{txt,bin}}  = RAW underwater colors")
        print(f"       beta_D_per_point.npy / beta_B_per_point.npy / B_inf_per_point.npy")
        print(f"       descattered/:            {os.path.join(args.outdir, 'descattered')}")
        print(f"       physics_params.npz:      {os.path.join(args.outdir, 'physics_params.npz')}")
        if not args.no_save_depth:
            print(f"       depth/:                  {os.path.join(args.outdir, 'depth')}/<stem>.npy")
        if args.save_medium_maps:
            print(f"       medium/:                 {os.path.join(args.outdir, 'medium')}/<stem>.npz")
    print()
    print(f"[viewer] python utils/viewer_viser.py --model swimm3r:{args.outdir}")


if __name__ == "__main__":
    main()
