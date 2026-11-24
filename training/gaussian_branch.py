"""Gaussian map branch of the Swimm3R training.

The pair branch of the training (underwater distillation and medium losses) is left as it is.
This branch runs next to it on the live student network and trains one extra head, the
Gaussian map head, so that many frames of a video give one Gaussian map without any
optimisation per scene.

One step:
  1. A window is read: an anchor frame, a random number of overlapping context frames and a
     few held out target frames (dataset/window_dataset.py).
  2. The student runs on the window without gradients. Every frame is the reference view of
     one pair pass with the anchor as partner. This gives the depth of the frame in its own
     camera, its confidence, the reference stream tokens, and the anchor pixels expressed in
     the camera of the frame.
  3. The anchor pixels are known in the anchor camera too, so a weighted Procrustes fit gives
     scale, rotation and translation of every frame in the anchor camera. No pose is given
     and no pose head is trained, the frames are tied together by our own pointmaps, the way
     our SfM ties them at inference.
  4. The head predicts one Gaussian per pixel of every context frame in the camera of that
     frame. The centre is the depth along the pixel ray, the size is relative to the pixel
     footprint at that depth.
  5. The Gaussians of all context frames are moved to the anchor camera and fused in voxels
     whose size follows the footprint, weighted by the pointmap confidence.
  6. The fused map is rendered in the target frames. The loss compares the render with the
     observed underwater frame and, through a second colour per Gaussian, with the restored
     frame.

At inference the same head is run once per frame and the SfM supplies pose and depth, so the
number of frames is not limited by the window used here.

Options, all off by default (docs/beta4d_keyhome_spec.md):
  --gs_kernel beta  the map is drawn with the Beta kernel of the gsplat fork (render_view_beta),
                    the Gaussian sigmas times K_BETA4. A 21 channel head (--gs_head_out 21)
                    predicts the Beta shape, else it is 4.
  --gs_4d           the gate, temporal width, shape and velocity of the 21 channel head split every
                    primitive into a static copy, fused as before, and a dynamic copy that keeps the
                    time of its frame (training/primitives4d.py). Every view is drawn at its own
                    time. The window carries the frame times (--gs_frame_times).
  --gs_key_pose     the targets are drawn at their key pose, a PnP of descriptor matches into the
                    anchor's own Gaussian centres (training/key_pose.py). A rejected target keeps
                    the Procrustes pose and a lower loss weight. --gs_key_ctx refits the context
                    frames as well, --gs_home_weight adds the homing hinge.
"""

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import mast3r.utils.path_to_dust3r  # noqa: F401
from dust3r.post_process import estimate_focal_knowing_depth

from dataset.window_dataset import UnderwaterWindowDataset
from model.gaussian_head import N_OUT, N_OUT_4D, OPACITY_BIAS, SCALE_BASE, VEL_UNIT, GaussianMapHead
from training.key_pose import (
    KeyPoseConfig, delta_from_poses, delta_pose, homing_loss, homing_perturbations, interp_pose, key_pose,
)
from training.primitives4d import (
    DYN_MIN_OPACITY, EPS2D_BETA4, K_BETA4, frame_gaps, primitives_at_time, rescale, split_static_dynamic,
    world_velocity,
)

SH_C0 = 0.28209479177387814
NEAR, FAR = 0.1, 1.0e4
NEAR_UNITS = 4.0


# ----------------------------------------------------------------------------- geometry


def weighted_umeyama(src, dst, w):
    """Similarity with dst ~ s R src + t. src, dst (M, N, 3), w (M, N) >= 0."""
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-20)
    mu_s = torch.einsum("mn,mnc->mc", w, src)
    mu_d = torch.einsum("mn,mnc->mc", w, dst)
    xs = src - mu_s[:, None]
    xd = dst - mu_d[:, None]
    cov = torch.bmm((xd * w[..., None]).transpose(1, 2), xs).double().cpu()
    var = (w * xs.square().sum(-1)).sum(1).double().cpu()
    u, sv, vh = torch.linalg.svd(cov)
    d = torch.sign(torch.linalg.det(u @ vh))
    d = torch.where(d == 0, torch.ones_like(d), d)
    dd = torch.ones_like(sv)
    dd[:, 2] = d
    rot = u @ torch.diag_embed(dd) @ vh
    s = (sv * dd).sum(-1) / var.clamp_min(1e-20)
    rot, s = rot.to(src), s.to(src)
    t = mu_d - s[:, None] * torch.einsum("mij,mj->mi", rot, mu_s)
    return s, rot, t


def align_to_anchor(src, dst, conf_src, conf_dst, iters=3):
    """Robust similarity of every row. Returns s, R, t and the median error relative to depth."""
    z = dst[..., 2]
    valid = torch.isfinite(src).all(-1) & torch.isfinite(dst).all(-1) & (z > 1e-3)
    # The error is measured relative to depth. Very small depths are floored at a tenth of
    # the median depth so that a few near pixels cannot take all the weight.
    zmed = torch.nanmedian(torch.where(valid, z, torch.full_like(z, float("nan"))), dim=1).values
    zc = torch.maximum(z, 0.1 * zmed.nan_to_num(1.0)[:, None]).clamp_min(1e-3)
    w0 = (conf_src.clamp_min(0) * conf_dst.clamp_min(0)).sqrt() / zc.square()
    w0 = torch.where(valid, w0, torch.zeros_like(w0))
    src = torch.where(valid[..., None], src, torch.zeros_like(src))
    dst = torch.where(valid[..., None], dst, torch.zeros_like(dst))
    w = w0
    for it in range(iters):
        s, rot, t = weighted_umeyama(src, dst, w)
        pred = s[:, None, None] * torch.einsum("mij,mnj->mni", rot, src) + t[:, None]
        r = (pred - dst).norm(dim=-1) / zc
        med = torch.nanmedian(torch.where(valid, r, torch.full_like(r, float("nan"))), dim=1).values
        if it < iters - 1:
            delta = (1.5 * med).nan_to_num(1.0).clamp_min(1e-3)
            w = w0 * (delta[:, None] / r.clamp_min(1e-9)).clamp(max=1.0)
    return s, rot, t, med


def rotmat_to_quat(m):
    """(..., 3, 3) to unit quaternion (..., 4) in the order w x y z."""
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    q_abs = torch.stack(
        [1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], dim=-1
    ).clamp_min(0).sqrt()
    cand = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]], dim=-1),
        torch.stack([m[..., 2, 1] - m[..., 1, 2], q_abs[..., 1] ** 2, m[..., 1, 0] + m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0]], dim=-1),
        torch.stack([m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] + m[..., 0, 1], q_abs[..., 2] ** 2, m[..., 1, 2] + m[..., 2, 1]], dim=-1),
        torch.stack([m[..., 1, 0] - m[..., 0, 1], m[..., 2, 0] + m[..., 0, 2], m[..., 2, 1] + m[..., 1, 2], q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)
    cand = cand / (2.0 * q_abs[..., None].clamp_min(0.1))
    idx = q_abs.argmax(dim=-1)
    q = torch.gather(cand, -2, idx[..., None, None].expand(*idx.shape, 1, 4)).squeeze(-2)
    return F.normalize(q, dim=-1)


def quat_to_rotmat(q):
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def quat_mul(a, b):
    """Hamilton product, R(a b) = R(a) R(b)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def pixel_rays(h, w, focal, device):
    """Rays with z = 1 through the pixel centres, principal point at the image centre."""
    v, u = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    return torch.stack([(u + 0.5 - w / 2.0) / focal, (v + 0.5 - h / 2.0) / focal, torch.ones_like(u)], dim=-1)


# ----------------------------------------------------------------------------- fusion


def voxel_fuse(xyz, weight, footprint, mean_attrs, pick_attrs, voxel_px=1.0):
    """Fuses Gaussians that fall into the same voxel.

    The voxel edge is voxel_px footprints, with the footprint rounded to a power of two of
    the median footprint, so near surfaces keep small voxels and far surfaces get large ones.
    Position and the attributes in mean_attrs are averaged with the weights. The attributes in
    pick_attrs (size and rotation) are taken from the member with the largest weight.
    """
    n = xyz.shape[0]
    fp_ref = footprint.median().clamp_min(1e-9)
    level = torch.round(torch.log2((footprint / fp_ref).clamp_min(1e-6))).clamp(-4, 4)
    cell = voxel_px * fp_ref * torch.pow(2.0, level)
    c = (torch.floor(xyz / cell[:, None]).long() + (1 << 18)).clamp(0, (1 << 19) - 1)
    key = ((level.long() + 4) << 57) | (c[:, 0] << 38) | (c[:, 1] << 19) | c[:, 2]
    uniq, inv = torch.unique(key, return_inverse=True)
    v = uniq.numel()
    w = weight.clamp_min(1e-6)
    wsum = torch.zeros(v, device=xyz.device, dtype=w.dtype).index_add_(0, inv, w)

    def mean(a):
        shape = (-1,) + (1,) * (a.dim() - 1)
        out = torch.zeros((v,) + a.shape[1:], device=a.device, dtype=a.dtype).index_add_(0, inv, a * w.view(shape))
        return out / wsum.view(shape)

    wmax = torch.full((v,), -1.0, device=xyz.device, dtype=w.dtype).scatter_reduce(0, inv, w, reduce="amax")
    is_max = w >= wmax[inv]
    idx = torch.arange(n, device=xyz.device)
    sel = torch.zeros(v, dtype=torch.long, device=xyz.device).scatter_(0, inv[is_max], idx[is_max])
    return mean(xyz), {k: mean(a) for k, a in mean_attrs.items()}, {k: a[sel] for k, a in pick_attrs.items()}


# ----------------------------------------------------------------------------- rendering


def _projection_t(near, far, tanx, tany, device, off_x=0.0, off_y=0.0):
    p = torch.zeros(4, 4, device=device)
    p[0, 0] = 1.0 / tanx
    p[1, 1] = 1.0 / tany
    p[0, 2] = off_x
    p[1, 2] = off_y
    p[3, 2] = 1.0
    p[2, 2] = far / (far - near)
    p[2, 3] = -(far * near) / (far - near)
    return p.t().contiguous()


def render_view(means, opacity, scales, quats, colors, c2w, focal, h, w, bg, pp=None):
    """One view of the map. colors (G, 3) in [0, 1]. Returns image (3, h, w), depth and alpha.

    pp is the principal point in continuous pixel coordinates (the centre of pixel j is at
    j + 0.5). Without it the principal point is the image centre.
    """
    from diff_gauss_pose import GaussianRasterizationSettings, GaussianRasterizer

    tanx, tany = 0.5 * w / focal, 0.5 * h / focal
    off_x = 0.0 if pp is None else (2.0 * float(pp[0]) - w) / w
    off_y = 0.0 if pp is None else (2.0 * float(pp[1]) - h) / h
    settings = GaussianRasterizationSettings(
        image_height=h, image_width=w, tanfovx=tanx, tanfovy=tany, bg=bg, scale_modifier=1.0,
        projmatrix=_projection_t(NEAR, FAR, tanx, tany, means.device, off_x, off_y), sh_degree=0,
        prefiltered=False, debug=False, enable_cov_grad=True, enable_sh_grad=True,
    )
    means2d = torch.zeros_like(means, requires_grad=True)
    shs = ((colors - 0.5) / SH_C0)[:, None, :].contiguous()
    view = torch.linalg.inv(c2w).t().contiguous()
    image, depth, _, alpha, _, _ = GaussianRasterizer(settings)(
        means3D=means, means2D=means2d, shs=shs, colors_precomp=None, opacities=opacity[:, None],
        scales=scales, rotations=quats, viewmatrix=view,
    )
    return image, depth, alpha


def render_view_beta(means, opacity, scales, quats, colors, c2w, focal, h, w, bg, pp=None, betas=4.0,
                     scale_factor=1.0, eps2d=0.3):
    """render_view with the Beta kernel of the gsplat fork in ubs/submodules (Universal Beta Splatting).

    Same inputs and outputs as render_view. The covariance comes from scales * scale_factor and
    the quaternions (w x y z), eps2d px^2 is added to its projection, and a primitive covers the
    pixels with Mahalanobis distance d2 < 1 under it, with alpha = opacity (1 - d2)^b. betas is one
    b or one per primitive. Colours are clamped at 0 like the SH colours of render_view, the
    background is added as (1 - alpha) bg, depth is the alpha blended depth along the axis.
    """
    from gsplat.rendering import rasterization

    n = means.shape[0]
    cx, cy = (0.5 * w, 0.5 * h) if pp is None else (float(pp[0]), float(pp[1]))
    k = torch.tensor([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], device=means.device, dtype=means.dtype)
    if not torch.is_tensor(betas):
        betas = torch.full((n,), float(betas), device=means.device, dtype=means.dtype)
    # near_plane 0.2 is the frustum test of the 3DGS rasteriser
    out, alpha, _ = rasterization(
        means=means, l_triagnles=quats, scales=scales * scale_factor, opacities=opacity.reshape(n),
        betas=betas.reshape(n), colors=colors.clamp_min(0.0), viewmats=torch.linalg.inv(c2w)[None], Ks=k[None],
        width=w, height=h, near_plane=0.2, eps2d=eps2d, render_mode="RGB+D")
    out, alpha = out[0].permute(2, 0, 1), alpha[0].permute(2, 0, 1)
    return out[:3] + (1.0 - alpha) * bg[:, None, None], out[3:], alpha


def psnr(pred, target):
    mse = (pred.clamp(0, 1) - target).square().flatten(1).mean(1).clamp_min(1e-10)
    return (-10.0 * torch.log10(mse)).mean()


# ----------------------------------------------------------------------------- branch


def check_flags(args):
    """Raises ValueError when the Beta, 4D and key pose flags of the branch do not fit together."""
    kernel = getattr(args, "gs_kernel", "gauss")
    head_out = int(getattr(args, "gs_head_out", N_OUT))
    key = bool(getattr(args, "gs_key_pose", False))
    bad = []
    if kernel not in ("gauss", "beta"):
        bad.append(f"unknown --gs_kernel {kernel}")
    if head_out not in (N_OUT, N_OUT_4D):
        bad.append(f"--gs_head_out must be {N_OUT} or {N_OUT_4D}")
    if head_out == N_OUT_4D and kernel != "beta":
        bad.append("--gs_head_out 21 needs --gs_kernel beta")
    if getattr(args, "gs_4d", False) and not (
            head_out == N_OUT_4D and kernel == "beta" and getattr(args, "gs_frame_times", False)):
        bad.append("--gs_4d needs --gs_head_out 21, --gs_kernel beta and --gs_frame_times")
    if getattr(args, "gs_key_ctx", False) and not key:
        bad.append("--gs_key_ctx needs --gs_key_pose")
    if float(getattr(args, "gs_home_weight", 0.0)) > 0 and not key:
        bad.append("--gs_home_weight above 0 needs --gs_key_pose")
    if bad:
        raise ValueError("; ".join(bad))


class GaussianBranch:
    def __init__(self, args, net, device, total_steps, save_dir):
        check_flags(args)
        self.args, self.net, self.device = args, net, device
        self.total_steps = max(1, int(total_steps))
        self.save_dir = save_dir
        self.debug_dir = os.path.join(save_dir, "gs_debug")
        os.makedirs(self.debug_dir, exist_ok=True)
        self.beta = getattr(args, "gs_kernel", "gauss") == "beta"
        self.four_d = bool(getattr(args, "gs_4d", False))
        self.key = bool(getattr(args, "gs_key_pose", False))
        self.key_ctx = bool(getattr(args, "gs_key_ctx", False))
        self.home_weight = float(getattr(args, "gs_home_weight", 0.0))
        head_out = int(getattr(args, "gs_head_out", N_OUT))

        self.head = (GaussianMapHead(net) if head_out == N_OUT else GaussianMapHead(net, n_out=head_out)).to(device)
        copied, total = self.head.init_from_pointmap_head(net)
        self.head.train()
        n_par = sum(p.numel() for p in self.head.parameters())
        print(f"[gs] Gaussian map head: {n_par / 1e6:.2f}M parameters, {copied}/{total} tensors start from the pointmap head")
        self.opt = torch.optim.AdamW(self.head.parameters(), lr=args.gs_lr, betas=(0.9, 0.95),
                                     weight_decay=args.gs_weight_decay)

        from lpips import LPIPS
        self.lpips = LPIPS(net="vgg", verbose=False).to(device).eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)
        self.bg = torch.zeros(3, device=device)

        seed = int(getattr(args, "seed", 0))
        seed = 0 if seed < 0 else seed
        jerlov_kwargs = dict(
            n_hue_families=int(getattr(args, "phys_jerlov_n_hue_families", 3)),
            beta_D_R_max=float(getattr(args, "phys_jerlov_beta_D_R_max", 2.0)),
            betaB_ratio_min=float(getattr(args, "phys_betaB_ratio_min", 0.4)),
            betaB_ratio_max=float(getattr(args, "phys_betaB_ratio_max", 1.0)),
            cyan_weight=float(getattr(args, "phys_jerlov_cyan_weight", 1.0)),
            family_ids=getattr(args, "phys_jerlov_family_ids", None),
            cyan_dead_beta_D_R=tuple(getattr(args, "phys_jerlov_cyan_band_beta_D_R", [0.6, 1.2])),
            family_cited_types=getattr(args, "phys_jerlov_family_cited_types", None),
        )
        timed = dict(frame_times=True, fps=float(args.gs_flsea_fps)) if getattr(args, "gs_frame_times", False) else {}
        dataset = UnderwaterWindowDataset(
            root_path=args.real_dataset_root,
            dataset_config_yml=args.real_dataset_config,
            precomputed_pairs=args.precomputed_pairs,
            long_edge=args.gs_long_edge,
            ctx_min=args.gs_ctx_min,
            ctx_max=args.gs_ctx_max,
            n_targets=args.gs_targets,
            radii=args.gs_radii,
            jerlov_p=args.gs_jerlov_p,
            jerlov_kwargs=jerlov_kwargs,
            jerlov_noise=float(getattr(args, "phys_jerlov_synth_noise", 0.0)),
            seed=seed + 7919,
            **timed,
        )
        gen = torch.Generator()
        gen.manual_seed(seed + 7919)
        workers = int(args.gs_workers)
        self.loader = DataLoader(
            dataset, batch_size=None, num_workers=workers, generator=gen,
            persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None,
        )
        self.it = iter(self.loader)
        self.rng = np.random.default_rng(seed + 104729)
        self.n = 0
        self.fail = 0
        self.consec_fail = 0
        self.skipped = 0
        self.disabled = False
        self.t_ema = None
        self.key_cfg = self.home_gen = None
        if self.key:
            self.key_cfg = KeyPoseConfig(
                stride=int(args.gs_key_stride), max_matches=int(args.gs_key_max_matches), iters=int(args.gs_key_iters),
                huber_px=float(args.gs_key_huber_px), inlier_px=float(args.gs_key_inlier_px),
                min_matches=int(args.gs_key_min_matches), min_inlier=float(args.gs_key_min_inlier),
                max_move_px=float(args.gs_key_max_move_px), trim_px=float(getattr(args, "gs_key_trim_px", 0.0)))
            self.home_gen = torch.Generator()
            self.home_gen.manual_seed(seed + 15485863)
        if self.beta or self.key:
            print(f"[gs] kernel {'beta' if self.beta else 'gauss'}, head {head_out} channels, "
                  f"{'4D' if self.four_d else 'static'} map, frame times {bool(timed)}, "
                  f"key pose {self.key_cfg if self.key else 'off'}{', context key poses' if self.key_ctx else ''}"
                  f"{f', homing hinge {self.home_weight}' if self.home_weight > 0 else ''}")

    # ------------------------------------------------------------------ pieces

    def lr(self):
        a = self.args
        warm = min(1.0, (self.n + 1) / max(1, a.gs_warmup_steps))
        prog = min(1.0, self.n / self.total_steps)
        return a.gs_lr * warm * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))

    @torch.no_grad()
    def geometry(self, img01, want_desc=False):
        """Runs the student on a window (F, 3, H, W) in [0, 1], frame 0 is the anchor.

        With want_desc the result also has the descriptors of the pair passes, desc_self (F, H, W, D)
        of every frame and desc_anchor (F, H, W, D) of the anchor in the pass with that frame.
        """
        net = self.net
        f, _, h, w = img01.shape
        dev = img01.device
        shape = torch.tensor([[h, w]], device=dev).repeat(f, 1)
        feat, pos, _ = net._encode_image(img01 * 2.0 - 1.0, shape)
        partner = torch.zeros(f, dtype=torch.long, device=dev)
        partner[0] = 1
        dec1, dec2 = net._decoder(feat, pos, feat[partner], pos[partner])
        dec1 = [t.float() for t in dec1]
        dec2 = [t.float() for t in dec2]
        with torch.autocast("cuda", enabled=False):
            res1 = net._downstream_head(1, dec1, shape)
            res2 = net._downstream_head(2, dec2, shape)
        x1, c1 = res1["pts3d"].float(), res1["conf"].float()
        x2, c2 = res2["pts3d"].float(), res2["conf"].float()

        pp = torch.tensor([w / 2.0, h / 2.0], device=dev).repeat(f, 1)
        focal = float(estimate_focal_knowing_depth(x1, pp, focal_mode="weiszfeld").median())
        if not (math.isfinite(focal) and focal > 1.0):
            raise ValueError(f"bad focal {focal}")

        src = x2[1:].reshape(f - 1, -1, 3)
        dst = x1[0].reshape(1, -1, 3).expand(f - 1, -1, -1)
        s, rot, t, resid = align_to_anchor(
            src, dst, c2[1:].reshape(f - 1, -1), c1[0].reshape(1, -1).expand(f - 1, -1))
        s = torch.cat([torch.ones(1, device=dev), s])
        rot = torch.cat([torch.eye(3, device=dev)[None], rot])
        t = torch.cat([torch.zeros(1, 3, device=dev), t])
        resid = torch.cat([torch.zeros(1, device=dev), resid])
        out = dict(tokens=dec1, z=x1[..., 2] * s[:, None, None], conf=c1, rot=rot, trans=t,
                   scale=s, resid=resid, focal=focal)
        if want_desc:
            out.update(desc_self=res1["desc"].float(), desc_anchor=res2["desc"].float())
        return out

    def assemble(self, geo, ci, act, h, w):
        """Gaussians of the context frames ci in the anchor camera, fused and rescaled."""
        focal = geo["focal"]
        dev = geo["z"].device
        rays = pixel_rays(h, w, focal, dev)
        z = geo["z"][ci]
        rot, trans = geo["rot"][ci], geo["trans"][ci]
        xyz = torch.einsum("kij,khwj->khwi", rot, rays[None] * z[..., None]) + trans[:, None, None]
        fp = z / focal
        scales = fp[..., None] * SCALE_BASE * torch.exp(act["log_scale"])
        quat = quat_mul(rotmat_to_quat(rot)[:, None, None].expand_as(act["quat"]), act["quat"])
        valid = torch.isfinite(xyz).all(-1) & torch.isfinite(z) & (z > 1e-4)
        xyz, fp, conf = xyz[valid], fp[valid], geo["conf"][ci][valid]
        opacity, scales, quat = act["opacity"][valid], scales[valid], quat[valid]
        color_raw, color_clean = act["color_raw"][valid], act["color_clean"][valid]
        n_in = xyz.shape[0]
        if n_in < 1000:
            raise ValueError(f"only {n_in} valid Gaussians")
        if self.args.gs_voxel_px > 0:
            xyz, m, p = voxel_fuse(
                xyz, conf, fp, dict(opacity=opacity, color_raw=color_raw, color_clean=color_clean),
                dict(scales=scales, quat=quat), self.args.gs_voxel_px)
            opacity, color_raw, color_clean = m["opacity"], m["color_raw"], m["color_clean"]
            scales, quat = p["scales"], p["quat"]
        zs = z[valid]
        zq = torch.quantile(zs[:: max(1, zs.numel() // 200000)], 0.01).clamp_min(1e-4)
        unit = NEAR_UNITS / zq
        return dict(means=xyz * unit, scales=scales * unit, quat=quat, opacity=opacity,
                    color_raw=color_raw, color_clean=color_clean, unit=unit, n_in=n_in, n_out=xyz.shape[0])

    def assemble_beta(self, geo, ci, act, h, w, times=None):
        """assemble for the Beta kernel. The primitives of the context frames are split by
        split_static_dynamic: the static copies are fused as in assemble, with --gs_4d the dynamic
        copies keep the time of their frame (times, (F,) seconds). Returns the split in the units of
        the render under 'split', the unit, the counts and the statistics of the head output."""
        a = self.args
        focal = geo["focal"]
        dev = geo["z"].device
        rays = pixel_rays(h, w, focal, dev)
        z = geo["z"][ci]
        rot, trans = geo["rot"][ci], geo["trans"][ci]
        xyz = torch.einsum("kij,khwj->khwi", rot, rays[None] * z[..., None]) + trans[:, None, None]
        fp = z / focal
        scales = fp[..., None] * SCALE_BASE * torch.exp(act["log_scale"])
        quat = quat_mul(rotmat_to_quat(rot)[:, None, None].expand_as(act["quat"]), act["quat"])
        valid = torch.isfinite(xyz).all(-1) & torch.isfinite(z) & (z > 1e-4)
        prim = dict(xyz=xyz[valid], fp=fp[valid], conf=geo["conf"][ci][valid], opacity=act["opacity"][valid],
                    scales=scales[valid], quat=quat[valid], color_raw=act["color_raw"][valid],
                    color_clean=act["color_clean"][valid])
        if "beta" in act:
            prim["beta"] = act["beta"][valid]
        four_d = self.four_d and "gate" in act
        if four_d:
            t = times[ci]
            prim.update(gate=act["gate"][valid], t_src=t[:, None, None].expand_as(z)[valid],
                        tw=(frame_gaps(t)[:, None, None] * act["tw_rel"])[valid], tb=act["tb"][valid],
                        vel=world_velocity(act["vel"], fp, rot[:, None, None])[valid])
        n_in = prim["xyz"].shape[0]
        if n_in < 1000:
            raise ValueError(f"only {n_in} valid Gaussians")
        split = split_static_dynamic(prim, voxel_fuse, a.gs_voxel_px, four_d=four_d,
                                     dyn_max=int(getattr(a, "gs_dyn_max", 0)))
        zs = z[valid]
        zq = torch.quantile(zs[:: max(1, zs.numel() // 200000)], 0.01).clamp_min(1e-4)
        unit = NEAR_UNITS / zq
        with torch.no_grad():
            b = prim.get("beta")
            st = dict(beta_mean=4.0, beta_min=4.0, beta_max=4.0) if b is None else dict(
                beta_mean=float(b.mean()), beta_min=float(b.min()), beta_max=float(b.max()))
            if four_d:
                gate = prim["gate"]
                dyn = (prim["opacity"] * gate.clamp_min(0.0)) > DYN_MIN_OPACITY
                vel = act["vel"][valid].norm(dim=-1)
                st.update(gate_mean=split["gate_mean"],
                          gate_p95=float(torch.quantile(gate[:: max(1, n_in // 200000)].float(), 0.95)),
                          dyn_rows=split["n_dyn"], dyn_frac=split["n_dyn"] / n_in,
                          tw_rel_mean=float(act["tw_rel"][valid].mean()), tb_mean=float(prim["tb"].mean()),
                          vel_px_s=float(vel[dyn].mean()) if bool(dyn.any()) else 0.0)
        return dict(split=rescale(split, unit), opacity=split["static"]["opacity"], unit=unit, n_in=n_in,
                    n_out=split["n_static"], n_dyn=split["n_dyn"], stats=st)

    def proc_pose(self, geo, i):
        """Camera to world (4, 4) of frame i in the units of the window, as the Procrustes fit (or a key pose in geo) gives it."""
        c2w = torch.eye(4, device=geo["rot"].device)
        c2w[:3, :3] = geo["rot"][i]
        c2w[:3, 3] = geo["trans"][i]
        return c2w

    def render_pose(self, g, c2w, focal, h, w, which=("color_raw",), t=None):
        """Images of the map at one camera to world pose in the units of the window, a dict over which.

        A map of assemble_beta is drawn with the Beta kernel at the time t (None for the static set).
        """
        c = c2w.clone()
        c[:3, 3] = c2w[:3, 3] * g["unit"]
        if "split" in g:
            p = primitives_at_time(g["split"], t, min_opacity=DYN_MIN_OPACITY)
            return {k: render_view_beta(p["xyz"], p["opacity"], p["scales"], p["quat"], p[k], c, focal, h, w, self.bg,
                                        betas=p["beta"], scale_factor=K_BETA4, eps2d=EPS2D_BETA4)[0] for k in which}
        return {k: render_view(g["means"], g["opacity"], g["scales"], g["quat"], g[k], c, focal, h, w, self.bg)[0]
                for k in which}

    def render(self, g, geo, views, h, w, which=("color_raw", "color_clean"), poses=None, times=None):
        """Views of the map at their poses in geo. poses {view: camera to world} replaces some of them,
        times (F,) gives the time of every view for a 4D map."""
        out = {k: [] for k in which}
        for i in views:
            if "split" in g or poses is not None:
                c2w = poses[i] if poses is not None and i in poses else self.proc_pose(geo, i)
                imgs = self.render_pose(g, c2w, geo["focal"], h, w, which, None if times is None else times[i])
                for k in which:
                    out[k].append(imgs[k])
                continue
            c2w = torch.eye(4, device=self.device)
            c2w[:3, :3] = geo["rot"][i]
            c2w[:3, 3] = geo["trans"][i] * g["unit"]
            for k in which:
                img, _, _ = render_view(g["means"], g["opacity"], g["scales"], g["quat"], g[k],
                                        c2w, geo["focal"], h, w, self.bg)
                out[k].append(img)
        return {k: torch.stack(v) for k, v in out.items()}

    @staticmethod
    def splat_act(rgb):
        """Head output of the plain pixel splat for frames rgb (K, H, W, 3)."""
        ident = rgb.new_tensor([1.0, 0.0, 0.0, 0.0]).expand(*rgb.shape[:3], 4)
        return dict(opacity=torch.full(rgb.shape[:3], 1.0 / (1.0 + math.exp(-OPACITY_BIAS)), device=rgb.device),
                    log_scale=torch.zeros_like(rgb), quat=ident, color_raw=rgb, color_clean=rgb)

    @torch.no_grad()
    def splat_baseline(self, geo, ci, img01, tgt, h, w, poses=None):
        """PSNR of the plain pixel splat, the output of the head at its start. A geometry check."""
        act = self.splat_act(img01[ci].permute(0, 2, 3, 1))
        g = self.assemble_beta(geo, ci, act, h, w) if self.beta else self.assemble(geo, ci, act, h, w)
        pred = self.render(g, geo, tgt, h, w, which=("color_raw",), poses=poses)["color_raw"]
        return float(psnr(pred, img01[torch.tensor(tgt, device=img01.device)]))

    def window_times(self, s, f):
        """Times of the frames of a window in seconds from the anchor, float32 (F,) on the device, and
        whether the window is timed. An untimed window (a photo set) has zeros."""
        if "times" in s and bool(s.get("timed", False)):
            t = s["times"].double()
            return (t - t[0]).float().to(self.device), True
        return torch.zeros(f, device=self.device), False

    @torch.no_grad()
    def key_poses(self, geo, frames, h, w):
        """Key pose of every frame in frames against the anchor's own Gaussian centres (training/key_pose.py).

        Returns {i: (c2w, accepted, stats)} with c2w (4, 4) in the units of the window, the Procrustes
        pose when rejected. stats also holds zmed, the median depth of the frame.
        """
        focal, dev = geo["focal"], geo["z"].device
        z0 = geo["z"][0]
        xyz_a = pixel_rays(h, w, focal, dev) * z0[..., None]
        valid_a = torch.isfinite(xyz_a).all(-1) & torch.isfinite(z0) & (z0 > 1e-4)
        pp = (0.5 * w, 0.5 * h)
        out = {}
        for i in frames:
            c2w, acc, st = key_pose(geo["desc_self"][i], geo["desc_anchor"][i], xyz_a, valid_a, self.proc_pose(geo, i),
                                    focal, pp, self.key_cfg)
            z = geo["z"][i]
            z = z[torch.isfinite(z) & (z > 1e-4)]
            st["zmed"] = float(z.median()) if z.numel() else float("nan")
            out[i] = (c2w, bool(acc), st)
        return out

    @staticmethod
    def with_keys(geo, keys):
        """geo with the rotation and translation of the accepted frames of keys replaced by their key pose."""
        rot, trans = geo["rot"].clone(), geo["trans"].clone()
        for i, (c2w, acc, _) in keys.items():
            if acc:
                rot[i], trans[i] = c2w[:3, :3], c2w[:3, 3]
        return dict(geo, rot=rot, trans=trans)

    @staticmethod
    def key_log(keys, frames, seconds):
        """Means over frames of the key pose statistics, the log keys key_*."""
        st = [keys[i][2] for i in frames]

        def mean(fn):
            v = [x for x in (float(fn(s)) for s in st) if math.isfinite(x)]
            return float(np.mean(v)) if v else float("nan")

        return {"key_n_match": mean(lambda s: s["n_match"]), "key_inlier0": mean(lambda s: s["inlier0"]),
                "key_inlier": mean(lambda s: s["inlier"]), "key_resid0_px": mean(lambda s: s["resid0_px"]),
                "key_resid_px": mean(lambda s: s["resid_px"]), "key_move_px": mean(lambda s: s["move_px"]),
                "key_rot_deg": mean(lambda s: s["rot_deg"]), "key_trans_rel": mean(lambda s: s["trans"] / s["zmed"]),
                "key_accept": float(np.mean([keys[i][1] for i in frames])), "key_seconds": seconds}

    def homing(self, g, geo, raw, tgt, keys, ctx, times, htimes, h, w):
        """Homing hinge of training/key_pose.py over the accepted targets. Returns the mean loss (None
        without an accepted target) and the share of violated perturbations."""
        a = self.args
        focal, pp = geo["focal"], (0.5 * w, 0.5 * h)
        ctx_c2w = torch.stack([self.proc_pose(geo, i) for i in ctx])
        ctx_t = htimes[torch.tensor(ctx, device=htimes.device)].cpu()
        losses, viol = [], []
        for i in tgt:
            c2w_key, acc, st = keys[i]
            if not acc or not math.isfinite(st["zmed"]):
                continue
            c2w_int = interp_pose(ctx_c2w, ctx_t, float(htimes[i]))
            xis = homing_perturbations(c2w_key, c2w_int, st["zmed"], focal, h, w, float(a.gs_home_px_max), self.home_gen)
            t_i = None if times is None else times[i]

            def render_fn(c2w, t_i=t_i):
                return self.render_pose(g, c2w, focal, h, w, ("color_raw",), t_i)["color_raw"]

            loss, v = homing_loss(render_fn, c2w_key, xis, raw[i], geo["z"][i], focal, pp, kappa=float(a.gs_home_kappa))
            losses.append(loss)
            viol.append(v)
        if not losses:
            return None, float("nan")
        return torch.stack(losses).mean(), float(np.mean(viol))

    # ------------------------------------------------------------------ step

    def _step(self, step):
        a, dev = self.args, self.device
        s = next(self.it)
        raw = s["raw"].to(dev, non_blocking=True).float() / 255.0
        clean = s["clean"].to(dev, non_blocking=True).float() / 255.0
        k, f = int(s["n_ctx"]), raw.shape[0]
        h, w = raw.shape[-2:]

        geo = self.geometry(raw, want_desc=True) if self.key else self.geometry(raw)
        ok = geo["resid"] < a.gs_max_resid
        ok[0] = True
        ctx = [i for i in range(k) if bool(ok[i])]
        tgt = [i for i in range(k, f) if bool(ok[i])]
        if len(ctx) < 2 or len(tgt) < 1:
            self.skipped += 1
            return None
        ci = torch.tensor(ctx, device=dev)
        times, timed = self.window_times(s, f) if self.four_d else (None, False)
        keys, poses, gm = None, None, geo
        if self.key:
            tk = time.time()
            keys = self.key_poses(geo, tgt + (ctx[1:] if self.key_ctx else []), h, w)
            key_seconds = time.time() - tk
            poses = {i: keys[i][0] for i in tgt}
            if self.key_ctx:
                gm = self.with_keys(geo, {i: keys[i] for i in ctx[1:]})

        out = self.head([t[ci] for t in geo["tokens"]], raw[ci] * 2.0 - 1.0, (h, w))
        act = GaussianMapHead.activate(out, raw[ci])
        g = self.assemble_beta(gm, ci, act, h, w, times) if self.beta else self.assemble(gm, ci, act, h, w)

        views = list(tgt)
        for _ in range(int(a.gs_ctx_render)):
            views.append(ctx[int(self.rng.integers(len(ctx)))])
        pred = self.render(g, gm, views, h, w, poses=poses, times=times)
        vi = torch.tensor(views, device=dev)
        t_raw, t_clean = raw[vi], clean[vi]
        p_raw, p_clean = pred["color_raw"], pred["color_clean"]
        nt = len(tgt)
        if keys is None:
            mse_raw = (p_raw - t_raw).square().mean()
            mse_clean = (p_clean - t_clean).square().mean()
            lp_raw = self.lpips(p_raw.clamp(0, 1), t_raw, normalize=True).mean()
            lp_clean = self.lpips(p_clean.clamp(0, 1), t_clean, normalize=True).mean()
        else:
            # weight 1 for accepted targets and context renders, the fallback weight for rejected targets
            wv = torch.tensor([1.0 if j >= nt or keys[i][1] else float(a.gs_key_fallback_weight)
                               for j, i in enumerate(views)], device=dev)
            wv = wv / wv.sum()
            mse_raw = ((p_raw - t_raw).square().flatten(1).mean(1) * wv).sum()
            mse_clean = ((p_clean - t_clean).square().flatten(1).mean(1) * wv).sum()
            lp_raw = (self.lpips(p_raw.clamp(0, 1), t_raw, normalize=True).flatten() * wv).sum()
            lp_clean = (self.lpips(p_clean.clamp(0, 1), t_clean, normalize=True).flatten() * wv).sum()
        loss = mse_raw + mse_clean + a.gs_lambda_lpips * (lp_raw + lp_clean)
        if self.four_d:
            p_plus = act["gate"].clamp_min(0.0)
            loss_gate = p_plus.mean()
            loss_vel = (p_plus.detach() * (act["vel"] / VEL_UNIT).square().sum(-1)).mean()
            loss = loss + a.gs_lambda_gate * loss_gate + a.gs_lambda_vel * loss_vel
        loss_home = viol = None
        if self.home_weight > 0 and self.n >= int(a.gs_home_warmup):
            htimes = times if timed else (s["frames"].double() - s["frames"][0]).float().to(dev)
            loss_home, viol = self.homing(g, gm, raw, tgt, keys, ctx, times, htimes, h, w)
            if loss_home is not None:
                loss = loss + self.home_weight * loss_home
        if not torch.isfinite(loss):
            raise FloatingPointError("non finite loss")

        lr = self.lr()
        for group in self.opt.param_groups:
            group["lr"] = lr
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.head.parameters(), a.gs_grad_clip)
        if not torch.isfinite(gnorm):
            self.opt.zero_grad(set_to_none=True)
            raise FloatingPointError("non finite gradient")
        self.opt.step()

        log = {
            "loss": float(loss), "mse_raw": float(mse_raw), "mse_clean": float(mse_clean),
            "lpips_raw": float(lp_raw), "lpips_clean": float(lp_clean),
            "psnr_raw": float(psnr(p_raw[:nt].detach(), t_raw[:nt])),
            "psnr_clean": float(psnr(p_clean[:nt].detach(), t_clean[:nt])),
            "n_ctx": len(ctx), "n_tgt": nt, "frames": f, "gauss_in": g["n_in"], "gauss_out": g["n_out"],
            "fuse_ratio": g["n_out"] / max(1, g["n_in"]), "focal": geo["focal"],
            "resid_med": float(geo["resid"][1:].nanmedian()), "resid_max": float(geo["resid"][1:].nan_to_num(9.0).max()),
            "scale_min": float(geo["scale"].min()), "scale_max": float(geo["scale"].max()),
            "opacity_mean": float(g["opacity"].detach().mean()), "grad_norm": float(gnorm), "lr": lr,
            "synth": float(bool(s["synth"])),
        }
        if self.beta:
            st = g["stats"]
            log.update({k: st[k] for k in ("beta_mean", "beta_min", "beta_max")})
            if self.four_d:
                four = ("gate_mean", "gate_p95", "dyn_rows", "dyn_frac", "tw_rel_mean", "tb_mean", "vel_px_s")
                log.update({k: st.get(k, 0.0) for k in four})
                log.update(loss_gate=float(loss_gate), loss_vel=float(loss_vel), timed=float(timed))
        if keys is not None:
            log.update(self.key_log(keys, tgt, key_seconds))
        if self.home_weight > 0:
            log.update(loss_home=0.0 if loss_home is None else float(loss_home), home_viol=float("nan") if viol is None else viol)
        if self.n < 3 or self.n % max(1, a.gs_debug_every) == 0:
            log["psnr_splat"] = self.splat_baseline(gm, ci, raw, tgt, h, w, poses=poses)
            if keys is not None:
                with torch.no_grad():
                    proc = self.render(g, gm, tgt, h, w, which=("color_raw",), times=times)["color_raw"]
                log["psnr_proc"] = float(psnr(proc, t_raw[:nt]))
            self.dump(step, t_raw[0], p_raw[0].detach(), t_clean[0], p_clean[0].detach(), s)
        self.n += 1
        return log

    def dump(self, step, t_raw, p_raw, t_clean, p_clean, s):
        from PIL import Image
        row = torch.cat([t_raw, p_raw.clamp(0, 1), t_clean, p_clean.clamp(0, 1)], dim=2)
        arr = (row.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        name = f"step_{int(step):06d}_{s['scene']}_{'synth' if bool(s['synth']) else 'real'}.jpg"
        Image.fromarray(arr).save(os.path.join(self.debug_dir, name), quality=90)

    def train_step(self, step, writer=None):
        if self.disabled or step < int(getattr(self.args, "gs_start", 0)):
            return
        t0 = time.time()
        try:
            log = self._step(step)
            self.consec_fail = 0
        except Exception as e:
            if getattr(self.args, "gs_strict", False):
                raise
            self.fail += 1
            self.consec_fail += 1
            self.opt.zero_grad(set_to_none=True)
            if self.fail <= 5 or self.fail % 100 == 0:
                print(f"[gs] step {step} failed ({self.fail} so far): {type(e).__name__}: {str(e)[:300]}")
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            if self.consec_fail >= 200:
                self.disabled = True
                print("[gs] branch disabled after 200 failures in a row, the pair branch continues")
            return
        if log is None:
            return
        dt = time.time() - t0
        self.t_ema = dt if self.t_ema is None else 0.95 * self.t_ema + 0.05 * dt
        if writer is not None:
            for key, val in log.items():
                writer.add_scalar(f"GS/{key}", val, step)
            writer.add_scalar("GS/seconds", dt, step)
        if self.n <= 3 or self.n % max(1, self.args.gs_print_every) == 0:
            extra = f" splat {log['psnr_splat']:.2f}" if "psnr_splat" in log else ""
            more = ""
            if "key_accept" in log:
                more += f" | key acc {log['key_accept']:.2f} inl {log['key_inlier']:.2f} move {log['key_move_px']:.1f} px"
                if "psnr_proc" in log:
                    more += f" proc {log['psnr_proc']:.2f}"
            if "gate_mean" in log:
                more += f" | gate {log['gate_mean']:.4f} dyn {log['dyn_frac']:.4f} beta {log['beta_mean']:.2f}"
            print(
                f"[gs] step {step} n {self.n} loss {log['loss']:.4f} psnr raw {log['psnr_raw']:.2f} "
                f"clean {log['psnr_clean']:.2f}{extra} | ctx {log['n_ctx']} tgt {log['n_tgt']} "
                f"gauss {log['gauss_in'] / 1e6:.2f}M to {log['gauss_out'] / 1e6:.2f}M | focal {log['focal']:.0f} "
                f"resid {log['resid_med']:.4f}/{log['resid_max']:.4f} | {self.t_ema:.2f}s "
                f"skipped {self.skipped} failed {self.fail}{more}", flush=True)

    # ------------------------------------------------------------------ io

    def save(self, weights_dir, epoch):
        path = os.path.join(weights_dir, f"{str(epoch).zfill(4)}_gauss.pth")
        torch.save({"head": self.head.state_dict(), "opt": self.opt.state_dict(), "n": self.n,
                    "skipped": self.skipped, "fail": self.fail}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.head.load_state_dict(ck["head"])
        if "opt" in ck:
            self.opt.load_state_dict(ck["opt"])
        self.n = int(ck.get("n", 0))
        print(f"[gs] resumed the Gaussian map head from {path} (n {self.n})")

    def warm_start(self, path):
        """Head weights of 14 or 21 channels as the start, rows the file lacks at zero. The optimiser
        and the step count are not loaded, self.n stays as it is."""
        ck = torch.load(path, map_location=self.device)
        expanded = self.head.load_base_state(ck["head"] if "head" in ck else ck)
        more = f", {len(expanded)} tensors expanded with zero rows: {expanded}" if expanded else ""
        print(f"[gs] head starts from {path}{more}", flush=True)

    # ------------------------------------------------------------------ key pose probe (T0)

    def register(self, g, c2w0, gt, focal, h, w, t, iters, lr):
        """Photometric registration of one camera against the fixed map from c2w0, L1 with Adam on a
        rotation vector and a translation in render units, the loop of refine_pose in tools/eval_gaussian_map.py."""
        if iters <= 0:
            return c2w0
        dr = torch.zeros(3, device=gt.device, requires_grad=True)
        dt = torch.zeros(3, device=gt.device, requires_grad=True)
        opt = torch.optim.Adam([dr, dt], lr=lr)
        unit = float(g["unit"])

        def pose():
            return c2w0 @ delta_pose(torch.cat([dr, dt / unit]))

        for _ in range(iters):
            opt.zero_grad(set_to_none=True)
            loss = F.l1_loss(self.render_pose(g, pose(), focal, h, w, ("color_raw",), t)["color_raw"], gt)
            loss.backward()
            opt.step()
        with torch.no_grad():
            return pose().detach()

    def probe_window(self, refine_iters, refine_lr):
        """One window of the key pose probe, the rows of its targets or None when the window is skipped."""
        a, dev = self.args, self.device
        s = next(self.it)
        raw = s["raw"].to(dev, non_blocking=True).float() / 255.0
        k, f = int(s["n_ctx"]), raw.shape[0]
        h, w = raw.shape[-2:]
        with torch.no_grad():
            geo = self.geometry(raw, want_desc=True)
            ok = geo["resid"] < a.gs_max_resid
            ok[0] = True
            ctx = [i for i in range(k) if bool(ok[i])]
            tgt = [i for i in range(k, f) if bool(ok[i])]
            if len(ctx) < 2 or len(tgt) < 1:
                return None
            ci = torch.tensor(ctx, device=dev)
            times = self.window_times(s, f)[0] if self.four_d else None
            tk = time.time()
            keys = self.key_poses(geo, tgt + (ctx[1:] if self.key_ctx else []), h, w)
            key_seconds = (time.time() - tk) / len(keys)
            gm = self.with_keys(geo, {i: keys[i] for i in ctx[1:]}) if self.key_ctx else geo
            act = GaussianMapHead.activate(self.head([t[ci] for t in geo["tokens"]], raw[ci] * 2.0 - 1.0, (h, w)), raw[ci])
            splat = self.splat_act(raw[ci].permute(0, 2, 3, 1))
            maps = {"head": self.assemble_beta(gm, ci, act, h, w, times) if self.beta else self.assemble(gm, ci, act, h, w),
                    "splat": self.assemble_beta(gm, ci, splat, h, w) if self.beta else self.assemble(gm, ci, splat, h, w)}
        focal = geo["focal"]
        rows = []
        for i in tgt:
            c2w_key, acc, st = keys[i]
            c2w_proc = self.proc_pose(gm, i)
            t_i = None if times is None else times[i]
            row = {"scene": s["scene"], "frame": int(s["frames"][i]), "synth": float(bool(s["synth"])), "n_ctx": len(ctx),
                   "key_accept": float(acc), "key_reason": st["reason"], "key_n_match": float(st["n_match"]),
                   "key_inlier0": st["inlier0"], "key_inlier": st["inlier"], "key_resid0_px": st["resid0_px"],
                   "key_resid_px": st["resid_px"], "key_move_px": st["move_px"], "key_rot_deg": st["rot_deg"],
                   "key_trans_rel": st["trans"] / st["zmed"], "key_seconds": key_seconds}
            for name, g in maps.items():
                with torch.no_grad():
                    for tag, c2w in (("proc", c2w_proc), ("key", c2w_key)):
                        img = self.render_pose(g, c2w, focal, h, w, ("color_raw",), t_i)["color_raw"]
                        row[f"psnr_{name}_{tag}"] = float(psnr(img[None], raw[i][None]))
                c2w_reg = self.register(g, c2w_proc, raw[i], focal, h, w, t_i, refine_iters, refine_lr)
                with torch.no_grad():
                    img = self.render_pose(g, c2w_reg, focal, h, w, ("color_raw",), t_i)["color_raw"]
                    row[f"psnr_{name}_reg"] = float(psnr(img[None], raw[i][None]))
                xi_k, xi_p = delta_from_poses(c2w_reg, c2w_key), delta_from_poses(c2w_reg, c2w_proc)
                row.update({f"{name}_regkey_deg": math.degrees(float(xi_k[:3].norm())),
                            f"{name}_regkey_trel": float(xi_k[3:].norm()) / st["zmed"],
                            f"{name}_regproc_deg": math.degrees(float(xi_p[:3].norm())),
                            f"{name}_regproc_trel": float(xi_p[3:].norm()) / st["zmed"]})
            rows.append(row)
        return rows

    def probe_key_pose(self, n_windows, out_json, refine_iters=300, refine_lr=8.0e-4):
        """Key pose test T0, nothing is trained. For n_windows windows and each of their targets the
        PSNR on color_raw of the head map and of the plain splat map at the Procrustes pose, at the key
        pose (the Procrustes pose when rejected) and after refine_iters registration steps from the
        Procrustes pose, with the key pose statistics. out_json gets the rows, their means ('mean') and
        the means over the accepted targets ('mean_accepted')."""
        if not self.key:
            raise ValueError("probe_key_pose needs --gs_key_pose")
        was_training = self.head.training
        self.head.eval()
        rows, used, tried, failed, t0 = [], 0, 0, 0, time.time()

        def write():
            num = sorted({k for r in rows for k, v in r.items() if isinstance(v, float) and k not in ("synth",)})

            def means(sel):
                r = [x for x in rows if sel(x)]
                out = {}
                for k in num:
                    v = [x[k] for x in r if k in x and math.isfinite(x[k])]
                    out[k] = float(np.mean(v)) if v else float("nan")
                return {"n": len(r), **out}

            res = {"windows": used, "windows_tried": tried, "windows_failed": failed, "targets": len(rows),
                   "refine_iters": refine_iters, "refine_lr": refine_lr, "kernel": "beta" if self.beta else "gauss",
                   "four_d": self.four_d, "key_ctx": self.key_ctx, "key_config": vars(self.key_cfg),
                   "seconds": time.time() - t0, "mean": means(lambda r: True),
                   "mean_accepted": means(lambda r: r["key_accept"] > 0), "rows": rows}
            os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
            with open(out_json, "w") as fh:
                json.dump(res, fh, indent=1)
            return res

        while used < n_windows and tried < 3 * n_windows:
            tried += 1
            try:
                got = self.probe_window(refine_iters, refine_lr)
            except Exception as e:
                if getattr(self.args, "gs_strict", False):
                    raise
                failed += 1
                print(f"[probe] window {tried} failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
                continue
            if got is None:
                self.skipped += 1
                continue
            used += 1
            rows += got
            m = [(r["psnr_head_proc"], r["psnr_head_key"], r["psnr_head_reg"], r["key_accept"]) for r in got]
            print("[probe] window %d/%d %s | head proc %s key %s reg %s | accepted %s" % (
                used, n_windows, got[0]["scene"], *(" ".join("%.2f" % x[j] for x in m) for j in range(3)),
                "".join("y" if x[3] else "n" for x in m)), flush=True)
            if used % 10 == 0:
                write()
        res = write()
        mn = res["mean"]
        if rows:
            print("[probe] %d windows, %d targets | head map proc %.2f key %.2f reg %.2f | splat map proc %.2f key %.2f reg %.2f "
                  "| accepted %.3f, inlier %.3f (at Procrustes %.3f), move %.2f px | %.0fs" % (
                      used, len(rows), mn["psnr_head_proc"], mn["psnr_head_key"], mn["psnr_head_reg"], mn["psnr_splat_proc"],
                      mn["psnr_splat_key"], mn["psnr_splat_reg"], mn["key_accept"], mn["key_inlier"], mn["key_inlier0"],
                      mn["key_move_px"], res["seconds"]), flush=True)
        self.head.train(was_training)
        return res
