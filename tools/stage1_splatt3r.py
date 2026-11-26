#!/usr/bin/env python3
"""Splatt3R Gaussian map of a Swimm3R Stage 1 output, coupled to the Swimm3R physics head (no Stage 2).

Every Stage 1 frame is passed once with its next frame as partner; the Splatt3R head (model/splatt3r_head.py)
reads the decoder tokens of the frame. A primitive sits at the Stage 1 depth of its pixel along the ray of the
Stage 1 camera plus the predicted offset (offset and scale taken from the units of the pass to the Stage 1
units by the median depth ratio), its colour is the Stage 1 restored colour of the pixel (descattered/<frame>.jpg
of the Stage 1 output, the restored points; the inversion of training/physics_splat.py when the file is
missing) plus the head's residual, so the restored render is the restored points drawn as Gaussians. The
frames are fused in voxels as tools/eval_gaussian_map.py fuses them.

Rendering: restored = J with a black background; underwater = every primitive J exp(-beta_D z) + B_inf
(1 - exp(-beta_B z)) at its depth z in the camera, background B_inf, with the medium of the physics head (the
mean of the two Stage 1 frames nearest in frame number, beta_D scaled as in Stage 1 with --phys_beta_D_scale).

Evaluation as tools/eval_gaussian_map.py: the Stage 1 frames themselves, and with --data_dir the held out frames
of split.json at the interpolated pose, before and after --pose_iters steps of photometric pose refinement
(underwater render against the frame), PSNR on plain MSE, torchmetrics SSIM, LPIPS alex. Restored renders of the
held out frames are saved (no restored ground truth). map.npz for tools/viz/view_ff_map.py.

    tools/stage1_splatt3r.py --scene_dir STAGE1/video1 --data_dir SEG/video1 --lora LORA --physics PHYS \\
        --head HEAD.pth | --splatt3r_ckpt epoch=19-step=1200.ckpt --out_dir OUT [--phys_beta_D_scale 0.6 1 1]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from utils import cuda_linalg_cpu  # noqa: E402

cuda_linalg_cpu.install()
import mast3r.utils.path_to_dust3r  # noqa: E402,F401
import eval_gaussian_map as E  # noqa: E402
from model.anchor_head import AnchorHead, ScaffoldHead, band_limit, build_anchors, describe, pixel_normals, slot_at_time  # noqa: E402
from model.splatt3r_head import Splatt3RHead, load_splatt3r  # noqa: E402
from training.gaussian_branch import NEAR_UNITS, quat_mul, render_view, render_view_beta, rotmat_to_quat, voxel_fuse  # noqa: E402
from training.primitives4d import EPS2D_BETA4, K_BETA4, frame_gaps  # noqa: E402
from training.physics_splat import background, invert, load_physics, medium, underwater_colours  # noqa: E402
from training import medium_raster as MR  # noqa: E402


def frame_passes(net, head, imgs, chunk=4, features=False):
    """Head output, depth of the pass and encoder tokens of every frame, partner the next frame. With features also
    the 128 channel features of the head before its last convolution (float16)."""
    n, _, h, w = imgs.shape
    dev = imgs.device
    shape = torch.tensor([[h, w]], device=dev)
    feats, poss = [], []
    for s in range(0, n, chunk):
        f, p, _ = net._encode_image(imgs[s:s + chunk] * 2.0 - 1.0, shape.repeat(min(chunk, n - s), 1))
        feats.append(f)
        poss.append(p)
    feat, pos = torch.cat(feats), torch.cat(poss)
    partner = torch.arange(n, device=dev) + 1
    partner[-1] = n - 2
    raws, zs, fps = [], [], []
    for s in range(0, n, chunk):
        i = torch.arange(s, min(s + chunk, n), device=dev)
        dec1, _ = net._decoder(feat[i], pos[i], feat[partner[i]], pos[partner[i]])
        dec1 = [t.float() for t in dec1]
        with torch.autocast("cuda", enabled=False):
            res1 = net._downstream_head(1, dec1, shape.repeat(len(i), 1))
            if features:
                r, fpx = head.forward_features(dec1, (h, w))
                raws.append(r)
                fps.append(fpx.half())
            else:
                raws.append(head(dec1, (h, w)))
        zs.append(res1["pts3d"][..., 2].float())
    if features:
        return torch.cat(raws), torch.cat(zs), feat.float(), torch.cat(fps)
    return torch.cat(raws), torch.cat(zs), feat.float()


def stage1_restored(scene_dir, names, h, w, dev):
    """Stage 1 restored frames descattered/<stem>.jpg at the processed size, or None for a missing one."""
    from PIL import Image
    out = []
    for nm in names:
        p = os.path.join(scene_dir, "descattered", os.path.splitext(nm)[0] + ".jpg")
        if not os.path.isfile(p):
            p = os.path.join(scene_dir, "descattered", nm)
        if not os.path.isfile(p):
            out.append(None)
            continue
        im = Image.open(p).convert("RGB")
        if im.size != (w, h):
            out.append(None)
            continue
        out.append(torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1).to(dev))
    return out


def main():
    ap = argparse.ArgumentParser(description="Splatt3R Gaussian map of a Stage 1 output with the physics coupling")
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--physics", required=True)
    ap.add_argument("--head", default=None, help="trained head (train_gaussian_head.py --gs_head splatt3r, *_splatt3r.pth)")
    ap.add_argument("--splatt3r_ckpt", default=None, help="public Splatt3R checkpoint, used when --head is not given")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mast3r", default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    ap.add_argument("--phys_beta_D_scale", type=float, nargs=3, default=None)
    ap.add_argument("--voxel_px", type=float, default=1.0)
    ap.add_argument("--min_conf_thr", type=float, default=1.5)
    ap.add_argument("--anchor_px", type=float, default=None,
                    help="anchor edge in pixel footprints when the head has an anchor head (default 8)")
    ap.add_argument("--pose_iters", type=int, default=300)
    ap.add_argument("--pose_lr", type=float, default=8e-4)
    ap.add_argument("--save_images", type=int, default=8)
    ap.add_argument("--medium", choices=("nearest", "field", "views", "both", "all"), default="field",
                    help="medium of a held out view: the mean of the two map frames nearest in frame number (nearest), "
                         "the medium field fitted to the map frames (field, model/medium_field.py, saved as "
                         "medium_field.pt), the B_inf maps of the map frames that saw each ray direction with no fitting "
                         "(views), nearest and field (both), or all three, each scored separately (keys with _field, _views)")
    ap.add_argument("--medium_iters", type=int, default=1500)
    ap.add_argument("--fps", type=float, default=5.0,
                    help="frames per second of the frame numbers: the time of a frame is its number / fps (4D scaffold heads)")
    ap.add_argument("--scaffold_mode", choices=("infer", "soft"), default="infer",
                    help="scaffold heads with use probabilities: keep the slots with use > 0.5 (infer), or draw every slot "
                         "at opacity times use as in the soft training, leaving out those up to opacity 1/255 (soft). "
                         "Heads trained always on draw the same map either way")
    ap.add_argument("--diag", type=int, default=0,
                    help="diagnosis of the map on N evenly spaced map frames (key diag of scores.json): the fused map and a map "
                         "built from that frame alone, both with the frame's own medium, PSNR on all pixels, on the pixels the "
                         "map covers (alpha > 0.5) and on the rest, the share of the squared error in uncovered pixels, the fused "
                         "map with the medium of the two nearest other frames, and the water alone (B_inf, no Gaussians)")
    ap.add_argument("--no_parent", action="store_true",
                    help="probe on a trained scaffold head: the parent Gaussians are left out of the map")
    ap.add_argument("--detail", type=float, nargs=2, default=None, metavar=("S", "FRAC"),
                    help="probe on a scaffold head trained without the detail layer: the FRAC of the cells with the most colour "
                         "detail drawn as S x S untrained fine slots (the fine cells themselves) in place of their child")
    ap.add_argument("--beta_fixed", type=float, default=None,
                    help="probe on a trained Beta scaffold head: every slot drawn with this Beta shape")
    ap.add_argument("--band", type=float, default=None, metavar="C",
                    help="probe: 3D band limit of the map (model/anchor_head.py band_limit) with the interval C slot pitches of a "
                         "scaffold head or C pixel footprints of the per pixel head, opacity unchanged")
    ap.add_argument("--medium_raster", choices=("s2", "preshade"), default="s2",
                    help="underwater render of a Beta kernel map: s2 gives the rasteriser the medium as three per pixel maps (veiling "
                         "light, beta_B, beta_D at the depth of the pixel) and it accumulates the water between the primitives along "
                         "the ray with its attenuation aware alpha, as the per scene renderer of Stage 2 does; preshade shades every "
                         "primitive before the rasteriser (the render up to v9c)")
    ap.add_argument("--time_bucket", type=int, default=0, metavar="K",
                    help="probe of a time local map: next to the map of all frames, one map per window of K consecutive map frames "
                         "(windows start every K // 2 frames), and every view is drawn from the window whose mean time is nearest to "
                         "the time of the view, so a primitive only exists on its stretch of the trajectory (0 off)")
    ap.add_argument("--exist_scale", type=float, default=1.0,
                    help="maps of a head with existence intervals: the width of every slot times this factor (0 draws every slot at every "
                         "time, the map without time)")
    ap.add_argument("--grad_deconf", action="store_true",
                    help="--medium_raster s2: the pose steps use the gradient separation of the final Stage 2 runs (the error the medium "
                         "explains is kept out of the gradient of the projected centres)")
    ap.add_argument("--far_layer", action="store_true",
                    help="anchor head maps: the pixels the confidence filter leaves out also get primitives, at the depth of the backbone, "
                         "as a second map drawn behind the map (and behind the time window). No ray is empty then, and the colour of a "
                         "far region comes from the attenuation and the backscatter at its depth as everywhere else; nothing is taken "
                         "as open water. Old render path only")
    ap.add_argument("--binf_veil", action="store_true",
                    help="medium field: B_inf is fitted to the map frames themselves through the image formation at the depth of every "
                         "pixel, not to the B_inf of the physics head alone. Where the map has a primitive the frame equals its water "
                         "free colour times exp(-beta_D z) plus B_inf (1 - exp(-beta_B z)) at the rendered depth; where it has none "
                         "the surface colour is unknown between 0 and 1 and the depth is the one of the backbone, so the frame lies "
                         "between B_inf (1 - exp(-beta_B z)) and that plus exp(-beta_D z). A far pixel fixes B_inf, a near one says "
                         "nothing, no pixel is declared water. The B_inf of the head stays as a prior of weight --binf_prior")
    ap.add_argument("--binf_prior", type=float, default=0.06,
                    help="--binf_veil: weight of the squared difference to the B_inf of the head against the squared excess of the frame")
    ap.add_argument("--time_fill", action="store_true",
                    help="with --time_bucket and --medium_raster preshade: the primitives of the window are drawn in front of the map of "
                         "all frames, which shows wherever the window has none, so the window decides what a view shows where it has "
                         "seen the surface and nothing is left uncovered")
    ap.add_argument("--scale_cap", type=float, default=None, metavar="R",
                    help="probe: soft bound R on the positive log size output of every slot of a scaffold head (ScaffoldHead scale_cap)")
    ap.add_argument("--overlap", type=int, default=0,
                    help="overlap of the map (key overlap of scores.json, tools/map_overlap.py): the in plane sigma over the distance "
                         "to the nearest neighbours, and on N map frames the share of the confident pixels with alpha > 0.5 at 1, 2 "
                         "and 4 times the focal length")
    ap.add_argument("--gate_test", type=int, default=0,
                    help="4D scaffold maps: whether the gate encodes content (key gate_test of scores.json, tools/gate_metric.py): "
                         "variance share of the slot kind, rank agreement inside every kind with the see through share and the "
                         "colour difference over the map frames, and on N held out frames the PSNR with the gate of the map, with "
                         "the gate shuffled inside every kind and with gate 0")
    ap.add_argument("--bench", action="store_true",
                    help="benchmark of the localization in place of the standard evaluation of the held out frames "
                         "(tools/bench_loc.py): tracking FPS at several numbers of pose steps, rendering FPS and the memory "
                         "footprint, printed as one table and saved as benchmark.txt")
    ap.add_argument("--bench_steps", type=int, nargs="+", default=[10, 25, 50, 100, 200, 300])
    ap.add_argument("--bench_init", nargs="+", choices=("interp", "motion"), default=["interp", "motion"])
    ap.add_argument("--bench_render_reps", type=int, default=5)
    ap.add_argument("--bench_warmup", type=int, default=2)
    ap.add_argument("--bench_mapping_seconds", type=float, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device(args.device)
    t0 = time.time()

    sc = E.load_scene(args.scene_dir, dev)
    names, imgs, depth, conf, intr, c2w = sc["names"], sc["imgs"], sc["depth"], sc["conf"], sc["intr"], sc["c2w"]
    n, _, h, w = imgs.shape
    diag_idx = sorted(set(np.linspace(0, n - 1, args.diag).round().astype(int).tolist())) if args.diag > 0 else []
    from model.lora_model import LoRAMast3r
    net = LoRAMast3r.from_pretrained(args.mast3r)
    net.load_pretrained_lora(args.lora)
    net = net.to(dev).eval()
    head = Splatt3RHead(net).to(dev).eval()
    anchor = None
    if args.head:
        ck = torch.load(args.head, map_location="cpu")
        head.load_state_dict(ck["head"])
        src = args.head
        if "anchor" in ck and args.anchor_px is not None and args.anchor_px <= 0:
            ap.error("this head was trained with the anchor head, its last 14 channel convolution never trained: give --anchor_px > 0")
        if "anchor" in ck:
            cfg = dict(ck["anchor_cfg"])
            probe = {}
            if cfg.get("scaffold"):
                if args.no_parent:
                    probe["parent_slot"] = False
                if args.detail:
                    probe.update(detail=int(args.detail[0]), detail_frac=float(args.detail[1]))
                if args.beta_fixed is not None:
                    probe["beta_fixed"] = float(args.beta_fixed)
                if args.band is not None:
                    probe["band"] = float(args.band)
                if args.scale_cap is not None:
                    probe["scale_cap"] = float(args.scale_cap)
                cfg.update(probe)
                anchor = ScaffoldHead(**cfg)
                anchor.mode = "infer_soft" if args.scaffold_mode == "soft" else "infer"
            else:
                anchor = AnchorHead(feat_dim=cfg["feat_dim"], width=cfg["width"], blocks=cfg["blocks"], opacity_bias=cfg["opacity_bias"])
            miss = anchor.load_state_dict(ck["anchor"], strict=not probe)
            if probe:
                if [k for k in miss.missing_keys if not k.startswith("detail_")] or miss.unexpected_keys:
                    ap.error(f"the head does not fit the probe {probe}: missing {miss.missing_keys[:6]}, unexpected {miss.unexpected_keys[:6]}")
                print(f"[probe] {probe}" + (f", {len(miss.missing_keys)} detail tensors untrained" if miss.missing_keys else ""), flush=True)
            anchor = anchor.to(dev).eval()
            if args.anchor_px is None:
                args.anchor_px = 8.0
            src = f"{args.head} ({describe(anchor)}, trained on edges {ck.get('anchor_px')}, map edge {args.anchor_px:g})"
    elif args.splatt3r_ckpt:
        load_splatt3r(head, args.splatt3r_ckpt)
        src = args.splatt3r_ckpt + " (public Splatt3R head, not trained on our data)"
    else:
        ap.error("--head or --splatt3r_ckpt")
    phys = load_physics(args.physics, dev, betaB_le_betaD=True)
    print(f"[s3r] {n} Stage 1 frames at {h}x{w} | head {src} | physics {phys.coupling}", flush=True)

    with torch.no_grad():
        if anchor is not None:
            raw, zpass, feat, featpx = frame_passes(net, head, imgs, features=True)
        else:
            raw, zpass, feat = frame_passes(net, head, imgs)
        act = Splatt3RHead.activate(raw)
        med = medium(phys, feat, depth, beta_D_scale=args.phys_beta_D_scale)
        J = invert(phys, imgs, depth, med)
        s1 = stage1_restored(args.scene_dir, names, h, w, dev)
        n_s1 = sum(1 for x in s1 if x is not None)
        for i, x in enumerate(s1):
            if x is not None:
                J[i] = x
        print(f"[s3r] restored colour: Stage 1 descattered frames for {n_s1}/{n}, inversion for the rest", flush=True)
        # units of the pass to the Stage 1 units
        ok = (depth > 1e-4) & (zpass > 1e-4)
        ratio = torch.stack([(depth[i][ok[i]] / zpass[i][ok[i]]).median() for i in range(n)])
        v, u = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32) + 0.5,
                              torch.arange(w, device=dev, dtype=torch.float32) + 0.5, indexing="ij")
        parts = {k: [] for k in ("xyz", "fp", "conf", "zsrc", "opacity", "scales", "quat", "J", "raw", "vid")}
        for i in range(n):
            f_, cx, cy = intr[i, 0, 0], intr[i, 0, 2] + 0.5, intr[i, 1, 2] + 0.5
            z = depth[i]
            cam = torch.stack([(u - cx) / f_ * z, (v - cy) / f_ * z, z], -1) + act["offset"][i] * ratio[i]
            xyz = cam @ c2w[i, :3, :3].T + c2w[i, :3, 3]
            good = torch.isfinite(xyz).all(-1) & (z > 1e-4) & (conf[i] > args.min_conf_thr)
            parts["xyz"].append(xyz[good])
            parts["fp"].append((z / f_)[good])
            parts["conf"].append(conf[i][good])
            parts["zsrc"].append(z[good])
            parts["opacity"].append(act["opacity"][i][good])
            sc_i = (act["scales"][i] * ratio[i])[good]
            parts["scales"].append(band_limit(sc_i, float(args.band) * (z / f_)[good]) if (args.band and anchor is None) else sc_i)
            parts["quat"].append(quat_mul(rotmat_to_quat(c2w[i, :3, :3])[None, None].expand(h, w, 4), act["quat"][i])[good])
            parts["J"].append(Splatt3RHead.colour(act["sh_dc"][i], J[i].permute(1, 2, 0))[good])
            parts["raw"].append(imgs[i].permute(1, 2, 0)[good])
            parts["vid"].append(torch.full((int(good.sum()),), i, device=dev, dtype=torch.long))
        pr = {k: torch.cat(vv) for k, vv in parts.items()}
        n_in = pr["xyz"].shape[0]
        tmaps = []
        g_far = None
        zs = depth[depth > 1e-4]
        unit = NEAR_UNITS / torch.quantile(zs[:: max(1, zs.numel() // 200000)], 0.01).clamp_min(1e-4)
        if anchor is not None:
            pa = {k: [] for k in ("xyz", "fp", "conf", "cam", "camx", "nrm", "vid", "feat", "J", "raw", "t")}
            pf = {k: [] for k in pa}
            tf = torch.tensor([E.fnum(x) / args.fps for x in names], device=dev, dtype=torch.float32)
            for i in range(n):
                f_, cx, cy = intr[i, 0, 0], intr[i, 0, 2] + 0.5, intr[i, 1, 2] + 0.5
                z = depth[i]
                xyz0 = torch.stack([(u - cx) / f_ * z, (v - cy) / f_ * z, z], -1) @ c2w[i, :3, :3].T + c2w[i, :3, 3]
                nrm = pixel_normals(xyz0[None], z[None])[0]
                good = torch.isfinite(xyz0).all(-1) & (z > 1e-4) & (conf[i] > args.min_conf_thr)
                k = int(good.sum())
                pa["xyz"].append(xyz0[good])
                pa["fp"].append((z / f_)[good])
                pa["conf"].append(conf[i][good])
                pa["cam"].append(c2w[i, :3, 3].expand(k, 3))
                pa["camx"].append(c2w[i, :3, 0].expand(k, 3))
                pa["nrm"].append(nrm[good])
                pa["vid"].append(torch.full((k,), i, device=dev, dtype=torch.long))
                pa["feat"].append(featpx[i][good])
                pa["J"].append(J[i].permute(1, 2, 0)[good])
                pa["raw"].append(imgs[i].permute(1, 2, 0)[good])
                pa["t"].append(tf[i].expand(k))
                if args.far_layer:
                    far = torch.isfinite(xyz0).all(-1) & (z > 1e-4) & ~good
                    kf = int(far.sum())
                    for k_, v_ in dict(xyz=xyz0[far], fp=(z / f_)[far], conf=conf[i][far], cam=c2w[i, :3, 3].expand(kf, 3), camx=c2w[i, :3, 0].expand(kf, 3),
                                       nrm=nrm[far], vid=torch.full((kf,), i, device=dev, dtype=torch.long), feat=featpx[i][far],
                                       J=J[i].permute(1, 2, 0)[far], raw=imgs[i].permute(1, 2, 0)[far], t=tf[i].expand(kf)).items():
                        pf[k_].append(v_)
            del featpx
            pa_single = {i: {k: pa[k][i] for k in pa} for i in diag_idx}
            pa = {k: torch.cat(vv) for k, vv in pa.items()}
            gap = float(frame_gaps(tf).median()) if n > 1 else 1.0

            def anchor_map(p):
                a_ = build_anchors(p["xyz"], p["fp"], p["conf"], p["cam"], p["camx"], p["nrm"], p["vid"], p["feat"], p["J"],
                                   p["raw"], args.anchor_px, quadrants=bool(getattr(anchor, "scaffold", False)),
                                   cells=int(getattr(anchor, "cells", 2)),
                                   times=p["t"] if getattr(anchor, "cfg", {}).get("four_d") else None,
                                   detail=int(getattr(anchor, "cfg", {}).get("detail", 0)),
                                   detail_frac=float(getattr(anchor, "cfg", {}).get("detail_frac", 0.0)))
                o_ = anchor(a_)
                gm = dict(xyz=o_["means"], means=o_["means"] * unit, scales=o_["scales"] * unit, quat=o_["quat"], opacity=o_["opacity"],
                          J=o_["J"], raw=o_["raw"], unit=unit, n_in=int(p["xyz"].shape[0]), n_out=o_["means"].shape[0])
                if "beta" in o_:
                    gm["beta"] = o_["beta"]
                if "gate" in o_:
                    gm.update(gate=o_["gate"], tw=o_["tw_rel"] * gap, tb=o_["tb"], vel=o_["vel"], t_src=o_["t_src"])
                if "transient" in o_:
                    gm["transient"] = o_["transient"]
                if "exist" in o_:
                    gm["exist"] = o_["exist"]
                if "slot" in o_:
                    gm["slot"] = o_["slot"]
                return a_, o_, gm

            def single_map(i):
                return anchor_map(pa_single[i])[2] if pa_single[i]["xyz"].shape[0] >= 16 else None

            anc, ga, g = anchor_map(pa)
            if args.far_layer:
                pf = {k: torch.cat(vv) for k, vv in pf.items()}
                if pf["xyz"].shape[0] >= 1000:
                    g_far = anchor_map(pf)[2]
                    zc_, zf_ = pa["fp"].median(), pf["fp"].median()
                    print(f"[far] {g_far['n_out'] / 1e6:.3f}M Gaussians from the {pf['xyz'].shape[0] / 1e6:.2f}M pixels the confidence filter leaves out "
                          f"({pf['xyz'].shape[0] / max(1, pf['xyz'].shape[0] + n_in):.3f} of the pixels with a depth), median footprint "
                          f"{float(zf_ / zc_.clamp_min(1e-9)):.2f} times the one of the map, opacity mean {float(g_far['opacity'].mean()):.3f}; "
                          f"drawn behind the map", flush=True)
            del pf
            if args.time_bucket > 0:
                order = torch.argsort(tf).tolist()
                kb, sb = int(args.time_bucket), max(1, int(args.time_bucket) // 2)
                for s0 in range(0, max(1, n - kb + sb), sb):
                    idx = order[s0:s0 + kb]
                    sel_ = torch.isin(pa["vid"], torch.tensor(idx, device=dev))
                    if not idx or int(sel_.sum()) < 1000:
                        continue
                    tmaps.append((float(tf[idx].mean()), anchor_map({k: v[sel_] for k, v in pa.items()})[2]))
                print(f"[time] {len(tmaps)} windows of {kb} map frames every {sb}: {sum(m_['n_out'] for _, m_ in tmaps) / 1e6:.2f}M Gaussians in all, "
                      f"{np.mean([m_['n_out'] for _, m_ in tmaps]) / 1e6:.3f}M per window (the map of all frames has {g['n_out'] / 1e6:.2f}M), a view "
                      f"is drawn from the window nearest in time", flush=True)
            del pa["feat"]
            if "exist" in ga:
                wq_ = torch.quantile(ga["tw_rel"].float()[:: max(1, ga["tw_rel"].numel() // 200000)], torch.tensor([0.05, 0.5, 0.95], device=dev))
                print(f"[4d] existence intervals: width p05/p50/p95 {float(wq_[0]):.2f}/{float(wq_[1]):.2f}/{float(wq_[2]):.2f} frame gaps "
                      f"(gap {gap:.3f} s), drawn with the widths times {args.exist_scale:g}", flush=True)
            elif "gate" in ga:
                dyn = (ga["opacity"] * ga["gate"].clamp(0, 1)) > 1.0 / 255.0
                print(f"[4d] gate mean {float(ga['gate'].mean()):.4f}, {int(dyn.sum())} slots with a dynamic copy "
                      f"({float(dyn.float().mean()):.4f}), velocity median {float(ga['vel'].norm(dim=-1).median()):.5f} units per s, "
                      f"frame gap {gap:.3f} s, beta mean {float(ga['beta'].mean()) if 'beta' in ga else 4.0:.2f}", flush=True)
            elif "beta" in ga:
                print(f"[beta] beta mean {float(ga['beta'].mean()):.2f}", flush=True)
            fpa = pa["fp"].median()
            se = ga["scale_edges"]
            print(f"[anchor] edge {args.anchor_px:g} footprints: {g['n_out'] / 1e6:.3f}M Gaussians from {n_in / 1e6:.2f}M pixels "
                  f"({g['n_out'] / n_in:.4f} per pixel), {float(anc['views'].mean()):.1f} views per anchor, largest axis "
                  f"{float((ga['scales'].max(-1).values / fpa).median()):.2f} median footprints, in edges median / p99 / max "
                  f"{float(se.median()):.2f} / {float(torch.quantile(se, 0.99)):.2f} / {float(se.max()):.2f}, offset "
                  f"{float(ga['offset'].norm(dim=-1).mean()):.3f} edges, non finite {int((~torch.isfinite(g['means']).all(-1)).sum())}",
                  flush=True)
            if "parent_kept" in ga:
                na = max(1, ga["n_anchor"])
                how = "always on" if anchor.cfg.get("always_on") else f"mode {anchor.mode}"
                print(f"[scaffold] {how}, {ga['n_anchor']} anchors: parents kept {ga['parent_kept'] / na:.2f}, children kept "
                      f"{ga['child_kept'] / na:.2f} of {ga['n_child'] / na:.2f} per anchor, {g['n_out'] / n_in:.4f} Gaussians per pixel, "
                      f"opacity mean {float(ga['opacity'].mean()):.3f}"
                      + (f", detail slots {ga['detail_kept'] / na:.2f} per anchor ({ga['detail_kept'] / max(1, g['n_out']):.2f} of the slots)"
                         if "detail_kept" in ga else ""), flush=True)
        else:
            def pixel_map(q):
                xyz_, m_, p_ = voxel_fuse(q["xyz"], q["conf"], q["fp"], dict(opacity=q["opacity"], J=q["J"], raw=q["raw"], zsrc=q["zsrc"]),
                                          dict(scales=q["scales"], quat=q["quat"]), args.voxel_px)
                return dict(xyz=xyz_, means=xyz_ * unit, scales=p_["scales"] * unit, quat=p_["quat"], opacity=m_["opacity"], J=m_["J"],
                            raw=m_["raw"], unit=unit, n_in=int(q["xyz"].shape[0]), n_out=xyz_.shape[0])

            def single_map(i):
                sel_ = pr["vid"] == i
                return pixel_map({k: v[sel_] for k, v in pr.items()}) if int(sel_.sum()) >= 16 else None

            g = pixel_map(pr)
    t_map = time.time() - t0
    print(f"[map] confidence > {args.min_conf_thr}: {n_in / 1e6:.2f}M of {n * h * w / 1e6:.2f}M pixels", flush=True)
    print(f"[map] {n_in / 1e6:.2f}M primitives, {g['n_out'] / 1e6:.2f}M after fusion, offset/depth "
          f"{float((act['offset'].norm(dim=-1) * ratio[:, None, None] / depth.clamp_min(1e-4))[ok].mean()):.4f}, "
          f"opacity {float(g['opacity'].mean()):.3f}, {t_map:.0f}s", flush=True)
    focal = float(intr[:, 0, 0].median())
    pp = (float(intr[:, 0, 2].median()) + 0.5, float(intr[:, 1, 2].median()) + 0.5)
    fnums = [E.fnum(x) for x in names]

    def med_of(order):
        idx = torch.as_tensor(np.asarray(order), device=dev)
        out = dict(params_D=med["params_D"][idx].mean(0), params_B=med["params_B"][idx].mean(0), B_inf=med["B_inf"][idx].mean(0))
        if "B_inf_map" in med:
            out["B_inf_map"] = med["B_inf_map"][idx].mean(0)
        if "beta_D_scale" in med:
            out["beta_D_scale"] = med["beta_D_scale"]
        return out

    def med_near(num):
        return med_of(np.argsort([abs(k - num) for k in fnums])[:2])

    def med_others(i):
        return med_of([j for j in np.argsort([abs(k - fnums[i]) for k in fnums]) if j != i][:2])

    want = dict(nearest=["nearest"], field=["field"], views=["views"], both=["nearest", "field"],
                all=["nearest", "field", "views"])[args.medium]
    field, views, medium_info = None, None, None
    if "field" in want or "views" in want:
        try:
            from model.medium_field import ViewMedium, binf_error, fit_medium_field, held_out_check
            tf = time.time()
            ho = held_out_check(med, c2w, intr, h, w, fnums, iters=args.medium_iters)
            obs_w, veil_info = None, None
            if "field" in want and args.binf_veil:
                from training.physics_splat import veil_terms
                al_, jp_, zs_ = [], [], []
                with torch.no_grad():
                    for i in range(n):
                        cu_ = c2w[i].clone()
                        cu_[:3, 3] = c2w[i, :3, 3] * unit
                        fi_, ci_ = float(intr[i, 0, 0]), (float(intr[i, 0, 2]) + 0.5, float(intr[i, 1, 2]) + 0.5)
                        one_ = lambda gm_: (render_view_beta(gm_["means"], gm_["opacity"], gm_["scales"], gm_["quat"], gm_["J"], cu_, fi_, h, w,
                                                             torch.zeros(3, device=dev), pp=ci_, betas=gm_["beta"], scale_factor=K_BETA4, eps2d=EPS2D_BETA4)
                                            if "beta" in gm_ else
                                            render_view(gm_["means"], gm_["opacity"], gm_["scales"], gm_["quat"], gm_["J"], cu_, fi_, h, w,
                                                        torch.zeros(3, device=dev), pp=ci_))
                        ji_, di_, a_ = one_(g)
                        if g_far is not None:
                            jf_, df_, af_ = one_(g_far)
                            ji_, di_, a_ = ji_ + (1.0 - a_) * jf_, di_ + (1.0 - a_) * df_, a_ + (1.0 - a_) * af_
                        al_.append(a_.reshape(h, w))
                        jp_.append(ji_.reshape(3, h, w))
                        zs_.append(MR.expected_depth(di_, a_, unit))
                    al_ = torch.stack(al_)
                    m_all = med_of(list(range(n)))
                    obs_w = dict(veil_terms(phys, m_all, al_, torch.stack(jp_), torch.stack(zs_), depth.float()), img=imgs.float())
                    em_ = al_ < 0.5
                    vr_, vs_ = obs_w["veil_reference"].mean(1), obs_w["veil_surface"].mean(1)
                    zr_ok = torch.isfinite(depth) & (depth > 1e-6)
                    veil_info = dict(empty=float(em_.float().mean()), veil_empty=float(vr_[em_].mean()) if bool(em_.any()) else 0.0,
                                     veil_empty_above_half=float((vr_[em_] > 0.5).float().mean()) if bool(em_.any()) else 0.0,
                                     veil_covered=float(vs_[~em_].mean()) if bool((~em_).any()) else 0.0,
                                     depth_empty_over_covered=float(depth[em_ & zr_ok].median() / depth[~em_ & zr_ok].median().clamp_min(1e-6))
                                     if bool((em_ & zr_ok).any()) and bool((~em_ & zr_ok).any()) else 0.0,
                                     empty_without_depth=float((em_ & ~zr_ok).float().sum() / em_.float().sum().clamp_min(1.0)), prior=args.binf_prior)
                print(f"[medium field] veil fit: {veil_info['empty']:.3f} of the map frame pixels have no primitive, their backbone depth is "
                      f"{veil_info['depth_empty_over_covered']:.2f} times the depth of the covered pixels (median), veil share 1 - exp(-beta_B z) there "
                      f"{veil_info['veil_empty']:.3f} (above 0.5 on {veil_info['veil_empty_above_half']:.2f} of them, no depth on "
                      f"{veil_info['empty_without_depth']:.2f}), on the covered pixels {veil_info['veil_covered']:.3f}", flush=True)
            if "field" in want:
                field = (fit_medium_field(med, c2w, intr, h, w, iters=args.medium_iters) if obs_w is None else
                         fit_medium_field(med, c2w, intr, h, w, iters=args.medium_iters, step=8, obs=obs_w, prior=args.binf_prior))
                torch.save(field.state(), os.path.join(args.out_dir, "medium_field.pt"))
                if veil_info is not None:
                    veil_info["moved"] = binf_error(field, med, c2w, intr, list(range(n)), h, w)
                    print(f"[medium field] veil fit: B_inf of the field differs from the B_inf of the head by {veil_info['moved']:.4f} on the map frames (mean absolute)", flush=True)
            if "views" in want:
                views = ViewMedium(med, c2w, intr, h, w)
            medium_info = dict(ho, seconds=time.time() - tf)
            if veil_info is not None:
                medium_info["binf_veil"] = veil_info
            print(f"[medium field] B_inf error on {ho['n_left_out']} left out map frames: field {ho['field']:.4f}, views "
                  f"{ho['views']:.4f}, mean of the two nearest frames {ho['nearest']:.4f} | spread over the frames of params_D "
                  f"{ho['spread_params_D']:.4f}, of B_inf {ho['spread_binf']:.4f} | {medium_info['seconds']:.0f}s", flush=True)
        except Exception as e:
            print(f"[medium field] failed, only the medium of the nearest frames is scored: {type(e).__name__}: {e}", flush=True)
            field, views = None, None
    modes = [m for m in want if m == "nearest" or (m == "field" and field is not None) or (m == "views" and views is not None)]
    if not modes:
        modes = ["nearest"]
    sfx = {m: ("" if m == modes[0] else "_" + m) for m in ("nearest", "field", "views")}
    tag = {m: ("" if m == modes[0] else " " + m) for m in ("nearest", "field", "views")}
    print(f"[medium] held out views use {modes[0]}" + (f", also scored with {', '.join(modes[1:])}" if len(modes) > 1 else ""), flush=True)

    def medium_for(mode, num, pose, f=focal, c=pp):
        if mode == "field":
            return field.medium_at(pose, f, c, h, w)
        if mode == "views":
            return views.medium_at(pose, f, c, h, w)
        return med_near(num)

    def at_time(t, gm=None):
        """The map gm (default g) at time t: the static and dynamic copies of a 4D scaffold (model/anchor_head.py
        slot_at_time) with xyz in Stage 1 units and means in render units, gm itself for a static map."""
        if gm is None and tmaps and t is not None:
            gm = tmaps[min(range(len(tmaps)), key=lambda j_: abs(tmaps[j_][0] - float(t)))][1]
        gm = g if gm is None else gm
        if t is None or "gate" not in gm or (float(gm.get("exist", 0.0) or 0.0) > 0.0 and args.exist_scale <= 0.0):
            return gm
        gt_ = slot_at_time(dict(means=gm["xyz"], opacity=gm["opacity"], scales=gm["scales"], quat=gm["quat"], J=gm["J"],
                                beta=gm["beta"], gate=gm["gate"], tw=gm["tw"] * (float(args.exist_scale) if gm.get("exist") else 1.0),
                                tb=gm["tb"], vel=gm["vel"], t_src=gm["t_src"],
                                transient=gm.get("transient", 0.0), exist=gm.get("exist", 0.0)), t)
        return dict(gt_, xyz=gt_["means"], means=gt_["means"] * unit)

    def rv(gg, colors, cu, f, hh, ww, bg, c):
        if "beta" in gg:
            return render_view_beta(gg["means"], gg["opacity"], gg["scales"], gg["quat"], colors, cu, f, hh, ww, bg, pp=c,
                                    betas=gg["beta"], scale_factor=K_BETA4, eps2d=EPS2D_BETA4)
        return render_view(gg["means"], gg["opacity"], gg["scales"], gg["quat"], colors, cu, f, hh, ww, bg, pp=c)

    def stack(t, gm):
        if gm is not None:
            return [gm]
        fill = bool(args.time_fill and tmaps and t is not None)
        return ([None, g] if fill else [None]) + ([g_far] if g_far is not None else [])

    def render(pose, m1, f=focal, c=pp, hw=(h, w), which="underwater", t=None, gm=None, alpha=False):
        ls = stack(t, gm)
        if len(ls) == 1 or not (args.medium_raster == "preshade" and (which == "restored" or "B_inf_map" in m1)):
            return render_one(pose, m1, f, c, hw, which, t, gm, alpha)
        img_, tr_ = 0.0, 1.0
        if which == "restored":
            cu_ = pose.clone()
            cu_[:3, 3] = pose[:3, 3] * unit
            for l_ in ls:
                gl_ = at_time(t, l_)
                r_, _, a_ = rv(gl_, gl_["J"], cu_, f, hw[0], hw[1], torch.zeros(3, device=dev), c)
                img_ = img_ + tr_ * r_
                tr_ = tr_ * (1.0 - a_.reshape(1, hw[0], hw[1]))
            return img_
        for l_ in ls:
            o_, a_ = render_one(pose, m1, f, c, hw, which, t, l_, True, obj=True)
            img_ = img_ + tr_ * o_
            tr_ = tr_ * (1.0 - a_.reshape(1, hw[0], hw[1]))
        img_ = img_ + tr_ * background(m1, hw[0], hw[1]).float()
        return (img_, 1.0 - tr_[0]) if alpha else img_

    def render_one(pose, m1, f=focal, c=pp, hw=(h, w), which="underwater", t=None, gm=None, alpha=False, obj=False):
        gg = at_time(t, gm)
        cu = pose.clone()
        cu[:3, 3] = pose[:3, 3] * unit
        if which == "restored":
            return rv(gg, gg["J"], cu, f, hw[0], hw[1], torch.zeros(3, device=dev), c)[0]
        if args.medium_raster == "s2" and "beta" in gg:
            with torch.no_grad():
                _, dacc, a0 = rv(gg, gg["J"], cu, f, hw[0], hw[1], torch.zeros(3, device=dev), c)
                m_rgb, m_bs, m_attn = MR.medium_maps(phys, m1, MR.expected_depth(dacc, a0, unit), unit)
            img, _, _ = MR.render_medium_view(gg["means"], gg["opacity"], gg["scales"], gg["quat"], gg["J"], gg["beta"], cu, f, hw[0], hw[1],
                                              c, m_rgb, m_bs, m_attn, scale_factor=K_BETA4, eps2d=EPS2D_BETA4, deconf=bool(args.grad_deconf))
            return (img, a0.reshape(hw[0], hw[1])) if alpha else img
        xc = (gg["xyz"] - pose[:3, 3]) @ pose[:3, :3]
        zc = xc[:, 2]
        if "B_inf_map" not in m1:
            col = underwater_colours(phys, gg["J"], zc, m1)
            img, _, alp = rv(gg, col, cu, f, hw[0], hw[1], m1["B_inf"].float().contiguous(), c)
            return (img, alp.reshape(hw[0], hw[1])) if alpha else img
        # direction dependent B_inf: the map of the Stage 1 grid (h, w) sampled at the pixel of every primitive,
        # open water the map resampled to the render size
        zs = torch.where(zc > 1e-6, zc, torch.ones_like(zc))
        uv = torch.stack([(f * xc[:, 0] / zs + c[0]) * w / hw[1], (f * xc[:, 1] / zs + c[1]) * h / hw[0]], dim=-1)
        col = underwater_colours(phys, gg["J"], zc, m1, uv)
        img, _, alp = rv(gg, col, cu, f, hw[0], hw[1], torch.zeros(3, device=dev), c)
        if obj:
            return img, alp.reshape(hw[0], hw[1])
        img = img + (1.0 - alp.reshape(1, hw[0], hw[1])) * background(m1, hw[0], hw[1]).float()
        return (img, alp.reshape(hw[0], hw[1])) if alpha else img

    def refine(p0, gt, m1, t=None):
        if args.pose_iters <= 0:
            return p0
        dr = torch.zeros(3, device=dev, requires_grad=True)
        dt = torch.zeros(3, device=dev, requires_grad=True)
        opt = torch.optim.Adam([dr, dt], lr=args.pose_lr)

        def pose():
            d = torch.eye(4, device=dev)
            d[:3, :3] = E.axis_angle_to_matrix(dr)
            d[:3, 3] = dt / float(unit)
            return p0 @ d

        for _ in range(args.pose_iters):
            opt.zero_grad(set_to_none=True)
            F.l1_loss(render(pose(), m1, t=t), gt).backward()
            opt.step()
        return pose().detach()

    from torchmetrics.functional.image import structural_similarity_index_measure as tm_ssim
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lp = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(dev)

    def score(img, gt):
        img = img.clamp(0.0, 1.0)
        return [float(-10.0 * torch.log10(((img - gt) ** 2).mean() + 1e-12)), float(tm_ssim(img[None], gt[None], data_range=1.0)),
                float(lp(img[None], gt[None]))]

    out = {"scene": args.scene_dir, "head": src, "lora": args.lora, "physics": args.physics, "frames_in_map": n,
           "primitives_before_fusion": n_in, "primitives": g["n_out"], "seconds_map": t_map, "stage1_restored_frames": n_s1,
           "beta_D_scale": args.phys_beta_D_scale, "medium": modes, "medium_primary": modes[0], "medium_field": medium_info, "far_layer": int(g_far["n_out"]) if g_far is not None else 0,
           "medium_raster": args.medium_raster if "beta" in g else "preshade",
           "time_fill": bool(args.time_fill and tmaps),
           "time_bucket": ({"frames": int(args.time_bucket), "windows": len(tmaps), "primitives_all_windows": int(sum(m_["n_out"] for _, m_ in tmaps)),
                            "primitives_per_window": float(np.mean([m_["n_out"] for _, m_ in tmaps]))} if tmaps else None),
           "scaffold_mode": None if anchor is None or not getattr(anchor, "scaffold", False) else ("always on" if anchor.cfg.get("always_on") else anchor.mode),
           "beta_fixed": float(getattr(anchor, "cfg", {}).get("beta_fixed", 0.0) or 0.0) if anchor is not None else 0.0,
           "anchor_cfg": dict(anchor.cfg) if anchor is not None else None}
    refined = []
    with torch.no_grad():
        for mode in modes:
            own, own_r = [], []
            for i in range(0, n, max(1, n // 12)):
                fi, ci = float(intr[i, 0, 0]), (float(intr[i, 0, 2]) + 0.5, float(intr[i, 1, 2]) + 0.5)
                m1 = medium_for(mode, fnums[i], c2w[i], fi, ci)
                ti_ = fnums[i] / args.fps
                own.append(score(render(c2w[i], m1, fi, ci, t=ti_), imgs[i]))
                if mode == modes[0]:
                    own_r.append(score(render(c2w[i], m1, fi, ci, which="restored", t=ti_), J[i]))
            out["map_frames_underwater" + sfx[mode]] = np.mean(own, 0).tolist()
            if own_r:
                out["map_frames_restored_vs_stage1"] = np.mean(own_r, 0).tolist()
            print("[map frames%s] underwater %.2f / %.4f / %.4f" % (tag[mode], *out["map_frames_underwater" + sfx[mode]])
                  + (" | restored against the Stage 1 restored frame %.2f / %.4f / %.4f" % tuple(out["map_frames_restored_vs_stage1"])
                     if own_r else ""), flush=True)
        if "beta" in g and args.medium_raster == "s2":
            pair, other = [], []
            for i in range(0, n, max(1, n // 12)):
                fi, ci = float(intr[i, 0, 0]), (float(intr[i, 0, 2]) + 0.5, float(intr[i, 1, 2]) + 0.5)
                m1 = medium_for(modes[0], fnums[i], c2w[i], fi, ci)
                r_s2 = render(c2w[i], m1, fi, ci, t=fnums[i] / args.fps)
                args.medium_raster = "preshade"
                r_pre = render(c2w[i], m1, fi, ci, t=fnums[i] / args.fps)
                args.medium_raster = "s2"
                pair.append(float(-10.0 * torch.log10((r_s2.clamp(0, 1) - r_pre.clamp(0, 1)).square().mean() + 1e-12)))
                other.append(score(r_pre, imgs[i]))
            out["raster_check"] = {"psnr_s2_against_preshade": float(np.mean(pair)), "psnr_min": float(np.min(pair)),
                                   "map_frames_underwater_preshade": np.mean(other, 0).tolist()}
            print("[raster] s2 against preshade on %d map frames: %.2f dB (lowest %.2f) | preshade underwater %.2f / %.4f / %.4f"
                  % (len(pair), np.mean(pair), np.min(pair), *out["raster_check"]["map_frames_underwater_preshade"]), flush=True)

    if diag_idx:
        def psnr_m(img, gt_, m=None):
            e = (img.clamp(0.0, 1.0) - gt_).square()
            if m is not None:
                if int(m.sum()) < 50:
                    return float("nan")
                e = e[:, m]
            return float(-10.0 * torch.log10(e.mean() + 1e-12))

        drows = []
        with torch.no_grad():
            for i in diag_idx:
                fi, ci = float(intr[i, 0, 0]), (float(intr[i, 0, 2]) + 0.5, float(intr[i, 1, 2]) + 0.5)
                ti_ = fnums[i] / args.fps
                gt_i = imgs[i]
                m_own, m_nb = med_of([i]), med_others(i)
                img_f, a_f = render(c2w[i], m_own, fi, ci, t=ti_, alpha=True)
                img_n = render(c2w[i], m_nb, fi, ci, t=ti_)
                gs = single_map(i)
                water = background(m_own, h, w).float() if "B_inf_map" in m_own else m_own["B_inf"].float()[:, None, None].expand(3, h, w)
                if gs is not None and gs["n_out"] > 0:
                    img_s, a_s = render(c2w[i], m_own, fi, ci, t=ti_, gm=gs, alpha=True)
                else:
                    img_s, a_s = water, torch.zeros(h, w, device=dev)
                cf, cs = a_f > 0.5, a_s > 0.5
                ef = (img_f.clamp(0.0, 1.0) - gt_i).square().sum(0)
                r = dict(frame=fnums[i], cover_fused=float(cf.float().mean()), cover_single=float(cs.float().mean()), gauss_single=0 if gs is None else int(gs["n_out"]),
                         fused=psnr_m(img_f, gt_i), single=psnr_m(img_s, gt_i), fused_nearest_other_medium=psnr_m(img_n, gt_i),
                         water=psnr_m(water, gt_i), fused_covered=psnr_m(img_f, gt_i, cf), fused_uncovered=psnr_m(img_f, gt_i, ~cf),
                         fused_on_single_cover=psnr_m(img_f, gt_i, cs), single_on_single_cover=psnr_m(img_s, gt_i, cs),
                         uncovered_error_share=float(ef[~cf].sum() / ef.sum().clamp_min(1e-12)))
                drows.append(r)
                del gs
                print(f"[diag] frame {r['frame']}: cover fused {r['cover_fused']:.2f} single {r['cover_single']:.2f} | PSNR fused {r['fused']:.2f} "
                      f"(covered {r['fused_covered']:.2f}, uncovered {r['fused_uncovered']:.2f}, {100 * r['uncovered_error_share']:.0f}% of the "
                      f"squared error uncovered) | on the single view cover: fused {r['fused_on_single_cover']:.2f} single "
                      f"{r['single_on_single_cover']:.2f} | fused with the medium of the two nearest other frames "
                      f"{r['fused_nearest_other_medium']:.2f} | water alone {r['water']:.2f}", flush=True)
        dmean = {k: float(np.nanmean([r[k] for r in drows])) for k in drows[0] if k != "frame"}
        out["diag"] = dict(frames=drows, mean=dmean)
        print(f"[diag mean] {len(drows)} map frames, own medium: fused {dmean['fused']:.2f} dB, single view {dmean['single']:.2f} dB, "
              f"water alone {dmean['water']:.2f} dB | cover fused {dmean['cover_fused']:.2f} single {dmean['cover_single']:.2f} | fused "
              f"covered {dmean['fused_covered']:.2f} uncovered {dmean['fused_uncovered']:.2f}, {100 * dmean['uncovered_error_share']:.0f}% of the "
              f"squared error uncovered | on the single view cover: fused {dmean['fused_on_single_cover']:.2f} single "
              f"{dmean['single_on_single_cover']:.2f} | medium of the nearest other frames {dmean['fused_nearest_other_medium']:.2f}", flush=True)

    if args.data_dir and not args.bench:
        split = json.load(open(os.path.join(args.data_dir, "split.json")))
        centre = set(E.fnum(x) for x in split.get("centre_files", []))
        tests = sorted(split["test_files"], key=E.fnum)
        start = E.interpolate_poses(c2w.cpu().numpy().astype(np.float64), [E.fnum(x) for x in tests], fnums)
        from PIL import Image
        os.makedirs(os.path.join(args.out_dir, "restored"), exist_ok=True)
        out["eval_hw"] = [h, w]
        for mode in modes:
            rows = []
            for j, name in enumerate(tests):
                gt = E.load_image(os.path.join(args.scene_dir, "images", name), max(h, w)).to(dev)
                num = E.fnum(name)
                p0 = torch.tensor(start[j], dtype=torch.float32, device=dev)
                m0 = medium_for(mode, num, p0)
                with torch.no_grad():
                    before = score(render(p0, m0, t=num / args.fps), gt)
                p1 = refine(p0, gt, m0, t=num / args.fps)
                with torch.no_grad():
                    m1 = medium_for(mode, num, p1)
                    pred = render(p1, m1, t=num / args.fps)
                    after = score(pred, gt)
                    rest = render(p1, m1, which="restored", t=num / args.fps).clamp(0, 1)
                    if mode == modes[0]:
                        Image.fromarray((rest.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(
                            os.path.join(args.out_dir, "restored", os.path.splitext(name)[0] + ".jpg"), quality=95)
                    if j < args.save_images:
                        row = torch.cat([gt, pred.clamp(0, 1), rest], dim=2)
                        Image.fromarray((row.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(
                            os.path.join(args.out_dir, "%s_gt_underwater_restored%s.jpg" % (os.path.splitext(name)[0], sfx[mode])),
                            quality=92)
                if mode == modes[0] and len(refined) < args.gate_test:
                    refined.append((num, p1, m1, gt))
                rows.append({"frame": num, "centre": num in centre, "dist": min(abs(num - k) for k in fnums), "before": before, "after": after})
                print("%06d %s%s before %.2f / %.4f / %.4f after %.2f / %.4f / %.4f" % (
                    num, "centre" if num in centre else "side  ", tag[mode], *before, *after), flush=True)

            def agg(sel):
                r = [x for x in rows if sel(x)]
                return {"n": len(r), **{k: np.mean([x[k] for x in r], 0).tolist() for k in ("before", "after")}} if r else None

            s = sfx[mode]
            out.update({"all" + s: agg(lambda r: True), "centre" + s: agg(lambda r: r["centre"]),
                        "side" + s: agg(lambda r: not r["centre"]), "frames" + s: rows})
            for k in ("all", "centre", "side"):
                a = out[k + s]
                if a:
                    print("[%s%s] n %d | underwater before %.3f / %.4f / %.4f after %.3f / %.4f / %.4f" % (
                        k, tag[mode], a["n"], *a["before"], *a["after"]))
    if args.overlap > 0:
        import map_overlap as MO
        with torch.no_grad():
            ov = MO.overlap_stats(g["xyz"].cpu().numpy(), (g["scales"] / unit).cpu().numpy(), g["beta"].cpu().numpy() if "beta" in g else None)
            rows = {1: [], 2: [], 4: []}
            for i in sorted(set(np.linspace(0, n - 1, args.overlap).round().astype(int).tolist())):
                fi, ci = float(intr[i, 0, 0]), (float(intr[i, 0, 2]) + 0.5, float(intr[i, 1, 2]) + 0.5)
                m_own = med_of([i])
                solid = (conf[i] > args.min_conf_thr)[None].float()
                for zf in rows:
                    _, alp = render(c2w[i], m_own, fi * zf, (zf * (ci[0] - 0.5 * w * (1 - 1 / zf)), zf * (ci[1] - 0.5 * h * (1 - 1 / zf))),
                                    t=fnums[i] / args.fps, alpha=True)
                    x0, y0 = 0.5 * w * (1 - 1 / zf), 0.5 * h * (1 - 1 / zf)
                    ys = ((torch.arange(h, device=dev) + 0.5) / zf + y0).long().clamp(0, h - 1)
                    xs = ((torch.arange(w, device=dev) + 0.5) / zf + x0).long().clamp(0, w - 1)
                    m = solid[0][ys][:, xs] > 0.5
                    if int(m.sum()) >= 50:
                        rows[zf].append(float((alp[m] > 0.5).float().mean()))
            ov["cover"] = {f"x{zf}": float(np.mean(v)) if v else None for zf, v in rows.items()}
        out["overlap"] = ov
        print(f"[overlap] sigma over the nearest neighbour distance, percentiles 10 / 50 / 90: largest axis "
              f"{' / '.join('%.2f' % v for v in ov['major_over_nn'])}, middle axis {' / '.join('%.2f' % v for v in ov['middle_over_nn'])} | "
              f"confident pixels with alpha > 0.5 at 1x / 2x / 4x focal: {ov['cover']['x1']} / {ov['cover']['x2']} / {ov['cover']['x4']}", flush=True)
    if args.gate_test > 0 and anchor is not None and "gate" in g and "slot" in g:
        import gate_metric as GM
        with torch.no_grad():
            kind = GM.slot_kind(g["slot"], anchor.cells)
            cs = GM.cross_frame_scores(g["xyz"], g["raw"], imgs, depth, conf, intr, c2w, args.min_conf_thr)
            seen = cs["n_seen"] >= 3
            gt_ = dict(type_r2=GM.type_r2(g["gate"], kind), slots_seen_by_3_frames=float(seen.float().mean()),
                       see_through_mean=float(cs["see_through"][seen].mean()),
                       auc_see_through=GM.within_type_auc(g["gate"][seen], cs["see_through"][seen], kind[seen]),
                       auc_colour=GM.within_type_auc(g["gate"][seen], cs["resid"][seen], kind[seen]))
            gen = torch.Generator().manual_seed(0)
            gates = dict(map=g["gate"], shuffled=GM.shuffle_within_type(g["gate"].cpu(), kind.cpu(), gen).to(dev), zero=torch.zeros_like(g["gate"]))
            ps = {k: [] for k in gates}
            for num, p1, m1, gt in refined:
                for k, gv in gates.items():
                    ps[k].append(score(render(p1, m1, t=num / args.fps, gm=dict(g, gate=gv)), gt)[0])
            gt_.update(frames=[r[0] for r in refined], psnr={k: float(np.mean(v)) if v else None for k, v in ps.items()})
            if refined:
                gt_["psnr_map_minus_shuffled"] = gt_["psnr"]["map"] - gt_["psnr"]["shuffled"]
                gt_["psnr_map_minus_zero"] = gt_["psnr"]["map"] - gt_["psnr"]["zero"]
        out["gate_test"] = gt_
        print(f"[gate test] kind explains {gt_['type_r2']:.3f} of the gate variance | rank agreement inside a kind (0.5 = none): see "
              f"through {gt_['auc_see_through']['mean']:.3f}, colour change {gt_['auc_colour']['mean']:.3f} | held out PSNR on "
              f"{len(refined)} frames: map gate {gt_['psnr']['map']}, shuffled inside a kind {gt_['psnr']['shuffled']}, gate 0 {gt_['psnr']['zero']}",
              flush=True)
    u8 = lambda c: (c.clamp(0, 1) * 255).round().byte().cpu().numpy()
    extra = {k: g[k].detach().float().cpu().numpy() for k in ("beta", "gate", "tw", "tb", "vel", "t_src") if k in g}
    if "slot" in g:
        extra["slot"] = g["slot"].cpu().numpy().astype(np.int16)
    np.savez_compressed(os.path.join(args.out_dir, "map.npz"), xyz=g["xyz"].float().cpu().numpy(),
                        scales=(g["scales"] / unit).half().cpu().numpy(), quat=g["quat"].half().cpu().numpy(),
                        opacity=g["opacity"].reshape(-1).half().cpu().numpy(), color_raw=u8(g["raw"]), color_clean=u8(g["J"]),
                        kf_c2w=c2w.cpu().numpy(), frame_c2w=c2w.cpu().numpy(), focal=np.float32(focal), pp=np.float32(pp),
                        hw=np.int32([h, w]), kernel="beta" if "beta" in g else "gauss", restored="swimm3r_stage1", **extra)
    text = None
    if args.bench and args.data_dir:
        import bench_loc
        tests = sorted(json.load(open(os.path.join(args.data_dir, "split.json")))["test_files"], key=E.fnum)
        poses_np = c2w.cpu().numpy().astype(np.float64)
        res = bench_loc.run(dict(
            dev=dev, tests=tests, steps=args.bench_steps, inits=args.bench_init, fps=args.fps, unit=unit, lr=args.pose_lr,
            warmup=args.bench_warmup, render_reps=args.bench_render_reps, map_nums=fnums, map_poses=poses_np,
            interp=E.interpolate_poses(poses_np, [E.fnum(x) for x in tests], fnums),
            load=lambda name: E.load_image(os.path.join(args.scene_dir, "images", name), max(h, w)).to(dev),
            medium=lambda num, pose: medium_for(modes[0], num, pose), render=render, score=score))
        mb = lambda q: os.path.getsize(q) / 1e6 if q and os.path.isfile(q) else 0.0
        n_far = int(g_far["n_out"]) if g_far is not None else 0
        n_win = int(sum(m_["n_out"] for _, m_ in tmaps)) if tmaps else 0
        text = bench_loc.report(res, dict(
            scene=args.scene_dir, map_frames=n, hw=[h, w], primitives=int(g["n_out"]) + n_far + n_win, warmup=args.bench_warmup,
            kernel="Gaussian", dynamic=("gate" in g and not g.get("transient") and not g.get("exist")),
            map_mb=sum(g[k].numel() for k in ("xyz", "scales", "quat", "opacity", "J", "gate", "tw", "tb", "vel", "t_src") if k in g) * 4 / 1e6,
            map_file_mb=mb(os.path.join(args.out_dir, "map.npz")), mapping_seconds=args.bench_mapping_seconds,
            weights_mb={"LoRA": mb(args.lora), "physics head": mb(args.physics), "Gaussian head": mb(args.head)}))
        text += (f"\nrender path {args.medium_raster}, veil fit {'on' if args.binf_veil else 'off'}, time windows "
                 f"{args.time_bucket if tmaps else 0} ({n_win} primitives), far layer {n_far} primitives")
        open(os.path.join(args.out_dir, "benchmark.txt"), "w").write(text + "\n")
        out["bench"] = res
    if dev.type == "cuda":
        out["gpu_peak_gib"] = max(torch.cuda.max_memory_allocated() / 2 ** 30, out.get("bench", {}).get("memory", {}).get("gpu_peak_gib", 0.0))
        print(f"[memory] peak GPU memory of the map and the localization {out['gpu_peak_gib']:.2f} GiB", flush=True)
    out["seconds_total"] = time.time() - t0
    json.dump(out, open(os.path.join(args.out_dir, "scores.json"), "w"), indent=1)
    print(f"[done] {args.out_dir}/scores.json and map.npz in {out['seconds_total']:.0f}s", flush=True)
    if text:
        print(text, flush=True)


if __name__ == "__main__":
    main()
