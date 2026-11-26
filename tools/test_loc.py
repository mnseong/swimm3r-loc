import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import mast3r.utils.path_to_dust3r
import bench_gscpe as C
import bench_proto as P
import eval_gaussian_map as E
import loc_ff
import loc_geom as G
import loc_ref as R
import loc_warp as Wp

H, W_, FOCAL, DEPTH = 128, 192, 180.0, 4.0
BALL, RADIUS = (2.4, 0.0, 11.0), 8.0
PP = (W_ / 2.0, H / 2.0)
X0, RX, Y0, RY = -3.5, 12.0, -3.0, 6.0


def true_pose(i):
    j = np.random.default_rng(100 + i)
    p = np.eye(4)
    p[:3, :3] = G.rot(np.radians(np.array([1.5 * np.sin(0.3 * i), 2.0 * np.sin(0.2 * i + 1.0), 1.0 * np.cos(0.25 * i)]) + j.normal(scale=0.7, size=3)))
    p[:3, 3] = np.array([0.12 * i, 0.15 * np.sin(0.15 * i), 0.1 * np.cos(0.1 * i)]) + j.normal(scale=0.04, size=3)
    return p


def surface_points(pose):
    v, u = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W_, dtype=torch.float32), indexing="ij")
    d = torch.stack([(u - (PP[0] - 0.5)) / FOCAL, (v - (PP[1] - 0.5)) / FOCAL, torch.ones_like(u)], -1) @ pose[:3, :3].T
    oc = pose[:3, 3] - torch.tensor(BALL, dtype=pose.dtype)
    a, b, c = (d * d).sum(-1), 2.0 * (d * oc).sum(-1), (oc * oc).sum() - RADIUS ** 2
    s = (-b - torch.sqrt((b * b - 4.0 * a * c).clamp_min(1e-9))) / (2.0 * a)
    return pose[:3, 3] + s[..., None] * d, s


def render(pose, med=None, t=None, alpha=False, which="underwater", depth=False):
    p, z = surface_points(pose)
    img = torch.stack([(p[..., 0] - X0) / RX, (p[..., 1] - Y0) / RY, 0.5 + 0.25 * torch.sin(3.0 * p[..., 0]) * torch.cos(2.0 * p[..., 1])])
    if which == "restored":
        img = img.clamp(0.0, 1.0) ** 0.6
    if depth:
        return img, torch.ones(H, W_), z
    return (img, torch.ones(H, W_)) if alpha else img


class StandIn:
    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.freq = torch.rand(16, 2, generator=g) * 2.0 - 1.0
        self.freq = self.freq * torch.linspace(2.0, 120.0, 16)[:, None]
        self.bank = []

    def _encode_image(self, x, shape):
        img = (x + 1.0) / 2.0
        ph = 2.0 * np.pi * torch.einsum("bchw,kc->bhwk", img[:, :2], self.freq)
        desc = F.normalize(torch.cat([torch.sin(ph), torch.cos(ph)], -1), dim=-1)
        tok = F.avg_pool2d(desc.permute(0, 3, 1, 2), 16).flatten(2).transpose(1, 2).contiguous()
        for b in range(tok.shape[0]):
            self.bank.append((tok[b].clone(), desc[b].clone()))
        return tok, torch.zeros(tok.shape[0], tok.shape[1], 2, dtype=torch.long), None

    def _decoder(self, f1, pos1, f2, pos2):
        return [f1], [f2]

    def _downstream_head(self, num, dec, shape):
        out = []
        for t in dec[-1]:
            out.append(min(self.bank, key=lambda e: float((e[0] - t).abs().sum()))[1])
        desc = torch.stack(out)
        return dict(desc=desc, desc_conf=torch.ones(desc.shape[:3]))


def check_geometry():
    rng = np.random.default_rng(0)
    K, hw = G.intrinsics(270.0, (256.0, 144.0)), (288, 512)
    true = np.eye(4)
    true[:3, :3] = G.rot(np.radians([3.0, -4.0, 2.0]))
    true[:3, 3] = [0.3, -0.1, 0.2]
    pts = np.c_[rng.uniform(-4, 4, 3000), rng.uniform(-2.5, 2.5, 3000), rng.uniform(3, 7, 3000)]
    uv, z = G.project(K, true, pts)
    ok = (z > 0) & G.inside(uv, hw)
    p3d, p2d = pts[ok], uv[ok] + rng.normal(scale=0.5, size=(int(ok.sum()), 2))
    bad = rng.random(len(p3d)) < 0.3
    p2d[bad] = np.c_[rng.uniform(0, 512, int(bad.sum())), rng.uniform(0, 288, int(bad.sum()))]
    r = G.solve_pnp(p3d, p2d, K, 4.7, G.perturb(true, 0.5, 6.0, rng))
    e = G.pose_error(r["c2w"], true, 5.0)
    print("PnP with 30 %% wrong matches: %.3f %% of depth, %.3f deg, %.3f px, solver %s" % (e[0], e[1], G.pixel_shift(K, true, r["c2w"], pts, hw), r["solver"]))
    assert e[0] < 0.2 and e[1] < 0.1, e
    yy, xx = np.mgrid[0:288, 0:512].astype(np.float64)
    tex = lambda x, y: 0.5 + 0.25 * np.sin(x / 7.0) * np.cos(y / 5.0) + 0.2 * np.sin((x + 2 * y) / 13.0) + 0.05 * np.sin(x / 2.3 + y / 3.1)
    f = G.flow_residual(np.stack([tex(xx, yy)] * 3, -1), np.stack([tex(xx - 3.0, yy - 2.0)] * 3, -1))
    print("optical flow of a known shift of 3.61 px: %s" % ("%.2f px" % f if np.isfinite(f) else "not available (no OpenCV optical flow)"))
    assert not np.isfinite(f) or abs(f - 3.61) < 0.3, f


def check_pose_files():
    import json
    import struct
    rng = np.random.default_rng(5)
    poses = []
    for i in range(4):
        q = np.eye(4)
        q[:3, :3] = G.rot(rng.normal(scale=0.6, size=3))
        q[:3, 3] = rng.normal(size=3)
        poses.append(q)
    names = ["%06d.jpg" % (3 * i) for i in range(4)]
    d = tempfile.mkdtemp()

    def qvec(r):
        v = G.rotvec(r)
        th = float(np.linalg.norm(v))
        return np.r_[np.cos(0.5 * th), np.sin(0.5 * th) * v / max(th, 1e-12)]

    w2c = [np.linalg.inv(q) for q in poses]
    with open(os.path.join(d, "images.txt"), "w") as f:
        f.write("# Image list\n")
        for i, (n, m) in enumerate(zip(names, w2c)):
            f.write("%d %s %s 1 frames/%s\n" % (i + 1, " ".join("%.17g" % x for x in qvec(m[:3, :3])), " ".join("%.17g" % x for x in m[:3, 3]), n))
            f.write("10.5 20.25 -1 30.0 40.0 7 1.0 2.0 3 4.0 5.0 6\n" if i % 2 else "\n")
    os.makedirs(os.path.join(d, "sparse", "0"))
    with open(os.path.join(d, "sparse", "0", "images.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(names)))
        for i, (n, m) in enumerate(zip(names, w2c)):
            f.write(struct.pack("<I7dI", i + 1, *qvec(m[:3, :3]), *m[:3, 3], 1))
            f.write(n.encode() + b"\x00")
            f.write(struct.pack("<Q", 2))
            f.write(struct.pack("<ddqddq", 1.0, 2.0, -1, 3.0, 4.0, 9))
    json.dump({n: q.tolist() for n, q in zip(names, poses)}, open(os.path.join(d, "poses.json"), "w"))
    worst = 0.0
    for src in (os.path.join(d, "images.txt"), d, os.path.join(d, "poses.json")):
        got_n, got_p = G.read_poses(src)
        assert got_n == names, (src, got_n)
        worst = max(worst, float(np.abs(got_p - np.asarray(poses)).max()))
    print("poses read back from a COLMAP text model, a binary model and a json: largest difference %.1e" % worst)
    assert worst < 1e-9, worst


def check_warp():
    tt = lambda p: torch.tensor(np.asarray(p), dtype=torch.float32)
    K, hw = G.intrinsics(FOCAL, PP), (H, W_)
    true = true_pose(12)
    with torch.no_grad():
        query = render(tt(true))
        pts = surface_points(tt(true))[0].reshape(-1, 3)[::37].double().numpy()
    rgbd = lambda p: render(tt(p), depth=True)
    zed = lambda p: rgbd(p)[2]
    start = G.perturb(true, 0.004 * DEPTH, 0.15, np.random.default_rng(9))
    img, alpha, depth = rgbd(start)
    mask, ok = Wp.visibility_mask(start, depth, alpha, K, zed)
    off, _ = Wp.visibility_mask(start, depth, alpha, K, lambda p: 1.05 * zed(p))
    print("visibility mask: %.3f of the pixels kept with consistent depth renders, %.3f when the nearby renders are 5 %% deeper" % (
        float(mask.float().mean()), float(off.float().mean())))
    assert float(mask.float().mean()) > 0.9 and float(off.float().mean()) < 0.05 and bool(ok.all())
    e0 = G.pixel_shift(K, true, start, pts, hw)
    out = {}
    for label, kw in (("3 pyramid levels with the mask", {}), ("full resolution only", dict(levels=0, steps=60)), ("no mask", dict(use_mask=False))):
        r = Wp.refine(query, start, K, rgbd, zed, **kw)
        out[label] = G.pixel_shift(K, true, r["pose"], pts, hw)
        print("warping refinement, %s: %.3f px -> %.3f px in %d steps, %d renders, %d mask renders, %d new anchors" % (
            label, e0, out[label], r["steps"], r["renders"], r["mask_renders"], r["rerenders"]))
        assert r["valid"] and r["steps"] == 60 and out[label] < 0.4 * e0, (label, e0, out[label])
    dim = Wp.refine((0.8 * query + 0.05).clamp(0.0, 1.0), start, K, rgbd, zed, gain=True)
    plain = Wp.refine((0.8 * query + 0.05).clamp(0.0, 1.0), start, K, rgbd, zed)
    e_dim, e_plain = G.pixel_shift(K, true, dim["pose"], pts, hw), G.pixel_shift(K, true, plain["pose"], pts, hw)
    print("warping refinement of a query at 0.8 of the brightness plus 0.05: %.3f px with the brightness fit, %.3f px without (start %.3f px)" % (e_dim, e_plain, e0))
    assert e_dim < e_plain and e_dim < e0, (e_dim, e_plain, e0)
    far = Wp.refine(query, G.perturb(true, 0.012 * DEPTH, 0.6, np.random.default_rng(4)), K, rgbd, zed)
    print("warping refinement from 1.2 %% of depth and 0.6 deg: %d new anchors" % far["rerenders"])
    assert far["rerenders"] >= 1 and far["renders"] == far["rerenders"] + 1 and far["mask_renders"] == 4 * far["renders"]
    lv = Wp.level_intrinsics(K, 2)
    assert abs(lv[0] - FOCAL / 4.0) < 1e-9 and abs(lv[2] - (K[0, 2] - 1.5) / 4.0) < 1e-9


def check_gscpe():
    dev = torch.device("cpu")
    tt = lambda p: torch.tensor(np.asarray(p), dtype=torch.float32)
    nums = list(range(2, 41))
    hold = [i for i in nums if i % 4 == 1 and nums[0] < i < nums[-1]]
    mapf = [i for i in nums if i not in hold]
    name = lambda i: "%06d.jpg" % i
    truth = {i: true_pose(i) for i in nums}
    with torch.no_grad():
        imgs = torch.stack([render(tt(truth[i])) for i in mapf])
        depth = torch.stack([surface_points(tt(truth[i]))[1] for i in mapf])
    c2w = torch.stack([tt(truth[i]) for i in mapf])
    intr = torch.tensor([[FOCAL, 0.0, PP[0] - 0.5], [0.0, FOCAL, PP[1] - 0.5], [0.0, 0.0, 1.0]]).expand(len(mapf), 3, 3).contiguous()
    net = StandIn()
    loc = loc_ff.Localizer(net, loc_ff.encode_frames(net, imgs), depth, torch.full_like(depth, 2.0), intr, c2w, FOCAL, PP, 1.0, dev, subsample=4, reproj=2.5)
    gen = torch.Generator().manual_seed(2)

    def load(nm):
        with torch.no_grad():
            return (render(tt(truth[E.fnum(nm)])) + 0.004 * torch.randn(3, H, W_, generator=gen)).clamp(0.0, 1.0)

    def score(img, gt):
        mae = float((img.clamp(0, 1) - gt).abs().mean())
        return [float(-10.0 * torch.log10(((img.clamp(0, 1) - gt) ** 2).mean() + 1e-12)), 1.0 - mae, mae]

    seen = {"underwater": 0, "restored": 0}

    def strict(pose, med=None, t=None, alpha=False, which="underwater", depth=False):
        assert which in seen and (which == "restored" or med is not None) and t is not None, (which, med, t)
        seen[which] += 1
        return render(pose, med, t, alpha, which, depth)

    with torch.no_grad():
        fq = loc.encode(render(tt(truth[hold[0]])))
    near = loc.near(truth[hold[0]], 1)[0]
    ir, al, dp = render(tt(loc.poses[near]), depth=True)
    p2d, p3d, found, st = loc.match_render(fq, ir, dp, al, loc.poses[near])
    q2d, q3d = loc.match(fq, [near])[:2]
    uv, z = G.project(loc.K, truth[hold[0]], p3d)
    err = np.linalg.norm(uv - p2d, axis=1)
    print("matching a query to a render at the pose of a map frame: %d matches lifted by the rendered depth, %.0f %% within 2 px of the true projection, "
          "largest difference to the matches of the map frame itself %.1e" % (len(p2d), 100.0 * np.mean(err < 2.0), float(np.abs(p3d - q3d).max())))
    assert found >= len(p2d) > 100 and float((torch.as_tensor(p3d) - torch.tensor(BALL, dtype=torch.float64)).norm(dim=-1).sub(RADIUS).abs().max()) < 1e-3
    assert np.mean(err < 2.0) > 0.9 and set(st) == {"encoder", "decoder", "matching"}, np.mean(err < 2.0)
    assert p2d.shape == q2d.shape and float(np.abs(p2d - q2d).max()) == 0.0 and float(np.abs(p3d - q3d).max()) < 1e-4

    out_dir = tempfile.mkdtemp()
    fit = dict(n=len(mapf), scale=1.0, t_med=0.0, t_max=0.0, r_med=0.0, r_max=0.0)
    ctx = dict(dev=dev, tests=[name(i) for i in hold] + ["999999.jpg"], fps=5.0, warmup=1, map_nums=mapf, load=load, medium=lambda num, pose: {},
               render=strict, score=score, loc=loc, loc_public=loc, out_dir=out_dir, ref_poses={name(i): truth[i] for i in hold}, ref_fit=fit,
               ref_source="truth", depth_m=2.0, fast=False)
    res = C.run(ctx)
    info = dict(scene="synthetic scene", map_frames=len(mapf), hw=[H, W_], primitives=0, map_mb=0.0, model="stand in", mapping_seconds=None,
                weights_mb={"none": 0.0}, reference="true poses of the synthetic scene", split="test : train = 1 : %.1f" % (len(mapf) / len(hold)))
    text = C.report(res, info)
    print(text)
    row = lambda label: next(r for r in res["rows"] if r["label"] == label)
    a0, a1, a2 = row("pose of the retrieved map frame"), row("coarse: render, match, PnP"), row("refine: + masked warping")
    assert res["frames"] == len(hold) and res["skipped"] == 1 and "%%" not in text and os.path.isfile(os.path.join(out_dir, "gscpe.json"))
    assert a1["fails"] == 0 and a1["acc"][0] == 100.0 and a1["acc"][1] == 100.0 and a1["t"] < 0.2 * a0["t"] and a1["px"] < 1.0, (a0, a1)
    assert a2["px"] <= a1["px"] + 0.3 and a2["acc"][1] == 100.0 and a2["s_mean"] > a1["s_mean"] > a0["s_mean"], (a1, a2)
    assert abs(a1["cm"] - 2.0 * a1["t"]) < 1e-9 and len(res["per_query"]["coarse: render, match, PnP"]) == len(hold)
    labels = [r["label"] for r in res["rows"]]
    for want in ("no render: match the map frame, PnP", "refine without the pyramid", "refine without the mask", "refine with a brightness fit", "no water model: coarse",
                 "no water model: refine", "public MASt3R, no render: match, PnP", "public MASt3R matcher: coarse", "public MASt3R matcher: refine"):
        assert want in labels, (want, labels)
    assert row("no render: match the map frame, PnP")["acc"][1] == 100.0 and res["stages"]["iters"] <= 3.0 and res["retrieval"]["top1"] >= 80.0
    assert "the same in cm" in text and "Runtime" in text and not [x for x in res["notes"] if G.cv2 is not None or "OpenCV is missing" not in x], res["notes"]
    assert (G.cv2 is None) == any("OpenCV is missing" in x for x in res["notes"]) and res["solver"] != "none", (res["solver"], res["notes"])
    dry = row("no water model: coarse")
    assert seen["restored"] > 4 * len(hold) and (dry["t"], dry["r"]) != (a1["t"], a1["r"]), (seen, dry, a1)
    assert row("public MASt3R matcher: coarse")["acc"][1] == 100.0 and "stages_public" in res
    same = C.report(dict(res, fit=dict(same=True, n=len(mapf), focal_map=FOCAL, focal_queries=FOCAL)), dict(info, mapping_seconds=10.0, mapping_frames=len(mapf) + len(hold)))
    assert "same frame and intrinsics as the map" in same and "moved into the map" not in same and "come from this same reconstruction" in same
    short = C.run(dict(ctx, out_dir=tempfile.mkdtemp(), ablation=False, loc_public=None, depth_m=None, coarse_iters=1, score=None))
    short_text = C.report(short, info)
    assert [r["label"] for r in short["rows"]] == labels[:4] and "the same in cm" not in short_text and short["stages"]["iters"] == 1.0
    assert "lpips" not in short["rows"][1] and "%%" not in short_text


def check_register():
    dev = torch.device("cpu")
    tt = lambda p: torch.tensor(np.asarray(p), dtype=torch.float32)
    nums = list(range(2, 41))
    hold = [n for k, n in enumerate(nums) if 4 <= k % 12 <= 6]
    mapf = [n for n in nums if n not in hold]
    name = lambda i: "%06d.jpg" % i
    truth = {i: true_pose(i) for i in nums}
    with torch.no_grad():
        imgs = torch.stack([render(tt(truth[i])) for i in mapf])
        depth = torch.stack([surface_points(tt(truth[i]))[1] for i in mapf])
    c2w = torch.stack([tt(truth[i]) for i in mapf])
    intr = torch.tensor([[FOCAL, 0.0, PP[0] - 0.5], [0.0, FOCAL, PP[1] - 0.5], [0.0, 0.0, 1.0]]).expand(len(mapf), 3, 3).contiguous()
    net = StandIn()
    loc = loc_ff.Localizer(net, loc_ff.encode_frames(net, imgs), depth, torch.full_like(depth, 2.0), intr, c2w, FOCAL, PP, 1.0, dev, subsample=4, reproj=2.5)
    before = (loc.poses.copy(), loc.xyz.clone(), loc.K.copy())
    gen = torch.Generator().manual_seed(6)

    def load(nm):
        with torch.no_grad():
            return (render(tt(truth[E.fnum(nm)])) + 0.004 * torch.randn(3, H, W_, generator=gen)).clamp(0.0, 1.0)

    tests = [name(i) for i in hold]
    poses_np = c2w.numpy().astype(np.float64)
    interp = E.interpolate_poses(poses_np, hold, mapf)
    poses, fit, rows = R.register_queries(loc, tests, load, mapf, interp, k=6, tol=1.5, verbose=False)
    assert loc.subsample == 4 and float(np.abs(loc.poses - before[0]).max()) == 0.0 and bool((loc.xyz == before[1]).all()) and float(np.abs(loc.K - before[2]).max()) == 0.0
    assert all(len(r["frames"]) == 6 and min(r["frames"]) < r["frame"] < max(r["frames"]) and r["frame"] not in r["frames"] for r in rows), rows[0]
    pts = surface_points(tt(truth[hold[4]]))[0].reshape(-1, 3)[::37].double().numpy()
    err = np.array([list(G.pose_error(poses[n], truth[E.fnum(n)], DEPTH)) + [G.pixel_shift(loc.K, truth[E.fnum(n)], poses[n], pts, (H, W_))] for n in poses])
    start = np.array([G.pixel_shift(loc.K, truth[h], interp[k], pts, (H, W_)) for k, h in enumerate(hold)])
    print("queries registered into the fixed map: %d of %d with a reference, against the true poses a median of %.3f %% of depth, %.3f deg, %.3f px "
          "(the interpolated start is %.1f px off); earlier and later map frames alone differ by a median of %.2f px" % (
              fit["n"], fit["total"], np.median(err[:, 0]), np.median(err[:, 1]), np.nanmedian(err[:, 2]), np.nanmedian(start), fit["half_px_med"]))
    assert fit["register"] and fit["n"] >= 7 and fit["total"] == 9 and fit["grid"] == 2, fit
    assert np.median(err[:, 0]) < 0.5 and np.median(err[:, 1]) < 0.3 and np.nanmedian(err[:, 2]) < 0.6 and np.nanmax(err[:, 2]) < 2.0, err
    lines = R.reference_lines(fit, skipped=fit["total"] - fit["n"])
    assert "registered into the fixed map" in lines[0] and "%%" not in "\n".join(lines)
    far, fit_far, _ = R.register_queries(loc, tests[:2], load, mapf, interp[:2], k=6, tol=1e-6, verbose=False)
    assert not far and fit_far["n"] == 0
    out_dir = tempfile.mkdtemp()

    def strict(pose, med=None, t=None, alpha=False, which="underwater", depth=False):
        assert which == "restored" or med is not None
        return render(pose, med, t, alpha, which, depth)

    def score(img, gt):
        mae = float((img.clamp(0, 1) - gt).abs().mean())
        return [float(-10.0 * torch.log10(((img.clamp(0, 1) - gt) ** 2).mean() + 1e-12)), 1.0 - mae, mae]

    res = C.run(dict(dev=dev, tests=tests, fps=5.0, warmup=1, map_nums=mapf, load=load, medium=lambda num, pose: {}, render=strict, score=score, loc=loc,
                     loc_public=None, out_dir=out_dir, ref_poses=poses, ref_fit=fit, ref_source="registration", ablation=False, fast=False))
    text = C.report(res, dict(scene="synthetic scene", map_frames=len(mapf), hw=[H, W_], primitives=0, map_mb=0.0, model="stand in", mapping_seconds=None,
                              weights_mb={"none": 0.0}, reference="every query registered into the fixed map", split="test : train = 1 : 3.3"))
    print(text[text.index(" Reference"):text.index(" Pipeline")])
    assert res["frames"] == fit["n"] and res["skipped"] == 9 - fit["n"] and "registered into the fixed map" in text and "Check       render of the map" in text
    assert res["reference_check"]["n"] == fit["n"] and res["reference_check"]["lpips"] < 0.01 and "%%" not in text, res["reference_check"]
    row = next(r for r in res["rows"] if r["label"] == "coarse: render, match, PnP")
    assert row["acc"][1] == 100.0 and row["px"] < 1.0, row


def main():
    torch.manual_seed(0)
    check_geometry()
    check_pose_files()
    check_warp()
    check_register()
    check_gscpe()
    dev = torch.device("cpu")
    nums = list(range(2, 41))
    map_nums = [i for i in nums if i % 8 in (2, 3, 4, 5, 6)]
    test_nums = [i for i in nums if i % 8 in (7, 0, 1) and 7 <= i <= 33]
    centre = [i for i in test_nums if i % 8 == 0]
    name = lambda i: "%06d.jpg" % i
    truth = {i: true_pose(i) for i in nums}
    tt = lambda p: torch.tensor(p, dtype=torch.float32)
    with torch.no_grad():
        imgs = torch.stack([render(tt(truth[i])) for i in map_nums])
        depth = torch.stack([surface_points(tt(truth[i]))[1] for i in map_nums])
    c2w = torch.stack([tt(truth[i]) for i in map_nums])
    intr = torch.tensor([[FOCAL, 0.0, PP[0] - 0.5], [0.0, FOCAL, PP[1] - 0.5], [0.0, 0.0, 1.0]]).expand(len(map_nums), 3, 3).contiguous()
    conf = torch.full_like(depth, 2.0)
    net = StandIn()
    loc = loc_ff.Localizer(net, loc_ff.encode_frames(net, imgs), depth, conf, intr, c2w, FOCAL, PP, 1.0, dev, subsample=4, reproj=2.5)
    ball = (loc.xyz - torch.tensor(BALL)).norm(dim=-1)
    assert tuple(loc.xyz.shape) == (len(map_nums), H, W_, 3) and float((ball - RADIUS).abs().max()) < 1e-3
    poses_np = c2w.numpy().astype(np.float64)
    tests = [name(i) for i in test_nums]
    gen = torch.Generator().manual_seed(1)

    def load(nm):
        with torch.no_grad():
            return (render(tt(truth[E.fnum(nm)])) + 0.004 * torch.randn(3, H, W_, generator=gen)).clamp(0.0, 1.0)

    def score(img, gt):
        mae = float((img.clamp(0, 1) - gt).abs().mean())
        return [float(-10.0 * torch.log10(((img.clamp(0, 1) - gt) ** 2).mean() + 1e-12)), 1.0 - mae, mae]

    one = loc.localize(load(tests[0]), prior=truth[test_nums[0] - 1], topk=1)
    e1 = G.pose_error(one["pose"], truth[test_nums[0]], DEPTH)
    print("one frame: %s, %d inliers of %d matches, %.3f %% of depth and %.3f deg from the true pose, solver %s" % (
        "solved" if one["ok"] else "failed", one["inliers"], one["matches"], e1[0], e1[1], one["solver"]))
    assert one["ok"] and e1[0] < 1.0 and e1[1] < 0.5, e1
    two = loc.localize(load(tests[0]), topk=3)
    e2 = G.pose_error(two["pose"], truth[test_nums[0]], DEPTH)
    print("no start pose: map frames %s chosen for frame %d, %.3f %% and %.3f deg" % ([map_nums[i] for i in two["idx"]], test_nums[0], e2[0], e2[1]))
    assert two["ok"] and e2[0] < 1.0 and e2[1] < 0.5, e2

    out_dir = tempfile.mkdtemp()
    ctx = dict(dev=dev, tests=tests, centre=[name(i) for i in centre], fps=5.0, unit=1.0, lr=4e-3, warmup=1, map_nums=map_nums, map_poses=poses_np,
               interp=E.interpolate_poses(poses_np, test_nums, map_nums), load=load, medium=lambda num, pose: {}, render=render, score=score,
               loc=loc, loc_public=loc, out_dir=out_dir, rc_steps=[5, 20, 60], ff_steps=[5, 10], topk=[1, 3], ref_steps=40, ref_check=120,
               ref_frames=4, ref_tol=1.0, px_tol=1.0, conv_ff_steps=5, conv_rc_steps=40, reloc_rc_max=20, levels=((2.0, 1.0), (10.0, 5.0)),
               map_names=[name(i) for i in map_nums], strides=[2, 4], gaps=[8])
    res = P.run(ctx)
    text = P.report(res, dict(scene="synthetic scene", map_frames=len(map_nums), hw=[H, W_], primitives=0, map_mb=0.0, model="stand in",
                              public=True, mapping_seconds=None, weights_mb={"none": 0.0}, warmup=1))
    print(text)
    ref = {int(k): v for k, v in __import__("json").load(open(os.path.join(out_dir, "reference.json")))["frames"].items()}
    order = [int(j) for j in np.argsort(test_nums)]
    err = np.array([G.pose_error(np.asarray(ref[j]["pose"]), truth[test_nums[j]], DEPTH) for j in order])
    print("reference against the true poses of the synthetic scene: median %.3f %% of depth, %.3f deg, largest %.3f %%, %.3f deg" % (
        np.median(err[:, 0]), np.median(err[:, 1]), err[:, 0].max(), err[:, 1].max()))
    assert res["reference"]["verified"] >= 0.8 * len(tests), res["reference"]
    assert res["reference"]["check_median"][2] < 0.6 and res["reference"]["pnp_median"][2] < 0.5, res["reference"]
    assert np.median(err[:, 0]) < 1.0 and np.median(err[:, 1]) < 0.5 and "%%" not in text, err
    row = lambda task, label: next(r for r in res[task] if r["label"] == label)
    row_of = lambda rs, task, label: next(r for r in rs[task] if r["label"] == label)
    a, a0 = row("tracking", "matching + PnP"), row("tracking", "start pose, no computation")
    assert a["s5"] == 100.0 and a["sp"] == 100.0 and a["fails"] == 0 and a["px"] < 0.5 and a["px"] < 0.2 * a0["px"], (a, a0)
    b = row("reloc", "matching + PnP, best 3")
    assert b["s5"] == 100.0 and b["sp"] == 100.0 and b["px"] < 0.5, b
    assert res["retrieval"]["top3"] >= 80.0, res["retrieval"]
    lv = res["convergence"]["levels"]
    assert all(x["ff"]["sp"] == 100.0 for x in lv) and lv[1]["start"]["px"] > 3.0 * lv[0]["start"]["px"] and lv[1]["start"]["sp"] == 0.0, lv
    assert res["exact"]["px"] < 0.5 and res["exact"]["sp"] == 100.0 and res["exact"]["ok"] == 100.0, res["exact"]
    assert row("tracking", "render and compare 60")["lpips"] <= row("tracking", "start pose, no computation")["lpips"] + 1e-6
    again = P.run(dict(ctx, only_reference=True))
    assert again["reference"]["seconds"] == 0.0 and again["reference"]["verified"] == res["reference"]["verified"]
    th = res["sparse"]
    assert [x["stride"] for x in th["strides"]] == [2, 4] and th["strides"][0]["all"]["sp"] >= 90.0 and th["strides"][1]["all"]["n"] > th["strides"][0]["all"]["n"], th["strides"]
    assert th["gaps"] and th["gaps"][0]["all"]["n"] >= 4 and "Thinned map" in text, th["gaps"]
    hard = P.run(dict(ctx, only_thinned=True, strides=[12], gaps=[24], accept_inliers=2000))
    assert not hard["tracking"] and hard["sparse"]["strides"][0]["all"]["again"] == 100.0 and hard["sparse"]["strides"][0]["all"]["accepted_good"] == 0.0, hard["sparse"]
    hard_text = P.report(hard, dict(scene="synthetic scene", map_frames=len(map_nums), hw=[H, W_], primitives=0, map_mb=0.0, model="stand in",
                                    public=False, mapping_seconds=None, weights_mb={"none": 0.0}, warmup=1))
    assert "Thinned map" in hard_text and "Tracking " not in hard_text
    print(hard_text[hard_text.index(" Thinned map"):hard_text.index(" Footprint")])
    rng = np.random.default_rng(3)
    sim = (1.7, G.rot(np.radians([20.0, -35.0, 50.0])), np.array([3.0, -2.0, 5.0]))
    line = [np.eye(4) for _ in range(6)]
    for i, q in enumerate(line):
        q[:3, :3] = G.rot(np.radians(rng.normal(scale=3.0, size=3)))
        q[:3, 3] = [0.3 * i, 0.0, 0.0]
    moved = [G.move_pose(*sim, q) for q in line]
    got, fit = G.transfer_poses(["f%d" % i for i in range(6)], np.array(line), ["f0", "f2", "f3", "f5"], np.array([moved[0], moved[2], moved[3], moved[5]]), ["f1", "f4"])
    back = max(float(np.abs(got["f1"] - moved[1]).max()), float(np.abs(got["f4"] - moved[4]).max()))
    print("poses moved between two reconstructions with camera centres on a line: scale %.4f, largest difference %.1e" % (fit["scale"], back))
    assert abs(fit["scale"] - 1.7) < 1e-9 and back < 1e-9 and max(fit["et"]) < 1e-9 and max(fit["er"]) < 1e-5, (fit, back)
    given = {name(i): truth[i] for i in test_nums}
    out2 = tempfile.mkdtemp()
    sp = P.run(dict(ctx, out_dir=out2, tasks=["reloc", "conv"], ref_poses=given,
                    ref_fit=dict(n=len(map_nums), scale=1.0, t_med=0.0, t_max=0.0, r_med=0.0, r_max=0.0)))
    sp_text = P.report(sp, dict(scene="synthetic scene", map_frames=len(map_nums), hw=[H, W_], primitives=0, map_mb=0.0, model="stand in",
                                public=False, mapping_seconds=None, weights_mb={"none": 0.0}, warmup=1))
    print(sp_text[sp_text.index(" Reference"):sp_text.index(" Footprint")])
    assert sp["reference"]["given"] and sp["reference"]["verified"] == len(tests) and not sp["tracking"] and "exact" not in sp and "sparse" not in sp, sp["reference"]
    assert row_of(sp, "reloc", "matching + PnP, best 1")["sp"] >= 90.0 and sp["difficulty"]["n"] == len(tests) and "offline mapping of all frames" in sp_text
    print("test_loc ok")


if __name__ == "__main__":
    main()
