import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import mast3r.utils.path_to_dust3r  # noqa: F401
import eval_gaussian_map as E
import loc_fast as LF
import loc_geom as G
import loc_warp as Wp
import loc_water as LW
import test_fast as TF
import test_loc as T

H, W_, FOCAL, DEPTH, PP = T.H, T.W_, T.FOCAL, T.DEPTH, T.PP
tt = lambda p: torch.tensor(np.asarray(p), dtype=torch.float32)
BINF = torch.tensor([0.04, 0.42, 0.48])
BETA_D = torch.tensor([0.45, 0.18, 0.15])
BETA_B = torch.tensor([0.30, 0.15, 0.12])
GAIN = torch.tensor([0.92, 1.0, 1.05])


def scene_at(pose, amp=1.0):
    J, alpha, z = TF.textured(pose)
    if amp != 1.0:
        J = (0.45 + amp * (J - 0.45)).clamp(0.0, 1.0)
    return J, alpha, z


def underwater(J, z):
    return J * torch.exp(-BETA_D[:, None, None] * z) + BINF[:, None, None] * (1.0 - torch.exp(-BETA_B[:, None, None] * z))


def betas(zz):
    n = zz.shape[0]
    return BETA_D[:, None].expand(3, n).to(zz.device), BETA_B[:, None].expand(3, n).to(zz.device)


def parts_at(pose, amp=1.0, half=False):
    J, alpha, z = scene_at(pose, amp)
    B = BINF[:, None, None].expand(3, H, W_).clone()
    I = underwater(J, z)
    if not half:
        return dict(I=I, J=J, alpha=alpha, depth=z, binf=B, betas=betas, level=0)
    d, ok = Wp.pool_depth(z, alpha > 0.5, 1)
    return dict(I=Wp.pool_image(I, 1), J=Wp.pool_image(J, 1), alpha=ok.float(), depth=torch.where(ok, d, torch.zeros_like(d)), binf=Wp.pool_image(B, 1),
                betas=betas, level=1)


def query_at(pose, gen, amp=1.0, noise=0.006):
    J, _, z = scene_at(pose, amp)
    return (GAIN[:, None, None] * underwater(J, z) + noise * torch.randn(3, H, W_, generator=gen)).clamp(0.0, 1.0)


def check_steps_match():
    assert LW.extension() is not None, "the C++ extension is not built: cd tools/csrc && python setup.py build_ext --inplace"
    K = G.intrinsics(FOCAL, PP)
    gen = torch.Generator().manual_seed(2)
    true = T.true_pose(9)
    start = G.perturb(true, 0.015 * DEPTH, 0.5, np.random.default_rng(3))
    q = query_at(true, gen)
    parts = parts_at(start)
    worst, rows = 0.0, []
    for transport in (True, False):
        for s in (2, 1, 0):
            fx, fy, cx, cy = Wp.level_intrinsics(K, s)
            Q = Wp.pool_image(q, s)
            stack = LW.gradients(Q)
            d, mk = Wp.pool_depth(parts["depth"], parts["alpha"] > 0.5, s)
            sel = mk.reshape(-1)
            X = Wp.backproject(d, (fx, fy, cx, cy)).reshape(-1, 3)[sel].contiguous()
            Iv = Wp.pool_image(parts["I"], s).reshape(3, -1)[:, sel].contiguous()
            Jv = Wp.pool_image(parts["J"], s).reshape(3, -1)[:, sel]
            Bv = Wp.pool_image(parts["binf"], s).reshape(3, -1)[:, sel]
            z0 = X[:, 2].contiguous()
            Dv = LW.slopes(Jv, Bv, z0, betas)
            Rt = torch.tensor(LW.start_state(), dtype=torch.float64)
            args = (stack, X, Iv, Dv, z0, Rt, fx, fy, cx, cy, DEPTH, transport, 1.345, 6, 1e-9, 64)
            t0 = time.time()
            a = LW.align_level("ext", *args)
            t1 = time.time()
            b = LW.align_level("torch", *args)
            t2 = time.time()
            diff = float(np.abs(a[0] - b[0]).max())
            worst = max(worst, diff)
            rows.append((transport, s, X.shape[0], a[1][0], b[1][0], diff, 1e3 * (t1 - t0) / max(a[1][0], 1), 1e3 * (t2 - t1) / max(b[1][0], 1)))
            assert a[1][0] == b[1][0] and a[1][2] == b[1][2] and diff < 1e-4, (transport, s, a[1], b[1], diff)
            assert torch.equal(a[4], b[4]) and float((a[2] - b[2]).abs().max()) < 1e-4 and float((a[3] - b[3]).abs().max()) < 1e-4
            e = LW.evaluate_torch(stack, X, Iv, Dv, z0, a[0], fx, fy, cx, cy, transport)
            assert torch.equal(e["use"], a[4]) and float((e["q"] - a[2]).abs().max()) < 1e-4 and float((e["m"] - a[3]).abs().max()) < 1e-4
            assert abs(e["sig"] - a[1][3]) < 1e-5 * max(1.0, abs(e["sig"]))
    for r in rows:
        print("C++ against PyTorch, %s water, level 1/%d, %d points: %d and %d steps, largest difference of R and t %.1e | %.2f ms against %.2f ms per step on the CPU"
              % ("moving" if r[0] else "frozen", 2 ** r[1], r[2], r[3], r[4], r[5], r[6], r[7]))
    assert LW.self_check(torch.device("cpu"))
    zz = torch.tensor([1.0, 2.0, 4.0])
    law = lambda z: ((0.2 * torch.exp(0.1 * (z - 5.0)))[None].expand(3, -1).clone(), (0.1 + 0.0 * z)[None].expand(3, -1).clone())
    sl = LW.slopes(torch.full((3, 3), 0.6), torch.full((3, 3), 0.3), zz, law)
    f = lambda z: 0.6 * torch.exp(-law(z)[0][0] * z) + 0.3 * (1.0 - torch.exp(-law(z)[1][0] * z))
    num = (f(zz * (1 + 1e-4)) - f(zz * (1 - 1e-4))) / (2e-4 * zz)
    print("depth slope of the water model with a depth dependent beta_D: %s against finite differences %s" % (sl[0].numpy().round(5), num.numpy().round(5)))
    assert float((sl[0] - num).abs().max()) < 1e-3, (sl, num)


def check_water_align():
    K, hw = G.intrinsics(FOCAL, PP), (H, W_)
    gen = torch.Generator().manual_seed(4)
    res = {k: [] for k in ("start", "ext", "torch", "half", "frozen")}
    secs = []
    for i in (6, 12, 18, 24, 30):
        true = T.true_pose(i)
        q = query_at(true, gen)
        pts = T.surface_points(tt(true))[0].reshape(-1, 3)[::37].double().numpy()
        for seed, (tr, rd) in enumerate(((0.01, 0.3), (0.02, 0.8))):
            start = G.perturb(true, tr * DEPTH, rd, np.random.default_rng(10 * i + seed))
            res["start"].append(G.pixel_shift(K, true, start, pts, hw))
            for key, kw in (("ext", dict(backend="ext")), ("torch", dict(backend="torch")), ("frozen", dict(backend="ext", transport=False))):
                r = LW.water_align(q, start, K, parts_at(start), levels=(2, 1, 0), iters=(6, 4, 3), **kw)
                assert r["ok"] and r["corr"] is not None and r["backend"] == kw["backend"], r
                res[key].append(G.pixel_shift(K, true, r["pose"], pts, hw))
                if key == "ext":
                    secs.append(r["stages"]["align"])
            r = LW.water_align(q, start, K, parts_at(start, half=True), levels=(2, 1), iters=(6, 6))
            assert r["ok"], r
            res["half"].append(G.pixel_shift(K, true, r["pose"], pts, hw))
    med = {k: float(np.median(v)) for k, v in res.items()}
    print("water aligned query (gain per channel, the water of the scene on it): start %.2f px -> C++ %.3f px, PyTorch %.3f px (largest %.3f), "
          "map parts at half resolution %.3f px, frozen water %.3f px | %.0f ms per query on the CPU" % (
              med["start"], med["ext"], med["torch"], max(res["ext"]), med["half"], med["frozen"], 1e3 * np.mean(secs)))
    assert med["ext"] < 0.1 and max(res["ext"]) < 0.5 and abs(med["ext"] - med["torch"]) < 0.01 and med["half"] < 0.3, med
    far = LW.water_align(query_at(T.true_pose(12), gen), T.true_pose(12), K, dict(parts_at(T.true_pose(12)), alpha=torch.zeros(H, W_)))
    assert not far["ok"] and far["steps"] == 0


def check_corr():
    g = torch.Generator().manual_seed(11)
    n = 4000
    a = torch.tensor([0.10, 0.45, 0.55])[:, None] + 0.01 * torch.randn(3, n, generator=g)
    b = torch.tensor([0.10, 0.45, 0.55])[:, None] + 0.01 * torch.randn(3, n, generator=g)
    am, bm = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    per = float((am * bm).sum() / (am.norm() * bm.norm()))
    pooled = float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()).norm() * (b - b.mean()).norm()))
    print("two unrelated images with the same colour cast: correlation centred per channel %.3f, centred over all channels %.3f" % (per, pooled))
    assert abs(per) < 0.1 and pooled > 0.9


def check_transport():
    K = G.intrinsics(FOCAL, PP)
    gen = torch.Generator().manual_seed(5)
    out = {True: [], False: []}
    for i in (6, 14, 22, 30):
        true = T.true_pose(i)
        q = query_at(true, gen, amp=0.04, noise=0.002)
        for sgn in (-1.0, 1.0):
            start = np.array(true, dtype=np.float64)
            start[:3, 3] += sgn * 0.04 * DEPTH * start[:3, 2]
            for transport in (True, False):
                r = LW.water_align(q, start, K, parts_at(start, amp=0.04), levels=(2, 1, 0), iters=(8, 6, 4), transport=transport, backend="ext")
                out[transport].append(G.pose_error(r["pose"], true, DEPTH)[0])
    m = {k: float(np.median(v)) for k, v in out.items()}
    print("weak texture, start 4.0 %% of depth along the optical axis: error after the alignment %.2f %% of depth with the water moving with the pose, "
          "%.2f %% with the water frozen at the start" % (m[True], m[False]))
    assert m[True] < 0.5 * m[False] and m[True] < 1.5, m


def check_fast_path():
    loc, hold, mapf, truth = TF.scene()
    hw = (H, W_)
    A = LF.TokenAnchors(loc)
    gen = torch.Generator().manual_seed(6)
    errs, tiers = [], []

    def water(p, i):
        img, alpha, z = T.render(tt(p), depth=True)
        return dict(I=img, J=img, alpha=alpha, depth=z, binf=torch.zeros(3, H, W_), betas=lambda zz: (torch.zeros(3, zz.shape[0]), torch.zeros(3, zz.shape[0])),
                    level=0)

    for n in hold:
        with torch.no_grad():
            img = (T.render(tt(truth[n])) + 0.004 * torch.randn(3, H, W_, generator=gen)).clamp(0.0, 1.0)
            pts = T.surface_points(tt(truth[n]))[0].reshape(-1, 3)[::37].double().numpy()
        r = LF.localize_fast(loc, img, None, anchors=A, water=water, levels=(2, 1, 0), iters=(6, 4, 3))
        assert set(r["stages"]) == {"encoder", "tokens", "pnp", "render", "align"} and r["backend"] == "ext" and r["ok"], r
        errs.append(G.pixel_shift(loc.K, truth[n], r["pose"], pts, hw))
        t = LF.localize_tiered(loc, img, None, anchors=A, water=water, levels=(2, 1, 0), iters=(6, 4, 3))
        tiers.append((t["fallback"], G.pixel_shift(loc.K, truth[n], t["pose"], pts, hw)))
    print("fast path with the water aligned step (anchors, PnP, one render, C++ steps): %.3f px (largest %.3f), with the check %.3f px, %d of %d fall back" % (
        np.median(errs), np.max(errs), np.median([x[1] for x in tiers]), sum(x[0] for x in tiers), len(tiers)))
    assert np.median(errs) < 0.6 and np.max(errs) < 2.0 and np.median([x[1] for x in tiers]) < 0.6


def check_bench():
    import tempfile
    import bench_gscpe as C
    loc, hold, mapf, truth = TF.scene()
    name = lambda i: "%06d.jpg" % i
    gen = torch.Generator().manual_seed(7)

    def load(nm):
        with torch.no_grad():
            return (T.render(tt(truth[E.fnum(nm)])) + 0.004 * torch.randn(3, H, W_, generator=gen)).clamp(0.0, 1.0)

    def strict(pose, med=None, t=None, alpha=False, which="underwater", depth=False):
        assert which == "restored" or med is not None
        return T.render(pose, med, t, alpha, which, depth)

    def score(img, gt):
        mae = float((img.clamp(0, 1) - gt).abs().mean())
        return [float(-10.0 * torch.log10(((img.clamp(0, 1) - gt) ** 2).mean() + 1e-12)), 1.0 - mae, mae]

    def water(p, num):
        img, alpha, z = T.render(tt(p), depth=True)
        return dict(I=img, J=img, alpha=alpha, depth=z, binf=torch.zeros(3, H, W_), betas=lambda zz: (torch.zeros(3, zz.shape[0]), torch.zeros(3, zz.shape[0])),
                    level=0)

    fit = dict(register=True, k=6, grid=2, tol=2.0, n=len(hold), total=len(hold), half_px_med=0.1, half_px_p90=0.2, half_t_med=0.1, half_r_med=0.1,
               inliers_med=100.0, matches_med=200.0, matcher="LoFTR (test)")
    res = C.run(dict(dev=torch.device("cpu"), tests=[name(i) for i in hold], fps=5.0, warmup=1, map_nums=mapf, load=load, medium=lambda num, pose: {},
                     render=strict, score=score, loc=loc, loc_public=None, out_dir=tempfile.mkdtemp(), ref_poses={name(i): truth[i] for i in hold}, ref_fit=fit,
                     ref_source="truth", fast_photo="affine", fast_levels=(2, 1, 0), fast_iters=(6, 4, 3), fast_dim=16, water=water))
    text = C.report(res, dict(scene="synthetic scene", map_frames=len(mapf), hw=[H, W_], primitives=0, map_mb=0.0, model="stand in", mapping_seconds=100.0,
                              map_seconds=5.0, weights_mb={"none": 0.0}, reference="registered", split="test : train = 1 : 3.3"))
    print(text[text.index(" In short"):text.index(" Reference")])
    labels = [r["label"] for r in res["rows"]]
    for want in ("anchors: water aligned", "anchors + check: water aligned", "anchors: water aligned, frozen water", "anchors: water aligned, PyTorch steps"):
        assert want in labels and len(want) <= 38, (want, labels)
    row = lambda label: next(r for r in res["rows"] if r["label"] == label)
    w, wt = row("anchors: water aligned"), row("anchors + check: water aligned")
    assert w["acc"][2] == 100.0 and w["px"] < 0.6 and wt["px"] < 0.6, (w, wt)
    fa = res["fast"]
    assert fa["water_cfg"] == dict(ext=True, levels=[2, 1, 0], iters=[6, 4, 3]), fa["water_cfg"]
    assert fa["water"]["backend"] == "ext" and fa["water_torch"]["backend"] == "torch" and fa["water_frozen"]["steps"] > 0
    assert "Water aligned: one render" in text and "in the C++ extension" in text and "anchors: water aligned" in text[:text.index(" Reference")]
    assert not res["notes"] or all("OpenCV" in x for x in res["notes"]), res["notes"]


def check_gpu():
    dev = torch.device("cuda")
    mod = LW.extension()
    assert mod is not None and mod.has_cuda(), "the C++ steps were built without CUDA: cd tools/csrc && rm -rf build && python setup.py build_ext --inplace"
    g = torch.Generator().manual_seed(9)
    for hs, ws in ((36, 64), (72, 128), (144, 256)):
        n = hs * ws
        img = LF.box(torch.rand(3, hs, ws, generator=g), 5)
        stack = LW.gradients(img).to(dev)
        X = torch.stack([(torch.rand(n, generator=g) - 0.5) * 2.4, (torch.rand(n, generator=g) - 0.5) * 1.4, 3.0 + torch.rand(n, generator=g)], -1).to(dev)
        mk = lambda lo, hi: (lo + (hi - lo) * torch.rand(3, n, generator=g)).to(dev)
        Iv, Dv = mk(0.1, 0.9), mk(-0.2, 0.05)
        z0 = X[:, 2].contiguous()
        St = torch.tensor(np.r_[LF.rodrigues(np.array([0.004, -0.006, 0.003])).reshape(-1), [0.01, -0.01, 0.02], [0.95, 1.0, 1.04]], dtype=torch.float64)
        f = ws * 0.55
        args = (stack, X, Iv, Dv, z0, St, f, f, (ws - 1) / 2.0, (hs - 1) / 2.0, 3.5, True, 1.345, 8, 1e-12, 64)
        for _ in range(2):
            LW.align_level("ext", *args)
            LW.align_level("torch", *args)
        torch.cuda.synchronize()
        t0 = time.time()
        a = LW.align_level("ext", *args)
        torch.cuda.synchronize()
        t1 = time.time()
        b = LW.align_level("torch", *args)
        torch.cuda.synchronize()
        t2 = time.time()
        diff = float(np.abs(a[0] - b[0]).max())
        print("GPU %s, level %d x %d (%d points): C++ %.2f ms against PyTorch %.2f ms per step, %d and %d steps, largest difference %.1e" % (
            torch.cuda.get_device_name(0), hs, ws, n, 1e3 * (t1 - t0) / max(a[1][0], 1), 1e3 * (t2 - t1) / max(b[1][0], 1), a[1][0], b[1][0], diff))
        assert abs(a[1][0] - b[1][0]) <= 1 and diff < 1e-3, (a[1], b[1], diff)
    assert LW.self_check(dev)


if __name__ == "__main__":
    torch.manual_seed(0)
    check_steps_match()
    check_water_align()
    check_corr()
    check_transport()
    check_fast_path()
    check_bench()
    if torch.cuda.is_available():
        check_gpu()
    print("test_water ok")
