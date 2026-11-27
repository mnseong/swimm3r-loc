import os
import subprocess
import sys
import tempfile
import time
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import mast3r.utils.path_to_dust3r
import bench_gap as GP
import bench_gscpe as C
import eval_gaussian_map as E
import loc_geom as G
import loc_ref as R
import test_fast as TF
import test_loc as T

H, W_, DEPTH = T.H, T.W_, T.DEPTH
tt = lambda p: torch.tensor(np.asarray(p), dtype=torch.float32)
name = lambda i: "%06d.jpg" % i

PATCH = (
    ('        res["per_query"][label] = [dict(r["ev"], ok=bool(r.get("ok", True)), sec=r.get("sec")) for r in rows]',
     '        res["per_query"][label] = [dict(r["ev"], ok=bool(r.get("ok", True)), sec=r.get("sec"), pose=r.get("pose")) for r in rows]'),
    ('            return [dict(ok=True, sec=x["retrieval"], ev=evaluate(j, x["start"], x["num"])) for j, x in enumerate(runs)]',
     '            return [dict(ok=True, sec=x["retrieval"], pose=x["start"], ev=evaluate(j, x["start"], x["num"])) for j, x in enumerate(runs)]'),
    ('            return [dict(ok=x["coarse"]["ok"], sec=x["retrieval"] + x["coarse"]["seconds"], ev=evaluate(j, x["coarse"]["pose"], x["num"])) for j, x in enumerate(runs)]',
     '            return [dict(ok=x["coarse"]["ok"], sec=x["retrieval"] + x["coarse"]["seconds"], pose=x["coarse"]["pose"], ev=evaluate(j, x["coarse"]["pose"], x["num"])) for j, x in enumerate(runs)]'),
    ('                     ev=evaluate(j, x["refine"]["pose"], x["num"])) for j, x in enumerate(runs)]',
     '                     pose=x["refine"]["pose"], ev=evaluate(j, x["refine"]["pose"], x["num"])) for j, x in enumerate(runs)]'),
    ('                                     ev=evaluate(j, r["pose"], x["num"])))',
     '                                     pose=r["pose"], ev=evaluate(j, r["pose"], x["num"])))'),
)


def gscpe_with_poses():
    src = open(C.__file__).read()
    if 'pose=r.get("pose")) for r in rows]' in src:
        return C, "bench_gscpe keeps the poses itself"
    for a, _ in PATCH:
        if src.count(a) != 1:
            return None, "bench_gscpe changed, the integration patch no longer applies"
    for a, b in PATCH:
        src = src.replace(a, b)
    mod = types.ModuleType("bench_gscpe_poses")
    mod.__file__ = C.__file__
    exec(compile(src, C.__file__, "exec"), mod.__dict__)
    return mod, "bench_gscpe with the integration patch applied in memory"


def world():
    loc, hold, mapf, truth = TF.scene()
    with torch.no_grad():
        imgs = torch.stack([T.render(tt(truth[i])) for i in mapf])
    return loc, hold, mapf, truth, imgs


def stand_matcher(loc, imgs, poses_of, seed=8):
    rng = np.random.default_rng(seed)
    calls = []

    def matcher(q, m):
        i = next(k for k in range(len(imgs)) if torch.equal(m, imgs[k]))
        calls.append(i)
        pose = poses_of[q.data_ptr()]
        v, u = np.meshgrid(np.arange(3, H, 6), np.arange(3, W_, 6), indexing="ij")
        u, v = u.reshape(-1), v.reshape(-1)
        X = loc.xyz[i][v, u].double().numpy()
        w2c = np.linalg.inv(pose)
        Xc = X @ w2c[:3, :3].T + w2c[:3, 3]
        z = np.where(Xc[:, 2] > 1e-6, Xc[:, 2], 1e-6)
        pq = np.stack([loc.K[0, 0] * Xc[:, 0] / z + loc.K[0, 2], loc.K[1, 1] * Xc[:, 1] / z + loc.K[1, 2]], 1) + rng.normal(0.0, 0.3, (len(u), 2))
        inside = (Xc[:, 2] > 1e-6) & (pq[:, 0] >= 0) & (pq[:, 0] <= W_ - 1) & (pq[:, 1] >= 0) & (pq[:, 1] <= H - 1)
        return pq[inside], np.stack([u, v], 1).astype(np.float64)[inside]

    matcher.name = "a stand in matcher"
    matcher.calls = calls
    return matcher


def loader(truth, poses_of, noise=0.0, seed=5):
    gen = torch.Generator().manual_seed(seed)

    def load(nm):
        with torch.no_grad():
            img = T.render(tt(truth[E.fnum(nm)]))
            if noise:
                img = (img + noise * torch.randn(3, H, W_, generator=gen)).clamp(0.0, 1.0)
        poses_of[img.data_ptr()] = truth[E.fnum(nm)]
        return img

    return load


def check_lifter():
    loc, hold, mapf, truth, imgs = world()
    seen = []

    def rgbd(i):
        seen.append(i)
        return T.render(tt(loc.poses[i]), which="restored", depth=True)

    lf = R.depth_lifter(loc, rgbd)
    rng = np.random.default_rng(0)
    pm = np.c_[rng.uniform(0, W_ - 1, 400), rng.uniform(0, H - 1, 400)]
    pq = rng.uniform(0, 100, (400, 2))
    a2, a3 = R.lift(loc, 3, pq, pm)
    b2, b3 = lf(loc, 3, pq, pm)
    lf(loc, 3, pq[:5], pm[:5])
    d = float(np.abs(a3 - b3).max())
    print("lifting by the rendered depth at the map camera against the Stage 1 points of the same pixels: largest difference %.1e over %d matches" % (d, len(b3)))
    assert a2.shape == b2.shape and float(np.abs(a2 - b2).max()) == 0.0 and d < 1e-3 and seen == [3], (d, seen)
    e2, e3 = lf(loc, 3, np.zeros((0, 2)), np.zeros((0, 2)))
    assert e2.shape == (0, 2) and e3.shape == (0, 3)

    def holed(i):
        img, al, z = T.render(tt(loc.poses[i]), which="restored", depth=True)
        al = al.clone()
        al[:, : W_ // 2] = 0.2
        return img, al, z

    lh = R.depth_lifter(loc, holed)
    h2, h3 = lh(loc, 3, pq, pm)
    right = np.round(pm[:, 0]) >= W_ // 2
    assert len(h2) == int(right.sum()) and np.allclose(h2, pq[right]) and "opacity above 0.5" in lh.name
    loc.valid[3][:, -W_ // 4:] = False
    lc = R.depth_lifter(loc, rgbd, common=True)
    c2, _ = lc(loc, 3, pq, pm)
    keep = np.round(pm[:, 0]) < W_ - W_ // 4
    loc.valid[3][:, -W_ // 4:] = True
    assert len(c2) == int(keep.sum()) and "Stage 1 has a point too" in lc.name, (len(c2), int(keep.sum()))
    print("support by opacity above 0.5 drops %d of %d matches on a half without support, the common support keeps %d" % (len(pq) - len(h2), len(pq), len(c2)))
    seen.clear()
    store = {}
    s1, s2 = R.depth_lifter(loc, rgbd, store=store), R.depth_lifter(loc, rgbd, common=True, store=store)
    x2, x3 = s1(loc, 5, pq, pm)
    y2, y3 = s2(loc, 5, pq, pm)
    assert seen == [5] and len(store) == 1 and float(np.abs(x3 - R.lift(loc, 5, pq, pm)[1]).max()) < 1e-3, (seen, len(store))
    assert len(x2) == len(y2) == len(pq) and float(np.abs(x3 - y3).max()) == 0.0, (len(x2), len(y2))
    four = R.depth_lifter(loc, lambda i: T.render(tt(loc.poses[i]), which="restored", depth=True) + (None,))
    assert len(four(loc, 2, pq, pm)[0]) == len(pq)


def check_register_gs():
    loc, hold, mapf, truth, imgs = world()
    poses_of = {}
    load = loader(truth, poses_of)
    m = stand_matcher(loc, imgs, poses_of)
    tests = [name(i) for i in hold]
    interp = E.interpolate_poses(loc.poses.copy(), hold, mapf)
    cache = {}
    pa, fa, _ = R.register_queries(loc, tests, load, mapf, interp, k=6, tol=1.5, verbose=False, matcher=m, images=imgs, cache=cache)
    n0 = len(m.calls)
    assert n0 == 6 * len(hold) and fa["matcher_calls"] == n0 and fa["reused"] == 0 and "lift" not in fa and len(cache) == n0, (n0, fa)
    plain, fp, _ = R.register_queries(loc, tests[:2], load, mapf, interp[:2], k=6, tol=1.5, verbose=False, matcher=m, images=imgs)
    assert "matcher_calls" not in fp and "lift" not in fp and len(m.calls) == n0 + 12
    m.calls.clear()
    rgbd = lambda i: T.render(tt(loc.poses[i]), which="restored", depth=True)
    pb, fb, _ = R.register_queries(loc, tests, load, mapf, interp, k=6, tol=1.5, verbose=False, matcher=m, images=imgs, lifter=R.depth_lifter(loc, rgbd),
                                   cache=cache)
    assert not m.calls and fb["matcher_calls"] == 0 and fb["reused"] == n0 and "Gaussian map" in fb["lift"], fb
    scale = float(np.median(loc.depths))
    d = GP.ref_diff(loc, pa, pb, tests, scale)

    def tilted(i):
        img, al, z = T.render(tt(loc.poses[i]), which="restored", depth=True)
        return img, al, z * (1.0 + 0.15 * (torch.arange(W_, dtype=torch.float32)[None] / W_ - 0.5))

    pt, ft, _ = R.register_queries(loc, tests, load, mapf, interp, k=6, tol=50.0, verbose=False, matcher=m, images=imgs, lifter=R.depth_lifter(loc, tilted),
                                   cache=cache)
    dt = GP.ref_diff(loc, pa, pt, tests, scale)
    print("reference B with the same matches (%d kept, %d new matcher calls): %d of %d queries, A and B differ by %.4f %% of depth, %.4f deg, %.3f px | "
          "with a depth tilted by 15 %% across the image %.3f %% of depth, %.3f deg, %.2f px" % (
              fb["reused"], fb["matcher_calls"], fb["n"], fb["total"], d["t_med"], d["r_med"], d["px_med"], dt["t_med"], dt["r_med"], dt["px_med"]))
    assert fb["n"] >= 7 and d["t_med"] < 0.05 and d["r_med"] < 0.02 and d["px_med"] < 0.1, d
    assert dt["n"] >= 5 and (dt["px_med"] > 10.0 * d["px_med"] + 0.2 or dt["r_med"] > 10.0 * d["r_med"] + 0.05), (d, dt)
    lines = R.reference_lines(fb, skipped=fb["total"] - fb["n"])
    text = "\n".join(lines)
    assert "lifted by the depth of the Gaussian map" in text and "with a stand in matcher" in text and "%%" not in text and len(lines) == 6, lines
    assert len(R.reference_lines(fa)) == 5 and "lifted by the points of the map" in "\n".join(R.reference_lines(fa))
    assert "floor" not in text and "floor" not in "\n".join(R.reference_lines(fa)) and "does not bound an error both share" in lines[-1], lines
    try:
        R.register_queries(loc, tests[:1], load, mapf, interp[:1], verbose=False, lifter=R.depth_lifter(loc, rgbd))
        raise AssertionError("lifting by another depth must need an outside matcher")
    except ValueError:
        pass
    return pa, fa


def check_splat():
    loc, hold, mapf, truth, imgs = world()
    img, al, dp = GP.splat(loc, imgs, [4], loc.poses[4])
    d_img = float((img - imgs[4]).abs().max())
    d_dep = float((dp - T.surface_points(tt(loc.poses[4]))[1]).abs().max())
    assert float(al.mean()) == 1.0 and d_img < 1e-5 and d_dep < 1e-3, (float(al.mean()), d_img, d_dep)
    q = truth[hold[4]]
    near = loc.near(q, 3)
    img, al, dp = GP.splat(loc, imgs, near, q)
    true_img, _, true_z = T.render(tt(q), depth=True)
    cov = float(al.mean())
    m = al > 0.5
    rel = float(((dp - true_z).abs() / true_z)[m].median())
    col = float((img - true_img).abs().mean(0)[m].mean())
    raw, alr, _ = GP.splat(loc, imgs, near, q, fill=0)
    print("keyframe splat of the 3 map frames nearest a held out pose: %.3f of the pixels covered (%.3f before hole filling), depth %.2f %% from the true depth, "
          "colour %.4f from the true view" % (cov, float(alr.mean()), 100.0 * rel, col))
    assert cov > 0.95 and cov >= float(alr.mean()) and rel < 0.01 and col < 0.03, (cov, rel, col)
    _, al2, _ = GP.splat(loc, imgs, near, np.asarray(q, dtype=np.float64) @ np.diag([-1.0, 1.0, -1.0, 1.0]))
    assert float(al2.mean()) < 0.05, float(al2.mean())
    i0, a0, d0 = GP.splat(loc, imgs, near, q, occl=0.0)
    i5, a5, d5 = GP.splat(loc, imgs, near, q)
    same = float((a0 == a5).float().mean())
    assert same > 0.995 and float((d0 - d5)[(a0 > 0) & (a5 > 0)].abs().max()) / DEPTH < 0.05, same
    n, f = 40, 40.0
    K = G.intrinsics(f, (n / 2.0, n / 2.0))
    v, u = torch.meshgrid(torch.arange(n, dtype=torch.float32), torch.arange(n, dtype=torch.float32), indexing="ij")
    ray = torch.stack([(u - float(K[0, 2])) / f, (v - float(K[1, 2])) / f, torch.ones_like(u)], -1)
    xyz = torch.stack([ray * 10.0, ray * 5.0])
    patch = (u >= 10) & (u < 30) & (v >= 10) & (v < 30)
    valid = torch.stack([torch.ones(n, n, dtype=torch.bool), patch & ((u + v) % 2 == 0)])
    cols = torch.stack([torch.tensor([1.0, 0.0, 0.0])[:, None, None].expand(3, n, n), torch.tensor([0.0, 1.0, 0.0])[:, None, None].expand(3, n, n)])
    two = types.SimpleNamespace(h=n, w=n, K=K, xyz=xyz, valid=valid)
    inner = (u >= 12) & (u < 28) & (v >= 12) & (v < 28)
    _, a0, d0 = GP.splat(two, cols, [0, 1], np.eye(4), occl=0.0)
    i1, a1, d1 = GP.splat(two, cols, [0, 1], np.eye(4))
    leak0, leak1 = float((d0[inner] > 7.0).float().mean()), float((d1[inner] > 7.0).float().mean())
    print("keyframe splat with a near patch of every other point in front of a far plane: %.2f of the patch shows the far plane without the visibility test, "
          "%.2f with it" % (leak0, leak1))
    assert leak0 > 0.4 and leak1 == 0.0 and float(a1.mean()) == 1.0 and float((d1[inner] - 5.0).abs().max()) < 1e-4, (leak0, leak1)
    assert float((i1[:, inner] - torch.tensor([0.0, 1.0, 0.0])[:, None]).abs().max()) < 1e-5 and float((d1[~patch & (u < 6)] - 10.0).abs().max()) < 1e-4
    nil = types.SimpleNamespace(h=n, w=n, K=K, xyz=xyz, valid=torch.zeros(2, n, n, dtype=torch.bool))
    ie, ae, de = GP.splat(nil, cols, [0, 1], np.eye(4))
    assert float(ae.sum()) == 0.0 and float(de.abs().sum()) == 0.0 and float(ie.abs().sum()) == 0.0


def check_parts():
    assert GP.parse_priors(["2:0", "0:5", "3"]) == [(2.0, 0.0), (0.0, 5.0), (3.0, 0.0)] and GP.parse_priors(None) == list(GP.LEVELS)
    assert GP.parse_priors([(1, 2)]) == [(1.0, 2.0)]
    assert [GP.bin_of(x, GP.T_EDGES) for x in (0.0, 2.99, 3.0, 5.9, 6.0, 12.0, 40.0)] == [0, 0, 1, 1, 2, 3, 3]
    assert GP.bin_names(GP.T_EDGES, "%") == ["below 3 %", "3 to 6 %", "6 to 12 %", "above 12 %"]
    lv = [(2.0, 0.0), (5.0, 0.0), (10.0, 0.0), (0.0, 2.0), (0.0, 5.0), (3.0, 3.0)]
    assert GP.basin(lv, [100.0, 90.0, 50.0, 85.0, 70.0, None]) == dict(translation=[5.0, 0.0], rotation=[0.0, 2.0], both=None)
    assert GP.basin(lv, [70.0, 100.0, 100.0, 79.9, 100.0, 80.0]) == dict(translation=None, rotation=None, both=[3.0, 3.0])
    scale = 2.0
    names = ["a", "b", "c", "d"]
    eye = np.eye(4)
    ref_a = {n: eye.copy() for n in names}
    ref_b = {n: eye.copy() for n in names[:3]}
    for k, n in enumerate(names[:3]):
        ref_b[n][:3, 3] = [0.02 * (k + 1), 0.0, 0.0]
    est = [eye.copy() for _ in names]
    est[1][:3, :3] = G.rot(np.radians([0.0, 3.0, 0.0]))
    res = dict(rows=[dict(label="x", family="f"), dict(label="y", family="f"), dict(label="short", family="f")],
               per_query=dict(x=[dict(t=0.0, r=0.0, pose=p) for p in est], y=[dict(t=0.0, r=0.0) for _ in est], short=[dict(t=0.0, r=0.0, pose=eye)]))
    rows, miss = GP.rescore(res, names, ref_a, ref_b, scale)
    assert miss == ["y"] and len(rows) == 1 and rows[0]["n"] == 3, (rows, miss)
    a, b = rows[0]["a"], rows[0]["b"]
    assert abs(a["t"]) < 1e-12 and abs(a["acc"][2] - 200.0 / 3.0) < 1e-9 and abs(b["t"] - 2.0) < 1e-9 and abs(b["r"]) < 1e-9 and abs(a["r"]) < 1e-9, (a, b)
    gaps = [dict(gap_t=1.0, gap_r=0.5), dict(gap_t=4.0, gap_r=6.0), dict(gap_t=13.0, gap_r=1.0), dict(gap_t=4.5, gap_r=12.0)]
    res2 = dict(rows=[dict(label="x", family="f")], per_query=dict(x=[dict(t=0.5, r=0.1, pose=eye), dict(t=3.0, r=0.4, pose=eye), dict(t=None, r=None),
                                                                      dict(t=8.0, r=1.0, pose=eye)]))
    st = GP.strata(res2, names, gaps, ref_b, scale)
    tb, rb = st[0]["t_bins"], st[0]["r_bins"]
    assert [c["n"] for c in tb] == [1, 2, 0, 1] and tb[2]["a"] is None and tb[3]["a"] is None and tb[1]["a"]["n"] == 2 and abs(tb[1]["a"]["t"] - 5.5) < 1e-12, tb
    assert [c["n"] for c in rb] == [2, 0, 1, 1] and tb[1]["b"]["n"] == 1 and abs(tb[1]["b"]["t"] - 2.0) < 1e-9, (rb, tb)
    good, bad = GP.usable([(2.0, 0.0), (0.0, 0.0), (-1.0, 2.0), (0.0, 3.0), (float("nan"), 1.0)])
    assert good == [(2.0, 0.0), (0.0, 3.0)] and len(bad) == 3 and bad[:2] == [(0.0, 0.0), (-1.0, 2.0)], (good, bad)
    ref = T.true_pose(7)
    a, b = GP.prior_pose(ref, 2.0, 0.0, 4.0, 3, 7), GP.prior_pose(ref, 10.0, 0.0, 4.0, 3, 7)
    da, db = a[:3, 3] - ref[:3, 3], b[:3, 3] - ref[:3, 3]
    assert abs(np.linalg.norm(da) - 0.08) < 1e-12 and abs(np.linalg.norm(db) - 0.4) < 1e-12 and np.allclose(da / 0.08, db / 0.4)
    assert np.allclose(a[:3, :3], ref[:3, :3]) and abs(G.motion(a, ref)[1]) < 1e-6
    ra, rb_ = GP.prior_pose(ref, 0.0, 2.0, 4.0, 3, 7), GP.prior_pose(ref, 0.0, 5.0, 4.0, 3, 7)
    wa, wb = G.rotvec(ref[:3, :3].T @ ra[:3, :3]), G.rotvec(ref[:3, :3].T @ rb_[:3, :3])
    assert abs(np.degrees(np.linalg.norm(wa)) - 2.0) < 1e-9 and abs(np.degrees(np.linalg.norm(wb)) - 5.0) < 1e-9 and np.allclose(wa / 2.0, wb / 5.0)
    assert np.allclose(ra[:3, 3], ref[:3, 3]) and abs(G.motion(ra, ref)[1] - 2.0) < 1e-6
    other = GP.prior_pose(ref, 2.0, 0.0, 4.0, 3, 8)[:3, 3] - ref[:3, 3]
    assert not np.allclose(other, da) and np.isfinite(GP.prior_pose(ref, 2.0, 1.0, 4.0, -1, 7)).all()
    fg = GP.frame_gaps(["000003.jpg", "000004.jpg", "000006.jpg", "000011.jpg"], [1, 2, 10], {"000003.jpg": 0, "000004.jpg": 0, "000011.jpg": 0}, {"000003.jpg": 0})
    assert fg == [dict(dist=1, n=2, a=2, b=1), dict(dist=2, n=1, a=1, b=0), dict(dist=4, n=1, a=0, b=0)], fg
    built = []

    def make():
        built.append(1)
        return lambda q, f: (q, f)

    lm = GP.lazy_matcher(make, "LoFTR, a stand in")
    assert not built and not lm.box and lm.name == "LoFTR, a stand in"
    assert lm(1, 2) == (1, 2) and lm(3, 4) == (3, 4) and len(built) == 1 and lm.box
    print("parts: priors parsed and checked, the prior direction kept across levels, bins, frame gaps, convergence basin, scoring against two references "
          "and the gap strata agree with hand counts")


def check_run():
    loc, hold, mapf, truth, imgs = world()
    poses_of = {}
    load = loader(truth, poses_of, noise=0.004)
    m = stand_matcher(loc, imgs, poses_of)
    tests = [name(i) for i in hold]
    interp = E.interpolate_poses(loc.poses.copy(), hold, mapf)
    cache = {}
    pa, fa, _ = R.register_queries(loc, tests, load, mapf, interp, k=6, tol=1.5, verbose=False, matcher=m, images=imgs, cache=cache)
    drop = [tests[2], tests[6]]
    given = {k: v for k, v in pa.items() if k not in drop}
    assert len(given) >= 6, len(given)

    def strict(pose, med=None, t=None, alpha=False, which="underwater", depth=False):
        assert which in ("underwater", "restored") and (which == "restored" or med is not None) and t is not None
        return T.render(pose, med, t, alpha, which, depth)

    def score(img, gt):
        mae = float((img.clamp(0, 1) - gt).abs().mean())
        return [float(-10.0 * torch.log10(((img.clamp(0, 1) - gt) ** 2).mean() + 1e-12)), 1.0 - mae, mae]

    out_dir = tempfile.mkdtemp()
    ctx = dict(dev=torch.device("cpu"), tests=tests, fps=5.0, warmup=1, map_nums=mapf, load=load, medium=lambda num, pose: {}, render=strict, score=score,
               loc=loc, loc_public=None, out_dir=out_dir, ref_poses=given, ref_fit=fa, ref_source="registration", fast_photo="affine",
               fast_levels=(2, 1, 0), fast_iters=(6, 4, 3), ablation=False, fast=False)
    mod, how = gscpe_with_poses()
    print("bench run for the gap test: %s" % how)
    a0 = time.time()
    res = (mod or C).run(ctx)
    m.calls.clear()
    gctx = dict(ctx, map_images=imgs, ref_matcher=m, ref_cache=cache, ref_k=6, ref_tol=1.5, gap_priors=["2:0", "10:0", "0:5"], gap_seed=3)
    out = GP.run_gap(gctx, res)
    text = GP.report_gap(out)
    print(text)
    print("gap run on the stand in: %.0f s" % (time.time() - a0))
    assert os.path.isfile(os.path.join(out_dir, "gap.json")) and os.path.isfile(os.path.join(out_dir, "gscpe.json"))
    assert not m.calls and out["fit_b"]["matcher_calls"] == 0 and out["fit_b"]["reused"] == len(cache) and out["diff"]["t_med"] < 0.05, out["fit_b"]
    assert "%%" not in text and ";" not in text and "real time" not in text and " -- " not in text
    assert out["scored"] == len(given) and out["free"] == 2 and out["queries"] == len(tests)
    if mod is not None:
        labels = [r["label"] for r in out["both"]]
        for want in ("pose of the retrieved map frame", "coarse: render, match, PnP", "refine: + masked warping", "no render: match the map frame, PnP"):
            assert want in labels, (want, labels)
        nb = len([x for x in given if x in out["ref_b"]])
        assert all(r["n"] == nb for r in out["both"]) and not out["missing_poses"]
        nr = next(r for r in out["both"] if r["label"] == "no render: match the map frame, PnP")
        assert nr["a"]["acc"][1] == 100.0 and nr["b"]["acc"][1] == 100.0
        assert nb < len(given) or abs(nr["a"]["t"] - next(x["t"] for x in res["rows"] if x["label"] == nr["label"])) < 1e-6
        assert "A   5/5" in text or " | A " in text
    st = out["strata"]
    assert st and all(sum(c["n"] for c in r["t_bins"]) == len(given) and sum(c["n"] for c in r["r_bins"]) == len(given) for r in st)
    assert len(out["gaps"]) == len(given) and all(g["gap_t"] > 0 for g in out["gaps"])
    nf = out["no_ref"]
    assert nf["names"] == drop and [r["label"] for r in nf["rows"]] == list(GP.FREE), nf
    for r in nf["rows"]:
        assert r["n"] == 2 and r["returned"] == 2 and r["s_mean"] is not None and (G.cv2 is None or r["flow"] is not None), r
        assert r["b"] is None or (r["b"]["n"] == len([x for x in drop if x in out["ref_b"]]) and r["b"]["acc"][1] == 100.0), r["b"]
    sw = out["priors"]
    assert [(x["t"], x["r"]) for x in sw] == [(2.0, 0.0), (10.0, 0.0), (0.0, 5.0)] and all([r["label"] for r in x["rows"]] == list(GP.ROWS) for x in sw)
    pr = {(x["t"], x["r"]): {r["label"]: r for r in x["rows"]} for x in sw}
    for (lt, lr), rows in pr.items():
        p0 = rows[GP.ROWS[0]]
        assert abs(p0["t"] - lt) < 1e-6 and abs(p0["r"] - lr) < 1e-6 and p0["success"] == 0.0 and "s_mean" not in p0, p0
    for k in GP.ROWS[1:]:
        for lv in ((2.0, 0.0), (10.0, 0.0), (0.0, 5.0)):
            assert pr[lv][k]["acc"][2] == 100.0 and pr[lv][k]["returned"] == len(out["prior_queries"]), (k, lv, pr[lv][k])
        assert pr[(2.0, 0.0)][k]["s_mean"] > 0.0 and pr[(2.0, 0.0)][k]["n_b"] == len([x for x in out["prior_queries"] if x in out["ref_b"]])
    for lv in ((2.0, 0.0), (10.0, 0.0), (0.0, 5.0)):
        assert pr[lv][GP.ROWS[2]]["success"] >= 80.0 and pr[lv][GP.ROWS[2]]["t"] < pr[lv][GP.ROWS[1]]["t"], pr[lv]
        assert pr[lv][GP.ROWS[4]]["s_mean"] > pr[lv][GP.ROWS[3]]["s_mean"] and pr[lv][GP.ROWS[5]]["s_mean"] >= pr[lv][GP.ROWS[3]]["s_mean"], pr[lv]
    passes_seen = []
    for x in sw:
        one, more = x["per_query"][GP.ROWS[3]], x["per_query"][GP.ROWS[5]]
        assert [e["name"] for e in one] == [e["name"] for e in more] and all(1 <= e["iters"] <= out["coarse_iters"] for e in more)
        for e1, e2 in zip(one, more):
            passes_seen.append(e2["iters"])
            if e2["iters"] == 1:
                assert e1["t"] == e2["t"] and e1["r"] == e2["r"] and e2["sec"] >= e1["sec"], (e1, e2)
    print("no render with the nearest frame taken again after every PnP: passes per query %s" % sorted(set(passes_seen)))
    gb = {g["name"]: GP.bin_of(g["gap_t"], GP.T_EDGES) for g in out["gaps"]}
    want_n = [sum(1 for q in out["prior_queries"] if gb[q] == b) for b in range(len(GP.T_EDGES) + 1)]
    for x in sw:
        for r in x["rows"]:
            bg = r["by_gap"]
            assert [c["n"] for c in bg] == want_n, (bg, want_n)
            tot = sum(c["success"] * c["n"] for c in bg if c["n"]) / float(sum(want_n))
            assert abs(tot - r["success"]) < 1e-9 and all(c["success"] is None for c in bg if not c["n"]), (r["label"], bg, r["success"])
    assert "success by the translation gap" in text
    bs = {b["label"]: b for b in out["basin"]}
    assert set(bs) == set(GP.ROWS) and bs[GP.ROWS[0]]["translation"] is None and bs[GP.ROWS[0]]["rotation"] is None and "both" not in bs[GP.ROWS[0]]
    assert bs[GP.ROWS[2]]["translation"] == [10.0, 0.0] and bs[GP.ROWS[2]]["rotation"] == [0.0, 5.0], bs[GP.ROWS[2]]
    for k in GP.ROWS[1:]:
        rates = [pr[lv][k]["success"] for lv in ((2.0, 0.0), (10.0, 0.0), (0.0, 5.0))]
        assert dict(bs[k], label=None) == dict(GP.basin([(2.0, 0.0), (10.0, 0.0), (0.0, 5.0)], rates), label=None), (k, bs[k], rates)
    assert "Navigation prior, simulated" in text and "Basin" in text and "Gap strata" in text and "No reference" in text and "Reference B" in text
    assert "Frame gap" in text and GP.ROWS[5] in text and "sensitivity check" in text and "drawn once per query" in text and "Stage 1 has a point as well" in text
    for line in text.split("\n"):
        prose = line.split(" | ")[0]
        for lab in list(GP.ROWS) + [r["label"] for r in res["rows"]]:
            prose = prose.replace(lab, "")
        assert ":" not in prose, line
    assert [g["label"] for g in out["global"]] == [GP.FREE[0]] and out["global"][0]["success"] is not None and out["global"][0]["n"] == len(out["prior_queries"])
    assert mod is None or out["global"][0]["n_b"] == len([x for x in out["prior_queries"] if x in out["ref_b"]])
    fgs = out["frame_gaps"]
    assert sum(g["n"] for g in fgs) == len(tests) and sum(g["a"] for g in fgs) == len(given) and sum(g["b"] for g in fgs) == len(out["ref_b"]), fgs
    assert out["fit_common"]["n"] == out["fit_b"]["n"] and out["diff_common"]["t_med"] < 0.05 and out["fit_b"]["matcher_built"], out["fit_common"]
    stripped = dict(res, per_query={k: [{kk: vv for kk, vv in e.items() if kk != "pose"} for e in v] for k, v in res["per_query"].items()})
    lean = GP.run_gap(dict(ctx, out_dir=tempfile.mkdtemp(), gap_priors=["5:0"], gap_queries=3, gap_flow=False), stripped)
    lt = GP.report_gap(lean)
    assert len(lean["prior_queries"]) == 3 and lean["prior_queries"][0] == res["names"][0] and lean["prior_queries"][-1] == res["names"][-1]
    assert not lean["both"] and lean["ref_b_n"] == 0 and any("reference B needs" in x for x in lean["notes"]), lean["notes"]
    assert any("splat row is left out" in x for x in lean["notes"]) and all(r["label"] != GP.ROWS[6] for r in lean["priors"][0]["rows"])
    assert all(r.get("flow") is None for r in lean["priors"][0]["rows"]) and "Reference B" not in lt and "%%" not in lt and ";" not in lt
    assert lean["global"][0]["n"] == 3 and not any("keeps no pose" in x for x in lean["notes"])
    print("without the matcher and the map frames: reference B and the splat row are left out with a note, %d prior queries" % len(lean["prior_queries"]))
    nop = GP.run_gap(dict(gctx, out_dir=tempfile.mkdtemp(), gap_priors=["2:0"], gap_queries=2, ref_matcher=None, ref_fit=dict(fa, matcher="LoFTR, a stand in name"),
                          gap_common_check=False), stripped)
    fb = nop["fit_b"]
    assert fb["matcher_calls"] == 0 and fb["matcher_built"] is False and fb["matcher"] == "LoFTR, a stand in name" and "fit_common" not in nop, fb
    assert not nop["both"] and len(nop["missing_poses"]) == len(res["rows"]) and any("keeps no pose" in x for x in nop["notes"]), nop["notes"]
    assert "gscpe.json keeps no pose" in GP.report_gap(nop) and nop["ref_b_n"] == out["ref_b_n"]
    print("an outside matcher named LoFTR is loaded only when a pair is missing from the cache of reference A, here never")
    tiny = GP.run_gap(dict(ctx, tests=list(res["names"]), out_dir=tempfile.mkdtemp(), ref_fit=dict(fa, register=False), gap_priors=["0:0", "0.5:0", "-1:2"],
                           gap_queries=1, gap_flow=False), res)
    tt_ = GP.report_gap(tiny)
    nts = " ".join(tiny["notes"])
    assert "made only for a reference registered" in nts and "simulated prior needs an error above zero" in nts and "already counts as a success" in nts, nts
    assert tiny["levels"] == [[0.5, 0.0]] and len(tiny["priors"]) == 1 and tiny["free"] == 0 and "no_ref" not in tiny and len(tiny["prior_queries"]) == 1
    assert tiny["priors"][0]["rows"][0]["success"] == 100.0 and "Navigation prior" in tt_
    orig = R.depth_lifter

    def broken(lz, fn, min_alpha=0.5, common=False, store=None):
        if common:
            raise RuntimeError("a stand in failure")
        return orig(lz, fn, min_alpha=min_alpha, common=common, store=store)

    GP.R.depth_lifter = broken
    try:
        bc = GP.run_gap(dict(gctx, tests=list(res["names"]), out_dir=tempfile.mkdtemp(), gap_priors=["5:0"], gap_queries=1, gap_flow=False), res)
    finally:
        GP.R.depth_lifter = orig
    assert bc["ref_b_n"] == len([x for x in res["names"] if x in out["ref_b"]]) and "fit_common" not in bc and bc["fit_b"]["matcher_built"]
    assert any("common support failed (RuntimeError, a stand in failure)" in x for x in bc["notes"]) and not any("reference B failed" in x for x in bc["notes"]), bc["notes"]
    none = GP.run_gap(dict(ctx, tests=tests[:2], ref_poses={}, out_dir=tempfile.mkdtemp(), gap_priors=["5:0"], gap_flow=False), None)
    nt = GP.report_gap(none)
    assert none["scored"] == 0 and none["free"] == 2 and not none.get("priors") and not none["global"] and "No reference" in nt and "Navigation prior" not in nt
    print("edge cases: prior errors of zero or below are left out, a single prior query, a reference that is not a registration, no reference at all")


def check_script():
    d = tempfile.mkdtemp()
    src = os.path.join(ROOT, "run_gap.sh")
    open(os.path.join(d, "run_gap.sh"), "w").write(open(src).read())
    stub = ("#!/usr/bin/env bash\n"
            "HB=$HOLD; [ \"$BLOCK\" = 1 ] || HB=${HOLD}b${BLOCK}\n"
            "O=\"$2/${1}_hold${HB}w${WIN}_loftr/loc\"\n"
            "mkdir -p \"$O\"\n"
            "echo \"$PWD|$HOLD|$BLOCK|$WIN|$HOLD_MAP|$REF_MATCHER|$*\" >> calls.txt\n"
            "echo gap > \"$O/gap.txt\"\n"
            "echo '{\"new\": 1}' > \"$O/gscpe.json\"\n")
    open(os.path.join(d, "run_gscpe.sh"), "w").write(stub)
    old = os.path.join(d, "out", "v13c_hold12b3w5_loftr", "loc")
    os.makedirs(old)
    open(os.path.join(old, "gscpe.json"), "w").write('{"old": 1}')
    open(os.path.join(old, "gscpe.txt"), "w").write("old table")
    env = dict(os.environ, GAP_BLOCKS="3 9 12", GAP_PRIORS="2:0 0:5", GAP_QUERIES="5")
    env.pop("TAG", None)
    p = subprocess.run(["bash", os.path.join(d, "run_gap.sh"), "v13c", "out", "--gscpe_no_ablation"], cwd="/", env=env, capture_output=True, text=True)
    calls = open(os.path.join(d, "calls.txt")).read().strip().split("\n")
    kept = [x for x in os.listdir(old) if x.startswith("before_gap_")]
    print(p.stdout.strip().split("\n")[-4:])
    assert p.returncode == 1 and len(calls) == 2 and "failed splits (BLOCK): 12" in p.stdout, (p.returncode, p.stdout, p.stderr)
    for c, b in zip(calls, ("3", "9")):
        cwd, hold, block, win, hm, rm, args = c.split("|")
        assert os.path.realpath(cwd) == os.path.realpath(d) and (hold, block, win, hm, rm) == ("12", b, "5", "register", "loftr"), c
        assert args == "v13c out --gap_bench --gap_priors 2:0 0:5 --gap_seed 0 --gap_queries 5 --gscpe_no_ablation", args
    assert len(kept) == 1 and open(os.path.join(old, kept[0], "gscpe.json")).read() == '{"old": 1}' and os.path.isfile(os.path.join(old, kept[0], "gscpe.txt"))
    assert p.stdout.count("gap.txt written") == 2 and os.path.join(os.path.realpath(d), "out", "v13c_hold12b9w5_loftr", "loc", "gap.txt") in p.stdout.replace(d, os.path.realpath(d))
    print("run_gap.sh with a stand in for run_gscpe.sh: splits 3 and 9 run from the repository folder with the gap options, 12 is refused, the earlier results "
          "of split 3 are kept in %s" % kept[0])


def main():
    torch.manual_seed(0)
    check_parts()
    check_script()
    check_lifter()
    check_register_gs()
    check_splat()
    check_run()
    print("test_gap ok")


if __name__ == "__main__":
    main()
