import json
import os
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import mast3r.utils.path_to_dust3r
import bench_verify as V
import eval_gaussian_map as E
import loc_geom as G
import loc_warp as Wp
import test_fast as TF
import test_loc as T

H, W_ = T.H, T.W_
tt = lambda p: torch.tensor(np.asarray(p), dtype=torch.float32)
name = lambda i: "%06d.jpg" % i


def check_metrics():
    y = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    s = np.array([3.0, 2.0, 1.0, 0.0, -1.0, -2.0])
    assert V.auroc(s, y) == 1.0 and V.auroc(-s, y) == 0.0 and V.auroc(np.zeros(6), y) == 0.5 and V.auroc(s[:3], y[:3]) is None
    closed = np.array([0.9, 0.8, -np.inf, 0.1, 0.2, -np.inf])
    assert abs(V.auroc(closed, y) - 6.5 / 9.0) < 1e-12
    s = np.arange(10.0)
    y = s >= 5
    assert V.keep_threshold(s, y, 0.9) == 5.0 and V.keep_threshold(s, y, 0.5) == 7.0 and V.keep_threshold(s, np.zeros(10, bool), 0.9) == np.inf
    assert V.precision_threshold(s, y, 0.95) == 5.0 and V.precision_threshold(s, ~y, 0.95) == np.inf
    noisy = np.array([9.0, 8.0, 7.5, 7.0, 6.0, 5.0])
    assert V.precision_threshold(noisy, np.array([1, 1, 0, 1, 1, 1], bool), 0.75) == 5.0
    sc = np.r_[10.0 + np.array([0.5, 0.6, 0.7, 0.0, 0.1, 0.2]), np.array([0.5, 0.6, 0.7, 0.0, 0.1, 0.2])]
    yy = np.array([1, 1, 1, 0, 0, 0] * 2, dtype=bool)
    fold = np.array([0] * 6 + [1] * 6)
    acc, th = V.held_out(sc, yy, fold)
    r = V.rates(acc["r90"], yy)
    print("held out thresholds on two folds with shifted scores: false acceptance %.2f and kept %.2f (in sample both folds would separate perfectly)" % (r["fa"], r["rec"]))
    assert th["1"]["r90"] == 0.5 and th["2"]["r90"] == 10.5 and not th["1"]["in_sample"]
    assert r["fa"] == 0.5 and r["rec"] == 0.5 and r["tp"] == 3 and r["fp"] == 3
    one, th1 = V.held_out(sc[:6], yy[:6], np.zeros(6, dtype=np.int64))
    assert th1["1"]["in_sample"] and V.rates(one["r90"], yy[:6])["fa"] == 0.0
    f, blocks = V.folds([6, 7, 8, 18, 19, 20, 30, 31, 32])
    assert f.tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1] and len(blocks) == 3
    f, _ = V.folds([30, 6, 19, 7, 31, 18], mode="alternate")
    assert f.tolist() == [0, 0, 1, 0, 0, 1]
    f, blocks = V.folds([5, 6, 7])
    assert sorted(f.tolist()) == [0, 1, 1] and len(blocks) == 1
    assert len(V.offsets()) == 29 and len(set(V.offsets())) == 29
    sc = np.array([-np.inf, -np.inf, 0.9, 0.8, 0.7, -np.inf, 0.1, 0.2] * 2)
    yy = np.array([1, 1, 1, 1, 1, 0, 0, 0] * 2, dtype=bool)
    fold = np.array([0] * 8 + [1] * 8)
    acc, th = V.held_out(sc, yy, fold)
    assert th["1"]["r90"] == -np.inf and not any(a[~np.isfinite(sc)].any() for a in acc.values())
    assert V.rates(acc["r90"], yy)["rec"] == 0.6 and V.rates(acc["r90"], yy)["fa"] == 4.0 / 6.0
    assert V.precision_threshold(np.array([-np.inf, 1.0, 2.0]), np.array([0, 1, 1], bool), 0.95) == 1.0
    assert V.precision_threshold(np.full(4, -np.inf), np.array([0, 1, 1, 0], bool), 0.95) == np.inf
    print("uncomputable scores are never accepted, even where the calibrated threshold falls to -inf (kept %.2f, false acceptance %.2f)" % (
        V.rates(acc["r90"], yy)["rec"], V.rates(acc["r90"], yy)["fa"]))
    js = np.repeat(np.arange(6), 4)
    y = np.tile(np.array([1, 1, 0, 0], dtype=bool), 6)
    good = np.tile(np.array([0.9, 0.8, 0.2, 0.1]), 6)
    noisy = good + np.random.default_rng(0).normal(scale=0.4, size=len(good))
    u = V.uncertainty(js, dict(d=good, b=noisy), y, 300, 0, pairs=(("d", "b", "x"),))
    u2 = V.uncertainty(js, dict(d=good, b=noisy), y, 300, 0, pairs=(("d", "b", "x"),))
    assert u == u2 and u["queries"] == 6 and u["auroc"]["d"] == [1.0, 1.0] and u["pairs"][0]["diff"] > 0.0
    assert u["pairs"][0]["ci"][0] >= 0.0 and u["pairs"][0]["above"] > 0.5 and u["auroc"]["b"][0] < u["auroc"]["b"][1]
    print("query resampling: a perfect check against a noisy one, AUROC difference %+.3f, interval %+.3f to %+.3f, above 0 in %.0f %% of draws" % (
        u["pairs"][0]["diff"], u["pairs"][0]["ci"][0], u["pairs"][0]["ci"][1], 100.0 * u["pairs"][0]["above"]))


def check_lcn_and_splat():
    g = torch.Generator().manual_seed(0)
    q = torch.rand(3, 48, 64, generator=g)
    img = (0.6 * q + 0.1 + 0.05 * torch.rand(3, 48, 64, generator=g)).clamp(0.0, 1.0)
    m = torch.ones(48, 64, dtype=torch.bool)
    m[:, :24] = False
    c1 = V.lcn_corr(q, img, m)
    q2, img2 = q.clone(), img.clone()
    q2[:, :, :24] = torch.rand(3, 48, 24, generator=g)
    img2[:, :, :24] = 5.0 * torch.rand(3, 48, 24, generator=g)
    c2 = V.lcn_corr(q2, img2, m)
    full = V.lcn_corr(q, q, torch.ones(48, 64, dtype=torch.bool))
    print("locally normalised correlation over the covered pixels %.4f, with other content where the map has no support %.4f, an image with itself %.4f" % (
        c1, c2, full))
    assert abs(c1 - c2) < 1e-6 and c1 > 0.5 and abs(full - 1.0) < 1e-5
    few = torch.zeros(48, 64, dtype=torch.bool)
    few[:4, :4] = True
    assert V.lcn_corr(q, img, few) is None
    from loc_fast import contrast
    a, _ = V.lcn(q, torch.ones(48, 64, dtype=torch.bool))
    assert float((a - contrast(q)).abs().max()) < 1e-5
    r1, r2 = V.rgb_corr(q, img, m), V.rgb_corr(q2, img2, m)
    same = V.rgb_corr(q, 0.5 * q + torch.tensor([0.1, 0.2, 0.3])[:, None, None], m)
    cast = V.rgb_corr(q, q * torch.tensor([1.0, 0.2, 0.2])[:, None, None], m)
    lcn_cast = V.lcn_corr(q, q * torch.tensor([1.0, 0.2, 0.2])[:, None, None], m)
    print("plain colour correlation %.4f, with other content where the map has no support %.4f, after one gain and offsets per channel %.4f, after a"
          " colour cast %.4f (locally normalised %.4f)" % (r1, r2, same, cast, lcn_cast))
    assert abs(r1 - r2) < 1e-6 and abs(same - 1.0) < 1e-5 and cast < 0.95 and V.rgb_corr(q, img, few) is None
    yy, xx = torch.meshgrid(torch.arange(48.0), torch.arange(64.0), indexing="ij")
    ramp = torch.stack([xx / 64.0, yy / 48.0, 0.5 + 0.0 * xx])
    hole = torch.ones(48, 64, dtype=torch.bool)
    hole[:, 10] = False
    hole[20:40, 30:50] = False
    ff = V.fill(ramp * hole.float()[None], hole)
    crack = float((ff[0, :, 10] - ramp[0, :, 10]).abs().max())
    print("gaps of a splat filled from their neighbours, covered pixels unchanged, largest error in a one pixel crack of a ramp %.4f" % crack)
    assert bool(torch.isfinite(ff).all()) and float((ff - ramp)[:, hole].abs().max()) == 0.0 and crack <= 1.0 / 64.0 + 1e-6
    assert float(ff[:, ~hole].min()) >= 0.0 and float(ff[0][~hole].max()) <= 63.0 / 64.0 + 1e-6 and float((ff[2][~hole] - 0.5).abs().max()) < 1e-6
    assert V.fill(ramp, torch.ones(48, 64, dtype=torch.bool)) is ramp and float(V.fill(0.0 * ramp, torch.zeros(48, 64, dtype=torch.bool)).abs().max()) == 0.0
    K = G.intrinsics(40.0, (32.0, 24.0))
    intr = Wp.level_intrinsics(K, 0)
    v, u = torch.meshgrid(torch.arange(48.0), torch.arange(64.0), indexing="ij")
    ray = torch.stack([(u - intr[2]) / intr[0], (v - intr[3]) / intr[1], torch.ones_like(u)], -1).reshape(-1, 3)
    X = torch.cat([2.0 * ray, 4.0 * ray, -3.0 * ray])
    col = torch.cat([torch.tensor([1.0, 0.0, 0.0])[:, None].expand(3, len(ray)), torch.tensor([0.0, 0.0, 1.0])[:, None].expand(3, len(ray)),
                     torch.tensor([0.0, 1.0, 0.0])[:, None].expand(3, len(ray))], 1)
    sp, cov, dep = V.splat(X, col, np.eye(4), intr, 48, 64)
    assert bool(cov.all()) and float((sp[0] - 1.0).abs().max()) < 1e-6 and float(sp[1:].abs().max()) < 1e-6 and float((dep - 2.0).abs().max()) < 1e-5
    sp2, cov2, _ = V.splat(X[len(ray):], col[:, len(ray):], np.eye(4), intr, 48, 64)
    assert bool(cov2.all()) and float((sp2[2] - 1.0).abs().max()) < 1e-6
    keep = torch.arange(len(X)) % 3 != 0
    k1, k2 = V.splat(X, col, np.eye(4), intr, 48, 64, keep), V.splat(X[keep], col[:, keep], np.eye(4), intr, 48, 64)
    assert all(torch.allclose(u.float(), v.float()) for u, v in zip(k1, k2)) and not bool(V.splat(X, col, np.eye(4), intr, 48, 64, keep & False)[1].any())
    loc, hold, mapf, truth = TF.scene()
    i = 7
    with torch.no_grad():
        im = T.render(tt(truth[mapf[i]]))
    X, c = loc.xyz[i][loc.valid[i]], im[:, loc.valid[i]]
    s0, m0, z0 = V.splat(X, c, loc.poses[i], Wp.level_intrinsics(loc.K, 0), H, W_)
    s1, m1, _ = V.splat(X, c, loc.poses[i], Wp.level_intrinsics(loc.K, 1), H // 2, W_ // 2)
    c1 = V.lcn_corr(Wp.pool_image(im, 1), s1, m1)
    print("a map frame drawn at its own pose from its own points, largest colour difference %.1e at full size, %.1f %% covered, correlation %.4f at half size" % (
        float((s0 - im).abs().max()), 100.0 * float(m1.float().mean()), c1))
    zt = T.surface_points(tt(truth[mapf[i]]))[1]
    assert bool(m0.all()) and float((s0 - im).abs().max()) < 1e-5 and float((z0 - zt).abs().max()) < 1e-4 and float(m1.float().mean()) > 0.99 and c1 > 0.97


def bench_ctx(out_dir, render=None):
    loc, hold, mapf, truth = TF.scene()

    def load(nm):
        gen = torch.Generator().manual_seed(1000 + E.fnum(nm))
        with torch.no_grad():
            return (T.render(tt(truth[E.fnum(nm)])) + 0.004 * torch.randn(3, H, W_, generator=gen)).clamp(0.0, 1.0)

    def strict(pose, med=None, t=None, alpha=False, which="underwater", depth=False):
        assert which == "restored" or med is not None
        return T.render(pose, med, t, alpha, which, depth)

    with torch.no_grad():
        mi = torch.stack([T.render(tt(truth[i])) for i in mapf])
    ctx = dict(dev=torch.device("cpu"), tests=[name(i) for i in hold], fps=5.0, warmup=1, map_nums=mapf, load=load, medium=lambda num, pose: {},
               render=render or strict, score=None, loc=loc, loc_public=None, out_dir=out_dir, ref_poses={name(i): truth[i] for i in hold},
               ref_fit=dict(n=len(mapf), scale=1.0, t_med=0.0, t_max=0.0, r_med=0.0, r_max=0.0), ref_source="truth", fast_photo="affine",
               fast_levels=(2, 1, 0), fast_iters=(6, 4, 3), fast_check=dict(min_corr=0.25), map_images=mi)
    return ctx, loc, hold, mapf, truth


def check_run():
    import bench_gscpe as C
    out_dir = tempfile.mkdtemp()
    ctx, loc, hold, mapf, truth = bench_ctx(out_dir)
    t0 = time.time()
    res = C.run(dict(ctx, fast=False, ablation=False, coarse_iters=1, levels=0, steps=2))
    t1 = time.time()
    out = V.run_verify(ctx, res)
    t2 = time.time()
    lines = V.report_verify(out)
    text = "\n".join(lines)
    print(text)
    print("benchmark %.1f s, verification %.1f s on the CPU stand in" % (t1 - t0, t2 - t1))
    cands = out["candidates"]
    assert out["queries"] == len(hold) == res["frames"] and out["n"] == len(cands) and os.path.isfile(os.path.join(out_dir, "verify.json"))
    back = json.load(open(os.path.join(out_dir, "verify.json")))
    every_key = {"a_tok", "a_dec", "b", "c", "d", "e", "b_rgb", "c_rgb", "d_rgb", "e_rgb", "g", "h"}
    assert back["n"] == out["n"] and len(back["candidates"]) == out["n"] and all(set(c["scores"]) >= every_key for c in back["candidates"])
    per = {}
    for c in cands:
        per.setdefault(c["frame"], []).append(c)
    assert all(len(v) == out["cap"] for v in per.values()), {k: len(v) for k, v in per.items()}
    srcs = set(c["source"] for c in cands if c["type"] == "method")
    assert {"no render, top 1", "fast path, token PnP", "fast path, after alignment", "anchors, after alignment"} <= srcs, srcs
    assert set(c["type"] for c in cands) == {"method", "perturbed", "wrong"}
    pert = [c for c in cands if c["type"] == "perturbed"]
    worst = max(max(abs(c["t"] - c["offset"][0]), abs(c["r"] - c["offset"][1])) for c in pert)
    assert worst < 1e-6 and all(c["ok2"] == (c["offset"][0] <= 2.0 and c["offset"][1] <= 2.0) for c in pert), worst
    assert all(c["ok5"] == (c["offset"][0] <= 5.0 and c["offset"][1] <= 5.0) for c in pert)
    seen = {}
    for c in pert:
        seen[tuple(c["offset"])] = seen.get(tuple(c["offset"]), 0) + 1
    assert len(seen) == 29 and max(seen.values()) - min(seen.values()) <= 1, seen
    far = [c for c in cands if c["source"] == "pose of a far map frame"]
    ret = [c for c in cands if c["source"].startswith("PnP on retrieved map frame")]
    assert len(far) == 2 * len(hold) and len(ret) == 4 * len(hold) and all(not c["ok5"] for c in far)
    for c in cands:
        dt, dr = G.motion(np.asarray(c["pose"]), truth[c["frame"]])
        assert abs(100.0 * dt / out["depth_units"] - c["t"]) < 1e-9 and abs(dr - c["r"]) < 1e-9
    ch = out["checks"]
    nf = sum(1 for c in cands if c["source"] in ("fast path, after alignment", "anchors, after alignment"))
    assert abs(ch["f"]["cover"] - nf / float(len(cands))) < 1e-9 and ch["f"]["2/2"]["all"]["n"] == nf and not ch["f"]["closed"]
    assert all(ch[k]["cover"] > 0.6 and ch[k]["closed"] for k in every_key), {k: ch[k]["cover"] for k in ch}
    au = {k: ch[k]["2/2"]["all"]["auroc"] for k in ch}
    print("AUROC within 2 %% and 2 deg on the stand in: %s" % ", ".join("%s %.3f" % (k, v) for k, v in au.items() if v is not None))
    assert au["d"] > 0.75 and au["e"] > 0.75 and au["b"] > 0.7 and au["c"] > 0.7 and au["a_dec"] > 0.8 and au["a_tok"] > 0.7 and au["g"] > 0.9 and au["h"] > 0.8, au
    assert all(au[k] > 0.6 for k in ("b_rgb", "c_rgb", "d_rgb", "e_rgb")), au
    for k in ("b", "c", "d", "e"):
        share = np.mean([c["ms"][k + "_rgb"] for c in cands]) / np.mean([c["ms"][k] for c in cands])
        assert share > 0.4, ("a plain colour row is charged the draw it shares", k, share)
    un = out["uncertainty"]["2/2"]
    assert set(un["auroc"]) == every_key and un["queries"] == len(hold) and un["n"] == 1000
    assert [(p["a"], p["b"]) for p in un["pairs"]] == [(a, b) for a, b, _ in V.PAIRS]
    for p in un["pairs"]:
        assert abs(p["diff"] - (au[p["a"]] - au[p["b"]])) < 1e-9 and p["ci"][0] <= p["ci"][1] and 0.0 <= p["above"] <= 1.0
    assert all(len(ch[k]["2/2"]["fa_folds"]) == 2 for k in ch) and not out["folds"]["in_sample"]
    for k in ("a_dec", "b", "d"):
        a = ch[k]["2/2"]["all"]
        assert a["n"] == len(cands) and a["r90"]["n1"] + a["r90"]["n0"] == len(cands) and set(ch[k]["2/2"]["thresholds"]) == {"1", "2"}
        assert a["r90"]["rec"] is not None and 0.0 <= a["r90"]["fa"] <= 1.0
        assert sum(ch[k]["2/2"]["by_type"][t]["n"] for t, _ in V.TYPES) == len(cands)
    assert all(v is not None and v >= 0.0 for v in [ch[k]["ms"] for k in ch])
    fr = out["folds"]["frames"]
    assert sorted(fr[0] + fr[1]) == sorted(hold) and fr[0] and fr[1] and max(fr[0]) < min(fr[1])
    assert all(c["fold"] == (1 if c["frame"] in fr[0] else 2) for c in cands)
    bad = [x for x in (";", " -- ", "%%", "real time", "real-time") if x in text] + [x for x in lines if ":" in x and not x.startswith(" note")]
    assert not bad and max(len(x) for x in lines) <= V.W and not out["notes"], (bad, max(len(x) for x in lines), out["notes"])
    assert "(d) minus (b)" in text and "(d) colour minus (e) colour" in text and "over queries" in text
    for key, label, _ in V.CHECKS:
        assert label in text, label
    assert "fold 1 frames 000006 to 000008" in text and "fast outputs 18/18" in text and "1/1 resolution" in text
    return ctx, out


def check_options(ctx, out):
    gen = dict(ctx, verify_checks=[], out_dir=tempfile.mkdtemp())
    again = V.run_verify(gen, None)
    a, b = out["candidates"], again["candidates"]
    assert len(a) == len(b) and all(x["source"] == y["source"] and np.allclose(x["pose"], y["pose"], atol=1e-9) and x["ok2"] == y["ok2"] for x, y in zip(a, b))
    assert not again["checks"] and all(not c["scores"] for c in b)
    other = V.run_verify(dict(gen, verify_seed=3), None)
    moved = [x for x, y in zip(a, other["candidates"]) if x["type"] == "perturbed" and not np.allclose(x["pose"], y["pose"])]
    assert moved, "another seed draws other perturbations"
    names = ctx["tests"][:3]
    sub = V.run_verify(dict(gen, verify_cap=6), dict(names=names))
    assert sub["queries"] == 3 and sub["names"] == names and all(sum(1 for c in sub["candidates"] if c["name"] == n) == 6 for n in names)
    assert all(sum(1 for c in sub["candidates"] if c["name"] == n and c["type"] == "perturbed") >= 1 for n in names)
    assert any("one block of frames" in x for x in sub["notes"]) and not sub["uncertainty"]
    one = V.run_verify(dict(ctx, verify_checks=["b", "d"], verify_cap=5, verify_methods=False, out_dir=tempfile.mkdtemp()), dict(names=names[:1]))
    t1 = "\n".join(V.report_verify(one))
    assert one["queries"] == 1 and one["n"] == 5 and one["folds"]["in_sample"] and "thresholds are in sample" in t1 and "in sample" in one["notes"][-1]
    assert one["uncertainty"]["2/2"]["queries"] == 1 and one["checks"]["d"]["2/2"]["thresholds"]["2"]["in_sample"]
    none = V.run_verify(dict(ctx, ref_poses=None, out_dir=tempfile.mkdtemp()), None)
    assert none["n"] == 0 and none["queries"] == 0 and "no candidate poses were scored (0 queries)" in "\n".join(V.report_verify(none))
    import copy
    blind = copy.copy(ctx["loc"])
    blind.match = lambda *a, **k: (np.zeros((0, 2)), np.zeros((0, 3)), 0, 0.0, 0.0)
    nm = V.run_verify(dict(ctx, loc=blind, verify_checks=["a_dec"], verify_cap=16, verify_far=0, verify_retrieved=1, verify_methods=False,
                           out_dir=tempfile.mkdtemp()), dict(names=[ctx["tests"][0], ctx["tests"][3]]))
    a = nm["checks"]["a_dec"]["2/2"]["all"]
    assert nm["checks"]["a_dec"]["cover"] == 0.0 and a["auroc"] == 0.5 and a["r90"]["fa"] == 0.0 and a["r90"]["rec"] == 0.0 and not nm["notes"]
    assert any(c["source"].endswith("no PnP pose") for c in nm["candidates"])
    print("edge cases: one query (thresholds in sample, flagged), no reference poses (nothing scored), no matches (check (a) undefined, never accepted)")

    def empty(pose, med=None, t=None, alpha=False, which="underwater", depth=False):
        img, al, z = T.render(pose, med, t, alpha, which, True)
        return img, torch.zeros_like(al), z

    hole = V.run_verify(dict(ctx, render=empty, verify_checks=["d", "e"], verify_cap=6, verify_methods=False, out_dir=tempfile.mkdtemp()), None)
    d = hole["checks"]["d"]
    assert d["cover"] == 0.0 and hole["checks"]["e"]["cover"] == 0.0 and d["2/2"]["all"]["auroc"] == 0.5 and d["2/2"]["all"]["r90"]["fa"] == 0.0
    assert d["2/2"]["all"]["r90"]["rec"] == 0.0 and d["2/2"]["all"]["r95"]["tp"] + d["2/2"]["all"]["r95"]["fp"] == 0
    nomap = V.run_verify(dict(ctx, map_images=None, verify_checks=["b", "c", "h", "b_rgb", "c_rgb", "a_tok"], verify_cap=5, verify_methods=False,
                              out_dir=tempfile.mkdtemp()), None)
    assert set(nomap["checks"]) == {"a_tok"} and any("splat checks are left out" in x for x in nomap["notes"])
    small = dict(ctx, verify_checks=["b"], verify_cap=3, verify_methods=False, out_dir=tempfile.mkdtemp())
    listed = V.run_verify(dict(small, map_images=list(ctx["map_images"])), dict(names=names[:1]))
    stacked = V.run_verify(small, dict(names=names[:1]))
    assert listed["checks"]["b"]["cover"] == 1.0 and [c["scores"]["b"] for c in listed["candidates"]] == [c["scores"]["b"] for c in stacked["candidates"]]
    lines = V.report_verify(nomap)
    assert "(b) splat" not in "\n".join(lines) and "(a) geometry, token matches" in "\n".join(lines)
    assert V.report_verify(dict(n=0, queries=0, notes=["x"]))[-2] == " note        x"
    every = V.run_verify(dict(ctx, verify_checks=None, verify_cap=4, verify_methods=False, out_dir=tempfile.mkdtemp()), dict(names=ctx["tests"][:2]))
    assert set(every["checks"]) == set(k for k, _, _ in V.CHECKS) and every["n"] == 8 and every["checks"]["f"]["cover"] == 0.0
    assert "2/2" not in every["checks"]["f"] and "(f) check of the fast path" not in "\n".join(V.report_verify(every))
    lc = V.run_verify(dict(ctx, fast_photo="lcn", verify_checks=["g", "h"], verify_cap=4, verify_methods=False, out_dir=tempfile.mkdtemp()),
                      dict(names=[ctx["tests"][0], ctx["tests"][3]]))
    assert lc["checks"]["g"]["cover"] > 0.5 and lc["checks"]["h"]["cover"] > 0.5 and not lc["notes"], lc["checks"]
    print("options: same candidates for the same seed, other perturbations for another seed, the query subset of the benchmark, renders without any "
          "opacity leave (d) and (e) undefined (coverage 0, rejected), no map images leave out the splat checks")


if __name__ == "__main__":
    torch.manual_seed(0)
    check_metrics()
    check_lcn_and_splat()
    ctx, out = check_run()
    check_options(ctx, out)
    print("test_verify ok")
