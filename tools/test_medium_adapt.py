import json
import os
import sys
import tempfile
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import mast3r.utils.path_to_dust3r
import eval_gaussian_map as E
import loc_ff
import loc_geom as G
import loc_medium_adapt as LM
import test_loc as T
from training.physics_splat import background, beta_at

H, W_, FOCAL, PP = T.H, T.W_, T.FOCAL, T.PP
tt = lambda p: torch.tensor(np.asarray(p), dtype=torch.float32)
PHYS = types.SimpleNamespace(coupling=dict(jerlov=False, seathru=True, le=True, beta_max=5.0, z_min=0.1, z_max=50.0, eps=0.01))
LAW = lambda m, z: beta_at(PHYS, m, z)
TRUE = np.array([[1.25, 0.85, 1.15], [1.35, 1.2, 0.85], [0.8, 1.15, 1.1]])
MILD = np.array([[1.04, 0.97, 1.03], [1.12, 1.08, 0.94], [0.97, 1.03, 1.02]])


def medium():
    v, u = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W_), indexing="ij")
    bmap = torch.stack([0.05 + 0.01 * u, 0.30 + 0.03 * v, 0.36 + 0.02 * u * v]).clamp(0.0, 1.0)
    pD = torch.tensor([[0.10, 0.045, 0.035], [-0.05, -0.04, -0.03], [0.06, 0.02, 0.02], [0.0, 0.0, 0.0]])
    pB = torch.tensor([[0.06, 0.03, 0.025], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    return dict(params_D=pD, params_B=pB, B_inf=bmap.mean((1, 2)), B_inf_map=bmap)


MED = medium()


def colour(pose):
    return T.render(tt(pose)).clamp(0.0, 1.0)


def hole():
    a = torch.ones(H, W_)
    a[:, :24] = 0.3
    return a


def underwater(pose, med):
    J = colour(pose)
    z = T.surface_points(tt(pose))[1]
    bD, bB = LAW(med, z.reshape(-1))
    zz = z.reshape(1, -1)
    I = J.reshape(3, -1) * torch.exp(-bD * zz) + background(med, H, W_).reshape(3, -1) * (1.0 - torch.exp(-bB * zz))
    return I.reshape(3, H, W_), J, z


def render(pose, med=None, t=None, alpha=False, which="underwater", depth=False):
    a = hole()
    if which == "restored":
        img, z = colour(pose), T.surface_points(tt(pose))[1]
    else:
        assert med is not None
        img, _, z = underwater(pose, med)
    if depth:
        return img, a, z
    return (img, a) if alpha else img


def parts(pose, num=None):
    I, J, z = underwater(pose, MED)
    return dict(I=I, J=J, alpha=hole(), depth=z, binf=background(MED, H, W_).float(), betas=lambda zz: LAW(MED, zz), level=0)


def query(pose, gen, noise=0.004, f=TRUE):
    I = underwater(pose, LM.apply_factors(MED, f))[0]
    I[:, :, :24] = 0.9
    return (I + noise * torch.randn(3, H, W_, generator=gen)).clamp(0.0, 1.0)


def check_apply():
    z = torch.linspace(1.0, 9.0, 50)
    f = np.array([[1.5, 0.7, 1.0], [2.0, 0.5, 1.2], [0.6, 1.8, 1.0]])
    m = LM.apply_factors(MED, f)
    bD0, bB0 = LAW(MED, z)
    bD1, bB1 = LAW(m, z)
    dD = float((bD1 - torch.tensor(f[1], dtype=torch.float32)[:, None] * bD0).abs().max())
    want_B = torch.minimum(torch.tensor(f[2], dtype=torch.float32)[:, None] * MED["params_B"][0][:, None].expand_as(bD0), bD1)
    dB = float((bB1 - want_B).abs().max())
    dV = float((background(m, H, W_) - torch.tensor(f[0], dtype=torch.float32)[:, None, None] * background(MED, H, W_)).abs().max())
    print("factors applied to the medium: attenuation off by %.1e, backscatter (with the cap at the attenuation) off by %.1e, veiling light off by %.1e" % (dD, dB, dV))
    assert dD < 1e-6 and dB < 1e-6 and dV < 1e-6
    assert torch.equal(MED["params_B"], medium()["params_B"]) and "beta_D_scale" not in MED, "the medium of the map must not change"
    one = LM.apply_factors(MED, np.ones((3, 3)))
    assert all(torch.equal(one[k], MED[k]) for k in MED)


def check_fit():
    gen = torch.Generator().manual_seed(1)
    errs, rows = [], []
    for i in (6, 14, 22):
        true = T.true_pose(i)
        q = query(true, gen)
        for law, name in ((LAW, "law"), (None, "scaled")):
            fit = LM.adapt_medium(q, parts(true), medium=MED, law=law)
            assert fit["ok"] and fit["law"] == (law is not None), fit
            f = np.array(fit["factors"])
            I1 = underwater(true, fit["apply"](MED))[0]
            I2 = underwater(true, LM.apply_factors(MED, TRUE))[0]
            img_err = float((I1 - I2)[:, :, 24:].abs().max())
            rows.append((i, name, f, fit["sig_before"], fit["sig_after"], img_err, fit["steps"], fit["seconds"]))
            if law is not None:
                errs.append((np.abs(f[1] / TRUE[1] - 1.0).max(), np.abs(f[0] * f[2] / (TRUE[0] * TRUE[2]) - 1.0).max(), img_err, fit["sig_after"]))
    for r in rows:
        print("pose %d, %s: veil %s attenuation %s backscatter %s | residual scale %.4f -> %.4f | largest image error %.4f | %d steps, %.0f ms" % (
            r[0], r[1], r[2][0].round(3), r[2][1].round(3), r[2][2].round(3), r[3], r[4], r[5], r[6], 1e3 * r[7]))
    e = np.array(errs)
    print("with the law: attenuation within %.1f %%, veil times backscatter within %.1f %%, image within %.4f, residual scale %.4f (noise 0.004)" % (
        100 * e[:, 0].max(), 100 * e[:, 1].max(), e[:, 2].max(), e[:, 3].max()))
    assert e[:, 0].max() < 0.06 and e[:, 1].max() < 0.08 and e[:, 2].max() < 0.02 and e[:, 3].max() < 0.006, e


def check_bound_and_support():
    gen = torch.Generator().manual_seed(2)
    true = T.true_pose(10)
    big = np.array([[6.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    q = underwater(true, LM.apply_factors(MED, big))[0]
    fit = LM.adapt_medium(q, parts(true), medium=MED, law=LAW, bound=1.5)
    f = np.array(fit["factors"])
    print("veiling light six times brighter in red, bound 1.5: red veil %.3f, red backscatter %.3f, %d factors at the bound, all within [%.3f, %.3f]" % (
        f[0, 0], f[2, 0], fit["at_bound"], f.min(), f.max()))
    assert abs(f[0, 0] - 1.5) < 1e-6 and abs(f[2, 0] - 1.5) < 1e-6 and fit["at_bound"] >= 2 and f.max() <= 1.5 + 1e-9 and f.min() >= 1 / 1.5 - 1e-9
    p = parts(true)
    p["alpha"] = torch.full((H, W_), 0.5)
    none = LM.adapt_medium(query(true, gen), p, medium=MED, law=LAW)
    assert not none["ok"] and none["factors"] == np.ones((3, 3)).tolist() and none["n"] == 0
    same = LM.adapt_medium(underwater(true, MED)[0], parts(true), medium=MED, law=LAW)
    print("query in the water of the map: factors within %.1e of 1" % float(np.abs(np.array(same["factors"]) - 1.0).max()))
    assert np.abs(np.array(same["factors"]) - 1.0).max() < 1e-3
    far = G.perturb(true, 0.25 * T.DEPTH, 8.0, np.random.default_rng(4))
    off = LM.adapt_medium(query(true, gen), parts(far), medium=MED, law=LAW)
    on = LM.adapt_medium(query(true, gen), parts(true), medium=MED, law=LAW)
    print("registration of query and render: %.2f at the true pose (fitted), %.2f a quarter of the depth and 8 deg away (not fitted, factors stay 1)" % (on["reg"], off["reg"]))
    assert on["registered"] and on["ok"] and not off["registered"] and not off["ok"] and off["factors"] == np.ones((3, 3)).tolist()
    print("no supported pixel (opacity 0.5 everywhere): no fit, factors stay 1")


def check_cells_and_prior():
    gen = torch.Generator().manual_seed(5)
    true = T.true_pose(12)
    q = query(true, gen)
    p = parts(true)
    full = LM.samples(q, p)
    a, b = LM.samples(q, p, cells=0), LM.samples(q, p, cells=1)
    assert a["n"] > 32 and b["n"] > 32 and a["n"] + b["n"] == full["n"] and full["level"] == a["level"] == b["level"], (a["n"], b["n"], full["n"])
    m = LM.held_out_mask(p["alpha"], full["level"] - p["level"], cells=0)
    blk = 4 * 2 ** (full["level"] - p["level"])
    on = LM.cell_parity(H, W_, blk, torch.device("cpu")) == 0
    assert not bool((m & on).any()) and bool(m.any()) and not bool((m & (p["alpha"] <= 0.9)).any())
    half = LM.adapt_medium(q, p, medium=MED, law=LAW, cells=0)
    f = np.array(half["factors"])
    assert half["ok"] and half["cells"] == 0 and np.abs(f[1] / TRUE[1] - 1.0).max() < 0.08, f
    I1 = underwater(true, half["apply"](MED))[0]
    assert LM.masked_psnr(I1, q, m) > LM.masked_psnr(p["I"], q, m) + 5.0
    pr = 0.5
    fit = LM.adapt_medium(q, p, medium=MED, law=LAW, prior=pr, iters=60, tol=1e-10)
    betas_of, _ = LM.water_law(p, MED, LAW)
    m0 = LM.per_pixel(full, betas_of, np.ones((3, 3)))
    norm = 1.0 / (full["n"] * LM.mad_scale(full["q"] - full["i"]) ** 2)
    res = lambda y: full["q"] - (full["i"] + LM.per_pixel(full, betas_of, np.exp(y)) - m0)
    x = np.array(fit["log"])
    c = 1.345 * LM.mad_scale(res(x))
    cost = lambda y: float(LM.huber(res(y), c).sum()) * norm + pr * float((y ** 2).sum())
    c0, worst = cost(x), 0.0
    for k in range(3):
        for ch in range(3):
            for d in (-0.01, 0.01):
                e = np.zeros((3, 3))
                e[k, ch] = d
                worst = max(worst, c0 - cost(x + e))
    print("fit on every other block, scored on the others: attenuation within %.1f %%, PSNR %.2f against %.2f dB with the map water | strong prior: "
          "the fit is a minimum of its cost (largest drop next to it %.1e)" % (100 * np.abs(f[1] / TRUE[1] - 1.0).max(), LM.masked_psnr(I1, q, m),
                                                                              LM.masked_psnr(p["I"], q, m), worst))
    assert worst < 2e-5, worst


def check_gain():
    true = T.true_pose(9)
    p = parts(true)
    g0, o0 = np.array([1.3, 0.8, 1.1]), np.array([0.05, -0.02, 0.01])
    q = (torch.tensor(g0, dtype=torch.float32)[:, None, None] * p["I"] + torch.tensor(o0, dtype=torch.float32)[:, None, None])
    q[:, :, :24] = 0.9
    fit = LM.fit_gain(q, p)
    assert fit["ok"] and np.abs(np.array(fit["gain"]) - g0).max() < 0.02 and np.abs(np.array(fit["offset"]) - o0).max() < 0.01, fit
    out = LM.apply_gain(p["I"], fit)
    assert float((out - q.clamp(0, 1))[:, :, 24:].abs().max()) < 0.02
    assert LM.apply_gain(p["I"], None) is p["I"] and LM.apply_gain(p["I"], dict(ok=False)) is p["I"]
    far = G.perturb(true, 0.25 * T.DEPTH, 8.0, np.random.default_rng(4))
    off = LM.fit_gain(q, parts(far))
    assert not off["ok"] and not off["registered"] and off["gain"] == [1.0, 1.0, 1.0]
    lo = dict(p, alpha=torch.full((H, W_), 0.5))
    assert not LM.fit_gain(q, lo)["ok"]
    print("control gain fit: gain %s offset %s recovered, not fitted when the render is not registered or not supported" % (
        np.round(fit["gain"], 3), np.round(fit["offset"], 3)))


def scene():
    nums = list(range(2, 41))
    hold = [n for k, n in enumerate(nums) if 4 <= k % 12 <= 6]
    mapf = [n for n in nums if n not in hold]
    truth = {i: T.true_pose(i) for i in nums}
    with torch.no_grad():
        imgs = torch.stack([underwater(truth[i], MED)[0] for i in mapf])
        depth = torch.stack([T.surface_points(tt(truth[i]))[1] for i in mapf])
    c2w = torch.stack([tt(truth[i]) for i in mapf])
    intr = torch.tensor([[FOCAL, 0.0, PP[0] - 0.5], [0.0, FOCAL, PP[1] - 0.5], [0.0, 0.0, 1.0]]).expand(len(mapf), 3, 3).contiguous()
    net = T.StandIn()
    loc = loc_ff.Localizer(net, loc_ff.encode_frames(net, imgs), depth, torch.full_like(depth, 2.0), intr, c2w, FOCAL, PP, 1.0, torch.device("cpu"),
                           subsample=4, reproj=2.5)
    return loc, hold, mapf, truth


def check_bench():
    import bench_medium as BM
    loc, hold, mapf, truth = scene()
    name = lambda i: "%06d.jpg" % i
    gen = torch.Generator().manual_seed(7)
    queries = {}

    def load(nm):
        n = E.fnum(nm)
        if n not in queries:
            with torch.no_grad():
                queries[n] = query(truth[n], gen, noise=0.002, f=MILD)
        return queries[n]

    def score(img, gt):
        mae = float((img.clamp(0, 1) - gt)[:, :, 24:].abs().mean())
        return [float(-10.0 * torch.log10(((img.clamp(0, 1) - gt)[:, :, 24:] ** 2).mean() + 1e-12)), 1.0 - mae, mae]

    d = tempfile.mkdtemp()
    out = BM.run_medium(dict(dev=torch.device("cpu"), tests=[name(i) for i in hold], fps=5.0, warmup=1, map_nums=mapf, load=load,
                             medium=lambda num, pose: dict(MED), render=render, score=score, loc=loc, out_dir=d, ref_poses={name(i): truth[i] for i in hold},
                             water=parts, law=LAW, fast_photo="affine", fast_levels=(2, 1, 0), fast_iters=(6, 4, 3),
                             query_info=[dict(name=name(i), heading_gap_deg=(10.0, 90.0, 170.0)[k % 3]) for k, i in enumerate(hold[1:])]), res={})
    text = BM.report_medium(out)
    print(text)
    labels = [r["label"] for r in out["rows"]]
    assert labels == list(BM.ROWS), labels
    assert all(len(x) <= 38 and ":" not in x for x in labels)
    assert not out["notes"] or all("OpenCV" in x for x in out["notes"]), out["notes"]
    row = lambda label: next(r for r in out["rows"] if r["label"] == label)
    ad, src = row("coarse, water adapted every pass"), row("coarse, map water")
    assert ad["adapt_s"] > 0.0 and src["adapt_s"] == 0.0 and ad["s_mean"] > src["s_mean"] * 0.8
    ctl = row("control, render gain fit every pass")
    assert ctl["family"] == "control" and ctl["adapt_s"] > 0.0 and ctl["passes"] >= 1
    assert out["gain"]["fitted"] > 0 and all(len(x) == 3 for x in (out["gain"]["gain_median"], out["gain"]["offset_median"]))
    nr = row("no render, match the map frame")
    assert nr["family"] == "ours" and nr["s_mean"] > 0.0 and np.isfinite(nr["t"])
    ph = out["photometric"]
    assert ph["adapted"]["psnr"] > ph["source"]["psnr"] and ph["ref_adapted"]["psnr"] > ph["ref_source"]["psnr"] + 3.0, ph
    assert ph["ref_held_adapted"]["held_psnr"] > ph["ref_held_source"]["held_psnr"] + 1.0, ph
    assert 0.1 < ph["held_source"]["held_share"] < 0.6 and "psnr" not in ph["held_adapted"] and ph["held_gain"]["held_psnr"] is not None, ph
    assert "held out pixels" in text and "in sample" in text and "not tuned on these queries" in text
    st = out["strata"]
    n3 = len(hold) - 1
    assert st["unknown"] == 1 and sum(st["counts"].values()) == n3 and [r["label"] for r in st["rows"]] == labels and "By the heading gap" in text, st
    assert sum(v["n"] for v in st["rows"][2]["bands"].values()) == n3
    up = np.array([out["factors_upper"][k]["median"] for k in LM.NAMES])
    assert np.abs(up[1] / MILD[1] - 1.0).max() < 0.02 and np.abs(up[0] * up[2] / (MILD[0] * MILD[2]) - 1.0).max() < 0.03, up
    assert ad["t"] < src["t"] and ad["acc"][0] >= src["acc"][0], (ad, src)
    assert row("upper bound, water adapted at ref pose")["px"] < 1.0
    assert out["registered"]["n"] == len(hold) and 0 < out["registered"]["any"] <= len(hold)
    saved = json.load(open(os.path.join(d, "medium.json")))
    assert [r["label"] for r in saved["rows"]] == labels and len(saved["per_query"]["coarse, water adapted every pass"]) == len(hold)
    assert any(q["factors"] is not None for q in saved["per_query"]["coarse, water adapted every pass"])
    assert any(q["gain"] is not None for q in saved["per_query"]["control, render gain fit every pass"])
    assert all(q["factors"] is None and q["gain"] is None for q in saved["per_query"]["coarse, map water"])
    prose = "\n".join(x for x in text.splitlines() if set(x.strip()) - {"-", "="})
    for bad in ("Beta", "beta kernel", ";", "--", "real time", "real-time"):
        assert bad not in prose, bad
    none = BM.run_medium(dict(dev=torch.device("cpu"), tests=[name(i) for i in hold], fps=5.0, map_nums=mapf, load=load, medium=lambda num, pose: dict(MED),
                              render=render, loc=loc, out_dir=d, ref_poses={name(i): truth[i] for i in hold}, water=None))
    assert not none["rows"] and none["notes"]
    got = {}
    noref = BM.run_medium(dict(dev=torch.device("cpu"), tests=[name(i) for i in hold], fps=5.0, map_nums=mapf, load=load, medium=lambda num, pose: dict(MED),
                               render=render, loc=loc, out_dir=d, ref_poses=None, water=parts, law=LAW), res=got)
    assert not noref["rows"] and noref["frames"] == 0 and noref["skipped"] == len(hold) and "no query has a reference pose" in noref["notes"][0]
    assert got["medium"]["notes"] == noref["notes"] and "no query has a reference pose" in BM.report_medium(noref)
    calls = {True: [], False: []}
    runs = {}
    for carry in (True, False):
        def counted(num, pose, c=carry):
            calls[c].append(num)
            return dict(MED)
        wat = (lambda p, n: dict(parts(p), medium=MED)) if carry else parts
        runs[carry] = BM.run_medium(dict(dev=torch.device("cpu"), tests=[name(hold[0])], fps=5.0, map_nums=mapf, load=load, medium=counted, render=render,
                                         score=score, loc=loc, out_dir=d, ref_poses={name(hold[0]): truth[hold[0]]}, water=wat, law=LAW, fast=False,
                                         units_per_m=2.0, medium_bound=2.0))
    one = runs[True]
    assert 0 < len(calls[True]) < len(calls[False]), (len(calls[True]), len(calls[False]))
    assert [(r["label"], round(r["t"], 6)) for r in runs[True]["rows"]] == [(r["label"], round(r["t"], 6)) for r in runs[False]["rows"]]
    assert [r["label"] for r in one["rows"]] == list(BM.ROWS[:7]) and all(r["n"] == 1 and r["s_mean"] > 0.0 for r in one["rows"][1:]), one["rows"]
    t1 = BM.report_medium(one)
    assert abs(one["depth_m"] - float(np.median(loc.depths)) / 2.0) < 1e-9 and one["depth_m_source"] == "reference" and "from the similarity" in t1
    assert not one["defaults"] and "set on the command line" in t1
    print("no reference: no rows and a note | one query: every coarse row, median depth %.3f m from the reference similarity, the fits take the medium "
          "of the render parts (%d medium calls against %d)" % (one["depth_m"], len(calls[True]), len(calls[False])))
    print("benchmark rows present, adapted water raises the PSNR of the render from %.2f to %.2f dB at the estimated pose" % (ph["source"]["psnr"], ph["adapted"]["psnr"]))


def check_sealoc_tools():
    import zipfile
    import make_sealoc_scene as S
    d = tempfile.mkdtemp()
    zp = os.path.join(d, "a.zip")
    rng = np.random.default_rng(3)
    payload = {"x/PR_1_LC16.tiff": rng.integers(0, 255, 70000, dtype=np.uint8).tobytes(), "x/PR_2_LC16.tiff": b"abc" * 5000}
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr(zipfile.ZipInfo("x/"), b"")
        with z.open("x/PR_1_LC16.tiff", "w", force_zip64=True) as f:
            f.write(payload["x/PR_1_LC16.tiff"])
        z.writestr("x/PR_2_LC16.tiff", payload["x/PR_2_LC16.tiff"], compress_type=zipfile.ZIP_DEFLATED)
    blob = open(zp, "rb").read()
    calls = []

    def fake(url, rng=None, tries=5):
        calls.append(rng)
        if rng is None:
            return blob, None
        if rng.startswith("-"):
            a = max(0, len(blob) - int(rng[1:]))
            return blob[a:], "bytes %d-%d/%d" % (a, len(blob) - 1, len(blob))
        a, b = [int(x) for x in rng.split("-")]
        return blob[a:b + 1], "bytes %d-%d/%d" % (a, b, len(blob))

    real = S.fetch
    S.fetch = fake
    try:
        cd = S.central_directory("u", os.path.join(d, "cd.json"))
        got = {k: S.member("u", m) for k, m in cd["members"].items() if k.endswith(".tiff")}
    finally:
        S.fetch = real
    assert got == {os.path.basename(k): v for k, v in payload.items()} and cd["total"] == len(blob), cd
    enu = S.Enu(113.95, -28.81, -18.0)
    e = enu(113.95, -28.81 + 1e-5, -18.0)
    assert abs(e[1] - 1.1086) < 2e-3 and abs(e[0]) < 1e-6 and abs(e[2]) < 1e-3, e
    cv = dict(world=True, sy=-1, off=0, swap=False, sp=1, sr=1)
    R = S.rotation(90.0, 0.0, 0.0, cv)
    assert np.allclose(R[:, 2], [0, 0, -1]) and np.allclose(R[:, 0], [0, -1, 0]) and np.allclose(R[:, 1], [-1, 0, 0]), R
    cal = dict(width=1360.0, height=1024.0, fx=1738.0, fy=1738.0, cx=680.0, cy=512.0, k1=0.16, k2=0.72, k3=0.0, p1=0.001, p2=0.002)
    m = np.eye(4)
    m[:3, :3] = S.rotation(37.0, 1.5, -2.0, cv)
    m[:3, 3] = [1.0, 2.0, -18.0]
    rays = S.corner_rays(cal)
    depth = np.array([2.1, 2.3, 1.9, 2.0])
    pts = m[:3, 3] + (m[:3, :3] @ (rays / rays[:, 2:3] * depth[:, None]).T).T
    assert S.corner_error(m, pts, rays) < 1e-6
    wrong = np.eye(4)
    wrong[:3, :3], wrong[:3, 3] = S.rotation(37.0, 1.5, -2.0, dict(cv, world=False, sr=-1)), m[:3, 3]
    assert S.corner_error(wrong, pts, rays) > 1.0
    assert np.allclose(S.corners("[(113.9, -28.8, -20.1), (113.91, -28.81, -20.2)]"), [[113.9, -28.8, -20.1], [113.91, -28.81, -20.2]])
    for bad in ("__import__('os').getcwd()", "[(1, 2, 3)] + [open('x')]"):
        try:
            S.corners(bad)
            raise AssertionError("code in a footprint was run")
        except ValueError:
            pass
    md = os.path.join(d, "meta")
    os.makedirs(md)
    open(os.path.join(md, "camera_calibrations.csv"), "w").write(
        "calibration_id,width,height,fx,fy,cx,cy,k1,k2,k3,p1,p2\n1,1360,1024,1738,1738,695,492,0.16,0.72,0,0.003,0.003\n"
        "3,1360,1024,1708,1708,691,499,0.15,0.69,0,0.003,0.003\n4,1360,1024,1711,1711,695,505,0.16,0.66,0,0.003,0.005\n"
        "5,1360,1024,1698,1698,666,526,0.14,0.67,0,-0.001,-0.002\n")
    open(os.path.join(md, "camera_sensors.csv"), "w").write(
        "sensor_id,sensor_label,sensor_width,sensor_height,sensor_location,sensor_rotation,sensor_calibration_id,camera_group_id,camera_bundle_id,master_sensor_id\n"
        "1,stereo_left,1360,1024,x,x,1,1,1,\n3,stereo_left,1360,1024,x,x,3,2,1,\n4,stereo_right,1360,1024,x,x,4,2,1,3.0\n5,stereo_left,1360,1024,x,x,5,3,1,\n")
    open(os.path.join(md, "camera_groups.csv"), "w").write(
        "camera_group_id,camera_group_label,meta,camera_bundle_id\n1,s_2010a,m,1\n2,s_2011a,m,1\n3,s_2012a,m,1\n")
    cals = S.calibrations(d, "s", ("2011a", "2012a"))
    assert cals["2011a"][1] == "3" and cals["2012a"][1] == "5" and cals["2012a"][0]["cx"] == 666.0, cals
    try:
        S.calibrations(d, "s", ("2013a",))
        raise AssertionError("a survey without a calibration was accepted")
    except SystemExit:
        pass
    sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert abs(S.iou(sq, sq) - 1.0) < 1e-9 and abs(S.iou(sq, sq + [0.5, 0.0]) - 1.0 / 3.0) < 1e-9 and S.iou(sq, sq + [3.0, 0.0]) == 0.0
    print("SEALOC tools: zip members read by byte ranges (%d range requests, ZIP64 and deflate), local east north up, orientation convention, footprint overlap" % len(calls))
    try:
        import cv2
    except Exception as e:
        print("geometry of the scene images not checked here (OpenCV unusable: %s)" % type(e).__name__)
        return
    two = dict(a=cal, b=dict(cal, fx=1698.0, fy=1698.0, cx=650.0, cy=530.0, k1=0.14, k2=0.67, p1=-0.001, p2=-0.002))
    geo = S.geometry(two, 512)
    assert (geo["w"], geo["h"]) == (512, 384) and geo["h"] % 16 == 0 and geo["zoom"] >= 1.0
    for key in two:
        img = S.prepare(np.full((1024, 1360, 3), 0.5, np.float32), geo, key)
        assert img.shape == (384, 512, 3) and float(img.min()) > 0.49, (key, float(img.min()))
        assert S.covers(*S.camera(two[key]), geo["K1"], 1360, 1024)
    one = S.geometry(dict(b=two["b"]), 512)
    alone_a = S.geometry(dict(a=cal), 512)["K1"]
    assert one["K1"][0, 0] <= geo["K1"][0, 0] + 1e-6 and alone_a[0, 0] <= geo["K1"][0, 0] + 1e-6 and not S.covers(*S.camera(two["b"]), alone_a, 1360, 1024)
    a, b = geo["K1"], one["K1"]
    print("scene images 512 x 384, each survey undistorted with its own calibration into one camera without an empty border (fx %.1f, %.1f %% zoom, alone %.1f)" % (
        a[0, 0], 100.0 * (geo["zoom"] - 1.0), b[0, 0]))


if __name__ == "__main__":
    torch.manual_seed(0)
    check_sealoc_tools()
    check_apply()
    check_fit()
    check_bound_and_support()
    check_cells_and_prior()
    check_gain()
    check_bench()
    print("test_medium_adapt ok")
