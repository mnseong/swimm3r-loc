import os
import sys
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import mast3r.utils.path_to_dust3r  # noqa: F401
import eval_gaussian_map as E
import loc_fast as LF
import loc_ff
import loc_geom as G
import test_loc as T

H, W_, FOCAL, DEPTH, PP = T.H, T.W_, T.FOCAL, T.DEPTH, T.PP
tt = lambda p: torch.tensor(np.asarray(p), dtype=torch.float32)


def textured(pose):
    p, z = T.surface_points(tt(pose))
    x, y = p[..., 0], p[..., 1]
    base = 0.45 + 0.12 * torch.sin(1.3 * x + 0.4) * torch.cos(1.1 * y) + 0.09 * torch.sin(4.0 * x + 2.0 * y) + 0.07 * torch.cos(9.0 * x - 7.0 * y) \
        + 0.05 * torch.sin(23.0 * x + 3.0) * torch.sin(19.0 * y)
    img = torch.stack([base + 0.05 * torch.sin(6.0 * y), base, base - 0.05 * torch.cos(5.0 * x)]).clamp(0.0, 1.0)
    return img, torch.ones(H, W_), z


def shuffle(x, r):
    b, c, h, w = x.shape
    return x.reshape(b, c // (r * r), r, r, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // (r * r), h * r, w * r)


def check_lean_head():
    from mast3r.catmlp_dpt_head import mast3r_head_factory
    import torch.nn.functional as F
    torch.manual_seed(0)
    probe = torch.arange(32.0).reshape(1, 8, 2, 2)
    assert bool((F.pixel_shuffle(probe, 2) == shuffle(probe, 2)).all())
    if not torch.cuda.is_available():
        F.pixel_shuffle = shuffle
    net = types.SimpleNamespace(enc_embed_dim=32, dec_embed_dim=24, dec_depth=12, depth_mode=("exp", -float("inf"), float("inf")),
                                conf_mode=("exp", 1, float("inf")), desc_mode=("norm",), two_confs=True, desc_conf_mode=("exp", 0, float("inf")),
                                patch_embed=types.SimpleNamespace(patch_size=(16, 16)))
    net.downstream_head1 = mast3r_head_factory("catmlp+dpt", "pts3d+desc24", net, has_conf=True).eval()
    net.downstream_head2 = mast3r_head_factory("catmlp+dpt", "pts3d+desc24", net, has_conf=True).eval()
    net._downstream_head = lambda num, dec, shape: (net.downstream_head1 if num == 1 else net.downstream_head2)(dec, (int(shape[0, 0]), int(shape[0, 1])))
    h, w = 64, 96
    loc = object.__new__(loc_ff.Localizer)
    loc.net, loc.h, loc.w, loc.name, loc.lean_ok, loc.lean_diff = net, h, w, "test", None, None
    sh = torch.tensor([[h, w]]).expand(2, 2)
    dec = lambda: [torch.randn(2, 24, 32)] + [torch.randn(2, 24, 24) for _ in range(12)]
    d1, d2 = dec(), dec()
    with torch.no_grad():
        full = loc.descriptors(d1, d2, sh, True)
        lean = loc.descriptors(d1, d2, sh, True)
        plain = loc.descriptors(d1, d2, sh, False)
    diff = max(float((a - b).abs().max()) for a, b in zip(full, lean))
    print("descriptors without the pointmap branch against the full head of MASt3R: largest difference %.1e, shape %s" % (diff, tuple(lean[0].shape)))
    assert loc.lean_ok and loc.lean_diff < 1e-5 and diff < 1e-5 and tuple(lean[0].shape) == (2, h, w, 24)
    assert max(float((a - b).abs().max()) for a, b in zip(full, plain)) == 0.0
    tall = object.__new__(loc_ff.Localizer)
    tall.net, tall.h, tall.w, tall.name, tall.lean_ok, tall.lean_diff = types.SimpleNamespace(_downstream_head=lambda num, d, s: dict(desc=d[0])), 96, 64, "tall", None, None
    tall.descriptors(d1, d2, sh, True)
    assert tall.lean_ok is False


def check_direct_align():
    K, hw = G.intrinsics(FOCAL, PP), (H, W_)
    gen = torch.Generator().manual_seed(1)
    v, u = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W_, dtype=torch.float32), indexing="ij")
    rows = {}
    for photo in ("lcn", "affine"):
        errs, starts, secs = [], [], []
        for i in (6, 12, 18, 24, 30):
            true = T.true_pose(i)
            with torch.no_grad():
                clean = textured(true)[0]
                query = (0.8 * clean + 0.06 + 0.05 * (u / W_)[None] + 0.01 * torch.randn(3, H, W_, generator=gen)).clamp(0.0, 1.0)
                pts = T.surface_points(tt(true))[0].reshape(-1, 3)[::37].double().numpy()
            for seed, (tr, rd) in enumerate(((0.01, 0.3), (0.02, 0.8))):
                start = G.perturb(true, tr * DEPTH, rd, np.random.default_rng(10 * i + seed))
                r = LF.direct_align(query, start, K, textured, levels=(2, 1, 0), iters=(6, 4, 3), photo=photo)
                assert r["ok"] and r["corr"] is not None, r
                starts.append(G.pixel_shift(K, true, start, pts, hw))
                errs.append(G.pixel_shift(K, true, r["pose"], pts, hw))
                secs.append(r["stages"]["align"])
        rows[photo] = (np.median(starts), np.median(errs), np.max(errs), np.mean(secs))
        print("direct alignment (%s) of a query at 0.8 of the brightness with an offset, a ramp and noise: start %.2f px -> %.3f px (largest %.3f) in %.0f ms on the CPU" % (
            photo, *rows[photo][:3], 1e3 * rows[photo][3]))
    assert rows["lcn"][1] < 0.15 and rows["lcn"][2] < 0.5 and rows["affine"][1] < 0.3, rows
    far = LF.direct_align(textured(T.true_pose(12))[0], G.perturb(T.true_pose(12), 0.0, 0.0, np.random.default_rng(0)), K, lambda p: (torch.zeros(3, H, W_), torch.zeros(H, W_), torch.zeros(H, W_)))
    assert not far["ok"] and far["steps"] == 0


def scene():
    nums = list(range(2, 41))
    hold = [n for k, n in enumerate(nums) if 4 <= k % 12 <= 6]
    mapf = [n for n in nums if n not in hold]
    truth = {i: T.true_pose(i) for i in nums}
    with torch.no_grad():
        imgs = torch.stack([T.render(tt(truth[i])) for i in mapf])
        depth = torch.stack([T.surface_points(tt(truth[i]))[1] for i in mapf])
    c2w = torch.stack([tt(truth[i]) for i in mapf])
    intr = torch.tensor([[FOCAL, 0.0, PP[0] - 0.5], [0.0, FOCAL, PP[1] - 0.5], [0.0, 0.0, 1.0]]).expand(len(mapf), 3, 3).contiguous()
    net = T.StandIn()
    loc = loc_ff.Localizer(net, loc_ff.encode_frames(net, imgs), depth, torch.full_like(depth, 2.0), intr, c2w, FOCAL, PP, 1.0, torch.device("cpu"), subsample=4, reproj=2.5)
    return loc, hold, mapf, truth


def check_fast():
    loc, hold, mapf, truth = scene()
    hw = (H, W_)
    gen = torch.Generator().manual_seed(3)
    ps, gh, gw = LF.token_grid(loc)
    assert (ps, gh, gw) == (16, H // 16, W_ // 16)
    render_at = lambda p, i: T.render(tt(p), depth=True)
    tok, fast, base, tier = [], [], [], []
    for n in hold:
        with torch.no_grad():
            img = (T.render(tt(truth[n])) + 0.004 * torch.randn(3, H, W_, generator=gen)).clamp(0.0, 1.0)
            pts = T.surface_points(tt(truth[n]))[0].reshape(-1, 3)[::37].double().numpy()
        r = LF.localize_fast(loc, img, render_at, topk=2, photo="affine", levels=(2, 1, 0), iters=(6, 4, 3))
        assert set(r["stages"]) == {"encoder", "tokens", "pnp", "render", "align"} and "fq" in r
        tok.append(G.pixel_shift(loc.K, truth[n], r["coarse"], pts, hw))
        fast.append(G.pixel_shift(loc.K, truth[n], r["pose"], pts, hw))
        b = loc.localize(img, topk=1)
        base.append(G.pixel_shift(loc.K, truth[n], b["pose"], pts, hw))
        ok = LF.localize_tiered(loc, img, render_at, topk=2, photo="affine", levels=(2, 1, 0), iters=(6, 4, 3))
        hard = LF.localize_tiered(loc, img, render_at, check=dict(min_corr=2.0), topk=2, photo="affine", levels=(2, 1, 0), iters=(6, 4, 3))
        assert "fq" not in ok and hard["fallback"] and any(k.startswith("fallback_") for k in hard["stages"]) and hard["seconds"] > ok["seconds"] * 0.5
        tier.append((ok["fallback"], G.pixel_shift(loc.K, truth[n], ok["pose"], pts, hw), G.pixel_shift(loc.K, truth[n], hard["pose"], pts, hw)))
    tier = np.array(tier, dtype=np.float64)
    print("no decoder: token matches and PnP %.2f px, after the direct alignment %.2f px (largest %.2f) | matching with the decoder %.2f px | with the check "
          "%.0f %% of the queries fall back, %.2f px; every query forced to fall back %.2f px" % (
              np.median(tok), np.median(fast), np.max(fast), np.median(base), 100 * tier[:, 0].mean(), np.median(tier[:, 1]), np.median(tier[:, 2])))
    assert np.median(tok) < 8.0 and np.median(fast) < 0.6 and np.max(fast) < 2.0 and np.median(fast) < np.median(tok), (tok, fast)
    assert np.median(tier[:, 1]) < 0.6 and np.median(tier[:, 2]) < 1.0


def check_anchors():
    loc, hold, mapf, truth = scene()
    hw = (H, W_)
    gen = torch.Generator().manual_seed(7)
    A = LF.TokenAnchors(loc)
    A2 = LF.TokenAnchors(loc, cell=0.5, min_views=2)
    A16 = LF.TokenAnchors(loc, dim=16)
    assert 64 <= A2.m < A.m < A.n_tokens and A.voxel > 0 and A.dim == loc.tokens.shape[-1] and A.basis is None and A.min_views == 1 and A.cell == 0.25
    assert float((A2.views >= 2).float().mean()) == 1.0 and A2.min_views == 2 and A.seconds > 0.0
    assert A16.dim == 16 and A16.m == A.m and 0.5 < A16.kept <= 1.0 and A16.megabytes() < A.megabytes()
    render_at = lambda p, i: T.render(tt(p), depth=True)
    kw = dict(photo="affine", levels=(2, 1, 0), iters=(6, 4, 3))
    rows = dict(frames=[], first=[], dense=[], final=[], low=[], tier=[], hard=[])
    counts = []
    for n in hold:
        with torch.no_grad():
            img = (T.render(tt(truth[n])) + 0.004 * torch.randn(3, H, W_, generator=gen)).clamp(0.0, 1.0)
            pts = T.surface_points(tt(truth[n]))[0].reshape(-1, 3)[::37].double().numpy()
        px = lambda pose: G.pixel_shift(loc.K, truth[n], pose, pts, hw)
        f = LF.localize_fast(loc, img, render_at, topk=2, **kw)
        a = LF.localize_fast(loc, img, render_at, anchors=A, guided=False, **kw)
        b = LF.localize_fast(loc, img, render_at, anchors=A, **kw)
        g = LF.localize_fast(loc, img, render_at, anchors=A, support=10 ** 6, **kw)
        c = LF.localize_fast(loc, img, render_at, anchors=A16, **kw)
        d = LF.localize_tiered(loc, img, render_at, anchors=A, **kw)
        e = LF.localize_tiered(loc, img, render_at, anchors=A, check=dict(min_corr=2.0), **kw)
        assert set(b["stages"]) == {"encoder", "tokens", "pnp", "render", "align"} and len(b["idx"]) == 1 and b["idx"][0] == loc.near(b["coarse"], 1)[0]
        assert a["seen"] == 0 and a["agree"] == 0 and not a["densified"] and a["matches"] == a["first"] and b["first"] == a["first"]
        assert b["seen"] >= b["agree"] > 0 and (b["densified"] or b["matches"] == b["first"]) and (b["densified"] or np.allclose(b["coarse"], a["coarse"]))
        assert g["densified"] and g["matches"] == g["agree"] > g["first"]
        assert not d["fallback"] and e["fallback"] and e["ok"]
        rows["frames"].append(px(f["coarse"]))
        rows["first"].append(px(a["coarse"]))
        rows["dense"].append(px(g["coarse"]))
        rows["final"].append(px(b["pose"]))
        rows["low"].append(px(c["pose"]))
        rows["tier"].append(px(d["pose"]))
        rows["hard"].append(px(e["pose"]))
        counts.append((a["first"], a["inliers"], g["agree"], g["seen"], g["inliers"]))
    med = {k: float(np.median(v)) for k, v in rows.items()}
    cm = np.median(np.array(counts, dtype=np.float64), 0)
    print("anchors: %d from %d tokens of %d map frames, cell %.2f of a token footprint, %.1f MB (16 dimensions keep %.0f %% of the energy, %.1f MB) | %d anchors seen "
          "by two frames or more at cell 0.5" % (A.m, A.n_tokens, loc.n, A.cell, A.megabytes(), 100 * A16.kept, A16.megabytes(), A2.m))
    print("no decoder: tokens of the 2 retrieved frames and PnP %.2f px (largest %.2f) | no retrieval either, anchors and PnP %.2f px (largest %.2f), %.0f inliers of %.0f "
          "matches | refit on the %.0f agreeing anchors of %.0f in view %.2f px (largest %.2f) | after the direct alignment %.3f px (largest %.3f) | 16 dimensions %.3f px" % (
              med["frames"], np.max(rows["frames"]), med["first"], np.max(rows["first"]), cm[1], cm[0], cm[2], cm[3], med["dense"], np.max(rows["dense"]),
              med["final"], np.max(rows["final"]), med["low"]))
    assert med["first"] < 4.0 and med["dense"] < 4.0 and cm[2] > cm[0] and cm[1] >= 0.5 * cm[0], (med, cm)
    assert med["final"] < 0.6 and np.max(rows["final"]) < 2.0 and med["low"] < 0.6 and med["tier"] < 0.6 and med["hard"] < 1.0, med
    far = np.array(truth[hold[0]], dtype=np.float64)
    far[:3, 3] += 50.0 * far[:3, 2]
    g2d, g3d, seen = A.guided(A.match(loc, loc.encode(T.render(tt(truth[hold[0]]))))[3], far, loc.K)
    assert len(g2d) < 0.2 * A.m
    assert LF.vote_frame(loc, np.zeros((0, 3))) is None
    v = LF.vote_frame(loc, T.surface_points(tt(truth[mapf[5]]))[0].reshape(-1, 3)[::11].numpy())
    assert abs(v - 5) <= 2, v


def check_bench():
    import tempfile
    import bench_gscpe as C
    loc, hold, mapf, truth = scene()
    name = lambda i: "%06d.jpg" % i
    gen = torch.Generator().manual_seed(5)

    def load(nm):
        with torch.no_grad():
            return (T.render(tt(truth[E.fnum(nm)])) + 0.004 * torch.randn(3, H, W_, generator=gen)).clamp(0.0, 1.0)

    def strict(pose, med=None, t=None, alpha=False, which="underwater", depth=False):
        assert which == "restored" or med is not None
        return T.render(pose, med, t, alpha, which, depth)

    def score(img, gt):
        mae = float((img.clamp(0, 1) - gt).abs().mean())
        return [float(-10.0 * torch.log10(((img.clamp(0, 1) - gt) ** 2).mean() + 1e-12)), 1.0 - mae, mae]

    fit = dict(register=True, k=6, grid=2, tol=2.0, n=len(hold), total=len(hold), half_px_med=0.1, half_px_p90=0.2, half_t_med=0.1, half_r_med=0.1,
               inliers_med=100.0, matches_med=200.0)
    res = C.run(dict(dev=torch.device("cpu"), tests=[name(i) for i in hold], fps=5.0, warmup=1, map_nums=mapf, load=load, medium=lambda num, pose: {},
                     render=strict, score=score, loc=loc, loc_public=loc, out_dir=tempfile.mkdtemp(), ref_poses={name(i): truth[i] for i in hold}, ref_fit=fit,
                     ref_source="truth", fast_photo="affine", fast_levels=(2, 1, 0), fast_iters=(6, 4, 3), fast_dim=16))
    text = C.report(res, dict(scene="synthetic scene", map_frames=len(mapf), hw=[H, W_], primitives=0, map_mb=0.0, model="stand in", mapping_seconds=None,
                              weights_mb={"none": 0.0}, reference="registered", split="test : train = 1 : 3.3"))
    print(text[text.index("   method"):text.index(" Runtime")])
    print(text[text.index(" Fast path"):text.index(" Footprint")])
    row = lambda label: next(r for r in res["rows"] if r["label"] == label)
    base, lean, fast, tier = (row("no render: match the map frame, PnP"), row("no render, descriptor branch only"), row("fast: tokens, PnP, direct alignment"),
                              row("fast + check (decoder when it fails)"))
    assert abs(lean["px"] - base["px"]) < 1e-9 and abs(lean["t"] - base["t"]) < 1e-9, (lean, base)
    assert fast["acc"][2] == 100.0 and fast["px"] < 0.6 and tier["px"] < 0.6, (fast, tier)
    assert res["fast"]["fast"]["fallback"] == 0.0 and res["fast"]["tiered"]["stages"]["align"] > 0.0 and res["fast"]["half"] is False
    assert res["fast"]["lean_ok"] is False and "are not available here" in text and "%%" not in text and not res["notes"] or all("OpenCV" in x for x in res["notes"]), res["notes"]
    labels = [r["label"] for r in res["rows"]]
    for want in ("fast, render without the water model", "public MASt3R: tokens, PnP, alignment", "anchors, PnP only, no alignment",
                 "anchors without the second pass", "anchors, render without the water", "anchors in 16 dimensions", "public MASt3R: anchors, alignment"):
        assert want in labels and len(want) <= 38, (want, labels)
    anc, anc_t, pnp = row("anchors: map tokens in 3D, alignment"), row("anchors + check (decoder if it fails)"), row("anchors, PnP only, no alignment")
    assert anc["acc"][2] == 100.0 and anc["px"] < 0.6 and anc_t["px"] < 0.6 and pnp["px"] < 4.0 and pnp["px"] > anc["px"], (anc, anc_t, pnp)
    fa = res["fast"]
    assert fa["anchors"]["n"] > 64 and fa["anchors"]["low_dim"] == 16 and fa["anchor"]["agree"] > 0 and fa["anchor_tiered"]["fallback"] == 0.0 and "anchor_low" in fa
    assert "Anchors: the tokens of all map frames" in text and "anchors in view agree" in text and "anchors + check, half precision" not in labels
    assert "no render, branch only, half prec." not in labels and "fast + check, half precision" not in labels


if __name__ == "__main__":
    check_lean_head()
    check_direct_align()
    check_fast()
    check_anchors()
    check_bench()
    print("test_fast ok")
