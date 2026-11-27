import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import bench_loc as B
import bench_proto as P
import eval_gaussian_map as E
import loc_fast as LF
import loc_geom as G
import loc_warp as Wp
import loc_water as LW

W = 138
OFF_T = (1.0, 2.0, 5.0, 10.0, 20.0)
OFF_R = (1.0, 2.0, 5.0, 10.0)
LABELS = ((2.0, 2.0), (5.0, 5.0))
BAND = 0.01
CHECKS = (("a_tok", "(a) geometry, token matches", False), ("a_dec", "(a) geometry, decoder matches", False),
          ("b", "(b) splat of the nearest map frame", False), ("c", "(c) splat of the 3 nearest map frames", False),
          ("d", "(d) Gaussian map with the medium", True), ("e", "(e) Gaussian map without the medium", True),
          ("b_rgb", "(b) nearest map frame, plain colour", False), ("c_rgb", "(c) 3 nearest map frames, plain colour", False),
          ("d_rgb", "(d) with the medium, plain colour", True), ("e_rgb", "(e) without the medium, plain colour", True),
          ("f", "(f) check of the fast path", True), ("g", "(g) alignment on the Gaussian map", True),
          ("h", "(h) alignment on the nearest map frame", False))
SPLAT = ("b", "c", "h", "b_rgb", "c_rgb")
DRAWN = ("b", "c", "d", "e", "b_rgb", "c_rgb", "d_rgb", "e_rgb")
TYPES = (("method", "outputs"), ("perturbed", "perturbed"), ("wrong", "wrong place"))
SHORT = dict(a_tok="(a) tokens", a_dec="(a) decoder", b="(b)", c="(c)", d="(d)", e="(e)", b_rgb="(b) colour", c_rgb="(c) colour", d_rgb="(d) colour",
             e_rgb="(e) colour", f="(f)", g="(g)", h="(h)")
PAIRS = (("d", "b", "map with the medium against the nearest map frame"), ("d", "c", "map with the medium against the 3 nearest map frames"),
         ("e", "c", "map without the medium against the 3 nearest map frames"), ("d", "e", "map with the medium against without it"),
         ("d_rgb", "c_rgb", "plain colour, map with the medium against 3 map frames"),
         ("d_rgb", "e_rgb", "plain colour, map with the medium against without it"), ("g", "h", "alignment on the map against on the nearest map frame"))


def tag_of(lab):
    return "%g/%g" % lab


def offsets():
    return [(t, 0.0) for t in OFF_T] + [(0.0, r) for r in OFF_R] + [(t, r) for t in OFF_T for r in OFF_R]


def folds(nums, mode="halves", gap=2):
    order = sorted(range(len(nums)), key=lambda k: nums[k])
    blocks = []
    for k in order:
        if blocks and nums[k] - nums[blocks[-1][-1]] <= gap:
            blocks[-1].append(k)
        else:
            blocks.append([k])
    f = np.zeros(len(nums), dtype=np.int64)
    if len(blocks) < 2:
        for k in order[len(order) // 2:]:
            f[k] = 1
        return f, blocks
    if mode == "alternate":
        for b, blk in enumerate(blocks):
            f[blk] = b % 2
    else:
        cum = np.cumsum([len(b) for b in blocks])[:-1]
        cut = int(np.argmin(np.abs(cum - 0.5 * len(nums)))) + 1
        for blk in blocks[cut:]:
            f[blk] = 1
    return f, blocks


def ranks(x):
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(cnt) - 0.5 * (cnt - 1))[inv]


def auroc(s, y):
    s, y = np.asarray(s, dtype=np.float64), np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return None
    r = ranks(s)
    return float((r[y].sum() - 0.5 * n1 * (n1 + 1)) / (n1 * n0))


def keep_threshold(s, y, q):
    p = np.sort(np.asarray(s, dtype=np.float64)[np.asarray(y, dtype=bool)])[::-1]
    if not len(p):
        return float("inf")
    return float(p[max(int(np.ceil(q * len(p) - 1e-9)) - 1, 0)])


def precision_threshold(s, y, q):
    s, y = np.asarray(s, dtype=np.float64), np.asarray(y, dtype=bool)
    best = float("inf")
    for t in np.unique(s[np.isfinite(s)])[::-1]:
        acc = s >= t
        if (acc & y).sum() >= q * acc.sum():
            best = float(t)
    return best


def held_out(s, y, fold):
    acc = {k: np.zeros(len(s), dtype=bool) for k in ("r95", "p95", "r90")}
    th = {}
    for f in (0, 1):
        ev = fold == f
        if not ev.any():
            continue
        cal = ~ev if (~ev).any() else ev
        t = dict(r95=keep_threshold(s[cal], y[cal], 0.95), p95=precision_threshold(s[cal], y[cal], 0.95), r90=keep_threshold(s[cal], y[cal], 0.90))
        for k in t:
            acc[k][ev] = (s[ev] >= t[k]) & np.isfinite(s[ev])
        th[str(f + 1)] = dict(t, in_sample=bool(not (~ev).any()))
    return acc, th


def rates(a, y):
    tp, fp = int((a & y).sum()), int((a & ~y).sum())
    n1, n0 = int(y.sum()), int((~y).sum())
    return dict(prec=tp / float(tp + fp) if tp + fp else None, rec=tp / float(n1) if n1 else None, fa=fp / float(n0) if n0 else None,
                tp=tp, fp=fp, n1=n1, n0=n0)


def stats(s, y, acc, mask):
    out = dict(n=int(mask.sum()), correct=int(y[mask].sum()), auroc=auroc(s[mask], y[mask]))
    for k, a in acc.items():
        out[k] = rates(a[mask], y[mask])
    return out


def interval(v):
    v = [x for x in v if x is not None]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None


def uncertainty(js, scores, y, n, seed, pairs=PAIRS):
    uq = np.unique(js)
    members = [np.nonzero(js == q)[0] for q in uq]
    rng = np.random.default_rng([int(seed), 7])
    draws = {k: [] for k in scores}
    for _ in range(int(n)):
        idx = np.concatenate([members[q] for q in rng.integers(0, len(uq), len(uq))])
        for k, s in scores.items():
            draws[k].append(auroc(s[idx], y[idx]))
    out = dict(n=int(n), queries=int(len(uq)), auroc={k: interval(v) for k, v in draws.items()}, pairs=[])
    for a, b, what in pairs:
        if a not in scores or b not in scores:
            continue
        pa, pb = auroc(scores[a], y), auroc(scores[b], y)
        d = [x - z for x, z in zip(draws[a], draws[b]) if x is not None and z is not None]
        out["pairs"].append(dict(a=a, b=b, what=what, diff=None if pa is None or pb is None else pa - pb, ci=interval(d),
                                 above=float(np.mean(np.array(d) > 0.0)) if d else None))
    return out


def lcn(img, m, k=7, eps=0.02):
    mf = m.float()[None]
    lum = torch.where(m[None], img.float().mean(0, keepdim=True), torch.zeros_like(mf))
    w = LF.box(mf, k)
    ws = w.clamp_min(1e-6)
    mu = LF.box(lum * mf, k) / ws
    var = (LF.box(lum * lum * mf, k) / ws - mu * mu).clamp_min(0.0)
    return (lum - mu) / torch.sqrt(var + eps * eps), w[0]


def lcn_corr(query, img, m, min_cover=0.1):
    m = m.bool()
    qn, w = lcn(query, m)
    an, _ = lcn(img, m)
    sel = m & (w >= 0.5)
    n = int(sel.sum())
    if n < max(64, min_cover * sel.numel()):
        return None
    return LW.centred_corr(qn[:, sel], an[:, sel])


def rgb_corr(query, img, m, min_cover=0.1):
    m = m.bool()
    n = int(m.sum())
    if n < max(64, min_cover * m.numel()):
        return None
    return LW.centred_corr(query.float()[:, m], img.float()[:, m])


def splat(X, col, pose, intr, hs, ws, keep=None):
    fx, fy, cx, cy = intr
    dev = X.device
    c = torch.as_tensor(np.asarray(pose), dtype=torch.float32, device=dev)
    pc = (X.float() - c[:3, 3]) @ c[:3, :3]
    z = pc[:, 2]
    zs = torch.where(z > 1e-6, z, torch.ones_like(z))
    u = torch.round(fx * pc[:, 0] / zs + cx).long()
    v = torch.round(fy * pc[:, 1] / zs + cy).long()
    ok = (z > 1e-6) & (u >= 0) & (u < ws) & (v >= 0) & (v < hs)
    if keep is not None:
        ok = ok & keep
    img = torch.zeros(col.shape[0], hs * ws, device=dev)
    cnt = torch.zeros(hs * ws, device=dev)
    zsum = torch.zeros(hs * ws, device=dev)
    if not bool(ok.any()):
        return img.reshape(-1, hs, ws), cnt.reshape(hs, ws) > 0, zsum.reshape(hs, ws)
    idx, z, col = (v * ws + u)[ok], z[ok], col.float()[:, ok]
    o = torch.argsort(z)
    idx, z, col = idx[o], z[o], col[:, o]
    o = torch.sort(idx, stable=True)[1]
    idx, z, col = idx[o], z[o], col[:, o]
    _, n = torch.unique_consecutive(idx, return_counts=True)
    zmin = torch.repeat_interleave(z[torch.cumsum(n, 0) - n], n)
    front = z <= zmin * (1.0 + BAND)
    img.index_add_(1, idx[front], col[:, front])
    cnt.index_add_(0, idx[front], torch.ones(int(front.sum()), device=dev))
    zsum.index_add_(0, idx[front], z[front])
    return (img / cnt.clamp_min(1.0)).reshape(-1, hs, ws), (cnt > 0).reshape(hs, ws), (zsum / cnt.clamp_min(1.0)).reshape(hs, ws)


def fill(img, m):
    if not bool(m.any()) or bool(m.all()):
        return img
    w = m.float()[None]
    pyr = [(img.float() * w, w)]
    while max(pyr[-1][1].shape[-2:]) > 1:
        a, b = pyr[-1]
        hw = ((a.shape[-2] + 1) // 2, (a.shape[-1] + 1) // 2)
        pyr.append((F.adaptive_avg_pool2d(a[None], hw)[0], F.adaptive_avg_pool2d(b[None], hw)[0]))
    up = None
    for a, b in reversed(pyr):
        c = a / b.clamp_min(1e-6)
        if up is not None:
            c = torch.where(b > 1e-6, c, F.interpolate(up[None], size=tuple(a.shape[-2:]), mode="nearest")[0])
        up = c
    return torch.where(m[None], img.float(), up)


def run_verify(ctx, res=None):
    dev, loc, fps = ctx["dev"], ctx["loc"], float(ctx["fps"])
    given = ctx.get("ref_poses") or {}
    tests = sorted([x for x in ctx["tests"] if x in given], key=E.fnum)
    if res and res.get("names"):
        keep = set(res["names"])
        tests = [x for x in tests if x in keep]
    map_nums = [int(x) for x in ctx["map_nums"]]
    scale = float(np.median(loc.depths))
    cap = int(ctx.get("verify_cap", 16))
    seed = int(ctx.get("verify_seed", 0))
    n_far, n_ret = int(ctx.get("verify_far", 2)), int(ctx.get("verify_retrieved", 4))
    n_boot = int(ctx.get("verify_boot", 1000))
    vc = ctx.get("verify_checks")
    want = tuple(k for k, _, _ in CHECKS if vc is None or k in vc)
    fk = dict(topk=int(ctx.get("fast_topk", 2)), photo=str(ctx.get("fast_photo", "lcn")), levels=tuple(ctx.get("fast_levels", (3, 2, 1))),
              iters=tuple(ctx.get("fast_iters", (5, 4, 3))))
    ak = dict(photo=fk["photo"], levels=fk["levels"], iters=fk["iters"], support=int(ctx.get("fast_support", 64)))
    chk = dict(ctx.get("fast_check") or {})
    lvl = int(ctx.get("verify_level", min(int(x) for x in fk["levels"])))
    min_cover = float(ctx.get("verify_min_cover", 0.1))
    lean = bool(ctx.get("verify_lean", True))
    hs, ws = loc.h // 2 ** lvl, loc.w // 2 ** lvl
    intr = Wp.level_intrinsics(loc.K, lvl)
    notes, failed, cache, drawn = [], set(), {}, {}
    try:
        thr_tok = 0.5 * LF.token_grid(loc)[0]
    except Exception as e:
        thr_tok = None
        want = tuple(k for k in want if k != "a_tok")
        notes.append("no token grid, check (a) with token matches left out, %s" % e)
    thr_dec = float(loc.reproj)

    def tens(p):
        return torch.tensor(np.asarray(p), dtype=torch.float32, device=dev)

    def rgbd(p, num, which):
        pt = tens(p)
        with torch.no_grad():
            return ctx["render"](pt, ctx["medium"](num, pt) if which == "underwater" else None, t=num / fps, which=which, depth=True)

    at_u = lambda p, i: rgbd(p, map_nums[i], "underwater")
    mi = ctx.get("map_images")
    if mi is None and ctx.get("map_names"):
        try:
            mi = [ctx["load"](x) for x in ctx["map_names"]]
        except Exception as e:
            notes.append("the map frames could not be loaded for the splat checks, %s, %s" % (type(e).__name__, e))
            mi = None
    if mi is not None and (len(mi) != loc.n or not len(mi) or tuple(mi[0].shape[-2:]) != (loc.h, loc.w)):
        notes.append("the map frame images do not match the store (%d of %d, size %s), the splat checks are left out" % (
            len(mi), loc.n, tuple(mi[0].shape[-2:]) if len(mi) else None))
        mi = None
    if mi is None:
        want = tuple(k for k in want if k not in SPLAT)
        if not any("splat" in x for x in notes):
            notes.append("no images of the map frames given (map_images or map_names), the splat checks are left out")
    else:
        mi = [x.to(dev).float() for x in mi]
    anchors = None
    if ctx.get("verify_fast", True) and ctx.get("verify_anchors", True):
        try:
            anchors = LF.TokenAnchors(loc, cell=float(ctx.get("fast_cell", 0.25)))
        except Exception as e:
            notes.append("the anchors could not be built, their outputs are left out, %s, %s" % (type(e).__name__, e))

    def geo(j, fq, p, kind):
        i = int(loc.near(p, 1)[0])
        key = (j, i, kind)
        fresh = key not in cache
        if fresh:
            B.sync(dev)
            a = time.time()
            got = LF.token_matches(loc, fq, [i])[:2] if kind == "token" else loc.match(fq, [i], lean=lean)[:2]
            B.sync(dev)
            cache[key] = (got[0], got[1], time.time() - a)
        p2d, p3d, sec = cache[key]
        if len(p2d) < 8:
            return None, 0.0 if fresh else sec
        uv, z = G.project(loc.K, p, p3d)
        e = np.linalg.norm(uv - p2d, axis=1)
        return float(np.mean((z > 1e-6) & (e < (thr_tok if kind == "token" else thr_dec)))), 0.0 if fresh else sec

    def draw(p, kind):
        if kind in drawn:
            return drawn[kind]
        B.sync(dev)
        a = time.time()
        if kind in ("b", "c"):
            ids = loc.near(p, 1 if kind == "b" else 3)
            A, m, _ = splat(torch.cat([loc.xyz[i].reshape(-1, 3) for i in ids]), torch.cat([mi[i].reshape(3, -1) for i in ids], 1), p, intr, hs, ws,
                            torch.cat([loc.valid[i].reshape(-1) for i in ids]))
        else:
            i = int(loc.near(p, 1)[0])
            ir, al, _ = rgbd(p, map_nums[i], "underwater" if kind == "d" else "restored")
            A = Wp.pool_image(ir.clamp(0.0, 1.0).float(), lvl)
            m = Wp.pool_image(al.float().reshape(1, loc.h, loc.w), lvl)[0] > 0.5
        B.sync(dev)
        drawn[kind] = (A, m, time.time() - a)
        return A, m, 0.0

    def compare(ql, p, key):
        A, m, extra = draw(p, key[0])
        return (rgb_corr if key.endswith("_rgb") else lcn_corr)(ql, A, m, min_cover), extra

    full = Wp.level_intrinsics(loc.K, 0)

    def frame_rgbd(q, i):
        sp, m, z = splat(loc.xyz[i].reshape(-1, 3), mi[i].reshape(3, -1), q, full, loc.h, loc.w, loc.valid[i].reshape(-1))
        return fill(sp, m), m.float(), z

    def aligned(img, p, kf=False):
        i = int(loc.near(p, 1)[0])
        draw_at = (lambda q: frame_rgbd(q, i)) if kf else (lambda q: rgbd(q, map_nums[i], "underwater"))
        al = LF.direct_align(img, p, loc.K, draw_at, levels=fk["levels"], iters=fk["iters"], photo=fk["photo"], sync_fn=lambda: B.sync(dev))
        if not al["ok"] or al["shift"] is None:
            return None, 0.0
        return -max(100.0 * float(al["shift"][0]) / LABELS[0][0], float(al["shift"][1]) / LABELS[0][1]), 0.0

    def fast_check(c):
        r = c.get("raw")
        if r is None:
            return None, 0.0
        return float(LF.passes(r, **chk)), float(r["stages"].get("render", 0.0) + r["stages"].get("align", 0.0))

    def score_all(j, img, ql, fq, c):
        drawn.clear()
        p = c["pose"]
        fns = dict(a_tok=lambda: geo(j, fq, p, "token"), a_dec=lambda: geo(j, fq, p, "decoder"), f=lambda: fast_check(c), g=lambda: aligned(img, p),
                   h=lambda: aligned(img, p, True))
        for k in DRAWN:
            fns[k] = lambda k=k: compare(ql, p, k)
        s, ms = {}, {}
        for key in want:
            if key == "f" and c.get("raw") is None:
                continue
            try:
                B.sync(dev)
                a = time.time()
                with torch.no_grad():
                    v, extra = fns[key]()
                B.sync(dev)
                sec = time.time() - a + extra
            except Exception as e:
                if key not in failed:
                    failed.add(key)
                    notes.append("check %s failed at least once and counts as not computed there, %s, %s" % (key, type(e).__name__, str(e)[:200]))
                v, sec = None, None
            s[key] = None if v is None or not np.isfinite(v) else float(v)
            if sec is not None:
                ms[key] = 1e3 * sec
        drawn.clear()
        return s, ms

    imgs = [ctx["load"](x) for x in tests]
    nums = [E.fnum(x) for x in tests]
    fold, blocks = folds(nums, str(ctx.get("verify_folds", "halves")), int(ctx.get("verify_block_gap", 2)))
    combos = offsets()
    combos = [combos[k] for k in np.random.default_rng(seed).permutation(len(combos))]
    cands, at = [], 0
    for j, name in enumerate(tests):
        w0 = time.time()
        ref = np.asarray(given[name], dtype=np.float64)
        img = imgs[j]
        ql = Wp.pool_image(img.float(), lvl)
        rng_far, rng_p = np.random.default_rng([seed, nums[j], 1]), np.random.default_rng([seed, nums[j], 2])
        fq = loc.encode(img)
        meth = []

        def add_method(src, fn):
            try:
                meth.extend(fn())
            except Exception as e:
                notes.append("the output of %s is missing for frame %06d, %s, %s" % (src, nums[j], type(e).__name__, str(e)[:200]))

        if ctx.get("verify_methods", True):
            def ff():
                r = loc.localize(img, topk=1, fq=fq)
                return [dict(type="method", source="no render, top 1" + ("" if r["ok"] else ", no PnP pose"), pose=r["pose"])]

            add_method("the no render path", ff)
            if ctx.get("verify_fast", True):
                def fast():
                    r = LF.localize_fast(loc, img, at_u, **fk)
                    r.pop("fq", None)
                    return [dict(type="method", source="fast path, token PnP", pose=r["coarse"]),
                            dict(type="method", source="fast path, after alignment", pose=r["pose"], raw=r)]

                add_method("the fast path", fast)
                if anchors is not None:
                    def anc():
                        r = LF.localize_fast(loc, img, at_u, anchors=anchors, **ak)
                        r.pop("fq", None)
                        return [dict(type="method", source="anchors, after alignment", pose=r["pose"], raw=r)]

                    add_method("the anchors", anc)
        meth = meth[:cap]
        rem = cap - len(meth)
        specs = []
        for k in range(max(n_ret, n_far)):
            if k < n_ret:
                specs.append(("ret", k))
            if k < n_far:
                specs.append(("far", k))
        specs = specs[:max(min(len(specs), rem - (rem + 2) // 3), 0)]
        idx = loc.retrieve(fq, 1 + n_ret)[0] if any(x[0] == "ret" for x in specs) else []
        order = loc.near(ref, loc.n)
        pool = order[len(order) // 2:]
        far_ids = [int(x) for x in rng_far.choice(pool, size=min(n_far, len(pool)), replace=False)] if n_far and pool else []
        wrong = []
        for kind, k in specs:
            try:
                if kind == "ret" and k + 1 < len(idx):
                    i = int(idx[k + 1])
                    r = loc.localize(img, idx=[i], fq=fq, lean=lean)
                    wrong.append(dict(type="wrong", source="PnP on retrieved map frame %d%s" % (k + 2, "" if r["ok"] else ", no PnP pose"), pose=r["pose"],
                                      frame_used=map_nums[i]))
                elif kind == "far" and k < len(far_ids):
                    i = far_ids[k]
                    wrong.append(dict(type="wrong", source="pose of a far map frame", pose=loc.poses[i], frame_used=map_nums[i]))
            except Exception as e:
                notes.append("a wrong place candidate is missing for frame %06d, %s, %s" % (nums[j], type(e).__name__, str(e)[:200]))
        n_p = max(min(cap - len(meth) - len(wrong), len(combos)), 0)
        pert = []
        for k in range(n_p):
            dt, dr = combos[(at + k) % len(combos)]
            pert.append(dict(type="perturbed", source="reference moved %g %% and turned %g deg" % (dt, dr), offset=[dt, dr],
                             pose=G.perturb(ref, dt / 100.0 * scale, dr, rng_p)))
        at += n_p
        mine = meth + wrong + pert
        fq = loc.encode(img)
        for c in mine:
            c["pose"] = np.asarray(c["pose"], dtype=np.float64)
            dt, dr = G.motion(c["pose"], ref)
            c.update(j=j, name=name, frame=nums[j], fold=int(fold[j]) + 1, t=100.0 * dt / scale, r=dr)
            for lab in LABELS:
                c["ok" + tag_of(lab)] = bool(c["t"] <= lab[0] + 1e-6 and c["r"] <= lab[1] + 1e-6)
        if j == 0 and mine and want:
            score_all(j, img, ql, fq, dict(mine[0]))
            cache.clear()
        for c in mine:
            c["s"], c["ms"] = score_all(j, img, ql, fq, c)
        cache.clear()
        cands += mine
        print("[verify] frame %06d, %d candidates (%d outputs, %d wrong place, %d perturbed), %d within 2 %% and 2 deg, %.1f s" % (
            nums[j], len(mine), len(meth), len(wrong), len(pert), sum(c["ok2/2"] for c in mine), time.time() - w0), flush=True)

    fc = np.array([c["fold"] - 1 for c in cands], dtype=np.int64)
    types = np.array([c["type"] for c in cands], dtype="U16")
    checks, counts, closed_s = {}, {}, {}
    in_sample = False
    for lab in LABELS:
        tg = tag_of(lab)
        y = np.array([c["ok" + tg] for c in cands], dtype=bool)
        counts[tg] = dict(all=[len(cands), int(y.sum())], **{t: [int((types == t).sum()), int(y[types == t].sum())] for t, _ in TYPES})
    for key, label, gs in CHECKS:
        if key not in want or not cands:
            continue
        vals = [c["s"].get(key) for c in cands]
        defined = np.array([v is not None for v in vals], dtype=bool)
        has = np.array([key in c["s"] for c in cands], dtype=bool)
        closed = key != "f"
        sel = np.ones(len(cands), dtype=bool) if closed else defined
        s = np.array([v if v is not None else -np.inf for v in vals], dtype=np.float64)
        if closed:
            closed_s[key] = s
        ms = [c["ms"][key] for c in cands if key in c["ms"]]
        m = dict(label=label, map=gs, closed=closed, cover=float(defined.mean()), asked=float(has.mean()), ms=float(np.mean(ms)) if ms else None,
                 ms_p90=float(np.percentile(ms, 90)) if ms else None, n=int(sel.sum()))
        fmask = np.array([c["s"].get("f") is not None for c in cands], dtype=bool)
        for lab in LABELS:
            tg = tag_of(lab)
            y = np.array([c["ok" + tg] for c in cands], dtype=bool)
            if not sel.any():
                continue
            acc_s, th = held_out(s[sel], y[sel], fc[sel])
            in_sample = in_sample or any(x["in_sample"] for x in th.values())
            acc = {k: np.zeros(len(cands), dtype=bool) for k in acc_s}
            for k in acc_s:
                acc[k][sel] = acc_s[k]
            m[tg] = dict(all=stats(s, y, acc, sel), by_type={t: stats(s, y, acc, sel & (types == t)) for t, _ in TYPES},
                         fast=stats(s, y, acc, sel & fmask), thresholds=th, auroc_folds=[auroc(s[sel & (fc == f)], y[sel & (fc == f)]) for f in (0, 1)],
                         fa_folds=[rates(acc["r90"][sel & (fc == f)], y[sel & (fc == f)])["fa"] for f in (0, 1)])
        checks[key] = m
    unc = {}
    if n_boot > 0 and closed_s:
        js = np.array([c["j"] for c in cands], dtype=np.int64)
        for lab in LABELS:
            tg = tag_of(lab)
            unc[tg] = uncertainty(js, closed_s, np.array([c["ok" + tg] for c in cands], dtype=bool), n_boot, seed)
    if len(blocks) < 2 and len(tests) > 1:
        notes.append("the queries form one block of frames, so the two folds split that block and neighbouring frames fall in different folds")
    if in_sample:
        notes.append("one fold is empty, so the thresholds were set and applied on the same queries (in sample) and the rates are optimistic")
    fr = [[nums[k] for k in range(len(nums)) if fold[k] == f] for f in (0, 1)]
    out = dict(n=len(cands), queries=len(tests), names=list(tests), cap=cap, seed=seed, far=n_far, retrieved=n_ret, level=lvl, min_cover=min_cover,
               offsets=dict(t=list(OFF_T), r=list(OFF_R), n=len(combos)), labels=[list(x) for x in LABELS], thr_px=dict(token=thr_tok, decoder=thr_dec),
               depth_units=scale, folds=dict(mode=str(ctx.get("verify_folds", "halves")), blocks=[[nums[k] for k in b] for b in blocks], frames=fr,
                                             in_sample=in_sample),
               counts=counts, checks=checks, uncertainty=unc, notes=notes,
               candidates=[dict(name=c["name"], frame=c["frame"], fold=c["fold"], type=c["type"], source=c["source"], offset=c.get("offset"),
                                frame_used=c.get("frame_used"), t=c["t"], r=c["r"], ok2=c["ok2/2"], ok5=c["ok5/5"], pose=c["pose"], scores=c["s"], ms=c["ms"])
                           for c in cands])
    out = P.clean(out)
    if ctx.get("out_dir"):
        try:
            json.dump(out, open(os.path.join(ctx["out_dir"], "verify.json"), "w"), indent=1)
        except Exception as e:
            print("[verify] verify.json not written, %s" % e, flush=True)
    return out


def fmt(v, spec):
    return "-" if v is None else spec % v


def pc(v):
    return "-" if v is None else "%.1f" % (100.0 * v)


def span(fr):
    return "%06d to %06d" % (min(fr), max(fr)) if fr else "none"


def table(out, tg, title):
    L = ["   %-40s %3s %5s %6s | %9s %6s | %9s %6s | %9s %6s | %6s" % (title, "map", "cover", "AUROC", "precision", "recall", "recall", "prec.",
                                                                     "false acc", "kept", "ms"),
         "   %-40s %3s %5s %6s | %16s | %16s | %16s | %6s" % ("", "", "%", "", "at 95 % recall", "at 95 % prec.", "at 90 % kept", "mean")]
    for key, label, gs in CHECKS:
        m = out["checks"].get(key)
        if not m or tg not in m:
            continue
        a = m[tg]["all"]
        L.append("   %-40s %3s %5.0f %6s | %9s %6s | %9s %6s | %9s %6s | %6s" % (
            label, "yes" if gs else "no", 100.0 * m["cover"], fmt(a["auroc"], "%.3f"), pc(a["r95"]["prec"]), pc(a["r95"]["rec"]), pc(a["p95"]["rec"]),
            pc(a["p95"]["prec"]), pc(a["r90"]["fa"]), pc(a["r90"]["rec"]), fmt(m.get("ms"), "%.1f")))
    return L


def by_type(out, tg):
    groups = [(t, name, out["counts"][tg][t]) for t, name in TYPES]
    head = "   %-38s" % ("by type, within %s %% and %s deg" % tuple(tg.split("/")))
    sub = "   %-38s" % "check"
    for t, name, (n, k) in groups:
        head += " | %-20s" % ("%s %d/%d" % (name, k, n))
        sub += " | %6s %6s %6s" % ("AUROC", "f acc", "kept")
    fz = [m[tg]["fast"] for m in out["checks"].values() if tg in m and m[tg]["fast"]["n"]]
    if fz:
        head += " | %-20s" % ("fast outputs %d/%d" % (fz[0]["correct"], fz[0]["n"]))
        sub += " | %6s %6s %6s" % ("AUROC", "f acc", "kept")
    L = [head.rstrip(), sub]
    for key, label, gs in CHECKS:
        m = out["checks"].get(key)
        if not m or tg not in m:
            continue
        row = "   %-38s" % label
        for t, _, _ in groups:
            a = m[tg]["by_type"][t]
            row += " | %6s %6s %6s" % (fmt(a["auroc"], "%.3f"), pc(a["r90"]["fa"]), pc(a["r90"]["rec"]))
        if fz:
            a = m[tg]["fast"]
            row += " | %6s %6s %6s" % (fmt(a["auroc"], "%.3f"), pc(a["r90"]["fa"]), pc(a["r90"]["rec"]))
        L.append(row)
    return L


def spread(out, tg):
    u = (out.get("uncertainty") or {}).get(tg) or {}
    L = ["   %-40s %6s %15s | %13s | %13s" % ("folds and intervals, %s %% and %s deg" % tuple(tg.split("/")), "AUROC", "95 % interval", "AUROC",
                                              "false acc"),
         "   %-40s %6s %15s | %6s %6s | %6s %6s" % ("", "", "over queries", "fold 1", "fold 2", "fold 1", "fold 2")]
    for key, label, gs in CHECKS:
        m = out["checks"].get(key)
        if not m or tg not in m:
            continue
        ci = (u.get("auroc") or {}).get(key)
        af, ff = m[tg].get("auroc_folds") or [None, None], m[tg].get("fa_folds") or [None, None]
        L.append("   %-40s %6s %7s %7s | %6s %6s | %6s %6s" % (
            label, fmt(m[tg]["all"]["auroc"], "%.3f"), fmt(ci[0] if ci else None, "%.3f"), fmt(ci[1] if ci else None, "%.3f"), fmt(af[0], "%.3f"),
            fmt(af[1], "%.3f"), pc(ff[0]), pc(ff[1])))
    if u.get("pairs"):
        L += ["", "   %-28s %-56s | %7s | %15s | %7s" % ("pair", "Gaussian map check against the check without it", "AUROC", "95 % interval",
                                                     "above 0"),
              "   %-28s %-56s | %7s | %15s | %7s" % ("", "", "diff.", "over queries", "% draws")]
        for p in u["pairs"]:
            ci = p.get("ci")
            L.append("   %-28s %-56s | %7s | %7s %7s | %7s" % (
                "%s minus %s" % (SHORT[p["a"]], SHORT[p["b"]]), p["what"], fmt(p.get("diff"), "%+.3f"), fmt(ci[0] if ci else None, "%+.3f"),
                fmt(ci[1] if ci else None, "%+.3f"), pc(p.get("above"))))
    if u:
        L += ["   The intervals draw the queries again with all their candidates (%d draws of %d queries), so they show how much these few" % (
            u["n"], u["queries"]),
              "   queries can tell. A difference whose interval contains 0 does not separate the two checks. The share above 0 is the share",
              "   of draws in which the Gaussian map check has the higher AUROC. Fold columns use the thresholds set on the other fold."]
    return L


def report_verify(out):
    L = ["=" * W, " Pose verification   does a check against the map reject wrong poses better than checks that need no Gaussian map (tools/bench_verify.py)",
         "-" * W]
    if not out or not out.get("n"):
        L.append(" no candidate poses were scored" + ("" if not out else " (%d queries)" % out.get("queries", 0)))
        for x in (out or {}).get("notes", []):
            L.append(" note        %s" % x)
        return L + ["=" * W]
    ct = out["counts"]
    a2, a5 = ct["2/2"]["all"], ct["5/5"]["all"]
    lv = 2 ** int(out["level"])
    L.append(" Candidates  %d poses for %d queries, at most %d per query, seed %d | %d localizer outputs, %d wrong places, %d perturbed" % (
        out["n"], out["queries"], out["cap"], out["seed"], ct["2/2"]["method"][0], ct["2/2"]["wrong"][0], ct["2/2"]["perturbed"][0]))
    L.append("             %d (%.1f %%) within 2 %% and 2 deg of the reference, %d (%.1f %%) within 5 %% and 5 deg, the boundary counted as within" % (
        a2[1], 100.0 * a2[1] / max(a2[0], 1), a5[1], 100.0 * a5[1] / max(a5[0], 1)))
    L += [" Outputs     raw poses before any check or fallback. No render top 1, the fast path after its token PnP and after its",
          "             alignment, the anchors after their alignment",
          " Wrong place poses of %d map frames drawn from the half of the map farthest from the reference pose of the query (test" % out["far"],
          "             inputs chosen with the reference), and PnP of the decoder matches to the retrieved map frames 2 to %d. Labelled by" % (
              1 + out["retrieved"]),
          "             their error like every candidate, so a PnP on a neighbour can be correct",
          " Perturbed   test inputs, never a localizer output. The reference pose moved by %s %% of the median depth in a seeded" % (
              ", ".join("%g" % x for x in out["offsets"]["t"])),
          "             random direction, turned by %s deg about a seeded random axis, or both. The %d offsets are in one" % (
              ", ".join("%g" % x for x in out["offsets"]["r"]), out["offsets"]["n"]),
          "             seeded order and each query takes the next ones"]
    fr = out["folds"]["frames"]
    L += [" Folds       the queries in two folds by blocks of frame numbers, fold 1 frames %s (%d queries)," % (span(fr[0]), len(fr[0])),
          "             fold 2 frames %s (%d queries). Every threshold is set on one fold and applied to the other, the rates below" % (
              span(fr[1]), len(fr[1])),
          "             pool the two held out folds. AUROC needs no threshold" + (", but here one fold is empty and the thresholds are in sample" if
                                                                                   out["folds"].get("in_sample") else "")]
    tp = out["thr_px"]
    L += [" Checks      (a) share of the matches of the query to the map frame nearest the candidate that reproject within the PnP",
          "             threshold under the candidate (tokens %s px, decoder %.1f px), the matches lifted by the Stage 1 points. For an" % (
              fmt(tp.get("token"), "%.1f"), tp["decoder"]),
          "             output fitted on those same matches (a) is its own PnP inlier share, and every candidate is charged the full",
          "             matching time even where the localizer already has the matches.",
          "             (b) and (c) the Stage 1 points of the nearest map frame or of the 3 nearest, with the colours of their own images,",
          "             drawn into the candidate view with a depth buffer at 1/%d resolution. (d) underwater render of the Gaussian map" % lv,
          "             with the medium field. (e) the restored render without the medium. (d) and (e) are rendered at full size and",
          "             averaged down to 1/%d, so their times include a full size render. (b) to (e) correlate the locally normalised" % lv,
          "             brightness with the query over the pixels the map covers (opacity above 0.5 in the renders). The plain colour rows",
          "             correlate the colours of the same images and pixels with only the mean of every channel removed, so the colour",
          "             of the water stays in the score. A pixel without support is left out and never taken as water. (f) the check",
          "             of the fast path, scored on its own outputs only, timed as the render and alignment it needs. (g) direct",
          "             alignment of the query from the candidate on the underwater render, the size of its correction in units of",
          "             2 % of the median rendered depth and 2 deg (larger rejects). (h) the same alignment on the splat of the nearest",
          "             map frame at full size, no Gaussian map. The gaps between its points are filled from their neighbours for the",
          "             image pyramid only and never count as support. A check that cannot be computed rejects the candidate at every",
          "             threshold (below %.0f %% of the pixels covered, fewer than 8 matches, no valid alignment). Times per check are" % (
              100.0 * out["min_cover"]),
          "             synchronised on the GPU. The encoder pass of the query is shared with the localizer and not counted, image",
          "             loading and scoring are not timed. The map column says whether the check renders the Gaussian map. Rates in %",
          "             of the candidates."]
    L += ["-" * W] + table(out, "2/2", "check, correct within 2 % and 2 deg")
    L += [""] + table(out, "5/5", "check, correct within 5 % and 5 deg")
    L += [""] + by_type(out, "2/2")
    L.append("   f acc is the share of wrong candidates accepted and kept the share of correct ones accepted, both at the threshold that")
    L.append("   keeps 90 % of the correct candidates of the other fold. A group heading gives correct of all candidates. AUROC is left")
    L.append("   blank where a group has only correct or only wrong candidates. Fast outputs are the poses of the fast path and the")
    L.append("   anchors after alignment, where (f) applies.")
    L += [""] + spread(out, "2/2")
    for x in out.get("notes", []):
        L.append(" note        %s" % x)
    L.append("=" * W)
    return L


def in_short_verify(out, tg="2/2", pairs=(("d", "c"), ("d_rgb", "c_rgb"), ("g", "h"))):
    if not out or not out.get("n"):
        return [" Verify      no candidate poses were scored, see verify.txt"]
    k2 = out["counts"][tg]["all"]
    L = [" Verify      %d candidate poses of %d queries, %d within %s %% and %s deg | AUROC of every check, 0.5 is chance | (d) (e) (g) render the" % (
        out["n"], out["queries"], k2[1], *tg.split("/")),
         "             Gaussian map, (a) (b) (c) (h) do not"]
    items = ["%s %s" % (SHORT[k], fmt(out["checks"][k][tg]["all"]["auroc"], "%.3f")) for k, _, _ in CHECKS if tg in out["checks"].get(k, {})]
    for s in range(0, len(items), 7):
        L.append("               " + " | ".join(items[s:s + 7]))
    got = {(p["a"], p["b"]): p for p in ((out.get("uncertainty") or {}).get(tg) or {}).get("pairs") or []}
    for a, b in pairs:
        p = got.get((a, b))
        if p:
            ci = p.get("ci")
            L.append("               %s minus %s %s, 95 %% interval over queries %s to %s" % (
                SHORT[a], SHORT[b], fmt(p.get("diff"), "%+.3f"), fmt(ci[0] if ci else None, "%+.3f"), fmt(ci[1] if ci else None, "%+.3f")))
    return L
