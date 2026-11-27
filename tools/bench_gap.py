import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import bench_gscpe as C
import bench_loc as B
import bench_proto as P
import eval_gaussian_map as E
import loc_fast as LF
import loc_geom as G
import loc_ref as R

LEVELS = ((2.0, 0.0), (5.0, 0.0), (10.0, 0.0), (20.0, 0.0), (0.0, 2.0), (0.0, 5.0), (0.0, 10.0))
T_EDGES = (3.0, 6.0, 12.0)
R_EDGES = (2.0, 5.0, 10.0)
W = C.W
ROWS = ("sim. prior itself, no computation", "sim. prior, render, match, PnP", "sim. prior, render, PnP, direct align",
        "sim. prior, no render, nearest frame", "sim. prior, no render, best of 3", "sim. prior, no render, frame per pass",
        "sim. prior, splat of 3 keyframes")
FREE_KEYS = ("no render: match the map frame, PnP", "coarse: render, match, PnP", "fast + check (decoder when it fails)")
FREE = ("no render, match the map frame, PnP", "coarse, render, match, PnP", "fast + check, decoder when it fails")


def parse_priors(v):
    out = []
    for x in v or ():
        if isinstance(x, str):
            a, _, b = x.partition(":")
            out.append((float(a), float(b or 0.0)))
        else:
            out.append((float(x[0]), float(x[1])))
    return out or [tuple(x) for x in LEVELS]


def usable(levels):
    good = [l for l in levels if np.isfinite(l).all() and min(l) >= 0.0 and max(l) > 0.0]
    return good, [l for l in levels if l not in good]


def prior_pose(pose, lt, lr, scale, seed, num):
    rng = np.random.default_rng([int(seed) % 2 ** 32, int(num)])
    return G.perturb(pose, lt * scale / 100.0, lr, rng)


def bin_of(x, edges):
    return int(np.searchsorted(np.asarray(edges, dtype=np.float64), float(x), side="right"))


def bin_names(edges, unit):
    e = ["%g" % x for x in edges]
    return ["below %s %s" % (e[0], unit)] + ["%s to %s %s" % (a, b, unit) for a, b in zip(e[:-1], e[1:])] + ["above %s %s" % (e[-1], unit)]


def err(p, q, scale):
    dt, dr = G.motion(p, q)
    return 100.0 * dt / scale, dr


def block(e, thr=C.THRESHOLDS):
    e = np.asarray(e, dtype=np.float64).reshape(-1, 2)
    if not len(e):
        return None
    t, r = e[:, 0], e[:, 1]
    return dict(n=len(e), acc=[float(100.0 * np.mean((t <= x) & (r <= y))) for x, y in thr], t=float(np.median(t)), r=float(np.median(r)),
                t_p90=float(np.percentile(t, 90)), r_p90=float(np.percentile(r, 90)))


def rate(e, succ):
    e = np.asarray(e, dtype=np.float64).reshape(-1, 2)
    return float(100.0 * np.mean((e[:, 0] <= succ[0]) & (e[:, 1] <= succ[1]))) if len(e) else None


def ref_diff(loc, ref_a, ref_b, names, scale):
    both = [n for n in names if n in ref_a and n in ref_b]
    if not both:
        return None
    e = np.array([err(ref_a[n], ref_b[n], scale) for n in both])
    px = np.array([G.pixel_shift(loc.K, ref_a[n], ref_b[n], loc.points[int(loc.near(ref_a[n], 1)[0])], (loc.h, loc.w)) for n in both])
    fin = px[np.isfinite(px)]
    return dict(n=len(both), t_med=float(np.median(e[:, 0])), t_p90=float(np.percentile(e[:, 0], 90)), r_med=float(np.median(e[:, 1])),
                r_p90=float(np.percentile(e[:, 1], 90)), px_med=float(np.median(fin)) if len(fin) else None,
                px_p90=float(np.percentile(fin, 90)) if len(fin) else None)


def rescore(res, names, ref_a, ref_b, scale):
    rows, missing = [], []
    pq_all = res.get("per_query") or {}
    for r in res.get("rows") or []:
        pq = pq_all.get(r["label"])
        if not pq or len(pq) != len(names):
            continue
        if any(e.get("pose") is None for e in pq):
            missing.append(r["label"])
            continue
        ea, eb = [], []
        for e, n in zip(pq, names):
            if n in ref_a and n in ref_b:
                p = np.asarray(e["pose"], dtype=np.float64)
                ea.append(err(p, ref_a[n], scale))
                eb.append(err(p, ref_b[n], scale))
        if ea:
            rows.append(dict(label=r["label"], family=r.get("family"), n=len(ea), a=block(ea), b=block(eb)))
    return rows, missing


def gaps_of(loc, poses, names, scale):
    hw = (loc.h, loc.w)
    out = []
    for n in names:
        p = np.asarray(poses[n], dtype=np.float64)
        near = int(loc.near(p, 1)[0])
        dt, dr = G.motion(loc.poses[near], p)
        seen = max(G.covisible(loc.K, p, loc.poses[k], loc.points[near], hw) for k in loc.near(p, 6))
        out.append(dict(name=n, near=near, gap_t=100.0 * dt / scale, gap_r=dr, seen=float(seen)))
    return out


def frame_gaps(tests, map_nums, ref_a, ref_b):
    mn = np.asarray(sorted(int(x) for x in map_nums), dtype=np.int64)
    rows = {}
    for n in tests:
        d = int(np.min(np.abs(mn - E.fnum(n)))) if len(mn) else 0
        r = rows.setdefault(d, dict(dist=d, n=0, a=0, b=0))
        r["n"] += 1
        r["a"] += int(n in ref_a)
        r["b"] += int(n in ref_b)
    return [rows[k] for k in sorted(rows)]


def strata(res, names, gaps, ref_b=None, scale=1.0, t_edges=T_EDGES, r_edges=R_EDGES):
    out = []
    pq_all = res.get("per_query") or {}
    bt = np.array([bin_of(g["gap_t"], t_edges) for g in gaps], dtype=np.int64)
    br = np.array([bin_of(g["gap_r"], r_edges) for g in gaps], dtype=np.int64)
    num = lambda v: np.nan if v is None else float(v)
    for r in res.get("rows") or []:
        pq = pq_all.get(r["label"])
        if not pq or len(pq) != len(names):
            continue
        e = np.array([[num(x.get("t")), num(x.get("r"))] for x in pq], dtype=np.float64).reshape(-1, 2)
        eb = np.full_like(e, np.nan)
        if ref_b:
            for j, (x, n) in enumerate(zip(pq, names)):
                if x.get("pose") is not None and n in ref_b:
                    eb[j] = err(np.asarray(x["pose"], dtype=np.float64), ref_b[n], scale)
        row = dict(label=r["label"], family=r.get("family"), t_bins=[], r_bins=[])
        for key, b, edges in (("t_bins", bt, t_edges), ("r_bins", br, r_edges)):
            for k in range(len(edges) + 1):
                m = (b == k) & np.isfinite(e).all(1)
                mb = (b == k) & np.isfinite(eb).all(1)
                row[key].append(dict(n=int((b == k).sum()), a=block(e[m]), b=block(eb[mb])))
        out.append(row)
    return out


def basin(levels, rates, need=80.0):
    out = {}
    kinds = (("translation", lambda l: l[0] > 0 and l[1] == 0), ("rotation", lambda l: l[0] == 0 and l[1] > 0), ("both", lambda l: l[0] > 0 and l[1] > 0))
    for kind, sel in kinds:
        ids = sorted([i for i, l in enumerate(levels) if sel(l)], key=lambda i: (levels[i][0], levels[i][1]))
        if not ids:
            continue
        best = None
        for i in ids:
            if rates[i] is None or rates[i] < need:
                break
            best = [float(levels[i][0]), float(levels[i][1])]
        out[kind] = best
    return out


def lazy_matcher(make, name):
    box = []

    def match(query, frame):
        if not box:
            box.append(make())
        return box[0](query, frame)

    match.name = name
    match.box = box
    return match


@torch.no_grad()
def splat(loc, colours, idx, pose, fill=2, occl=0.05, win=5):
    h, w = loc.h, loc.w
    dev = loc.xyz.device
    c = torch.as_tensor(np.asarray(pose), dtype=torch.float32, device=dev)
    pts = torch.cat([loc.xyz[i][loc.valid[i]] for i in idx])
    col = torch.cat([colours[i].to(dev).float().permute(1, 2, 0)[loc.valid[i]] for i in idx])
    pc = (pts - c[:3, 3]) @ c[:3, :3]
    z = pc[:, 2]
    zs = torch.where(z > 1e-6, z, torch.ones_like(z))
    u = torch.round(float(loc.K[0, 0]) * pc[:, 0] / zs + float(loc.K[0, 2])).long()
    v = torch.round(float(loc.K[1, 1]) * pc[:, 1] / zs + float(loc.K[1, 2])).long()
    ok = (z > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    pix, zz, cc = (v * w + u)[ok], z[ok], col[ok]
    zb = torch.full((h * w,), float("inf"), device=dev)
    if len(pix):
        zb.scatter_reduce_(0, pix, zz, "amin", include_self=True)
    hit = torch.isfinite(zb)
    if occl > 0 and win > 1:
        zf = zb.reshape(1, 1, h, w)
        near = -F.max_pool2d(-zf, int(win), 1, int(win) // 2)
        hit = hit & (zf <= near * (1.0 + float(occl))).reshape(-1)
    img = torch.zeros(h * w, 3, device=dev)
    top = (zz <= zb[pix]) & hit[pix]
    img[pix[top]] = cc[top]
    depth = torch.where(hit, zb, torch.zeros_like(zb)).reshape(h, w)
    img = img.reshape(h, w, 3).permute(2, 0, 1).contiguous()
    alpha = hit.reshape(h, w).float()
    for _ in range(int(fill)):
        cnt = F.avg_pool2d(alpha[None, None], 3, 1, 1)[0, 0] * 9.0
        si = F.avg_pool2d((img * alpha)[None], 3, 1, 1)[0] * 9.0
        sd = F.avg_pool2d((depth * alpha)[None, None], 3, 1, 1)[0, 0] * 9.0
        add = (alpha < 0.5) & (cnt >= 3.0)
        if not bool(add.any()):
            break
        img = torch.where(add[None], si / cnt.clamp_min(1.0), img)
        depth = torch.where(add, sd / cnt.clamp_min(1.0), depth)
        alpha = torch.where(add, torch.ones_like(alpha), alpha)
    return img, alpha, depth


def run_gap(ctx, res):
    dev, loc, fps = ctx["dev"], ctx["loc"], float(ctx["fps"])
    given = ctx["ref_poses"]
    map_nums = [int(x) for x in ctx["map_nums"]]
    tests = sorted(ctx["tests"], key=E.fnum)
    names = list((res or {}).get("names") or [x for x in tests if x in given])
    free = [x for x in tests if x not in given]
    hw = (loc.h, loc.w)
    scale = float(np.median(loc.depths))
    warm = int(ctx.get("warmup", 2))
    depth_m = ctx.get("depth_m")
    n_c = int(ctx.get("coarse_iters", 3))
    eps_t, eps_r = float(ctx.get("eps_t", 0.0002)), float(ctx.get("eps_r", 0.02))
    succ = tuple(float(x) for x in ctx.get("gap_success", (1.0, 1.0)))
    need = float(ctx.get("gap_need", 80.0))
    levels, bad = usable(parse_priors(ctx.get("gap_priors")))
    seed = int(ctx.get("gap_seed", 0))
    flow_on = bool(ctx.get("gap_flow", True))
    occl = float(ctx.get("gap_splat_occl", 0.05))
    colours = ctx.get("map_images")
    fa = ctx.get("ref_fit") or {}
    fk = dict(topk=int(ctx.get("fast_topk", 2)), photo=str(ctx.get("fast_photo", "lcn")), levels=tuple(ctx.get("fast_levels", (3, 2, 1))),
              iters=tuple(ctx.get("fast_iters", (5, 4, 3))))
    chk = dict(ctx.get("fast_check") or {})
    out = dict(map_frames=int(loc.n), queries=len(tests), scored=len(names), free=len(free), depth_units=scale, success=list(succ), need=need,
               levels=[list(x) for x in levels], seed=seed, coarse_iters=n_c, warmup=warm, notes=[], ref_source=ctx.get("ref_source", ""),
               fit_a=fa, t_edges=list(T_EDGES), r_edges=list(R_EDGES), splat=colours is not None, splat_occl=occl)
    if bad:
        out["notes"].append("the prior errors %s are left out, a simulated prior needs an error above zero and no negative part" % (
            ", ".join("%g %% and %g deg" % l for l in bad)))
    inside = [l for l in levels if l[0] <= succ[0] and l[1] <= succ[1]]
    if inside:
        out["notes"].append("at the prior errors %s the prior alone already counts as a success" % ", ".join("%g %% and %g deg" % l for l in inside))
    path = os.path.join(ctx["out_dir"], "gap.json")

    def save():
        try:
            json.dump(P.clean(out), open(path, "w"), indent=1)
        except Exception as e:
            print("[gap] gap.json not written: %s" % e, flush=True)

    cache = {}

    def image(n):
        if n not in cache:
            cache[n] = ctx["load"](n)
        return cache[n]

    def tens(p):
        return torch.tensor(np.asarray(p), dtype=torch.float32, device=dev)

    def rgbd(p, num, which):
        pt = tens(p)
        with torch.no_grad():
            return ctx["render"](pt, ctx["medium"](num, pt) if which == "underwater" else None, t=num / fps, which=which, depth=True)

    def flow_at(img, pose, num):
        if not flow_on:
            return None
        pt = tens(pose)
        with torch.no_grad():
            im, al = ctx["render"](pt, ctx["medium"](num, pt), t=num / fps, alpha=True)
        f = G.flow_residual(img.permute(1, 2, 0).cpu().numpy(), im.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy(), (al > 0.5).cpu().numpy())
        return f if f is not None and np.isfinite(f) else None

    def passes(fq, p0, view):
        T = np.asarray(p0, dtype=np.float64)
        sec = dict(render=0.0, encoder=0.0, decoder=0.0, matching=0.0, pnp=0.0)
        its, ok, matches, inliers, conv = 0, False, 0, 0, False
        for _ in range(n_c):
            B.sync(dev)
            a = time.time()
            ir, al, dp = view(T)[:3]
            B.sync(dev)
            sec["render"] += time.time() - a
            p2d, p3d, _, st = loc.match_render(fq, ir, dp, al, T)
            for k in st:
                sec[k] += st[k]
            a = time.time()
            sol = G.solve_pnp(p3d, p2d, loc.K, loc.reproj, T) if len(p2d) >= 6 else None
            sec["pnp"] += time.time() - a
            its += 1
            if sol is None or not np.isfinite(sol["c2w"]).all():
                break
            ok, matches, inliers = True, int(len(p2d)), int(sol["inliers"])
            dt, dr = G.motion(T, sol["c2w"])
            T = np.asarray(sol["c2w"], dtype=np.float64)
            if dt <= eps_t * scale and dr <= eps_r:
                conv = True
                break
        return dict(pose=T, ok=ok, iters=its, matches=matches, inliers=inliers, converged=conv, stages=sec, seconds=float(sum(sec.values())))

    ref_b, store, lifter, m = {}, {}, None, None
    try:
        m = ctx.get("ref_matcher")
        mname = str(fa.get("matcher") or "")
        if m is None and "LoFTR" in mname:
            m = lazy_matcher(lambda: R.loftr_matcher(dev, conf_min=float(ctx.get("ref_loftr_conf", 0.5))), mname)
        if not fa.get("register"):
            out["notes"].append("reference B is made only for a reference registered into the fixed map (HOLD_MAP=register), it is left out")
        elif m is None:
            out["notes"].append("reference B needs the outside matcher of reference A (REF_MATCHER=loftr), it is left out")
        else:
            mk = lambda common: R.depth_lifter(loc, lambda i: rgbd(loc.poses[i], map_nums[i], "restored"), min_alpha=float(ctx.get("gap_alpha", 0.5)),
                                               common=common, store=store)
            lifter = mk(bool(ctx.get("gap_common", False)))
            interp = E.interpolate_poses(np.asarray(loc.poses, dtype=np.float64), [E.fnum(x) for x in tests], map_nums)
            rc = ctx.get("ref_cache")
            rc = rc if rc is not None else {}
            reg = lambda lf, tag: R.register_queries(loc, tests, image, map_nums, interp, k=int(ctx.get("ref_k", 6)), tol=float(ctx.get("ref_tol", 2.0)),
                                                     matcher=m, images=colours, lifter=lf, cache=rc, tag=tag)
            ref_b, fit_b, rows_b = reg(lifter, "ref gs")
            fit_b["support"] = float(np.mean([float(x[1].float().mean()) for x in lifter.maps.values()])) if lifter.maps else None
            out.update(fit_b=fit_b, rows_b=rows_b, ref_b={k: v for k, v in ref_b.items()})
            print("[gap] reference B: %d of %d queries, %d matcher calls, %d pairs reused" % (fit_b["n"], fit_b["total"], fit_b.get("matcher_calls", 0),
                                                                                            fit_b.get("reused", 0)), flush=True)
            if not ctx.get("gap_common", False) and ctx.get("gap_common_check", True):
                try:
                    both_ref, fit_c, _ = reg(mk(True), "ref gs common")
                    out["fit_common"] = fit_c
                    out["diff_common"] = ref_diff(loc, given, both_ref, tests, scale)
                except Exception as e:
                    out["notes"].append("reference B on the common support failed (%s, %s)" % (type(e).__name__, e))
            fit_b["matcher_built"] = bool(getattr(m, "box", [None]))
    except Exception as e:
        out["notes"].append("reference B failed (%s, %s)" % (type(e).__name__, e))
    finally:
        if getattr(m, "box", None):
            m.box.clear()
        store.clear()
        if lifter is not None:
            lifter.maps.clear()
    out["ref_a_n"], out["ref_b_n"] = len([x for x in tests if x in given]), len(ref_b)
    out["frame_gaps"] = frame_gaps(tests, map_nums, given, ref_b)
    try:
        out["diff"] = ref_diff(loc, given, ref_b, tests, scale)
        out["both"], miss = rescore(res or {}, names, given, ref_b, scale)
        if miss and ref_b:
            out["notes"].append("gscpe.json keeps no pose per query for %d rows, they are not scored against reference B" % len(miss))
        out["missing_poses"] = miss
    except Exception as e:
        out["notes"].append("scoring against both references failed (%s, %s)" % (type(e).__name__, e))
    save()

    try:
        out["gaps"] = gaps_of(loc, given, names, scale)
        out["strata"] = strata(res or {}, names, out["gaps"], ref_b, scale)
    except Exception as e:
        out["notes"].append("gap strata failed (%s, %s)" % (type(e).__name__, e))
    save()

    if free:
        try:
            at = lambda which: (lambda p, i: rgbd(p, map_nums[i], which))

            def coarse_free(img):
                B.sync(dev)
                a = time.time()
                fq = loc.encode(img)
                idx, _ = loc.retrieve(fq, 1)
                B.sync(dev)
                t_ret = time.time() - a
                num = map_nums[idx[0]]
                c = passes(fq, loc.poses[idx[0]], lambda T: rgbd(T, num, "underwater"))
                return dict(ok=c["ok"], pose=c["pose"], seconds=t_ret + c["seconds"], idx=list(idx))

            fns = ((FREE[0], lambda im: loc.localize(im, topk=1)), (FREE[1], coarse_free),
                   (FREE[2], lambda im: LF.localize_tiered(loc, im, at("underwater"), check=chk, **fk)))
            rows = []
            for label, fn in fns:
                try:
                    fn(image(free[0]))
                    got = []
                    for n in free:
                        r = fn(image(n))
                        got.append(dict(name=n, ok=bool(r["ok"]), sec=float(r["seconds"]), pose=np.asarray(r["pose"], dtype=np.float64),
                                        num=map_nums[r["idx"][0]]))
                    for g in got:
                        g["flow"] = flow_at(image(g["name"]), g["pose"], g["num"])
                        if g["name"] in ref_b:
                            g["t_b"], g["r_b"] = err(g["pose"], ref_b[g["name"]], scale)
                    fl = [g["flow"] for g in got if g["flow"] is not None]
                    s = np.array([g["sec"] for g in got[min(warm, max(len(got) - 1, 0)):]], dtype=np.float64)
                    eb = [(g["t_b"], g["r_b"]) for g in got if "t_b" in g]
                    rows.append(dict(label=label, n=len(got), returned=int(sum(g["ok"] for g in got)), flow=float(np.median(fl)) if fl else None,
                                     flow_n=len(fl), s_mean=float(s.mean()) if len(s) else None, b=block(eb), per_query=got))
                except Exception as e:
                    out["notes"].append("%s on the queries without a reference failed (%s, %s)" % (label, type(e).__name__, e))
            by = {r["label"]: r for r in (res or {}).get("rows") or []}
            out["no_ref"] = dict(names=free, rows=rows, compare={FREE[i]: by[k].get("flow") for i, k in enumerate(FREE_KEYS) if k in by})
        except Exception as e:
            out["notes"].append("queries without a reference failed (%s, %s)" % (type(e).__name__, e))
        save()

    qs = list(names)
    cap = int(ctx.get("gap_queries", 0) or 0)
    if 0 < cap < len(qs):
        qs = [qs[int(i)] for i in np.unique(np.linspace(0, len(qs) - 1, cap).round().astype(int))]
    out["prior_queries"] = qs
    if qs and not levels:
        out["notes"].append("no usable prior error is left, the navigation prior sweep is left out")
    if qs and levels:
        try:
            def from_prior(img, prior):
                o = {ROWS[0]: dict(ok=True, pose=np.asarray(prior, dtype=np.float64), sec=None)}
                B.sync(dev)
                a = time.time()
                fq = loc.encode(img)
                near = loc.near(prior, 3)
                num = map_nums[near[0]]
                B.sync(dev)
                t0 = time.time() - a
                c = passes(fq, prior, lambda T: rgbd(T, num, "underwater"))
                o[ROWS[1]] = dict(ok=c["ok"], pose=c["pose"], sec=t0 + c["seconds"], iters=c["iters"], inliers=c["inliers"])
                al = LF.direct_align(img, c["pose"], loc.K, lambda p: rgbd(p, num, "underwater"), levels=fk["levels"], iters=fk["iters"], photo=fk["photo"],
                                     sync_fn=lambda: B.sync(dev))
                o[ROWS[2]] = dict(ok=bool(c["ok"] or al["ok"]), pose=np.asarray(al["pose"] if al["ok"] else c["pose"], dtype=np.float64),
                                  sec=t0 + c["seconds"] + al["seconds"], corr=al.get("corr"))
                r = loc.localize(img, idx=near[:1], prior=prior, fq=fq)
                o[ROWS[3]] = dict(ok=bool(r["ok"]), pose=np.asarray(r["pose"], dtype=np.float64), sec=t0 + r["seconds"], inliers=r["inliers"])
                best, sec = (r if r["ok"] else None), t0 + r["seconds"]
                for i in near[1:3]:
                    q = loc.localize(img, idx=[i], prior=prior, fq=fq)
                    sec += q["seconds"]
                    if q["ok"] and (best is None or q["inliers"] > best["inliers"]):
                        best = q
                o[ROWS[4]] = dict(ok=best is not None, pose=np.asarray(best["pose"] if best else prior, dtype=np.float64), sec=sec,
                                  inliers=best["inliers"] if best else 0, frame=map_nums[best["idx"][0]] if best else None)
                cur, used, sec, its = r, [near[0]], t0 + r["seconds"], 1
                while cur["ok"] and its < n_c:
                    a = time.time()
                    i = loc.near(cur["pose"], 1)[0]
                    sec += time.time() - a
                    if i in used:
                        break
                    q = loc.localize(img, idx=[i], prior=cur["pose"], fq=fq)
                    sec += q["seconds"]
                    used.append(i)
                    its += 1
                    if not q["ok"]:
                        break
                    cur = q
                o[ROWS[5]] = dict(ok=bool(cur["ok"]), pose=np.asarray(cur["pose"], dtype=np.float64), sec=sec, iters=its,
                                  frames=[map_nums[i] for i in used])
                if colours is not None:
                    c = passes(fq, prior, lambda T: splat(loc, colours, loc.near(T, 3), T, occl=occl))
                    o[ROWS[6]] = dict(ok=c["ok"], pose=c["pose"], sec=t0 + c["seconds"], iters=c["iters"], inliers=c["inliers"])
                return o, num

            from_prior(image(qs[0]), prior_pose(given[qs[0]], levels[0][0], levels[0][1], scale, seed, E.fnum(qs[0])))
            if colours is None:
                out["notes"].append("the map frames are not in the context, the keyframe splat row is left out")
            sweep = []
            gmap = {g["name"]: bin_of(g["gap_t"], T_EDGES) for g in out.get("gaps") or []}
            for lt, lr in levels:
                per = {k: [] for k in ROWS}
                for n in qs:
                    img, ref = image(n), np.asarray(given[n], dtype=np.float64)
                    pts = loc.points[int(loc.near(ref, 1)[0])]
                    o, num = from_prior(img, prior_pose(ref, lt, lr, scale, seed, E.fnum(n)))
                    for k, r in o.items():
                        et, er = err(r["pose"], ref, scale)
                        ev = dict(t=et, r=er, px=G.pixel_shift(loc.K, ref, r["pose"], pts, hw), flow=flow_at(img, r["pose"], num))
                        if n in ref_b:
                            ev["t_b"], ev["r_b"] = err(r["pose"], ref_b[n], scale)
                        per[k].append(dict(r, name=n, ev=ev))
                rows = []
                for k in ROWS:
                    if not per[k]:
                        continue
                    s = C.summary(per[k], warm, depth_m)
                    e = [(x["ev"]["t"], x["ev"]["r"]) for x in per[k]]
                    eb = [(x["ev"]["t_b"], x["ev"]["r_b"]) for x in per[k] if "t_b" in x["ev"]]
                    s.update(label=k, success=rate(e, succ), success_b=rate(eb, succ), n_b=len(eb),
                             returned=int(sum(bool(x["ok"]) for x in per[k])))
                    if gmap:
                        s["by_gap"] = [dict(n=len(sel), success=rate(sel, succ)) for sel in
                                       ([(x["ev"]["t"], x["ev"]["r"]) for x in per[k] if gmap.get(x["name"]) == b] for b in range(len(T_EDGES) + 1))]
                    rows.append(s)
                sweep.append(dict(t=lt, r=lr, rows=rows, per_query={k: [dict(x["ev"], name=x["name"], ok=bool(x["ok"]), sec=x["sec"], iters=x.get("iters"))
                                                                        for x in v] for k, v in per.items() if v}))
                print("[gap] prior error %g %% of depth, %g deg: %s" % (lt, lr, " | ".join("%s %s" % (r["label"], C.fmt(r["success"], "%.0f %%")) for r in rows)),
                      flush=True)
                out["priors"] = sweep
                save()
            labels = [k for k in ROWS if any(any(r["label"] == k for r in x["rows"]) for x in sweep)]
            out["basin"] = []
            for k in labels:
                rates = [next((r["success"] for r in x["rows"] if r["label"] == k), None) for x in sweep]
                out["basin"].append(dict(label=k, **basin(levels, rates, need)))
        except Exception as e:
            out["notes"].append("the navigation prior sweep failed (%s, %s)" % (type(e).__name__, e))
    by = {r["label"]: r for r in (res or {}).get("rows") or []}
    pq_all = (res or {}).get("per_query") or {}
    out["global"] = []
    pick = [names.index(n) for n in qs]
    for key, label in ((FREE_KEYS[0], FREE[0]), (FREE_KEYS[2], FREE[2])):
        pq = pq_all.get(key)
        if key not in by or not pq or len(pq) != len(names):
            continue
        use = [j for j in pick if pq[j].get("t") is not None and pq[j].get("r") is not None]
        e = [(pq[j]["t"], pq[j]["r"]) for j in use]
        eb = [err(np.asarray(pq[j]["pose"], dtype=np.float64), ref_b[names[j]], scale) for j in pick if pq[j].get("pose") is not None and names[j] in ref_b]
        out["global"].append(dict(label=label, n=len(e), success=rate(e, succ), success_b=rate(eb, succ), n_b=len(eb), s_mean=by[key].get("s_mean")))
    out = P.clean(out)
    save()
    return out


def fmt(v, spec):
    return C.fmt(v, spec)


def report_gap(out):
    L = ["=" * W, " Swimm3R localization, viewpoint gaps and a simulated navigation prior, with the sensitivity of the scores to the reference", "-" * W]
    L.append(" split       %d map frames, %d queries, %d with reference A (scored in the main table), %d without" % (
        out["map_frames"], out["queries"], out["scored"], out["free"]))
    fb = out.get("fit_b") or {}
    fa = out.get("fit_a") or {}
    mn = "LoFTR" if "LoFTR" in str(fa.get("matcher") or "") else (fa.get("matcher") or "our backbone")
    L.append(" Reference A the reference of the main table, the matches of %s lifted by the Stage 1 points of the map, the same 3D the no render rows use" % mn)
    if fb:
        how = ("the same matches of %s, kept from reference A and not matched again" % mn if fb.get("matcher_calls") == 0 else
               "%s run again on the same image pairs (%d calls, %d pairs kept from reference A)" % (mn, fb.get("matcher_calls", 0), fb.get("reused", 0)))
        L += [" Reference B %s," % how,
              "             lifted by %s." % fb.get("lift", "the depth of the Gaussian map"),
              "             The same check as A (the registrations on the earlier and on the later map frames alone agree within %.1f px). The support covers" % fb.get("tol", 2.0),
              "             %s of the pixels of the map frames. A for %d, B for %d of %d queries." % (
                  fmt(None if fb.get("support") is None else 100.0 * fb["support"], "%.0f %%"), out["ref_a_n"], out["ref_b_n"], out["queries"]),
              "             B keeps the Stage 1 poses of the map cameras and the matches of A and changes only the 3D, so it is a sensitivity check and"
              " not an independent reference."]
        d = out.get("diff")
        if d:
            L.append("             A and B differ by a median of %.2f %% of depth and %.2f deg (p90 %.2f %%, %.2f deg), map points shift by %s px (p90 %s), %d queries." % (
                d["t_med"], d["r_med"], d["t_p90"], d["r_p90"], fmt(d["px_med"], "%.2f"), fmt(d["px_p90"], "%.2f"), d["n"]))
        dc, fc = out.get("diff_common"), out.get("fit_common") or {}
        if fc:
            L.append("             With support only where Stage 1 has a point as well, B for %d queries%s" % (
                fc.get("n", 0), ", A and B differ by a median of %.2f %% of depth and %.2f deg (p90 %.2f %%, %.2f deg)." % (
                    dc["t_med"], dc["r_med"], dc["t_p90"], dc["r_p90"]) if dc else "."))
        L.append("             Neither is ground truth. A row that is close to one reference and far from the other shares the geometry of that reference.")
        rows = out.get("both") or []
        if rows:
            hc = "%5s %5s %5s %6s %5s" % ("5/5", "2/2", "1/1", "t %", "deg")
            cell = lambda a: "%5.1f %5.1f %5.1f %6.2f %5.2f" % (a["acc"][1], a["acc"][2], a["acc"][3], a["t"], a["r"])
            L.append("   every row of the main table on the queries both references have, share within 5 % and 5 deg, 2 % and 2 deg, 1 % and 1 deg, medians")
            L.append("   %-38s %4s | A %s | B %s" % ("method", "n", hc, hc))
            for r in rows:
                L.append("   %-38s %4d | A %s | B %s" % (r["label"], r["n"], cell(r["a"]), cell(r["b"])))
        if out.get("missing_poses"):
            L.append("   %d rows are not scored against B, gscpe.json keeps no pose per query for them" % len(out["missing_poses"]))
    fg = out.get("frame_gaps") or []
    if fg:
        L += ["-" * W, " Frame gap   every query by its distance in frames to the nearest map frame, and how many of them have reference A and B. Where the share",
              "             with A falls with the distance, the scored queries lean to small gaps and the rows below see fewer of the far queries."]
        for g in fg:
            L.append("   %3d %-6s from a map frame %5d queries | with A %4d | with B %4d" % (g["dist"], "frame" if g["dist"] == 1 else "frames", g["n"], g["a"], g["b"]))
    fr = out.get("no_ref")
    if fr and fr.get("rows"):
        L += ["-" * W, " No reference %d queries whose registration did not pass the check, so no error against A. flow is the median optical flow in px between" % len(fr["names"]),
              "             the query and the underwater render at the estimate, no reference involved. Against B where B has a pose."]
        L.append("   %-38s %11s | %8s %8s | B %3s %5s %6s %5s" % ("method", "returned", "flow px", "s mean", "n", "5/5", "t %", "deg"))
        for r in fr["rows"]:
            b = r.get("b")
            L.append("   %-38s %4d of %3d | %8s %8s | B %s" % (r["label"], r["returned"], r["n"], fmt(r["flow"], "%.2f"), fmt(r["s_mean"], "%.3f"),
                                                            "%3d %5.1f %6.2f %5.2f" % (b["n"], b["acc"][1], b["t"], b["r"]) if b else "  -"))
        cmp_ = fr.get("compare") or {}
        if cmp_:
            L.append("   for comparison, flow on the queries with a reference " + " | ".join("%s %s px" % (k, fmt(v, "%.2f")) for k, v in cmp_.items()))
    st = out.get("strata")
    if st:
        L += ["-" * W, " Gap strata  every scored query by its distance to the nearest map frame (the frame the localizer calls nearest), errors against A.",
              "             Each cell gives the queries in the bin, the share within 2 % and 2 deg, within 5 % and 5 deg, and the median translation error in % of depth",
              "             (translation bins) or the median rotation error in deg (rotation bins)."]
        for key, edges, unit, col in (("t_bins", out["t_edges"], "%", "t"), ("r_bins", out["r_edges"], "deg", "r")):
            nm = bin_names(edges, unit)
            L.append("   %-38s" % ("by translation gap" if key == "t_bins" else "by rotation gap") + "".join(" | %-22s" % x for x in nm))
            for r in st:
                cells = []
                for c in r[key]:
                    a = c["a"]
                    cells.append(" | %3d %5.1f %5.1f %6.2f" % (c["n"], a["acc"][2], a["acc"][1], a[col]) if a else " | %3d %18s" % (c["n"], "-"))
                L.append("   %-38s%s" % (r["label"], "".join(cells)))
    sw = out.get("priors")
    if sw:
        su = out["success"]
        L += ["-" * W, " Navigation prior, simulated (sim. prior). The start pose is reference A plus a seeded error of the stated size (seed %d). The direction" % out["seed"],
              "             of the error is drawn once per query and only its size changes from level to level. Every method starts from the same prior.",
              "             It stands in for a navigation estimate and is not one. No method sees the reference otherwise. Only queries with reference A",
              "             take part, see the frame gap above for those that are missing.",
              "             Success means within %g %% of depth and %g deg of reference A (B in the last column). %d queries." % (su[0], su[1], len(out.get("prior_queries") or [])),
              "   %-38s the underwater render of the map at the prior (water of the map frame nearest the prior), matched with our backbone, lifted" % ROWS[1],
              "   %-38s by the rendered depth, PnP with RANSAC, at most %d passes" % ("", out["coarse_iters"]),
              "   %-38s the same, then direct alignment of the query to the underwater render (locally normalised intensity)" % ROWS[2],
              "   %-38s the query matched to the map frame nearest the prior, lifted by the Stage 1 points, PnP" % ROWS[3],
              "   %-38s the same with each of the 3 map frames nearest the prior, the pose with the most inliers is kept" % ROWS[4],
              "   %-38s the same, the map frame nearest the last estimate is taken again after every PnP, at most %d passes, the counterpart" % (
                  ROWS[5], out["coarse_iters"]),
              "   %-38s of the passes of the render rows" % ""]
        if out.get("splat"):
            L += ["   %-38s no Gaussian. The Stage 1 points of the 3 map frames nearest the pose with the colours of those frames as captured," % ROWS[6],
                  "   %-38s splatted into the view (z buffer, points more than %.0f %% behind the nearest point within 2 px dropped, two hole filling" % (
                      "", 100.0 * float(out.get("splat_occl") or 0.0)),
                  "   %-38s passes), matched with our backbone, lifted by the splat depth, PnP, at most %d passes" % ("", out["coarse_iters"])]
        for x in sw:
            L.append(" prior error %g %% of depth and %g deg" % (x["t"], x["r"]))
            L.append("   %-38s %7s | %5s %5s %5s %5s | %7s %7s %6s %5s | %7s %6s | %7s" % (
                "method", "success", "10/5", "5/5", "2/2", "1/1", "t %", "rot deg", "px", "flow", "s mean", "FPS", "succ. B"))
            for r in x["rows"]:
                L.append("   %-38s %7s | %5.1f %5.1f %5.1f %5.1f | %7.3f %7.3f %6s %5s | %7s %6s | %7s" % (
                    r["label"], fmt(r["success"], "%.1f"), r["acc"][0], r["acc"][1], r["acc"][2], r["acc"][3], r["t"], r["r"], fmt(r.get("px"), "%.2f"),
                    fmt(r.get("flow"), "%.1f"), fmt(r.get("s_mean"), "%.3f"), fmt(r.get("fps"), "%.2f"), fmt(r.get("success_b"), "%.1f")))
            if any(r.get("by_gap") for r in x["rows"]):
                L.append("   %-38s" % "success by the translation gap" + "".join(" | %-14s" % y for y in bin_names(out["t_edges"], "%")))
                for r in x["rows"]:
                    if r.get("by_gap"):
                        L.append("   %-38s%s" % (r["label"], "".join(" | %5s of %3d  " % (fmt(c["success"], "%.0f"), c["n"]) if c["n"] else " | %14s" % "-"
                                                                for c in r["by_gap"])))
        bs = out.get("basin") or []
        if bs:
            L.append(" Basin       largest prior error up to which every level has at least %.0f %% success, translation alone, rotation alone, both together" % out["need"])
            show = lambda v: "none" if v is None else "%g %% and %g deg" % (v[0], v[1])
            for b in bs:
                L.append("   %-38s %s" % (b["label"], " | ".join("%s %s" % (k, show(b[k])) for k in ("translation", "rotation", "both") if k in b)))
        gl = out.get("global") or []
        if gl:
            L.append(" Without a prior, retrieval of the map frame as in the main table, the same %d queries" % gl[0]["n"])
            for g in gl:
                L.append("   %-38s success %s, against B %s, %s s" % (g["label"], fmt(g["success"], "%.1f %%"), fmt(g.get("success_b"), "%.1f %%"),
                                                                    fmt(g["s_mean"], "%.3f")))
    L += ["-" * W, " Time per query runs from the query on the GPU to the pose (encoder, render or splat, matching, PnP, alignment). Image loading and scoring",
          " are not timed. The first %d queries of every row are left out of the time as warm-up. The rows of the prior share one encoder pass of the query," % out["warmup"],
          " its time is counted in every row."]
    for n in out.get("notes", []):
        L.append(" note        %s" % n)
    L.append("=" * W)
    return "\n".join(L)


def in_short_gap(out):
    d = out.get("diff")
    L = [" Gap         reference A for %d queries, B for %d (B lifted by the depth of the Gaussian map)%s" % (
        out.get("ref_a_n", 0), out.get("ref_b_n", 0), ", A and B differ by %.2f %% of depth and %.2f deg (median)" % (d["t_med"], d["r_med"]) if d else "")]
    fr = (out.get("no_ref") or {}).get("rows") or []
    if fr:
        nick = dict(zip(FREE, ("no render", "coarse", "fast + check")))
        L.append("               queries without reference A %d, median flow px of the render at the estimate | %s" % (
            len(out["no_ref"]["names"]), " | ".join("%s %s" % (nick.get(r["label"], r["label"]), fmt(r.get("flow"), "%.2f")) for r in fr)))
    sw = out.get("priors") or []
    if sw:
        su = out["success"]
        L.append("               %-38s %s" % ("sim. prior error, % of depth/deg"," ".join("%6s" % ("%g/%g" % (x["t"], x["r"])) for x in sw)))
        for k in ROWS[1:]:
            v = [next((r.get("success") for r in x["rows"] if r["label"] == k), None) for x in sw]
            if any(x is not None for x in v):
                L.append("               %-38s %s" % (k, " ".join("%6s" % fmt(x, "%.1f") for x in v)))
        L.append("               share of the queries within %g %% of depth and %g deg of reference A, %d queries per prior error" % (
            su[0], su[1], len(out.get("prior_queries") or [])))
    return L
