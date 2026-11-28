import numpy as np
import torch

import eval_gaussian_map as E
import loc_geom as G


def loftr_matcher(dev, pretrained="outdoor", conf_min=0.5):
    try:
        import kornia.feature as KF
    except ImportError as e:
        raise RuntimeError("the LoFTR reference needs kornia in this environment: pip install kornia") from e
    net = KF.LoFTR(pretrained=pretrained).to(dev).eval()

    def grey(x):
        h8, w8 = (int(x.shape[-2]) // 8) * 8, (int(x.shape[-1]) // 8) * 8
        x = x[:, :h8, :w8].float().clamp(0.0, 1.0)
        return (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2])[None, None]

    @torch.no_grad()
    def match(query, frame):
        out = net({"image0": grey(query.to(dev)), "image1": grey(frame.to(dev))})
        keep = out["confidence"] >= conf_min
        return out["keypoints0"][keep].double().cpu().numpy(), out["keypoints1"][keep].double().cpu().numpy()

    match.name = "LoFTR (%s weights of kornia, confidence at least %.2f)" % (pretrained, conf_min)
    return match


def lift(loc, i, pq, pm):
    if len(pm) == 0:
        return np.zeros((0, 2)), np.zeros((0, 3))
    u = torch.as_tensor(np.clip(np.round(pm[:, 0]), 0, loc.w - 1).astype(np.int64), device=loc.xyz.device)
    v = torch.as_tensor(np.clip(np.round(pm[:, 1]), 0, loc.h - 1).astype(np.int64), device=loc.xyz.device)
    ok = loc.valid[i][v, u].cpu().numpy()
    return np.ascontiguousarray(pq[ok], dtype=np.float64), loc.xyz[i][v, u][torch.as_tensor(ok, device=loc.xyz.device)].double().cpu().numpy()


def depth_lifter(loc, rgbd, min_alpha=0.5, common=False, store=None):
    maps = {}
    raw = {} if store is None else store

    def get(i):
        if i not in maps:
            if i not in raw:
                with torch.no_grad():
                    _, al, dp = rgbd(i)[:3]
                raw[i] = (dp.reshape(loc.h, loc.w).float(), al.reshape(loc.h, loc.w).float())
            dp, al = raw[i]
            ok = (al > min_alpha) & torch.isfinite(dp) & (dp > 1e-6)
            if common:
                ok = ok & loc.valid[i].to(ok.device)
            maps[i] = (dp, ok)
        return maps[i]

    def run(lz, i, pq, pm):
        if len(pm) == 0:
            return np.zeros((0, 2)), np.zeros((0, 3))
        dp, ok = get(int(i))
        u = torch.as_tensor(np.clip(np.round(pm[:, 0]), 0, lz.w - 1).astype(np.int64), device=dp.device)
        v = torch.as_tensor(np.clip(np.round(pm[:, 1]), 0, lz.h - 1).astype(np.int64), device=dp.device)
        keep = ok[v, u]
        z = dp[v, u][keep].double()
        uu, vv = u[keep].double(), v[keep].double()
        K = lz.K
        c = torch.as_tensor(np.asarray(lz.poses[int(i)]), dtype=torch.float64, device=dp.device)
        cam = torch.stack([(uu - float(K[0, 2])) / float(K[0, 0]) * z, (vv - float(K[1, 2])) / float(K[1, 1]) * z, z], -1)
        return np.ascontiguousarray(pq[keep.cpu().numpy()], dtype=np.float64), (cam @ c[:3, :3].T + c[:3, 3]).cpu().numpy()

    run.name = "the depth of the Gaussian map rendered at the map camera (restored render, opacity above %.1f%s)" % (
        min_alpha, ", only where Stage 1 has a point too" if common else "")
    run.maps = maps
    return run


def register_queries(loc, tests, load, map_nums, interp, k=6, tol=2.0, min_inliers=50, min_ratio=0.3, verbose=True, matcher=None, images=None,
                     lifter=None, cache=None, tag="ref"):
    if lifter is not None and matcher is None:
        raise ValueError("lifting by another depth needs an outside matcher, the matcher of the map lifts by its own points")
    up = lift if lifter is None else lifter
    nums = [E.fnum(x) for x in tests]
    map_nums = [int(x) for x in map_nums]
    hw = (loc.h, loc.w)
    k = max(2, int(k))
    sub = loc.subsample
    loc.subsample = max(2, int(sub) // 2)
    poses, rows = {}, []
    calls = [0, 0]
    try:
        for j, name in enumerate(tests):
            lo = sorted([i for i in range(loc.n) if map_nums[i] < nums[j]], key=lambda i: nums[j] - map_nums[i])[:k // 2]
            hi = sorted([i for i in range(loc.n) if map_nums[i] > nums[j]], key=lambda i: map_nums[i] - nums[j])[:k - k // 2]
            if not lo or not hi:
                order = sorted(range(loc.n), key=lambda i: abs(map_nums[i] - nums[j]))[:k]
                lo, hi = order[0::2], order[1::2]
            start = np.asarray(interp[j], dtype=np.float64)
            img = load(name)
            if matcher is None:
                fq = loc.encode(img)
                a2, a3 = loc.match(fq, lo)[:2]
                b2, b3 = loc.match(fq, hi)[:2]
            else:
                def pair(i, name=name, img=img):
                    key = (name, int(i))
                    if cache is not None and key in cache:
                        calls[1] += 1
                        return cache[key]
                    calls[0] += 1
                    got = matcher(img, images[i])
                    if cache is not None:
                        cache[key] = got
                    return got

                def side(ids):
                    got = [up(loc, i, *pair(i)) for i in ids]
                    return np.concatenate([g[0] for g in got]), np.concatenate([g[1] for g in got])
                a2, a3 = side(lo)
                b2, b3 = side(hi)
            p2, p3 = np.concatenate([a2, b2]), np.concatenate([a3, b3])
            solve = lambda q3, q2: G.solve_pnp(q3, q2, loc.K, loc.reproj, start) if len(q2) >= 6 else None
            s, sa, sb = solve(p3, p2), solve(a3, a2), solve(b3, b2)
            good = all(x is not None and bool(np.isfinite(x["c2w"]).all()) for x in (s, sa, sb))
            row = dict(frame=nums[j], name=name, frames=[map_nums[i] for i in lo + hi], matches=int(len(p2)), inliers=int(s["inliers"]) if s else 0,
                       half_px=None, half_t=None, half_r=None, verified=False)
            if good:
                near = int(loc.near(s["c2w"], 1)[0])
                pts, depth = loc.points[near], float(loc.depths[near])
                px = G.pixel_shift(loc.K, sa["c2w"], sb["c2w"], pts, hw)
                et, er = G.pose_error(sa["c2w"], sb["c2w"], depth)
                row.update(half_px=float(px), half_t=float(et), half_r=float(er))
                row["verified"] = bool(np.isfinite(px) and px <= tol and s["inliers"] >= max(min_inliers, min_ratio * len(p2)))
                if row["verified"]:
                    poses[name] = np.asarray(s["c2w"], dtype=np.float64)
            rows.append(row)
            if verbose:
                print("[%s] frame %06d: registered on the map frames %s with %d inliers of %d matches | earlier and later map frames alone differ by %s | %s" % (
                    tag, nums[j], row["frames"], row["inliers"], row["matches"],
                    "%.2f px (%.2f %% of depth, %.2f deg)" % (row["half_px"], row["half_t"], row["half_r"]) if row["half_px"] is not None else "no pose",
                    "verified" if row["verified"] else "not verified"), flush=True)
    finally:
        loc.subsample = sub
    done = [r for r in rows if r["half_px"] is not None and np.isfinite(r["half_px"])]
    col = lambda key, q: float(np.percentile([r[key] for r in done], q)) if done else None
    fit = dict(register=True, k=k, tol=float(tol), n=len(poses), total=len(tests), half_px_med=col("half_px", 50), half_px_p90=col("half_px", 90),
               half_t_med=col("half_t", 50), half_r_med=col("half_r", 50), inliers_med=float(np.median([r["inliers"] for r in rows])) if rows else None,
               matches_med=float(np.median([r["matches"] for r in rows])) if rows else None, grid=int(max(2, int(sub) // 2)),
               matcher=getattr(matcher, "name", "matcher") if matcher is not None else None)
    if lifter is not None:
        fit["lift"] = getattr(lifter, "name", "another depth")
    if cache is not None:
        fit.update(matcher_calls=calls[0], reused=calls[1])
    return poses, fit, rows


def reference_lines(fit, text=None, skipped=0):
    f = fit or {}
    gone = " %d queries without a reference are left out." % skipped if skipped else ""
    if f.get("register"):
        num = lambda key, spec: "-" if f.get(key) is None else spec % f[key]
        if f.get("matcher") and f.get("lift"):
            how = ["             changed. The query is matched to the %d map frames nearest in time with %s," % (f.get("k", 0), f["matcher"]),
                   "             a matcher none of the methods below uses. The matches are lifted by %s," % f["lift"],
                   "             PnP with RANSAC and a refit on"]
        elif f.get("matcher"):
            how = ["             changed: the query is matched to the %d map frames nearest in time with %s," % (f.get("k", 0), f["matcher"]),
                   "             a matcher none of the methods below uses; the matches are lifted by the points of the map, PnP with RANSAC and a refit on"]
        else:
            how = ["             changed: the query is matched to the %d map frames nearest in time (grid every %d px), the matches are lifted by the points of" % (
                       f.get("k", 0), f.get("grid", 0)),
                   "             the map, PnP with RANSAC and a refit on"]
        how[-1] += " all matches gives the pose (median %s inliers of %s matches). A reference counts when the" % (
            num("inliers_med", "%.0f"), num("matches_med", "%.0f"))
        return ([" Reference   every query registered into the fixed map after the mapping. The map is built from the map frames alone and nothing of it is"]
                + how
                + ["             registrations on the earlier and on the later map frames alone agree within %.1f px: %d of %d queries.%s" % (
                       f.get("tol", float("nan")), f.get("n", 0), f.get("total", 0), gone),
                   "             The two differ by a median of %s px (p90 %s), %s %% of depth and %s deg, which does not bound an error both share." % (
                       num("half_px_med", "%.2f"), num("half_px_p90", "%.2f"), num("half_t_med", "%.2f"), num("half_r_med", "%.2f"))])
    out = [" Reference   %s" % (text or "poses of a reconstruction of all frames, the queries included")]
    if f.get("same"):
        out.append("             same frame and intrinsics as the map, nothing moved between reconstructions (focal of the map frames %.2f, of the queries %.2f).%s" % (
            f.get("focal_map", float("nan")), f.get("focal_queries", float("nan")), gone))
    else:
        out += ["             %s on the %d map frames (scale %.3f): after the fit the map frames differ by a median of" % (
                    "one similarity fitted" if f.get("local") else "moved into the map by a similarity fitted", f.get("n", 0), f.get("scale", float("nan"))),
                "             %.2f %% of depth and %.2f deg (largest %.2f %%, %.2f deg), which does not bound the error of a query pose.%s" % (
                    f.get("t_med", float("nan")), f.get("r_med", float("nan")), f.get("t_max", float("nan")), f.get("r_max", float("nan")), gone)]
        if f.get("depth_common"):
            out.append("             Depth for the percentages: the median depth of the reference frames %.4f times the scale = %.4f map units, the same for every" % (
                f["depth_ref"], f["depth_common"]))
            out.append("             method (the median depth of this map would be %.4f)." % f.get("depth_map", float("nan")))
        lo = f.get("local")
        if lo:
            out.append("             Query by query: each reference pose moved in by a similarity on the %d map frames nearest to it in the reference" % lo["k"])
            out.append("             (chosen from the reference poses alone), the nearest %.2f %% median of depth away, the %d th %.2f %% median, %.2f %% max;" % (
                lo["near_med"], lo["k"], lo["far_med"], lo["far_max"]))
            out.append("             local scale %.2f %% to %.2f %% (p5 to p95) from the scale on all map frames, %d queries fell back to the one similarity." % (
                lo["scale_p5"], lo["scale_p95"], lo["fallback"]))
        h = f.get("holdout")
        if h and h.get("local"):
            out.append("             Held out map frames (each left out in turn, similarity on its %d nearest map frames): %.2f %% median, %.2f %% p95, %.2f %% max" % (
                h["local"], h["t_med"], h["t_p95"], h["t_max"]))
            out.append("             of depth, %.2f / %.2f / %.2f deg; accuracy thresholds below this are not resolved by the reference." % (
                h["r_med"], h["r_p95"], h["r_max"]))
            g = f.get("holdout_global")
            if g:
                out.append("             With one similarity on all map frames (3 of %d time blocks): %.2f %% median, %.2f %% p95 of depth, %.2f / %.2f deg." % (
                    g["blocks"], g["t_med"], g["t_p95"], g["r_med"], g["r_p95"]))
        elif h:
            out.append("             Held out map frames (similarity fitted on 3 of %d time blocks, tested on the rest): %.2f %% median, %.2f %% p95, %.2f %% max of depth," % (
                h["blocks"], h["t_med"], h["t_p95"], h["t_max"]))
            out.append("             %.2f / %.2f / %.2f deg, scale spread %.2f %% over the blocks; accuracy thresholds below this are not resolved by the reference." % (
                h["r_med"], h["r_p95"], h["r_max"], h["scale_spread"]))
    return out
