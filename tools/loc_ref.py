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


def register_queries(loc, tests, load, map_nums, interp, k=6, tol=2.0, min_inliers=50, min_ratio=0.3, verbose=True, matcher=None, images=None):
    nums = [E.fnum(x) for x in tests]
    map_nums = [int(x) for x in map_nums]
    hw = (loc.h, loc.w)
    k = max(2, int(k))
    sub = loc.subsample
    loc.subsample = max(2, int(sub) // 2)
    poses, rows = {}, []
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
                def side(ids):
                    got = [lift(loc, i, *matcher(img, images[i])) for i in ids]
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
                print("[ref] frame %06d: registered on the map frames %s with %d inliers of %d matches | earlier and later map frames alone differ by %s | %s" % (
                    nums[j], row["frames"], row["inliers"], row["matches"],
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
    return poses, fit, rows


def reference_lines(fit, text=None, skipped=0):
    f = fit or {}
    gone = " %d queries without a reference are left out." % skipped if skipped else ""
    if f.get("register"):
        num = lambda key, spec: "-" if f.get(key) is None else spec % f[key]
        if f.get("matcher"):
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
                   "             The two differ by a median of %s px (p90 %s), %s %% of depth and %s deg: the floor of every pose error below." % (
                       num("half_px_med", "%.2f"), num("half_px_p90", "%.2f"), num("half_t_med", "%.2f"), num("half_r_med", "%.2f"))])
    out = [" Reference   %s" % (text or "poses of a reconstruction of all frames, the queries included")]
    if f.get("same"):
        out.append("             same frame and intrinsics as the map, nothing moved between reconstructions (focal of the map frames %.2f, of the queries %.2f).%s" % (
            f.get("focal_map", float("nan")), f.get("focal_queries", float("nan")), gone))
    else:
        out += ["             moved into the map by a similarity fitted on the %d map frames (scale %.3f): after the fit the map frames differ by a median of" % (
                    f.get("n", 0), f.get("scale", float("nan"))),
                "             %.2f %% of depth and %.2f deg (largest %.2f %%, %.2f deg), the floor of every pose error below.%s" % (
                    f.get("t_med", float("nan")), f.get("r_med", float("nan")), f.get("t_max", float("nan")), f.get("r_max", float("nan")), gone)]
    return out
