import json
import os
import resource
import sys
import time

import numpy as np
import torch

import bench_loc as B
import eval_gaussian_map as E
import loc_geom as G

LEVELS = ((2.0, 1.0), (5.0, 2.0), (10.0, 5.0), (20.0, 10.0))
BUDGETS = (0.5, 1.0, 2.0, 5.0, 25.0)
W = 122


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    return o


def agg(rows, k, warmup=0, px_tol=2.0):
    ev = [r["ev"][k] for r in rows if k in r["ev"]]
    if not ev:
        return None
    out = {"n": len(ev)}
    for m in ("psnr", "ssim", "lpips"):
        out[m] = float(np.mean([e[m] for e in ev]))
    fl = [e["flow"] for e in ev if e.get("flow") is not None and np.isfinite(e["flow"])]
    out["flow"] = float(np.median(fl)) if fl else None
    pe = [e for e in ev if "et" in e]
    if pe:
        et, er = np.array([e["et"] for e in pe]), np.array([e["er"] for e in pe])
        px = np.array([e["px"] for e in pe], dtype=np.float64)
        out.update(n_ref=len(pe), et=float(np.median(et)), er=float(np.median(er)),
                   px=float(np.nanmedian(px)) if np.isfinite(px).any() else None,
                   s5=float(100.0 * np.mean((et <= 5.0) & (er <= 5.0))), sp=float(100.0 * np.mean(np.nan_to_num(px, nan=np.inf) <= px_tol)))
    timed = [r["sec"][k] for r in rows[min(warmup, max(len(rows) - 1, 0)):] if k in r.get("sec", {})]
    if timed:
        s = np.asarray(timed)
        out.update(ms_med=float(1e3 * np.median(s)), ms_p90=float(1e3 * np.percentile(s, 90)), ms_mean=float(1e3 * s.mean()), fps=float(1.0 / s.mean()))
    return out


def run(ctx):
    dev, loc, fps = ctx["dev"], ctx["loc"], float(ctx["fps"])
    tests = ctx["tests"]
    nums = [E.fnum(x) for x in tests]
    order = [int(j) for j in np.argsort(nums)]
    centre = set(E.fnum(x) for x in ctx.get("centre", []))
    map_pose = {int(n): p for n, p in zip(ctx["map_nums"], ctx["map_poses"])}
    hw = (loc.h, loc.w)
    warm = int(ctx.get("warmup", 2))
    px_tol, ref_tol = float(ctx.get("px_tol", 2.0)), float(ctx.get("ref_tol", 2.0))
    cache = {}
    whole_peak = torch.cuda.max_memory_allocated() if dev.type == "cuda" else 0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tegra = B.Tegra()
    res = {"frames": len(order), "interior": len([j for j in order if nums[j] in centre]), "tracking": [], "reloc": [], "notes": []}

    def save():
        try:
            json.dump(clean(res), open(os.path.join(ctx["out_dir"], "protocol.json"), "w"), indent=1)
        except Exception as e:
            print("[proto] protocol.json not written: %s" % e, flush=True)

    def image(j):
        if j not in cache:
            cache[j] = ctx["load"](tests[j])
        return cache[j]

    def tens(p):
        return torch.tensor(np.asarray(p), dtype=torch.float32, device=dev)

    def medium_timed(num, p):
        B.sync(dev)
        a = time.time()
        m = ctx["medium"](num, p)
        B.sync(dev)
        return m, time.time() - a

    def steps_from(j, pose, marks, num=None):
        num = nums[j] if num is None else int(num)
        p0 = tens(pose)
        med, tm = medium_timed(num, p0)
        ps, ss = B.track(ctx["render"], image(j), p0, med, num / fps, ctx["unit"], ctx["lr"], list(marks), dev)
        return {k: ps[k].cpu().numpy().astype(np.float64) for k in marks}, {k: tm + ss[k] for k in marks}

    def evaluate(j, pose, ref):
        num, gt = nums[j], image(j)
        p = tens(pose)
        with torch.no_grad():
            img, alpha = ctx["render"](p, ctx["medium"](num, p), t=num / fps, alpha=True)
            sc = ctx["score"](img, gt)
            flow = G.flow_residual(gt.permute(1, 2, 0).cpu().numpy(), img.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy(),
                                   (alpha > 0.5).cpu().numpy())
        out = dict(psnr=sc[0], ssim=sc[1], lpips=sc[2], flow=flow)
        r = ref.get(j)
        if r and r["verified"]:
            et, er = G.pose_error(pose, r["pose"], r["depth"])
            out.update(et=et, er=er, px=G.pixel_shift(loc.K, r["pose"], pose, r["points"], hw))
        return out

    def gap_of(j, ref):
        r = ref.get(j)
        if not r or not r["verified"]:
            return None
        et, er = G.pose_error(loc.poses[r["near"]], r["pose"], r["depth"])
        seen = max(G.covisible(loc.K, r["pose"], loc.poses[k], r["points"], hw) for k in loc.near(r["pose"], 6))
        return dict(t=et, r=er, seen=float(seen))

    def reference():
        given = ctx.get("ref_poses")
        if given:
            ref, t0 = {}, time.time()
            for j in order:
                known = tests[j] in given
                p = np.asarray(given[tests[j]] if known else ctx["interp"][j], dtype=np.float64)
                near = int(loc.near(p, 1)[0])
                depth, pts = float(loc.depths[near]), loc.points[near]
                r = loc.localize(image(j), prior=p, idx=loc.near(p, int(ctx["ref_frames"])))
                geo = np.asarray(r["pose"], dtype=np.float64)
                pn = list(G.pose_error(geo, p, depth)) + [G.pixel_shift(loc.K, p, geo, pts, hw)]
                ref[j] = dict(pose=p, geo=geo, near=near, depth=depth, points=pts, verified=known, check=[0.0, 0.0, 0.0], pnp=pn,
                              pnp_ok=bool(r["ok"]), matches=r["matches"], inliers=r["inliers"], frame=nums[j])
            return ref, time.time() - t0
        path = os.path.join(ctx["out_dir"], "reference.json")
        key = dict(steps=int(ctx["ref_steps"]), check=int(ctx["ref_check"]), frames=int(ctx["ref_frames"]), tol=ref_tol,
                   tests=[tests[j] for j in order], lr=float(ctx["lr"]), map_frames=int(loc.n))
        if os.path.isfile(path):
            try:
                d = json.load(open(path))
                if d.get("key") == key:
                    print("[proto] reference poses read from %s" % path, flush=True)
                    return {int(k): dict(v, pose=np.asarray(v["pose"]), geo=np.asarray(v["geo"]), points=loc.points[int(v["near"])])
                            for k, v in d["frames"].items()}, 0.0
            except Exception:
                pass
        ref = {}
        t0 = time.time()
        for j in order:
            num, gt = nums[j], image(j)
            kf = sorted(range(loc.n), key=lambda i: abs(ctx["map_nums"][i] - num))[:int(ctx["ref_frames"])]
            r = loc.localize(gt, prior=ctx["interp"][j], idx=kf)
            geo = np.asarray(r["pose"], dtype=np.float64)
            pa = steps_from(j, geo, [int(ctx["ref_steps"])])[0][int(ctx["ref_steps"])] if ctx["ref_steps"] > 0 else geo
            pb = steps_from(j, ctx["interp"][j], [int(ctx["ref_check"])])[0][int(ctx["ref_check"])] if ctx["ref_check"] > 0 else pa
            near = int(loc.near(pa, 1)[0])
            depth = float(loc.depths[near])
            pts = loc.points[near]
            ck = list(G.pose_error(pb, pa, depth)) + [G.pixel_shift(loc.K, pa, pb, pts, hw)]
            pn = list(G.pose_error(geo, pa, depth)) + [G.pixel_shift(loc.K, pa, geo, pts, hw)]
            ok = bool(r["ok"]) and bool(np.isfinite(ck[2])) and ck[2] <= ref_tol and ck[0] <= 5.0 and ck[1] <= 5.0
            ref[j] = dict(pose=pa, geo=geo, near=near, depth=depth, points=pts, verified=ok, check=ck, pnp=pn, pnp_ok=bool(r["ok"]),
                          matches=r["matches"], inliers=r["inliers"], frame=num)
            print("[proto] reference frame %06d: %d matches on %d map frames, %d inliers | second run differs by %.2f px (%.2f %% of depth, %.2f deg) | "
                  "PnP differs by %.2f px (%.2f %%, %.2f deg) | %s" % (num, r["matches"], len(kf), r["inliers"], ck[2], ck[0], ck[1], pn[2], pn[0], pn[1],
                                                                       "verified" if ok else "not verified"), flush=True)
        try:
            json.dump(clean(dict(key=key, frames={j: {k: v for k, v in r.items() if k != "points"} for j, r in ref.items()})), open(path, "w"), indent=1)
        except Exception as e:
            print("[proto] reference.json not written: %s" % e, flush=True)
        return ref, time.time() - t0

    def pass_ff(lz, mode, topk, marks, ref, label):
        known, rows = dict(map_pose), []
        w0 = time.time()
        for j in order:
            num = nums[j]
            prior = B.predict_pose(known, num) if mode == "track" else None
            r = lz.localize(image(j), prior=prior, topk=topk)
            poses, sec = {0: np.asarray(r["pose"], dtype=np.float64)}, {0: r["seconds"]}
            if marks:
                ps, ss = steps_from(j, poses[0], marks, None if mode == "track" else ctx["map_nums"][r["idx"][0]])
                for k in marks:
                    poses[k], sec[k] = ps[k], r["seconds"] + ss[k]
            known[num] = poses[max(poses)]
            ev = {k: evaluate(j, poses[k], ref) for k in poses}
            rows.append(dict(frame=num, ok=bool(r["ok"]), idx=r["idx"], matches=r["matches"], inliers=r["inliers"], stages=r["stages"],
                             solver=r["solver"], sec=sec, ev=ev, gap=gap_of(j, ref)))
            print("[proto] %s, frame %06d: %d inliers of %d matches in %.0f ms, LPIPS %.3f%s" % (
                label, num, r["inliers"], r["matches"], 1e3 * r["seconds"], ev[0]["lpips"],
                ", %.2f %% of depth and %.2f deg from the reference" % (ev[0]["et"], ev[0]["er"]) if "et" in ev[0] else ""), flush=True)
        return rows, (w0, time.time())

    def pass_rc(start, marks, ref, label):
        known, rows = dict(map_pose), []
        w0 = time.time()
        for j in order:
            num = nums[j]
            p0, pre, extra = start(j, known)
            ps, ss = steps_from(j, p0, marks, ctx["map_nums"][extra["idx"][0]] if "idx" in extra else None)
            poses = dict(ps)
            poses[0] = np.asarray(p0, dtype=np.float64)
            known[num] = poses[marks[-1]]
            ev = {k: evaluate(j, poses[k], ref) for k in poses}
            rows.append(dict(frame=num, ok=True, sec={k: pre + ss[k] for k in marks}, ev=ev, **extra))
            print("[proto] %s, frame %06d: %d steps in %.2f s, LPIPS %.3f -> %.3f" % (
                label, num, marks[-1], pre + ss[marks[-1]], ev[0]["lpips"], ev[marks[-1]]["lpips"]), flush=True)
        return rows, (w0, time.time())

    def line(label, family, rows, k, window, extra=None):
        a = agg(rows, k, warm, px_tol)
        if a is None:
            return None
        a.update(label=label, family=family, fails=int(sum(1 for r in rows if not r.get("ok", True))))
        p = B.power_summary(tegra.window(*window)) if window else None
        if p and "ms_mean" in a:
            a.update(power_w=p["total_w"], joule=p["total_w"] * a["ms_mean"] / 1e3)
        if extra:
            a.update(extra)
        return a

    def stages(rows):
        keys = ("encoder", "select", "decoder", "matching", "pnp")
        use = rows[min(warm, max(len(rows) - 1, 0)):]
        return {k: float(1e3 * np.mean([r["stages"].get(k, 0.0) for r in use])) for k in keys}

    j0 = order[0]
    for lz in [loc] + ([ctx["loc_public"]] if ctx.get("loc_public") is not None else []):
        r0 = lz.localize(image(j0), prior=ctx["interp"][j0], topk=1)
        e0 = evaluate(j0, r0["pose"], {})
        print("[proto] first frame with %s: %d inliers of %d matches (%s), %.0f ms, LPIPS %.3f, flow %s" % (
            lz.name, r0["inliers"], r0["matches"], r0["solver"], 1e3 * r0["seconds"], e0["lpips"],
            "%.2f px" % e0["flow"] if np.isfinite(e0["flow"]) else "not available"), flush=True)
    steps_from(j0, ctx["interp"][j0], [2])

    def med_cols(a, q=50):
        return [float(np.nanpercentile(a[:, i], q)) if np.isfinite(a[:, i]).any() else None for i in range(a.shape[1])]

    ref, ref_seconds = reference()
    ver = [j for j in order if ref[j]["verified"]]
    num_row = lambda v: [np.nan if x is None else float(x) for x in v]
    ck = np.array([num_row(ref[j]["check"]) for j in order], dtype=np.float64)
    pn = np.array([num_row(ref[j]["pnp"]) for j in order if ref[j]["pnp_ok"]], dtype=np.float64)
    res["reference"] = dict(verified=len(ver), frames=len(order), steps=int(ctx["ref_steps"]), check_steps=int(ctx["ref_check"]),
                            map_frames=int(ctx["ref_frames"]), tol=ref_tol, seconds=ref_seconds,
                            check_median=med_cols(ck), check_p90=med_cols(ck, 90),
                            pnp_median=med_cols(pn) if len(pn) else None, pnp_ok=int(len(pn)),
                            depth_median=float(np.median([ref[j]["depth"] for j in order])), given=bool(ctx.get("ref_poses")), fit=ctx.get("ref_fit"))
    tasks = set(ctx.get("tasks") or ["track", "reloc", "conv", "exact", "thinned"])
    save()
    if ctx.get("only_reference"):
        tegra.stop()
        return clean(res)

    def passes():
        rc_marks = sorted(set(int(k) for k in ctx["rc_steps"] if int(k) > 0))
        ff_marks = sorted(set(int(k) for k in ctx["ff_steps"] if int(k) > 0))
        if "track" in tasks:
            rows, win = pass_rc(lambda j, known: (B.predict_pose(known, nums[j]), 0.0, {}), rc_marks, ref, "tracking, render and compare")
            res["tracking"].append(line("start pose, no computation", "prior", rows, 0, None))
            for k in rc_marks:
                res["tracking"].append(line("render and compare %d" % k, "render and compare", rows, k, win))
            save()
            rows, win = pass_ff(loc, "track", 1, [], ref, "tracking, matching and PnP")
            res["tracking"].append(line("matching + PnP", "feed forward", rows, 0, win))
            res["stages"] = stages(rows)
            solvers = [r["solver"] for r in rows if r["ok"]]
            res["solver"] = max(set(solvers), key=solvers.count) if solvers else "none"
            if ff_marks:
                rows, win = pass_ff(loc, "track", 1, ff_marks, ref, "tracking, matching and PnP with steps")
                for k in ff_marks:
                    res["tracking"].append(line("matching + PnP + %d steps" % k, "feed forward with steps", rows, k, win))
            save()
            if ctx.get("loc_public") is not None:
                try:
                    rows, win = pass_ff(ctx["loc_public"], "track", 1, [], ref, "tracking, public MASt3R")
                    res["tracking"].append(line("public MASt3R, matching + PnP", "public", rows, 0, win))
                except Exception as e:
                    res["notes"].append("public MASt3R tracking failed: %s: %s" % (type(e).__name__, e))
            res["tracking"] = [r for r in res["tracking"] if r]
            save()

        if "reloc" in tasks:
            try:
                good = {1: 0, 3: 0}
                for j in ver:
                    idx, _ = loc.retrieve(loc.encode(image(j)), 3)
                    cv = [G.covisible(loc.K, ref[j]["pose"], loc.poses[i], ref[j]["points"], hw) >= float(ctx.get("covisible", 0.3)) for i in idx]
                    good[1] += int(cv[0])
                    good[3] += int(any(cv))
                res["retrieval"] = dict(n=len(ver), top1=100.0 * good[1] / max(len(ver), 1), top3=100.0 * good[3] / max(len(ver), 1),
                                        covisible=float(ctx.get("covisible", 0.3)))
            except Exception as e:
                res["notes"].append("retrieval check failed: %s: %s" % (type(e).__name__, e))
            for k in sorted(set(int(x) for x in ctx["topk"])):
                rows, win = pass_ff(loc, "reloc", k, [], ref, "relocalization, best %d map frames" % k)
                res["reloc"].append(line("matching + PnP, best %d" % k, "feed forward", rows, 0, win))
                if k == 1:
                    res["stages_reloc"] = stages(rows)
                    res["difficulty"] = difficulty(rows, px_tol)
            if ff_marks:
                rows, win = pass_ff(loc, "reloc", 1, ff_marks[-1:], ref, "relocalization, matching and PnP with steps")
                res["reloc"].append(line("matching + PnP, best 1 + %d steps" % ff_marks[-1], "feed forward with steps", rows, ff_marks[-1], win))
            save()

            def start_retrieved(j, known):
                B.sync(dev)
                a = time.time()
                idx, _ = loc.retrieve(loc.encode(image(j)), 1)
                B.sync(dev)
                return loc.poses[idx[0]], time.time() - a, dict(idx=idx)

            rl_marks = [k for k in rc_marks if k <= int(ctx.get("reloc_rc_max", 100))] or rc_marks[:1]
            rows, win = pass_rc(start_retrieved, rl_marks, ref, "relocalization, render and compare")
            res["reloc"].append(line("pose of the best map frame", "prior", rows, 0, None))
            for k in rl_marks:
                res["reloc"].append(line("best frame, render and compare %d" % k, "render and compare", rows, k, win))
            if ctx.get("loc_public") is not None:
                try:
                    k = max(int(x) for x in ctx["topk"])
                    rows, win = pass_ff(ctx["loc_public"], "reloc", k, [], ref, "relocalization, public MASt3R")
                    res["reloc"].append(line("public MASt3R, matching + PnP, best %d" % k, "public", rows, 0, win))
                except Exception as e:
                    res["notes"].append("public MASt3R relocalization failed: %s: %s" % (type(e).__name__, e))
            res["reloc"] = [r for r in res["reloc"] if r]
            save()

        sel = [j for j in ver if nums[j] in centre] or ver
        cap = int(ctx.get("conv_max", 12))
        if len(sel) > cap:
            sel = sel[::int(np.ceil(len(sel) / float(cap)))]
        if "conv" in tasks:
            try:
                if not sel:
                    raise RuntimeError("no verified frame")
                conv = []
                kf, kr = int(ctx.get("conv_ff_steps", 10)), int(ctx.get("conv_rc_steps", 100))
                for li, (tp, rd) in enumerate(ctx.get("levels", LEVELS)):
                    rows = {"start": [], "ff": [], "ffk": [], "rc": []}
                    for j in sel:
                        r = ref[j]
                        s = G.perturb(r["pose"], tp / 100.0 * r["depth"], rd, np.random.default_rng(1000 * li + nums[j]))
                        rows["start"].append(dict(ev={0: evaluate(j, s, ref)}))
                        a = loc.localize(image(j), prior=s, topk=1)
                        rows["ff"].append(dict(ev={0: evaluate(j, a["pose"], ref)}, ok=a["ok"]))
                        if kf > 0:
                            rows["ffk"].append(dict(ev={0: evaluate(j, steps_from(j, a["pose"], [kf])[0][kf], ref)}))
                        if kr > 0:
                            rows["rc"].append(dict(ev={0: evaluate(j, steps_from(j, s, [kr])[0][kr], ref)}))
                    conv.append(dict(level=[tp, rd], n=len(sel), **{k: agg(v, 0, 0, px_tol) for k, v in rows.items() if v}))
                    print("[proto] convergence from %.0f %% of depth and %.0f deg: within %.1f px of the reference after matching and PnP %.0f %%" % (
                        tp, rd, px_tol, conv[-1]["ff"].get("sp", float("nan"))), flush=True)
                res["convergence"] = dict(levels=conv, ff_steps=kf, rc_steps=kr, frames=len(sel))
            except Exception as e:
                res["notes"].append("convergence test failed: %s: %s" % (type(e).__name__, e))
            save()

        if "exact" in tasks:
            try:
                tp, rd = ctx.get("exact_level", (5.0, 2.0))
                er = []
                for j in sel:
                    r = ref[j]
                    true = G.perturb(r["pose"], tp / 100.0 * r["depth"], rd, np.random.default_rng(77 + nums[j]))
                    p = tens(true)
                    with torch.no_grad():
                        img = ctx["render"](p, ctx["medium"](nums[j], p), t=nums[j] / fps).clamp(0.0, 1.0)
                    a = loc.localize(img, prior=r["pose"], topk=1)
                    er.append(list(G.pose_error(a["pose"], true, r["depth"])) + [G.pixel_shift(loc.K, true, a["pose"], r["points"], hw), float(a["ok"])])
                er = np.asarray(er, dtype=np.float64)
                if len(er):
                    res["exact"] = dict(n=len(er), level=[tp, rd], et=float(np.median(er[:, 0])), er=float(np.median(er[:, 1])),
                                        px=float(np.nanmedian(er[:, 2])) if np.isfinite(er[:, 2]).any() else None,
                                        sp=float(100.0 * np.mean(np.nan_to_num(er[:, 2], nan=np.inf) <= px_tol)), ok=float(100.0 * er[:, 3].mean()))
            except Exception as e:
                res["notes"].append("exact pose test failed: %s: %s" % (type(e).__name__, e))

    if not ctx.get("only_thinned"):
        passes()
    try:
        if "thinned" in tasks and (ctx.get("strides") or ctx.get("gaps")):
            res["sparse"] = thinned(ctx, loc, hw, px_tol, [dict(num=nums[j], img=(lambda j=j: image(j)), pose=ref[j]["pose"], depth=ref[j]["depth"],
                                                               points=ref[j]["points"], held=True) for j in ver])
    except Exception as e:
        res["notes"].append("thinned map test failed: %s: %s" % (type(e).__name__, e))
    save()

    bud = []
    for b in BUDGETS:
        row = {"budget": b}
        for task in ("tracking", "reloc"):
            fit = [r for r in res[task] if r.get("ms_p90") is not None and r["ms_p90"] <= 1e3 * b]
            row[task] = min(fit, key=lambda r: r["lpips"]) if fit else None
        bud.append(row)
    res["budget"] = bud
    ram = [r for r in tegra.window(0, time.time()) if "ram_used_mb" in r]
    tegra.stop()
    peak = torch.cuda.max_memory_allocated() if dev.type == "cuda" else 0
    res["memory"] = {"gpu_peak_gib": max(whole_peak, peak) / 2 ** 30, "gpu_peak_localization_gib": peak / 2 ** 30,
                     "rss_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024) / 2 ** 30,
                     "store_mb": loc.megabytes()}
    if ram:
        res["memory"].update(system_ram_peak_mb=max(r["ram_used_mb"] for r in ram), system_ram_total_mb=ram[-1]["ram_total_mb"])
    res["device"] = dict(zip(("gpu", "board", "power_mode"), B.device_lines(dev)))
    res["flow"] = bool(G.cv2 is not None and hasattr(G.cv2, "DISOpticalFlow_create"))
    res["reproj_px"] = float(loc.reproj)
    res["px_tol"] = px_tol
    res["subsample"] = int(loc.subsample)
    res = clean(res)
    save()
    return res


def thinned(ctx, loc, hw, px_tol, held):
    fps = float(ctx["fps"])
    map_nums = [int(x) for x in ctx["map_nums"]]
    names = ctx.get("map_names")
    min_inl, min_ratio, cov = int(ctx.get("accept_inliers", 50)), float(ctx.get("accept_ratio", 0.3)), float(ctx.get("covisible", 0.3))
    cache = {}

    def map_query(i):
        def img():
            if i not in cache:
                cache[i] = ctx["load"](names[i])
            return cache[i]
        return dict(num=map_nums[i], img=img, pose=loc.poses[i], depth=float(loc.depths[i]), points=loc.points[i], held=False)

    def accept(r):
        return bool(r["ok"]) and r["inliers"] >= max(min_inl, min_ratio * r["matches"])

    def one(sub, kept_nums, q):
        a = sub.localize(q["img"](), topk=1)
        first = float(G.covisible(loc.K, q["pose"], sub.poses[a["idx"][0]], q["points"], hw))
        near = sorted(range(sub.n), key=lambda i: abs(kept_nums[i] - q["num"]))[:6]
        seen = max(float(G.covisible(loc.K, q["pose"], sub.poses[i], q["points"], hw)) for i in near)
        r, sec, again = a, a["seconds"], False
        if not accept(a):
            b = sub.localize(q["img"](), topk=3)
            sec, again = sec + b["seconds"], True
            if b["inliers"] > a["inliers"]:
                r = b
        row = dict(num=q["num"], held=q["held"], dist=int(min(abs(k - q["num"]) for k in kept_nums)), first=first, seen=seen, again=again,
                   accepted=accept(r), sec=sec, sec1=a["seconds"])
        for key, x in (("", r), ("1", a)):
            et, er = G.pose_error(x["pose"], q["pose"], q["depth"])
            px = G.pixel_shift(loc.K, q["pose"], x["pose"], q["points"], hw)
            row.update({"et" + key: et, "er" + key: er, "px" + key: px, "good" + key: bool(np.isfinite(px) and px <= px_tol)})
        return row

    def summary(rows):
        if not rows:
            return None
        px = np.array([r["px"] for r in rows], dtype=np.float64)
        acc = np.array([r["accepted"] for r in rows])
        good = np.array([r["good"] for r in rows])
        return dict(n=len(rows), sp=float(100.0 * good.mean()), sp1=float(100.0 * np.mean([r["good1"] for r in rows])),
                    s5=float(100.0 * np.mean([(r["et"] <= 5.0) and (r["er"] <= 5.0) for r in rows])),
                    px=float(np.nanmedian(px)) if np.isfinite(px).any() else None, et=float(np.median([r["et"] for r in rows])),
                    er=float(np.median([r["er"] for r in rows])), first=float(100.0 * np.mean([r["first"] >= cov for r in rows])),
                    again=float(100.0 * np.mean([r["again"] for r in rows])), ms_med=float(1e3 * np.median([r["sec"] for r in rows])),
                    ms_mean=float(1e3 * np.mean([r["sec"] for r in rows])), ms1_med=float(1e3 * np.median([r["sec1"] for r in rows])),
                    accepted_good=float(100.0 * np.mean(acc & good)), accepted_off=float(100.0 * np.mean(acc & ~good)),
                    rejected_good=float(100.0 * np.mean(~acc & good)), rejected_off=float(100.0 * np.mean(~acc & ~good)))

    out = dict(strides=[], gaps=[], accept=[min_inl, min_ratio], covisible=cov, px_tol=px_tol)
    for s in [int(x) for x in ctx.get("strides", []) if int(x) > 1]:
        keep = sorted(set(list(range(0, loc.n, s)) + [loc.n - 1]))
        if len(keep) < 2 or len(keep) >= loc.n:
            continue
        kept_nums = [map_nums[i] for i in keep]
        sub = loc.sub(keep)
        qs = list(held) + ([map_query(i) for i in range(loc.n) if i not in set(keep)] if names else [])
        rows = [one(sub, kept_nums, q) for q in qs]
        bins = []
        for lo, hi in ((1, 1), (2, 2), (3, 4), (5, 8), (9, 10 ** 6)):
            g = [r for r in rows if lo <= r["dist"] <= hi]
            if g:
                bins.append(dict(lo=lo, hi=hi, n=len(g), sp=float(100.0 * np.mean([r["good"] for r in g]))))
        out["strides"].append(dict(stride=s, kept=len(keep), all=summary(rows), held=summary([r for r in rows if r["held"]]), bins=bins))
        a = out["strides"][-1]["all"]
        print("[proto] map thinned to every %d (%d of %d map frames), %d queries without a start pose: within %.1f px %.1f %%, median %.0f ms" % (
            s, len(keep), loc.n, a["n"], px_tol, a["sp"], a["ms_med"]), flush=True)
        del sub
    lo_num, hi_num = min(map_nums), max(map_nums)
    for g in [int(x) for x in ctx.get("gaps", []) if int(x) > 0]:
        rows, windows = [], 0
        for a0 in range(lo_num + g, hi_num - 2 * g + 1, 2 * g):
            inside = lambda n: a0 <= n < a0 + g
            keep = [i for i in range(loc.n) if not inside(map_nums[i])]
            if len(keep) < 2 or len(keep) == loc.n:
                continue
            kept_nums = [map_nums[i] for i in keep]
            sub = loc.sub(keep)
            qs = [q for q in held if inside(q["num"])] + ([map_query(i) for i in range(loc.n) if inside(map_nums[i])] if names else [])
            rows += [one(sub, kept_nums, q) for q in qs]
            windows += 1
            del sub
        if not rows:
            continue
        bins = []
        for lo, hi in ((0.8, 1.01), (0.5, 0.8), (0.2, 0.5), (-1.0, 0.2)):
            gq = [r for r in rows if lo <= r["seen"] < hi]
            bins.append(dict(lo=max(lo, 0.0), hi=min(hi, 1.0), n=len(gq), sp=float(100.0 * np.mean([r["good"] for r in gq])) if gq else None))
        out["gaps"].append(dict(frames=g, seconds=g / fps, windows=windows, all=summary(rows), held=summary([r for r in rows if r["held"]]), bins=bins))
        a = out["gaps"][-1]["all"]
        print("[proto] map without a stretch of %.1f s (%d places), %d queries inside without a start pose: within %.1f px %.1f %%" % (
            g / fps, windows, a["n"], px_tol, a["sp"]), flush=True)
    return out


def difficulty(rows, px_tol):
    use = [r for r in rows if r.get("gap") and "px" in r["ev"][0]]
    if len(use) < 6:
        return None
    good = np.array([bool(np.isfinite(r["ev"][0]["px"]) and r["ev"][0]["px"] <= px_tol) for r in use])
    out = dict(n=len(use), median={k: float(np.median([r["gap"][k] for r in use])) for k in ("t", "r", "seen")},
               largest={k: float(np.max([r["gap"][k] for r in use])) for k in ("t", "r")})
    for key in ("t", "r", "seen"):
        v = np.array([r["gap"][key] for r in use])
        e = np.quantile(v, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        bins = []
        for i in range(3):
            m = (v >= e[i]) & ((v <= e[i + 1]) if i == 2 else (v < e[i + 1]))
            bins.append(dict(lo=float(e[i]), hi=float(e[i + 1]), n=int(m.sum()), sp=float(100.0 * good[m].mean()) if m.any() else None))
        out[key] = bins
    return out


def fmt(v, spec):
    return "-" if v is None else spec % v


def table_row(r):
    return "   %-38s %5s %5s %6s %6s %6s | %6.3f %6.3f %6.2f %5s | %8s %8s %6s %7s" % (
        r["label"], fmt(r.get("s5"), "%.1f"), fmt(r.get("sp"), "%.1f"), fmt(r.get("et"), "%.2f"), fmt(r.get("er"), "%.2f"),
        fmt(r.get("px"), "%.1f"), r["lpips"], r["ssim"], r["psnr"], fmt(r.get("flow"), "%.1f"), fmt(r.get("ms_med"), "%.1f"),
        fmt(r.get("ms_p90"), "%.1f"), fmt(r.get("fps"), "%.2f"), fmt(r.get("joule"), "%.1f"))


HEAD = "   %-38s %5s %5s %6s %6s %6s | %6s %6s %6s %5s | %8s %8s %6s %7s" % (
    "method", "ok@5", "ok@px", "t %", "rot", "px", "LPIPS", "SSIM", "PSNR", "flow", "ms med", "p90", "FPS", "J")


def report(res, info):
    L = ["=" * W, " Swimm3R efficient localization protocol", "-" * W]
    d = res["device"]
    L.append(" device      %s%s%s | torch %s%s" % (
        d["gpu"], " | " + d["board"] if d["board"] else "", " | power mode " + d["power_mode"] if d["power_mode"] else "",
        torch.__version__, " | CUDA " + str(torch.version.cuda) if torch.version.cuda else ""))
    L.append(" scene       %s | %d map frames | %d held-out frames (%d interior) | %d x %d" % (
        info["scene"], info["map_frames"], res["frames"], res["interior"], info["hw"][0], info["hw"][1]))
    st = res["memory"]["store_mb"]
    L.append(" map         %.2f M primitives | %.1f MB as float32 tensors | map frame tokens %.1f MB | map frame points %.1f MB" % (
        info["primitives"] / 1e6, info["map_mb"], st["tokens"], st["points"]))
    L.append(" model       %s%s" % (info["model"], " | baseline public MASt3R in the same pipeline on the same map" if info.get("public") else ""))
    if info.get("mapping_seconds"):
        L.append(" mapping     %.0f s for %d frames (%.2f s per frame), run once before localization" % (
            info["mapping_seconds"], info["map_frames"], info["mapping_seconds"] / info["map_frames"]))
    r = res["reference"]
    if r.get("given") and ((r.get("fit") or {}).get("register") or (r.get("fit") or {}).get("same")):
        import loc_ref
        L += ["-" * W] + loc_ref.reference_lines(r.get("fit"), "poses of the queries in the Stage 1 reconstruction of all frames", r["frames"] - r["verified"])
    elif r.get("given"):
        f = r.get("fit") or {}
        L += ["-" * W, " Reference   poses of an offline mapping of all frames (the map frames and the queries together), no external reconstruction,",
              "             moved into this map by a similarity fitted on the %d map frames (scale %.3f): after the fit the map frames differ" % (
                  f.get("n", 0), f.get("scale", float("nan"))),
              "             by a median of %.2f %% of depth and %.2f deg (largest %.2f %%, %.2f deg), the floor of every pose error below. %d of %d queries have one." % (
                  f.get("t_med", float("nan")), f.get("r_med", float("nan")), f.get("t_max", float("nan")), f.get("r_max", float("nan")), r["verified"], r["frames"])]
    else:
        L += ["-" * W, " Reference   every held-out frame registered to the map offline with no time limit, no external reconstruction",
              "             PnP on the matches to the %d map frames nearest in time, then %d render and compare steps" % (r["map_frames"], r["steps"]),
              "             second run of %d steps from the interpolated pose as a check: verified %d of %d (the map points shift by at most %.1f px)" % (
                  r["check_steps"], r["verified"], r["frames"], r["tol"]),
              "             the two runs differ by a median of %.2f %% of depth, %.2f deg, %s px (p90 %.2f %%, %.2f deg)" % (
                  r["check_median"][0], r["check_median"][1], fmt(r["check_median"][2], "%.1f"), r["check_p90"][0], r["check_p90"][1])]
    if r.get("pnp_median"):
        L.append("             PnP against the reference: median %.2f %% of depth, %.2f deg, %s px (the floor of the PnP rows below)" % (
            r["pnp_median"][0], r["pnp_median"][1], fmt(r["pnp_median"][2], "%.1f")))
    if res["tracking"]:
        L += ["-" * W, " Tracking    one frame at a time, start pose constant velocity from the two previous estimates (causal)", HEAD]
        L += [table_row(x) for x in res["tracking"]]
    if res.get("stages"):
        s = res["stages"]
        L.append("   stages of matching + PnP: encoder %.0f ms | map frame choice %.1f ms | decoder and heads %.0f ms | matching %.0f ms | PnP %.0f ms (%s)" % (
            s["encoder"], s["select"], s["decoder"], s["matching"], s["pnp"], res.get("solver", "")))
    if res["reloc"]:
        L += ["-" * W, " Relocalization   no start pose, map frames chosen by the similarity of the encoder tokens to the %d map frames" % info["map_frames"]]
    if res.get("retrieval"):
        q = res["retrieval"]
        L.append("   map frame choice: the first one sees the view in %.1f %% of %d frames, one of the first three in %.1f %% (at least %.0f %% of the view)" % (
            q["top1"], q["n"], q["top3"], 100.0 * q["covisible"]))
    if res["reloc"]:
        L.append(HEAD)
        L += [table_row(x) for x in res["reloc"]]
    df = res.get("difficulty")
    if df:
        third = lambda key, spec: " | ".join((spec + " to " + spec + ": %s (%d)") % (b["lo"], b["hi"], fmt(b["sp"], "%.1f"), b["n"]) for b in df[key])
        L += ["   the queries lie a median of %.1f %% of depth and %.1f deg from the nearest map frame (largest %.1f %%, %.1f deg), which sees %.0f %% of their view" % (
            df["median"]["t"], df["median"]["r"], df["largest"]["t"], df["largest"]["r"], 100.0 * df["median"]["seen"]),
              "   ok@px of matching + PnP, best 1, by thirds of the %d queries: distance in %% of depth  %s" % (df["n"], third("t", "%.1f")),
              "       turn in deg  %s" % third("r", "%.1f"), "       share of the view seen  %s" % third("seen", "%.2f")]
    if res.get("stages_reloc"):
        s = res["stages_reloc"]
        L.append("   stages with one map frame: encoder %.0f ms | map frame choice %.1f ms | decoder and heads %.0f ms | matching %.0f ms | PnP %.0f ms" % (
            s["encoder"], s["select"], s["decoder"], s["matching"], s["pnp"]))
    sp = res.get("sparse")
    if sp and (sp["strides"] or sp["gaps"]):
        L += ["-" * W, " Thinned map   the store of map frames is cut down, the queries are localized without a start pose (first the best map frame,",
              "               the best three when the pose is not accepted). Queries: the verified held-out frames and the map frames taken out,",
              "               whose mapping pose is the reference. The 3D map stays the full mapping, only the frames to match against are fewer.",
              "               The map frames taken out did take part in the mapping, the rows of the held-out frames alone are free of that."]
        if sp["strides"]:
            L.append("   %-22s %7s %6s %6s %6s %6s %6s | %10s %11s %10s | %8s %8s" % (
                "kept map frames", "queries", "ok@px", "ok@5", "px", "t %", "rot", "first try", "frame right", "second try", "ms med", "ms mean"))
            for x in sp["strides"]:
                for name, a in (("every %d (%d frames)" % (x["stride"], x["kept"]), x["all"]), ("  held-out frames only", x.get("held"))):
                    if a:
                        L.append("   %-22s %7d %6.1f %6.1f %6s %6.2f %6.2f | %10.1f %11.1f %10.1f | %8.1f %8.1f" % (
                            name, a["n"], a["sp"], a["s5"], fmt(a["px"], "%.1f"), a["et"], a["er"], a["sp1"], a["first"], a["again"], a["ms_med"], a["ms_mean"]))
                L.append("     ok@px by the distance to the nearest kept map frame: " + " | ".join(
                    "%s frames %.1f (%d)" % (("%d" % b["lo"]) if b["lo"] == b["hi"] else ("%d+" % b["lo"] if b["hi"] > 1000 else "%d to %d" % (b["lo"], b["hi"])),
                                             b["sp"], b["n"]) for b in x["bins"]))
        if sp["gaps"]:
            L.append("   %-22s %7s %6s %6s %6s | ok@px by the share of the view that the remaining map frames see" % ("stretch taken out", "queries", "ok@px", "ok@5", "px"))
            for x in sp["gaps"]:
                a = x["all"]
                L.append("   %-22s %7d %6.1f %6.1f %6s | %s" % (
                    "%.1f s, %d places" % (x["seconds"], x["windows"]), a["n"], a["sp"], a["s5"], fmt(a["px"], "%.1f"),
                    " | ".join("%.0f to %.0f %%: %s (%d)" % (100 * b["lo"], 100 * b["hi"], fmt(b["sp"], "%.1f"), b["n"]) for b in x["bins"])))
                h = x.get("held")
                if h:
                    L.append("   %-22s %7d %6.1f %6.1f %6s |" % ("  held-out frames only", h["n"], h["sp"], h["s5"], fmt(h["px"], "%.1f")))
        rows_all = [x["all"] for x in sp["strides"] + sp["gaps"]]
        tot = float(sum(a["n"] for a in rows_all))
        w = lambda k: sum(a[k] * a["n"] for a in rows_all) / max(tot, 1.0)
        L.append("   self check (a pose is accepted with at least %d inliers and %.0f %% of the matches): accepted and within %.1f px %.1f %%, accepted but off %.1f %%, "
                 "rejected though right %.1f %%, rejected and off %.1f %%" % (sp["accept"][0], 100 * sp["accept"][1], sp["px_tol"], w("accepted_good"),
                                                                         w("accepted_off"), w("rejected_good"), w("rejected_off")))
    c = res.get("convergence")
    if c:
        L += ["-" * W, " Convergence   start pose = reference moved by a known amount, %d verified interior frames, share within %.1f px of the reference" % (
            c["frames"], res["px_tol"]),
              "   %-22s %10s %10s | %14s %20s %24s" % ("start offset", "px (known)", "px (flow)", "matching + PnP", "+ %d steps" % c["ff_steps"],
                                                       "render and compare %d" % c["rc_steps"])]
        for lv in c["levels"]:
            g = lambda k: fmt((lv.get(k) or {}).get("sp"), "%.1f")
            L.append("   %-22s %10s %10s | %14s %20s %24s" % (
                "%.0f %% of depth, %.0f deg" % tuple(lv["level"]), fmt((lv.get("start") or {}).get("px"), "%.1f"),
                fmt((lv.get("start") or {}).get("flow"), "%.1f"), g("ff"), g("ffk"), g("rc")))
    e = res.get("exact")
    if e:
        L += ["-" * W, " Exact pose  %d queries rendered from the map at poses known exactly (%.0f %% of depth and %.0f deg from the reference)" % (
            e["n"], e["level"][0], e["level"][1]),
            "             matching + PnP: median %.2f %% of depth, %.2f deg, %s px | within %.1f px %.1f %% | solved %.1f %%" % (
                e["et"], e["er"], fmt(e.get("px"), "%.2f"), res["px_tol"], e["sp"], e["ok"])]
    if res["tracking"] or res["reloc"]:
        L += ["-" * W, " Budget      lowest LPIPS among the rows whose p90 time per frame fits the budget",
              "   %-8s %-16s %-38s %6s %6s %6s %9s" % ("budget", "task", "method", "LPIPS", "ok@px", "px", "p90 ms")]
    for b in res["budget"] if (res["tracking"] or res["reloc"]) else []:
        for task, name in (("tracking", "tracking"), ("reloc", "relocalization")):
            x = b[task]
            if x:
                L.append("   %-8s %-16s %-38s %6.3f %6s %6s %9.1f" % ("%.1f s" % b["budget"], name, x["label"], x["lpips"], fmt(x.get("sp"), "%.1f"),
                                                                      fmt(x.get("px"), "%.1f"), x["ms_p90"]))
            else:
                L.append("   %-8s %-16s %-38s" % ("%.1f s" % b["budget"], name, "nothing fits"))
    m = res["memory"]
    L += ["-" * W, " Footprint   GPU memory peak %.2f GiB allocated (%.2f GiB during localization) | process memory peak %.2f GiB" % (
        m["gpu_peak_gib"], m["gpu_peak_localization_gib"], m["rss_peak_gib"])]
    if "system_ram_peak_mb" in m:
        L.append("             system memory peak %d MB of %d MB (tegrastats, CPU and GPU share it)" % (m["system_ram_peak_mb"], m["system_ram_total_mb"]))
    L.append("             weights %s (the MASt3R base weights are not counted)" % " + ".join("%s %.1f MB" % (k, v) for k, v in info["weights_mb"].items()))
    L += ["-" * W,
          " ok@5: share of the verified frames within 5 %% of depth and 5 deg of the reference pose. ok@px: share whose map points shift by at most %.1f px." % res["px_tol"],
          " t %: distance of the camera centres over the median depth of the nearest map frame. px: median shift of the map points in the image",
          " between the estimate and the reference (a sideways shift and a turn can cancel in the image, so px is the measure to trust).",
          " flow: median optical flow in px between the frame and the render at the estimated pose, %s" % (
              "no reference involved." if res.get("flow") else "not available here (no OpenCV optical flow)."),
          " The reference is the photometric optimum of the map, so the render and compare rows approach it by construction.",
          " Time per frame runs from the frame on the GPU to the pose: encoder, map frame choice, decoder, heads, matching, PnP, medium lookup",
          " and pose steps. Image loading, scoring and the reference are not timed. The first %d frames of every pass are left out as warm-up." % info["warmup"],
          " J: mean power of the pass (tegrastats) times the mean time per frame. PnP threshold %.1f px, matching grid every %d px." % (
              res["reproj_px"], res["subsample"])]
    for n in res.get("notes", []):
        L.append(" note: %s" % n)
    L.append("=" * W)
    return "\n".join(L)
