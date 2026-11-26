import json
import os
import resource
import sys
import time

import numpy as np
import torch

import bench_loc as B
import bench_proto as P
import eval_gaussian_map as E
import loc_fast as LF
import loc_geom as G
import loc_ref as R
import loc_warp as Wp

THRESHOLDS = ((10.0, 5.0), (5.0, 5.0), (2.0, 2.0), (1.0, 1.0))
W = 138


def summary(rows, warm, depth_m=None):
    if not rows:
        return None
    t = np.array([r["ev"]["t"] for r in rows], dtype=np.float64)
    a = np.array([r["ev"]["r"] for r in rows], dtype=np.float64)
    px = np.array([r["ev"]["px"] for r in rows], dtype=np.float64)
    out = dict(n=len(rows), t=float(np.median(t)), r=float(np.median(a)), px=float(np.nanmedian(px)) if np.isfinite(px).any() else None,
               acc=[float(100.0 * np.mean((t <= x) & (a <= y))) for x, y in THRESHOLDS], fails=int(sum(1 for r in rows if not r.get("ok", True))))
    if depth_m:
        cm = t * float(depth_m)
        out.update(cm=float(np.median(cm)), acc_cm=[float(100.0 * np.mean((cm <= x) & (a <= y))) for x, y in THRESHOLDS])
    for k in ("lpips", "ssim", "psnr"):
        v = [r["ev"][k] for r in rows if k in r["ev"]]
        if v:
            out[k] = float(np.mean(v))
    fl = [r["ev"]["flow"] for r in rows if r["ev"].get("flow") is not None and np.isfinite(r["ev"]["flow"])]
    if fl:
        out["flow"] = float(np.median(fl))
    s = np.array([r["sec"] for r in rows[min(warm, max(len(rows) - 1, 0)):] if r.get("sec") is not None], dtype=np.float64)
    if len(s):
        out.update(s_mean=float(s.mean()), s_med=float(np.median(s)), s_p90=float(np.percentile(s, 90)), fps=float(1.0 / max(s.mean(), 1e-9)))
    return out


def run(ctx):
    dev, loc, fps = ctx["dev"], ctx["loc"], float(ctx["fps"])
    given = ctx["ref_poses"]
    tests = sorted([x for x in ctx["tests"] if x in given], key=E.fnum)
    hw = (loc.h, loc.w)
    warm = int(ctx.get("warmup", 2))
    map_nums = [int(x) for x in ctx["map_nums"]]
    scale = float(np.median(loc.depths))
    depth_m = ctx.get("depth_m")
    n_c = int(ctx.get("coarse_iters", 3))
    eps_t, eps_r = float(ctx.get("eps_t", 0.0002)), float(ctx.get("eps_r", 0.02))
    levels, steps = int(ctx.get("levels", 2)), int(ctx.get("steps", 20))
    wp = dict(levels=levels, steps=steps, lr=float(ctx.get("lr", 3e-4)), views=int(ctx.get("views", 4)), shift=float(ctx.get("shift", 0.02)),
              tau_d=float(ctx.get("tau_d", 0.01)), tau_e=float(ctx.get("tau_e", 0.3)), tau_alpha=float(ctx.get("tau_alpha", 0.5)),
              eps_t=float(ctx.get("rerender_t", 0.005)), eps_r=float(ctx.get("rerender_r", 0.5)))
    whole_peak = torch.cuda.max_memory_allocated() if dev.type == "cuda" else 0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tegra = B.Tegra()
    res = dict(frames=len(tests), skipped=len(ctx["tests"]) - len(tests), rows=[], notes=[], thresholds=[list(x) for x in THRESHOLDS],
               depth_units=scale, depth_m=depth_m, coarse_iters=n_c, eps=[eps_t, eps_r], warp=wp, fit=ctx.get("ref_fit"),
               ref_source=ctx.get("ref_source", ""), names=list(tests), per_query={})

    def save():
        try:
            json.dump(P.clean(res), open(os.path.join(ctx["out_dir"], "gscpe.json"), "w"), indent=1)
        except Exception as e:
            print("[gscpe] gscpe.json not written: %s" % e, flush=True)

    def tens(p):
        return torch.tensor(np.asarray(p), dtype=torch.float32, device=dev)

    def rgbd(p, num, which):
        if which not in ("underwater", "restored"):
            raise ValueError("unknown render %r" % (which,))
        pt = tens(p)
        with torch.no_grad():
            return ctx["render"](pt, ctx["medium"](num, pt) if which == "underwater" else None, t=num / fps, which=which, depth=True)

    imgs = [ctx["load"](x) for x in tests]
    ref = []
    for x in tests:
        p = np.asarray(given[x], dtype=np.float64)
        near = int(loc.near(p, 1)[0])
        dt, dr = G.motion(loc.poses[near], p)
        seen = max(G.covisible(loc.K, p, loc.poses[k], loc.points[near], hw) for k in loc.near(p, 6))
        ref.append(dict(pose=p, near=near, points=loc.points[near], gap_t=100.0 * dt / scale, gap_r=dr, seen=float(seen)))

    def flow_of(j, img, alpha):
        return G.flow_residual(imgs[j].permute(1, 2, 0).cpu().numpy(), img.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy(), (alpha > 0.5).cpu().numpy())

    def evaluate(j, pose, num):
        dt, dr = G.motion(pose, ref[j]["pose"])
        out = dict(t=100.0 * dt / scale, r=dr, px=G.pixel_shift(loc.K, ref[j]["pose"], pose, ref[j]["points"], hw))
        if ctx.get("score") is not None:
            pt = tens(pose)
            with torch.no_grad():
                img, alpha = ctx["render"](pt, ctx["medium"](num, pt), t=num / fps, alpha=True)
                sc = ctx["score"](img, imgs[j])
            out.update(psnr=sc[0], ssim=sc[1], lpips=sc[2], flow=flow_of(j, img, alpha))
        return out

    def check_reference():
        fl, sc = [], []
        for j in range(len(tests)):
            num = E.fnum(tests[j])
            pt = tens(ref[j]["pose"])
            with torch.no_grad():
                img, alpha = ctx["render"](pt, ctx["medium"](num, pt), t=num / fps, alpha=True)
                if ctx.get("score") is not None:
                    sc.append(ctx["score"](img, imgs[j]))
            fl.append(flow_of(j, img, alpha))
        fl = [x for x in fl if x is not None and np.isfinite(x)]
        return dict(n=len(tests), flow=float(np.median(fl)) if fl else None, lpips=float(np.mean([x[2] for x in sc])) if sc else None,
                    psnr=float(np.mean([x[0] for x in sc])) if sc else None)

    def coarse(fq, p0, num, which, lz=loc):
        T = np.asarray(p0, dtype=np.float64)
        sec = dict(render=0.0, encoder=0.0, decoder=0.0, matching=0.0, pnp=0.0)
        its, ok, matches, inliers, conv, solver = 0, False, 0, 0, False, "none"
        for _ in range(n_c):
            B.sync(dev)
            a = time.time()
            ir, al, dp = rgbd(T, num, which)
            B.sync(dev)
            sec["render"] += time.time() - a
            p2d, p3d, _, st = lz.match_render(fq, ir, dp, al, T)
            for k in st:
                sec[k] += st[k]
            a = time.time()
            sol = G.solve_pnp(p3d, p2d, lz.K, lz.reproj, T) if len(p2d) >= 6 else None
            sec["pnp"] += time.time() - a
            its += 1
            if sol is None or not np.isfinite(sol["c2w"]).all():
                break
            ok, matches, inliers, solver = True, int(len(p2d)), int(sol["inliers"]), sol["solver"]
            dt, dr = G.motion(T, sol["c2w"])
            T = np.asarray(sol["c2w"], dtype=np.float64)
            if dt <= eps_t * scale and dr <= eps_r:
                conv = True
                break
        return dict(pose=T, ok=ok, iters=its, matches=matches, inliers=inliers, converged=conv, solver=solver, stages=sec,
                    seconds=float(sum(sec.values())))

    def warp(j, pose, num, which, **over):
        return Wp.refine(imgs[j], pose, loc.K, lambda p: rgbd(p, num, which), lambda p: rgbd(p, num, "restored")[2],
                         sync=lambda: B.sync(dev), **dict(wp, **over))

    def pipeline(which, label, lz=loc):
        out = []
        w0 = time.time()
        for j in range(len(tests)):
            B.sync(dev)
            a = time.time()
            fq = lz.encode(imgs[j])
            idx, _ = lz.retrieve(fq, 1)
            B.sync(dev)
            t_ret = time.time() - a
            num = map_nums[idx[0]]
            c = coarse(fq, loc.poses[idx[0]], num, which, lz)
            r = warp(j, c["pose"], num, which)
            out.append(dict(idx=int(idx[0]), num=num, retrieval=t_ret, start=np.asarray(loc.poses[idx[0]], dtype=np.float64), coarse=c, refine=r))
            print("[gscpe] %s, frame %06d: map frame %06d, coarse %d inliers of %d matches in %d passes (%.2f s), refinement %d steps, %d new anchors (%.2f s)" % (
                label, E.fnum(tests[j]), num, c["inliers"], c["matches"], c["iters"], t_ret + c["seconds"], r["steps"], r["rerenders"], r["seconds"]), flush=True)
        return out, (w0, time.time())

    def add(label, family, rows, window=None):
        a = summary(rows, warm, depth_m)
        if a is None:
            return
        a.update(label=label, family=family)
        res["per_query"][label] = [dict(r["ev"], ok=bool(r.get("ok", True)), sec=r.get("sec")) for r in rows]
        p = B.power_summary(tegra.window(*window)) if window else None
        if p and "s_mean" in a:
            a.update(power_w=p["total_w"], joule=p["total_w"] * a["s_mean"])
        res["rows"].append(a)
        save()

    def stage_means(runs):
        use = runs[min(warm, max(len(runs) - 1, 0)):]
        m = lambda f: float(np.mean([f(x) for x in use]))
        out = dict(retrieval=m(lambda x: x["retrieval"]), coarse=m(lambda x: x["coarse"]["seconds"]), refine=m(lambda x: x["refine"]["seconds"]),
                   iters=m(lambda x: x["coarse"]["iters"]), converged=m(lambda x: 100.0 * float(x["coarse"]["converged"])),
                   steps=m(lambda x: x["refine"]["steps"]), rerenders=m(lambda x: x["refine"]["rerenders"]),
                   mask_share=m(lambda x: 100.0 * float(x["refine"]["mask_share"] or 0.0)),
                   inliers=m(lambda x: x["coarse"]["inliers"]), matches=m(lambda x: x["coarse"]["matches"]))
        for k in ("render", "encoder", "decoder", "matching", "pnp"):
            out["coarse_" + k] = m(lambda x, k=k: x["coarse"]["stages"][k] / max(x["coarse"]["iters"], 1))
        for k in ("render", "mask", "steps"):
            out["refine_" + k] = m(lambda x, k=k: x["refine"]["stages"][k])
        return out

    def rows_of(runs, key):
        if key == "start":
            return [dict(ok=True, sec=x["retrieval"], ev=evaluate(j, x["start"], x["num"])) for j, x in enumerate(runs)]
        if key == "coarse":
            return [dict(ok=x["coarse"]["ok"], sec=x["retrieval"] + x["coarse"]["seconds"], ev=evaluate(j, x["coarse"]["pose"], x["num"])) for j, x in enumerate(runs)]
        return [dict(ok=x["coarse"]["ok"] and x["refine"]["valid"], sec=x["retrieval"] + x["coarse"]["seconds"] + x["refine"]["seconds"],
                     ev=evaluate(j, x["refine"]["pose"], x["num"])) for j, x in enumerate(runs)]

    if tests:
        j0 = 0
        fq = loc.encode(imgs[j0])
        i0, _ = loc.retrieve(fq, 1)
        c0 = coarse(fq, loc.poses[i0[0]], map_nums[i0[0]], "underwater")
        warp(j0, c0["pose"], map_nums[i0[0]], "underwater")
        print("[gscpe] first query: %d inliers of %d matches after %d passes" % (c0["inliers"], c0["matches"], c0["iters"]), flush=True)

    try:
        res["reference_check"] = check_reference() if tests else None
    except Exception as e:
        res["notes"].append("check of the reference failed: %s: %s" % (type(e).__name__, e))
    res["difficulty"] = dict(t=float(np.median([r["gap_t"] for r in ref])) if ref else None, r=float(np.median([r["gap_r"] for r in ref])) if ref else None,
                             t_max=float(np.max([r["gap_t"] for r in ref])) if ref else None, r_max=float(np.max([r["gap_r"] for r in ref])) if ref else None,
                             seen=float(np.median([r["seen"] for r in ref])) if ref else None)
    runs, win = pipeline("underwater", "GS-CPE pipeline")
    cov = float(ctx.get("covisible", 0.3))
    res["retrieval"] = dict(n=len(runs), covisible=cov,
                            top1=float(100.0 * np.mean([G.covisible(loc.K, ref[j]["pose"], loc.poses[x["idx"]], ref[j]["points"], hw) >= cov
                                                        for j, x in enumerate(runs)])) if runs else None)
    res["stages"] = stage_means(runs) if runs else None
    solvers = [x["coarse"]["solver"] for x in runs if x["coarse"]["ok"]]
    res["solver"] = max(set(solvers), key=solvers.count) if solvers else "none"
    if solvers and not res["solver"].startswith("cv2"):
        res["notes"].append("OpenCV is missing here, PnP ran as a robust least squares fit from the start pose in place of RANSAC")
    add("pose of the retrieved map frame", "start", rows_of(runs, "start"))
    add("coarse: render, match, PnP", "coarse", rows_of(runs, "coarse"), win)
    add("refine: + masked warping", "refine", rows_of(runs, "refine"), win)

    try:
        rows, w0 = [], time.time()
        for j in range(len(tests)):
            r = loc.localize(imgs[j], topk=1)
            rows.append(dict(ok=bool(r["ok"]), sec=r["seconds"], pose=np.asarray(r["pose"], dtype=np.float64), num=map_nums[r["idx"][0]]))
        win_ff = (w0, time.time())
        for j, r in enumerate(rows):
            r["ev"] = evaluate(j, r["pose"], r["num"])
        add("no render: match the map frame, PnP", "ours", rows, win_ff)
    except Exception as e:
        res["notes"].append("matching against the retrieved map frame failed: %s: %s" % (type(e).__name__, e))

    if ctx.get("ablation", True) and runs:
        for label, over in (("refine without the pyramid", dict(levels=0, steps=steps * (levels + 1))), ("refine without the mask", dict(use_mask=False)),
                            ("refine with a brightness fit", dict(gain=True))):
            try:
                rows = []
                for j, x in enumerate(runs):
                    r = warp(j, x["coarse"]["pose"], x["num"], "underwater", **over)
                    rows.append(dict(ok=x["coarse"]["ok"] and r["valid"], sec=x["retrieval"] + x["coarse"]["seconds"] + r["seconds"],
                                     ev=evaluate(j, r["pose"], x["num"])))
                add(label, "ablation", rows)
            except Exception as e:
                res["notes"].append("%s failed: %s: %s" % (label, type(e).__name__, e))
        try:
            dry, win_dry = pipeline("restored", "no water model")
            res["stages_no_water"] = stage_means(dry)
            add("no water model: coarse", "no water", rows_of(dry, "coarse"), win_dry)
            add("no water model: refine", "no water", rows_of(dry, "refine"), win_dry)
        except Exception as e:
            res["notes"].append("the run without the water model failed: %s: %s" % (type(e).__name__, e))

    if ctx.get("loc_public") is not None:
        try:
            lp = ctx["loc_public"]
            rows, w0 = [], time.time()
            for j in range(len(tests)):
                r = lp.localize(imgs[j], topk=1)
                rows.append(dict(ok=bool(r["ok"]), sec=r["seconds"], pose=np.asarray(r["pose"], dtype=np.float64), num=map_nums[r["idx"][0]]))
            win_pub = (w0, time.time())
            for j, r in enumerate(rows):
                r["ev"] = evaluate(j, r["pose"], r["num"])
            add("public MASt3R, no render: match, PnP", "public", rows, win_pub)
            pub, win_pub = pipeline("underwater", "public MASt3R as retrieval and matcher", lp)
            res["stages_public"] = stage_means(pub)
            add("public MASt3R matcher: coarse", "public", rows_of(pub, "coarse"), win_pub)
            add("public MASt3R matcher: refine", "public", rows_of(pub, "refine"), win_pub)
        except Exception as e:
            res["notes"].append("public MASt3R failed: %s: %s" % (type(e).__name__, e))

    if ctx.get("fast", True) and tests:
        fk = dict(topk=int(ctx.get("fast_topk", 2)), photo=str(ctx.get("fast_photo", "lcn")), levels=tuple(ctx.get("fast_levels", (3, 2, 1))),
                  iters=tuple(ctx.get("fast_iters", (5, 4, 3))))
        chk = dict(ctx.get("fast_check") or {})
        at = lambda which: (lambda p, i: rgbd(p, map_nums[i], which))
        half = dev.type == "cuda"
        if half:
            try:
                loc.match(loc.encode(imgs[0], half=True), [0], lean=True, half=True)
            except Exception as e:
                half = False
                res["notes"].append("half precision is not usable here: %s: %s" % (type(e).__name__, e))
        info = dict(options=dict(fk, check=chk), half=bool(half))

        def timed(label, family, fn, keep=None):
            try:
                fn(imgs[0])
                rows, w0 = [], time.time()
                for j in range(len(tests)):
                    r = fn(imgs[j])
                    r.pop("fq", None)
                    rows.append(dict(ok=bool(r["ok"]), sec=r["seconds"], pose=np.asarray(r["pose"], dtype=np.float64), num=map_nums[r["idx"][0]], raw=r))
                win_f = (w0, time.time())
                for j, r in enumerate(rows):
                    r["ev"] = evaluate(j, r["pose"], r["num"])
                add(label, family, rows, win_f)
                if keep:
                    use = [r["raw"] for r in rows[min(warm, max(len(rows) - 1, 0)):]]
                    mean = lambda f: float(np.mean([f(x) for x in use]))
                    st = {k: mean(lambda x, k=k: x["stages"].get(k, 0.0)) for k in ("encoder", "tokens", "pnp", "render", "align", "select", "decoder", "matching")}
                    st.update({"fallback_" + k: mean(lambda x, k=k: x["stages"].get("fallback_" + k, 0.0)) for k in ("decoder", "matching", "pnp")})
                    extra = {}
                    if "corr" in use[0]:
                        extra = dict(matches=mean(lambda x: x["matches"]), inliers=mean(lambda x: x["inliers"]), steps=mean(lambda x: x["steps"]),
                                     corr=float(np.median([x["corr"] for x in use if x["corr"] is not None])) if any(x["corr"] is not None for x in use) else None,
                                     valid=mean(lambda x: x["valid"]), fallback=mean(lambda x: 100.0 * float(x.get("fallback", False))))
                    info[keep] = dict(stages=st, **extra)
            except Exception as e:
                res["notes"].append("%s failed: %s: %s" % (label, type(e).__name__, e))

        timed("no render, descriptor branch only", "lean", lambda im: loc.localize(im, topk=1, lean=True), "lean")
        if half:
            timed("no render, branch only, half prec.", "lean", lambda im: loc.localize(im, topk=1, lean=True, half=True), "lean_half")
        timed("fast: tokens, PnP, direct alignment", "fast", lambda im: LF.localize_fast(loc, im, at("underwater"), **fk), "fast")
        timed("fast + check (decoder when it fails)", "fast", lambda im: LF.localize_tiered(loc, im, at("underwater"), check=chk, **fk), "tiered")
        if half:
            timed("fast + check, half precision", "fast", lambda im: LF.localize_tiered(loc, im, at("underwater"), check=chk, half=True, **fk), "tiered_half")
        if ctx.get("ablation", True):
            timed("fast, render without the water model", "fast", lambda im: LF.localize_fast(loc, im, at("restored"), **fk))
        if ctx.get("loc_public") is not None:
            timed("public MASt3R: tokens, PnP, alignment", "public", lambda im: LF.localize_fast(ctx["loc_public"], im, at("underwater"), **fk))
        info.update(lean_ok=loc.lean_ok, lean_diff=loc.lean_diff)
        res["fast"] = info
        save()

    ram = [r for r in tegra.window(0, time.time()) if "ram_used_mb" in r]
    tegra.stop()
    peak = torch.cuda.max_memory_allocated() if dev.type == "cuda" else 0
    res["memory"] = {"gpu_peak_gib": max(whole_peak, peak) / 2 ** 30, "gpu_peak_localization_gib": peak / 2 ** 30,
                     "rss_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024) / 2 ** 30,
                     "store_mb": loc.megabytes()}
    if ram:
        res["memory"].update(system_ram_peak_mb=max(r["ram_used_mb"] for r in ram), system_ram_total_mb=ram[-1]["ram_total_mb"])
    res["device"] = dict(zip(("gpu", "board", "power_mode"), B.device_lines(dev)))
    res["reproj_px"], res["subsample"], res["warmup"] = float(loc.reproj), int(loc.subsample), warm
    res = P.clean(res)
    save()
    return res


def fmt(v, spec):
    return "-" if v is None else spec % v


def table(rows, key_t, key_acc, unit):
    head = "   %-38s %6s %6s %6s %6s | %8s %7s %6s %5s | %6s %6s %6s | %7s %7s %6s %7s" % (
        "method", "10/5", "5/5", "2/2", "1/1", "t " + unit, "rot deg", "px", "flow", "LPIPS", "SSIM", "PSNR", "s mean", "s med", "FPS", "J")
    out = [head]
    for r in rows:
        if r.get(key_acc) is None:
            continue
        out.append("   %-38s %6.1f %6.1f %6.1f %6.1f | %8.3f %7.3f %6s %5s | %6s %6s %6s | %7s %7s %6s %7s" % (
            r["label"], r[key_acc][0], r[key_acc][1], r[key_acc][2], r[key_acc][3], r[key_t], r["r"], fmt(r.get("px"), "%.2f"), fmt(r.get("flow"), "%.1f"),
            fmt(r.get("lpips"), "%.3f"), fmt(r.get("ssim"), "%.3f"), fmt(r.get("psnr"), "%.2f"), fmt(r.get("s_mean"), "%.3f"),
            fmt(r.get("s_med"), "%.3f"), fmt(r.get("fps"), "%.2f"), fmt(r.get("joule"), "%.1f")))
    return out


def report(res, info):
    L = ["=" * W, " Swimm3R localization, protocol of GS-CPE (arXiv 2608.10938): split, pipeline, metrics and pseudo ground truth", "-" * W]
    d = res["device"]
    L.append(" device      %s%s%s | torch %s%s" % (
        d["gpu"], " | " + d["board"] if d["board"] else "", " | power mode " + d["power_mode"] if d["power_mode"] else "",
        torch.__version__, " | CUDA " + str(torch.version.cuda) if torch.version.cuda else ""))
    L.append(" scene       %s | %d x %d" % (info["scene"], info["hw"][0], info["hw"][1]))
    L.append(" split       %d map frames (training set: the map and the retrieval database) | %d queries (test set, never in the mapping)%s" % (
        info["map_frames"], res["frames"], " | %s" % info["split"] if info.get("split") else ""))
    st = res["memory"]["store_mb"]
    L.append(" map         %.2f M primitives, one feed forward pass, no training on the scene | %.1f MB as float32 tensors | tokens %.1f MB | points %.1f MB" % (
        info["primitives"] / 1e6, info["map_mb"], st["tokens"], st["points"]))
    L.append(" model       %s" % info["model"])
    if info.get("mapping_seconds"):
        mf = int(info.get("mapping_frames") or info["map_frames"])
        L.append(" mapping     %.0f s for %d frames (%.2f s per frame), run once before localization%s" % (
            info["mapping_seconds"], mf, info["mapping_seconds"] / mf,
            "; the poses of the queries come from this same reconstruction" if mf != info["map_frames"] else ""))
    L += ["-" * W] + R.reference_lines(res.get("fit"), info.get("reference"), res.get("skipped"))
    rc = res.get("reference_check") or {}
    if rc.get("n"):
        L.append(" Check       render of the map at the reference poses against the queries, no estimate involved: optical flow median %s px, LPIPS %s, PSNR %s (%d queries)" % (
            fmt(rc.get("flow"), "%.2f"), fmt(rc.get("lpips"), "%.3f"), fmt(rc.get("psnr"), "%.2f"), rc["n"]))
    df = res.get("difficulty") or {}
    if df.get("t") is not None:
        L.append(" Queries     a median of %.1f %% of depth and %.1f deg from the nearest map frame (largest %.1f %%, %.1f deg), the map frames near it see %.0f %% of the view" % (
            df["t"], df["r"], df["t_max"], df["r_max"], 100.0 * df["seen"]))
    q = res.get("retrieval") or {}
    if q.get("top1") is not None:
        L.append(" Retrieval   best map frame by the similarity of the encoder tokens, it sees at least %.0f %% of the view for %.1f %% of the %d queries" % (
            100.0 * q["covisible"], q["top1"], q["n"]))
    wp = res["warp"]
    L += ["-" * W, " Pipeline    retrieval -> render of the map at the pose (underwater RGB and depth in one pass) -> matching of the query and the render with the backbone",
          "             -> the matches of the render lifted by the rendered depth -> PnP with RANSAC, at most %d passes -> warping refinement: mask voted by" % res["coarse_iters"],
          "             %d nearby views, %d pyramid levels of %d Adam steps at rate %g on a quaternion and a translation, new anchor render after %.1f %% of depth or %.1f deg" % (
              wp["views"], wp["levels"] + 1, wp["steps"], wp["lr"], 100.0 * wp["eps_t"], wp["eps_r"]),
          " Accuracy    share of the queries within both thresholds, translation / rotation in deg. Median errors next to it.",
          " Units       the map has no metric scale. Translation is given in % of the median depth of the map, so 1 % is 1 cm at 1 m range (thresholds 10, 5, 2, 1 %)."]
    L += table(res["rows"], "t", "acc", "%")
    if res.get("depth_m"):
        L += ["   the same in cm with the median depth of the map taken as %.2f m (given by the user, not measured here): thresholds 10cm/5deg, 5cm/5deg, 2cm/2deg, 1cm/1deg" % res["depth_m"]]
        L += table(res["rows"], "cm", "acc_cm", "cm")
    bad = ["%s %d" % (r["label"], r["fails"]) for r in res["rows"] if r.get("fails")]
    if bad:
        L.append("   queries for which PnP gave no pose or the refinement had no valid pixel (the pose before that step is scored): " + " | ".join(bad))
    s = res.get("stages")
    if s:
        L += ["-" * W, " Runtime     mean per query, the first %d queries left out as warm-up" % res["warmup"],
              "   retrieval %.0f ms (encoder of the query and map frame choice) | coarse %.0f ms in %.2f passes, a pass: render %.0f ms, encoder of the render %.0f ms," % (
                  1e3 * s["retrieval"], 1e3 * s["coarse"], s["iters"], 1e3 * s["coarse_render"], 1e3 * s["coarse_encoder"]),
              "   decoder and heads %.0f ms, matching %.0f ms, PnP %.0f ms | refinement %.0f ms: anchor renders %.0f ms, mask %.0f ms, %.0f steps %.0f ms, %.2f new anchors" % (
                  1e3 * s["coarse_decoder"], 1e3 * s["coarse_matching"], 1e3 * s["coarse_pnp"], 1e3 * s["refine"], 1e3 * s["refine_render"], 1e3 * s["refine_mask"],
                  s["steps"], 1e3 * s["refine_steps"], s["rerenders"]),
              "   coarse stage: %.0f inliers of %.0f matches on average, stopped before the last pass for %.0f %% of the queries | the mask keeps %.0f %% of the pixels" % (
                  s["inliers"], s["matches"], s["converged"], s["mask_share"])]
    fa = res.get("fast")
    if fa:
        o = fa["options"]
        L += ["-" * W, " Fast path   one encoder pass of the query and no decoder: its tokens are matched to the tokens of the best %d map frames (mutual nearest" % o["topk"],
              "             neighbours, refined inside the 3 x 3 neighbours), the matches are lifted by the points of the map at the token centres and PnP gives a",
              "             first pose. The query is then aligned directly to the underwater render of the map at that pose, Gauss-Newton on %s over the" % (
                  "the locally normalised intensity" if o["photo"] == "lcn" else "the colours with one gain and offset per channel"),
              "             pyramid levels 1/%s, no network. The check accepts the result or sends the query through the decoder (descriptor branch only)." % (
                  ", 1/".join(str(2 ** int(x)) for x in o["levels"]))]
        ms = lambda d, k: 1e3 * d["stages"].get(k, 0.0)
        for key, name in (("fast", "fast"), ("tiered", "fast + check"), ("tiered_half", "fast + check, half precision")):
            d = fa.get(key)
            if not d:
                continue
            L.append("   %-28s encoder %.0f ms | tokens %.0f ms | PnP %.0f ms | render %.0f ms | alignment %.0f ms in %.1f steps | %.0f inliers of %.0f token matches | "
                     "correlation %s" % (name, ms(d, "encoder"), ms(d, "tokens"), ms(d, "pnp"), ms(d, "render"), ms(d, "align"), d["steps"], d["inliers"], d["matches"],
                                         fmt(d.get("corr"), "%.2f")))
            if key != "fast":
                L.append("   %-28s %.0f %% of the queries fail the check and go through the decoder (mean over all queries: decoder %.0f ms, matching %.0f ms, PnP %.0f ms)" % (
                    "", d["fallback"], ms(d, "fallback_decoder"), ms(d, "fallback_matching"), ms(d, "fallback_pnp")))
        for key, name in (("lean", "descriptor branch only"), ("lean_half", "branch only, half precision")):
            d = fa.get(key)
            if d:
                L.append("   %-28s encoder %.0f ms | map frame choice %.1f ms | decoder and heads %.0f ms | matching %.0f ms | PnP %.0f ms" % (
                    name, ms(d, "encoder"), ms(d, "select"), ms(d, "decoder"), ms(d, "matching"), ms(d, "pnp")))
        L.append("   the descriptors without the pointmap branch %s; half precision %s" % (
            "differ from the full head by at most %.1e" % fa["lean_diff"] if fa.get("lean_ok") else "are not available here, those rows use the full head",
            "available" if fa.get("half") else "not available on this device"))
    m = res["memory"]
    L += ["-" * W, " Footprint   GPU memory peak %.2f GiB allocated (%.2f GiB during localization) | process memory peak %.2f GiB" % (
        m["gpu_peak_gib"], m["gpu_peak_localization_gib"], m["rss_peak_gib"])]
    if "system_ram_peak_mb" in m:
        L.append("             system memory peak %d MB of %d MB (tegrastats, CPU and GPU share it)" % (m["system_ram_peak_mb"], m["system_ram_total_mb"]))
    L.append("             weights %s (the MASt3R base weights are not counted)" % " + ".join("%s %.1f MB" % (k, v) for k, v in info["weights_mb"].items()))
    L += ["-" * W,
          " px: median shift of the map points in the image between the estimate and the reference. flow: median optical flow in px between the query and the",
          " render at the estimated pose, no reference involved. LPIPS, SSIM, PSNR: underwater render at the estimated pose against the query. The reference",
          " is made with the matcher of the map itself, so the rows that match with the same backbone are not independent of it. Time per query runs from the query on the GPU to the pose. Image loading and scoring are not timed. J: mean power of the pass",
          " (tegrastats) times the mean time per query. PnP threshold %.1f px, solver: %s. Matching grid every %d px." % (
              res["reproj_px"], res.get("solver", "none"), res["subsample"]),
          " Taken from the paper: test 1 to train 3 split with unseen queries, pose of the best retrieved frame as the start, render, match, lift by the rendered",
          " depth, PnP with RANSAC (2000 draws) in at most 3 passes, warping refinement with a visibility mask (consistency share 0.3), 3 pyramid levels of 20 Adam steps at",
          " rate 3e-4, median errors and accuracy at four thresholds, runtime per query. Different here: the map is one feed forward pass in place of a 3DGS",
          " trained for 30000 steps, the retrieval and the matcher are the backbone of the map in place of NetVLAD and VGGT, lengths are relative to the depth",
          " (the paper gives 0.02 m for the depth agreement of the mask, here %.1f %% of the depth; stop of the coarse stage below %.2f %% and %.2f deg)." % (
              100.0 * wp["tau_d"], 100.0 * res["eps"][0], res["eps"][1]),
          " Not stated in the paper, chosen here: %d mask views shifted sideways by %.0f %% of the depth, opacity threshold %.1f." % (
              wp["views"], 100.0 * wp["shift"], wp["tau_alpha"]),
          " Rows 'refine without ...' start from the same coarse poses (without the pyramid: the same number of steps at full resolution). 'refine with a",
          " brightness fit' is not in the paper: before every step one gain and offset per colour channel is fitted between the warped query and the render.",
          " 'no water model' renders the restored colours of the map without the medium in both stages. 'no render' matches the query to the retrieved",
          " map frame itself, with no rendering at all.",
          " 'public MASt3R matcher' runs the same pipeline on the same map with the public weights for the retrieval and the matching.",
          " 'descriptor branch only' skips the pointmap regression of both views, which localization never reads; the descriptors are the same."]
    for n in res.get("notes", []):
        L.append(" note: %s" % n)
    L.append("=" * W)
    return "\n".join(L)
