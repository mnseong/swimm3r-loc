import json
import os
import time

import numpy as np
import torch

import bench_gscpe as C
import bench_loc as B
import bench_proto as P
import eval_gaussian_map as E
import loc_fast as LF
import loc_geom as G
import loc_medium_adapt as LM

W = C.W
ROWS = ("pose of the retrieved map frame", "no render, match the map frame", "coarse, map water", "coarse, water adapted every pass",
        "control, render gain fit every pass", "coarse, restored render, no water", "upper bound, water adapted at ref pose", "fast alignment, map water",
        "fast alignment, adapted water", "anchors, water aligned, map water", "anchors, water aligned, adapted water")
DEFAULTS = dict(bound=3.0, prior=0.02, min_reg=0.4, level=4, min_alpha=0.9)
BANDS = (("below 30 deg", 0.0, 30.0), ("30 to 150 deg", 30.0, 150.0), ("150 deg and more", 150.0, 181.0))


def strata(out, info):
    qi = {str(x["name"]): x for x in (info or []) if isinstance(x, dict) and x.get("name") is not None and x.get("heading_gap_deg") is not None}
    if not qi:
        return None
    names = out["names"]
    idx = [(b, [j for j, n in enumerate(names) if n in qi and lo <= float(qi[n]["heading_gap_deg"]) < hi]) for b, lo, hi in BANDS]
    x, y = C.THRESHOLDS[0]
    rows = []
    for r in out["rows"]:
        pq = out["per_query"].get(r["label"], [])
        d = {}
        for b, ids in idx:
            v = [pq[j] for j in ids if j < len(pq)]
            if v:
                t, a = np.array([q["t"] for q in v], dtype=np.float64), np.array([q["r"] for q in v], dtype=np.float64)
                d[b] = dict(n=len(v), acc=float(100.0 * np.mean((t <= x) & (a <= y))), t=float(np.median(t)), r=float(np.median(a)))
        rows.append(dict(label=r["label"], bands=d))
    return dict(bands=[b for b, _, _ in BANDS], counts={b: len(ids) for b, ids in idx}, unknown=int(sum(n not in qi for n in names)), threshold=[x, y], rows=rows)


def run_medium(ctx, res=None):
    dev, loc, fps = ctx["dev"], ctx["loc"], float(ctx["fps"])
    given = ctx.get("ref_poses") or {}
    tests = sorted([x for x in ctx["tests"] if x in given], key=E.fnum)
    hw = (loc.h, loc.w)
    warm = int(ctx.get("warmup", 2))
    map_nums = [int(x) for x in ctx["map_nums"]]
    scale = float(np.median(loc.depths))
    depth_m, depth_src = ctx.get("depth_m"), "user"
    upm = ctx.get("units_per_m")
    depth_ref = scale / float(upm) if upm else None
    if depth_m is None and depth_ref is not None:
        depth_m, depth_src = depth_ref, "reference"
    n_c = int(ctx.get("coarse_iters", 3))
    eps_t, eps_r = float(ctx.get("eps_t", 0.0002)), float(ctx.get("eps_r", 0.02))
    water, law = ctx.get("water"), ctx.get("law")
    mk = dict(bound=float(ctx.get("medium_bound", 3.0)), prior=float(ctx.get("medium_prior", 0.02)), level=int(ctx.get("medium_level", 4)),
              iters=int(ctx.get("medium_iters", 8)), min_alpha=float(ctx.get("medium_min_alpha", 0.9)), min_reg=float(ctx.get("medium_min_reg", 0.4)))
    gk = {k: mk[k] for k in ("bound", "level", "min_alpha", "min_reg")}
    out = dict(frames=len(tests), skipped=len(ctx["tests"]) - len(tests), rows=[], notes=[], thresholds=[list(x) for x in C.THRESHOLDS], depth_units=scale,
               depth_m=depth_m, depth_m_source=depth_src, depth_m_reference=depth_ref, coarse_iters=n_c, adapt=dict(mk, law=law is not None),
               defaults=all(abs(mk[k] - v) < 1e-12 for k, v in DEFAULTS.items()), names=list(tests), per_query={}, warmup=warm)
    path = os.path.join(ctx["out_dir"], "medium.json")

    def save():
        try:
            json.dump(P.clean(out), open(path, "w"), indent=1)
        except Exception as e:
            print("[medium] medium.json not written: %s" % e, flush=True)

    def finish():
        out["strata"] = strata(out, ctx.get("query_info"))
        save()
        o = P.clean(out)
        if res is not None:
            res["medium"] = o
        return o

    if water is None:
        out["notes"].append("the medium adaptation needs the render parts of the s2 rasteriser with the medium field (ctx water), no rows were run")
        return finish()
    if not tests:
        out["notes"].append("no query has a reference pose, no rows were run")
        return finish()

    def tens(p):
        return torch.tensor(np.asarray(p), dtype=torch.float32, device=dev)

    def med_of(num, pt, f):
        m = ctx["medium"](num, pt)
        return m if f is None else LM.apply_factors(m, f)

    def rgbd(p, num, which, f=None):
        pt = tens(p)
        with torch.no_grad():
            return ctx["render"](pt, med_of(num, pt, f) if which == "underwater" else None, t=num / fps, which=which, depth=True)

    imgs = [ctx["load"](x) for x in tests]
    refp = [np.asarray(given[x], dtype=np.float64) for x in tests]
    pts = [loc.points[int(loc.near(p, 1)[0])] for p in refp]

    def medium_of(parts, num, pose):
        m = parts.get("medium")
        return m if m is not None else ctx["medium"](num, tens(pose))

    def fit_at(j, pose, num, x0=None, cells=None):
        with torch.no_grad():
            parts = water(np.asarray(pose, dtype=np.float64), num)
        return LM.adapt_medium(imgs[j], parts, medium=medium_of(parts, num, pose), law=law, start=x0, sync_fn=lambda: B.sync(dev), cells=cells, **mk), parts

    def gain_at(j, pose, num, cells=None):
        with torch.no_grad():
            parts = water(np.asarray(pose, dtype=np.float64), num)
        return LM.fit_gain(imgs[j], parts, sync_fn=lambda: B.sync(dev), cells=cells, **gk)

    def flow_of(j, img, alpha):
        return G.flow_residual(imgs[j].permute(1, 2, 0).cpu().numpy(), img.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy(), (alpha > 0.5).cpu().numpy())

    def photo(j, pose, num, f=None, which="underwater", gain=None, held=None, whole=True):
        pt = tens(pose)
        with torch.no_grad():
            img, alpha = ctx["render"](pt, med_of(num, pt, f) if which == "underwater" else None, t=num / fps, which=which, depth=True)[:2]
            img = LM.apply_gain(img, gain)
            o = {}
            if whole:
                o["flow"] = flow_of(j, img, alpha)
                if ctx.get("score") is not None:
                    sc = ctx["score"](img, imgs[j])
                    o.update(psnr=sc[0], ssim=sc[1], lpips=sc[2])
            if held is not None:
                m = LM.held_out_mask(alpha, held, cells=0, min_alpha=mk["min_alpha"])
                o.update(held_psnr=LM.masked_psnr(img, imgs[j], m), held_share=float(m.float().mean()))
        return o

    def evaluate(j, pose, num, f=None, gain=None):
        dt, dr = G.motion(pose, refp[j])
        o = dict(t=100.0 * dt / scale, r=dr, px=G.pixel_shift(loc.K, refp[j], pose, pts[j], hw))
        if ctx.get("score") is not None:
            o.update(photo(j, pose, num, f, gain=gain))
        return o

    def coarse(j, fq, p0, num, which, mode=None, fixed=None):
        T = np.asarray(p0, dtype=np.float64)
        sec = dict(adapt=0.0, render=0.0, encoder=0.0, decoder=0.0, matching=0.0, pnp=0.0)
        its, ok, matches, inliers, conv = 0, False, 0, 0, False
        f, x, gn, fits = fixed, None, None, []
        for _ in range(n_c):
            if mode is not None:
                B.sync(dev)
                a = time.time()
                if mode == "medium":
                    fit, _ = fit_at(j, T, num, x)
                    if fit["ok"]:
                        f, x = fit["factors"], fit["factors"]
                else:
                    fit = gain_at(j, T, num)
                    if fit["ok"]:
                        gn = fit
                B.sync(dev)
                sec["adapt"] += time.time() - a
                fits.append({k: v for k, v in fit.items() if k != "apply"})
            B.sync(dev)
            a = time.time()
            ir, al, dp = rgbd(T, num, which, f)[:3]
            ir = LM.apply_gain(ir, gn)
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
        gain = None if gn is None else dict(ok=True, gain=gn["gain"], offset=gn["offset"])
        return dict(pose=T, ok=ok, iters=its, matches=matches, inliers=inliers, converged=conv, stages=sec, seconds=float(sum(sec.values())), factors=f,
                    gain=gain, fits=fits)

    def retrieve(j):
        B.sync(dev)
        a = time.time()
        fq = loc.encode(imgs[j])
        idx, _ = loc.retrieve(fq, 1)
        B.sync(dev)
        return fq, int(idx[0]), time.time() - a

    def add(label, family, rows, extra=None):
        a = C.summary(rows, warm, depth_m)
        if a is None:
            return
        a.update(label=label, family=family)
        if extra:
            a.update(extra)
        out["per_query"][label] = [dict(r["ev"], ok=bool(r.get("ok", True)), sec=r.get("sec"), factors=r.get("factors"), gain=r.get("gain")) for r in rows]
        out["rows"].append(a)
        save()

    def stage_mean(runs, key):
        use = runs[min(warm, max(len(runs) - 1, 0)):]
        return float(np.mean([x["coarse"]["stages"].get(key, 0.0) for x in use])) if use else 0.0

    def coarse_rows(label, family, which, mode=None, upper=False):
        runs = []
        for j in range(len(tests)):
            fq, i, t_ret = retrieve(j)
            num = map_nums[i]
            fixed, t_up = None, 0.0
            if upper:
                B.sync(dev)
                a = time.time()
                fit, _ = fit_at(j, refp[j], num)
                B.sync(dev)
                t_up = time.time() - a
                fixed = fit["factors"] if fit["ok"] else None
            c = coarse(j, fq, loc.poses[i], num, which, mode=mode, fixed=fixed)
            c["stages"]["adapt"] += t_up
            c["seconds"] += t_up
            runs.append(dict(idx=i, num=num, retrieval=t_ret, start=np.asarray(loc.poses[i], dtype=np.float64), coarse=c))
            got = ""
            if c["factors"] is not None:
                got = ", factors veil %s attenuation %s backscatter %s" % tuple("/".join("%.2f" % v for v in row) for row in LM.as_factors(c["factors"]))
            elif c["gain"] is not None:
                got = ", gain %s offset %s" % ("/".join("%.2f" % v for v in c["gain"]["gain"]), "/".join("%.3f" % v for v in c["gain"]["offset"]))
            print("[medium] %s, frame %06d: map frame %06d, %d inliers of %d matches in %d passes (%.2f s, fit %.2f s)%s" % (
                label, E.fnum(tests[j]), num, c["inliers"], c["matches"], c["iters"], t_ret + c["seconds"], c["stages"]["adapt"], got), flush=True)
        rows = [dict(ok=x["coarse"]["ok"], sec=x["retrieval"] + x["coarse"]["seconds"], factors=x["coarse"]["factors"], gain=x["coarse"]["gain"],
                     ev=evaluate(j, x["coarse"]["pose"], x["num"], x["coarse"]["factors"] if which == "underwater" else None, x["coarse"]["gain"]))
                for j, x in enumerate(runs)]
        add(label, family, rows, dict(adapt_s=stage_mean(runs, "adapt"), render_s=stage_mean(runs, "render"),
                                      passes=float(np.mean([x["coarse"]["iters"] for x in runs])) if runs else None))
        return runs

    fq, i, _ = retrieve(0)
    fit_at(0, loc.poses[i], map_nums[i])
    gain_at(0, loc.poses[i], map_nums[i])
    coarse(0, fq, loc.poses[i], map_nums[i], "underwater", mode="medium")
    loc.localize(imgs[0], topk=1)

    try:
        start_rows = []
        for j in range(len(tests)):
            fq, i, t_ret = retrieve(j)
            start_rows.append(dict(ok=True, sec=t_ret, ev=evaluate(j, loc.poses[i], map_nums[i])))
        add(ROWS[0], "start", start_rows)
    except Exception as e:
        out["notes"].append("start rows failed: %s: %s" % (type(e).__name__, e))
    try:
        rows = []
        for j in range(len(tests)):
            r = loc.localize(imgs[j], topk=1)
            rows.append(dict(ok=bool(r["ok"]), sec=r["seconds"], pose=np.asarray(r["pose"], dtype=np.float64), num=map_nums[r["idx"][0]]))
        for j, r in enumerate(rows):
            r["ev"] = evaluate(j, r["pose"], r["num"])
        add(ROWS[1], "ours", rows)
    except Exception as e:
        out["notes"].append("%s failed: %s: %s" % (ROWS[1], type(e).__name__, e))
    runs = {}
    for key, label, family, which, kw in (("source", ROWS[2], "coarse", "underwater", {}), ("adapted", ROWS[3], "adapted", "underwater", dict(mode="medium")),
                                          ("gain", ROWS[4], "control", "underwater", dict(mode="gain")),
                                          ("restored", ROWS[5], "no water", "restored", {}), ("upper", ROWS[6], "upper bound", "underwater", dict(upper=True))):
        try:
            runs[key] = coarse_rows(label, family, which, **kw)
        except Exception as e:
            out["notes"].append("%s failed: %s: %s" % (label, type(e).__name__, e))
    ad = runs.get("adapted")
    if ad:
        out["factors"] = LM.summary([x["coarse"]["fits"][-1] for x in ad if x["coarse"]["fits"]])
        out["factors_first"] = LM.summary([x["coarse"]["fits"][0] for x in ad if x["coarse"]["fits"]])
        regs = [x["coarse"]["fits"][0]["reg"] for x in ad if x["coarse"]["fits"] and x["coarse"]["fits"][0]["reg"] is not None]
        out["registered"] = dict(n=len(ad), first=int(sum(bool(x["coarse"]["fits"] and x["coarse"]["fits"][0]["ok"]) for x in ad)),
                                 any=int(sum(any(y["ok"] for y in x["coarse"]["fits"]) for x in ad)), reg_first=float(np.median(regs)) if regs else None)
    gr = runs.get("gain")
    if gr:
        gs = [x["coarse"]["gain"] for x in gr if x["coarse"]["gain"] is not None]
        out["gain"] = dict(n=len(gr), fitted=len(gs), gain_median=np.median([g["gain"] for g in gs], 0).tolist() if gs else None,
                           offset_median=np.median([g["offset"] for g in gs], 0).tolist() if gs else None)
    if runs.get("upper"):
        ub = []
        for j, x in enumerate(runs["upper"]):
            f = x["coarse"]["factors"]
            ub.append(dict(ok=f is not None, factors=f if f is not None else np.ones((3, 3)).tolist(), sig_before=None, sig_after=None))
        out["factors_upper"] = LM.summary(ub)

    try:
        keys = ("source", "adapted", "restored", "held_source", "held_adapted", "held_gain", "held_restored")
        ph = {p + k: [] for p in ("", "ref_") for k in keys}
        shares = []
        src = ad if ad else runs.get("source", [])
        for j, x in enumerate(src):
            num = x["num"]
            for pre, pose in (("", x["coarse"]["pose"]), ("ref_", refp[j])):
                fa, _ = fit_at(j, pose, num)
                fh, _ = fit_at(j, pose, num, cells=0)
                gh = gain_at(j, pose, num, cells=0)
                lv = int(fh["level"])
                both = photo(j, pose, num, held=lv)
                ph[pre + "source"].append(both)
                ph[pre + "held_source"].append(both)
                ph[pre + "adapted"].append(photo(j, pose, num, fa["factors"] if fa["ok"] else None))
                ph[pre + "held_adapted"].append(photo(j, pose, num, fh["factors"] if fh["ok"] else None, held=lv, whole=False))
                ph[pre + "held_gain"].append(photo(j, pose, num, gain=gh, held=lv, whole=False))
                dry = photo(j, pose, num, which="restored", held=lv)
                ph[pre + "restored"].append(dry)
                ph[pre + "held_restored"].append(dry)
                if not pre:
                    shares.append(fa.get("share"))

        def agg(rows):
            o = {}
            for k in ("psnr", "ssim", "lpips", "held_psnr", "held_share"):
                v = [r[k] for r in rows if r.get(k) is not None]
                if v:
                    o[k] = float(np.mean(v))
            fl = [r["flow"] for r in rows if r.get("flow") is not None and np.isfinite(r["flow"])]
            o["flow"] = float(np.median(fl)) if fl else None
            return o
        out["photometric"] = dict({k: agg(v) for k, v in ph.items()}, n=len(src),
                                  pose="final pose of the row coarse, water adapted every pass" if ad else "final pose of the row coarse, map water",
                                  support=float(np.median([s for s in shares if s is not None])) if any(s is not None for s in shares) else None)
        save()
    except Exception as e:
        out["notes"].append("photometric comparison failed: %s: %s" % (type(e).__name__, e))

    if ctx.get("fast", True):
        fk = dict(topk=int(ctx.get("fast_topk", 2)), photo=str(ctx.get("fast_photo", "lcn")), levels=tuple(ctx.get("fast_levels", (3, 2, 1))),
                  iters=tuple(ctx.get("fast_iters", (5, 4, 3))))
        info = dict(options=fk)

        def source_at(p, i):
            return rgbd(p, map_nums[i], "underwater")[:3]

        def adapted_at(j):
            got = {}

            def at(p, i):
                fit, _ = fit_at(j, p, map_nums[i])
                got["fit"] = fit
                return rgbd(p, map_nums[i], "underwater", fit["factors"] if fit["ok"] else None)[:3]
            return at, got

        def timed(label, family, fn):
            try:
                fn(0)
                rows = []
                for j in range(len(tests)):
                    r, f = fn(j)
                    r.pop("fq", None)
                    rows.append(dict(ok=bool(r["ok"]), sec=r["seconds"], pose=np.asarray(r["pose"], dtype=np.float64), num=map_nums[r["idx"][0]], factors=f))
                for j, r in enumerate(rows):
                    r["ev"] = evaluate(j, r["pose"], r["num"], r["factors"])
                add(label, family, rows)
            except Exception as e:
                out["notes"].append("%s failed: %s: %s" % (label, type(e).__name__, e))

        def fast_adapted(j):
            at, got = adapted_at(j)
            r = LF.localize_fast(loc, imgs[j], at, **fk)
            fit = got.get("fit")
            return r, (fit["factors"] if fit is not None and fit["ok"] else None)

        timed(ROWS[7], "fast", lambda j: (LF.localize_fast(loc, imgs[j], source_at, **fk), None))
        timed(ROWS[8], "fast", fast_adapted)
        anchors = None
        try:
            anchors = LF.TokenAnchors(loc, cell=float(ctx.get("fast_cell", 0.25)))
        except Exception as e:
            out["notes"].append("the anchors of the map could not be built: %s: %s" % (type(e).__name__, e))
        if anchors is not None:
            wk = dict(levels=tuple(ctx.get("water_levels", fk["levels"])), iters=tuple(ctx.get("water_iters", fk["iters"])), support=int(ctx.get("fast_support", 64)))

            def water_src(p, i):
                return water(p, map_nums[i])

            def water_adapted(j):
                got = {}

                def wf(p, i):
                    num = map_nums[i]
                    fit, parts = fit_at(j, p, num)
                    got["fit"] = fit
                    if not fit["ok"]:
                        return parts
                    lv = int(parts.get("level", 0))
                    I = rgbd(p, num, "underwater", fit["factors"])[0]
                    if lv:
                        I = torch.nn.functional.avg_pool2d(I[None], 2 ** lv)[0]
                    if tuple(I.shape[-2:]) != tuple(parts["I"].shape[-2:]):
                        return parts
                    return LM.adapted_parts(parts, fit, medium=medium_of(parts, num, p), law=law, I=I)
                return wf, got

            def anchors_adapted(j):
                wf, got = water_adapted(j)
                r = LF.localize_fast(loc, imgs[j], None, anchors=anchors, water=wf, **wk)
                fit = got.get("fit")
                return r, (fit["factors"] if fit is not None and fit["ok"] else None)

            timed(ROWS[9], "water", lambda j: (LF.localize_fast(loc, imgs[j], None, anchors=anchors, water=water_src, **wk), None))
            timed(ROWS[10], "water", anchors_adapted)
        out["fast"] = info
    return finish()


def fmt(v, spec):
    return "-" if v is None else spec % v


def factor_lines(fs, name):
    if not fs:
        return []
    L = ["   %s, %d queries%s" % (name, fs["n"], ", %d factors at the bound" % fs["at_bound"] if fs.get("at_bound") else "")]
    for k in LM.NAMES:
        d = fs[k]
        L.append("     %-12s median R/G/B %s | 25 to 75 %% %s | range %s" % (
            k, " ".join("%.2f" % v for v in d["median"]), " ".join("%.2f-%.2f" % (a, b) for a, b in zip(d["p25"], d["p75"])),
            " ".join("%.2f-%.2f" % (a, b) for a, b in zip(d["min"], d["max"]))))
    if fs.get("sig_before") is not None:
        L.append("     robust residual scale of query against render on the supported pixels %.4f before, %.4f after (median)" % (fs["sig_before"], fs["sig_after"]))
    return L


def report_medium(out):
    L = ["=" * W, " Swimm3R localization with the medium adapted to the query, the Gaussians and their in air colour stay fixed, only the water changes", "-" * W]
    a = out.get("adapt", {})
    L += [" Method      at the current pose the map is rendered once (in air colour, opacity, depth, veiling light and coefficients of the medium field),",
          "             then nine numbers are fitted, one scale per colour channel of the veiling light, the attenuation and the backscatter, on the pixels",
          "             where the opacity of the map is above %.1f (1/%d resolution), robust least squares on the per pixel model J exp(-b_D z) + B (1 - exp(-b_B z))" % (
              a.get("min_alpha", 0.9), 2 ** int(a.get("level", 4))),
          "             anchored to the s2 render, each scale within a factor %.1f and pulled towards 1 (weight %.3f). The scales then go into the medium given" % (
              a.get("bound", 3.0), a.get("prior", 0.02)),
          "             to the renderer, so every later render uses the adapted water. Coefficient law of the physics head %s." % (
              "used for the attenuation and backscatter (caps included)" if a.get("law") else "not passed, the coefficients are scaled directly"),
          "             The fit runs only when query and render are registered, the correlation of their locally normalised intensity at least %.2f," % a.get("min_reg", 0.4),
          "             otherwise the water of the previous pass (at first the water of the map) is kept. %s" % (
              "Gate, bound and prior are the defaults of tools/loc_medium_adapt.py, not tuned on these queries." if out.get("defaults") else
              "Gate, bound or prior were set on the command line, so they may have been chosen on these queries."),
          " Control     the row 'control' fits one gain and one offset per colour channel of the render to the query on the same pixels, with the same gate",
          "             and the same passes, and applies them to the render, not to the query. Same number of renders and fits as the adapted water row."]
    L += [" Queries     %d scored (%d without a reference left out), the first %d left out of the timing" % (out["frames"], out["skipped"], out["warmup"])]
    L += C.table(out["rows"], "t", "acc", "%")
    if out.get("depth_m"):
        L += ["   the same in cm with the median depth of the map taken as %.2f m (%s)" % (
            out["depth_m"], "given by the user" if out.get("depth_m_source") == "user" else "from the similarity between the metric reference and the map")]
        L += C.table(out["rows"], "cm", "acc_cm", "cm")
    if out.get("depth_m_reference") is not None and out.get("depth_m_source") == "user":
        L.append("   the similarity between the metric reference and the map gives a median depth of %.2f m" % out["depth_m_reference"])
    bad = ["%s %d" % (r["label"], r["fails"]) for r in out["rows"] if r.get("fails")]
    if bad:
        L.append("   PnP gave no pose for some queries, the pose before is scored | " + " | ".join(bad))
    ad = [r for r in out["rows"] if r.get("adapt_s")]
    if ad:
        L.append("   mean fit time per query, inside s mean | " + " | ".join("%s %.0f ms" % (r["label"], 1e3 * r["adapt_s"]) for r in ad))
    L += ["-" * W, " Fitted scales (1 means the water of the map)"]
    rg = out.get("registered")
    if rg:
        L.append("   registered at the pose of the retrieved map frame for %d of %d queries (median correlation %s), adapted in at least one pass for %d" % (
            rg["first"], rg["n"], fmt(rg.get("reg_first"), "%.2f"), rg["any"]))
    L += factor_lines(out.get("factors_first"), "at the pose of the retrieved map frame")
    L += factor_lines(out.get("factors"), "in the last pass")
    L += factor_lines(out.get("factors_upper"), "upper bound, fitted at the reference pose")
    gn = out.get("gain")
    if gn and gn.get("gain_median") is not None:
        L.append("   control, gain fitted for %d of %d queries, median gain R/G/B %s, offset %s" % (
            gn["fitted"], gn["n"], " ".join("%.2f" % v for v in gn["gain_median"]), " ".join("%.3f" % v for v in gn["offset_median"])))
    ph = out.get("photometric")
    if ph:
        L += ["-" * W, " Query against the render, %d queries, at the %s and at the reference poses (no estimate involved there)" % (ph["n"], ph["pose"]),
              "   whole image, the adapted water fitted on the same query, so its gain is in sample"]
        for key, name in (("source", "water of the map"), ("adapted", "adapted water"), ("restored", "restored render, no water"),
                          ("ref_source", "at the ref pose, map water"), ("ref_adapted", "at the ref pose, adapted"), ("ref_restored", "at the ref pose, restored")):
            d = ph.get(key) or {}
            L.append("     %-30s PSNR %s | SSIM %s | LPIPS %s | flow %s px" % (name, fmt(d.get("psnr"), "%.2f"), fmt(d.get("ssim"), "%.3f"), fmt(d.get("lpips"), "%.3f"),
                                                                             fmt(d.get("flow"), "%.2f")))
        hs = (ph.get("held_source") or {}).get("held_share")
        L.append("   held out pixels, water and gain fitted on every other block of the image and PSNR taken on the supported pixels of the other blocks%s" % (
            "" if hs is None else " (%.0f %% of the image)" % (100.0 * hs)))
        for pre, where in (("", "estimated pose"), ("ref_", "reference pose")):
            L.append("     %-30s PSNR map water %s | adapted water %s | gain control %s | restored %s" % (
                "at the " + where, *[fmt((ph.get(pre + k) or {}).get("held_psnr"), "%.2f") for k in ("held_source", "held_adapted", "held_gain", "held_restored")]))
        if ph.get("support") is not None:
            L.append("   the map supports %.0f %% of the pixels at the coarse level (median)" % (100.0 * ph["support"]))
    st = out.get("strata")
    if st:
        L += ["-" * W, " By the heading gap between the query and its best overlapping map frame (scene.json), queries per band %s%s" % (
            ", ".join("%s %d" % (b, st["counts"][b]) for b in st["bands"]), ", %d without a gap left out" % st["unknown"] if st.get("unknown") else ""),
              "   share within %.0f %% and %.0f deg | median error %% of depth, deg" % tuple(st["threshold"])]
        for r in st["rows"]:
            L.append("   %-38s %s" % (r["label"], " | ".join("%s %s" % (b, "%5.1f %% %6.2f %% %6.2f deg" % (r["bands"][b]["acc"], r["bands"][b]["t"], r["bands"][b]["r"])
                                                                                   if b in r["bands"] else "-") for b in st["bands"])))
    L += ["-" * W,
          " LPIPS, SSIM, PSNR and flow of each row are taken on the render at the estimated pose with the water that row used (source rows the water of the",
          " map, adapted rows their fitted scales, the control its gain), so these columns differ between rows also by that water. Time per query runs",
          " from the query on the GPU to the pose, the fits and their renders included. Image loading and scoring are not timed. The upper bound row",
          " fits the scales at the reference pose and is not a method. 'no render' matches the query to the retrieved map frame itself."]
    for n in out.get("notes", []):
        L.append(" note: %s" % n)
    L.append("=" * W)
    return "\n".join(L)


def in_short_medium(out, keys=(ROWS[1], ROWS[2], ROWS[3], ROWS[4], ROWS[9], ROWS[10])):
    rows = {r["label"]: r for r in out.get("rows") or []}
    if not rows:
        return [" Medium      no rows were run" + (", %s" % out["notes"][0] if out.get("notes") else "")]
    L = [" Medium      water adapted to the query, %d queries | within 10 %% and 5 deg, within 2 %% and 2 deg | median error %% of depth, deg | s mean" % out["frames"]]
    for k in keys:
        r = rows.get(k)
        if r is None or r.get("acc") is None:
            continue
        L.append("               %-38s %5.1f %%  %5.1f %% | %5.2f %%  %5.2f deg | %s s" % (k, r["acc"][0], r["acc"][2], r["t"], r["r"], fmt(r.get("s_mean"), "%.2f")))
    st = out.get("strata")
    if st:
        b = st["bands"][0]
        got = {r["label"]: r["bands"].get(b) for r in st["rows"]}
        nick = ((ROWS[1], "no render"), (ROWS[2], "map water"), (ROWS[3], "adapted water"), (ROWS[4], "gain control"))
        L.append("               heading gap %s, %d queries, share within 10 %% and 5 deg" % (b, st["counts"].get(b, 0)))
        L.append("               " + " | ".join("%s %s" % (s, fmt(got[k]["acc"] if got.get(k) else None, "%.1f %%")) for k, s in nick))
    return L
