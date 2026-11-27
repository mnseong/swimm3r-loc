import json
import os
import traceback

ORDER = ("verify", "gap", "medium")
NAMES = dict(verify="Verify", gap="Gap", medium="Medium")


def scene_queries(data_dir):
    p = os.path.join(data_dir, "scene.json") if data_dir else None
    if not p or not os.path.isfile(p):
        return None
    try:
        return json.load(open(p)).get("queries")
    except Exception as e:
        print("[strategies] %s not read (%s, %s), the medium rows have no heading strata" % (p, type(e).__name__, e), flush=True)
        return None


def options(args, map_images=None, ref_cache=None, ref_fit=None, law=None, query_info=None):
    want = {}
    if getattr(args, "verify", False):
        want["verify"] = dict(map_images=map_images, verify_cap=args.verify_cap, verify_seed=args.verify_seed, verify_boot=args.verify_boot,
                              verify_checks=args.verify_checks)
    if getattr(args, "gap_bench", False):
        want["gap"] = dict(map_images=map_images, ref_cache=ref_cache, ref_k=args.ref_register, ref_tol=args.proto_ref_tol, ref_loftr_conf=args.ref_loftr_conf,
                           gap_priors=args.gap_priors, gap_seed=args.gap_seed, gap_queries=args.gap_queries)
    if getattr(args, "medium_adapt", False):
        metric = bool(args.gscpe_ref_metric and args.proto_ref_poses)
        want["medium"] = dict(law=law, units_per_m=(ref_fit or {}).get("scale") if metric else None, query_info=query_info,
                              medium_bound=args.gscpe_medium_bound, medium_prior=args.gscpe_medium_prior, medium_min_reg=args.gscpe_medium_min_reg)
    return want


def one(key, ctx, res):
    if key == "verify":
        import bench_verify as V
        o = V.run_verify(ctx, res)
        return o, "\n".join(V.report_verify(o)), V.in_short_verify(o)
    if key == "gap":
        import bench_gap as GP
        o = GP.run_gap(ctx, res)
        return o, GP.report_gap(o), GP.in_short_gap(o)
    if key == "medium":
        import bench_medium as BM
        o = BM.run_medium(ctx, res)
        return o, BM.report_medium(o), BM.in_short_medium(o)
    raise ValueError("unknown experiment %r" % (key,))


def run(ctx, res, want, out_dir, text=None, path=None, log=None):
    log = log or (lambda m: print(m, flush=True))
    texts, short, done = [], [], {}
    for key in ORDER:
        if key not in want:
            continue
        try:
            o, t, s = one(key, dict(ctx, **want[key]), res)
            open(os.path.join(out_dir, key + ".txt"), "w").write(t + "\n")
            texts.append(t)
            short += list(s)
            done[key] = o
        except Exception as e:
            why = "%s, %s" % (type(e).__name__, " ".join(str(e).split())[:300])
            err = os.path.join(out_dir, key + "_error.txt")
            try:
                open(err, "w").write(traceback.format_exc())
            except Exception:
                err = None
            log("[%s] the experiment failed and the run goes on (%s)%s" % (key, why, ", traceback in %s" % err if err else ""))
            short.append(" %-11s failed (%s)" % (NAMES[key], why))
        if path:
            try:
                open(path, "w").write("\n".join([x for x in [text] + texts if x]) + "\n")
            except Exception as e:
                log("[%s] %s not rewritten (%s, %s)" % (key, path, type(e).__name__, e))
    return texts, short, done
