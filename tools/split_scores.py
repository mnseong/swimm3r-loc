import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loc_geom as G
from bench_gscpe import KEY_ROWS, THRESHOLDS


def errors(g, rows, names, against):
    if against == "local":
        return np.array([r["t"] for r in rows], dtype=np.float64), np.array([r["r"] for r in rows], dtype=np.float64)
    gp = ((g.get("fit") or {}).get("global_poses")) or {}
    den = float(g.get("depth_units") or 0.0)
    if not gp or den <= 0.0:
        return None
    t, a = [], []
    for r, n in zip(rows, names):
        if r.get("pose") is None or n not in gp:
            t.append(np.inf)
            a.append(180.0)
            continue
        dt, dr = G.motion(np.asarray(r["pose"], dtype=np.float64), np.asarray(gp[n], dtype=np.float64))
        t.append(100.0 * dt / den)
        a.append(dr)
    return np.array(t), np.array(a)


def score_rows(g, lab, idx=None, against="local"):
    names = list(g.get("names") or [])
    rows = (g.get("per_query") or {}).get(lab)
    if not rows or len(rows) != len(names):
        return None
    e = errors(g, rows, names, against)
    if e is None:
        return None
    idx = list(range(len(names))) if idx is None else list(idx)
    if not idx:
        return None
    t, r = e[0][idx], e[1][idx]
    sec = [rows[i]["sec"] for i in idx if rows[i].get("sec") is not None]
    return dict(n=len(idx), acc=[float(100.0 * np.mean((t <= x) & (r <= y))) for x, y in THRESHOLDS], t=float(np.median(t)), r=float(np.median(r)),
                fails=int(sum(1 for i in idx if not rows[i].get("ok", True))), fps=float(1.0 / np.mean(sec)) if sec else None)


def subset_rows(scores, groups, against="local"):
    g = scores.get("gscpe") or {}
    names = list(g.get("names") or [])
    out = {}
    for lab in (g.get("per_query") or {}):
        for gname, pick in groups.items():
            v = score_rows(g, lab, [i for i, n in enumerate(names) if pick(n)], against)
            if v:
                out[(lab, gname)] = v
    return out


def block(res, groups):
    print(" %-40s %-46s %4s | %6s %6s %6s %6s | %6s %6s %5s %6s" % ("method", "queries", "n", "10/5", "5/5", "2/2", "1/1", "t %", "deg", "fails", "FPS"))
    for lab in KEY_ROWS:
        first = True
        for gname in groups:
            v = res.get((lab, gname))
            if not v:
                continue
            print(" %-40s %-46s %4d | %6.1f %6.1f %6.1f %6.1f | %6.2f %6.2f %5d %6s" % (lab if first else "", gname, v["n"], *v["acc"], v["t"], v["r"], v["fails"],
                  "-" if v["fps"] is None else "%.2f" % v["fps"]))
            first = False


def main():
    groups = {"held out frames of the map pass (session 1)": lambda n: not os.path.basename(n).startswith("1"),
              "cross session queries (session 2)": lambda n: os.path.basename(n).startswith("1")}
    for path in sys.argv[1:]:
        s = json.load(open(path))
        fit = (s.get("gscpe") or {}).get("fit") or {}
        h = fit.get("holdout") or {}
        print("=" * 120)
        print(" %s" % path)
        if h:
            how = ("similarity on the %d nearest map frames of each query" % h["local"]) if h.get("local") else "one similarity on all map frames"
            print(" reference (%s), check on held out map frames: %.2f %% median, %.2f %% p95 of depth, %.2f / %.2f deg median / p95; thresholds below this are not resolved" % (
                how, h["t_med"], h["t_p95"], h["r_med"], h["r_p95"]))
        block(subset_rows(s, groups), groups)
        if fit.get("global_poses"):
            hg = fit.get("holdout_global") or {}
            print("-" * 120)
            print(" the same estimates against the reference moved in by one similarity on all map frames%s" % (
                "" if not hg else " (check on held out map frames: %.2f %% median, %.2f %% p95 of depth, %.2f / %.2f deg)" % (hg["t_med"], hg["t_p95"], hg["r_med"], hg["r_p95"])))
            block(subset_rows(s, groups, "global"), groups)


if __name__ == "__main__":
    main()
