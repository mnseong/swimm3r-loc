import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_gscpe import KEY_ROWS, THRESHOLDS


def subset_rows(scores, groups):
    g = scores.get("gscpe") or {}
    names = list(g.get("names") or [])
    out = {}
    for lab, rows in (g.get("per_query") or {}).items():
        if len(rows) != len(names):
            continue
        for gname, pick in groups.items():
            idx = [i for i, n in enumerate(names) if pick(n)]
            if not idx:
                continue
            t = np.array([rows[i]["t"] for i in idx], dtype=np.float64)
            r = np.array([rows[i]["r"] for i in idx], dtype=np.float64)
            sec = [rows[i]["sec"] for i in idx if rows[i].get("sec") is not None]
            out[(lab, gname)] = dict(n=len(idx), acc=[float(100.0 * np.mean((t <= x) & (r <= y))) for x, y in THRESHOLDS],
                                     t=float(np.median(t)), r=float(np.median(r)), fails=int(sum(1 for i in idx if not rows[i].get("ok", True))),
                                     fps=float(1.0 / np.mean(sec)) if sec else None)
    return out


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
            print(" reference check on held out map frames: %.2f %% median, %.2f %% p95 of depth, %.2f / %.2f deg median / p95; thresholds below this are not resolved" % (
                h["t_med"], h["t_p95"], h["r_med"], h["r_p95"]))
        print(" %-40s %-46s %4s | %6s %6s %6s %6s | %6s %6s %5s %6s" % ("method", "queries", "n", "10/5", "5/5", "2/2", "1/1", "t %", "deg", "fails", "FPS"))
        res = subset_rows(s, groups)
        for lab in KEY_ROWS:
            first = True
            for gname in groups:
                v = res.get((lab, gname))
                if not v:
                    continue
                print(" %-40s %-46s %4d | %6.1f %6.1f %6.1f %6.1f | %6.2f %6.2f %5d %6s" % (lab if first else "", gname, v["n"], *v["acc"], v["t"], v["r"], v["fails"],
                      "-" if v["fps"] is None else "%.2f" % v["fps"]))
                first = False


if __name__ == "__main__":
    main()
