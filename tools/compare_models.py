import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_gscpe import KEY_ROWS
from split_scores import score_rows

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
MODE = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("CMP_MODE", "loftr")
WIN = os.environ.get("WIN", "5")
MODELS = [("v18e10", "ours, v18 epoch 10"), ("splatt3r", "Splatt3R (public)")]
VIDEOS = ["video1", "video2", "video3", "video4"]


def scores(m, v):
    tag = "" if v == "video1" else "_" + v
    p = os.path.join(OUT, "%s%s_hold12b3w%s_%s" % (m, tag, WIN, MODE), "loc", "scores.json")
    try:
        return json.load(open(p)).get("gscpe")
    except Exception:
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def f(x, spec):
    return "  -  " if x is None else spec % x


res = {(m, v): scores(m, v) for m, _ in MODELS for v in VIDEOS}
W = 132
if MODE == "moved":
    ref = (" reference: GLUEMAP reconstruction of all frames per video (the pseudo reference of the paper, data/videoN/ref_meta.json), moved into each map by a similarity on its map frames,",
           " errors in percent of the reference depth moved into each map (one denominator for both models)")
else:
    ref = (" reference: LoFTR registration of every query into the map of that model, errors in percent of the median map depth",
           " (the reference and the scored queries differ between the models)")
L = ["=" * W, " Splatt3R against ours with the same localization code, Barbados videos 1 to 4, GS-CPE split (3 of every 12 frames are queries, never mapped),"] + list(ref) + ["-" * W]
L.append(" queries scored   " + " | ".join("%-18s %s" % (lab, " ".join(f((res[(m, v)] or {}).get("frames"), "%3d") for v in VIDEOS)) for m, lab in MODELS))
L.append(" queries held out " + " | ".join("%-18s %s" % (lab, " ".join(f(((res[(m, v)] or {}).get("frames") or 0) + ((res[(m, v)] or {}).get("skipped") or 0) if res[(m, v)] else None, "%3d") for v in VIDEOS)) for m, lab in MODELS))
same = []
for v in VIDEOS:
    ns = [tuple((res[(m, v)] or {}).get("names") or ()) for m, _ in MODELS]
    same.append("%s %s" % (v.replace("video", "v"), "-" if not all(ns) else ("same" if len(set(ns)) == 1 else "DIFFERENT")))
L.append(" query lists of the two models: " + ", ".join(same))
for m, lab in MODELS:
    hs = []
    for v in VIDEOS:
        h = ((res[(m, v)] or {}).get("fit") or {}).get("holdout")
        hs.append("%s %s" % (v.replace("video", "v"), "-" if not h else "%.2f %% / %.2f deg" % (h["t_med"], h["r_med"])))
    if any(not x.endswith("-") for x in hs):
        L.append(" reference check, held out map frames, median (%s): %s" % (lab, ", ".join(hs)))
for m, lab in MODELS:
    hs = []
    for v in VIDEOS:
        h = ((res[(m, v)] or {}).get("fit") or {}).get("holdout")
        hs.append("%s %s" % (v.replace("video", "v"), "%.2f %% / %.2f deg" % (h["t_p95"], h["r_p95"]) if h and "t_p95" in h else "-"))
    if any(not x.endswith("-") for x in hs):
        L.append(" reference check, held out map frames, p95 (%s): %s" % (lab, ", ".join(hs)))
L.append(" per scene: FPS (1 / mean seconds per query), then queries within 5 % and 5 deg | mean of the four scenes: FPS, within 10/5, 5/5, 2/2, 1/1, median %, deg")
L.append(" %-38s %-20s %s | %s | %6s %6s %6s %6s %6s %6s %6s" % ("method", "model", " ".join("%5s" % v.replace("video", "v") for v in VIDEOS),
                                                      " ".join("%5s" % v.replace("video", "v") for v in VIDEOS), "FPS", "10/5", "5/5", "2/2", "1/1", "t %", "deg"))
for lab in KEY_ROWS:
    block = []
    for m, name in MODELS:
        rows = [{r["label"]: r for r in (res[(m, v)] or {}).get("rows", [])}.get(lab) for v in VIDEOS]
        if not any(r is not None and r.get("acc") is not None for r in rows):
            continue
        fps = [r.get("fps") if r else None for r in rows]
        a10 = [r["acc"][0] if r and r.get("acc") else None for r in rows]
        a5 = [r["acc"][1] if r and r.get("acc") else None for r in rows]
        a2 = [r["acc"][2] if r and r.get("acc") else None for r in rows]
        a1 = [r["acc"][3] if r and r.get("acc") else None for r in rows]
        t = [r.get("t") if r else None for r in rows]
        rr = [r.get("r") if r else None for r in rows]
        block.append(" %-38s %-20s %s | %s | %6s %6s %6s %6s %6s %6s %6s" % (lab if not block else "", name, " ".join(f(x, "%5.2f") for x in fps),
                     " ".join(f(x, "%5.1f") for x in a5), f(mean(fps), "%6.2f"), f(mean(a10), "%6.1f"), f(mean(a5), "%6.1f"), f(mean(a2), "%6.1f"),
                     f(mean(a1), "%6.1f"), f(mean(t), "%6.2f"), f(mean(rr), "%6.2f")))
    if block:
        L.extend(block)
loc_k = sorted({(((res[(m, v)] or {}).get("fit") or {}).get("local") or {}).get("k") for m, _ in MODELS for v in VIDEOS} - {None})
if loc_k:
    L[4:4] = [" each query reference moved in by a similarity on the %s map frames nearest to it in the reference (chosen from the reference alone)" % "/".join(map(str, loc_k))]
    L.append("-" * W)
    L.append(" the same estimates against the reference moved in by one similarity on all map frames: mean of the four scenes, within 10/5, 5/5, 2/2, 1/1, median %, deg")
    for m, lab in MODELS:
        hs = []
        for v in VIDEOS:
            h = ((res[(m, v)] or {}).get("fit") or {}).get("holdout_global")
            hs.append("%s %s" % (v.replace("video", "v"), "-" if not h else "%.2f %% / %.2f deg" % (h["t_med"], h["r_med"])))
        L.append(" reference check with one similarity, held out blocks, median (%s): %s" % (lab, ", ".join(hs)))
    for lab in KEY_ROWS:
        block = []
        for m, name in MODELS:
            got = [score_rows(res[(m, v)], lab, against="global") if res[(m, v)] else None for v in VIDEOS]
            if not any(got):
                continue
            col = lambda k, i=None: mean([(g[k] if i is None else g[k][i]) if g else None for g in got])
            block.append(" %-38s %-20s %6s %6s %6s %6s %6s %6s" % (lab if not block else "", name, f(col("acc", 0), "%6.1f"), f(col("acc", 1), "%6.1f"),
                                                             f(col("acc", 2), "%6.1f"), f(col("acc", 3), "%6.1f"), f(col("t"), "%6.2f"), f(col("r"), "%6.2f")))
        L.extend(block)
L.append("=" * W)
text = "\n".join(L)
open(os.path.join(OUT, "splatt3r_vs_ours%s%s.txt" % ("" if MODE == "loftr" else "_" + MODE, "" if WIN == "5" else "_w" + WIN)), "w").write(text + "\n")
print(text)
