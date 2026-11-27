import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_gscpe import KEY_ROWS

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
MODELS = [("v18e10", "ours, v18 epoch 10"), ("splatt3r", "Splatt3R (public)")]
VIDEOS = ["video1", "video2", "video3", "video4"]


def scores(m, v):
    tag = "" if v == "video1" else "_" + v
    p = os.path.join(OUT, "%s%s_hold12b3w5_loftr" % (m, tag), "loc", "scores.json")
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
W = 118
L = ["=" * W, " Splatt3R against ours with the same localization code, Barbados videos 1 to 4, GS-CPE split (3 of every 12 frames are queries, never mapped),",
     " reference: LoFTR registration of every query into the map of that model, errors in percent of the median map depth", "-" * W]
L.append(" queries scored   " + " | ".join("%-18s %s" % (lab, " ".join(f((res[(m, v)] or {}).get("frames"), "%3d") for v in VIDEOS)) for m, lab in MODELS))
L.append(" per scene: FPS (1 / mean seconds per query), then queries within 5 %% and 5 deg | mean of the four scenes: FPS, within 5/5, within 2/2, median %%, deg")
L.append(" %-38s %-20s %s | %s | %6s %6s %6s %6s %6s" % ("method", "model", " ".join("%5s" % v.replace("video", "v") for v in VIDEOS),
                                                      " ".join("%5s" % v.replace("video", "v") for v in VIDEOS), "FPS", "5/5", "2/2", "t %", "deg"))
for lab in KEY_ROWS:
    block = []
    for m, name in MODELS:
        rows = [{r["label"]: r for r in (res[(m, v)] or {}).get("rows", [])}.get(lab) for v in VIDEOS]
        if not any(r is not None and r.get("acc") is not None for r in rows):
            continue
        fps = [r.get("fps") if r else None for r in rows]
        a5 = [r["acc"][1] if r and r.get("acc") else None for r in rows]
        a2 = [r["acc"][2] if r and r.get("acc") else None for r in rows]
        t = [r.get("t") if r else None for r in rows]
        rr = [r.get("r") if r else None for r in rows]
        block.append(" %-38s %-20s %s | %s | %6s %6s %6s %6s %6s" % (lab if not block else "", name, " ".join(f(x, "%5.2f") for x in fps),
                     " ".join(f(x, "%5.1f") for x in a5), f(mean(fps), "%6.2f"), f(mean(a5), "%6.1f"), f(mean(a2), "%6.1f"), f(mean(t), "%6.2f"), f(mean(rr), "%6.2f")))
    if block:
        L.extend(block)
L.append("=" * W)
text = "\n".join(L)
open(os.path.join(OUT, "splatt3r_vs_ours.txt"), "w").write(text + "\n")
print(text)
