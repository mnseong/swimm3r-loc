import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loc_geom as G
import loc_ref as R


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def poses(n, rng):
    out = []
    for i in range(n):
        m = np.eye(4)
        m[:3, :3] = rotz(0.02 * i) @ G.rot(rng.normal(size=3) * 0.05)
        m[:3, 3] = [0.1 * i, 0.3 * np.sin(i / 7.0), 0.2 * np.cos(i / 9.0)]
        out.append(m)
    return np.array(out)


def check_validate():
    rng = np.random.default_rng(0)
    ref = poses(40, rng)
    s, Rm, t = 2.5, rotz(0.7) @ G.rot(np.array([0.2, -0.1, 0.05])), np.array([1.0, -2.0, 0.5])
    mp = np.array([G.move_pose(s, Rm, t, p) for p in ref])
    names = ["%06d.jpg" % i for i in range(40)]
    v = G.validate_transfer(names, ref, names, mp, blocks=4)
    exact = max(v["et"]) < 1e-9 and max(v["er"]) < 1e-4 and max(abs(x - s) for x in v["scales"]) < 1e-9
    bent = mp.copy()
    for k in range(30, 40):
        bent[k] = bent[k] @ np.block([[G.rot(np.array([0, 0.05, 0])), np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])
        bent[k][:3, 3] += [0.05 * (k - 29), 0, 0]
    w = G.validate_transfer(names, ref, names, bent, blocks=4)
    worse = max(w["et"][30:]) > 0.1 and max(w["er"][30:]) > 1.0
    few = G.validate_transfer(names[:10], ref[:10], names[:10], mp[:10], blocks=4) is None
    print("validate_transfer: exact similarity recovered on held out blocks %s, a bent last block shows up (%.3f, %.2f deg) %s, too few frames gives None %s" % (
        exact, max(w["et"][30:]), max(w["er"][30:]), worse, few))
    assert exact and worse and few


def check_angle():
    a, b = np.eye(4), np.eye(4)
    b[:3, :3] = G.rot(np.array([0.0, 0.0, np.radians(1.0)])) * 1.002
    got = G.pose_error(a, b, 1.0)[1]
    print("rotation error of a 1 deg rotation stored with a 0.2 %% scale: %.4f deg (the trace alone would clip it to 0)" % got)
    assert abs(got - 1.0) < 1e-6


def check_lines():
    fit = dict(n=40, scale=2.5, t_med=0.5, t_max=1.0, r_med=0.4, r_max=0.9, depth_ref=3.0, depth_common=7.5, depth_map=6.9,
               holdout=dict(blocks=4, t_med=1.2, t_p95=2.0, t_max=2.5, r_med=0.8, r_p95=1.3, r_max=1.5, scale_spread=0.7))
    out = "\n".join(R.reference_lines(fit, "poses of a COLMAP reconstruction", 0))
    ok = "7.5000 map units" in out and "Held out map frames" in out and "not resolved by the reference" in out
    print("reference lines name the common depth and the held out check %s" % ok)
    assert ok


def check_table():
    d = tempfile.mkdtemp(prefix="cmp_")
    for m in ("v18e10", "splatt3r"):
        for v in ("video1", "video2", "video3", "video4"):
            tag = "" if v == "video1" else "_" + v
            p = os.path.join(d, "%s%s_hold12b3w5_moved" % (m, tag), "loc")
            os.makedirs(p)
            names = ["%06d.jpg" % i for i in range(5, 30, 4)] if (m == "v18e10" or v != "video3") else ["%06d.jpg" % i for i in range(5, 26, 4)]
            row = dict(label="fast + check, TensorRT encoder", acc=[100.0, 90.0, 80.0, 50.0], t=0.8, r=0.4, fps=3.5)
            g = dict(frames=len(names), skipped=7 - len(names), names=names, rows=[row],
                     fit=dict(holdout=dict(t_med=1.1, r_med=0.7)))
            json.dump(dict(gscpe=g), open(os.path.join(p, "scores.json"), "w"))
    out = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_models.py"), d, "moved"],
                         capture_output=True, text=True).stdout
    ok = "v3 DIFFERENT" in out and "v1 same" in out and "1.10 % / 0.70 deg" in out and os.path.isfile(os.path.join(d, "splatt3r_vs_ours_moved.txt")) and "one denominator" in out
    print("comparison table in the common reference mode: query lists checked, reference check printed, file written %s" % ok)
    assert ok, out


if __name__ == "__main__":
    check_validate()
    check_angle()
    check_lines()
    check_table()
    print("test_ref_common ok")
