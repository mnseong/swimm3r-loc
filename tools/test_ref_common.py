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


def drift(ps, g=0.003):
    out = [ps[0].copy()]
    for k in range(1, len(ps)):
        Rk = G.rot(np.array([0.0, g * k, 0.0]))
        m = ps[k].copy()
        m[:3, :3] = Rk @ ps[k][:3, :3]
        m[:3, 3] = out[-1][:3, 3] + Rk @ (ps[k][:3, 3] - ps[k - 1][:3, 3])
        out.append(m)
    return np.array(out)


def check_local():
    rng = np.random.default_rng(1)
    ref = poses(60, rng)
    s, Rm, t = 1.7, rotz(-0.4) @ G.rot(np.array([0.1, 0.2, -0.1])), np.array([0.3, 0.1, -1.0])
    true = np.array([G.move_pose(s, Rm, t, p) for p in drift(ref)])
    names = ["%06d.jpg" % i for i in range(60)]
    q = [i for i in range(60) if i % 6 == 3]
    mi = [i for i in range(60) if i not in q]
    mnames, mposes, qn = [names[i] for i in mi], true[mi], [names[i] for i in q]
    glob, _ = G.transfer_poses(names, ref, mnames, mposes, qn)
    loc, lf = G.transfer_poses_local(names, ref, mnames, mposes, qn, 10)
    eg = max(np.linalg.norm(glob[n][:3, 3] - true[i][:3, 3]) for n, i in zip(qn, q))
    el = max(np.linalg.norm(loc[n][:3, 3] - true[i][:3, 3]) for n, i in zip(qn, q))
    rl = max(G.angle(loc[n][:3, :3], true[i][:3, :3]) for n, i in zip(qn, q))
    lin = lf["k"] == 10 and len(lf["scales"]) == len(q) and not lf["fallback"]
    mp = np.array([G.move_pose(s, Rm, t, p) for p in ref])
    v = G.validate_transfer_local(names, ref, names, mp, 10)
    exact = max(v["et"]) < 1e-9 and max(v["er"]) < 1e-4 and max(abs(x - s) for x in v["scales"]) < 1e-9 and len(v["et"]) == 60
    vb = G.validate_transfer_local(mnames, ref[mi], mnames, mposes, 10)
    hb = G.validate_transfer(mnames, ref[mi], mnames, mposes, blocks=4)
    selfx = G.transfer_poses_local(names, ref, names, mp, ["000005.jpg"], 3)
    _, near, d = G.local_fit(ref, mp, ref[5][:3, 3], 3, [5])
    skip = 5 not in list(near) and d[0] > 0
    few = G.transfer_poses_local(names[:5], ref[:5], names[:5], mp[:5], ["000001.jpg"], 10) is None
    print("transfer_poses_local: on a bent map the queries are off by %.4f with one similarity and %.4f (%.3f deg) with 10 nearest frames %s; "
          "exact similarity recovered by the leave one out check %s; on the bent map the local check gives %.4f median against %.4f for held out blocks; "
          "a query that is also a map frame is left out of its own fit %s; too few frames gives None %s" % (
              eg, el, rl, el < 0.2 * eg and lin, exact, np.median(vb["et"]), np.median(hb["et"]), skip and selfx is not None, few))
    assert el < 0.2 * eg and rl < 1.0 and lin and exact and np.median(vb["et"]) < 0.5 * np.median(hb["et"]) and skip and selfx is not None and few
    return names, q, true, glob, loc


def check_rescore(names, q, true, glob, loc):
    import split_scores as S
    qn = [names[i] for i in q]
    est = [true[i] @ np.block([[np.eye(3), np.array([[0.01], [0], [0]])], [np.zeros((1, 3)), np.ones((1, 1))]]) for i in q]
    den = 0.5
    rows = []
    for n, e in zip(qn, est):
        dt, dr = G.motion(e, loc[n])
        rows.append(dict(t=100.0 * dt / den, r=dr, ok=True, sec=0.1, pose=e.tolist()))
    g = dict(names=qn, depth_units=den, per_query={"fast + check, TensorRT encoder": rows},
             fit=dict(global_poses={n: glob[n].tolist() for n in qn}, local=dict(k=10)))
    a = S.score_rows(g, "fast + check, TensorRT encoder")
    b = S.score_rows(g, "fast + check, TensorRT encoder", against="global")
    want = np.median([100.0 * np.linalg.norm(e[:3, 3] - glob[n][:3, 3]) / den for n, e in zip(qn, est)])
    ok = abs(b["t"] - want) < 1e-9 and a["t"] < b["t"] and a["n"] == b["n"] == len(qn)
    g2 = dict(g, fit={})
    none = S.score_rows(g2, "fast + check, TensorRT encoder", against="global") is None
    print("split_scores: local median %.3f %%, the same estimates against one similarity %.3f %% (expected %.3f) %s, no global poses gives None %s" % (a["t"], b["t"], want, ok, none))
    assert ok and none
    return g


def check_local_lines_and_table(g):
    fit = dict(n=40, scale=2.5, t_med=0.5, t_max=1.0, r_med=0.4, r_max=0.9, depth_ref=3.0, depth_common=7.5, depth_map=6.9,
               local=dict(k=10, queries=7, near_med=0.8, near_max=2.0, far_med=3.0, far_max=6.0, scale_p5=-1.0, scale_p95=1.5, fallback=0),
               holdout=dict(local=10, n=40, t_med=0.2, t_p95=0.9, t_max=1.4, r_med=0.5, r_p95=1.3, r_max=1.6, scale_spread=2.5),
               holdout_global=dict(blocks=4, t_med=3.3, t_p95=6.1, t_max=6.8, r_med=1.6, r_p95=3.9, r_max=4.0, scale_spread=3.0))
    out = "\n".join(R.reference_lines(fit, "poses of a SIFT reconstruction", 0))
    ok = "10 map frames nearest" in out and "each left out in turn" in out and "3.30 % median" in out and "one similarity fitted on the 40" in out
    print("reference lines in the local mode name the neighbours, the leave one out check and the one similarity check %s" % ok)
    assert ok, out
    d = tempfile.mkdtemp(prefix="cmpl_")
    for m in ("v18e10", "splatt3r"):
        for v in ("video1", "video2", "video3", "video4"):
            tag = "" if v == "video1" else "_" + v
            p = os.path.join(d, "%s%s_hold12b3w5_moved" % (m, tag), "loc")
            os.makedirs(p)
            gg = json.loads(json.dumps(g))
            gg.update(frames=len(gg["names"]), skipped=0, rows=[dict(label="fast + check, TensorRT encoder", acc=[100.0, 100.0, 100.0, 100.0], t=0.2, r=0.0, fps=3.0)])
            gg["fit"].update(holdout=fit["holdout"], holdout_global=fit["holdout_global"])
            json.dump(dict(gscpe=gg), open(os.path.join(p, "scores.json"), "w"))
    out = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_models.py"), d, "moved"],
                         capture_output=True, text=True)
    ok = "10 map frames nearest" in out.stdout and "one similarity on all map frames" in out.stdout and "0.90 % / 1.30 deg" in out.stdout and "3.30 % / 1.60 deg" in out.stdout
    print("comparison table in the local mode adds the p95 check and the scores against one similarity %s" % ok)
    assert ok, out.stdout + out.stderr


if __name__ == "__main__":
    check_validate()
    check_angle()
    check_lines()
    check_table()
    check_local_lines_and_table(check_rescore(*check_local()))
    print("test_ref_common ok")
