import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import mast3r.utils.path_to_dust3r
import bench_gscpe as C
import bench_strategies as S
import eval_gaussian_map as E
import test_loc as T
import test_medium_adapt as TM

tt = lambda p: torch.tensor(np.asarray(p), dtype=torch.float32)
name = lambda i: "%06d.jpg" % i
REQ = ["--scene_dir", "s", "--lora", "l", "--physics", "p", "--out_dir", "o"]
OUTSIDE = 100005


def check_parser():
    try:
        import stage1_splatt3r as S1
    except ModuleNotFoundError as e:
        print("parser not checked with this Python, tools/stage1_splatt3r.py does not import (%s)" % e)
        return
    ap = S1.parser()
    base = ap.parse_args(REQ + ["--gscpe"])
    assert not base.verify and not base.gap_bench and not base.medium_adapt and S.options(base) == {}
    assert base.gap_priors == ["2:0", "5:0", "10:0", "20:0", "0:2", "0:5", "0:10"] and base.verify_cap == 16 and base.gscpe_medium_bound == 3.0
    a = ap.parse_args(REQ + ["--gscpe", "--verify_bench", "--gap_bench", "--medium_adapt", "--verify_cap", "5", "--verify_checks", "d", "c",
                             "--gap_priors", "2:0", "0:5", "--gap_queries", "4", "--gscpe_medium_min_reg", "0.5", "--ref_register", "6"])
    b = ap.parse_args(REQ + ["--gscpe", "--verify", "--gscpe_medium"])
    assert b.verify and b.medium_adapt and not b.gap_bench
    cache, imgs, law = {}, torch.zeros(2, 3, 4, 4), (lambda m, z: None)
    w = S.options(a, map_images=imgs, ref_cache=cache, ref_fit=dict(scale=3.0), law=law, query_info=[{"name": "x"}])
    assert set(w) == {"verify", "gap", "medium"}
    assert w["verify"]["map_images"] is imgs and w["verify"]["verify_cap"] == 5 and w["verify"]["verify_checks"] == ["d", "c"]
    assert w["gap"]["ref_cache"] is cache and w["gap"]["gap_priors"] == ["2:0", "0:5"] and w["gap"]["gap_queries"] == 4 and w["gap"]["ref_k"] == 6
    assert w["gap"]["ref_tol"] == a.proto_ref_tol and w["gap"]["ref_loftr_conf"] == a.ref_loftr_conf
    assert w["medium"]["law"] is law and w["medium"]["medium_min_reg"] == 0.5 and w["medium"]["units_per_m"] is None and w["medium"]["query_info"]
    m = ap.parse_args(REQ + ["--gscpe", "--medium_adapt", "--gscpe_ref_metric", "--proto_ref_poses", "r.json"])
    assert S.options(m, ref_fit=dict(scale=3.0))["medium"]["units_per_m"] == 3.0
    assert S.options(ap.parse_args(REQ + ["--gscpe", "--medium_adapt", "--gscpe_ref_metric"]), ref_fit=dict(scale=3.0))["medium"]["units_per_m"] is None
    import contextlib
    import io
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            ap.parse_args(REQ + ["--verify_checks", "z"])
        raise AssertionError("an unknown check must be refused")
    except SystemExit:
        pass
    d = tempfile.mkdtemp()
    assert S.scene_queries(d) is None and S.scene_queries(None) is None
    json.dump(dict(queries=[dict(name="100000.jpg", heading_gap_deg=10.0)]), open(os.path.join(d, "scene.json"), "w"))
    assert S.scene_queries(d)[0]["heading_gap_deg"] == 10.0
    sj = os.path.join(ROOT, "data", "sealoc_s1")
    if os.path.isfile(os.path.join(sj, "scene.json")):
        q = S.scene_queries(sj)
        assert len(q) == 24 and all("heading_gap_deg" in x for x in q)
    print("parser: the three experiments are off by default, --verify_bench, --gap_bench and --medium_adapt (aliases --verify and --gscpe_medium) "
          "and their options reach the context of each experiment")


def check_sample():
    import torch.nn.functional as F
    import loc_warp as Wp
    g = torch.Generator().manual_seed(3)
    img = torch.rand(2, 7, 9, generator=g)
    u0 = (torch.rand(5, 6, generator=g) * 14.0 - 3.0).requires_grad_(True)
    v0 = (torch.rand(5, 6, generator=g) * 12.0 - 3.0).requires_grad_(True)
    u1, v1 = u0.detach().clone().requires_grad_(True), v0.detach().clone().requires_grad_(True)
    ref = F.grid_sample(img[None], torch.stack([2.0 * u0 / 8.0 - 1.0, 2.0 * v0 / 6.0 - 1.0], -1)[None], mode="bilinear", padding_mode="zeros",
                        align_corners=True)[0]
    got = Wp.sample(img, u1, v1)
    ref.sum().backward()
    got.sum().backward()
    assert float((got - ref).detach().abs().max()) < 1e-6 and float((u1.grad - u0.grad).abs().max()) < 1e-6 and float((v1.grad - v0.grad).abs().max()) < 1e-6
    far = Wp.sample(img, torch.tensor([[1e13, -1e13, 3.0]]), torch.tensor([[2.0, 2.0, 1e12]]))
    assert float(far.abs().max()) == 0.0
    print("warp sampling: coordinates far outside the image are held two pixels outside, values and gradients unchanged, no crash on huge coordinates")


def check_mapping_lines():
    info = dict(map_frames=90, mapping_frames=90, mapping_seconds=100.0, load_seconds=20.0, map_seconds=30.0, field_seconds=40.0)
    tot, how = C.mapping_parts(info)
    assert tot == 190.0 and how == "Stage 1 100 s, reading the Stage 1 output and loading the networks 20 s, Gaussian map 30 s, medium field fit 40 s", how
    assert C.mapping_parts(dict(info, field_seconds=None))[1].endswith("Gaussian map 30 s, no medium field fitted")
    assert C.mapping_parts(dict(info, field_seconds=None))[0] == 150.0
    old = dict(map_frames=90, mapping_seconds=100.0, map_seconds=5.0)
    assert C.mapping_parts(old) == (105.0, "Stage 1 100 s, Gaussian map and medium field 5 s")
    assert C.mapping_parts(dict(map_frames=90, mapping_seconds=100.0))[1] == "Stage 1 100 s, Gaussian map and medium field not timed"
    assert C.mapping_parts(dict(map_frames=90)) == (None, None)
    row = lambda lab, s: dict(label=lab, acc=[100.0, 95.0, 90.0, 50.0], s_mean=s, fps=1.0 / s, t=0.3, r=0.2, flow=0.5)
    res = dict(frames=21, skipped=6, rows=[row("no render: match the map frame, PnP", 0.6), row("no render, descriptor branch only", 0.5),
                                           row("no render, branch only, half prec.", 0.4), row("something else", 0.1)])
    L = C.in_short(res, info)
    text = "\n".join(L)
    assert L[0] == " In short    mapping, run once: 90 frames in 190 s, 2.1 s per frame" and L[1] == "             " + how
    assert "21 queries scored, 6 of the 27 queries of the split have no accepted reference and are left out" in text
    labs = [x.strip().split("  ")[0] for x in L[4:]]
    assert labs == ["no render: match the map frame, PnP", "no render, descriptor branch only", "no render, branch only, half prec."], labs
    assert C.KEY_ROWS.index("no render, descriptor branch only") == 1 and C.KEY_ROWS.index("no render, branch only, half prec.") == 2
    zero = C.in_short(dict(res, skipped=0), dict(map_frames=90))
    assert zero[0].startswith(" In short    localization") and "21 queries scored, 0 of the 21 queries" in zero[1]
    print("mapping time: Stage 1, loading, Gaussian map and the fit of the medium field each named and summed, the no render rows and the "
          "queries left out without a reference in the short block")


def scene():
    loc, hold, mapf, truth = TM.scene()
    gen = torch.Generator().manual_seed(11)
    poses = {name(i): truth[i] for i in hold}
    poses[name(OUTSIDE)] = truth[hold[1]]
    poses["999999.jpg"] = truth[hold[2]]
    queries = {}

    def load(nm):
        if nm not in queries:
            with torch.no_grad():
                queries[nm] = TM.query(poses[nm], gen, noise=0.002, f=TM.MILD)
        return queries[nm]

    seen = []

    def render(pose, med=None, t=None, alpha=False, which="underwater", depth=False):
        assert t is not None and (which == "restored" or med is not None)
        seen.append(float(t))
        return TM.render(pose, med, t, alpha, which, depth)

    def score(img, gt):
        mae = float((img.clamp(0, 1) - gt)[:, :, 24:].abs().mean())
        return [float(-10.0 * torch.log10(((img.clamp(0, 1) - gt)[:, :, 24:] ** 2).mean() + 1e-12)), 1.0 - mae, mae]

    with torch.no_grad():
        mi = torch.stack([TM.underwater(truth[i], TM.MED)[0] for i in mapf])
    tests = [name(i) for i in hold] + [name(OUTSIDE), "999999.jpg"]
    ref = {k: v for k, v in poses.items() if k != "999999.jpg"}
    ctx = dict(dev=torch.device("cpu"), tests=tests, fps=5.0, warmup=1, map_nums=mapf, load=load, medium=lambda num, pose: dict(TM.MED), render=render,
               score=score, loc=loc, loc_public=None, out_dir=tempfile.mkdtemp(), ref_poses=ref, ref_fit=dict(n=len(mapf), scale=1.0, t_med=0.0, t_max=0.0,
                                                                                                              r_med=0.0, r_max=0.0),
               ref_source="truth", fast_photo="affine", fast_levels=(2, 1, 0), fast_iters=(6, 4, 3), fast_dim=16, ablation=False,
               fast_check=dict(min_corr=0.25), water=lambda p, n: dict(TM.parts(p), medium=TM.MED))
    return ctx, mi, seen, loc, hold, mapf


def check_run():
    ctx, mi, seen, loc, hold, mapf = scene()
    a0 = time.time()
    res = C.run(ctx)
    t_bench = time.time() - a0
    names = res["names"]
    assert res["skipped"] == 1 and res["frames"] == len(hold) + 1 and name(OUTSIDE) in names
    top = max(mapf) / ctx["fps"] + 1e-6
    assert seen and max(seen) <= top, ("a query outside the map frame numbers is rendered at the time of a map frame", max(seen))
    assert res["reference_check"]["n"] == len(names) and res["reference_check"]["psnr"] is not None, res["reference_check"]
    pq = res["per_query"]
    for r in res["rows"]:
        assert len(pq[r["label"]]) == len(names) and all(e["pose"] is not None and np.asarray(e["pose"]).shape == (4, 4) for e in pq[r["label"]]), r["label"]
    lab = "no render, descriptor branch only"
    assert lab in [r["label"] for r in res["rows"]]
    info = dict(scene="stand in", map_frames=len(mapf), hw=[T.H, T.W_], primitives=0, map_mb=0.0, model="stand in", weights_mb={"none": 0.0},
                reference="truth", mapping_seconds=100.0, load_seconds=3.0, map_seconds=4.0, field_seconds=5.0, split="1 : 3")
    gtext = C.report(res, info)
    short = C.in_short(res, info)
    assert any(lab in x for x in short) and "1 of the %d queries of the split have no accepted reference" % (len(names) + 1) in "\n".join(short)
    assert " mapping     112 s for %d frames" % len(mapf) in gtext and "\n             Stage 1 100 s, reading the Stage 1 output and loading the networks 3 s, " \
        "Gaussian map 4 s, medium field fit 5 s\n" in gtext
    print("benchmark on the stand in with a query numbered %d: every row keeps its poses, the reference check renders it at the time of a map "
          "frame (%.0f s)" % (OUTSIDE, t_bench))

    want = dict(verify=dict(map_images=mi, verify_cap=8, verify_boot=40, verify_seed=1),
                gap=dict(map_images=mi, ref_cache={}, gap_priors=["5:0"], gap_queries=2, gap_flow=False),
                medium=dict(law=TM.LAW, fast=False, query_info=[dict(name=n, heading_gap_deg=(10.0, 90.0)[k % 2]) for k, n in enumerate(names)]))
    out_dir = ctx["out_dir"]
    gpath = os.path.join(out_dir, "gscpe.txt")
    open(gpath, "w").write(gtext + "\n")
    logs = []
    a0 = time.time()
    texts, sshort, done = S.run(ctx, res, want, out_dir, text=gtext, path=gpath, log=logs.append)
    t_all = time.time() - a0
    assert set(done) == {"verify", "gap", "medium"} and len(texts) == 3 and not logs, logs
    assert max(seen) <= top and done["gap"]["free"] == 1 and done["gap"]["no_ref"]["rows"], done["gap"].get("notes")
    assert not done["verify"]["notes"] and not [x for x in done["gap"]["notes"] if "failed" in x] and not [x for x in done["medium"]["notes"] if "failed" in x]
    for k in ("verify", "gap", "medium"):
        assert os.path.isfile(os.path.join(out_dir, k + ".txt")) and os.path.isfile(os.path.join(out_dir, k + ".json")), k
    whole = open(gpath).read()
    i0, i1, i2, i3 = (whole.index(gtext), whole.index(" Pose verification"), whole.index("viewpoint gaps and a simulated navigation prior"),
                      whole.index("with the medium adapted to the query"))
    assert i0 == 0 and i0 < i1 < i2 < i3, (i0, i1, i2, i3)
    assert res["medium"]["rows"] and done["verify"]["queries"] == len(names) and done["gap"]["scored"] == len(names)
    assert done["verify"]["names"] == names and done["medium"]["names"] == names
    block = "\n".join(["=" * C.W] + short + sshort + ["=" * C.W])
    print(block)
    heads = [x[:12] for x in sshort if not x.startswith("  ")]
    assert heads == [" Verify     ", " Gap        ", " Medium     "], heads
    for x in sshort:
        assert ":" not in x and ";" not in x and "%%" not in x and " -- " not in x and "real time" not in x and len(x) <= C.W, (len(x), x)
    assert any("(d) minus (c)" in x for x in sshort) and any("sim. prior, render, match, PnP" in x for x in sshort)
    assert any("coarse, water adapted every pass" in x for x in sshort) and any("heading gap below 30 deg" in x for x in sshort)
    print("the three experiments ran after the benchmark on the same context (%.0f s), each report appended to gscpe.txt in order" % t_all)

    import bench_verify as V
    orig = V.run_verify

    def broken(ctx_, res_=None):
        raise RuntimeError("a stand in failure: of the verification")

    V.run_verify = broken
    try:
        d2 = tempfile.mkdtemp()
        g2 = os.path.join(d2, "gscpe.txt")
        logs = []
        small = dict(verify=want["verify"], gap=dict(want["gap"], gap_queries=1), medium=dict(want["medium"], tests=names[:2]))
        t2, s2, done2 = S.run(dict(ctx, out_dir=d2), dict(res), small, d2, text="BASE", path=g2, log=logs.append)
    finally:
        V.run_verify = orig
    assert set(done2) == {"gap", "medium"} and len(t2) == 2 and len(logs) == 1 and "[verify] the experiment failed and the run goes on" in logs[0], logs
    assert s2[0].startswith(" Verify      failed (RuntimeError, a stand in failure") and any(x.startswith(" Gap ") for x in s2) and any(x.startswith(" Medium ") for x in s2)
    assert "RuntimeError" in open(os.path.join(d2, "verify_error.txt")).read() and not os.path.isfile(os.path.join(d2, "verify.txt"))
    g2t = open(g2).read()
    assert g2t.startswith("BASE\n") and "viewpoint gaps" in g2t and "medium adapted" in g2t
    t3, s3, done3 = S.run(ctx, res, {}, tempfile.mkdtemp(), text="BASE")
    assert not t3 and not s3 and not done3
    print("a failing experiment prints one line, keeps its traceback in verify_error.txt, and the others still run")


def fake_tree(script):
    d = tempfile.mkdtemp()
    open(os.path.join(d, "run.sh"), "w").write(script)
    for sc in ("video1", "sealoc_s1"):
        src = os.path.join(ROOT, "data", sc)
        os.makedirs(os.path.join(d, "data", sc, "images"))
        for f in os.listdir(os.path.join(src, "images")):
            open(os.path.join(d, "data", sc, "images", f), "w").close()
        for f in ("train_files.txt", "split.json", "reference_poses.json"):
            if os.path.isfile(os.path.join(src, f)):
                shutil.copy(os.path.join(src, f), os.path.join(d, "data", sc, f))
    for f in ("ckpt/joint_s3r_v12/weights/0019_lora.pth", "ckpt/joint_s3r_v12/weights/0019_physics.pth", "ckpt/joint_s3r_v12/train_encoder_args.json",
              "ckpt/head_s3r_v13c/final_splatt3r.pth"):
        os.makedirs(os.path.dirname(os.path.join(d, f)), exist_ok=True)
        open(os.path.join(d, f), "w").close()
    os.makedirs(os.path.join(d, "tools", "csrc"))
    stub = os.path.join(d, "bin")
    os.makedirs(stub)
    open(os.path.join(stub, "python"), "w").write(
        "#!/usr/bin/env bash\n"
        "if [ \"$1\" = -c ]; then\n"
        "  case \"$2\" in *water_gn_ext*|*kornia*) exit 0 ;; esac\n"
        "  exec \"$REALPY\" \"$@\"\n"
        "fi\n"
        "echo \"python $*\" >> \"$CALLS\"\n")
    os.chmod(os.path.join(stub, "python"), 0o755)
    return d


def launch(script, env, args):
    d = fake_tree(script)
    e = {k: v for k, v in os.environ.items() if k not in ("HOLD", "BLOCK", "EVERY", "TAG", "SCENE", "FIXED_SPLIT", "REF", "HOLD_MAP", "REF_MATCHER", "TRT",
                                                          "WIN", "STRIDE", "LORA", "PHYS", "HEAD", "WEIGHTS", "REF_K")}
    e.update(env, PATH=os.path.join(d, "bin") + os.pathsep + os.environ["PATH"], CALLS=os.path.join(d, "calls.txt"), REALPY=sys.executable)
    p = subprocess.run(["bash", os.path.join(d, "run.sh")] + args, cwd="/", env=e, capture_output=True, text=True)
    calls = open(os.path.join(d, "calls.txt")).read().strip().split("\n") if os.path.isfile(os.path.join(d, "calls.txt")) else []
    norm = lambda s: re.sub(r"--bench_mapping_seconds \d+", "--bench_mapping_seconds N", re.sub(r"done in \d+ s", "done in N s", s))
    files = {}
    for r, _, fs in os.walk(os.path.join(d, "out")):
        for f in fs:
            q = os.path.join(r, f)
            files[os.path.relpath(q, d)] = "" if f == "mapping_seconds" else open(q).read()
    return dict(code=p.returncode, out=norm(p.stdout), err=p.stderr, calls=[norm(c) for c in calls], files=files, dir=d)


def check_run_sh():
    new = open(os.path.join(ROOT, "run.sh")).read()
    try:
        old = subprocess.run(["git", "show", "HEAD:run.sh"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except Exception as e:
        old = None
        print("run.sh of the last commit not readable (%s), the comparison with it is left out" % e)
    cases = [({}, ["v13c", "out"]), ({"HOLD": "12", "BLOCK": "3", "REF_MATCHER": "loftr", "TRT": "1"}, ["v13c", "out", "--gscpe", "--gscpe_no_ablation"]),
             ({"EVERY": "4"}, ["v13c", "out", "--proto"]), ({"EVERY": "1"}, ["v13c", "out"]),
             ({"HOLD": "12", "HOLD_MAP": "moved", "REF": "data/video1/split.json"}, ["v13c", "out", "--gscpe"]),
             ({"HOLD": "12", "BLOCK": "3", "HOLD_MAP": "same"}, ["v13c", "out", "--gscpe"]), ({"HOLD": "4", "HOLD_MAP": "moved"}, ["v13c", "out2"]),
             ({"TAG": "x", "STRIDE": "2"}, ["v13c", "out"]), ({"HOLD": "12", "BLOCK": "3", "REF_MATCHER": "ours", "TAG": "t"}, ["v13c", "out", "--gscpe"])]
    if old is not None and old != new:
        for env, args in cases:
            a, b = launch(old, env, args), launch(new, env, args)
            for k in ("code", "out", "calls", "files"):
                assert a[k] == b[k], (env, args, k, a[k], b[k])
            assert b["calls"] or b["code"], (env, b)
        print("run.sh with the default scene: %d launch cases give the same exit code, screen output, commands and files as the last commit" % len(cases))
    ref = "data/sealoc_s1/reference_poses.json"
    s = launch(new, {"SCENE": "data/sealoc_s1", "FIXED_SPLIT": "1", "REF": ref, "TRT": "1"}, ["v13c", "out", "--gscpe", "--medium_adapt"])
    assert s["code"] == 0 and len(s["calls"]) == 2, s
    c0, c1 = s["calls"]
    assert c0.startswith("python swimm3r_colmap.py --imagedir data/sealoc_s1/images --outdir out/map_v13c_sealoc_s1_fixed ") and \
        "--subset_list out/map_v13c_sealoc_s1_fixed/frames.txt" in c0, c0
    for want in ("--scene_dir out/map_v13c_sealoc_s1_fixed --data_dir data/sealoc_s1 ", "--out_dir out/v13c_sealoc_s1_fixed_moved/loc ",
                 "--split_json out/map_v13c_sealoc_s1_fixed/split.json --proto_ref_poses %s --proto_ref_frames 2 --trt_encoder --gscpe --medium_adapt" % ref):
        assert want in c1, (want, c1)
    tf = open(os.path.join(ROOT, "data", "sealoc_s1", "train_files.txt")).read()
    assert s["files"]["out/map_v13c_sealoc_s1_fixed/frames.txt"] == tf and json.loads(s["files"]["out/map_v13c_sealoc_s1_fixed/split.json"])["test_files"][0] == "100000.jpg"
    nr = launch(new, {"SCENE": "data/sealoc_s1", "FIXED_SPLIT": "1"}, ["v13c", "out", "--gscpe"])
    assert nr["code"] == 1 and "FIXED_SPLIT=1 needs REF" in nr["out"] and not nr["calls"], nr
    hm = launch(new, {"SCENE": "data/sealoc_s1", "FIXED_SPLIT": "1", "REF": ref, "HOLD_MAP": "register"}, ["v13c", "out"])
    assert hm["code"] == 1 and "HOLD_MAP=moved only" in hm["out"] and not hm["calls"], hm
    ho = launch(new, {"SCENE": "data/sealoc_s1", "HOLD": "12", "BLOCK": "3", "REF_MATCHER": "loftr"}, ["v13c", "out", "--gscpe"])
    assert ho["code"] == 0 and "--outdir out/map_v13c_sealoc_s1_hold12b3w5 " in ho["calls"][0] and "--out_dir out/v13c_sealoc_s1_hold12b3w5_loftr/loc " in ho["calls"][1]
    assert len(ho["files"]["out/map_v13c_sealoc_s1_hold12b3w5/all.txt"].split()) == 114 and "--data_dir data/sealoc_s1 " in ho["calls"][1]
    mv = launch(new, {"SCENE": "data/sealoc_s1", "HOLD": "12", "HOLD_MAP": "moved"}, ["v13c", "out"])
    assert mv["code"] == 1 and "--imagedir data/sealoc_s1/images --outdir out/map_v13c_sealoc_s1_all " in mv["calls"][0] and \
        "no reference mapping at out/map_v13c_sealoc_s1_all" in mv["out"], mv
    bad = launch(new, {"SCENE": "data/none"}, ["v13c", "out"])
    assert bad["code"] == 1 and "no images at data/none/images" in bad["out"] and not bad["calls"]
    same = launch(new, {"SCENE": "./data/video1/"}, ["v13c", "out"])
    dflt = launch(new, {}, ["v13c", "out"])
    assert [c.replace("./data/video1", "data/video1") for c in same["calls"]] == dflt["calls"] and same["files"] == dflt["files"] and same["code"] == 0
    print("run.sh with SCENE=data/sealoc_s1 FIXED_SPLIT=1 REF=...: maps train_files.txt, localizes the test_files of split.json, moves the reference in, "
          "folders named after the scene, refused without REF or with another HOLD_MAP, ./data/video1/ is the default scene")


def check_run_strategies():
    d = tempfile.mkdtemp()
    shutil.copy(os.path.join(ROOT, "run_strategies.sh"), os.path.join(d, "run_strategies.sh"))
    os.makedirs(os.path.join(d, "data", "sealoc_s1"))
    for f in ("reference_poses.json", "split.json"):
        open(os.path.join(d, "data", "sealoc_s1", f), "w").write("{}")
    rec = "echo \"$(basename \"$0\")|$HF_HUB_OFFLINE|${SCENE:-}|${FIXED_SPLIT:-}|${HOLD:-}|${BLOCK:-}|${HOLD_MAP:-}|${REF:-}|${REF_MATCHER:-}|${GAP_BLOCKS:-}|${TRT:-}|$*\" >> \"$PWD/calls.txt\"\n"
    mk = lambda o, tag: "mkdir -p \"%s\"; echo table > \"%s/gscpe.txt\"; echo \"SHORT %s\" > \"%s/short.txt\"\n" % (o, o, tag, o)
    open(os.path.join(d, "run_gscpe.sh"), "w").write("#!/usr/bin/env bash\n" + rec + mk("$2/${1}_hold12b3w${WIN}_loftr/loc", "ONE"))
    open(os.path.join(d, "run_gap.sh"), "w").write("#!/usr/bin/env bash\n" + rec + mk("$2/${1}_hold12b6w${WIN}_loftr/loc", "SIX") + "[ -z \"${FAIL_GAP:-}\" ]\n")
    open(os.path.join(d, "run.sh"), "w").write("#!/usr/bin/env bash\n" + rec + mk("$2/${1}_sealoc_s1_fixed_moved/loc", "THREE"))
    env = {k: v for k, v in os.environ.items() if k not in ("TAG", "SCENE", "HOLD", "BLOCK", "GAP_BLOCKS", "TRT", "WIN", "SEALOC")}
    p = subprocess.run(["bash", os.path.join(d, "run_strategies.sh"), "v13c", "out", "--gscpe_no_ablation"], cwd="/", env=dict(env, FAIL_GAP="1"),
                       capture_output=True, text=True)
    calls = open(os.path.join(d, "calls.txt")).read().strip().split("\n")
    print("\n".join(p.stdout.strip().split("\n")[-14:]))
    assert p.returncode == 1 and len(calls) == 3, (p.returncode, p.stdout, p.stderr)
    g, a, s = [c.split("|") for c in calls]
    assert g == ["run_gscpe.sh", "1", "data/video1", "0", "12", "3", "register", "", "loftr", "", "",
                 "v13c out --verify_bench --gap_bench --medium_adapt --gscpe_no_ablation"], g
    assert a[:3] == ["run_gap.sh", "1", "data/video1"] and a[9] == "6 9" and a[11] == "v13c out --gscpe_no_ablation", a
    assert s == ["run.sh", "1", "data/sealoc_s1", "1", "0", "", "moved", "data/sealoc_s1/reference_poses.json", "", "", "1",
                 "v13c out --gscpe --verify_bench --gap_bench --medium_adapt --gscpe_ref_metric --gscpe_depth_m 2.05 --gscpe_no_ablation"], s
    real = os.path.realpath(d)
    o = p.stdout.replace(d, real)
    for sub in ("v13c_hold12b3w5_loftr", "v13c_hold12b6w5_loftr", "v13c_hold12b9w5_loftr", "v13c_sealoc_s1_fixed_moved"):
        assert os.path.join(real, "out", sub, "loc", "gscpe.txt") in o, sub
    assert "step 2 failed" in o and "step 1 done" in o and "step 3 done" in o and "b9w5_loftr/loc/gscpe.txt (missing)" in o
    tail = o[o.index("strategies done"):]
    assert tail.index("SHORT ONE") < tail.index("SHORT SIX") < tail.index("SHORT THREE"), tail
    q = subprocess.run(["bash", os.path.join(d, "run_strategies.sh")], cwd="/", env=dict(env, GAP_BLOCKS="9"), capture_output=True, text=True)
    assert q.returncode == 0 and "step 2 done" in q.stdout and "v17_hold12b9w5_loftr" in q.stdout, (q.returncode, q.stdout)
    assert open(os.path.join(d, "calls.txt")).read().strip().split("\n")[4].split("|")[9] == "9"
    print("run_strategies.sh with stand ins for the launchers: three steps in order with HF_HUB_OFFLINE=1, a failed step does not stop the next, "
          "every gscpe.txt is named and the short blocks are printed again at the end")


if __name__ == "__main__":
    torch.manual_seed(0)
    check_parser()
    check_sample()
    check_mapping_lines()
    check_run_sh()
    check_run_strategies()
    check_run()
    print("test_strategies ok")
