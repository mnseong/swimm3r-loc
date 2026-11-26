import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import mast3r.utils.path_to_dust3r
import bench_proto as P
import eval_gaussian_map as E
import loc_ff
import loc_geom as G

H, W_, FOCAL, DEPTH = 128, 192, 180.0, 4.0
BALL, RADIUS = (2.4, 0.0, 11.0), 8.0
PP = (W_ / 2.0, H / 2.0)
X0, RX, Y0, RY = -3.5, 12.0, -3.0, 6.0


def true_pose(i):
    j = np.random.default_rng(100 + i)
    p = np.eye(4)
    p[:3, :3] = G.rot(np.radians(np.array([1.5 * np.sin(0.3 * i), 2.0 * np.sin(0.2 * i + 1.0), 1.0 * np.cos(0.25 * i)]) + j.normal(scale=0.7, size=3)))
    p[:3, 3] = np.array([0.12 * i, 0.15 * np.sin(0.15 * i), 0.1 * np.cos(0.1 * i)]) + j.normal(scale=0.04, size=3)
    return p


def surface_points(pose):
    v, u = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W_, dtype=torch.float32), indexing="ij")
    d = torch.stack([(u - (PP[0] - 0.5)) / FOCAL, (v - (PP[1] - 0.5)) / FOCAL, torch.ones_like(u)], -1) @ pose[:3, :3].T
    oc = pose[:3, 3] - torch.tensor(BALL, dtype=pose.dtype)
    a, b, c = (d * d).sum(-1), 2.0 * (d * oc).sum(-1), (oc * oc).sum() - RADIUS ** 2
    s = (-b - torch.sqrt((b * b - 4.0 * a * c).clamp_min(1e-9))) / (2.0 * a)
    return pose[:3, 3] + s[..., None] * d, s


def render(pose, med=None, t=None, alpha=False):
    p, _ = surface_points(pose)
    img = torch.stack([(p[..., 0] - X0) / RX, (p[..., 1] - Y0) / RY, 0.5 + 0.25 * torch.sin(3.0 * p[..., 0]) * torch.cos(2.0 * p[..., 1])])
    return (img, torch.ones(H, W_)) if alpha else img


class StandIn:
    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.freq = torch.rand(16, 2, generator=g) * 2.0 - 1.0
        self.freq = self.freq * torch.linspace(2.0, 120.0, 16)[:, None]
        self.bank = []

    def _encode_image(self, x, shape):
        img = (x + 1.0) / 2.0
        ph = 2.0 * np.pi * torch.einsum("bchw,kc->bhwk", img[:, :2], self.freq)
        desc = F.normalize(torch.cat([torch.sin(ph), torch.cos(ph)], -1), dim=-1)
        tok = F.avg_pool2d(desc.permute(0, 3, 1, 2), 16).flatten(2).transpose(1, 2).contiguous()
        for b in range(tok.shape[0]):
            self.bank.append((tok[b].clone(), desc[b].clone()))
        return tok, torch.zeros(tok.shape[0], tok.shape[1], 2, dtype=torch.long), None

    def _decoder(self, f1, pos1, f2, pos2):
        return [f1], [f2]

    def _downstream_head(self, num, dec, shape):
        out = []
        for t in dec[-1]:
            out.append(min(self.bank, key=lambda e: float((e[0] - t).abs().sum()))[1])
        desc = torch.stack(out)
        return dict(desc=desc, desc_conf=torch.ones(desc.shape[:3]))


def check_geometry():
    rng = np.random.default_rng(0)
    K, hw = G.intrinsics(270.0, (256.0, 144.0)), (288, 512)
    true = np.eye(4)
    true[:3, :3] = G.rot(np.radians([3.0, -4.0, 2.0]))
    true[:3, 3] = [0.3, -0.1, 0.2]
    pts = np.c_[rng.uniform(-4, 4, 3000), rng.uniform(-2.5, 2.5, 3000), rng.uniform(3, 7, 3000)]
    uv, z = G.project(K, true, pts)
    ok = (z > 0) & G.inside(uv, hw)
    p3d, p2d = pts[ok], uv[ok] + rng.normal(scale=0.5, size=(int(ok.sum()), 2))
    bad = rng.random(len(p3d)) < 0.3
    p2d[bad] = np.c_[rng.uniform(0, 512, int(bad.sum())), rng.uniform(0, 288, int(bad.sum()))]
    r = G.solve_pnp(p3d, p2d, K, 4.7, G.perturb(true, 0.5, 6.0, rng))
    e = G.pose_error(r["c2w"], true, 5.0)
    print("PnP with 30 %% wrong matches: %.3f %% of depth, %.3f deg, %.3f px, solver %s" % (e[0], e[1], G.pixel_shift(K, true, r["c2w"], pts, hw), r["solver"]))
    assert e[0] < 0.2 and e[1] < 0.1, e
    yy, xx = np.mgrid[0:288, 0:512].astype(np.float64)
    tex = lambda x, y: 0.5 + 0.25 * np.sin(x / 7.0) * np.cos(y / 5.0) + 0.2 * np.sin((x + 2 * y) / 13.0) + 0.05 * np.sin(x / 2.3 + y / 3.1)
    f = G.flow_residual(np.stack([tex(xx, yy)] * 3, -1), np.stack([tex(xx - 3.0, yy - 2.0)] * 3, -1))
    print("optical flow of a known shift of 3.61 px: %s" % ("%.2f px" % f if np.isfinite(f) else "not available (no OpenCV optical flow)"))
    assert not np.isfinite(f) or abs(f - 3.61) < 0.3, f


def main():
    torch.manual_seed(0)
    check_geometry()
    dev = torch.device("cpu")
    nums = list(range(2, 41))
    map_nums = [i for i in nums if i % 8 in (2, 3, 4, 5, 6)]
    test_nums = [i for i in nums if i % 8 in (7, 0, 1) and 7 <= i <= 33]
    centre = [i for i in test_nums if i % 8 == 0]
    name = lambda i: "%06d.jpg" % i
    truth = {i: true_pose(i) for i in nums}
    tt = lambda p: torch.tensor(p, dtype=torch.float32)
    with torch.no_grad():
        imgs = torch.stack([render(tt(truth[i])) for i in map_nums])
        depth = torch.stack([surface_points(tt(truth[i]))[1] for i in map_nums])
    c2w = torch.stack([tt(truth[i]) for i in map_nums])
    intr = torch.tensor([[FOCAL, 0.0, PP[0] - 0.5], [0.0, FOCAL, PP[1] - 0.5], [0.0, 0.0, 1.0]]).expand(len(map_nums), 3, 3).contiguous()
    conf = torch.full_like(depth, 2.0)
    net = StandIn()
    loc = loc_ff.Localizer(net, loc_ff.encode_frames(net, imgs), depth, conf, intr, c2w, FOCAL, PP, 1.0, dev, subsample=4, reproj=2.5)
    ball = (loc.xyz - torch.tensor(BALL)).norm(dim=-1)
    assert tuple(loc.xyz.shape) == (len(map_nums), H, W_, 3) and float((ball - RADIUS).abs().max()) < 1e-3
    poses_np = c2w.numpy().astype(np.float64)
    tests = [name(i) for i in test_nums]
    gen = torch.Generator().manual_seed(1)

    def load(nm):
        with torch.no_grad():
            return (render(tt(truth[E.fnum(nm)])) + 0.004 * torch.randn(3, H, W_, generator=gen)).clamp(0.0, 1.0)

    def score(img, gt):
        mae = float((img.clamp(0, 1) - gt).abs().mean())
        return [float(-10.0 * torch.log10(((img.clamp(0, 1) - gt) ** 2).mean() + 1e-12)), 1.0 - mae, mae]

    one = loc.localize(load(tests[0]), prior=truth[test_nums[0] - 1], topk=1)
    e1 = G.pose_error(one["pose"], truth[test_nums[0]], DEPTH)
    print("one frame: %s, %d inliers of %d matches, %.3f %% of depth and %.3f deg from the true pose, solver %s" % (
        "solved" if one["ok"] else "failed", one["inliers"], one["matches"], e1[0], e1[1], one["solver"]))
    assert one["ok"] and e1[0] < 1.0 and e1[1] < 0.5, e1
    two = loc.localize(load(tests[0]), topk=3)
    e2 = G.pose_error(two["pose"], truth[test_nums[0]], DEPTH)
    print("no start pose: map frames %s chosen for frame %d, %.3f %% and %.3f deg" % ([map_nums[i] for i in two["idx"]], test_nums[0], e2[0], e2[1]))
    assert two["ok"] and e2[0] < 1.0 and e2[1] < 0.5, e2

    out_dir = tempfile.mkdtemp()
    ctx = dict(dev=dev, tests=tests, centre=[name(i) for i in centre], fps=5.0, unit=1.0, lr=4e-3, warmup=1, map_nums=map_nums, map_poses=poses_np,
               interp=E.interpolate_poses(poses_np, test_nums, map_nums), load=load, medium=lambda num, pose: {}, render=render, score=score,
               loc=loc, loc_public=loc, out_dir=out_dir, rc_steps=[5, 20, 60], ff_steps=[5, 10], topk=[1, 3], ref_steps=40, ref_check=120,
               ref_frames=4, ref_tol=1.0, px_tol=1.0, conv_ff_steps=5, conv_rc_steps=40, reloc_rc_max=20, levels=((2.0, 1.0), (10.0, 5.0)),
               map_names=[name(i) for i in map_nums], strides=[2, 4], gaps=[8])
    res = P.run(ctx)
    text = P.report(res, dict(scene="synthetic scene", map_frames=len(map_nums), hw=[H, W_], primitives=0, map_mb=0.0, model="stand in",
                              public=True, mapping_seconds=None, weights_mb={"none": 0.0}, warmup=1))
    print(text)
    ref = {int(k): v for k, v in __import__("json").load(open(os.path.join(out_dir, "reference.json")))["frames"].items()}
    order = [int(j) for j in np.argsort(test_nums)]
    err = np.array([G.pose_error(np.asarray(ref[j]["pose"]), truth[test_nums[j]], DEPTH) for j in order])
    print("reference against the true poses of the synthetic scene: median %.3f %% of depth, %.3f deg, largest %.3f %%, %.3f deg" % (
        np.median(err[:, 0]), np.median(err[:, 1]), err[:, 0].max(), err[:, 1].max()))
    assert res["reference"]["verified"] >= 0.8 * len(tests), res["reference"]
    assert res["reference"]["check_median"][2] < 0.6 and res["reference"]["pnp_median"][2] < 0.5, res["reference"]
    assert np.median(err[:, 0]) < 1.0 and np.median(err[:, 1]) < 0.5 and "%%" not in text, err
    row = lambda task, label: next(r for r in res[task] if r["label"] == label)
    a, a0 = row("tracking", "matching + PnP"), row("tracking", "start pose, no computation")
    assert a["s5"] == 100.0 and a["sp"] == 100.0 and a["fails"] == 0 and a["px"] < 0.5 and a["px"] < 0.2 * a0["px"], (a, a0)
    b = row("reloc", "matching + PnP, best 3")
    assert b["s5"] == 100.0 and b["sp"] == 100.0 and b["px"] < 0.5, b
    assert res["retrieval"]["top3"] >= 80.0, res["retrieval"]
    lv = res["convergence"]["levels"]
    assert all(x["ff"]["sp"] == 100.0 for x in lv) and lv[1]["start"]["px"] > 3.0 * lv[0]["start"]["px"] and lv[1]["start"]["sp"] == 0.0, lv
    assert res["exact"]["px"] < 0.5 and res["exact"]["sp"] == 100.0 and res["exact"]["ok"] == 100.0, res["exact"]
    assert row("tracking", "render and compare 60")["lpips"] <= row("tracking", "start pose, no computation")["lpips"] + 1e-6
    again = P.run(dict(ctx, only_reference=True))
    assert again["reference"]["seconds"] == 0.0 and again["reference"]["verified"] == res["reference"]["verified"]
    th = res["sparse"]
    assert [x["stride"] for x in th["strides"]] == [2, 4] and th["strides"][0]["all"]["sp"] >= 90.0 and th["strides"][1]["all"]["n"] > th["strides"][0]["all"]["n"], th["strides"]
    assert th["gaps"] and th["gaps"][0]["all"]["n"] >= 4 and "Thinned map" in text, th["gaps"]
    hard = P.run(dict(ctx, only_thinned=True, strides=[12], gaps=[24], accept_inliers=2000))
    assert not hard["tracking"] and hard["sparse"]["strides"][0]["all"]["again"] == 100.0 and hard["sparse"]["strides"][0]["all"]["accepted_good"] == 0.0, hard["sparse"]
    hard_text = P.report(hard, dict(scene="synthetic scene", map_frames=len(map_nums), hw=[H, W_], primitives=0, map_mb=0.0, model="stand in",
                                    public=False, mapping_seconds=None, weights_mb={"none": 0.0}, warmup=1))
    assert "Thinned map" in hard_text and "Tracking " not in hard_text
    print(hard_text[hard_text.index(" Thinned map"):hard_text.index(" Footprint")])
    print("test_loc ok")


if __name__ == "__main__":
    main()
