import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import mast3r.utils.path_to_dust3r  # noqa: F401
import trt_encoder as TE

try:
    import huggingface_hub  # noqa: F401
except ModuleNotFoundError:
    import types

    class _Mixin:
        def __init_subclass__(cls, **kw):
            super().__init_subclass__()

    sys.modules["huggingface_hub"] = types.SimpleNamespace(__version__="0.22.0", PyTorchModelHubMixin=_Mixin)


def small_net():
    from mast3r.model import AsymmetricMASt3R
    torch.manual_seed(0)
    net = AsymmetricMASt3R(output_mode="pts3d+desc24", head_type="catmlp+dpt", depth_mode=("exp", -float("inf"), float("inf")),
                           conf_mode=("exp", 1, float("inf")), enc_embed_dim=64, enc_depth=2, enc_num_heads=4, dec_embed_dim=48, dec_depth=12, dec_num_heads=4,
                           img_size=(64, 96), patch_embed_cls="PatchEmbedDust3R", pos_embed="RoPE100", two_confs=True, desc_conf_mode=("exp", 0, float("inf")))
    return net.eval()


def check_rope():
    import dust3r.utils.path_to_croco  # noqa: F401
    from models.pos_embed import RoPE2D
    torch.manual_seed(1)
    tokens = torch.randn(2, 4, 24, 16)
    pos = torch.stack(torch.meshgrid(torch.arange(4), torch.arange(6), indexing="ij"), -1).reshape(1, 24, 2).expand(2, 24, 2).contiguous()
    ref = RoPE2D(freq=100.0)(tokens.clone(), pos)
    mine = TE.TorchRoPE2D(freq=100.0)(tokens.clone(), pos)
    diff = float((ref - mine).abs().max())
    print("PyTorch RoPE used for the export against the RoPE of the model (%s): largest difference %.1e" % (RoPE2D.__name__, diff))
    assert diff < 1e-6


def check_export():
    net = small_net()
    h, w = 64, 96
    x = torch.rand(1, 3, h, w) * 2.0 - 1.0
    shape = torch.tensor([[h, w]])
    with torch.no_grad():
        ref = net._encode_image(x, shape)[0]
        saved = TE.swap_rope(net)
        swapped = TE.EncoderExport(net, h, w)(x)
        TE.restore_rope(saved)
        back = net._encode_image(x, shape)[0]
    assert len(saved) == len(net.enc_blocks) and all(type(blk.attn.rope).__name__ != "TorchRoPE2D" for blk in net.enc_blocks)
    d1, d2 = float((ref - swapped).abs().max()), float((ref - back).abs().max())
    print("encoder with the export RoPE against the model: largest difference %.1e (restored afterwards %.1e), tokens %s" % (d1, d2, tuple(ref.shape)))
    assert d1 < 1e-5 and d2 == 0.0
    path = os.path.join(tempfile.mkdtemp(), "enc.onnx")
    try:
        TE.export_onnx(net, h, w, path, torch.device("cpu"))
        print("ONNX export of the encoder: %.1f MB written" % (os.path.getsize(path) / 1e6))
        assert os.path.getsize(path) > 1e4
    except ModuleNotFoundError as e:
        print("ONNX export not checked here: %s" % e)


def check_localizer():
    import loc_ff
    import test_loc as T

    class FakeTRT:
        def __init__(self, net, hw):
            self.net, self.hw, self.calls = net, hw, 0

        def __call__(self, image):
            self.calls += 1
            return self.net._encode_image(image, torch.tensor([[self.hw[0], self.hw[1]]]))[0] + 0.0

    net = T.StandIn()
    H, W_ = T.H, T.W_
    imgs = torch.stack([T.render(torch.tensor(np.asarray(T.true_pose(i)), dtype=torch.float32)) for i in (2, 3, 4)])
    depth = torch.stack([T.surface_points(torch.tensor(np.asarray(T.true_pose(i)), dtype=torch.float32))[1] for i in (2, 3, 4)])
    c2w = torch.stack([torch.tensor(np.asarray(T.true_pose(i)), dtype=torch.float32) for i in (2, 3, 4)])
    intr = torch.tensor([[T.FOCAL, 0.0, T.PP[0] - 0.5], [0.0, T.FOCAL, T.PP[1] - 0.5], [0.0, 0.0, 1.0]]).expand(3, 3, 3).contiguous()
    loc = loc_ff.Localizer(net, loc_ff.encode_frames(net, imgs), depth, torch.full_like(depth, 2.0), intr, c2w, T.FOCAL, T.PP, 1.0, torch.device("cpu"))
    fake = FakeTRT(net, (H, W_))
    f0 = loc.encode(imgs[0])
    loc.trt, loc.trt_pos = fake, torch.full((1, 3, 2), 7, dtype=torch.long)
    f1 = loc.encode(imgs[0])
    assert fake.calls == 0 and not bool((loc.pos == 7).all()), "the engine must stay off unless use_trt is set"
    loc.use_trt = True
    f2 = loc.encode(imgs[0])
    assert fake.calls == 1 and bool((loc.pos == 7).all()) and float((f2 - f0).abs().max()) < 1e-6 and float((f1 - f0).abs().max()) == 0.0
    small = loc.encode(imgs[0][:, :64, :96])
    assert fake.calls == 1 and small.shape[1] != f2.shape[1], "another image size goes through PyTorch"
    print("localizer: the TensorRT encoder is used only when switched on and for its own image size, positions taken from the PyTorch pass")


def check_bench():
    import bench_gscpe as C
    import test_fast as TF
    import test_loc as T
    import eval_gaussian_map as E
    loc, hold, mapf, truth = TF.scene()

    class FakeTRT:
        def __init__(self, net, hw):
            self.net, self.hw = net, hw

        def __call__(self, image):
            return self.net._encode_image(image, torch.tensor([[self.hw[0], self.hw[1]]]))[0]

    loc.trt = FakeTRT(loc.net, (T.H, T.W_))
    loc.trt_pos = loc.net._encode_image(torch.zeros(1, 3, T.H, T.W_), None)[1]
    loc.trt_info = dict(built=False, seconds_build=0.0, cos_mean=0.99999, cos_min=0.9995, ms_trt=50.0, ms_torch=150.0, ok=True)
    tt = lambda p: torch.tensor(np.asarray(p), dtype=torch.float32)
    gen = torch.Generator().manual_seed(5)
    name = lambda i: "%06d.jpg" % i

    def load(nm):
        with torch.no_grad():
            return (T.render(tt(truth[E.fnum(nm)])) + 0.004 * torch.randn(3, T.H, T.W_, generator=gen)).clamp(0.0, 1.0)

    def strict(pose, med=None, t=None, alpha=False, which="underwater", depth=False):
        return T.render(pose, med, t, alpha, which, depth)

    def score(img, gt):
        mae = float((img.clamp(0, 1) - gt).abs().mean())
        return [float(-10.0 * torch.log10(((img.clamp(0, 1) - gt) ** 2).mean() + 1e-12)), 1.0 - mae, mae]

    res = C.run(dict(dev=torch.device("cpu"), tests=[name(i) for i in hold], fps=5.0, warmup=1, map_nums=mapf, load=load, medium=lambda num, pose: {},
                     render=strict, score=score, loc=loc, loc_public=None, out_dir=tempfile.mkdtemp(), ref_poses={name(i): truth[i] for i in hold},
                     ref_fit=dict(n=len(mapf), scale=1.0, t_med=0.0, t_max=0.0, r_med=0.0, r_max=0.0), ref_source="truth", fast_photo="affine",
                     fast_levels=(2, 1, 0), fast_iters=(6, 4, 3), fast_dim=16, ablation=False))
    text = C.report(res, dict(scene="synthetic", map_frames=len(mapf), hw=[T.H, T.W_], primitives=0, map_mb=0.0, model="stand in", mapping_seconds=None,
                              weights_mb={"none": 0.0}, reference="truth", split="1 : 3"))
    labels = [r["label"] for r in res["rows"]]
    assert "fast: TensorRT encoder" in labels and "fast + check, TensorRT encoder" in labels, labels
    assert "TensorRT encoder: FP16 engine" in text and "fast: TensorRT encoder" in text[:text.index(" Reference")] and loc.use_trt is False
    row = next(r for r in res["rows"] if r["label"] == "fast + check, TensorRT encoder")
    assert row["px"] < 0.6, row
    print("benchmark: TensorRT rows present and accurate on the stand in (%.2f px), the engine is switched off after them" % row["px"])


if __name__ == "__main__":
    check_rope()
    check_export()
    check_localizer()
    check_bench()
    print("test_trt ok")
