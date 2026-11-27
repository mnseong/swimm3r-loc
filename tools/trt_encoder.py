import hashlib
import importlib.util
import inspect
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchRoPE2D(nn.Module):
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = float(freq)
        self.F0 = float(F0)
        self.cache = {}
        self.fixed_len = 0

    def get_cos_sin(self, D, seq_len, device, dtype):
        key = (D, seq_len, device, dtype)
        if key not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            self.cache[key] = (freqs.cos(), freqs.sin())
        return self.cache[key]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        cos = F.embedding(pos1d, cos)[:, None, :, :]
        sin = F.embedding(pos1d, sin)[:, None, :, :]
        return tokens * cos + self.rotate_half(tokens) * sin

    def forward(self, tokens, positions):
        D = tokens.size(3) // 2
        n = self.fixed_len if self.fixed_len > 0 else int(positions.max()) + 1
        cos, sin = self.get_cos_sin(D, n, tokens.device, tokens.dtype)
        y, x = tokens.chunk(2, dim=-1)
        y = self.apply_rope1d(y, positions[:, :, 0], cos, sin)
        x = self.apply_rope1d(x, positions[:, :, 1], cos, sin)
        return torch.cat((y, x), dim=-1)


class EncoderExport(nn.Module):
    def __init__(self, net, h, w):
        super().__init__()
        self.net = net
        pe = net.patch_embed
        dev = next(net.parameters()).device
        with torch.no_grad():
            _, pos = pe(torch.zeros(1, 3, int(h), int(w), device=dev), true_shape=torch.tensor([[int(h), int(w)]], device=dev))
        self.register_buffer("pos", pos.clone())
        self.portrait = type(pe).__name__ == "ManyAR_PatchEmbed" and int(h) > int(w)
        for blk in net.enc_blocks:
            if isinstance(getattr(blk.attn, "rope", None), TorchRoPE2D):
                blk.attn.rope.fixed_len = int(pos.max()) + 1

    def forward(self, image):
        pe = self.net.patch_embed
        x = pe.proj(image.transpose(-1, -2) if self.portrait else image)
        x = pe.norm(x.flatten(2).transpose(1, 2))
        pos = self.pos.expand(image.shape[0], -1, -1)
        for blk in self.net.enc_blocks:
            x = blk(x, pos)
        return self.net.enc_norm(x)


def swap_rope(net):
    saved = []
    for blk in net.enc_blocks:
        r = getattr(blk.attn, "rope", None)
        if r is not None:
            saved.append((blk.attn, r))
            blk.attn.rope = TorchRoPE2D(freq=getattr(r, "base", 100.0), F0=getattr(r, "F0", 1.0))
    return saved


def restore_rope(saved):
    for attn, r in saved:
        attn.rope = r


def export_onnx(net, h, w, path, dev):
    if int(torch.__version__.split(".")[0]) >= 2 and importlib.util.find_spec("onnx") is None:
        raise ModuleNotFoundError("the ONNX export of torch %s needs the onnx package: python -m pip install onnx \"numpy==$(python -c 'import numpy; "
                                  "print(numpy.__version__)')\" in this environment" % torch.__version__)
    saved = swap_rope(net)
    try:
        wrap = EncoderExport(net, h, w).to(dev).eval()
        dummy = torch.zeros(1, 3, int(h), int(w), device=dev)
        kw = dict(input_names=["image"], output_names=["tokens"], opset_version=17, do_constant_folding=True)
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            kw["dynamo"] = False
        with torch.no_grad():
            torch.onnx.export(wrap, (dummy,), path, **kw)
    finally:
        restore_rope(saved)
    return path


def import_tensorrt():
    try:
        import tensorrt
        return tensorrt
    except ImportError:
        pass
    for p in ("/usr/lib/python3.%d/dist-packages" % sys.version_info.minor, "/usr/lib/python3/dist-packages"):
        if os.path.isdir(os.path.join(p, "tensorrt")) and p not in sys.path:
            sys.path.append(p)
            try:
                import tensorrt
                return tensorrt
            except ImportError:
                pass
            finally:
                sys.path.remove(p)
    raise ImportError("no tensorrt python module in this environment or in /usr/lib/python3*/dist-packages")


def build_engine(onnx_path, engine_path, fp16=True, workspace_gb=4.0):
    trt = import_tensorrt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH") and int(trt.__version__.split(".")[0]) < 10:
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError("the ONNX graph of the encoder does not parse: " + errs[:500])
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    data = builder.build_serialized_network(network, config)
    if data is None:
        raise RuntimeError("TensorRT did not build the encoder engine")
    with open(engine_path, "wb") as f:
        f.write(bytes(data))
    return engine_path


class TRTEncoder:
    def __init__(self, engine_path, dev, hw):
        trt = import_tensorrt()
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("the engine %s does not load with this TensorRT" % engine_path)
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inp = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.outp = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        dt = {trt.float32: torch.float32, trt.float16: torch.float16}
        self.out = torch.empty(tuple(self.engine.get_tensor_shape(self.outp)), dtype=dt[self.engine.get_tensor_dtype(self.outp)], device=dev)
        self.in_dtype = dt[self.engine.get_tensor_dtype(self.inp)]
        self.hw = tuple(int(x) for x in hw)
        self.dev = dev

    def __call__(self, image):
        x = image.to(self.dev, self.in_dtype).contiguous()
        self.ctx.set_tensor_address(self.inp, x.data_ptr())
        self.ctx.set_tensor_address(self.outp, self.out.data_ptr())
        if not self.ctx.execute_async_v3(torch.cuda.current_stream(self.dev).cuda_stream):
            raise RuntimeError("TensorRT execution of the encoder failed")
        return self.out.float().clone()


def cache_key(lora_path, h, w):
    trt = import_tensorrt()
    st = os.stat(lora_path) if lora_path and os.path.isfile(lora_path) else None
    raw = "%s|%s|%s|%d|%d|%s|%s|fp16" % (os.path.abspath(lora_path or ""), st.st_size if st else 0, int(st.st_mtime) if st else 0, int(h), int(w),
                                         trt.__version__, torch.__version__)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


@torch.no_grad()
def compare(enc_trt, net, frames, shape, reps=10):
    cos, rel = [], []
    for img in frames:
        x = img[None].to(enc_trt.dev) * 2.0 - 1.0
        ref = net._encode_image(x, shape)[0].float()
        out = enc_trt(x)
        cos.append(F.cosine_similarity(out[0], ref[0], dim=-1))
        rel.append(float((out - ref).norm() / ref.norm().clamp_min(1e-12)))
    cos = torch.cat(cos)
    x = frames[0][None].to(enc_trt.dev) * 2.0 - 1.0
    times = {}
    for name, fn in (("tensorrt", lambda: enc_trt(x)), ("pytorch", lambda: net._encode_image(x, shape)[0])):
        fn()
        torch.cuda.synchronize()
        a = time.time()
        for _ in range(int(reps)):
            fn()
        torch.cuda.synchronize()
        times[name] = (time.time() - a) / float(reps)
    return dict(cos_mean=float(cos.mean()), cos_min=float(cos.min()), rel=float(max(rel)), ms_trt=1e3 * times["tensorrt"], ms_torch=1e3 * times["pytorch"])


def attach(loc, net, lora_path, h, w, dev, cache_dir, frames, min_cos_mean=0.999, min_cos=0.98, log=print):
    os.makedirs(cache_dir, exist_ok=True)
    key = cache_key(lora_path, h, w)
    engine = os.path.join(cache_dir, "encoder_%dx%d_%s.engine" % (int(h), int(w), key))
    info = dict(engine=engine, built=False, seconds_build=0.0)
    if not os.path.isfile(engine):
        onnx_path = os.path.join(cache_dir, "encoder_%dx%d_%s.onnx" % (int(h), int(w), key))
        a = time.time()
        log("[trt] exporting the encoder (LoRA merged, %d x %d) to ONNX and building an FP16 TensorRT engine, once (minutes)" % (int(h), int(w)))
        export_onnx(net, h, w, onnx_path, dev)
        try:
            build_engine(onnx_path, engine)
        finally:
            for f in os.listdir(cache_dir):
                if f.startswith(os.path.basename(onnx_path)) or f == os.path.basename(onnx_path):
                    os.remove(os.path.join(cache_dir, f))
        info.update(built=True, seconds_build=time.time() - a)
    enc = TRTEncoder(engine, dev, (h, w))
    shape = torch.tensor([[int(h), int(w)]], device=dev)
    with torch.no_grad():
        _, pos, _ = net._encode_image(frames[0][None].to(dev) * 2.0 - 1.0, shape)
    info.update(compare(enc, net, frames, shape))
    info["ok"] = bool(info["cos_mean"] >= min_cos_mean and info["cos_min"] >= min_cos)
    log("[trt] engine %s (%s): tokens against PyTorch cosine mean %.5f, lowest %.4f, relative difference %.4f | encoder %.1f ms against %.1f ms | %s" % (
        os.path.basename(engine), "built in %.0f s" % info["seconds_build"] if info["built"] else "cached", info["cos_mean"], info["cos_min"], info["rel"],
        info["ms_trt"], info["ms_torch"], "used" if info["ok"] else "not used, the tokens differ too much"))
    if info["ok"]:
        loc.trt, loc.trt_pos, loc.trt_info = enc, pos, info
    return info
