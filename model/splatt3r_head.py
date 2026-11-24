"""Splatt3R Gaussian head on Swimm3R (Smart et al. 2024, github.com/btsmart/splatt3r, CC BY-NC 4.0).

The head of Splatt3R as it is: a DPT head (dust3r PixelwiseTaskWithDPT, the pointmap head design) next to the
pointmap head of the reference view, on the decoder tokens of hooks 0, 6, 9, 12, with 14 output channels:
centre offset 3, scale 3, rotation 4 (x y z w), colour 3 (the DC spherical harmonic, sh_degree 1), opacity 1.
Activations of Splatt3R: offset = unit direction * (exp(|o| - 6) - exp(-6)), scale = exp, rotation normalised,
opacity sigmoid, colour a DC residual added to the base colour in SH units. The last layer is initialised per
slice as in Splatt3R (xavier gains and biases: offset 0.001/0.001, scale 3e-5/-7, rotation 1/0, colour 1/0,
opacity 1/-2), the trunk keeps the constructor initialisation, and load_splatt3r() copies the trained head of
the public checkpoint (encoder.downstream_head1.gaussian_dpt).

Coupling to Swimm3R (not in Splatt3R): the base colour is the restored colour J of the pixel, the Stage 1
inversion of the Swimm3R physics head (training/physics_splat.py), so a primitive carries the medium free
radiance; the underwater appearance is rendered through the image formation model with the medium of the
physics head. Centres: the pixel at the depth of our own geometry plus the predicted offset.
"""

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

import mast3r.utils.path_to_dust3r  # noqa: F401
from dust3r.heads.dpt_head import PixelwiseTaskWithDPT

SH_C0 = 0.28209479177387814
N_SPLATT3R = 14
SLICES = ((0, 3), (3, 6), (6, 10), (10, 13), (13, 14))  # offset, scale, rotation, colour DC, opacity
INIT = ((3, 0.001, 0.001), (3, 0.00003, -7.0), (4, 1.0, 0.0), (3, 1.0, 0.0), (1, 1.0, -2.0))
OFFSET_SHIFT = 6.0


def rgb_to_sh(rgb):
    return (rgb - 0.5) / SH_C0


def sh_to_rgb(sh):
    return sh * SH_C0 + 0.5


def reg_dense_offsets(xyz, shift=OFFSET_SHIFT):
    """Offsets of Splatt3R: unit direction times exp(|o| - shift) - exp(-shift), zero at o = 0."""
    d = xyz.norm(dim=-1, keepdim=True)
    return xyz / d.clip(min=1e-8) * (torch.exp(d - shift) - torch.exp(torch.zeros_like(d) - shift))


class Splatt3RHead(nn.Module):
    def __init__(self, net):
        super().__init__()
        l2 = net.dec_depth
        ed, dd = net.enc_embed_dim, net.dec_embed_dim
        self.gaussian_dpt = PixelwiseTaskWithDPT(
            num_channels=N_SPLATT3R, feature_dim=256, last_dim=128,
            hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2], dim_tokens=[ed, dd, dd, dd],
            postprocess=None, depth_mode=net.depth_mode, conf_mode=net.conf_mode, head_type="regression")
        self.init_last_layer()

    def last_conv(self):
        convs = [m for m in self.gaussian_dpt.dpt.head.modules() if isinstance(m, nn.Conv2d) and m.out_channels == N_SPLATT3R]
        if not convs:
            raise RuntimeError("no last convolution with 14 outputs in the Gaussian DPT")
        return convs[-1]

    @torch.no_grad()
    def init_last_layer(self):
        conv = self.last_conv()
        start = 0
        for n, gain, bias in INIT:
            nn.init.xavier_uniform_(conv.weight[start:start + n], gain=gain)
            nn.init.constant_(conv.bias[start:start + n], bias)
            start += n

    def forward(self, dec_tokens, hw):
        """dec_tokens: the decoder outputs of the reference view (list, encoder output first). Returns (B, H, W, 14)."""
        out = self.gaussian_dpt([t.float() for t in dec_tokens], (int(hw[0]), int(hw[1])))
        return out.permute(0, 2, 3, 1)

    def forward_features(self, dec_tokens, hw):
        """forward() and the 128 channel features before the last convolution, (B, H, W, 14) and (B, H, W, 128)."""
        store = {}
        hook = self.gaussian_dpt.dpt.head[-2].register_forward_hook(lambda m, i, o: store.__setitem__("f", o))
        try:
            out = self.forward(dec_tokens, hw)
        finally:
            hook.remove()
        return out, store["f"].permute(0, 2, 3, 1)

    @staticmethod
    def activate(raw):
        """Splatt3R activations of (..., 14): offset, scales, quaternion w x y z (for our rasteriser), colour DC
        residual (SH units), opacity."""
        o, s, r, c, a = (raw[..., i:j] for i, j in SLICES)
        rot = r / (r.norm(dim=-1, keepdim=True) + 1e-8)  # x y z w
        return dict(offset=reg_dense_offsets(o), scales=s.exp(), quat=rot[..., [3, 0, 1, 2]], sh_dc=c,
                    opacity=a[..., 0].sigmoid())

    @staticmethod
    def colour(sh_dc, base_rgb):
        """Base colour (the restored colour J) plus the DC residual, in RGB, clamped below at 0 (Splatt3R adds the
        residual to the input image colour in SH units)."""
        return sh_to_rgb(rgb_to_sh(base_rgb) + sh_dc).clamp_min(0.0)


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__["_state"] = state

    def __call__(self, *a, **k):
        return self


class _StubFinder:
    """Import hook that hands out empty modules (every attribute a _Stub) for the packages in names, so that a
    Lightning checkpoint loads without its training packages (omegaconf and so on)."""

    def __init__(self, names):
        self.names = set(names)

    def find_spec(self, fullname, path=None, target=None):
        import importlib.util
        if fullname.split(".")[0] in self.names:
            return importlib.util.spec_from_loader(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__path__ = []
        m.__getattr__ = lambda name: _Stub
        return m

    def exec_module(self, module):
        pass


def _load_lightning(path):
    import sys
    finder = _StubFinder(())
    sys.meta_path.insert(0, finder)
    try:
        for _ in range(20):
            try:
                return torch.load(path, map_location="cpu", weights_only=False)
            except ModuleNotFoundError as e:
                top = (e.name or "").split(".")[0]
                if not top or top in finder.names:
                    raise
                finder.names.add(top)
        raise RuntimeError(f"could not load {path}")
    finally:
        sys.meta_path.remove(finder)
        for k in [k for k in sys.modules if k.split(".")[0] in finder.names]:
            del sys.modules[k]


def splatt3r_head_state(path, view=1):
    """State dict of the Gaussian head of one view from the public Splatt3R checkpoint (epoch=19-step=1200.ckpt)."""
    ck = _load_lightning(path)
    sd = ck.get("state_dict", ck)
    prefix = f"encoder.downstream_head{view}.gaussian_dpt."
    out = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not out:
        raise KeyError(f"no '{prefix}' keys in {path}; first keys {list(sd)[:5]}")
    return out


def load_splatt3r(head, path, view=1):
    """Copies the trained Gaussian head of Splatt3R into head (strict). Returns the number of tensors."""
    state = splatt3r_head_state(path, view)
    head.gaussian_dpt.load_state_dict(state, strict=True)
    return len(state)
