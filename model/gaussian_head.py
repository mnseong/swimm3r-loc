"""Gaussian map head of Swimm3R.

One DPT head on the reference stream of a pair pass. It reads the token layers the pointmap
head reads, and for every pixel of a frame it predicts the attributes of one Gaussian in the
camera frame of that frame. The centre is not predicted. It is the depth of our own geometry
along the pixel ray, so a frame can be placed in any map with the pose and depth of the SfM.

The DPT architecture is the Gaussian parameter head of SPFSplat, taken as code only from
spfsplat/src/model/encoder/heads. Its trunk starts from the weights of our pointmap head and
the last layer starts at zero, so the first output is a plain splat of the input pixels.

Output channels: opacity 1, log scale 3, rotation 4 (w x y z), colour offset of the observed
(underwater) appearance 3, colour offset of the restored appearance 3. A head with N_OUT_4D
channels adds the Beta shape 1, the dynamic gate 1, the temporal width 1, the temporal shape 1
and a velocity 3, read by training/primitives4d.py. At a zero last layer they give beta 4 and
a static map.
"""

import importlib
import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

_HEADS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "spfsplat", "src", "model", "encoder", "heads",
)
_PKG = "_swimm3r_spf_heads"

N_OUT = 14
OPACITY_BIAS = 2.0
SCALE_BASE = 0.6
LOG_SCALE_RANGE = 4.0

N_OUT_4D = 21
BETA_BASE, BETA_RANGE = 4.0, 2.0
GATE_P0 = 0.04742587317756678      # sigmoid(-3)
TW_BASE, TW_MIN, TW_MAX = 1.5, -1.5, 3.0
TB_RANGE = 2.0
VEL_UNIT, VEL_RANGE = 10.0, 20.0


def _spf_gs_head_module():
    """Loads the head files alone, without importing the rest of the SPFSplat package."""
    if _PKG not in sys.modules:
        if not os.path.isdir(_HEADS_DIR):
            raise FileNotFoundError(f"Gaussian head code not found: {_HEADS_DIR}")
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [_HEADS_DIR]
        sys.modules[_PKG] = pkg
    return importlib.import_module(_PKG + ".dpt_gs_head")


def head_out_channels(state):
    """Output channels (14 or 21) of a head state dict, from the bias of its last layer."""
    n = None
    for k, v in state.items():
        if k.startswith("gs.dpt.head.") and v.dim() == 1 and v.shape[0] in (N_OUT, N_OUT_4D):
            n = int(v.shape[0])
    if n is None:
        raise ValueError("no output layer of 14 or 21 channels under gs.dpt.head.")
    return n


def expand_head_state(state, ref_state):
    """State with the layout of ref_state. Tensors of 14 rows where ref_state has 21 get zero rows 14 to 20.

    Returns the new state and the list of expanded keys. Any other difference raises.
    """
    missing = [k for k in ref_state if k not in state]
    extra = [k for k in state if k not in ref_state]
    if missing or extra:
        raise KeyError(f"head state does not match: missing {missing[:5]}, unexpected {extra[:5]}")
    out, expanded = {}, []
    for k, ref in ref_state.items():
        v = state[k]
        if v.shape == ref.shape:
            out[k] = v.clone()
        elif (v.dim() == ref.dim() and v.dim() >= 1 and v.shape[0] == N_OUT and ref.shape[0] == N_OUT_4D
              and v.shape[1:] == ref.shape[1:]):
            new = v.new_zeros(ref.shape)
            new[:N_OUT] = v
            out[k] = new
            expanded.append(k)
        else:
            raise ValueError(f"cannot load {k}: {tuple(v.shape)} into {tuple(ref.shape)}")
    return out, expanded


class GaussianMapHead(nn.Module):
    def __init__(self, net, n_out=N_OUT):
        super().__init__()
        mod = _spf_gs_head_module()
        self.gs = mod.create_gs_dpt_head(net, has_conf=False, out_nchan=n_out, postprocess_func=None)
        head = self.gs.dpt.head
        for name, m in list(head.named_children()):
            if isinstance(m, nn.Dropout):
                setattr(head, name, nn.Identity())
        last = [m for m in head.modules() if isinstance(m, nn.Conv2d)][-1]
        assert last.out_channels == n_out
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def load_base_state(self, state):
        """Loads a head state of 14 or 21 channels. Rows the state lacks start at zero. Returns the expanded keys."""
        full, expanded = expand_head_state(state, self.state_dict())
        self.load_state_dict(full)
        return expanded

    @torch.no_grad()
    def init_from_pointmap_head(self, net):
        """Copies the DPT trunk of the pointmap head of `net`. Returns (copied, total) tensors."""
        src = net.downstream_head1.dpt.state_dict()
        dst = self.gs.dpt.state_dict()
        copied = 0
        for k, v in dst.items():
            if k.startswith("head.") or k.startswith("input_merger."):
                continue
            if k in src and src[k].shape == v.shape:
                v.copy_(src[k])
                copied += 1
        return copied, len(dst)

    def forward(self, tokens, image, hw):
        """tokens: list of (B, S, C) of one decoder stream. image: (B, 3, H, W) in [-1, 1]."""
        return self.gs(tokens, image, hw)

    @staticmethod
    def activate(raw, image01):
        """raw (B, 14 or 21, H, W), image01 (B, 3, H, W) in [0, 1]. Everything comes back as (B, H, W, C).

        With 21 channels it adds beta, gate, tw_rel, tb (B, H, W) and vel (B, H, W, 3).
        """
        raw = raw.permute(0, 2, 3, 1)
        rgb = image01.permute(0, 2, 3, 1)
        ident = raw.new_tensor([1.0, 0.0, 0.0, 0.0])
        out = {
            "opacity": torch.sigmoid(raw[..., 0] + OPACITY_BIAS),
            "log_scale": raw[..., 1:4].clamp(-LOG_SCALE_RANGE, LOG_SCALE_RANGE),
            "quat": F.normalize(raw[..., 4:8] + ident, dim=-1),
            "color_raw": rgb + raw[..., 8:11],
            "color_clean": rgb + raw[..., 11:14],
        }
        if raw.shape[-1] == N_OUT:
            return out
        if raw.shape[-1] != N_OUT_4D:
            raise ValueError(f"head output has {raw.shape[-1]} channels, expected {N_OUT} or {N_OUT_4D}")
        # (sigmoid(raw - 3) - sigmoid(-3)) / (1 - sigmoid(-3)), exactly 0 at raw 0
        u = torch.expm1(raw[..., 15].clamp(-20.0, 20.0))
        out["beta"] = BETA_BASE * torch.exp(raw[..., 14].clamp(-BETA_RANGE, BETA_RANGE))
        out["gate"] = GATE_P0 * u / (1.0 + GATE_P0 * u)
        out["tw_rel"] = TW_BASE * torch.exp(raw[..., 16].clamp(TW_MIN, TW_MAX))
        out["tb"] = raw[..., 17].clamp(-TB_RANGE, TB_RANGE)
        out["vel"] = VEL_UNIT * raw[..., 18:21].clamp(-VEL_RANGE, VEL_RANGE)
        return out
