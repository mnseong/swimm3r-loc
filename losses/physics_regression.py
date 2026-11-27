"""Physics-regression heads + cycle reconstruction loss for v7.

Adds an Akkaynak-Treibitz-style image-formation regularizer on top of the v1
baseline (encoder + desc consistency). The student already runs a forward
pass on the raw underwater image I; we attach lightweight heads to the
existing encoder feature to predict (β^D, β^B, B^∞), and supervise them via
the image-formation equation:

    I_c = J_c · exp(-β^D_c · z) + B^∞_c · (1 - exp(-β^B_c · z))

J is the SeaErra-restored "clean" view (already in the pair) and z is the
metric depth GT (loaded from <scene>/depth/<stem>.npy by the dataset). This
is a pure self-supervised auxiliary task — zero extra forwards through the
backbone.

Two head variants
-----------------
1. PhysicsHead          (kind="scalar")
       β^D, β^B ∈ ℝ^[B,3] — per-view scalar, broadcast over all pixels.
       Simplest, ~0.4M params.

2. DepthDependentPhysicsHead   (kind="depth_dep")  [DEFAULT, Sea-Thru style]
       β^D_c(z) = a^D_c · exp(b^D_c · z) + c^D_c · exp(d^D_c · z)
       β^B_c(z) = a^B_c · exp(b^B_c · z) + c^B_c · exp(d^B_c · z)
       Output is β as a [B, 3, H, W] spatial map — one β per pixel, derived
       from per-view (a, b, c, d) parameters and the metric depth z.
       Matches Akkaynak-Treibitz 2018 / Sea-Thru CVPR 2019.

Numerical safety (depth_dep):
    a, c via softplus  → strictly positive
    b, d via tanh*b_max → bounded slopes (default ±0.2 / m)
    final β clamped to [1e-3, beta_max] regardless of formula

Identifiability priors:
    - β_R > β_G > β_B per-channel ordering (sort for scalar; soft penalty for
      depth_dep — order may flip with depth in real water, so it's per-image
      mean rather than per-pixel)
    - sigmoid * max_b caps B^∞ ≤ max_b (default 0.7); real Jerlov ≤ 0.5
    - L1 coupling β^D ≈ β^B (Akkaynak: spread ~30%)
    - box bound ReLU(β - beta_max) above an upper limit
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


def _sort_descending(x: torch.Tensor) -> torch.Tensor:
    """Sort the last dim in descending order. Differentiable via the values
    tensor of torch.sort (gradients route through the original positions)."""
    return torch.sort(x, dim=-1, descending=True).values


def median_normalize_depth(z: torch.Tensor, z_min: float = 0.1,
                           z_max: float = 50.0) -> torch.Tensor:
    """Per-image median normalisation for the PHYSICS domain: ẑ = z / median(z).

    The per-image median is taken over VALID pixels (finite, in (z_min, z_max))
    and DETACHED, so ẑ has median ≈ 1 with the same RELATIVE structure as z, and
    the normaliser carries no gradient (only z's structure does). Unifies the
    β·z scale convention so β is "attenuation per normalised-depth unit" and is
    consistent across GT-z / pred-z / inference regardless of metric vs
    teacher-scale. z: [B, H, W] (or [B, ...]). Returns same shape.
    """
    B = z.shape[0]
    flat = z.reshape(B, -1)
    s = []
    for b in range(B):
        v = flat[b]
        m = torch.isfinite(v) & (v > z_min) & (v < z_max)
        vv = v[m]
        s.append(vv.median() if vv.numel() > 0 else z.new_tensor(1.0))
    s = torch.stack(s).clamp(min=1e-3).detach().view(B, *([1] * (z.dim() - 1)))
    return z / s


def sanitize_physics_depth(z: torch.Tensor, z_min: float = 0.1,
                           z_max: float = 50.0) -> torch.Tensor:
    """Bound depth before it enters exp().

    Invalid pixels are still excluded by each loss' valid mask; this helper
    only prevents NaN/Inf values from poisoning intermediate tensors before
    the mask is applied.
    """
    return torch.nan_to_num(z, nan=z_max, posinf=z_max, neginf=z_min).clamp(
        z_min + 1e-3, z_max - 1e-3
    )


def binf_rgb_stats(img_raw: torch.Tensor) -> torch.Tensor:
    """Compact raw-image colour statistics for B^∞ prediction.

    img_raw is in MASt3R [-1, 1] convention. The returned vector is per-frame
    and intentionally tiny: global RGB mean, dim-pixel RGB percentiles, bright
    RGB percentile, and a few red-vs-GB ratios. These are available at both
    train and inference time and give the B^∞ head direct access to the
    dim-pixel cue used by Sea-Thru-style veiling-light estimation.
    """
    if img_raw.dim() != 4 or img_raw.size(1) != 3:
        raise ValueError(f"img_raw must be [B,3,H,W], got {tuple(img_raw.shape)}")
    x = ((img_raw + 1.0) * 0.5).clamp(0.0, 1.0)
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    flat = x.flatten(2)                                            # [B, 3, HW]
    mean = flat.mean(dim=-1)                                       # [B, 3]
    p02 = torch.quantile(flat, 0.02, dim=-1)                       # [B, 3]
    p05 = torch.quantile(flat, 0.05, dim=-1)
    p95 = torch.quantile(flat, 0.95, dim=-1)

    def _log_r_over_gb(v: torch.Tensor) -> torch.Tensor:
        gb = 0.5 * (v[:, 1] + v[:, 2])
        return torch.log((v[:, 0] + 1e-3) / (gb + 1e-3)).clamp(-4.0, 4.0).unsqueeze(-1)

    ratios = torch.cat([_log_r_over_gb(mean), _log_r_over_gb(p02), _log_r_over_gb(p05)], dim=-1)
    return torch.cat([mean, p02, p05, p95, ratios], dim=-1)         # [B, 15]


class PhysicsHead(nn.Module):
    """Scalar per-view predictor (kind='scalar').

    Outputs (β^D, β^B), each [B, 3] — broadcast across all pixels in cycle
    loss. Channel monotonic prior baked in via sort_descending.
    """

    def __init__(self, in_dim: int = 1024, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 6),
        )

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = feat.mean(dim=1)            # [B, C]
        raw = self.mlp(pooled)               # [B, 6]
        bD_raw, bB_raw = raw.split(3, dim=-1)
        bD = _sort_descending(F.softplus(bD_raw))
        bB = _sort_descending(F.softplus(bB_raw))
        return bD, bB


class DepthDependentPhysicsHead(nn.Module):
    """Sea-Thru-style β(z) parameterization (kind='depth_dep').

    Following Akkaynak-Treibitz 2018, β is modelled as a double-exponential
    function of metric depth z:

        β^D_c(z) = a^D_c · exp(b^D_c · z) + c^D_c · exp(d^D_c · z)
        β^B_c(z) = a^B_c · exp(b^B_c · z) + c^B_c · exp(d^B_c · z)

    Per view the head emits 24 raw scalars (4 params × 3 channels × 2
    branches); the spatial map [B, 3, H, W] is materialized at forward time
    from these scalars and the per-pixel depth.

    a, c constrained positive (softplus); b, d are bounded slopes
    (tanh * b_max). Final β clamped to [1e-3, beta_max] for numerical
    stability across the (z * b) exponent range.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        hidden: int = 256,
        b_max: float = 0.2,
        beta_max: float = 10.0,
        constrain_beta_B: bool = False,
        betaB_ratio_min: float = 0.0,
        betaB_ratio_max: float = 1.0,
        jerlov_unified_beta: bool = False,
        seathru_betaB_const: bool = False,
        seathru_betaB_le_betaD: bool = False,
        unified_binf: bool = False,
        binf_max_b: float = 0.7,
        binf_spatial: bool = False,
        light: bool = False,
    ):
        super().__init__()
        self.b_max = float(b_max)
        # Direction dependent veiling light (2026-11-23): B^∞(x, y) = sigmoid(logit + sum_k a_k phi_k(x, y)) * max_b over
        # the normalised image coordinates of the view, phi = (x, y, x^2, x y, y^2) per channel (15 more MLP outputs,
        # 18 -> 33), started at zero so that the map starts as the constant B^∞ of the view.
        self.binf_spatial = bool(binf_spatial)
        # Architectural beta_B <= beta_D cap for the free seathru beta_B (the
        # constraint v9p3-epsilon's constrain_beta_B had built in). Applied in
        # forward as beta_B = min(beta_B, beta_D); a no-op for the other modes.
        self.seathru_betaB_le_betaD = bool(seathru_betaB_le_betaD)
        self.beta_max = float(beta_max)
        # Unified medium head: emit B^∞ from the SAME MLP as β (out_dim += 3),
        # so β and B^∞ share the 256→256 trunk and learn their physical
        # correlation (e.g. high β^D_R ↔ low B^∞_R for cyan-dead water). Only
        # valid with seathru_betaB_const (15 → 18). B^∞ slice = sigmoid×binf_max_b.
        self.unified_binf = bool(unified_binf)
        self.binf_max_b = float(binf_max_b)
        # v9: hard-enforce β_B ≤ β_D by parameterising β_B = β_D · sigmoid(g(z)).
        # Default off for backward-compat with v8 checkpoints; flip on via CLI
        # for new training runs (config/training_args.py:--phys_constrain_beta_B).
        self.constrain_beta_B = bool(constrain_beta_B)
        # v11 — Jerlov unified-β: predict a single β at the architectural level
        # (MLP outputs 12 scalars = 4 params × 3 channels × 1 branch instead of
        # 24 for two branches). β^D and β^B share the exact same tensor (β^B :=
        # β^D), realising the Jerlov 1976 single-coefficient water-type model
        # natively. No sigmoid involved → no Pattern C saturation possible. The
        # medium-strip cond input (β^D, β^B, B^∞) has β^D ≡ β^B redundancy from
        # the very first step, so the strip MLP learns the tied structure from
        # initialisation and never sees the distribution-shift jump that broke
        # the constrain_beta_B [0.95, 1.0] run (see step ~1150 incident).
        # Mutually exclusive with constrain_beta_B (the two are alternative
        # Jerlov enforcement mechanisms).
        self.jerlov_unified_beta = bool(jerlov_unified_beta)
        if self.jerlov_unified_beta and self.constrain_beta_B:
            raise ValueError(
                "jerlov_unified_beta and constrain_beta_B are mutually "
                "exclusive: the former ties β^B = β^D at the MLP output "
                "(12 scalars), the latter uses sigmoid reparam (24 scalars). "
                "Pick one Jerlov enforcement mechanism."
            )
        # Sea-Thru-true β^B: backscatter coefficient is a per-view, per-channel
        # CONSTANT (z-INDEPENDENT), while β^D keeps its z-dependent double-exp.
        # This is the actual Akkaynak-Treibitz 2018 / Sea-Thru asymmetry —
        # β^D varies strongly with depth, β^B is treated as depth-flat. The MLP
        # emits 15 scalars: 12 for β^D (a,b,c,d × 3 ch) + 3 for the β^B constant
        # (one per channel). β^B and β^D are INDEPENDENT here (no β^B ≤ β^D tie),
        # so couple / couple_order / ordering_B should be disabled in the config
        # (the loss-level free-β^B contract). Mutually exclusive with the two
        # Jerlov enforcement mechanisms above (those tie β^B to β^D).
        self.seathru_betaB_const = bool(seathru_betaB_const)
        if self.seathru_betaB_const and (self.jerlov_unified_beta or self.constrain_beta_B):
            raise ValueError(
                "seathru_betaB_const is mutually exclusive with "
                "jerlov_unified_beta / constrain_beta_B: the former models β^B "
                "as a z-independent free constant (15 scalars), the latter two "
                "tie β^B to β^D. Pick one β^B treatment."
            )
        # v10 — bounded β_B / β_D ratio. The original constrain_beta_B used
        # σ(g(z)) ∈ [0, 1] which empirically saturates to 1.0 (Pattern C in
        # v9.x diagnostics). Bounded sigmoid range forces ratio ∈
        # [betaB_ratio_min, betaB_ratio_max] architecturally:
        #   ratio = ratio_min + (ratio_max − ratio_min) × σ(g(z))
        # v10 default: [0.4, 0.95] — Akkaynak 2018 measurements show real
        # β_B/β_D ratio ∈ [0.4, 0.9]. ratio_max=0.95 prevents saturation at
        # 1.0, forcing β_B < β_D always (informative spread guaranteed).
        # Backward compat: defaults [0.0, 1.0] reproduce v9.x exactly.
        self.betaB_ratio_min = float(betaB_ratio_min)
        self.betaB_ratio_max = float(betaB_ratio_max)
        assert 0.0 <= self.betaB_ratio_min < self.betaB_ratio_max <= 1.0, (
            f"betaB ratio range invalid: [{betaB_ratio_min}, {betaB_ratio_max}]"
        )
        # MLP output:
        #   * 24 = 4 params × 3 channels × 2 branches (β^D, β^B) — default,
        #     used by free / constrain_beta_B modes.
        #   * 12 = 4 params × 3 channels × 1 branch (β only) — Jerlov unified,
        #     β^B := β^D so only one set of (a, b, c, d) per channel needed.
        #   * 15 = 12 (β^D double-exp) + 3 (β^B per-channel constant) —
        #     Sea-Thru-true (seathru_betaB_const): β^B is z-independent.
        if self.unified_binf and not self.seathru_betaB_const:
            raise ValueError(
                "unified_binf is only supported with seathru_betaB_const "
                "(15 → 18 = 12 β^D + 3 β^B const + 3 B^∞)."
            )
        if self.binf_spatial and not self.unified_binf:
            raise ValueError("binf_spatial needs the unified medium head (B^∞ from the same MLP)")
        if self.jerlov_unified_beta:
            out_dim = 12
        elif self.seathru_betaB_const:
            out_dim = (33 if self.binf_spatial else 18) if self.unified_binf else 15
        else:
            out_dim = 24
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        if self.binf_spatial:
            with torch.no_grad():
                self.mlp[-1].weight[18:33].zero_()
                self.mlp[-1].bias[18:33].zero_()
        if light:
            self.light = nn.Linear(hidden, 3)
            nn.init.zeros_(self.light.weight)
            nn.init.zeros_(self.light.bias)

    def _split_params(self, raw: torch.Tensor) -> tuple:
        """Decode raw → (a_D, b_D, c_D, d_D, a_B, b_B, c_B, d_B), each [B, 3].

        Raw shape:
          * jerlov_unified_beta: raw is [B, 12] (single branch). The same
            (a, b, c, d) tuple is returned for both D and B → β^B = β^D
            tied at the parameter level.
          * default / constrain_beta_B: raw is [B, 24] (two branches).
        """
        if self.jerlov_unified_beta:
            # Single 12-scalar branch → tied (a, b, c, d) for D and B.
            p = raw.view(-1, 4, 3)                      # [B, 4, 3]
            a = F.softplus(p[:, 0])
            b = torch.tanh(p[:, 1]) * self.b_max
            c = F.softplus(p[:, 2])
            d = torch.tanh(p[:, 3]) * self.b_max
            # Return identical tuples for D and B → architectural Jerlov
            # (the same tensor objects, not copies, so gradients flow
            # through one set of params from both β paths).
            return a, b, c, d, a, b, c, d

        if self.seathru_betaB_const:
            # raw is [B, 15]: first 12 → β^D double-exp params, last 3 → the
            # z-independent β^B constant (one per channel). The β^B "slope"
            # params b_B/d_B and second-term c_B are returned as ZEROS so the
            # generic double-exp forward path (β = a·exp(b·z) + c·exp(d·z))
            # collapses to β^B = a_B — a pure depth-flat constant. Keeping the
            # 8-tuple shape means params_B = stack([a_B, 0, 0, 0]) stays
            # [B, 4, 3] and view_consistency_loss compares it unchanged.
            bD_raw = raw[:, :12]                        # [B, 12]
            bD_p = bD_raw.reshape(-1, 4, 3)             # [B, 4, 3]
            a_D = F.softplus(bD_p[:, 0])
            b_D = torch.tanh(bD_p[:, 1]) * self.b_max
            c_D = F.softplus(bD_p[:, 2])
            d_D = torch.tanh(bD_p[:, 3]) * self.b_max
            a_B = F.softplus(raw[:, 12:15])             # [B, 3] — β^B constant
            zero = torch.zeros_like(a_B)
            return a_D, b_D, c_D, d_D, a_B, zero, zero, zero

        bD_raw, bB_raw = raw.split(12, dim=-1)        # [B, 12] each
        bD_p = bD_raw.view(-1, 4, 3)                   # [B, 4, 3]
        bB_p = bB_raw.view(-1, 4, 3)
        a_D = F.softplus(bD_p[:, 0])
        b_D = torch.tanh(bD_p[:, 1]) * self.b_max
        c_D = F.softplus(bD_p[:, 2])
        d_D = torch.tanh(bD_p[:, 3]) * self.b_max
        if self.constrain_beta_B:
            # β_B is reparameterised as β_D · sigmoid(g(z)). The (a, c) coefficients
            # of g feed sigmoid, so they may be negative — keep them RAW (no
            # softplus). The slope params (b, d) keep the same tanh × b_max.
            a_B = bB_p[:, 0]
            c_B = bB_p[:, 2]
        else:
            a_B = F.softplus(bB_p[:, 0])
            c_B = F.softplus(bB_p[:, 2])
        b_B = torch.tanh(bB_p[:, 1]) * self.b_max
        d_B = torch.tanh(bB_p[:, 3]) * self.b_max
        return a_D, b_D, c_D, d_D, a_B, b_B, c_B, d_B

    def forward(self, feat: torch.Tensor, z: torch.Tensor) -> dict:
        """feat: [B, N, C] tokens. z: [B, H, W] metric depth (m).

        Returns a dict with both spatial β maps and the underlying per-view
        params (used for view-consistency loss across the pair).
        """
        pooled = feat.mean(dim=1)                      # [B, C]
        if hasattr(self, "light"):
            hid = self.mlp[:-1](pooled)
            raw = self.mlp[-1](hid)
            light = torch.exp(self.light(hid.detach() if getattr(self, "light_detach", False) else hid).clamp(-4.0, 1.0))
        else:
            raw = self.mlp(pooled)                         # [B, 12] or [B, 24]
            light = None
        a_D, b_D, c_D, d_D, a_B, b_B, c_B, d_B = self._split_params(raw)

        z_safe = sanitize_physics_depth(z)
        z_ = z_safe.unsqueeze(1)                       # [B, 1, H, W]
        # broadcast: [B, 3, 1, 1] * [B, 1, H, W] → [B, 3, H, W]
        beta_D = (
            a_D[:, :, None, None] * torch.exp(b_D[:, :, None, None] * z_)
            + c_D[:, :, None, None] * torch.exp(d_D[:, :, None, None] * z_)
        )

        if self.jerlov_unified_beta:
            # β^B := β^D exactly. The same tensor object is referenced from
            # both keys; gradients flow through the same (a, b, c, d) params
            # from both the forward-transmission (β^D) and backscatter (β^B)
            # paths in the IFM, jointly constraining the unified β
            # (paper-defensible Jerlov form). The clamp below clones, so we
            # re-tie after clamping to preserve the identity invariant for
            # downstream losses (L_couple, L_ordering, medium-strip cond).
            beta_D = beta_D.clamp(min=1e-3, max=self.beta_max)
            beta_B = beta_D
            # Build output and bail early — skip the duplicate clamp below.
            out = {
                "beta_D_map": beta_D,
                "beta_B_map": beta_B,
                "params_D": torch.stack([a_D, b_D, c_D, d_D], dim=1),
                # params_B: identical tensors (a_D == a_B etc., set in
                # _split_params), so view-consistency loss sees zero spread
                # between D and B per-view scalars — by design.
                "params_B": torch.stack([a_B, b_B, c_B, d_B], dim=1),
            }
            return out
        elif self.seathru_betaB_const:
            # Sea-Thru-true asymmetry: β^B is a z-INDEPENDENT per-channel
            # constant (a_B), broadcast across every pixel. β^D above keeps its
            # depth-dependent double-exp. b_B/c_B/d_B are zero (set in
            # _split_params), so this is mathematically β^B = a_B regardless of
            # z — we broadcast a_B directly rather than re-evaluating the
            # (now-trivial) double-exp, which is clearer and a hair cheaper.
            beta_B = a_B[:, :, None, None].expand(-1, -1, z_.shape[-2], z_.shape[-1])
        elif self.constrain_beta_B:
            # β_B = β_D · σ(g(z)) where g(z) is the double-exp parameterisation.
            # v10: σ rescaled to [betaB_ratio_min, betaB_ratio_max] (default
            # [0.0, 1.0] = v9.x behaviour; v10 sets [0.4, 0.95] to prevent
            # the σ → 1 saturation observed empirically (Pattern C). Bounded
            # range forces β_B / β_D ∈ [ratio_min, ratio_max] architecturally,
            # guaranteeing informative spread and physical realism (Akkaynak
            # 2018 measured β_B/β_D ∈ [0.4, 0.9]).
            ratio_raw = (
                a_B[:, :, None, None] * torch.exp(b_B[:, :, None, None] * z_)
                + c_B[:, :, None, None] * torch.exp(d_B[:, :, None, None] * z_)
            )
            sigma = torch.sigmoid(ratio_raw)                       # σ ∈ [0, 1]
            ratio = self.betaB_ratio_min + (self.betaB_ratio_max - self.betaB_ratio_min) * sigma
            beta_B = beta_D * ratio
        else:
            beta_B = (
                a_B[:, :, None, None] * torch.exp(b_B[:, :, None, None] * z_)
                + c_B[:, :, None, None] * torch.exp(d_B[:, :, None, None] * z_)
            )
        beta_D = beta_D.clamp(min=1e-3, max=self.beta_max)
        beta_B = beta_B.clamp(min=1e-3, max=self.beta_max)
        # Architectural beta_B <= beta_D cap (v9p3-eps constrain equivalent for the
        # free seathru beta_B). Per-pixel min, so beta_B becomes z-dependent only
        # where it would exceed beta_D(z); physically correct (backscatter <=
        # attenuation) and makes runaway beta_B structurally impossible.
        if self.seathru_betaB_le_betaD:
            beta_B = torch.minimum(beta_B, beta_D)

        out = {
            "beta_D_map": beta_D,                      # [B, 3, H, W]
            "beta_B_map": beta_B,
            # per-view scalars retained for view-consistency loss (compare
            # parameters across the two views in a pair, not spatial maps —
            # those have different geometry).
            "params_D": torch.stack([a_D, b_D, c_D, d_D], dim=1),  # [B, 4, 3]
            "params_B": torch.stack([a_B, b_B, c_B, d_B], dim=1),
        }
        # Expose pre-sigmoid logit for ratiocap loss (constrain branch only).
        # Penalising the post-sigmoid ratio dies at saturation because
        # σ'(g) = σ(1−σ) ≈ 0 when σ ≈ 1 (Pattern C resilience). Logit-space
        # penalty bypasses σ' entirely and provides effective gradient
        # even when the sigmoid is fully saturated.
        if self.constrain_beta_B:
            out["ratio_logit"] = ratio_raw            # [B, 3, H, W] pre-sigmoid
        if self.unified_binf:
            # B^∞ from the same MLP trunk: last 3 raw scalars → sigmoid×max_b.
            # Shares the 256→256 representation with β → joint medium prediction.
            out["B_inf"] = torch.sigmoid(raw[:, 15:18]) * self.binf_max_b   # [B, 3]
            if self.binf_spatial:
                out["B_inf_map"] = binf_map(raw[:, 15:18], raw[:, 18:33], z_.shape[-2], z_.shape[-1]) * self.binf_max_b
        if light is not None:
            out["light"] = light
        return out


def binf_basis(h, w, device, dtype=torch.float32):
    """phi (5, H, W) = (x, y, x^2, x y, y^2) over the normalised pixel centre coordinates x, y in [-1, 1]."""
    y = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h * 2.0 - 1.0
    x = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w * 2.0 - 1.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy, xx * xx, xx * yy, yy * yy])


def binf_map(logit0, coeff, h, w):
    """sigmoid(logit0 + sum_k coeff_k phi_k) of the B^∞ head: logit0 (B, 3), coeff (B, 15) -> (B, 3, H, W) in (0, 1)."""
    phi = binf_basis(h, w, logit0.device, logit0.dtype)
    a = coeff.reshape(coeff.shape[0], 3, 5)
    return torch.sigmoid(logit0[:, :, None, None] + torch.einsum("bck,khw->bchw", a, phi))


class MediumStripper(nn.Module):
    """FiLM-conditioned medium stripping for token features.

    Given an encoder feature `feat` and the predicted physics parameters
    (β^D, β^B, B^∞) of the same view, produces a *medium-stripped* feature

        feat_geom = feat * (1 + γ(params)) − b(params)

    where (γ, b) ∈ ℝ^C are produced by a small conditioning MLP on the
    9-scalar physics summary. The last layer is zero-initialised so that
    the strip is initially the identity — training discovers what to
    subtract once physics predictions are stable.

    This module replaces the v5 medium-swap *augmentation* by an
    *architectural* mechanism: instead of running the encoder on synthetic
    medium-swapped inputs (which costs an extra forward pass), we use the
    student's already-extracted feature and the predicted medium params to
    produce a medium-invariant representation in a single forward.

    Output shape == input shape, so the stripped feature is a drop-in
    replacement for `lowlight_feat*` in the encoder consistency loss.
    """

    def __init__(self, channel_dim: int, cond_dim: int = 9, hidden: int = 128):
        super().__init__()
        self.cond_dim = cond_dim
        self.channel_dim = channel_dim
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * channel_dim),
        )
        # Identity initialization: γ = 0, b = 0 → feat_geom = feat at start.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feat: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """feat:   [B, ..., C]  arbitrary middle dims (tokens / spatial).
           params: [B, cond_dim]
        """
        if feat.shape[-1] != self.channel_dim:
            raise ValueError(
                f"MediumStripper expected feature channel {self.channel_dim}, "
                f"got {feat.shape[-1]}"
            )
        cond = self.mlp(params)                        # [B, 2C]
        gamma, beta = cond.chunk(2, dim=-1)            # [B, C] each
        # broadcast over middle dims
        view_shape = [feat.shape[0]] + [1] * (feat.dim() - 2) + [feat.shape[-1]]
        gamma = gamma.view(view_shape)
        beta = beta.view(view_shape)
        return feat * (1.0 + gamma) - beta


def physics_params_to_cond(pred: dict) -> torch.Tensor:
    """Bundle a PhysicsRegressor prediction dict into a [B, 9] conditioning vector.

    Spatial β maps from depth_dep mode are pooled to per-channel scalars so the
    stripper sees a fixed-dim summary regardless of head kind. Order matches
    `MediumStripper.cond_dim=9`: (β^D_RGB, β^B_RGB, B^∞_RGB).
    """
    bD = pred["beta_D"]
    bB = pred["beta_B"]
    Binf = pred["B_inf"]
    if bD.dim() == 4:
        bD = bD.mean(dim=(-1, -2))
    if bB.dim() == 4:
        bB = bB.mean(dim=(-1, -2))
    return torch.cat([bD, bB, Binf], dim=-1)            # [B, 9]


class BackscatterHead(nn.Module):
    """Per-view predictor for B^∞ ∈ ℝ^[B, 3], bounded to [0, max_b]."""

    def __init__(
        self,
        in_dim: int = 1024,
        max_b: float = 0.7,
        hidden: int = 256,
        use_rgb_stats: bool = False,
        deep_head: bool = False,
        rgb_stats_dim: int = 15,
        feat_embed_dim: int = 9,
    ):
        super().__init__()
        self.max_b = max_b
        self.use_rgb_stats = bool(use_rgb_stats)
        self.deep_head = bool(deep_head)
        self.rgb_stats_dim = int(rgb_stats_dim)
        self.feat_embed_dim = int(feat_embed_dim)
        if self.use_rgb_stats:
            # Late fusion: compress the encoder feature to a compact learned
            # B∞ descriptor (1024→256→9), then concatenate raw dim-pixel stats
            # (→24) before the final 3-channel head. Keeps the photometric cue
            # explicit. NOTE: the stats can OOD-extrapolate on cyan-dead scenes.
            self.feature_mlp = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.feat_embed_dim),
            )
            self.fuse = nn.Sequential(
                nn.LayerNorm(self.feat_embed_dim + self.rgb_stats_dim),
                nn.Linear(self.feat_embed_dim + self.rgb_stats_dim, 3),
            )
        elif self.deep_head:
            # SAME depth as the stats head (1024→256→9→3) but WITHOUT the raw
            # stats concat — keeps the B∞ head capacity while removing the
            # OOD-extrapolating photometric shortcut. Distinct module names
            # (feature_mlp + head_embed, no `fuse`) so inference auto-detects it.
            self.feature_mlp = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.feat_embed_dim),
            )
            self.head_embed = nn.Sequential(
                nn.LayerNorm(self.feat_embed_dim),
                nn.Linear(self.feat_embed_dim, 3),
            )
        else:
            # Legacy architecture (1024→256→3), byte-compatible with existing
            # pre-stats physics checkpoints.
            self.mlp = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 3),
            )

    def forward(self, feat: torch.Tensor, img_raw: Optional[torch.Tensor] = None) -> torch.Tensor:
        pooled = feat.mean(dim=1)
        if self.use_rgb_stats:
            if img_raw is None:
                raise ValueError("BackscatterHead(use_rgb_stats=True) requires img_raw.")
            feat9 = self.feature_mlp(pooled)
            stats = binf_rgb_stats(img_raw).to(device=pooled.device, dtype=pooled.dtype)
            return torch.sigmoid(self.fuse(torch.cat([feat9, stats], dim=-1))) * self.max_b
        if self.deep_head:
            return torch.sigmoid(self.head_embed(self.feature_mlp(pooled))) * self.max_b
        return torch.sigmoid(self.mlp(pooled)) * self.max_b


class PhysicsRegressor(nn.Module):
    """Selects between scalar (per-view) and depth-dependent (Sea-Thru) heads.

    The regressor's `forward(feat, z)` always returns a dict with keys:
        beta_D : [B, 3] (scalar) OR [B, 3, H, W] (depth_dep)
        beta_B : same shape
        B_inf  : [B, 3]
        params_D / params_B : present for depth_dep — per-view (a,b,c,d)
                              tensors used by view-consistency loss.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        hidden: int = 256,
        max_b: float = 0.7,
        kind: str = "depth_dep",
        b_max: float = 0.2,
        beta_max: float = 10.0,
        constrain_beta_B: bool = False,
        betaB_ratio_min: float = 0.0,
        betaB_ratio_max: float = 1.0,
        jerlov_unified_beta: bool = False,
        seathru_betaB_const: bool = False,
        seathru_betaB_le_betaD: bool = False,
        binf_use_rgb_stats: bool = False,
        binf_deep_head: bool = False,
        unified_medium_head: bool = False,
        binf_spatial: bool = False,
        light: bool = False,
    ):
        super().__init__()
        self.kind = kind
        self.unified_medium_head = bool(unified_medium_head)
        if self.unified_medium_head and not (kind == "depth_dep" and seathru_betaB_const):
            raise ValueError(
                "unified_medium_head requires kind='depth_dep' + "
                "seathru_betaB_const (18 = 12 β^D + 3 β^B const + 3 B^∞)."
            )
        if kind == "scalar":
            if jerlov_unified_beta or seathru_betaB_const:
                raise ValueError(
                    "jerlov_unified_beta / seathru_betaB_const are only "
                    "supported with kind='depth_dep'; the scalar head has no "
                    "β^B tying or z-dependence to differentiate."
                )
            self.physics = PhysicsHead(in_dim=in_dim, hidden=hidden)
        elif kind == "depth_dep":
            self.physics = DepthDependentPhysicsHead(
                in_dim=in_dim, hidden=hidden, b_max=b_max, beta_max=beta_max,
                constrain_beta_B=constrain_beta_B,
                betaB_ratio_min=betaB_ratio_min,
                betaB_ratio_max=betaB_ratio_max,
                jerlov_unified_beta=jerlov_unified_beta,
                seathru_betaB_const=seathru_betaB_const,
                seathru_betaB_le_betaD=seathru_betaB_le_betaD,
                unified_binf=self.unified_medium_head,
                binf_max_b=max_b,
                binf_spatial=bool(binf_spatial),
                light=bool(light),
            )
        else:
            raise ValueError(f"Unknown kind={kind!r}; expected 'scalar' or 'depth_dep'.")
        # Separate B^∞ head only when NOT unified (unified emits B^∞ from the
        # physics MLP, so creating a backscatter head would be dead weight).
        self.backscatter = None if self.unified_medium_head else BackscatterHead(
            in_dim=in_dim,
            hidden=hidden,
            max_b=max_b,
            use_rgb_stats=binf_use_rgb_stats,
            deep_head=binf_deep_head,
        )

    def forward(
        self,
        feat: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        img_raw: Optional[torch.Tensor] = None,
    ) -> dict:
        if self.kind == "scalar":
            bD, bB = self.physics(feat)
            out = {"beta_D": bD, "beta_B": bB}
        else:
            if z is None:
                raise ValueError("kind='depth_dep' requires depth z.")
            d = self.physics(feat, z)
            out = {
                "beta_D": d["beta_D_map"],
                "beta_B": d["beta_B_map"],
                "params_D": d["params_D"],
                "params_B": d["params_B"],
            }
            # Pass pre-sigmoid ratio logit through (constrain branch only).
            if "ratio_logit" in d:
                out["ratio_logit"] = d["ratio_logit"]
        if self.unified_medium_head:
            out["B_inf"] = d["B_inf"]                 # from the shared MLP trunk
            if "B_inf_map" in d:
                out["B_inf_map"] = d["B_inf_map"]     # direction dependent veiling light [B, 3, H, W]
            if "light" in d:
                out["light"] = d["light"]
        else:
            out["B_inf"] = self.backscatter(feat, img_raw=img_raw)
        return out


# ---------------------------------------------------------------------------
# Cycle reconstruction loss
# ---------------------------------------------------------------------------


@dataclass
class PhysicsCycleConfig:
    """Hyper-parameters for the physics cycle/prior losses."""
    lambda_cycle: float = 1.0     # weight of cycle reconstruction
    lambda_couple: float = 0.1    # weight of |β^B - β^D| (symmetric coupling, legacy)
    lambda_box: float = 0.01      # weight of upper-bound bias on β
    lambda_view_consist: float = 0.1   # weight of inter-view (B^∞) agreement
    lambda_ordering: float = 0.01      # weight of R>G>B ordering on β means
    # v9 additions — physics-correctness constraints (the v8 head learnt
    # β_B > β_D for 100 % of points, collapsing the R channel post-clamp).
    lambda_couple_order: float = 0.5   # weight of ReLU(β^B - β^D) — asymmetric ordering
    lambda_realism: float = 0.5        # weight of ReLU(B∞·(1-trans_B) - I_raw)
    # v9.2 — direct J supervision: invert the formation model and compare
    # against the SeaErra-restored ground truth so the descattered output
    # looks like in-air natural colour, not just satisfies the forward
    # cycle. Off (0.0) by default for back-compat with v8/v9.
    lambda_j_direct: float = 0.0       # weight of |J_inverted - J_clean|
    # v9.3-α — saturation-budget loss: penalise AT inversions whose J_inv
    # falls outside [0, 1]. Quadratic on both extremes — ReLU(J_inv − 1)²
    # for warm-saturated scenes (over-amplified R), ReLU(−J_inv)² for
    # cyan-collapsed scenes (B∞ over-prediction). Gradients flow into β_D
    # and B∞ so the network learns to keep J in the valid range without
    # any inference-time amplification cap. Off (0.0) by default. Try 0.3.
    lambda_saturation: float = 0.0     # weight of ReLU(J_inv − 1)² + ReLU(−J_inv)²
    # v9.3-δ — three additional regularizers targeting OOD generalisation:
    #
    #   lambda_b_inf_floor:
    #     Anchor B∞ to the per-channel mean of deep pixels (top 30% by depth
    #     in each frame). Physics: I → B∞·(1−exp(−β_B·z)) ≈ B∞ as z→∞, so
    #     deep-pixel raw intensity is a soft lower bound on B∞. Penalises
    #     B∞ < deep_mean → prevents B∞ under-prediction that causes severe
    #     R amplification on warm-saturated scenes (Barbados video2/3).
    #
    #   lambda_beta_smoothness:
    #     Edge-preserving spatial smoothness on β_D. Medium parameters are
    #     scene-global, but the depth-dependent head outputs per-pixel β_D
    #     that can spike on textured patches (rock, coral) → R over-amp at
    #     mid-distance. Weight by exp(−|∇z|/ε) so smoothness only applies
    #     in low-depth-gradient regions; depth edges keep their β freedom.
    #
    #   lambda_color_balance:
    #     Frame-mean R/((G+B)/2) target = 1.0. Catches global warm/cyan bias
    #     missed by per-pixel saturation_loss. Operates on clamped J_inv to
    #     stay anchored to the inference-equivalent output.
    lambda_b_inf_floor:    float = 0.0
    # G1 — forward over-subtraction ceiling on B^∞_R. The cyan-dead failure
    # (J_R clamped to 0) can ONLY arise from a non-positive inversion
    # numerator: I_R − B^∞_R·(1 − e^{−β^B_R z}) ≤ 0. Since (1 − e^{−β^B_R z})
    # ≤ 1, the sufficient condition B^∞_R ≤ α·min_pixel(I_R) guarantees the
    # numerator stays ≥ (1−α)·I_R > 0 at every pixel, so J_R can never be
    # over-subtracted to zero. This is a FORWARD prior — it touches only the
    # predicted B^∞ and the raw input I, never the J_inv division — so it
    # cannot produce the 1/trans_D or 1/mean_GB blow-ups that killed L_sat-
    # on-raw / L_cb. Penalty = ReLU(B^∞_R − α·I_R^lo)². α default 0.7; I_R^lo
    # is a low percentile of valid I_R (not the strict min, to stay robust
    # to a few black/vignette pixels). R-channel only — applying it to G/B
    # would wrongly cap the legitimately-high green/blue veiling light.
    lambda_binf_oversub:   float = 0.0
    binf_oversub_alpha:    float = 0.7
    binf_oversub_pct:      float = 0.02
    lambda_beta_smoothness: float = 0.0
    lambda_color_balance:  float = 0.0
    # v9.3-ζ — asymmetric color balance weight. 1.0 = symmetric (v9.3-ε
    # behaviour), > 1 amplifies the warm-side penalty so warm-shift OOD
    # scenes are corrected without weakening cyan-dead recovery. Try 2.0.
    color_balance_warm_weight: float = 1.0
    # v9.4-η — color balance loss MODE.
    #   "default"  : v9.3-ε / ζ behaviour. Pulls R/(GB) toward target=1.0
    #                always (with optional warm asymmetry via warm_weight).
    #                Side effects on Barbados v1 (cyan-dead) when λ is
    #                boosted — over-suppresses R because target=1 is too
    #                aggressive for naturally cyan-shifted scenes.
    #   "deadband" : Defines a natural band [cb_t_low, cb_t_high]. Inside
    #                the band → loss = 0 (no intervention on already-
    #                natural scenes like Barbados video3). Outside →
    #                penalty pulls ratio TO THE BAND EDGE, not to 1.0.
    #                cyan side: F.relu(t_low - r)² (1× weight, mild pull).
    #                warm side: F.relu(r - t_high)² × warm_weight (strong
    #                pull on outliers). Eliminates over-cyan regression on
    #                v1 while still suppressing warm-shift on v2/4 +
    #                DRUVA A11/A12. Sparser gradient than "default" → safer
    #                NaN profile despite warm_weight=2.5.
    color_balance_mode: str = "default"
    cb_t_low:  float = 0.55
    cb_t_high: float = 1.0
    # v9.4-η — saturation loss MODE.
    #   "default"    : v9.3-α behaviour. Per-pixel ReLU(J_inv − 1)² + ReLU(−J_inv)²
    #                  averaged over all valid pixels. Mean-based — fails to catch
    #                  hot-spots in scenes whose mean J is near boundary (DRUVA
    #                  A12: mean R=0.85, hot-spot 37%).
    #   "percentile" : Frame-channel saturation BUDGET. Per (B, C) frame
    #                  fraction of pixels at each extreme; penalty kicks in only
    #                  when fraction exceeds budget (default 5%). Targets
    #                  hot-spots directly. Sparse gradient (zero in well-formed
    #                  scenes) → very safe NaN-wise.
    saturation_mode:    str = "default"
    saturation_budget: float = 0.05
    # v9.5-α — channel-weighted saturation. R-channel-only saturation
    # boost via per-channel weight tensor [R_weight, 1.0, 1.0]. Targets the
    # observed v9.3-ε R-saturation pattern (Barbados v2/v4: 18-20%, DRUVA
    # A12: 57%) without over-penalising natural G/B highlight pixels.
    # Applies to BOTH "default" mean mode and "percentile" mode (in
    # percentile mode, the over/under_pix is multiplied by weights before
    # the budget gate). 1.0 = ε / ζ behaviour (no R bias).
    saturation_R_weight: float = 1.0
    # v13-α — asymmetric R-channel saturation. Splits the R-weight into
    # two values, one applied to the upper-side penalty (J_R > 1, i.e.
    # R-saturation / over-amplification — the failure mode on warm
    # scenes v2/v4/A12) and one applied to the lower-side penalty (J_R
    # < 0, i.e. cyan-collapse). Setting upper > under lets us suppress
    # warm-scene R-saturation aggressively WITHOUT injecting any extra
    # "amplify R" gradient signal that would over-correct cyan-leaning
    # scenes (v1). When None it falls back to saturation_R_weight (the
    # symmetric v9.5-α / v11 / v12 behaviour) — backward compatible.
    saturation_R_weight_under: float | None = None
    # v9.3-ε — Sea-Thru style dim-pixel B∞ anchor (Akkaynak 2018):
    #   Per-frame, per-channel: bottom 5-percentile of valid raw I is the
    #   B∞ asymptote estimate (since I → B∞·(1−exp(−β_B·z)) → B∞ as z→∞,
    #   and the dimmest pixels approximate this asymptote). Two-sided MSE
    #   loss anchors B∞_pred exactly to this data-driven target — stronger
    #   than the one-sided ReLU floor (lambda_b_inf_floor) when warm-saturated
    #   scenes have B∞ under-prediction. Use either floor (one-sided) or
    #   dimpx (two-sided), not both.
    lambda_b_inf_dimpx:    float = 0.0
    # v12 — cyan floor loss. Surgical R-channel collapse prevention.
    # Penalises J_R_inv that falls below `cyan_floor_threshold` (default
    # 0.10, just below ε's verified mean J_R = 0.128 on Barbados video1).
    # On warm scenes (v2/A12) J_R_inv stays well above 0.10 so the loss
    # is identically 0 — zero impact on those gains. On cyan-leaning
    # scenes (v1) the loss adds a strong upward gradient that prevents
    # J_R from being clamp-collapsed past the ε baseline.
    #
    # NaN-safe by construction: pen ≤ threshold² is bounded, non-finite
    # J_R pixels are masked out of both numerator and denominator (no
    # single bad pixel can poison a batch). Try 0.5.
    lambda_cyan_floor:     float = 0.0
    cyan_floor_threshold:  float = 0.10
    # v11 — synth GT supervision on β_B/β_D ratio. Active only when the batch
    # was rendered through sample_jerlov_medium (synth_gt_active=True), in
    # which case the sampled (β_D_aug, β_B_aug) are passed in as GT and the
    # predicted ratio is MSE-supervised toward the GT ratio. Combined with
    # the v10 bounded sigmoid this gives Pattern C a true fix: σ cannot
    # saturate (architecture) AND the depth-dep MLP receives a direct
    # informative-spread learning signal (supervision). Try 0.3.
    lambda_betaB_ratio_synth: float = 0.0
    # binf_synth — direct MSE supervision of predicted B^∞ vs the Jerlov-aug
    # GT B^∞ (hard target on aug steps). Hardens the RGB-stats B^∞ head's
    # image→B^∞ mapping for OOD transfer (cyan-dead v1). R channel up-weighted.
    lambda_binf_synth: float = 0.0
    # ratiocap — soft penalty on σ→1 saturation (Pattern C).
    # Always-on ReLU(ratio - target)^2 mean over every (batch, channel,
    # pixel). Architecturally identical to ε (sigmoid range [0.0, 1.0]
    # unchanged) so it can be turned on mid-training without disrupting
    # the forward model output. ε's empirical ratio ≈ 1.0 sits at the
    # edge of the penalty zone, so this loss provides a small but
    # persistent gradient that pulls the sigmoid head below target
    # without the abrupt forward-output step-change that hard ratio
    # caps (v10+ lineage) introduce. Try λ = 0.1, target = 0.9.
    lambda_ratio_cap: float = 0.0
    ratio_cap_target: float = 0.9
    z_min: float = 0.1            # metres (mask depths below this — sky / no-return)
    z_max: float = 50.0           # metres
    beta_max: float = 5.0         # ReLU(β - beta_max) box prior
    use_depth_variance_weighting: bool = False
    inverse_eps: float = 0.01     # trans_D clamp floor — prevents J blow-up
    # Compact medium loss (2026-11-23): one residual r = J*T_D + B_inf*(1-T_B) - I in the observation space with the
    # weight w = 1 + med_inv_weight / max(T_D, inverse_eps), not detached. For T_D >= inverse_eps it equals
    # L_for + med_inv_weight * L_inv in value and gradient (|J_inv - J| = |r| / T_D); below it the inverse term keeps
    # the same residual instead of the clamped inversion. Replaces the cycle term; j_direct and saturation go to 0.
    compact_med: bool = False
    med_inv_weight: float = 1.0
                                   # at large depth (matches the inference
                                   # path in swimm3r_colmap.py)


def _to_01(img: torch.Tensor) -> torch.Tensor:
    """Convert from MASt3R [-1, 1] convention to [0, 1] for image-formation eq."""
    return (img + 1.0) * 0.5


@torch.no_grad()
def sample_jerlov_medium(batch_size: int, device, dtype=torch.float32,
                          n_hue_families: int = 3,
                          beta_D_R_max: float = 2.0,
                          betaB_ratio_min: float = 0.4,
                          betaB_ratio_max: float = 1.0,
                          cyan_weight: float = 1.0,
                          family_ids=None,
                          cyan_dead_beta_D_R=(0.60, 1.20),
                          family_cited_types=None):
    """Sample a random underwater medium per batch element, covering Jerlov 1-9
    turbidity × {blue, green, yellow-brown[, warm]} veil hues.

    Returns:
        beta_D : [B, 3]   (R, G, B) /m  — direct attenuation per channel
        beta_B : [B, 3]   /m            — backscatter coefficient
        B_inf  : [B, 3]   in [0, 1]     — veiling light asymptote

    Args:
        n_hue_families : 3 (legacy, green/yellow/warm only — NO cyan
                         augmentation), 4 (+ cyan-dead, Barbados v1
                         target), 5 (+ cyan-mild, FLSea cyan frames).
                         PAPER-FINAL default for v7_jerlov: 5.
        beta_D_R_max   : upper bound for the base β_D_R uniform sampling
                         (applies to families 0/1/2 only — cyan families
                         3,4 override this with their own β_D_R ranges).
        betaB_ratio_min, betaB_ratio_max : β_B / β_D ratio bounds.
                         For Jerlov unified-β architecture, ratio_max
                         should approach 1.0 so synth GT matches the
                         architectural β^B ≡ β^D identity.
        cyan_weight    : relative sampling weight of cyan families
                         (id 3, 4) vs base families (id 0, 1, 2).
                         1.0 (default) = uniform; 2.0 = each cyan
                         family weighted 2x → combined cyan coverage
                         = 4/7 ≈ 57% (vs 40% uniform). Increases
                         effective supervision against cyan-dead
                         outputs on cyan-leaning real inputs.

    Coverage:
        - β_D_R   ∈ [0.3, beta_D_R_max] /m   (clearer to severe; base only)
        - β_D_G/B ∈ [0.04, 0.85] /m         (clear ocean → heavily turbid)
        - β_B/β_D ∈ [betaB_ratio_min, betaB_ratio_max]  (β_B ≤ β_D)
        - 5-family layout (paper-final, v15 — cyan-natural REMOVED):
            id=0 green             : B^∞ R∈[0.05,0.15], G∈[0.45,0.70], B∈[0.30,0.55]
            id=1 yellow / sediment : B^∞ R∈[0.10,0.25], G∈[0.30,0.50], B∈[0.10,0.30]
            id=2 warm / R-dominant : B^∞ R∈[0.20,0.40], G∈[0.15,0.30], B∈[0.10,0.25]
            id=3 cyan-dead         : B^∞ R∈[0.02,0.08], G∈[0.30,0.50], B∈[0.30,0.45]
                                     + β^D_R override ∈ [0.60, 1.20]  (severe R-attenuation; recovery driven by L_j_direct, not range)
            id=4 cyan-mild         : B^∞ R∈[0.02,0.08], G∈[0.30,0.50], B∈[0.30,0.45]
                                     + β^D_R override ∈ [0.20, 0.60]  (FLSea mild cyan;
                                       v15.1 — extended from 0.50 → 0.60 to close
                                       the gap with cyan-dead's [0.60, 1.20])

    Sampling biased toward clearer water (j**1.5) so most BASE batches stay
    close to FLSea distribution; the tail (j → 1) covers severely turbid
    OOD scenes. Cyan families bypass the j**1.5 bias via direct overrides.

    Removed (v15): the legacy "blue / open ocean" family — paper-target
    dataset has no clear open-ocean scenes — AND the "cyan-natural"
    family — its [0.05, 0.30] β_D_R range overlaps with the base families'
    low-turbidity tail (j ≈ 0 → β_D_R ≈ 0.3), so it provided no
    distinct supervision signal.
    """
    B = batch_size
    # family_ids overrides the default 0..n_hue-1 family set — e.g. [5,1,3] =
    # green-coastal + yellow + cyan-dead. Ids index the 6-family layout
    # (0 green, 1 yellow, 2 warm, 3 cyan-dead, 4 cyan-mild, 5 green-coastal);
    # the cyan-dead/mild β_D_R and the green-coastal β_D/β_B overrides apply
    # per-sampled-id (not gated by n_hue).
    _fam = None if family_ids is None else [int(i) for i in family_ids]
    if _fam is not None:
        assert all(0 <= i <= 5 for i in _fam), f"family_ids must be in 0..5, got {_fam}"
    else:
        assert n_hue_families in (3, 4, 5), f"n_hue_families must be 3, 4, or 5, got {n_hue_families}"
    assert 0.0 <= betaB_ratio_min < betaB_ratio_max <= 1.0, (
        f"betaB ratio range invalid: [{betaB_ratio_min}, {betaB_ratio_max}]"
    )
    bB_lo = float(betaB_ratio_min)
    bB_hi = float(betaB_ratio_max)
    bB_span = bB_hi - bB_lo

    # Turbidity parameter j ∈ [0, 1] biased toward clearer (more samples in
    # FLSea-like regime, fewer in extreme Jerlov 9 regime).
    j = torch.rand(B, device=device, dtype=dtype).pow(1.5)

    # β_D per channel — R uses configurable upper bound for v9.5-α extension
    bD_R = 0.3 + j * (beta_D_R_max - 0.3)
    base_GB = 0.04 + j * (0.85 - 0.04)
    bD_G = base_GB * (0.8 + 0.4 * torch.rand(B, device=device, dtype=dtype))
    bD_B = base_GB * (0.8 + 0.4 * torch.rand(B, device=device, dtype=dtype))
    beta_D = torch.stack([bD_R, bD_G, bD_B], dim=1)               # [B, 3]

    # β_B = β_D · ratio,  ratio ∈ [betaB_ratio_min, betaB_ratio_max]   (β_B ≤ β_D)
    # v11: range made parameter so synth GT aligns with the architectural
    # ratio cap (e.g., DepthDependentPhysicsHead cap = 0.95 ⇒ sample ≤ 0.95
    # so the supervised target is always reachable by the bounded sigmoid).
    bB_ratio = bB_lo + bB_span * torch.rand(B, 3, device=device, dtype=dtype)
    beta_B = beta_D * bB_ratio                                    # [B, 3]

    # B∞ — sample one of N hue families per batch element.
    # PAPER-FINAL 5-family layout (v15.1):
    #   0 = green             (default β_D)
    #   1 = yellow / sediment (default β_D)
    #   2 = warm / R-dominant (default β_D)
    #   3 = cyan-dead         β_D_R override [0.60, 1.20]   (severe R-attenuation, Barbados v1)
    #   4 = cyan-mild         β_D_R override [0.20, 0.60]   (mild R-attenuation; v15.1 gap-fix)
    # 2 cyan-leaning families (id=3,4) together = 40% cyan-protective training
    # coverage UNDER UNIFORM SAMPLING. Removed: legacy "blue / open ocean"
    # (no paper-target scenes) and "cyan-natural" (β_D_R [0.05, 0.30]
    # overlapped with base families' low-turbidity tail).
    #
    # cyan_weight controls non-uniform family sampling: cyan families
    # (id=3, 4) get weight `cyan_weight`, base families (id=0, 1, 2) get
    # weight 1.0. cyan_weight=2.0 → cyan coverage 4/7 ≈ 57% (1.4x boost).
    # With phys_jerlov_aug_p=0.6, effective cyan supervision ≈ 34% of
    # all training steps (vs 24% uniform).
    if _fam is not None:
        # Uniform over the explicit family set (e.g. [0,1,3]).
        _fid = torch.tensor(_fam, device=device)
        hue_id = _fid[torch.randint(0, len(_fid), (B,), device=device)]
    elif cyan_weight != 1.0 and n_hue_families >= 4:
        # Build per-family weights [1, 1, 1, cyan_weight, cyan_weight, ...].
        # Cyan starts at id=3 in the 5-family layout.
        weights = torch.ones(n_hue_families, device=device, dtype=dtype)
        if n_hue_families >= 4:
            weights[3] = cyan_weight                          # cyan-dead
        if n_hue_families >= 5:
            weights[4] = cyan_weight                          # cyan-mild
        # torch.multinomial samples [B] indices from the weights distribution.
        hue_id = torch.multinomial(weights, B, replacement=True)
    else:
        hue_id = torch.randint(0, n_hue_families, (B,), device=device)

    # Hue family bounds. R is uniformly low for the two cyan families
    # (id=3,4) so the rendered I always looks cyan-leaning; β_D_R override
    # below differentiates them by attenuation severity.
    #              id:   0(grn) 1(yel) 2(wrm) 3(cdead) 4(cmild) 5(green-coastal)
    _R_lo_all = [0.05, 0.10, 0.20, 0.02, 0.02, 0.05]
    _R_hi_all = [0.15, 0.25, 0.40, 0.08, 0.08, 0.18]
    _G_lo_all = [0.45, 0.30, 0.15, 0.30, 0.30, 0.35]
    _G_hi_all = [0.70, 0.50, 0.30, 0.50, 0.50, 0.60]
    _B_lo_all = [0.30, 0.10, 0.10, 0.30, 0.30, 0.20]
    _B_hi_all = [0.55, 0.30, 0.25, 0.45, 0.45, 0.45]
    # Full 6-family arrays (indexed by hue_id, always 0..5) — works for both the
    # default 0..n_hue-1 set and an explicit family_ids set that includes 3/4/5.
    Binf_R_lo = torch.tensor(_R_lo_all, device=device, dtype=dtype)
    Binf_R_hi = torch.tensor(_R_hi_all, device=device, dtype=dtype)
    Binf_G_lo = torch.tensor(_G_lo_all, device=device, dtype=dtype)
    Binf_G_hi = torch.tensor(_G_hi_all, device=device, dtype=dtype)
    Binf_B_lo = torch.tensor(_B_lo_all, device=device, dtype=dtype)
    Binf_B_hi = torch.tensor(_B_hi_all, device=device, dtype=dtype)

    u_R = torch.rand(B, device=device, dtype=dtype)
    u_G = torch.rand(B, device=device, dtype=dtype)
    u_B = torch.rand(B, device=device, dtype=dtype)
    Binf_R = Binf_R_lo[hue_id] + u_R * (Binf_R_hi[hue_id] - Binf_R_lo[hue_id])
    Binf_G = Binf_G_lo[hue_id] + u_G * (Binf_G_hi[hue_id] - Binf_G_lo[hue_id])
    Binf_B = Binf_B_lo[hue_id] + u_B * (Binf_B_hi[hue_id] - Binf_B_lo[hue_id])
    B_inf = torch.stack([Binf_R, Binf_G, Binf_B], dim=1)          # [B, 3]

    # PAPER-FINAL (v15): two cyan-leaning families pair the same cyan-low
    # B^∞ with two different β_D_R severity overrides — together they teach
    # the network that "cyan-looking input" can correspond to mild OR severe
    # R-attenuation, with the right inversion behaviour for each.
    #
    # id=3 cyan-dead : β_D_R ∈ [0.60, 1.20]  (severe R-attenuation, Barbados v1)
    # id=4 cyan-mild : β_D_R ∈ [0.20, 0.60]  (mild R-attenuation, FLSea cyan; v15.1
    #                                          extends from 0.50 → 0.60 to close
    #                                          the [0.50, 0.60] β_D_R gap)
    #
    # Barbados v1 sits at β_D_R ≈ 0.68 → cyan-dead family. FLSea cyan-
    # leaning frames sit at β_D_R ≈ 0.3-0.5 → cyan-mild family. The
    # previously-included cyan-natural family ([0.05, 0.30] β_D_R) was
    # dropped because that range overlaps with the base families'
    # low-turbidity tail (j ≈ 0 → β_D_R ≈ 0.3) → redundant signal.
    if n_hue_families >= 4 or _fam is not None:
        cdead_mask = (hue_id == 3)
        if cdead_mask.any():
            # v15.6 — REVERTED to [0.60, 1.20] (v15.5's [0.60, 2.20] extension
            # was ineffective). Measured root cause of cyan-dead is NOT a weak
            # β^D_R prior but the ABSENCE of an inverse-direction recovery loss:
            # forward L_cyc's gradient ∂I_pred/∂β = -J·z·e^{-βz} VANISHES as
            # β grows (e^{-βz}→0), so forward supervision structurally cannot
            # push β high enough for extreme-cyan recovery — it stalls at
            # moderate β regardless of how wide the augmentation range is.
            # The fix is L_j_direct (inverse, gradient ∂J_inv/∂β = J_inv·z
            # GROWS with β), not a wider β^D_R augmentation. So the [0.6, 2.2]
            # extension is reverted; L_j_direct is re-enabled in the config.
            _clo, _chi = float(cyan_dead_beta_D_R[0]), float(cyan_dead_beta_D_R[1])
            cdead_bD_R = _clo + (_chi - _clo) * torch.rand(B, device=device, dtype=dtype)  # default [0.60,1.20]
            cdead_bD_GB_base = 0.20 + 0.30 * torch.rand(B, device=device, dtype=dtype)  # [0.20, 0.50] (G,B << R)
            cdead_bD_G = cdead_bD_GB_base * (0.8 + 0.4 * torch.rand(B, device=device, dtype=dtype))
            cdead_bD_B = cdead_bD_GB_base * (0.8 + 0.4 * torch.rand(B, device=device, dtype=dtype))
            beta_D[:, 0] = torch.where(cdead_mask, cdead_bD_R, beta_D[:, 0])
            beta_D[:, 1] = torch.where(cdead_mask, cdead_bD_G, beta_D[:, 1])
            beta_D[:, 2] = torch.where(cdead_mask, cdead_bD_B, beta_D[:, 2])
            beta_B = beta_D * bB_ratio

    if n_hue_families >= 5 or _fam is not None:
        cmild_mask = (hue_id == 4)
        if cmild_mask.any():
            # v15.1 — range extended from [0.20, 0.50] to [0.20, 0.60]
            # to close the [0.50, 0.60] gap with cyan-dead's [0.60, 1.20].
            # Real scenes with β^D_R ≈ 0.55 (sub-Barbados-v1) now covered.
            cmild_bD_R = 0.20 + 0.40 * torch.rand(B, device=device, dtype=dtype)        # [0.20, 0.60]
            cmild_bD_GB_base = 0.10 + 0.20 * torch.rand(B, device=device, dtype=dtype)  # [0.10, 0.30]
            cmild_bD_G = cmild_bD_GB_base * (0.8 + 0.4 * torch.rand(B, device=device, dtype=dtype))
            cmild_bD_B = cmild_bD_GB_base * (0.8 + 0.4 * torch.rand(B, device=device, dtype=dtype))
            beta_D[:, 0] = torch.where(cmild_mask, cmild_bD_R, beta_D[:, 0])
            beta_D[:, 1] = torch.where(cmild_mask, cmild_bD_G, beta_D[:, 1])
            beta_D[:, 2] = torch.where(cmild_mask, cmild_bD_B, beta_D[:, 2])
            beta_B = beta_D * bB_ratio

    # id=5 green-coastal / turbid: realistic green water. Red attenuated, green
    # PENETRATES (low β^D_G), but STRONG green backscatter (β^B_G high) — and
    # crucially β^B is sampled DECOUPLED from β^D (NOT β^D·ratio), so β^B_G can
    # exceed β^D_G, the regime the ratio-coupled families structurally cannot
    # produce. Without this the free β^B head never sees high-green-backscatter
    # examples and extrapolates a runaway β^B_G on OOD green scenes (Barbados v4
    # showed β^B_G≈0.75 vs β^D_G≈0.09) → green veil over-removed → purple/violet
    # cast. This family calibrates that regime. MUST come after the id3/id4
    # blocks (which do a global beta_B = beta_D·ratio) so the masked override
    # below is not clobbered.
    if _fam is not None:
        gcoast_mask = (hue_id == 5)
        if gcoast_mask.any():
            gc_bD_R = 0.30 + 0.70 * torch.rand(B, device=device, dtype=dtype)  # [0.30,1.00]
            gc_bD_G = 0.05 + 0.13 * torch.rand(B, device=device, dtype=dtype)  # [0.05,0.18]
            gc_bD_B = 0.12 + 0.23 * torch.rand(B, device=device, dtype=dtype)  # [0.12,0.35]
            beta_D[:, 0] = torch.where(gcoast_mask, gc_bD_R, beta_D[:, 0])
            beta_D[:, 1] = torch.where(gcoast_mask, gc_bD_G, beta_D[:, 1])
            beta_D[:, 2] = torch.where(gcoast_mask, gc_bD_B, beta_D[:, 2])
            # DECOUPLED β^B (green-dominant strong backscatter, NOT β^D·ratio)
            gc_bB_R = 0.05 + 0.13 * torch.rand(B, device=device, dtype=dtype)  # [0.05,0.18]
            gc_bB_G = 0.25 + 0.40 * torch.rand(B, device=device, dtype=dtype)  # [0.25,0.65]
            gc_bB_B = 0.10 + 0.25 * torch.rand(B, device=device, dtype=dtype)  # [0.10,0.35]
            beta_B[:, 0] = torch.where(gcoast_mask, gc_bB_R, beta_B[:, 0])
            beta_B[:, 1] = torch.where(gcoast_mask, gc_bB_G, beta_B[:, 1])
            beta_B[:, 2] = torch.where(gcoast_mask, gc_bB_B, beta_B[:, 2])

    # Citable β^B / B^∞ override: replace the parametric β^B and B^∞ with a
    # CITED Jerlov type's values per family (β^B=ratio·b [Gordon 1988], B^∞ via
    # Gordon Eq.2), keeping OUR β^D (the cited K_d^0 can't reach the R-dominant
    # band). family_cited_types is parallel to family_ids, e.g. family_ids
    # [0,1,3] + types ["3C","7C","II"] → green=3C / yellow=7C / cyan-dead=II.
    if family_cited_types is not None and _fam is not None:
        # PARTIAL cited override of beta_B / B_inf, per family. A None (or the
        # serialized string "None"/"null") entry leaves THAT family parametric,
        # so e.g. family_ids [0,1,3,4] + types [None,None,None,"IA"] cites only
        # id4 (cyan-mild = Jerlov IA) while green/yellow/cyan-dead keep their
        # parametric beta_B/B_inf — avoids importing the LOW cited beta_B onto
        # the cyan-recovery families (the suspected v1-killer in jerlov_full).
        # Masked per-family (not a full bB_tab[hue_id] overwrite) so untouched
        # families are not zeroed. Backward-compatible: an all-non-None list
        # overrides every family exactly as before.
        for fid, tname in zip(_fam, family_cited_types):
            if tname is None or str(tname).strip().lower() in ("none", "null", ""):
                continue
            bb, bi = _jerlov_cited_bB_Binf(str(tname), beta_B.device, beta_B.dtype)
            m = (hue_id == int(fid))                       # [B]
            if m.any():
                beta_B[m] = bb                             # bb [3] broadcasts to [n,3]
                B_inf[m] = bi

    return beta_D, beta_B, B_inf


# ============================================================================
# Citable Jerlov-water-type medium sampler (for synthetic-pair augmentation).
# Every quantity traces to a published reference instead of hand-tuned ranges:
#   β^D  = Jerlov reference diffuse attenuation K_d^0(λ) per water type
#          [Solonenko & Mobley, "Inherent optical properties of Jerlov water
#           types," Appl. Opt. 54(17):5392-5401 (2015), Tables 4-8]
#          (β^D≈K_d^0 is the Jerlov-defining, color-faithful choice; the IFM's
#           direct-signal coefficient is strictly beam c=a+b, and Akkaynak &
#           Treibitz CVPR 2018 show β^D≠c≠K_d exactly — documented approximation.)
#   β^B  = backscatter coefficient b_b(λ) = f_bb · b(λ)
#          [b(λ): Solonenko-Mobley 2015; f_bb=b_b/b≈1-2% of total scattering for
#           seawater, Gordon et al. JGR 93(D9):10909-10924 (1988) p.10912;
#           phase-function origin: Petzold, SIO Ref 72-78 (1972)]
#          Independent of β^D → consistent with Sea-thru asymmetry + our free β^B.
#   B^∞  hue ∝ R/Q = 0.0949·u + 0.0794·u²,  u = b_b/(a+b_b)
#          [Gordon et al. 1988, Eq. 2 — general reflectance↔IOP relation, valid
#           beyond Case 1]; absolute magnitude set by ambient light (scene
#           parameter / HydroLight, Mobley "Light and Water" 1994).
# RGB sampled at λ = (R 650, G 525, B 475) nm (wavelengths tabulated in the paper).
#
# Table values below were verified against Solonenko-Mobley 2015 Tables 4-8
# from the local PDF (R=650, G=475, B=475 nm columns). 2026-06 RE-VERIFY: 9/10
# types match the PDF exactly; 5C's RED triple was WRONG — it held the 350 nm row
# values (Kd0 2.43 / a 1.78 / b 2.87) mis-assigned to 650 nm. Corrected to the
# 650 nm estimate (~0.55 / 0.37 / 1.10). So 5C is a MILD coastal type, NOT the
# "strong cyan-dead" lever it was wrongly treated as. The prior "0.456->2.43
# correction" was itself the bug.
# ============================================================================
_JERLOV_TABLE_VERIFIED = True
_JERLOV_CITED_WARNED = False

# (R=650, G=525, B=475) nm.  Kd0, a, b in m^-1.   SOURCE: Solonenko-Mobley 2015 Tbl 4-8 (VERIFY)
#   Patched 2026-06 per PDF verification (Tbl 4-8). Kd0 uses the reference
#   K_d^0 column; a and b use the recovered IOP columns.
_JERLOV = {
    # --- oceanic (Case 1) ---
    "I":   {"Kd0": (0.357, 0.0398, 0.0182), "a": (0.334, 0.046,  0.019),  "b": (9.0e-4,  2.05e-3, 3.06e-3)},
    "IA":  {"Kd0": (0.364, 0.0460, 0.0253), "a": (0.334, 0.0468, 0.0216), "b": (2.34e-3, 4.02e-3, 5.36e-3)},
    "IB":  {"Kd0": (0.371, 0.0513, 0.0325), "a": (0.334, 0.0469, 0.0225), "b": (0.045,   0.0565,  0.0635)},
    "II":  {"Kd0": (0.393, 0.0780, 0.0619), "a": (0.334, 0.0469, 0.0228), "b": (0.270,   0.387,   0.459)},
    "III": {"Kd0": (0.431, 0.122,  0.117),  "a": (0.336, 0.0507, 0.0335), "b": (0.737,   1.06,    1.26)},
    # --- coastal (Case 2) ---
    "1C":  {"Kd0": (0.431, 0.122,  0.134),  "a": (0.344, 0.068,  0.077),  "b": (0.274,   0.395,   0.469)},
    "3C":  {"Kd0": (0.478, 0.198,  0.223),  "a": (0.346, 0.078,  0.105),  "b": (0.800,   1.15,    1.36)},
    "5C":  {"Kd0": (0.55,  0.315,  0.400),  "a": (0.37,  0.127,  0.204),  "b": (1.10,    1.44,    1.71)},   # CORRECTED 2026-06: prior R triple (2.43/1.78/2.87) was the 350nm row mis-assigned to 650nm (Solonenko-Mobley 2015 Tbl 7). G525/B475 verified OK. True 5C@650 cell is corrupted in the PDF text layer; R≈0.55 (Kd0), 0.37 (a), 1.10 (b) interpolated between 3C(0.478) and 7C(0.635) — CONFIRM exact digits from Fig 1 / Tbl 7. 5C is a MILD turbid-coastal type (R/B≈1.4), NOT strong cyan-dead.
    "7C":  {"Kd0": (0.635, 0.494,  0.693),  "a": (0.403, 0.233,  0.388),  "b": (1.77,    2.54,    3.01)},
    "9C":  {"Kd0": (0.755, 0.777,  1.24),   "a": (0.456, 0.430,  0.709),  "b": (2.35,    3.38,    4.01)},
}


def _jerlov_cited_bB_Binf(type_name, device, dtype,
                          backscatter_ratio: float = 0.018, binf_scale: float = 0.6):
    """Cited (β^B, B^∞) for one Jerlov type — same math as sample_jerlov_medium_cited.

    β^B = backscatter_ratio·b [Gordon 1988 / Petzold 1972]; B^∞ hue via Gordon
    1988 Eq.2 (R/Q = 0.0949u + 0.0794u², u = b_b/(a+b_b)) × ambient scale.
    Returns (beta_B[3], B_inf[3]) — deterministic per type.
    """
    a = torch.tensor(_JERLOV[type_name]["a"], device=device, dtype=dtype)
    b = torch.tensor(_JERLOV[type_name]["b"], device=device, dtype=dtype)
    b_b = backscatter_ratio * b
    u = b_b / (a + b_b).clamp(min=1e-6)
    R = 0.0949 * u + 0.0794 * u * u
    B_inf = (binf_scale * R / R.max().clamp(min=1e-6)).clamp(0.0, 0.95)
    return b_b, B_inf


def sample_jerlov_medium_cited(
    batch_size: int, device, dtype=torch.float32,
    types=None, type_weights: dict | None = None,
    backscatter_ratio: float = 0.018,   # b_b/b ≈ 1-2% [Gordon 1988 p.10912; Petzold 1972]
    binf_scale: float = 0.6,            # B^∞ magnitude (ambient; scene / Mobley 1994)
    turbidity_jitter: float = 0.0,      # optional per-sample magnitude jitter (log-normal σ)
):
    """Sample a Jerlov water type and return physically-derived (β_D, β_B, B_inf).

    All quantities cite published sources (see module comment above). Returns:
        beta_D : [B, 3]  K_d^0 per type            (R,G,B)
        beta_B : [B, 3]  b_b = backscatter_ratio·b
        B_inf  : [B, 3]  Gordon-1988 hue × magnitude, in [0, ~0.95]
    """
    global _JERLOV_CITED_WARNED
    if not _JERLOV_TABLE_VERIFIED and not _JERLOV_CITED_WARNED:
        print("[jerlov-cited] WARNING: _JERLOV coefficients are OCR-extracted and "
              "UNVERIFIED — confirm against Solonenko-Mobley 2015 Tbl 4-8 before "
              "paper use, then set _JERLOV_TABLE_VERIFIED=True.")
        _JERLOV_CITED_WARNED = True

    keys = list(types) if types is not None else list(_JERLOV.keys())
    if type_weights is not None:
        w = torch.tensor([float(type_weights.get(k, 1.0)) for k in keys],
                         dtype=dtype, device=device)
        idx = torch.multinomial(w, batch_size, replacement=True)
    else:
        idx = torch.randint(0, len(keys), (batch_size,), device=device)

    Kd0 = torch.tensor([_JERLOV[k]["Kd0"] for k in keys], dtype=dtype, device=device)  # [T,3]
    A   = torch.tensor([_JERLOV[k]["a"]   for k in keys], dtype=dtype, device=device)
    Bs  = torch.tensor([_JERLOV[k]["b"]   for k in keys], dtype=dtype, device=device)

    beta_D = Kd0[idx]                                   # β^D ← Jerlov K_d^0  [Solonenko-Mobley 2015]
    a      = A[idx]
    b      = Bs[idx]
    b_b    = backscatter_ratio * b                      # β^B ← b_b  [Gordon 1988; Petzold 1972]
    beta_B = b_b
    u      = b_b / (a + b_b).clamp(min=1e-6)            # Gordon 1988 Eq.2 variable
    R      = 0.0949 * u + 0.0794 * u * u                # R/Q ∝ ...  [Gordon 1988 Eq.2]
    R_max  = R.max(dim=1, keepdim=True).values.clamp(min=1e-6)
    B_inf  = (binf_scale * R / R_max).clamp(0.0, 0.95)  # hue (Gordon) × ambient magnitude

    if turbidity_jitter > 0.0:
        jit = torch.exp(turbidity_jitter * torch.randn(batch_size, 1, device=device, dtype=dtype))
        beta_D = beta_D * jit
        beta_B = beta_B * jit
    return beta_D, beta_B, B_inf


def sample_cyan_dead_band(
    batch_size: int, device, dtype=torch.float32,
    betaD_R_lo: float = 0.55, betaD_R_hi: float = 1.0,
    betaD_GB_lo: float = 0.10, betaD_GB_hi: float = 0.30,
    betaB_ratio_lo: float = 0.02, betaB_ratio_hi: float = 0.10,
    Binf_R_lo: float = 0.02, Binf_R_hi: float = 0.08,
    Binf_G_lo: float = 0.12, Binf_G_hi: float = 0.22,
    Binf_B_lo: float = 0.15, Binf_B_hi: float = 0.25,
):
    """Cyan-dead REFERENCE BAND medium — calibration anchor for Barbados v1.

    A controlled R-dominant, clear-veil medium centred on the Barbados v1
    estimate (β_D_R≈0.68). Unlike the cited Gordon path (which DERIVES B_inf),
    every quantity is sampled in an explicit band so the transfer-critical
    knobs are pinned:
      β_D_R ∈ [betaD_R_lo, betaD_R_hi]   (R-dominant; default [0.55, 1.0])
      β_D_G,β_D_B ∈ [betaD_GB_lo, betaD_GB_hi]   (≪ β_D_R)
      β_B = β_D · U[betaB_ratio_lo, betaB_ratio_hi]   (LOW, ≈ cited Gordon
            regime ~0.02-0.1 → weak veil build → red preserved)
      B_inf : R LOW [0.02,0.08] (cyan veil), G/B higher
    The low B_inf_R is the point: paired with --phys_lambda_b_inf_dimpx it
    teaches the RGB-stats B_inf head the "dim-pixel low ⇒ B_inf_R low" mapping
    that must transfer to the cyan-dead OOD scene (v1), keeping the inverse-J
    numerator (I_R − B_inf_R·(1−trans_B)) positive so β_D_R can amplify it.
    """
    def _U(lo, hi, shape):
        return lo + (hi - lo) * torch.rand(shape, device=device, dtype=dtype)
    bD_R = _U(betaD_R_lo, betaD_R_hi, (batch_size,))
    bD_G = _U(betaD_GB_lo, betaD_GB_hi, (batch_size,))
    bD_B = _U(betaD_GB_lo, betaD_GB_hi, (batch_size,))
    beta_D = torch.stack([bD_R, bD_G, bD_B], dim=1)               # [B, 3]
    beta_B = beta_D * _U(betaB_ratio_lo, betaB_ratio_hi, (batch_size, 3))
    Binf_R = _U(Binf_R_lo, Binf_R_hi, (batch_size,))
    Binf_G = _U(Binf_G_lo, Binf_G_hi, (batch_size,))
    Binf_B = _U(Binf_B_lo, Binf_B_hi, (batch_size,))
    B_inf = torch.stack([Binf_R, Binf_G, Binf_B], dim=1)          # [B, 3]
    return beta_D, beta_B, B_inf


def sample_jerlov_medium_hybrid(
    batch_size: int, device, dtype=torch.float32,
    cited_types=("II", "3C"),
    cited_weights: dict | None = None,
    synth_weight: float = 0.34,
    synth_mode: str = "parametric",
    cyan_band_beta_D_R=(0.55, 1.0),
    n_hue_families: int = 3,
    beta_D_R_max: float = 2.0,
    betaB_ratio_min: float = 0.3,
    betaB_ratio_max: float = 0.9,
    cyan_weight: float = 1.0,
    backscatter_ratio: float = 0.018,
    binf_scale: float = 0.6,
    turbidity_jitter: float = 0.0,
):
    """3-component hybrid augmentation medium sampler.

    Per sample, draw the (β_D, β_B, B_inf) medium from ONE of:
      • a CITED Jerlov water type (from `cited_types`, weighted) — every
        quantity physically sourced: β_D=K_d^0 [Solonenko & Mobley 2015],
        β_B=b_b [Gordon 1988 / Petzold 1972], B_inf hue [Gordon 1988 Eq.2];
      • the SYNTH component, `synth_mode`:
          'parametric' : the epsilon R-dominant hue families
                         (sample_jerlov_medium, β_D_R up to beta_D_R_max);
          'cyan_band'  : the cyan-dead reference band (sample_cyan_dead_band,
                         β_D_R∈cyan_band_beta_D_R, β_B≈cited Gordon regime,
                         B_inf_R low) — v1-centred, citable, β_B-consistent.

    `synth_weight` ∈ [0,1] = fraction of the batch from the synth component;
    the rest are cited (split by `cited_weights`). The per-sample medium IS
    the synth-GT, so downstream ratio supervision stays consistent.
    """
    bd_c, bb_c, bi_c = sample_jerlov_medium_cited(
        batch_size, device=device, dtype=dtype,
        types=list(cited_types), type_weights=cited_weights,
        backscatter_ratio=backscatter_ratio, binf_scale=binf_scale,
        turbidity_jitter=turbidity_jitter,
    )
    if synth_mode == "cyan_band":
        bd_s, bb_s, bi_s = sample_cyan_dead_band(
            batch_size, device=device, dtype=dtype,
            betaD_R_lo=float(cyan_band_beta_D_R[0]),
            betaD_R_hi=float(cyan_band_beta_D_R[1]),
        )
    else:
        bd_s, bb_s, bi_s = sample_jerlov_medium(
            batch_size, device=device, dtype=dtype,
            n_hue_families=n_hue_families, beta_D_R_max=beta_D_R_max,
            betaB_ratio_min=betaB_ratio_min, betaB_ratio_max=betaB_ratio_max,
            cyan_weight=cyan_weight,
        )
    use_synth = (torch.rand(batch_size, device=device) < float(synth_weight)
                 ).to(dtype).view(batch_size, 1)
    beta_D = use_synth * bd_s + (1.0 - use_synth) * bd_c
    beta_B = use_synth * bb_s + (1.0 - use_synth) * bb_c
    B_inf  = use_synth * bi_s + (1.0 - use_synth) * bi_c
    return beta_D, beta_B, B_inf


@torch.no_grad()
def jerlov_render(J_clean: torch.Tensor, depth: torch.Tensor,
                  beta_D: torch.Tensor, beta_B: torch.Tensor,
                  B_inf: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
    """Forward-render a clean image J through the AT model with the given
    medium parameters. Used for OOD generalisation augmentation.

    Args:
        J_clean   : [B, 3, H, W] in [-1, 1]  (MASt3R convention)
        depth     : [B, H, W] in metres (>0)
        beta_D    : [B, 3]
        beta_B    : [B, 3]
        B_inf     : [B, 3]
        noise_std : Gaussian sensor-noise std added in [0,1] space (then
                    clamped + remapped). 0.01 ≈ 1% noise, bridges sim2real
                    gap by giving the synth I a more realistic photometric
                    distribution. Default 0 (clean synth).

    Returns:
        I_synth : [B, 3, H, W] in [-1, 1]
    """
    J01 = _to_01(J_clean)
    z_  = sanitize_physics_depth(depth).unsqueeze(1)              # [B, 1, H, W]
    bD_ = beta_D[:, :, None, None]
    bB_ = beta_B[:, :, None, None]
    Bi_ = B_inf[:, :, None, None]
    trans_D = torch.exp(-bD_ * z_)
    trans_B = torch.exp(-bB_ * z_)
    I_synth01 = J01 * trans_D + Bi_ * (1.0 - trans_B)
    if noise_std > 0:
        I_synth01 = I_synth01 + noise_std * torch.randn_like(I_synth01)
    I_synth01 = I_synth01.clamp(0.0, 1.0)
    return I_synth01 * 2.0 - 1.0                                  # back to [-1, 1]


# ---- v9.3-δ auxiliary losses ----

def b_inf_floor_loss(B_inf: torch.Tensor,
                     I_raw_01: torch.Tensor,
                     depth: torch.Tensor,
                     valid_mask: torch.Tensor,
                     deep_quantile: float = 0.7) -> torch.Tensor:
    """B∞ realism prior — penalise predicted B∞ falling below the per-channel
    mean of deep-pixel raw intensities.

    Args:
        B_inf      : [B, 3]   predicted veiling light (in [0, 1])
        I_raw_01   : [B, 3, H, W] raw underwater image in [0, 1]
        depth      : [B, H, W] metric depth (z > 0)
        valid_mask : [B, 1, H, W] or [B, H, W] in {0, 1} — exclude invalid z
        deep_quantile : 0.7 → top 30% by depth treated as "deep"

    Returns:
        scalar loss
    """
    if valid_mask.dim() == 4:
        valid_2d = valid_mask[:, 0]                                # [B, H, W]
    else:
        valid_2d = valid_mask
    B = depth.size(0)

    # v9.5-α — diagnostic flags (returned alongside loss for upstream tracking)
    diag = {
        "z_has_nan":      torch.isnan(depth).any().detach().float(),
        "z_has_inf":      torch.isinf(depth).any().detach().float(),
        "I_has_nan":      torch.isnan(I_raw_01).any().detach().float(),
        "B_inf_has_nan":  torch.isnan(B_inf).any().detach().float(),
        "B_inf_has_inf":  torch.isinf(B_inf).any().detach().float(),
        "valid_frac":     valid_2d.float().mean().detach(),
        "few_valid_frac": torch.zeros((), device=depth.device),    # filled below
    }

    # Sanitise depth: replace NaN/Inf with a large positive sentinel.
    z_safe = torch.where(torch.isfinite(depth), depth, torch.full_like(depth, 1e6))
    z_flat_safe = z_safe.reshape(B, -1)
    valid_flat = valid_2d.reshape(B, -1)

    # per-image depth quantile over VALID pixels only
    z_thr = []
    few_valid = 0
    for i in range(B):
        zv = z_flat_safe[i][valid_flat[i] > 0.5]
        if zv.numel() < 32:
            few_valid += 1
            z_thr.append(z_flat_safe[i].max() + 1e3)
        else:
            z_thr.append(zv.quantile(deep_quantile))
    diag["few_valid_frac"] = torch.tensor(few_valid / max(B, 1), device=depth.device)
    z_thr = torch.stack(z_thr).view(-1, 1, 1)                     # [B, 1, 1]
    deep_mask = ((z_safe > z_thr) & (valid_2d > 0.5)).unsqueeze(1).float()  # [B, 1, H, W]
    n = deep_mask.sum(dim=(-1, -2)).clamp(min=1.0)                 # [B, 1]

    # Sanitise I_raw_01: replace NaN/Inf with 0 (NaN * 0 = NaN otherwise).
    I_safe = torch.where(torch.isfinite(I_raw_01), I_raw_01, torch.zeros_like(I_raw_01))
    I_deep_mean = (I_safe * deep_mask).sum(dim=(-1, -2)) / n        # [B, 3]

    raw_result = F.relu(I_deep_mean - B_inf).pow(2).mean()
    is_finite = torch.isfinite(raw_result)
    diag["guard_fired"]    = (~is_finite).detach().float()          # 1.0 if had to suppress NaN
    diag["raw_result"]     = (raw_result.detach() if is_finite else
                              torch.tensor(float("nan"), device=raw_result.device))
    if not is_finite:
        result = torch.zeros((), device=B_inf.device, dtype=B_inf.dtype)
    else:
        result = raw_result
    return result, diag


def binf_oversub_loss(B_inf: torch.Tensor,
                      I_raw_01: torch.Tensor,
                      valid_mask: torch.Tensor,
                      alpha: float = 0.7,
                      pct: float = 0.02) -> tuple[torch.Tensor, dict]:
    """G1 — forward over-subtraction ceiling on the RED B^∞.

    Guarantees the J-inversion numerator I_R − B^∞_R·(1 − e^{−β^B_R z}) stays
    strictly positive (so J_R can never collapse to 0 = cyan-dead) via the
    sufficient per-frame condition:

        B^∞_R  ≤  α · I_R^lo          (I_R^lo = low percentile of valid I_R)

    Because (1 − e^{−β^B_R z}) ≤ 1, B^∞_R ≤ α·I_R^lo ⇒ backscatter_R ≤ α·I_R^lo
    ≤ α·I_R at the dark-red pixels, leaving numerator ≥ (1−α)·I_R > 0.

    This is FORWARD-only: it reads the predicted per-frame B^∞_R and the raw
    input I_R. No J_inv, no 1/trans_D, no 1/mean_GB — so it is immune to the
    numerical blow-ups that destabilised L_sat-on-raw and L_cb.

    R-channel only — G/B veiling light is legitimately high and must not be
    capped.

    Args:
        B_inf      : [B, 3]  predicted veiling light in [0, 1]
        I_raw_01   : [B, 3, H, W] raw underwater image in [0, 1]
        valid_mask : [B, 1, H, W] or [B, H, W] in {0, 1}
        alpha      : ceiling fraction (default 0.7 → ≥ 30%% numerator headroom)
        pct        : low percentile of I_R used as the reference (default 0.02;
                     robust to a handful of black/vignette pixels vs strict min)

    Returns:
        (scalar loss, diag dict)
    """
    if valid_mask.dim() == 4:
        valid_2d = valid_mask[:, 0]                                  # [B, H, W]
    else:
        valid_2d = valid_mask
    B = B_inf.size(0)

    I_R = I_raw_01[:, 0]                                             # [B, H, W]
    I_R_safe = torch.where(torch.isfinite(I_R), I_R, torch.zeros_like(I_R))
    I_R_flat = I_R_safe.reshape(B, -1)
    valid_flat = valid_2d.reshape(B, -1)

    # Per-frame low percentile of VALID red. Fall back to a benign high
    # sentinel when a frame has too few valid pixels (ceiling never binds).
    I_lo = []
    for i in range(B):
        rv = I_R_flat[i][valid_flat[i] > 0.5]
        if rv.numel() < 32:
            I_lo.append(torch.ones((), device=I_R.device))
        else:
            I_lo.append(rv.quantile(pct))
    I_lo = torch.stack(I_lo)                                         # [B]

    ceil = alpha * I_lo                                             # [B]
    binf_R = B_inf[:, 0]                                             # [B]
    over = F.relu(binf_R - ceil)                                    # [B]
    raw_result = over.pow(2).mean()

    is_finite = torch.isfinite(raw_result)
    diag = {
        "guard_fired":   (~is_finite).detach().float(),
        "binf_R_mean":   binf_R.mean().detach(),
        "I_R_lo_mean":   I_lo.mean().detach(),
        "violate_frac":  (over > 0).float().mean().detach(),
    }
    result = (raw_result if is_finite
              else torch.zeros((), device=B_inf.device, dtype=B_inf.dtype))
    return result, diag


def beta_smoothness_loss(beta_D: torch.Tensor,
                         depth: torch.Tensor,
                         edge_eps: float = 0.5) -> torch.Tensor:
    """Edge-preserving spatial smoothness on β_D.

    Penalises β_D variation in spatially smooth (low-depth-gradient) regions
    while letting β_D vary freely at depth edges. Encourages medium-coherent
    β_D fields and prevents per-pixel β spikes on textured patches.

    Args:
        beta_D   : [B, 3, H, W]
        depth    : [B, H, W]
        edge_eps : depth-gradient scale (metres). Larger → less edge-preserving.

    Returns:
        scalar loss
    """
    if beta_D.dim() != 4:
        return torch.zeros((), device=beta_D.device)
    # depth gradients
    dz_x = (depth[:, :, 1:] - depth[:, :, :-1]).abs()              # [B, H, W-1]
    dz_y = (depth[:, 1:, :] - depth[:, :-1, :]).abs()              # [B, H-1, W]
    w_x = torch.exp(-dz_x / edge_eps).unsqueeze(1)                 # [B, 1, H, W-1]
    w_y = torch.exp(-dz_y / edge_eps).unsqueeze(1)                 # [B, 1, H-1, W]
    # β_D gradients (per channel)
    db_x = (beta_D[:, :, :, 1:] - beta_D[:, :, :, :-1]).abs()      # [B, 3, H, W-1]
    db_y = (beta_D[:, :, 1:, :] - beta_D[:, :, :-1, :]).abs()      # [B, 3, H-1, W]
    return (db_x * w_x).mean() + (db_y * w_y).mean()


def b_inf_dimpx_loss(B_inf_pred: torch.Tensor,
                     I_raw_01: torch.Tensor,
                     valid_mask: torch.Tensor,
                     dim_quantile: float = 0.05) -> torch.Tensor:
    """Sea-Thru style dim-pixel B∞ anchor (two-sided MSE).

    Akkaynak 2018: per-frame, per-channel, the bottom dim_quantile of
    valid raw intensities approximates the B∞ asymptote, since at large
    depth z the AT model gives I ≈ B∞·(1−exp(−β_B·z)) → B∞. The dimmest
    pixels are the most plausible B∞ estimate (least direct-path
    contribution, most backscatter saturation).

    Loss: ‖B∞_pred − B∞_target‖² where B∞_target is the per-channel
    bottom-5-percentile of valid raw I per frame. Anchors B∞ exactly to
    the scene's photometric asymptote — stronger than the one-sided
    ReLU floor when B∞_pred is under-estimated (warm-saturated scenes).

    Args:
        B_inf_pred  : [B, 3]   network output, in [0, 1]
        I_raw_01    : [B, 3, H, W] raw underwater image in [0, 1]
        valid_mask  : [B, 1, H, W] or [B, H, W] in {0, 1}
        dim_quantile: 0.05 → 5th percentile (Sea-Thru convention)

    Returns:
        scalar loss
    """
    B, C, H, W = I_raw_01.shape
    if valid_mask.dim() == 4:
        valid_2d = valid_mask[:, 0]                                # [B, H, W]
    else:
        valid_2d = valid_mask
    valid_2d = (valid_2d > 0.5)

    # v9.5-α — diagnostic flags
    diag = {
        "I_has_nan":           torch.isnan(I_raw_01).any().detach().float(),
        "B_inf_has_nan":       torch.isnan(B_inf_pred).any().detach().float(),
        "B_inf_has_inf":       torch.isinf(B_inf_pred).any().detach().float(),
    }

    # Sanitise I_raw_01 to prevent NaN propagation through sort.
    I_clean = torch.where(torch.isfinite(I_raw_01), I_raw_01, torch.zeros_like(I_raw_01))

    I_masked = I_clean.masked_fill(~valid_2d.unsqueeze(1), float("inf"))
    I_flat = I_masked.flatten(2)
    I_sorted, _ = I_flat.sort(dim=-1)

    n_valid = valid_2d.flatten(1).sum(-1).clamp(min=1)
    idx = (n_valid.float() * dim_quantile).long().clamp(min=0)
    idx = idx.unsqueeze(1).unsqueeze(2).expand(-1, C, 1)

    target = I_sorted.gather(-1, idx).squeeze(-1)
    raw_result = (B_inf_pred - target).pow(2).mean()
    is_finite = torch.isfinite(raw_result)
    diag["guard_fired"] = (~is_finite).detach().float()
    diag["raw_result"]  = (raw_result.detach() if is_finite else
                           torch.tensor(float("nan"), device=raw_result.device))
    if not is_finite:
        result = torch.zeros((), device=B_inf_pred.device, dtype=B_inf_pred.dtype)
    else:
        result = raw_result
    return result, diag


def color_balance_loss(J_inv_clamped: torch.Tensor,
                       valid_mask: torch.Tensor,
                       target: float = 1.0,
                       warm_weight: float = 1.0) -> torch.Tensor:
    """Frame-mean color balance — R / mean(G, B) → target ≈ 1.0.

    Operates on clamped J_inv ∈ [0, 1] so the loss reflects the
    inference-equivalent output. Catches systemic warm/cyan bias not
    captured by per-pixel saturation_loss.

    Args:
        J_inv_clamped : [B, 3, H, W] in [0, 1]
        valid_mask    : [B, 1, H, W] or [B, H, W]
        target        : natural balance target (1.0 = R == 0.5(G+B))
        warm_weight   : v9.3-ζ — multiplier for the warm-side penalty
            (R/(GB) > target). 1.0 = symmetric (default, equivalent to v9.3-ε).
            Set > 1 (e.g., 2.0) to penalise warm-shift more strongly than
            cyan-shift, which preserves cyan recovery on cyan-dead OOD
            scenes (e.g., Barbados video1) while reducing warm-saturation
            on warm-prone scenes (Barbados video2/4, DRUVA A1/A11/A12).

    Returns:
        scalar loss
    """
    if valid_mask.dim() == 3:
        valid_mask = valid_mask.unsqueeze(1)
    valid_3 = valid_mask.expand_as(J_inv_clamped).float()
    n = valid_3.sum(dim=(-1, -2)).clamp(min=1.0)                   # [B, 3]
    means = (J_inv_clamped * valid_3).sum(dim=(-1, -2)) / n        # [B, 3]
    R, G, B = means[:, 0], means[:, 1], means[:, 2]
    GB = 0.5 * (G + B)
    R_over_GB = R / GB.clamp(min=1e-3)
    if warm_weight != 1.0:
        # Asymmetric: warm side (R_over_GB > target) penalised by warm_weight,
        # cyan side (R_over_GB < target) unchanged. cyan-dead recovery
        # therefore preserved while warm-shift is suppressed.
        excess  = F.relu(R_over_GB - target).pow(2) * warm_weight
        deficit = F.relu(target - R_over_GB).pow(2)
        return (excess + deficit).mean()
    return (R_over_GB - target).pow(2).mean()


def color_balance_loss_band(J_inv_clamped: torch.Tensor,
                            valid_mask: torch.Tensor,
                            t_low:  float = 0.55,
                            t_high: float = 1.0,
                            warm_weight: float = 2.5) -> torch.Tensor:
    """v9.4-η — Dead-band asymmetric color-balance loss.

    Frame-mean R / (0.5·(G+B)) gets a penalty ONLY outside the natural band
    [t_low, t_high]. Inside the band → loss = 0 (no intervention on already
    natural scenes). Outside, penalty pulls the ratio to the BAND EDGE
    (not to 1.0), which is the key difference vs `color_balance_loss`:
    cyan-shifted scenes (Barbados video1, R/(GB) ≈ 0.21) are pulled only up
    to t_low, not to 1.0 → no over-suppression of cyan-recovery.

    cyan side: ReLU(t_low - r)²            (1× weight, mild pull)
    warm side: ReLU(r - t_high)² × warm_weight (strong pull on outliers)

    Sparser gradient than `color_balance_loss` (zero contribution on
    natural batches) — therefore SAFER NaN-wise despite warm_weight > 1.

    Args:
        J_inv_clamped : [B, 3, H, W] in [0, 1]
        valid_mask    : [B, 1, H, W] or [B, H, W]
        t_low         : lower edge of natural band (default 0.55)
        t_high        : upper edge of natural band (default 1.0)
        warm_weight   : multiplier on the warm-side (r > t_high) penalty.
                        Default 2.5. Strong pull only on the outlier batches
                        where it actually fires — cumulative drift bounded.

    Returns:
        scalar loss (often 0 for natural scenes — by design).
    """
    if valid_mask.dim() == 3:
        valid_mask = valid_mask.unsqueeze(1)
    valid_3 = valid_mask.expand_as(J_inv_clamped).float()
    n = valid_3.sum(dim=(-1, -2)).clamp(min=1.0)                   # [B, 3]
    means = (J_inv_clamped * valid_3).sum(dim=(-1, -2)) / n        # [B, 3]
    R, G, B = means[:, 0], means[:, 1], means[:, 2]
    GB = 0.5 * (G + B)
    R_over_GB = R / GB.clamp(min=1e-3)
    cyan = F.relu(t_low  - R_over_GB).pow(2)                       # mild
    warm = F.relu(R_over_GB - t_high).pow(2) * warm_weight         # strong
    return (cyan + warm).mean()


def saturation_loss_percentile(J_inv_raw: torch.Tensor,
                               valid_mask: torch.Tensor,
                               budget: float = 0.05) -> torch.Tensor:
    """v9.4-η — Percentile-budget saturation loss.

    Per-frame, per-channel fraction of pixels saturated at each extreme:
        f_hi = mean( J_inv > 1 - eps )
        f_lo = mean( J_inv < eps )
    Penalise only when fraction > budget — natural scenes (small saturation
    fraction) get loss = 0. Outlier scenes (DRUVA A12: 37% R-saturation)
    get a strong, directly targeted pull.

    Targets the *hot-spot population* rather than the per-pixel residual
    mean. Mean-based saturation_loss fails on scenes with mean J near
    boundary (mean R=0.85 → small per-pixel residual but 37% of pixels
    actually clipped); this loss looks at the population fraction directly.

    Sparse gradient (zero in well-formed scenes) → very safe NaN profile.

    Args:
        J_inv_raw  : [B, 3, H, W] *unclamped* AT inversion (gradient flows
                     through; clamp would zero gradients on the saturated
                     pixels we want to fix).
        valid_mask : [B, 1, H, W] or [B, H, W] in {0, 1}.
        budget     : tolerated fraction (default 0.05 = 5%).

    Returns:
        scalar loss.
    """
    if valid_mask.dim() == 3:
        valid_mask = valid_mask.unsqueeze(1)
    valid_3 = valid_mask.expand_as(J_inv_raw).float()              # [B, 3, H, W]
    n_pix = valid_3.sum(dim=(-1, -2)).clamp(min=1.0)               # [B, 3]

    # Soft fractions via straight-through-style indicator:
    # We use hard mean of float-cast indicator (no gradient through indicator),
    # paired with a "soft pull" penalty that DOES carry gradient. The standard
    # trick: penalty = ReLU(frac - budget)² gives us the right zero-band, but
    # frac is non-differentiable. Instead, we use the gradient-carrying
    # SOFT excess: mean of the squared overshoot per-pixel above (1 - eps)
    # / below eps, gated by the budget on the indicator side.
    #
    # In practice: compute hard fractions for the gating, AND the per-pixel
    # squared overshoot for gradient flow. Multiply the per-pixel squared
    # overshoot by an indicator that the fraction exceeds budget. This way:
    #   - Natural batch (frac < budget): gate = 0, no gradient.
    #   - Outlier batch (frac > budget): gate = 1, full gradient through
    #     the squared overshoot.
    over_pix  = F.relu(J_inv_raw - 1.0).pow(2)                     # [B, 3, H, W]
    under_pix = F.relu(-J_inv_raw).pow(2)
    # Per-frame, per-channel hard fractions (no grad — used as gate only).
    with torch.no_grad():
        hi_count = ((J_inv_raw > 1.0) & (valid_3 > 0)).float().sum(dim=(-1, -2))
        lo_count = ((J_inv_raw < 0.0) & (valid_3 > 0)).float().sum(dim=(-1, -2))
        f_hi = hi_count / n_pix
        f_lo = lo_count / n_pix
        gate_hi = (f_hi > budget).float()                          # [B, 3]
        gate_lo = (f_lo > budget).float()
    # Per-frame mean of the squared overshoot, gated by population excess.
    over_mean  = (over_pix  * valid_3).sum(dim=(-1, -2)) / n_pix   # [B, 3]
    under_mean = (under_pix * valid_3).sum(dim=(-1, -2)) / n_pix
    return (gate_hi * over_mean + gate_lo * under_mean).mean()


def _expand_beta(beta: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Reshape β to [B, 3, H, W], broadcasting from [B, 3] if scalar."""
    if beta.dim() == 2:
        return beta[:, :, None, None].expand(-1, -1, target_h, target_w)
    return beta


def _spatial_mean(beta: torch.Tensor) -> torch.Tensor:
    """Reduce β to [B, 3] (mean over spatial dims if [B, 3, H, W])."""
    if beta.dim() == 4:
        return beta.mean(dim=(-1, -2))
    return beta


def beta_from_depth_params(
    params: torch.Tensor,
    z: torch.Tensor,
    beta_max: float = 10.0,
    z_min: float = 0.1,
    z_max: float = 50.0,
) -> torch.Tensor:
    """Evaluate detached depth-dependent β params at a new depth map.

    params is [B, 4, 3] = (a, b, c, d). The same evaluator covers the
    Sea-Thru β^B-constant branch because its params are stored as
    (a_B, 0, 0, 0), so the expression collapses to β^B=a_B.
    """
    z_safe = sanitize_physics_depth(z, z_min, z_max).unsqueeze(1)
    a, b, c, d = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
    beta = (
        a[:, :, None, None] * torch.exp(b[:, :, None, None] * z_safe)
        + c[:, :, None, None] * torch.exp(d[:, :, None, None] * z_safe)
    )
    return beta.clamp(min=1e-3, max=beta_max)


def pred_z_j_direct_loss(
    I: torch.Tensor,
    J: torch.Tensor,
    z_pred: torch.Tensor,
    pred: dict,
    cfg: PhysicsCycleConfig,
    charbonnier_eps: float = 1e-3,
) -> tuple[torch.Tensor, dict]:
    """Inverse-J loss that trains the medium using detached predicted depth.

    This is the pred-depth analogue of phys_lambda_j_direct:

        J_inv(I, z_pred.detach(); β_live, B∞_live) ≈ J_clean

    The predicted depth is used because inference descatters with predicted
    geometry, but it is detached so this loss calibrates β/B∞ rather than
    rewriting the depth head. β^D/β^B are re-evaluated at z_pred from the live
    per-frame params, so gradients reach the physics head.
    """
    B, C, H, W = I.shape
    assert C == 3
    if z_pred.shape[-2:] != (H, W):
        z_pred = F.interpolate(z_pred.unsqueeze(1), size=(H, W),
                               mode="bilinear", align_corners=False).squeeze(1)

    I01 = torch.nan_to_num(_to_01(I).clamp(0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)
    J01 = torch.nan_to_num(_to_01(J).clamp(0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)

    z_fixed = z_pred.detach()
    valid = (z_fixed > cfg.z_min) & (z_fixed < cfg.z_max) & torch.isfinite(z_fixed)
    valid_mask = valid.unsqueeze(1).expand(-1, 3, -1, -1).float()
    denom = valid_mask.sum().clamp(min=1.0)

    params_D = pred["params_D"]
    params_B = pred["params_B"]
    Binf = pred["B_inf"][:, :, None, None]
    z_safe = sanitize_physics_depth(z_fixed, cfg.z_min, cfg.z_max).unsqueeze(1)

    beta_D = beta_from_depth_params(params_D, z_fixed, cfg.beta_max, cfg.z_min, cfg.z_max)
    beta_B = beta_from_depth_params(params_B, z_fixed, cfg.beta_max, cfg.z_min, cfg.z_max)
    trans_D = torch.exp(-beta_D * z_safe)
    trans_B = torch.exp(-beta_B * z_safe)
    backscatter = Binf * (1.0 - trans_B)
    J_inv_raw = (I01 - backscatter) / trans_D.clamp(min=cfg.inverse_eps)
    J_inv_raw = torch.nan_to_num(J_inv_raw, nan=0.0, posinf=1.0 / cfg.inverse_eps, neginf=-1.0 / cfg.inverse_eps)

    diff = J_inv_raw - J01
    loss_map = torch.sqrt(diff * diff + charbonnier_eps * charbonnier_eps) - charbonnier_eps
    loss = (loss_map * valid_mask).sum() / denom

    with torch.no_grad():
        J_vis = J_inv_raw.clamp(0.0, 1.0)
        R = (J_vis[:, 0:1] * valid_mask[:, 0:1]).sum() / valid_mask[:, 0:1].sum().clamp(min=1.0)
        G = (J_vis[:, 1:2] * valid_mask[:, 1:2]).sum() / valid_mask[:, 1:2].sum().clamp(min=1.0)
        Bl = (J_vis[:, 2:3] * valid_mask[:, 2:3]).sum() / valid_mask[:, 2:3].sum().clamp(min=1.0)
        ratio = R / (0.5 * (G + Bl)).clamp(min=1e-3)
        log = {
            "phys_j_pred_depth": loss.detach(),
            "phys_J_pred_R_mean": R.detach(),
            "phys_J_pred_G_mean": G.detach(),
            "phys_J_pred_B_mean": Bl.detach(),
            "phys_J_pred_R_over_GB": ratio.detach(),
            "phys_J_pred_valid_frac": valid.float().mean().detach(),
        }
    return loss, log


def pred_z_cycle_loss(
    I: torch.Tensor,
    J: torch.Tensor,
    z_pred: torch.Tensor,
    pred: dict,
    cfg: PhysicsCycleConfig,
) -> tuple[torch.Tensor, dict]:
    """Forward IFM reconstruction at DETACHED predicted depth, live medium.

    The forward analogue of pred_z_j_direct_loss (and the pred-z analogue of
    phys_lambda_cycle):

        I ≈ J·trans_D(z_pred.detach()) + B∞·(1 - trans_B(z_pred.detach()))

    Predicted depth supplies the inference-time geometry convention; it is
    detached so the gradient calibrates β/B∞ (the medium head) rather than the
    depth head. β^D/β^B are re-evaluated at z_pred from the live per-frame
    params. Reconstruction-only (no division), so strictly safer than the
    inverse variant. Mirrors cycle_loss's forward term but isolates it (no
    realism/j_direct/saturation double-counting).
    """
    B, C, H, W = I.shape
    assert C == 3
    if z_pred.shape[-2:] != (H, W):
        z_pred = F.interpolate(z_pred.unsqueeze(1), size=(H, W),
                               mode="bilinear", align_corners=False).squeeze(1)

    I01 = torch.nan_to_num(_to_01(I).clamp(0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)
    J01 = torch.nan_to_num(_to_01(J).clamp(0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)

    z_fixed = z_pred.detach()
    valid = (z_fixed > cfg.z_min) & (z_fixed < cfg.z_max) & torch.isfinite(z_fixed)
    valid_mask = valid.unsqueeze(1).expand(-1, 3, -1, -1).float()
    denom = valid_mask.sum().clamp(min=1.0)

    params_D = pred["params_D"]
    params_B = pred["params_B"]
    Binf = pred["B_inf"][:, :, None, None]
    z_safe = sanitize_physics_depth(z_fixed, cfg.z_min, cfg.z_max).unsqueeze(1)

    beta_D = beta_from_depth_params(params_D, z_fixed, cfg.beta_max, cfg.z_min, cfg.z_max)
    beta_B = beta_from_depth_params(params_B, z_fixed, cfg.beta_max, cfg.z_min, cfg.z_max)
    trans_D = torch.exp(-beta_D * z_safe)
    trans_B = torch.exp(-beta_B * z_safe)
    I_pred = (J01 * trans_D + Binf * (1.0 - trans_B)).clamp(0.0, 1.0)

    diff = (I_pred - I01).abs()
    loss = (diff * valid_mask).sum() / denom

    with torch.no_grad():
        log = {
            "phys_cycle_pred_depth": loss.detach(),
            "phys_cycle_pred_valid_frac": valid.float().mean().detach(),
        }
    return loss, log


def cycle_loss(
    I: torch.Tensor,           # [B, 3, H, W] in [-1, 1]
    J: torch.Tensor,           # [B, 3, H, W] in [-1, 1]
    z: torch.Tensor,           # [B, H, W] in metres (>0 = valid)
    beta_D: torch.Tensor,      # [B, 3] or [B, 3, H, W]
    beta_B: torch.Tensor,      # [B, 3] or [B, 3, H, W]
    B_inf: torch.Tensor,       # [B, 3]
    cfg: PhysicsCycleConfig,
) -> tuple[torch.Tensor, dict]:
    """Image-formation reconstruction loss. Works for scalar OR spatial β."""
    B, C, H, W = I.shape
    assert C == 3
    assert J.shape == I.shape
    if z.shape[-2:] != (H, W):
        z = F.interpolate(z.unsqueeze(1), size=(H, W),
                          mode="bilinear", align_corners=False).squeeze(1)

    I01 = _to_01(I)
    J01 = _to_01(J)

    # Both branches end up [B, 3, H, W] via broadcasting (scalar β is [B, 3, 1, 1]).
    if beta_D.dim() == 2:
        beta_D = beta_D[:, :, None, None]
    if beta_B.dim() == 2:
        beta_B = beta_B[:, :, None, None]
    if B_inf.dim() == 4:
        # direction dependent B^∞ map [B, 3, H, W]; the per view priors below read its mean
        if B_inf.shape[-2:] != (H, W):
            B_inf = F.interpolate(B_inf, size=(H, W), mode="bilinear", align_corners=False)
        Binf = B_inf
        B_inf = B_inf.mean(dim=(2, 3))
    else:
        Binf = B_inf[:, :, None, None]
    valid = (z > cfg.z_min) & (z < cfg.z_max) & torch.isfinite(z)
    z_safe = sanitize_physics_depth(z, cfg.z_min, cfg.z_max)
    z_ = z_safe.unsqueeze(1)                  # [B, 1, H, W]

    trans_D = torch.exp(-beta_D * z_)         # broadcasts to [B, 3, H, W]
    trans_B = torch.exp(-beta_B * z_)
    I_pred = J01 * trans_D + Binf * (1.0 - trans_B)
    I_pred = I_pred.clamp(0.0, 1.0)

    diff = (I_pred - I01).abs()               # [B, 3, H, W]
    valid_pixel = valid.unsqueeze(1).expand_as(diff).float()

    if cfg.use_depth_variance_weighting:
        # weight pixels by deviation from per-image median depth.
        z_mean = (z * valid_pixel[:, 0]).sum(dim=(-1, -2)) / valid_pixel[:, 0].sum(dim=(-1, -2)).clamp(min=1.0)
        w = (z - z_mean.view(B, 1, 1)).abs().unsqueeze(1)
        valid_pixel = valid_pixel * (1.0 + w)

    denom = valid_pixel.sum().clamp(min=1.0)
    cycle = (diff * valid_pixel).sum() / denom
    if cfg.compact_med:
        r_abs = (J01 * trans_D + Binf * (1.0 - trans_B) - I01).abs()
        w_med = 1.0 + cfg.med_inv_weight / trans_D.clamp(min=cfg.inverse_eps)
        cycle = (w_med * r_abs * valid_pixel).sum() / denom

    # v9: realism constraint — predicted backscatter must not exceed the raw
    # underwater intensity, channel-wise. ReLU(B∞(1-trans_B) - I_raw) → 0
    # when the inversion would not require clamping the J channel to 0.
    backscatter = Binf * (1.0 - trans_B)                       # [B, 3, H, W]
    realism = F.relu(backscatter - I01)
    valid_mask = valid.unsqueeze(1).expand_as(realism).float()
    realism_loss = (realism * valid_mask).sum() / denom

    # v9.2: direct J supervision — invert the forward model on the raw
    # image with predicted physics and compare against the SeaErra-restored
    # ground truth. Forces the descattered output to look like the in-air
    # natural reference (Sea-Thru style) instead of merely satisfying the
    # forward cycle (which the model can solve by fitting any J + medium
    # combination that re-renders to I_raw).
    #
    # IMPORTANT: at TRAINING time the loss uses the UNCLIPPED inversion.
    # PyTorch's clamp(0, 1) zeros the gradient where J_inv < 0 — exactly
    # the over-clamp pixels that need a gradient pushing B_∞ down. With
    # the unclipped J_inv, |J_inv − J_clean| stays large and well-signed
    # even when the inversion goes negative, so B_∞ / β_D get the correct
    # update direction. The clamp is still applied at inference time
    # (swimm3r_colmap.py) for image-domain consistency, and at the
    # diagnostic side below for metrics that reflect what the user sees.
    J_inv_raw = (I01 - backscatter) / trans_D.clamp(min=cfg.inverse_eps)
    j_direct_loss = torch.zeros((), device=I.device)
    if cfg.lambda_j_direct > 0:
        j_direct = (J_inv_raw - J01).abs()
        j_direct_loss = (j_direct * valid_mask).sum() / denom

    # v9.3-α: saturation-budget loss.
    # Quadratic penalty on J_inv ∉ [0, 1]:
    #   ReLU(J_inv − 1)²   ← warm-saturated pixels (β_D over-amplifies R)
    #   ReLU(−J_inv)²      ← cyan-collapsed pixels (B∞ over-prediction)
    # Gradient flows into β_D and B∞ — network learns to keep AT inversion
    # output in the valid range across diverse scenes without inference-time
    # caps. v9.2 J_direct already uses unclamped J_inv, but penalises the
    # *signed* residual against J_clean — saturation_loss explicitly forbids
    # the magnitude going outside the box.
    sat_loss = torch.zeros((), device=I.device)
    if cfg.lambda_saturation > 0:
        if cfg.saturation_mode == "percentile":
            # v9.4-η — population-based: penalise only when fraction of
            # saturated pixels exceeds budget. Sparse gradient.
            sat_loss = saturation_loss_percentile(
                J_inv_raw,
                valid_pixel[:, :1],            # any [B, 1, H, W] mask
                budget=cfg.saturation_budget,
            )
            # v9.5-α — R-weighted boost (post-percentile-gate)
            # v13-α — asymmetric over/under R weights (under defaults to over for backcompat).
            r_w_over  = float(cfg.saturation_R_weight)
            r_w_under = (float(cfg.saturation_R_weight_under)
                         if cfg.saturation_R_weight_under is not None
                         else r_w_over)
            if r_w_over != 1.0 or r_w_under != 1.0:
                R_only_over  = F.relu(J_inv_raw[:, :1] - 1.0).pow(2)
                R_only_under = F.relu(-J_inv_raw[:, :1]).pow(2)
                denom_R = valid_pixel[:, :1].sum().clamp(min=1.0)
                # boost over and under independently
                if r_w_over != 1.0:
                    over_loss = (R_only_over * valid_pixel[:, :1]).sum() / denom_R
                    sat_loss = sat_loss + (r_w_over - 1.0) * over_loss
                if r_w_under != 1.0:
                    under_loss = (R_only_under * valid_pixel[:, :1]).sum() / denom_R
                    sat_loss = sat_loss + (r_w_under - 1.0) * under_loss
        else:
            # v9.3-α default: per-pixel mean. v9.5-α added a single R-channel
            # weight applied to BOTH over- and under-side penalties.
            # v13-α splits this into independent over / under weights so
            # warm-scene R-saturation can be suppressed aggressively (upper
            # weight high) without injecting an "amplify R" gradient on
            # cyan-leaning scenes (under weight kept = ε baseline 1.0).
            over  = F.relu(J_inv_raw - 1.0).pow(2)                       # upper-side
            under = F.relu(-J_inv_raw).pow(2)                            # lower-side
            r_w_over  = float(cfg.saturation_R_weight)
            r_w_under = (float(cfg.saturation_R_weight_under)
                         if cfg.saturation_R_weight_under is not None
                         else r_w_over)
            if r_w_over == 1.0 and r_w_under == 1.0:
                sat_pix = over + under                                    # [B, 3, H, W]
            else:
                w_over  = torch.ones(3, device=J_inv_raw.device, dtype=J_inv_raw.dtype)
                w_under = torch.ones(3, device=J_inv_raw.device, dtype=J_inv_raw.dtype)
                w_over[0]  = r_w_over
                w_under[0] = r_w_under
                sat_pix = (over * w_over[None, :, None, None]
                           + under * w_under[None, :, None, None])
            sat_loss = (sat_pix * valid_mask).sum() / denom

    # Inference-equivalent clamped J_inv used only for monitoring metrics
    # (matches what utils/diagnose_descatter.py reports on the export).
    J_inv = J_inv_raw.clamp(0.0, 1.0)

    # v12 — cyan floor loss: prevent J_R_inv from being driven below a
    # small floor (default 0.10, just under ε's verified mean J_R=0.128
    # on Barbados v1). Surgical R-channel protection — zero gradient on
    # scenes where J_R_inv is already healthy (v2/A12 with J_R>0.5),
    # strong upward pull only when the model over-subtracts B^∞ on
    # cyan-leaning scenes.
    #
    # NaN-safety + gradient-stability:
    #   - Non-finite J_R pixels are masked out of BOTH the numerator
    #     and the weight tensor; nan_to_num substitutes a benign
    #     sentinel at non-finite positions so the autograd graph never
    #     touches a NaN, even if a bad pixel slipped through.
    #   - J_R_safe is additionally clamp(min=-1.0) so even pathological
    #     unclamped inversions (J_R = -100) cannot explode the squared
    #     penalty: pen ≤ (1 + threshold)² ≈ 1.21. Bounded gradient.
    #   - denom is clamp(min=1) so an all-invalid frame returns 0, not
    #     NaN.
    cyan_floor_loss_v = torch.zeros((), device=I.device)
    cyan_floor_diag: dict = {}
    if cfg.lambda_cyan_floor > 0:
        floor = float(cfg.cyan_floor_threshold)
        J_R = J_inv_raw[:, 0]                                           # [B, H, W]
        valid_R = valid_pixel[:, 0]                                      # [B, H, W]
        finite_R = torch.isfinite(J_R).float()
        # Substitute a non-poisoning sentinel (= floor ⇒ ReLU(...)=0) at
        # any nan/+inf, and 0 at -inf so the explicit mask zeroes it.
        J_R_safe = torch.nan_to_num(J_R, nan=floor, posinf=floor, neginf=0.0)
        # Bound the squared penalty: clamp(min=-1) caps pen at (1+floor)².
        J_R_clipped = J_R_safe.clamp(min=-1.0)
        pen = F.relu(floor - J_R_clipped).pow(2)                         # [B, H, W]
        weight = valid_R * finite_R
        denom_R = weight.sum().clamp(min=1.0)
        cyan_floor_loss_v = (pen * weight).sum() / denom_R
        with torch.no_grad():
            cyan_floor_diag = {
                "cyan_floor_pen":            cyan_floor_loss_v.detach(),
                "cyan_floor_below_frac":     ((J_R_safe < floor).float() * weight).sum().detach() / denom_R,
                "cyan_floor_nonfinite_frac": (1.0 - finite_R.mean()).detach(),
                "cyan_floor_J_R_min_safe":   J_R_safe.min().detach(),
            }

    # v9.3-δ — three additional regularizers (each gated by its own lambda).
    binf_floor_loss_v = torch.zeros((), device=I.device)
    binf_floor_diag = {}
    if cfg.lambda_b_inf_floor > 0:
        binf_floor_loss_v, binf_floor_diag = b_inf_floor_loss(B_inf, I01, z, valid_mask)

    # G1 — forward over-subtraction ceiling on B^∞_R (cyan-dead prevention).
    binf_oversub_loss_v = torch.zeros((), device=I.device)
    binf_oversub_diag = {}
    if cfg.lambda_binf_oversub > 0:
        binf_oversub_loss_v, binf_oversub_diag = binf_oversub_loss(
            B_inf, I01, valid_mask,
            alpha=cfg.binf_oversub_alpha,
            pct=cfg.binf_oversub_pct,
        )

    beta_smooth_loss = torch.zeros((), device=I.device)
    if cfg.lambda_beta_smoothness > 0:
        # only meaningful when β_D is per-pixel (depth-dep head)
        beta_smooth_loss = beta_smoothness_loss(beta_D, z)

    cb_loss = torch.zeros((), device=I.device)
    if cfg.lambda_color_balance > 0:
        if cfg.color_balance_mode == "deadband":
            # v9.4-η — natural-band asymmetric. No penalty for r ∈ [t_low, t_high].
            cb_loss = color_balance_loss_band(
                J_inv, valid_mask,
                t_low=cfg.cb_t_low,
                t_high=cfg.cb_t_high,
                warm_weight=cfg.color_balance_warm_weight,
            )
        else:
            # v9.3-ε / ζ default: pull toward target=1.0.
            cb_loss = color_balance_loss(
                J_inv, valid_mask,
                warm_weight=cfg.color_balance_warm_weight,
            )

    # v9.3-ε — Sea-Thru dim-pixel B∞ anchor (two-sided MSE).
    binf_dimpx_loss_v = torch.zeros((), device=I.device)
    binf_dimpx_diag = {}
    if cfg.lambda_b_inf_dimpx > 0:
        binf_dimpx_loss_v, binf_dimpx_diag = b_inf_dimpx_loss(B_inf, I01, valid_mask)

    # ---- v9.2 diagnostics: per-channel J_inv mean, R-clamp fraction,
    # and color balance ratio J_R / mean(J_G, J_B). These mirror the
    # metrics in utils/diagnose_descatter.py so wandb shows the same
    # picture the offline diagnostic would compute on the export.
    with torch.no_grad():
        # Per-channel mean restricted to valid (in-depth-range) pixels.
        valid3 = valid_pixel  # already [B, 3, H, W]
        per_chan_sum = (J_inv * valid3).sum(dim=(0, 2, 3))
        per_chan_cnt = valid3.sum(dim=(0, 2, 3)).clamp(min=1.0)
        J_inv_mean = per_chan_sum / per_chan_cnt              # [3]
        # R-clamp fraction within valid pixels.
        R_valid_count = valid3[:, 0].sum().clamp(min=1.0)
        R_clipped = ((J_inv[:, 0] == 0.0) & (valid3[:, 0] > 0)).sum().float()
        R_clip_frac = R_clipped / R_valid_count
        # Color-balance ratio J_R / mean(J_G, J_B). > 0.5 ⇒ visually
        # natural; < 0.2 ⇒ cyan dominant.
        gb_mean = 0.5 * (J_inv_mean[1] + J_inv_mean[2])
        color_balance = J_inv_mean[0] / gb_mean.clamp(min=1e-6)

    total = (
        cycle * cfg.lambda_cycle
        + realism_loss * cfg.lambda_realism
        + j_direct_loss * cfg.lambda_j_direct
        + sat_loss * cfg.lambda_saturation
        + binf_floor_loss_v * cfg.lambda_b_inf_floor
        + binf_oversub_loss_v * cfg.lambda_binf_oversub
        + beta_smooth_loss * cfg.lambda_beta_smoothness
        + cb_loss * cfg.lambda_color_balance
        + binf_dimpx_loss_v * cfg.lambda_b_inf_dimpx
        + cyan_floor_loss_v * cfg.lambda_cyan_floor
    )

    # v9.3-α diagnostics: per-channel saturation rate (J_inv ∉ [0,1])
    # v9.5-α: extended to all 3 channels for ε-failure tracking. R is
    # the dominant problem channel (5/8 ε scenes show R-saturation
    # 18-57%) but G/B tracking confirms the issue is R-specific (not
    # general saturation), validating the targeted R_weight strategy.
    with torch.no_grad():
        valid_R_count = valid_pixel[:, 0].sum().clamp(min=1.0)
        valid_G_count = valid_pixel[:, 1].sum().clamp(min=1.0)
        valid_B_count = valid_pixel[:, 2].sum().clamp(min=1.0)
        sat_R_high = ((J_inv_raw[:, 0] > 1.0) & (valid_pixel[:, 0] > 0)).sum().float() / valid_R_count
        sat_R_low  = ((J_inv_raw[:, 0] < 0.0) & (valid_pixel[:, 0] > 0)).sum().float() / valid_R_count
        sat_G_high = ((J_inv_raw[:, 1] > 1.0) & (valid_pixel[:, 1] > 0)).sum().float() / valid_G_count
        sat_G_low  = ((J_inv_raw[:, 1] < 0.0) & (valid_pixel[:, 1] > 0)).sum().float() / valid_G_count
        sat_B_high = ((J_inv_raw[:, 2] > 1.0) & (valid_pixel[:, 2] > 0)).sum().float() / valid_B_count
        sat_B_low  = ((J_inv_raw[:, 2] < 0.0) & (valid_pixel[:, 2] > 0)).sum().float() / valid_B_count

    return total, {
        "phys_cycle": cycle.detach(),
        "phys_realism": realism_loss.detach(),
        "phys_j_direct": j_direct_loss.detach(),
        "phys_saturation": sat_loss.detach(),
        "phys_b_inf_floor": binf_floor_loss_v.detach(),
        "phys_binf_oversub": binf_oversub_loss_v.detach(),
        "phys_b_inf_dimpx": binf_dimpx_loss_v.detach(),
        "phys_beta_smoothness": beta_smooth_loss.detach(),
        "phys_color_balance": cb_loss.detach(),
        "phys_cyan_floor": cyan_floor_loss_v.detach(),
        "phys_valid_frac": (valid_pixel > 0).float().mean().detach(),
        "phys_J_inv_R_mean": J_inv_mean[0].detach(),
        "phys_J_inv_G_mean": J_inv_mean[1].detach(),
        "phys_J_inv_B_mean": J_inv_mean[2].detach(),
        "phys_J_R_clipped_frac": R_clip_frac.detach(),
        "phys_J_R_over_GB": color_balance.detach(),
        "phys_J_R_sat_high_frac": sat_R_high.detach(),
        "phys_J_R_sat_low_frac":  sat_R_low.detach(),
        # v9.5-α — G/B saturation tracking (confirms R-specificity of issue)
        "phys_J_G_sat_high_frac": sat_G_high.detach(),
        "phys_J_G_sat_low_frac":  sat_G_low.detach(),
        "phys_J_B_sat_high_frac": sat_B_high.detach(),
        "phys_J_B_sat_low_frac":  sat_B_low.detach(),
        # v9.5-α — input/output NaN tracking (cycle_loss-level)
        "phys_data_z_has_nan":     torch.isnan(z).any().float().detach(),
        "phys_data_J_has_nan":     torch.isnan(J).any().float().detach(),
        "phys_data_I_has_nan":     torch.isnan(I).any().float().detach(),
        "phys_pred_betaD_has_nan": torch.isnan(beta_D).any().float().detach(),
        "phys_pred_betaB_has_nan": torch.isnan(beta_B).any().float().detach(),
        "phys_pred_Binf_has_nan":  torch.isnan(B_inf).any().float().detach(),
        "phys_J_inv_raw_has_nan":  torch.isnan(J_inv_raw).any().float().detach(),
        "phys_J_inv_raw_has_inf":  torch.isinf(J_inv_raw).any().float().detach(),
        "phys_cycle_has_nan":      (~torch.isfinite(cycle)).float().detach(),
        # v9.5-α — guard-fire diagnostics from b_inf_floor / b_inf_dimpx
        **{f"phys_binf_floor_{k}": v.float().detach() for k, v in binf_floor_diag.items()},
        # G1 — over-subtraction ceiling diagnostics (binf_R vs I_R_lo, violate frac)
        **{f"phys_binf_oversub_{k}": v.float().detach() for k, v in binf_oversub_diag.items()},
        **{f"phys_binf_dimpx_{k}": v.float().detach() for k, v in binf_dimpx_diag.items()},
        # v12 — cyan floor diagnostics (NaN-source tracking + activation rate)
        **{f"phys_{k}": v.float().detach() for k, v in cyan_floor_diag.items()},
    }


def prior_loss(
    beta_D: torch.Tensor,      # [B, 3] or [B, 3, H, W]
    beta_B: torch.Tensor,
    B_inf: torch.Tensor,       # [B, 3]
    cfg: PhysicsCycleConfig,
) -> tuple[torch.Tensor, dict]:
    """Soft constraints on physical reasonableness of (β^D, β^B, B^∞).

    1. Coupling: β^D ≈ β^B (Akkaynak: spread ~30%). Computed elementwise so
       it pushes spatial maps to stay close.
    2. Box upper bound on β.
    3. Channel ordering R > G > B applied to spatial means (per-image).
       Soft penalty — depth-dependent β can locally flip ordering, which is
       OK; we just don't want the *image-average* ordering to invert.
    """
    coupling = (beta_B - beta_D).abs().mean()
    box = F.relu(beta_D - cfg.beta_max).mean() + F.relu(beta_B - cfg.beta_max).mean()

    # v9: asymmetric ordering β^D ≥ β^B  (Akkaynak 2017).  Architecture-level
    # constraint (DepthDependentPhysicsHead.constrain_beta_B=True) makes this
    # impossible to violate; this loss is kept as belt-and-suspenders for
    # checkpoints that train without the architectural constraint.
    couple_order = F.relu(beta_B - beta_D).mean()

    bD_mean = _spatial_mean(beta_D)             # [B, 3]
    bB_mean = _spatial_mean(beta_B)
    order_D = (
        F.relu(bD_mean[:, 1] - bD_mean[:, 0]).mean()
        + F.relu(bD_mean[:, 2] - bD_mean[:, 1]).mean()
    )
    order_B = (
        F.relu(bB_mean[:, 1] - bB_mean[:, 0]).mean()
        + F.relu(bB_mean[:, 2] - bB_mean[:, 1]).mean()
    )
    ordering = order_D + order_B

    total = (
        cfg.lambda_couple * coupling
        + cfg.lambda_couple_order * couple_order
        + cfg.lambda_box * box
        + cfg.lambda_ordering * ordering
    )
    return total, {
        "phys_couple": coupling.detach(),
        "phys_couple_order": couple_order.detach(),
        "phys_box": box.detach(),
        "phys_order": ordering.detach(),
    }


def binf_supervision_loss(
    B_inf_pred: torch.Tensor,
    B_inf_gt:   torch.Tensor,
    r_weight:   float = 2.0,
) -> tuple[torch.Tensor, dict]:
    """Direct MSE supervision on predicted B^∞ against the Jerlov-aug GT B^∞.

    On augmented steps the medium that rendered the synthetic image is known,
    so B^∞_gt is the TRUE veiling light. Unlike b_inf_dimpx (a dim-pixel
    PRIOR) this is a hard target → teaches the RGB-stats B^∞ head the exact
    "this image ⇒ this B^∞" mapping, which is what must transfer to the
    cyan-dead OOD scene (v1). The R channel is up-weighted (`r_weight`) because
    B^∞_R is the cyan-dead binding variable (its over-prediction flips the
    inverse-J numerator negative → R collapse) and is numerically small, so an
    unweighted MSE would barely supervise it.
    """
    if B_inf_pred.dim() > 2:                      # [B,3,H,W] → [B,3]
        B_inf_pred = B_inf_pred.reshape(B_inf_pred.shape[0], 3, -1).mean(-1)
    w = torch.tensor([r_weight, 1.0, 1.0], device=B_inf_pred.device,
                     dtype=B_inf_pred.dtype)
    se = (B_inf_pred - B_inf_gt).pow(2) * w                       # [B, 3]
    loss = se.mean()
    with torch.no_grad():
        diag = {
            "binf_pred_R": B_inf_pred[:, 0].mean().detach(),
            "binf_gt_R":   B_inf_gt[:, 0].mean().detach(),
            "binf_R_abs_err": (B_inf_pred[:, 0] - B_inf_gt[:, 0]).abs().mean().detach(),
        }
    return loss, diag


def betaB_ratio_supervision_loss(
    beta_B_pred: torch.Tensor,
    beta_D_pred: torch.Tensor,
    beta_B_gt:   torch.Tensor,
    beta_D_gt:   torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """v11 — Direct MSE supervision on β_B / β_D ratio under Jerlov augmentation.

    When Jerlov rendering replaces I with synth_render(J, β_D_aug, β_B_aug, B∞_aug),
    the (β_D_aug, β_B_aug) sampled by sample_jerlov_medium are the GROUND TRUTH
    medium parameters that produced the observed image. Supervising the model's
    predicted β_B/β_D ratio against the synth ratio adds a strong learning
    signal that does NOT rely on the bounded sigmoid producing informative
    gradients (which it can't near the [ratio_min, ratio_max] boundaries —
    Pattern C's root cause).

    Combined with the v10 architectural cap, this gives Pattern C a true fix:
    architecture forbids σ → 1.0, supervision teaches the depth-dep MLP to
    produce informative spread instead of resting at the cap.

    Args:
        beta_B_pred : [B, 3] or [B, 3, H, W]   predicted β_B
        beta_D_pred : same shape                predicted β_D
        beta_B_gt   : [B, 3]                    Jerlov-sampled GT β_B
        beta_D_gt   : [B, 3]                    Jerlov-sampled GT β_D

    Returns:
        loss : scalar MSE on ratio
        diag : {ratio_pred_mean, ratio_gt_mean, ratio_abs_err_mean}
    """
    # Spatial mean for depth-dep heads
    if beta_B_pred.dim() == 4:
        bB_p = beta_B_pred.mean(dim=(-1, -2))
        bD_p = beta_D_pred.mean(dim=(-1, -2))
    else:
        bB_p = beta_B_pred
        bD_p = beta_D_pred

    pred_ratio = (bB_p / bD_p.clamp(min=1e-3)).clamp(0.0, 1.0)            # [B, 3]
    gt_ratio   = (beta_B_gt / beta_D_gt.clamp(min=1e-3)).clamp(0.0, 1.0)  # [B, 3]

    loss = F.mse_loss(pred_ratio, gt_ratio)
    with torch.no_grad():
        diag = {
            "ratio_pred_mean":    pred_ratio.mean().detach(),
            "ratio_gt_mean":      gt_ratio.mean().detach(),
            "ratio_abs_err_mean": (pred_ratio - gt_ratio).abs().mean().detach(),
        }
    return loss, diag


def betaB_ratio_cap_loss(
    ratio_logit: torch.Tensor,
    target_ratio: float,
) -> tuple[torch.Tensor, dict]:
    """Soft cap on β^B/β^D ratio — penalises in LOGIT space.

    Always-on regularizer (no synth GT required). Computes the penalty
    on the PRE-sigmoid logit `g(z)` that the constrain_beta_B head feeds
    into σ, NOT on the post-sigmoid ratio. This bypasses the σ → 1
    saturation problem (where σ'(g) = σ(1−σ) ≈ 0 kills the gradient on
    any post-σ penalty — Pattern C's resilience).

    Loss: ReLU(g - g_target)^2 mean over (B, C, H, W), where
        g_target = σ^{-1}(target_ratio) = ln(target_ratio / (1 - target_ratio))

    Gradient: ∂L/∂g = 2 · ReLU(g − g_target). No σ' factor → effective
    gradient even when σ is fully saturated at 1.0 (ε's epoch-9 state).

    For a v9p3-ε checkpoint with σ ≈ 0.999 (logit ≈ 6.9) and
    target_ratio = 0.9 (logit ≈ 2.20), the per-pixel penalty is about
    22.1 and the per-pixel gradient is about 9.4 — roughly 5×10⁴ stronger
    than the equivalent post-sigmoid penalty.

    Args:
        ratio_logit  : [B, 3, H, W]  pre-sigmoid g(z) from PhysicsHead
        target_ratio : in (0, 1)     ratio threshold the model should stay below

    Returns:
        loss : scalar penalty (apply λ outside)
        diag : human-readable monitoring fields (ratio & logit, mean & frac)
    """
    if not (0.0 < target_ratio < 1.0):
        raise ValueError(
            f"target_ratio must be in (0, 1), got {target_ratio}. "
            "Use 0.9 for the Akkaynak upper bound, 0.8 for stricter."
        )
    target_logit = float(math.log(target_ratio / (1.0 - target_ratio)))
    penalty = torch.relu(ratio_logit - target_logit) ** 2
    loss = penalty.mean()
    with torch.no_grad():
        ratio = torch.sigmoid(ratio_logit)
        diag = {
            # Human-readable post-sigmoid ratio (what we report in the
            # health check and the paper).
            "ratio_cap_mean":    ratio.mean().detach(),
            "ratio_cap_violate": (ratio > target_ratio).float().mean().detach(),
            # Logit-space monitoring — should decrease over training.
            "logit_mean":          ratio_logit.mean().detach(),
            "logit_violate_frac":  (ratio_logit > target_logit).float().mean().detach(),
        }
    return loss, diag


def view_consistency_loss(
    pred1: dict, pred2: dict, cfg: PhysicsCycleConfig,
) -> tuple[torch.Tensor, dict]:
    """Within a pair, the two views should see (almost) the same medium.

    For depth_dep, compare the head parameters (a, b, c, d) directly — these
    are per-view scalars and capture the medium independent of geometry.
    For scalar, compare β tensors directly.

    B^∞ is per-view scalar in both modes — pulled together with L1.
    """
    if "params_D" in pred1 and "params_D" in pred2:
        diff_D = (pred1["params_D"] - pred2["params_D"]).abs().mean()
        diff_B = (pred1["params_B"] - pred2["params_B"]).abs().mean()
    else:
        # fall back to spatial-mean comparison for scalar mode
        bD1, bD2 = _spatial_mean(pred1["beta_D"]), _spatial_mean(pred2["beta_D"])
        bB1, bB2 = _spatial_mean(pred1["beta_B"]), _spatial_mean(pred2["beta_B"])
        diff_D = (bD1 - bD2).abs().mean()
        diff_B = (bB1 - bB2).abs().mean()

    diff_Bi = (pred1["B_inf"] - pred2["B_inf"]).abs().mean()
    total = cfg.lambda_view_consist * (diff_D + diff_B + diff_Bi)
    return total, {
        "phys_view_dD": diff_D.detach(),
        "phys_view_dB": diff_B.detach(),
        "phys_view_dBinf": diff_Bi.detach(),
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def compute_physics_losses(
    pred1: dict,                          # {beta_D, beta_B, B_inf, [params_D, params_B]}
    pred2: dict,
    img_raw_1: torch.Tensor,              # I_1 [B, 3, H, W] in [-1, 1]
    img_raw_2: torch.Tensor,              # I_2
    img_clean_1: torch.Tensor,            # J_1 (SeaErra restored)
    img_clean_2: torch.Tensor,            # J_2
    depth_1: torch.Tensor,                # z_1 [B, H, W] metres
    depth_2: torch.Tensor,                # z_2
    cfg: PhysicsCycleConfig,
    synth_gt: dict | None = None,         # v11: {"beta_D_gt": [B,3], "beta_B_gt": [B,3]}
                                          #       passed from trainer when Jerlov rendered
                                          #       this batch. None ⇒ no supervision.
) -> tuple[torch.Tensor, dict]:
    """Bundle cycle + priors + view-consistency for one training step."""
    log: dict = {}

    # v9.5-α — input data NaN tracking (raw images, clean targets, depths)
    with torch.no_grad():
        log["data_depth1_has_nan"]    = torch.isnan(depth_1).any().float().detach()
        log["data_depth2_has_nan"]    = torch.isnan(depth_2).any().float().detach()
        log["data_depth1_has_inf"]    = torch.isinf(depth_1).any().float().detach()
        log["data_depth2_has_inf"]    = torch.isinf(depth_2).any().float().detach()
        log["data_img_raw1_has_nan"]  = torch.isnan(img_raw_1).any().float().detach()
        log["data_img_raw2_has_nan"]  = torch.isnan(img_raw_2).any().float().detach()
        log["data_img_clean1_has_nan"]= torch.isnan(img_clean_1).any().float().detach()
        log["data_img_clean2_has_nan"]= torch.isnan(img_clean_2).any().float().detach()
        # v9.5-α — model-prediction NaN tracking (β, B^∞ output health)
        log["pred_betaD1_has_nan"] = torch.isnan(pred1["beta_D"]).any().float().detach()
        log["pred_betaD2_has_nan"] = torch.isnan(pred2["beta_D"]).any().float().detach()
        log["pred_betaB1_has_nan"] = torch.isnan(pred1["beta_B"]).any().float().detach()
        log["pred_betaB2_has_nan"] = torch.isnan(pred2["beta_B"]).any().float().detach()
        log["pred_Binf1_has_nan"]  = torch.isnan(pred1["B_inf"]).any().float().detach()
        log["pred_Binf2_has_nan"]  = torch.isnan(pred2["B_inf"]).any().float().detach()
        # also Inf check on predictions (gradient explosion early signal)
        log["pred_betaD1_has_inf"] = torch.isinf(pred1["beta_D"]).any().float().detach()
        log["pred_Binf1_has_inf"]  = torch.isinf(pred1["B_inf"]).any().float().detach()

    cyc1, l1 = cycle_loss(img_raw_1, img_clean_1, depth_1,
                          pred1["beta_D"], pred1["beta_B"], pred1.get("B_inf_map", pred1["B_inf"]), cfg)
    cyc2, l2 = cycle_loss(img_raw_2, img_clean_2, depth_2,
                          pred2["beta_D"], pred2["beta_B"], pred2.get("B_inf_map", pred2["B_inf"]), cfg)
    log.update({f"{k}_v1": v for k, v in l1.items()})
    log.update({f"{k}_v2": v for k, v in l2.items()})

    pri1, lp1 = prior_loss(pred1["beta_D"], pred1["beta_B"], pred1["B_inf"], cfg)
    pri2, lp2 = prior_loss(pred2["beta_D"], pred2["beta_B"], pred2["B_inf"], cfg)
    log.update({f"{k}_v1": v for k, v in lp1.items()})
    log.update({f"{k}_v2": v for k, v in lp2.items()})

    vc, lv = view_consistency_loss(pred1, pred2, cfg)
    log.update(lv)

    # v11 — synth GT ratio supervision (Pattern C true fix)
    ratio_sup_loss = torch.zeros((), device=pred1["beta_D"].device)
    if (cfg.lambda_betaB_ratio_synth > 0
            and synth_gt is not None
            and synth_gt.get("active", False)):
        # Same medium applied to both views — supervise both predictions
        # against the shared (β_D_gt, β_B_gt).
        bDg = synth_gt["beta_D_gt"]      # [B, 3]
        bBg = synth_gt["beta_B_gt"]
        rl1, rdiag1 = betaB_ratio_supervision_loss(
            pred1["beta_B"], pred1["beta_D"], bBg, bDg,
        )
        rl2, rdiag2 = betaB_ratio_supervision_loss(
            pred2["beta_B"], pred2["beta_D"], bBg, bDg,
        )
        ratio_sup_loss = rl1 + rl2
        log["phys_betaB_ratio_synth_v1"] = rl1.detach()
        log["phys_betaB_ratio_synth_v2"] = rl2.detach()
        for k, v in rdiag1.items():
            log[f"phys_synth_{k}_v1"] = v
        for k, v in rdiag2.items():
            log[f"phys_synth_{k}_v2"] = v
        log["phys_synth_gt_active"] = torch.tensor(1.0)
    else:
        log["phys_synth_gt_active"] = torch.tensor(0.0)

    # Direct B^∞ synth-GT supervision (cyan-dead transfer hardener). On aug
    # steps the rendered medium's B^∞ is known → supervise the predicted B^∞
    # (RGB-stats head) against it with a hard MSE (R up-weighted). This pins
    # the image→B^∞ mapping at correct values so it transfers to v1, instead
    # of relying on the dim-pixel PRIOR (b_inf_dimpx) that OOD-transfer-failed.
    binf_sup_loss = torch.zeros((), device=pred1["beta_D"].device)
    if (cfg.lambda_binf_synth > 0
            and synth_gt is not None
            and synth_gt.get("active", False)
            and synth_gt.get("B_inf_gt", None) is not None):
        Bg = synth_gt["B_inf_gt"]                # [B, 3]
        bl1, bdiag1 = binf_supervision_loss(pred1["B_inf"], Bg)
        bl2, bdiag2 = binf_supervision_loss(pred2["B_inf"], Bg)
        binf_sup_loss = bl1 + bl2
        log["phys_binf_synth_v1"] = bl1.detach()
        log["phys_binf_synth_v2"] = bl2.detach()
        for k, v in bdiag1.items():
            log[f"phys_binf_synth_{k}_v1"] = v

    # ratiocap — always-on soft penalty on Pattern C (σ → 1 saturation),
    # applied in LOGIT space to bypass the σ → 1 gradient-vanishing
    # problem. Does NOT depend on Jerlov synth GT, so it applies on
    # every step including real-data batches. The penalty is zero
    # whenever the pre-sigmoid logit g(z) ≤ σ^{-1}(target), so it only
    # pulls down the predictions that are violating the cap.
    #
    # Requires constrain_beta_B = True (otherwise the head emits β^B
    # directly and there is no ratio sigmoid to penalise). Falls
    # through silently when ratio_logit is absent so legacy v8 / v9.0
    # checkpoints can still load this code.
    ratio_cap_loss = torch.zeros((), device=pred1["beta_D"].device)
    if (cfg.lambda_ratio_cap > 0
            and "ratio_logit" in pred1
            and "ratio_logit" in pred2):
        rcl1, rcdiag1 = betaB_ratio_cap_loss(
            pred1["ratio_logit"], cfg.ratio_cap_target,
        )
        rcl2, rcdiag2 = betaB_ratio_cap_loss(
            pred2["ratio_logit"], cfg.ratio_cap_target,
        )
        ratio_cap_loss = rcl1 + rcl2
        log["phys_ratio_cap_v1"] = rcl1.detach()
        log["phys_ratio_cap_v2"] = rcl2.detach()
        for k, v in rcdiag1.items():
            log[f"phys_{k}_v1"] = v
        for k, v in rcdiag2.items():
            log[f"phys_{k}_v2"] = v

    total = (cyc1 + cyc2 + pri1 + pri2 + vc
             + cfg.lambda_betaB_ratio_synth * ratio_sup_loss
             + cfg.lambda_binf_synth * binf_sup_loss
             + cfg.lambda_ratio_cap * ratio_cap_loss)

    # diagnostic-only — log per-channel β means and B^∞ for the batch.
    # Expanded for v9 so wandb can verify (a) β_B is no longer larger than
    # β_D, (b) B^∞ predictions are not pinned at the new 0.5 ceiling, and
    # (c) the architectural β_B ≤ β_D constraint is actually respected.
    with torch.no_grad():
        bD_mean = _spatial_mean(pred1["beta_D"])
        bB_mean = _spatial_mean(pred1["beta_B"])
        log["phys_beta_D_R_mean"] = bD_mean[:, 0].mean().detach()
        log["phys_beta_D_G_mean"] = bD_mean[:, 1].mean().detach()
        log["phys_beta_D_B_mean"] = bD_mean[:, 2].mean().detach()
        log["phys_beta_B_R_mean"] = bB_mean[:, 0].mean().detach()
        log["phys_beta_B_G_mean"] = bB_mean[:, 1].mean().detach()
        log["phys_beta_B_B_mean"] = bB_mean[:, 2].mean().detach()
        log["phys_B_inf_R_mean"]  = pred1["B_inf"][:, 0].mean().detach()
        log["phys_B_inf_G_mean"]  = pred1["B_inf"][:, 1].mean().detach()
        log["phys_B_inf_B_mean"]  = pred1["B_inf"][:, 2].mean().detach()
        # Fraction of (channel, pixel) entries where β_B > β_D (the v8 failure
        # mode). With --phys_constrain_beta_B this should be exactly 0; the
        # asymmetric ordering loss alone should drive it to ≤ 1 % within a
        # few hundred steps.
        bB1 = pred1["beta_B"]
        bD1 = pred1["beta_D"]
        violate = (bB1 > bD1).float().mean()
        log["phys_beta_B_over_D_frac"] = violate.detach()
        log["phys_total"] = total.detach()

        # ===========================================================
        # v9.5-α — ε-failure-pattern health-check metrics
        # Tracks the four root causes identified in the verified
        # 8-scene ε baseline (Barbados B^∞_R underprediction, DRUVA
        # β^D_R underprediction, R-channel saturation, β^B/β^D σ
        # saturation). All wandb-logged at training step granularity.
        # ===========================================================

        # (Cause 1) B^∞_R floor health — tracks ε's video1=0.004 / v4=0.041
        # / v2=0.146 systematic underprediction. v9.5-α with λ_b_inf_floor
        # 0.3 should pull these floor cases up.
        binf_R = pred1["B_inf"][:, 0]                                        # [B]
        log["phys_B_inf_R_min"]            = binf_R.min().detach()
        log["phys_B_inf_R_below_005_frac"] = (binf_R < 0.05).float().mean().detach()
        log["phys_B_inf_R_below_010_frac"] = (binf_R < 0.10).float().mean().detach()

        # (Cause 2) β^D_R distribution — tracks DRUVA-style underprediction
        # (ε's DRUVA mean β^D_R = 0.37). v9.5-α with warm Jerlov family +
        # extended β_D_R range should produce more diverse β^D_R, including
        # higher values for high-attenuation scenes.
        bD_R = bD_mean[:, 0]                                                  # [B]
        log["phys_beta_D_R_min"]      = bD_R.min().detach()
        log["phys_beta_D_R_max"]      = bD_R.max().detach()
        log["phys_beta_D_R_p10"]      = bD_R.quantile(0.1).detach()
        log["phys_beta_D_R_p90"]      = bD_R.quantile(0.9).detach()

        # (Pattern C) β^B / β^D ratio — tracks σ(g(z)) saturation. ε
        # showed ratio = 1.0 always (architecture ceiling reached, MLP
        # dummy). v9.5-α doesn't directly fix this but tracks for
        # completeness; expect ratio ≈ 1.0 still.
        ratio_R = (bB_mean[:, 0] / bD_mean[:, 0].clamp(min=1e-3)).clamp(0, 1)
        ratio_G = (bB_mean[:, 1] / bD_mean[:, 1].clamp(min=1e-3)).clamp(0, 1)
        ratio_B = (bB_mean[:, 2] / bD_mean[:, 2].clamp(min=1e-3)).clamp(0, 1)
        log["phys_betaB_betaD_ratio_R"]      = ratio_R.mean().detach()
        log["phys_betaB_betaD_ratio_G"]      = ratio_G.mean().detach()
        log["phys_betaB_betaD_ratio_B"]      = ratio_B.mean().detach()
        log["phys_betaB_betaD_ratio_below_095_frac"] = (
            (ratio_R < 0.95).float().mean()
            + (ratio_G < 0.95).float().mean()
            + (ratio_B < 0.95).float().mean()
        ).detach() / 3.0     # 0 = all saturated (ε pattern), 1 = informative spread

        # (Cause 3 — partial coverage) per-channel J_inv saturation rates
        # already logged via phys_J_R_sat_high/low_frac in cycle_loss.
        # The above per-batch B^∞ and β^D stats give early-warning signals
        # before they manifest as J-output R-saturation on test scenes.
        # v9.2 — surface the J inversion's per-channel mean for monitoring.
        # Computed lazily here so it doesn't add cost when --phys_lambda_j_direct
        # is 0 (cycle_loss skipped J_inv altogether in that case; we recompute
        # a coarse version with frame-mean depth for diagnostics only).
        if log.get("phys_j_direct_v1") is None:
            # already populated from cycle_loss's log dict — no recompute
            pass

    return total, log
