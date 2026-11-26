"""Physics coupling of the Gaussian map with the Swimm3R physics head.

Image formation (Akkaynak and Treibitz, the model of losses/physics_regression.py):
    I = J * exp(-beta_D(z) z) + B_inf * (1 - exp(-beta_B z)),
beta_D(z) = a exp(b z) + c exp(d z) per channel (the depth dependent head), beta_B a per channel constant
capped at beta_D (the Sea-Thru variant with --phys_seathru_betaB_le_betaD), B_inf per view, or with
--phys_binf_spatial a map B_inf(x, y) per view over the viewing directions (sigmoid of the per view logit plus a
quadratic in the image coordinates).

- load_physics builds the physics head of a checkpoint as Stage 1 does (swimm3r_colmap.py,
  compute_descatter_per_frame: the architecture read from the last layer, the Stage 1 defaults).
- medium() runs it on the encoder tokens and the depth of a view.
- invert() is the restored colour of the Stage 1 default path, J = (I - B_inf (1 - T_B)) / max(T_D, eps).
- underwater_colours() is the colour of each primitive seen from a camera at depth z of the primitive in that
  camera: J exp(-beta_D(z) z) + B_inf (1 - exp(-beta_B z)); the background of the underwater render is B_inf
  (open water), the background of the restored render black. With a B_inf map a primitive takes B_inf of the
  pixel it projects to and the background is the map.
"""

import torch
import torch.nn.functional as F

# Stage 1 defaults (swimm3r_colmap.py parse_args)
STAGE1 = dict(phys_head_hidden=256, phys_max_backscatter=0.7, phys_b_max=0.2, phys_beta_clamp_max=10.0,
              phys_betaB_ratio_min=0.0, phys_betaB_ratio_max=1.0, phys_constrain_beta_B=True, inverse_eps=0.01,
              phys_z_min=0.1, phys_z_max=50.0)


def load_physics(path, device, betaB_le_betaD=True, binf_spatial=False, **over):
    """The physics head of a checkpoint, built as Stage 1 builds it. Frozen, in eval mode. binf_spatial extends a
    unified head (18 outputs) to the B_inf map head with zero spatial rows, the same medium (a check of the map path)."""
    from losses.physics_regression import PhysicsRegressor
    cfg = {**STAGE1, **over}
    state = torch.load(path, map_location="cpu", weights_only=True)
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    mlp = [k for k in state if k.startswith("physics.mlp.") and k.endswith(".weight")]
    last = max(mlp, key=lambda k: int(k.split(".")[2])) if mlp else None
    out_dim = state[last].shape[0] if mlp else None
    if binf_spatial and out_dim == 18:
        wl, bl = state[last], state[last[:-len("weight")] + "bias"]
        state[last] = torch.cat([wl, wl.new_zeros(15, wl.shape[1])])
        state[last[:-len("weight")] + "bias"] = torch.cat([bl, bl.new_zeros(15)])
        out_dim = 33
    if out_dim not in (12, 15, 18, 24, 33):
        raise ValueError(f"unexpected physics MLP out_features {out_dim} in {path}")
    jerlov, seathru, unified = out_dim == 12, out_dim in (15, 18, 33), out_dim in (18, 33)
    phys = PhysicsRegressor(
        in_dim=1024, hidden=cfg["phys_head_hidden"], max_b=cfg["phys_max_backscatter"], kind="depth_dep",
        b_max=cfg["phys_b_max"], beta_max=cfg["phys_beta_clamp_max"],
        constrain_beta_B=cfg["phys_constrain_beta_B"] and not (jerlov or seathru),
        betaB_ratio_min=cfg["phys_betaB_ratio_min"], betaB_ratio_max=cfg["phys_betaB_ratio_max"],
        jerlov_unified_beta=jerlov, seathru_betaB_const=seathru, seathru_betaB_le_betaD=bool(betaB_le_betaD),
        unified_medium_head=unified, binf_spatial=out_dim == 33)
    phys.load_state_dict(state)
    phys = phys.to(device).eval()
    for p in phys.parameters():
        p.requires_grad_(False)
    attach_coupling(phys, eps=cfg["inverse_eps"], z_min=cfg["phys_z_min"], z_max=cfg["phys_z_max"])
    phys.coupling["out_dim"] = out_dim
    return phys


def attach_coupling(phys, eps=0.01, z_min=0.1, z_max=50.0):
    """Reads the beta treatment of a PhysicsRegressor (loaded, or the live one of a joint training) for beta_at()."""
    inner = phys.physics
    phys.coupling = dict(jerlov=bool(getattr(inner, "jerlov_unified_beta", False)),
                         seathru=bool(getattr(inner, "seathru_betaB_const", False)),
                         binf_spatial=bool(getattr(inner, "binf_spatial", False)),
                         le=bool(getattr(inner, "seathru_betaB_le_betaD", False)),
                         beta_max=float(getattr(inner, "beta_max", 10.0)), eps=eps, z_min=z_min, z_max=z_max)
    return phys


@torch.no_grad()
def medium(phys, feat, z, img_raw=None, beta_D_scale=None):
    """feat (B, N, 1024) encoder tokens, z (B, H, W) depth. Returns beta_D, beta_B (B, 3, H, W), B_inf (B, 3) and
    the per view parameters params_D, params_B (B, 4, 3), with a spatial head also B_inf_map (B, 3, H, W).
    beta_D_scale (3,) is the Stage 1 per channel scale."""
    return medium_live(phys, feat, z, img_raw, beta_D_scale)


def medium_live(phys, feat, z, img_raw=None, beta_D_scale=None):
    out = phys(feat, z=z, img_raw=img_raw)
    if beta_D_scale is not None:
        sc = torch.as_tensor(beta_D_scale, dtype=out["beta_D"].dtype, device=out["beta_D"].device).reshape(1, 3, 1, 1)
        out = dict(out, beta_D=out["beta_D"] * sc, beta_D_scale=sc.reshape(3))
    return out


def detach_medium(med):
    return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in med.items()}


def limit_medium(med, mode, weight=1.0):
    out = dict(med)
    if mode == "binf":
        for k in ("beta_D", "beta_B", "params_D", "params_B"):
            if k in out:
                out[k] = out[k].detach()
    if weight != 1.0:
        for k in ("beta_D", "beta_B", "params_D", "params_B", "B_inf", "B_inf_map"):
            if k in out and torch.is_tensor(out[k]) and out[k].requires_grad:
                out[k] = out[k].detach() + float(weight) * (out[k] - out[k].detach())
    return out


def trainable_medium(phys, mode):
    mlp = phys.physics.mlp
    if mode == "all":
        ps = list(mlp.parameters())
        for p in ps:
            p.requires_grad_(True)
        return ps
    if mode != "binf":
        raise ValueError(f"medium training mode {mode}: off, binf or all")
    last = mlp[-1]
    if last.out_features < 18:
        raise ValueError(f"the physics head has {last.out_features} outputs, B_inf rows need the unified head (18 or 33)")
    keep = torch.zeros(last.out_features, dtype=torch.bool)
    keep[15:] = True
    ps = [last.weight, last.bias]
    for p in ps:
        p.requires_grad_(True)
        p.register_hook(lambda g_, k=keep: g_ * k.to(g_.device).reshape((-1,) + (1,) * (g_.dim() - 1)).to(g_.dtype))
    return ps


def beta_at(phys, med, z, view=None):
    """beta_D, beta_B (3, N) of one medium at depths z (N,): the double exponential of params_D, the constant of
    params_B for the Sea-Thru head, the cap beta_B <= beta_D, the clamp of the head. med holds params_D (4, 3),
    params_B (4, 3) of one view (or the mean of several)."""
    c = phys.coupling
    zz = z.clamp(c["z_min"], c["z_max"])[None]
    a, b, cc, d = (med["params_D"][i][:, None] for i in range(4))
    bD = (a * torch.exp(b * zz) + cc * torch.exp(d * zz)).clamp(1e-3, c["beta_max"])
    if "beta_D_scale" in med:
        bD = bD * med["beta_D_scale"][:, None]
    if c["jerlov"]:
        bB = bD
    elif c["seathru"]:
        bB = med["params_B"][0][:, None].expand_as(bD).clamp(1e-3, c["beta_max"])
    else:
        a, b, cc, d = (med["params_B"][i][:, None] for i in range(4))
        bB = (a * torch.exp(b * zz) + cc * torch.exp(d * zz)).clamp(1e-3, c["beta_max"])
    if c["le"]:
        bB = torch.minimum(bB, bD)
    return bD, bB


def invert(phys, img01, z, med):
    """Restored colour of the Stage 1 default path: (I - B_inf (1 - T_B)) / max(T_D, eps), in [0, 1].
    img01 (B, 3, H, W), z (B, H, W), med from medium()."""
    zz = z[:, None]
    td = torch.exp(-med["beta_D"] * zz)
    tb = torch.exp(-med["beta_B"] * zz)
    binf = med["B_inf_map"] if "B_inf_map" in med else med["B_inf"][:, :, None, None]
    j = (img01 - binf * (1.0 - tb)) / td.clamp_min(phys.coupling["eps"])
    return j.clamp(0.0, 1.0)


def select(med, idx):
    """The media of the views idx (a batch of views kept)."""
    out = {k: med[k][idx] for k in ("beta_D", "beta_B", "B_inf", "B_inf_map", "params_D", "params_B") if k in med}
    if "beta_D_scale" in med:
        out["beta_D_scale"] = med["beta_D_scale"]
    return out


def view_medium(med, i):
    """The medium of one view i: params_D, params_B (4, 3), B_inf (3,) and B_inf_map (3, H, W) if the head has one."""
    out = dict(params_D=med["params_D"][i], params_B=med["params_B"][i], B_inf=med["B_inf"][i])
    if "B_inf_map" in med:
        out["B_inf_map"] = med["B_inf_map"][i]
    if "beta_D_scale" in med:
        out["beta_D_scale"] = med["beta_D_scale"]
    return out


def mean_medium(med, idx):
    """One medium for a novel view: the mean of the per view parameters, of B_inf and of the B_inf maps of the views idx."""
    out = dict(params_D=med["params_D"][idx].mean(0), params_B=med["params_B"][idx].mean(0), B_inf=med["B_inf"][idx].mean(0))
    if "B_inf_map" in med:
        out["B_inf_map"] = med["B_inf_map"][idx].mean(0)
    if "beta_D_scale" in med:
        out["beta_D_scale"] = med["beta_D_scale"]
    return out


def binf_at(med1, uv=None):
    """B_inf (N, 3) of primitives that project to the pixel coordinates uv (N, 2) of the camera (pixel centres at
    j + 0.5): the map of med1 sampled there (bilinear, border), else the one B_inf of the view."""
    if "B_inf_map" not in med1 or uv is None:
        return med1["B_inf"][None].expand(1 if uv is None else uv.shape[0], 3)
    mp = med1["B_inf_map"]
    h, w = mp.shape[-2:]
    grid = torch.stack([2.0 * uv[:, 0] / w - 1.0, 2.0 * uv[:, 1] / h - 1.0], dim=-1)[None, None].to(mp.dtype)
    return F.grid_sample(mp[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0].t()


def background(med1, h, w):
    """Open water behind the primitives (3, h, w): the B_inf map of the camera or its one B_inf."""
    if "B_inf_map" in med1:
        mp = med1["B_inf_map"]
        return mp if tuple(mp.shape[-2:]) == (h, w) else F.interpolate(mp[None], size=(h, w), mode="bilinear", align_corners=False)[0]
    return med1["B_inf"][:, None, None].expand(3, h, w)


def underwater_colours(phys, J, z, med1, uv=None):
    """Colour of every primitive (N, 3) seen at depth z (N,) through the medium med1 of the camera; uv (N, 2) the
    pixel it projects to, for the B_inf map."""
    bD, bB = beta_at(phys, med1, z.clamp_min(1e-4))
    zz = z.clamp_min(1e-4)[None]
    return (J.t() * torch.exp(-bD * zz) + binf_at(med1, uv).t() * (1.0 - torch.exp(-bB * zz))).t()


def veil_terms(phys, med1, alpha, j_pre, z_surf, z_ref):
    n, h, w = alpha.shape
    a = alpha[:, None].clamp(0.0, 1.0)

    def tw(z):
        ok = torch.isfinite(z) & (z > 1e-6)
        zz = torch.where(ok, z, torch.ones_like(z)).reshape(-1)
        bD, bB = beta_at(phys, med1, zz)
        td = torch.exp(-bD * zz[None]).reshape(3, n, h, w).permute(1, 0, 2, 3)
        wv = (1.0 - torch.exp(-bB * zz[None])).reshape(3, n, h, w).permute(1, 0, 2, 3)
        m = ok[:, None]
        return torch.where(m, td, torch.ones_like(td)), torch.where(m, wv, torch.zeros_like(wv))

    td_s, w_s = tw(z_surf)
    td_r, w_r = tw(z_ref)
    return dict(known=j_pre * td_s, gain=a * w_s + (1.0 - a) * w_r, slack=(1.0 - a) * td_r, veil_surface=w_s, veil_reference=w_r)


def veil_loss(med, raw, z):
    ok = (torch.isfinite(z) & (z > 1e-6))[:, None]
    zz = torch.where(ok, z[:, None], torch.ones_like(z[:, None]))
    td = torch.exp(-med["beta_D"].detach() * zz)
    wv = 1.0 - torch.exp(-med["beta_B"].detach() * zz)
    b = med["B_inf_map"] if "B_inf_map" in med else med["B_inf"][:, :, None, None]
    lo = b * wv
    r = torch.relu(lo - raw) + torch.relu(raw - lo - td)
    m = ok.to(raw.dtype)
    return (r.square() * m).sum() / (3.0 * m.sum().clamp_min(1.0))
