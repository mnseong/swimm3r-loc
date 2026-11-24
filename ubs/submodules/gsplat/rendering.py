import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Literal

from .cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
)
from .utils import depth_to_normal


def rasterization(
    means: Tensor,  # [N, 3]
    l_triagnles: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    betas: Tensor,  # [N]
    colors: Tensor,  # [(C,) N, D] or [(C,) N, K, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    # MAX projected-radius cap (px). Gaussians whose projected radius
    # exceeds this value are skipped in tile intersection for THIS frame
    # (radii zeroed → isect_tiles treats them as out-of-view). Bounds
    # rasterization memory/time against scene-specific blow-ups (a single
    # bloated Gaussian covering thousands of tiles → tile-overdraw cascade).
    # Per-frame skip is self-healing: the Gaussian still exists, contributes
    # in any view where its projection is below the cap, and natural
    # opacity decay + relocate eventually prunes it if it never recovers.
    # 0 = disabled (default). Recommended ~ min(W,H)//2.
    max_radius_px: float = 0.0,
    eps2d: float = 0.3,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal[
        "RGB", "Normal", "Diffuse", "Specular", "Depth", "EDepth", "RGB+D", "RGB+ED"
    ] = "RGB",
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    covars: Optional[Tensor] = None,
    # --- Medium parameters for underwater rendering ---
    medium_rgb: Optional[Tensor] = None,  # [C, H, W, 3]
    medium_bs: Optional[Tensor] = None,  # [C, H, W, 3]
    medium_attn: Optional[Tensor] = None,  # [C, H, W, 3]
    # --- Deconfounded primitive gradient sharpening (CUDA backward) ---
    # --- Dual-path backward geometry correction (sole novelty) ---
    grad_deconf_enabled: bool = False,
    grad_deconf_gamma: float = 1.0,
    grad_deconf_lambda_med: float = 1.0,
    grad_deconf_eps: float = 1e-12,
    grad_deconf_residual_eta: float = 0.25,
    grad_deconf_fade: float = 0.0,
    # Per-parameter-block switches. The stable default is projected-center
    # correction only; conic/opacity/medium paths are opt-in.
    grad_deconf_apply_means2d: bool = True,
    grad_deconf_apply_conic: bool = False,
    # ρ-based base-gradient reweighting. 0=off, 1=damp (base|=ρ|S|),
    # 2=amp (base=(2−ρ)|S|). Competing hypotheses for how medium-confused
    # primitives should be treated; default off = pure dual-path filter.
    grad_deconf_rho_mode: int = 0,
    # --- Phase 3 opacity soft deconf (backward only) ---
    grad_deconf_apply_opacity: bool = False,
    grad_deconf_gamma_opacity: float = 0.5,
    grad_deconf_eta_opacity: float = 0.15,
    # --- Phase 4 medium symmetric soft deconf (backward only) ---
    grad_deconf_gamma_medium: float = 0.0,
    grad_deconf_lambda_obj: float = 1.0,
    grad_deconf_medium_bs_floor: float = 0.3,
    grad_deconf_medium_attn_exact: bool = True,
) -> Tuple[Tensor, Tensor, Dict]:

    meta = {}

    N = means.shape[0]
    C = viewmats.shape[0]
    device = means.device
    assert means.shape == (N, 3), means.shape
    if covars is None:
        assert l_triagnles.shape == (N, 4), l_triagnles.shape
        assert scales.shape == (N, 3), scales.shape
    else:
        assert covars.shape == (N, 3, 3), covars.shape
        l_triagnles, scales = None, None
        # convert covars from 3x3 matrix to upper-triangular 6D vector
        tri_indices = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
        covars = covars[..., tri_indices[0], tri_indices[1]]
    assert opacities.shape == (N,), opacities.shape
    assert betas.shape == (N,), betas.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape
    assert render_mode in [
        "RGB",
        "Normal",
        "Diffuse",
        "Specular",
        "Depth",
        "EDepth",
        "RGB+D",
        "RGB+ED",
    ], render_mode

    # treat colors as post-activation values, should be in shape [N, D] or [C, N, D]
    assert (colors.dim() == 2 and colors.shape[0] == N) or (
        colors.dim() == 3 and colors.shape[:2] == (C, N)
    ), colors.shape

    camtoworlds = torch.inverse(viewmats)
    # dirs = means[primitive_ids, :] - camtoworlds[camera_ids, :3, 3]

    # Project Gaussians to 2D. Directly pass in {l_triagnles, scales} is faster than precomputing covars.
    proj_results = fully_fused_projection(
        means,
        covars,
        l_triagnles,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        sparse_grad=False,
        calc_compensations=(rasterize_mode == "antialiased"),
        ortho=False,
    )

    radii, means2d, depths, conics, compensations = proj_results
    opacities = opacities.repeat(C, 1)  # [C, N]
    betas = betas.repeat(C, 1)  # [C, N]
    camera_ids, primitive_ids = None, None

    if compensations is not None:
        opacities = opacities * compensations

    # MAX projected-radius cap: zero out radii of Gaussians that would
    # cover too many tiles (one bloated primitive can blow up
    # rasterization memory polynomially). isect_tiles treats radii==0 as
    # out-of-view → no tile intersections, no rasterization cost, no
    # gradient through this primitive for the current frame.
    # Keep the pre-cap radii: the soft radius penalty (train.py
    # lambda_max_radius) must see the TRUE projected radius — the capped
    # radii report 0 for the worst offenders, letting exactly the over-cap
    # splats escape the shrink loss.
    radii_uncapped = radii
    if max_radius_px > 0:
        _over = radii > float(max_radius_px)
        if _over.any():
            # SAFEGUARD: if the cap would zero >90% of currently visible
            # primitives in this frame, the rasterized output would have
            # no gradient (no Gaussian contributed) and the next backward
            # crashes with "element 0 of tensors does not require grad
            # and does not have a grad_fn". Detected on mast3r-sfm/video1
            # iter 1886. In that pathological frame we SKIP the cap and
            # let rasterization proceed with full radii; the cap protects
            # other frames as usual.
            _n_visible = int((radii > 0).sum().item())
            if (_n_visible > 0
                    and float(_over.sum().item()) / _n_visible < 0.9):
                radii = torch.where(
                    _over, torch.zeros_like(radii), radii
                )

    meta.update(
        {
            # global camera_ids
            "camera_ids": camera_ids,
            # local primitive_ids
            "primitive_ids": primitive_ids,
            "radii": radii,
            # true projected radii BEFORE the max_radius_px cull (== radii
            # when the cap is off or skipped by the >=90% safeguard)
            "radii_uncapped": radii_uncapped,
            "means2d": means2d,
            "depths": depths,
            "conics": conics,
            "opacities": opacities,
            "betas": betas,
        }
    )

    # Turn colors into [C, N, D] or [nnz, D] to pass into rasterize_to_pixels()
    if colors.dim() == 2:
        # Turn [N, D] into [C, N, D]
        colors = colors.expand(C, -1, -1)
    else:
        # colors is already [C, N, D]
        pass

    # Rasterize to pixels
    if render_mode in ["RGB+D", "RGB+ED"]:
        colors = torch.cat((colors, depths[..., None]), dim=-1)
        if backgrounds is not None:
            backgrounds = torch.cat(
                [backgrounds, torch.zeros(C, 1, device=backgrounds.device)], dim=-1
            )
    elif render_mode in ["Depth", "EDepth", "Normal"]:
        colors = depths[..., None]
        if backgrounds is not None:
            backgrounds = torch.zeros(C, 1, device=backgrounds.device)
    else:  # RGB
        pass

    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        n_cameras=C,
        camera_ids=camera_ids,
        primitive_ids=primitive_ids,
    )
    # print("rank", world_rank, "Before isect_offset_encode")
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    meta.update(
        {
            "tile_width": tile_width,
            "tile_height": tile_height,
            "tiles_per_gauss": tiles_per_gauss,
            "isect_ids": isect_ids,
            "flatten_ids": flatten_ids,
            "isect_offsets": isect_offsets,
            "width": width,
            "height": height,
            "tile_size": tile_size,
            "n_cameras": C,
        }
    )
    # print("rank", world_rank, "Before rasterize_to_pixels")
    render_medium = None
    render_clear = None
    # The CUDA dual-path correction rewrites v_means2d in the backward pass.
    # In chunked mode later channel chunks run the legacy path, so we only
    # enable the correction when the whole colour tensor fits in a single
    # chunk to keep the geometry gradient consistent across channels.
    grad_deconf_active = (
        bool(grad_deconf_enabled) and colors.shape[-1] <= channel_chunk
    )
    if colors.shape[-1] > channel_chunk:
        n_chunks = (colors.shape[-1] + channel_chunk - 1) // channel_chunk
        render_colors, render_alphas = [], []
        for i in range(n_chunks):
            colors_chunk = colors[..., i * channel_chunk : (i + 1) * channel_chunk]
            backgrounds_chunk = (
                backgrounds[..., i * channel_chunk : (i + 1) * channel_chunk]
                if backgrounds is not None
                else None
            )
            # Only pass medium params on the first chunk (medium is RGB-only).
            if i == 0:
                render_colors_, render_alphas_, render_medium, render_clear = rasterize_to_pixels(
                    means2d,
                    conics,
                    colors_chunk,
                    opacities,
                    betas,
                    width,
                    height,
                    tile_size,
                    isect_offsets,
                    flatten_ids,
                    backgrounds=backgrounds_chunk,
                    medium_rgb=medium_rgb,
                    medium_bs=medium_bs,
                    medium_attn=medium_attn,
                    depths=depths,
                    # Deconf is disabled in chunked mode by ``grad_deconf_active``.
                )
            else:
                render_colors_, render_alphas_, _, _ = rasterize_to_pixels(
                    means2d,
                    conics,
                    colors_chunk,
                    opacities,
                    betas,
                    width,
                    height,
                    tile_size,
                    isect_offsets,
                    flatten_ids,
                    backgrounds=backgrounds_chunk,
                )
            render_colors.append(render_colors_)
            render_alphas.append(render_alphas_)
        render_colors = torch.cat(render_colors, dim=-1)
        render_alphas = render_alphas[0]  # discard the rest
    else:
        render_colors, render_alphas, render_medium, render_clear = rasterize_to_pixels(
            means2d,
            conics,
            colors,
            opacities,
            betas,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
            medium_rgb=medium_rgb,
            medium_bs=medium_bs,
            medium_attn=medium_attn,
            depths=depths,
            grad_deconf_enabled=grad_deconf_active,
            grad_deconf_gamma=grad_deconf_gamma,
            grad_deconf_lambda_med=grad_deconf_lambda_med,
            grad_deconf_eps=grad_deconf_eps,
            grad_deconf_residual_eta=grad_deconf_residual_eta,
            grad_deconf_fade=grad_deconf_fade,
            grad_deconf_apply_means2d=grad_deconf_apply_means2d,
            grad_deconf_apply_conic=grad_deconf_apply_conic,
            grad_deconf_rho_mode=grad_deconf_rho_mode,
            grad_deconf_apply_opacity=grad_deconf_apply_opacity,
            grad_deconf_gamma_opacity=grad_deconf_gamma_opacity,
            grad_deconf_eta_opacity=grad_deconf_eta_opacity,
            grad_deconf_gamma_medium=grad_deconf_gamma_medium,
            grad_deconf_lambda_obj=grad_deconf_lambda_obj,
            grad_deconf_medium_bs_floor=grad_deconf_medium_bs_floor,
            grad_deconf_medium_attn_exact=grad_deconf_medium_attn_exact,
        )
    if render_mode in ["ED", "RGB+ED"]:
        # normalize the accumulated depth to get the expected depth
        render_colors = torch.cat(
            [
                render_colors[..., :-1],
                render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
            ],
            dim=-1,
        )
    if render_mode == "Normal":
        render_colors = depth_to_normal(render_colors, camtoworlds, Ks)
        render_colors = (render_colors + 1) / 2

    if render_medium is not None:
        meta["render_medium"] = render_medium
    if render_clear is not None:
        meta["render_clear"] = render_clear

    return render_colors, render_alphas, meta
