from typing import Callable, Optional, Tuple
import warnings

import torch
from torch import Tensor


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda


def cond_mean_convariance_opacity(
    means: Tensor,  # [N, 6]
    covars: Tensor,  # [N, 6, 6]
    opacities: Tensor,  # [N, 1]
    betas: Tensor,  # [N, 1]
    query: Tensor,  # [N, 3]
) -> Tuple[Tensor, Tensor, Tensor]:
    return _CondMeanConvarianceOpacity.apply(
        means.contiguous(),
        covars.contiguous(),
        opacities.contiguous(),
        betas.contiguous(),
        query.contiguous(),
    )


def l_triangle_to_rotmat(l_triangle: Tensor) -> Tensor:
    assert l_triangle.shape[1] == 3, l_triangle.shape
    return _LTriangleToRotmat.apply(l_triangle.contiguous())


def rot_scale_l_triangle_to_covar(
    rot: Tensor,  # [N, 3, 3]
    scale: Tensor,  # [N, D]
    l_triangle: Tensor,  # [N, D*(D-1)//2]
    rest_i: Tensor,
    rest_j: Tensor,
    spatial_block: bool = False,
) -> Tensor:
    # ensure contiguity for all CUDA calls
    return _RotScaleLTriangleToCovar.apply(
        rot.contiguous(),
        scale.contiguous(),
        l_triangle.contiguous(),
        rest_i,  # integer tensors, no grad needed
        rest_j,  # integer tensors, no grad needed
        spatial_block,
    )


def quat_scale_to_covar_preci(
    quats: Tensor,  # [N, 4],
    scales: Tensor,  # [N, 3],
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Converts quaternions and scales to covariance and precision matrices.

    Args:
        quats: Quaternions (No need to be normalized). [N, 4]
        scales: Scales. [N, 3]
        compute_covar: Whether to compute covariance matrices. Default: True. If False,
            the returned covariance matrices will be None.
        compute_preci: Whether to compute precision matrices. Default: True. If False,
            the returned precision matrices will be None.
        triu: If True, the return matrices will be upper triangular. Default: False.

    Returns:
        A tuple:

        - **Covariance matrices**. If `triu` is True the returned shape is [N, 6], otherwise [N, 3, 3].
        - **Precision matrices**. If `triu` is True the returned shape is [N, 6], otherwise [N, 3, 3].
    """
    assert quats.dim() == 2 and quats.size(1) == 4, quats.size()
    assert scales.dim() == 2 and scales.size(1) == 3, scales.size()
    quats = quats.contiguous()
    scales = scales.contiguous()
    covars, precis = _QuatScaleToCovarPreci.apply(
        quats, scales, compute_covar, compute_preci, triu
    )
    return covars if compute_covar else None, precis if compute_preci else None


def persp_proj(
    means: Tensor,  # [C, N, 3]
    covars: Tensor,  # [C, N, 3, 3]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """Perspective projection on Gaussians.
    DEPRECATED: please use `proj` with `ortho=False` instead.

    Args:
        means: Gaussian means. [C, N, 3]
        covars: Gaussian covariances. [C, N, 3, 3]
        Ks: Camera intrinsics. [C, 3, 3]
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **Projected means**. [C, N, 2]
        - **Projected covariances**. [C, N, 2, 2]
    """
    warnings.warn(
        "persp_proj is deprecated and will be removed in a future release. "
        "Use proj with ortho=False instead.",
        DeprecationWarning,
    )
    return proj(means, covars, Ks, width, height, ortho=False)


def proj(
    means: Tensor,  # [C, N, 3]
    covars: Tensor,  # [C, N, 3, 3]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    ortho: bool,
) -> Tuple[Tensor, Tensor]:
    """Projection of Gaussians (perspective or orthographic).

    Args:
        means: Gaussian means. [C, N, 3]
        covars: Gaussian covariances. [C, N, 3, 3]
        Ks: Camera intrinsics. [C, 3, 3]
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **Projected means**. [C, N, 2]
        - **Projected covariances**. [C, N, 2, 2]
    """
    C, N, _ = means.shape
    assert means.shape == (C, N, 3), means.size()
    assert covars.shape == (C, N, 3, 3), covars.size()
    assert Ks.shape == (C, 3, 3), Ks.size()
    means = means.contiguous()
    covars = covars.contiguous()
    Ks = Ks.contiguous()
    return _Proj.apply(means, covars, Ks, width, height, ortho)


def world_to_cam(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 3, 3]
    viewmats: Tensor,  # [C, 4, 4]
) -> Tuple[Tensor, Tensor]:
    """Transforms Gaussians from world to camera coordinate system.

    Args:
        means: Gaussian means. [N, 3]
        covars: Gaussian covariances. [N, 3, 3]
        viewmats: World-to-camera transformation matrices. [C, 4, 4]

    Returns:
        A tuple:

        - **Gaussian means in camera coordinate system**. [C, N, 3]
        - **Gaussian covariances in camera coordinate system**. [C, N, 3, 3]
    """
    C = viewmats.size(0)
    N = means.size(0)
    assert means.size() == (N, 3), means.size()
    assert covars.size() == (N, 3, 3), covars.size()
    assert viewmats.size() == (C, 4, 4), viewmats.size()
    means = means.contiguous()
    covars = covars.contiguous()
    viewmats = viewmats.contiguous()
    return _WorldToCam.apply(means, covars, viewmats)


def fully_fused_projection(
    means: Tensor,  # [N, 3]
    covars: Optional[Tensor],  # [N, 6] or None
    quats: Optional[Tensor],  # [N, 4] or None
    scales: Optional[Tensor],  # [N, 3] or None
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    sparse_grad: bool = False,
    calc_compensations: bool = False,
    ortho: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Projects Gaussians to 2D.

    This function fuse the process of computing covariances
    (:func:`quat_scale_to_covar_preci()`), transforming to camera space (:func:`world_to_cam()`),
    and projection (:func:`proj()`).

    .. note::

        During projection, we ignore the Gaussians that are outside of the camera frustum.
        So not all the elements in the output tensors are valid. The output `radii` could serve as
        an indicator, in which zero radii means the corresponding elements are invalid in
        the output tensors and will be ignored in the next rasterization process.

    .. note::

        This functions supports projecting Gaussians with either covariances or {quaternions, scales},
        which will be converted to covariances internally in a fused CUDA kernel. Either `covars` or
        {`quats`, `scales`} should be provided.

    Args:
        means: Gaussian means. [N, 3]
        covars: Gaussian covariances (flattened upper triangle). [N, 6] Optional.
        quats: Quaternions (No need to be normalized). [N, 4] Optional.
        scales: Scales. [N, 3] Optional.
        viewmats: Camera-to-world matrices. [C, 4, 4]
        Ks: Camera intrinsics. [C, 3, 3]
        width: Image width.
        height: Image height.
        eps2d: A epsilon added to the 2D covariance for numerical stability. Default: 0.3.
        near_plane: Near plane distance. Default: 0.01.
        far_plane: Far plane distance. Default: 1e10.
        radius_clip: Gaussians with projected radii smaller than this value will be ignored. Default: 0.0.
        sparse_grad: This is only effective when `packed` is True. If True, during backward the gradients
          of {`means`, `covars`, `quats`, `scales`} will be a sparse Tensor in COO layout. Default: False.
        calc_compensations: If True, a view-dependent opacity compensation factor will be computed, which
          is useful for anti-aliasing. Default: False.

    Returns:
        A tuple:
        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [C, N].
        - **means**. Projected Gaussian means in 2D. [C, N, 2]
        - **depths**. The z-depth of the projected Gaussians. [C, N]
        - **conics**. Inverse of the projected covariances. Return the flattend upper triangle with [C, N, 3]
        - **compensations**. The view-dependent opacity compensation factor. [C, N]
    """
    C = viewmats.size(0)
    N = means.size(0)
    assert means.size() == (N, 3), means.size()
    assert viewmats.size() == (C, 4, 4), viewmats.size()
    assert Ks.size() == (C, 3, 3), Ks.size()
    means = means.contiguous()
    if covars is not None:
        assert covars.size() == (N, 6), covars.size()
        covars = covars.contiguous()
    else:
        assert quats is not None, "covars or quats is required"
        assert scales is not None, "covars or scales is required"
        assert quats.size() == (N, 4), quats.size()
        assert scales.size() == (N, 3), scales.size()
        quats = quats.contiguous()
        scales = scales.contiguous()

    viewmats = viewmats.contiguous()
    Ks = Ks.contiguous()

    return _FullyFusedProjection.apply(
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        calc_compensations,
        ortho,
    )


@torch.no_grad()
def isect_tiles(
    means2d: Tensor,  # [C, N, 2] or [nnz, 2]
    radii: Tensor,  # [C, N] or [nnz]
    depths: Tensor,  # [C, N] or [nnz]
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
    n_cameras: Optional[int] = None,
    camera_ids: Optional[Tensor] = None,
    primitive_ids: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Maps projected Gaussians to intersecting tiles.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        radii: Maximum radii of the projected Gaussians. [C, N] if packed is False, [nnz] if packed is True.
        depths: Z-depth of the projected Gaussians. [C, N] if packed is False, [nnz] if packed is True.
        tile_size: Tile size.
        tile_width: Tile width.
        tile_height: Tile height.
        sort: If True, the returned intersections will be sorted by the intersection ids. Default: True.
        packed: If True, the input tensors are packed. Default: False.
        n_cameras: Number of cameras. Required if packed is True.
        camera_ids: The row indices of the projected Gaussians. Required if packed is True.
        primitive_ids: The column indices of the projected Gaussians. Required if packed is True.

    Returns:
        A tuple:

        - **Tiles per Gaussian**. The number of tiles intersected by each Gaussian.
          Int32 [C, N] if packed is False, Int32 [nnz] if packed is True.
        - **Intersection ids**. Each id is an 64-bit integer with the following
          information: camera_id (Xc bits) | tile_id (Xt bits) | depth (32 bits).
          Xc and Xt are the maximum number of bits required to represent the camera and
          tile ids, respectively. Int64 [n_isects]
        - **Flatten ids**. The global flatten indices in [C * N] or [nnz] (packed). [n_isects]
    """

    C, N, _ = means2d.shape
    assert means2d.shape == (C, N, 2), means2d.size()
    assert radii.shape == (C, N), radii.size()
    assert depths.shape == (C, N), depths.size()

    tiles_per_gauss, isect_ids, flatten_ids = _make_lazy_cuda_func("isect_tiles")(
        means2d.contiguous(),
        radii.contiguous(),
        depths.contiguous(),
        camera_ids,
        primitive_ids,
        C,
        tile_size,
        tile_width,
        tile_height,
        sort,
        True,  # DoubleBuffer: memory efficient radixsort
    )
    return tiles_per_gauss, isect_ids, flatten_ids


@torch.no_grad()
def isect_offset_encode(
    isect_ids: Tensor, n_cameras: int, tile_width: int, tile_height: int
) -> Tensor:
    """Encodes intersection ids to offsets.

    Args:
        isect_ids: Intersection ids. [n_isects]
        n_cameras: Number of cameras.
        tile_width: Tile width.
        tile_height: Tile height.

    Returns:
        Offsets. [C, tile_height, tile_width]
    """
    return _make_lazy_cuda_func("isect_offset_encode")(
        isect_ids.contiguous(), n_cameras, tile_width, tile_height
    )


def rasterize_to_pixels(
    means2d: Tensor,  # [C, N, 2] or [nnz, 2]
    conics: Tensor,  # [C, N, 3] or [nnz, 3]
    colors: Tensor,  # [C, N, channels] or [nnz, channels]
    opacities: Tensor,  # [C, N] or [nnz]
    betas: Tensor,  # [C, N] or [nnz]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [C, channels]
    masks: Optional[Tensor] = None,  # [C, tile_height, tile_width]
    # --- Medium parameters for underwater rendering ---
    medium_rgb: Optional[Tensor] = None,  # [C, H, W, 3]
    medium_bs: Optional[Tensor] = None,  # [C, H, W, 3]
    medium_attn: Optional[Tensor] = None,  # [C, H, W, 3]
    depths: Optional[Tensor] = None,  # [C, N]
    # --- Deconfounded gradient sharpening (CUDA backward) ---
    # --- Dual-path backward geometry correction (sole novelty) ---
    grad_deconf_enabled: bool = False,
    grad_deconf_gamma: float = 1.0,
    grad_deconf_lambda_med: float = 1.0,
    grad_deconf_eps: float = 1e-12,
    grad_deconf_residual_eta: float = 0.25,
    grad_deconf_fade: float = 0.0,
    grad_deconf_apply_means2d: bool = True,
    grad_deconf_apply_conic: bool = False,
    grad_deconf_rho_mode: int = 0,
    # --- Phase 3 opacity soft correction (backward only) ---
    grad_deconf_apply_opacity: bool = False,
    grad_deconf_gamma_opacity: float = 0.5,
    grad_deconf_eta_opacity: float = 0.15,
    # --- Phase 4 medium symmetric correction (backward only) ---
    grad_deconf_gamma_medium: float = 0.0,
    grad_deconf_lambda_obj: float = 1.0,
    grad_deconf_medium_bs_floor: float = 0.3,
    grad_deconf_medium_attn_exact: bool = True,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Rasterizes Gaussians to pixels.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        conics: Inverse of the projected covariances with only upper triangle values. [C, N, 3] if packed is False, [nnz, 3] if packed is True.
        colors: Gaussian colors or ND features. [C, N, channels] if packed is False, [nnz, channels] if packed is True.
        opacities: Gaussian opacities that support per-view values. [C, N] if packed is False, [nnz] if packed is True.
        betas: Gaussian sharpness that support per-view values. [C, N] if packed is False, [nnz] if packed is True.
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [C, tile_height, tile_width]
        flatten_ids: The global flatten indices in [C * N] or [nnz] from  `isect_tiles()`. [n_isects]
        backgrounds: Background colors. [C, channels]. Default: None.
        masks: Optional tile mask to skip rendering GS to masked tiles. [C, tile_height, tile_width]. Default: None.
        medium_rgb: Per-pixel medium scattering color. [C, H, W, 3]. Default: None.
        medium_bs: Per-pixel medium backscatter coefficient. [C, H, W, 3]. Default: None.
        medium_attn: Per-pixel medium attenuation coefficient. [C, H, W, 3]. Default: None.
        depths: Per-gaussian depths for medium computation. [C, N]. Default: None.

    Returns:
        A tuple:

        - **Rendered colors**. [C, image_height, image_width, channels]
        - **Rendered alphas**. [C, image_height, image_width, 1]
        - **Rendered medium**. [C, image_height, image_width, 3] or None if medium params not provided.
    """

    C = isect_offsets.size(0)
    device = means2d.device
    N = means2d.size(1)
    assert means2d.shape == (C, N, 2), means2d.shape
    assert conics.shape == (C, N, 3), conics.shape
    assert colors.shape[:2] == (C, N), colors.shape
    assert opacities.shape == (C, N), opacities.shape
    assert betas.shape == (C, N), betas.shape
    if backgrounds is not None:
        assert backgrounds.shape == (C, colors.shape[-1]), backgrounds.shape
        backgrounds = backgrounds.contiguous()
    if masks is not None:
        assert masks.shape == isect_offsets.shape, masks.shape
        masks = masks.contiguous()
    if medium_rgb is not None:
        assert medium_rgb.shape == (C, image_height, image_width, 3), medium_rgb.shape
        medium_rgb = medium_rgb.contiguous()
    if medium_bs is not None:
        assert medium_bs.shape == (C, image_height, image_width, 3), medium_bs.shape
        medium_bs = medium_bs.contiguous()
    if medium_attn is not None:
        assert medium_attn.shape == (C, image_height, image_width, 3), medium_attn.shape
        medium_attn = medium_attn.contiguous()
    if depths is not None:
        assert depths.shape == (C, N), depths.shape
        depths = depths.contiguous()

    # Pad the channels to the nearest supported number if necessary
    channels = colors.shape[-1]
    if channels > 513 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (
        1,
        2,
        3,
        4,
        5,
        8,
        9,
        16,
        17,
        32,
        33,
        64,
        65,
        128,
        129,
        256,
        257,
        512,
        513,
    ):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [
                colors,
                torch.zeros(*colors.shape[:-1], padded_channels, device=device),
            ],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0

    tile_height, tile_width = isect_offsets.shape[1:3]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    render_colors, render_alphas, render_medium, render_clear = _RasterizeToPixels.apply(
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        betas.contiguous(),
        backgrounds,
        masks,
        medium_rgb,
        medium_bs,
        medium_attn,
        depths,
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        bool(grad_deconf_enabled),
        float(grad_deconf_gamma),
        float(grad_deconf_lambda_med),
        float(grad_deconf_eps),
        float(grad_deconf_residual_eta),
        float(grad_deconf_fade),
        bool(grad_deconf_apply_means2d),
        bool(grad_deconf_apply_conic),
        int(grad_deconf_rho_mode),
        bool(grad_deconf_apply_opacity),
        float(grad_deconf_gamma_opacity),
        float(grad_deconf_eta_opacity),
        float(grad_deconf_gamma_medium),
        float(grad_deconf_lambda_obj),
        float(grad_deconf_medium_bs_floor),
        bool(grad_deconf_medium_attn_exact),
    )

    if padded_channels > 0:
        render_colors = render_colors[..., :-padded_channels]
    return render_colors, render_alphas, render_medium, render_clear


@torch.no_grad()
def rasterize_to_indices_in_range(
    range_start: int,
    range_end: int,
    transmittances: Tensor,  # [C, image_height, image_width]
    means2d: Tensor,  # [C, N, 2]
    conics: Tensor,  # [C, N, 3]
    opacities: Tensor,  # [C, N]
    betas: Tensor,  # [C, N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Rasterizes a batch of Gaussians to images but only returns the indices.

    .. note::

        This function supports iterative rasterization, in which each call of this function
        will rasterize a batch of Gaussians from near to far, defined by `[range_start, range_end)`.
        If a one-step full rasterization is desired, set `range_start` to 0 and `range_end` to a really
        large number, e.g, 1e10.

    Args:
        range_start: The start batch of Gaussians to be rasterized (inclusive).
        range_end: The end batch of Gaussians to be rasterized (exclusive).
        transmittances: Currently transmittances. [C, image_height, image_width]
        means2d: Projected Gaussian means. [C, N, 2]
        conics: Inverse of the projected covariances with only upper triangle values. [C, N, 3]
        opacities: Gaussian opacities that support per-view values. [C, N]
        betas: Gaussian sharpness that support per-view values. [C, N]
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [C, tile_height, tile_width]
        flatten_ids: The global flatten indices in [C * N] from  `isect_tiles()`. [n_isects]

    Returns:
        A tuple:

        - **Gaussian ids**. Gaussian ids for the pixel intersection. A flattened list of shape [M].
        - **Pixel ids**. pixel indices (row-major). A flattened list of shape [M].
        - **Camera ids**. Camera indices. A flattened list of shape [M].
    """

    C, N, _ = means2d.shape
    assert conics.shape == (C, N, 3), conics.shape
    assert opacities.shape == (C, N), opacities.shape
    assert betas.shape == (C, N), betas.shape
    assert isect_offsets.shape[0] == C, isect_offsets.shape

    tile_height, tile_width = isect_offsets.shape[1:3]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    out_gauss_ids, out_indices = _make_lazy_cuda_func("rasterize_to_indices_in_range")(
        range_start,
        range_end,
        transmittances.contiguous(),
        means2d.contiguous(),
        conics.contiguous(),
        opacities.contiguous(),
        betas.contiguous(),
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
    )
    out_pixel_ids = out_indices % (image_width * image_height)
    out_camera_ids = out_indices // (image_width * image_height)
    return out_gauss_ids, out_pixel_ids, out_camera_ids


class _CondMeanConvarianceOpacity(torch.autograd.Function):
    """Condition mean, covariance and opacity on a query."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [N, 6]
        covars: Tensor,  # [N, 6, 6]
        opacities: Tensor,  # [N, 1]
        betas: Tensor,  # [N, 1]
        query: Tensor,  # [N, 3]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        cond_means, cond_covars, cond_opacities = _make_lazy_cuda_func(
            "cond_mean_convariance_opacity_fwd"
        )(means, covars, opacities, betas, query)
        ctx.save_for_backward(means, covars, opacities, betas, query)
        return cond_means, cond_covars, cond_opacities

    @staticmethod
    def backward(
        ctx,
        v_cond_means: Tensor,
        v_cond_covars: Tensor,
        v_cond_opacities: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, None]:
        means, covars, opacities, betas, query = ctx.saved_tensors
        v_means, v_covars, v_opacities, v_betas = _make_lazy_cuda_func(
            "cond_mean_convariance_opacity_bwd"
        )(
            means.contiguous(),
            covars.contiguous(),
            opacities.contiguous(),
            betas.contiguous(),
            query.contiguous(),
            v_cond_means.contiguous(),
            v_cond_covars.contiguous(),
            v_cond_opacities.contiguous(),
        )
        return v_means, v_covars, v_opacities, v_betas, None


class _LTriangleToRotmat(torch.autograd.Function):
    """Converts a lower triangular matrix to a rotation matrix."""

    @staticmethod
    def forward(ctx, l_triangle: Tensor) -> Tensor:
        rots = _make_lazy_cuda_func("l_triangle_to_rotmat_fwd")(l_triangle)
        ctx.save_for_backward(l_triangle)
        return rots

    @staticmethod
    def backward(ctx, v_rots: Tensor):
        (l_triangle,) = ctx.saved_tensors
        v_l_triangle = _make_lazy_cuda_func("l_triangle_to_rotmat_bwd")(
            l_triangle,
            v_rots.contiguous(),
        )
        return v_l_triangle


class _RotScaleLTriangleToCovar(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        rot: Tensor,
        scale: Tensor,
        l_triangle: Tensor,
        rest_i: Tensor,
        rest_j: Tensor,
        spatial_block: bool,
    ) -> Tensor:
        # call into the fused CUDA forward kernel
        covar = _make_lazy_cuda_func("rot_scale_l_triangle_to_covar_fwd")(
            rot, scale, l_triangle, rest_i, rest_j, spatial_block
        )
        # save inputs for backward
        ctx.save_for_backward(rot, scale, l_triangle, rest_i, rest_j)
        ctx.spatial_block = spatial_block
        return covar

    @staticmethod
    def backward(ctx, v_covar: Tensor):
        # unpack saved
        rot, scale, l_triangle, rest_i, rest_j = ctx.saved_tensors
        spatial_block = ctx.spatial_block

        # if no one needs gradients, early exit
        if not any(ctx.needs_input_grad[:3]):
            return (None, None, None, None, None, None)

        # call into the fused CUDA backward kernel
        v_rot, v_scale, v_l_triangle = _make_lazy_cuda_func(
            "rot_scale_l_triangle_to_covar_bwd"
        )(
            rot.contiguous(),
            scale.contiguous(),
            l_triangle.contiguous(),
            rest_i,
            rest_j,
            spatial_block,
            v_covar.contiguous(),
        )

        # return one entry per forward arg
        return (
            v_rot,  # grad w.r.t. rot
            v_scale,  # grad w.r.t. scale
            v_l_triangle,  # grad w.r.t. l_triangle
            None,  # rest_i (int, no grad)
            None,  # rest_j (int, no grad)
            None,  # spatial_block (bool, no grad)
        )


class _QuatScaleToCovarPreci(torch.autograd.Function):
    """Converts quaternions and scales to covariance and precision matrices."""

    @staticmethod
    def forward(
        ctx,
        quats: Tensor,  # [N, 4],
        scales: Tensor,  # [N, 3],
        compute_covar: bool = True,
        compute_preci: bool = True,
        triu: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        covars, precis = _make_lazy_cuda_func("quat_scale_to_covar_preci_fwd")(
            quats, scales, compute_covar, compute_preci, triu
        )
        ctx.save_for_backward(quats, scales)
        ctx.compute_covar = compute_covar
        ctx.compute_preci = compute_preci
        ctx.triu = triu
        return covars, precis

    @staticmethod
    def backward(ctx, v_covars: Tensor, v_precis: Tensor):
        quats, scales = ctx.saved_tensors
        compute_covar = ctx.compute_covar
        compute_preci = ctx.compute_preci
        triu = ctx.triu
        if compute_covar and v_covars.is_sparse:
            v_covars = v_covars.to_dense()
        if compute_preci and v_precis.is_sparse:
            v_precis = v_precis.to_dense()
        v_quats, v_scales = _make_lazy_cuda_func("quat_scale_to_covar_preci_bwd")(
            quats,
            scales,
            v_covars.contiguous() if compute_covar else None,
            v_precis.contiguous() if compute_preci else None,
            triu,
        )
        return v_quats, v_scales, None, None, None


class _Proj(torch.autograd.Function):
    """Perspective fully_fused_projection on Gaussians."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [C, N, 3]
        covars: Tensor,  # [C, N, 3, 3]
        Ks: Tensor,  # [C, 3, 3]
        width: int,
        height: int,
        ortho: bool,
    ) -> Tuple[Tensor, Tensor]:
        means2d, covars2d = _make_lazy_cuda_func("proj_fwd")(
            means, covars, Ks, width, height, ortho
        )
        ctx.save_for_backward(means, covars, Ks)
        ctx.width = width
        ctx.height = height
        ctx.ortho = ortho
        return means2d, covars2d

    @staticmethod
    def backward(ctx, v_means2d: Tensor, v_covars2d: Tensor):
        means, covars, Ks = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        ortho = ctx.ortho
        v_means, v_covars = _make_lazy_cuda_func("proj_bwd")(
            means,
            covars,
            Ks,
            width,
            height,
            ortho,
            v_means2d.contiguous(),
            v_covars2d.contiguous(),
        )
        return v_means, v_covars, None, None, None, None


class _WorldToCam(torch.autograd.Function):
    """Transforms Gaussians from world to camera space."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [N, 3]
        covars: Tensor,  # [N, 3, 3]
        viewmats: Tensor,  # [C, 4, 4]
    ) -> Tuple[Tensor, Tensor]:
        means_c, covars_c = _make_lazy_cuda_func("world_to_cam_fwd")(
            means, covars, viewmats
        )
        ctx.save_for_backward(means, covars, viewmats)
        return means_c, covars_c

    @staticmethod
    def backward(ctx, v_means_c: Tensor, v_covars_c: Tensor):
        means, covars, viewmats = ctx.saved_tensors
        v_means, v_covars, v_viewmats = _make_lazy_cuda_func("world_to_cam_bwd")(
            means,
            covars,
            viewmats,
            v_means_c.contiguous(),
            v_covars_c.contiguous(),
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[2],
        )
        if not ctx.needs_input_grad[0]:
            v_means = None
        if not ctx.needs_input_grad[1]:
            v_covars = None
        if not ctx.needs_input_grad[2]:
            v_viewmats = None
        return v_means, v_covars, v_viewmats


class _FullyFusedProjection(torch.autograd.Function):
    """Projects Gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [N, 3]
        covars: Tensor,  # [N, 6] or None
        quats: Tensor,  # [N, 4] or None
        scales: Tensor,  # [N, 3] or None
        viewmats: Tensor,  # [C, 4, 4]
        Ks: Tensor,  # [C, 3, 3]
        width: int,
        height: int,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
        calc_compensations: bool,
        ortho: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # "covars" and {"quats", "scales"} are mutually exclusive
        radii, means2d, depths, conics, compensations = _make_lazy_cuda_func(
            "fully_fused_projection_fwd"
        )(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
            ortho,
        )
        if not calc_compensations:
            compensations = None
        ctx.save_for_backward(
            means, covars, quats, scales, viewmats, Ks, radii, conics, compensations
        )
        ctx.width = width
        ctx.height = height
        ctx.eps2d = eps2d
        ctx.ortho = ortho

        return radii, means2d, depths, conics, compensations

    @staticmethod
    def backward(ctx, v_radii, v_means2d, v_depths, v_conics, v_compensations):
        (
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            radii,
            conics,
            compensations,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        eps2d = ctx.eps2d
        ortho = ctx.ortho
        if v_compensations is not None:
            v_compensations = v_compensations.contiguous()
        v_means, v_covars, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            "fully_fused_projection_bwd"
        )(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            ortho,
            radii,
            conics,
            compensations,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
            v_compensations,
            ctx.needs_input_grad[4],  # viewmats_requires_grad
        )
        if not ctx.needs_input_grad[0]:
            v_means = None
        if not ctx.needs_input_grad[1]:
            v_covars = None
        if not ctx.needs_input_grad[2]:
            v_quats = None
        if not ctx.needs_input_grad[3]:
            v_scales = None
        if not ctx.needs_input_grad[4]:
            v_viewmats = None
        return (
            v_means,
            v_covars,
            v_quats,
            v_scales,
            v_viewmats,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# Module-level side channel for per-primitive diagnostics.
# _RasterizeToPixels.backward overwrites this after every backward pass; the
# training loop can read the latest values for wandb / ρ-guided MCMC.
class _RasterizeToPixels(torch.autograd.Function):
    """Rasterize gaussians.

    Side-channel attributes written in backward each time the dual-path is
    active. All three are detached [C, N] tensors, ``None`` when the
    dual-path was not active for the last call. Consumers (BetaModel) EMA
    these into persistent buffers; nothing else in the forward chain
    depends on them, so they are safe to read any time after
    ``loss.backward()``.

    * ``last_rho_per_primitive`` — identifiability score ρ ∈ [0, 1].
      Used by ρ-guided MCMC relocation and wandb histograms.
    * ``last_cos_sm_per_primitive`` — cos(S, M) ∈ [-1, 1], direction
      confounding of the medium-path gradient relative to object-path.
      Gradient-flow validity diagnostic.
    * ``last_magratio_per_primitive`` — |M|_2 / |S|_2 magnitude ratio.
      Gradient-flow magnitude diagnostic.
    * ``last_grad_vis_per_primitive`` — α-weighted mean of T·exp(-β_D·d)
      over contributing pixels; effective gradient multiplier after the
      Beer-Lambert + occlusion chain. Low values ⇒ transmittance-starved
      primitive (Mode #1 compound diagnostic). Populated when medium is
      active, independent of the dual-path correction being enabled.
    * ``last_T_per_view`` — α-weighted T only (occlusion component).
      var(T) across views reflects primitive-ordering instability — a
      generic 3DGS failure mode present in any scene.
    * ``last_exp_per_view`` — α-weighted mean(exp(-β_D·d)) only (medium
      component). var(exp) across views reflects Beer-Lambert
      amplification of per-view depth drift — the underwater-specific
      failure mode. The ratio var(exp)/var(T) quantifies how much
      underwater exacerbates view-dependent gradient variance beyond
      baseline occlusion-reordering effects.
    """

    # Class-level side channels (not autograd-tracked).
    last_rho_per_primitive: Optional[Tensor] = None
    last_cos_sm_per_primitive: Optional[Tensor] = None
    last_magratio_per_primitive: Optional[Tensor] = None
    last_grad_vis_per_primitive: Optional[Tensor] = None
    last_T_per_view: Optional[Tensor] = None
    last_exp_per_view: Optional[Tensor] = None

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,  # [C, N, 2]
        conics: Tensor,  # [C, N, 3]
        colors: Tensor,  # [C, N, D]
        opacities: Tensor,  # [C, N]
        betas: Tensor,  # [C, N]
        backgrounds: Tensor,  # [C, D], Optional
        masks: Tensor,  # [C, tile_height, tile_width], Optional
        # --- Medium parameters (optional) ---
        medium_rgb: Tensor,  # [C, H, W, 3], Optional
        medium_bs: Tensor,  # [C, H, W, 3], Optional
        medium_attn: Tensor,  # [C, H, W, 3], Optional
        depths: Tensor,  # [C, N], Optional
        # ---
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,  # [C, tile_height, tile_width]
        flatten_ids: Tensor,  # [n_isects]
        # --- Deconfounded sharpening config (backward only) ---
        # --- Dual-path backward geometry correction config (backward only) ---
        grad_deconf_enabled: bool = False,
        grad_deconf_gamma: float = 1.0,
        grad_deconf_lambda_med: float = 1.0,
        grad_deconf_eps: float = 1e-12,
        grad_deconf_residual_eta: float = 0.25,
        grad_deconf_fade: float = 0.0,
        grad_deconf_apply_means2d: bool = True,
        grad_deconf_apply_conic: bool = False,
        grad_deconf_rho_mode: int = 0,
        # --- Phase 3 opacity soft correction (backward only) ---
        grad_deconf_apply_opacity: bool = False,
        grad_deconf_gamma_opacity: float = 0.5,
        grad_deconf_eta_opacity: float = 0.15,
        # --- Phase 4 medium symmetric correction (backward only) ---
        grad_deconf_gamma_medium: float = 0.0,
        grad_deconf_lambda_obj: float = 1.0,
        grad_deconf_medium_bs_floor: float = 0.3,
        grad_deconf_medium_attn_exact: bool = True,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        render_colors, render_alphas, last_ids, render_medium, render_clear = _make_lazy_cuda_func(
            "rasterize_to_pixels_fwd"
        )(
            means2d,
            conics,
            colors,
            opacities,
            betas,
            backgrounds,
            masks,
            # medium parameters
            medium_rgb,
            medium_bs,
            medium_attn,
            depths,
            # image size
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
        )

        ctx.save_for_backward(
            means2d,
            conics,
            colors,
            opacities,
            betas,
            backgrounds,
            masks,
            medium_rgb,
            medium_bs,
            medium_attn,
            depths,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.has_medium = medium_rgb is not None
        # Dual-path backward config (consumed only by the backward pass).
        ctx.grad_deconf_enabled = bool(grad_deconf_enabled)
        ctx.grad_deconf_gamma = float(grad_deconf_gamma)
        ctx.grad_deconf_lambda_med = float(grad_deconf_lambda_med)
        ctx.grad_deconf_eps = float(grad_deconf_eps)
        ctx.grad_deconf_residual_eta = float(grad_deconf_residual_eta)
        ctx.grad_deconf_fade = float(grad_deconf_fade)
        ctx.grad_deconf_apply_means2d = bool(grad_deconf_apply_means2d)
        ctx.grad_deconf_apply_conic   = bool(grad_deconf_apply_conic)
        ctx.grad_deconf_rho_mode      = int(grad_deconf_rho_mode)
        # Phase 3 opacity soft correction (consumed only by backward).
        ctx.grad_deconf_apply_opacity = bool(grad_deconf_apply_opacity)
        ctx.grad_deconf_gamma_opacity = float(grad_deconf_gamma_opacity)
        ctx.grad_deconf_eta_opacity   = float(grad_deconf_eta_opacity)
        # Phase 4 medium symmetric correction (consumed only by backward).
        ctx.grad_deconf_gamma_medium  = float(grad_deconf_gamma_medium)
        ctx.grad_deconf_lambda_obj    = float(grad_deconf_lambda_obj)
        ctx.grad_deconf_medium_bs_floor = float(grad_deconf_medium_bs_floor)
        ctx.grad_deconf_medium_attn_exact = bool(grad_deconf_medium_attn_exact)

        # double to float
        render_alphas = render_alphas.float()
        # Return None for render_medium/render_clear when medium is not used
        # (CUDA returns empty({0}) tensor when medium is disabled)
        if render_medium.dim() < 4:
            render_medium = None
        if render_clear.dim() < 4:
            render_clear = None
        return render_colors, render_alphas, render_medium, render_clear

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,  # [C, H, W, 3]
        v_render_alphas: Tensor,  # [C, H, W, 1]
        v_render_medium: Optional[Tensor],  # [C, H, W, 3] or None
        v_render_clear: Optional[Tensor],  # [C, H, W, 3] or None — not used
    ):
        (
            means2d,
            conics,
            colors,
            opacities,
            betas,
            backgrounds,
            masks,
            medium_rgb,
            medium_bs,
            medium_attn,
            depths,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size

        (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_betas,
            v_medium_rgb,
            v_medium_bs,
            v_medium_attn,
            rho_per_primitive,
            cos_sm_per_primitive,
            magratio_per_primitive,
            grad_vis_per_primitive,
            T_per_view,
            exp_per_view,
        ) = _make_lazy_cuda_func(
            "rasterize_to_pixels_bwd"
        )(
            means2d,
            conics,
            colors,
            opacities,
            betas,
            backgrounds,
            masks,
            # medium parameters
            medium_rgb,
            medium_bs,
            medium_attn,
            depths,
            # image size
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            # forward outputs
            render_alphas,
            last_ids,
            # gradients of outputs
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            v_render_medium.contiguous() if v_render_medium is not None else None,
            # dual-path geometry correction config
            ctx.grad_deconf_enabled,
            ctx.grad_deconf_gamma,
            ctx.grad_deconf_lambda_med,
            ctx.grad_deconf_eps,
            ctx.grad_deconf_residual_eta,
            ctx.grad_deconf_fade,
            ctx.grad_deconf_apply_means2d,
            ctx.grad_deconf_apply_conic,
            ctx.grad_deconf_rho_mode,
            # Phase 3 opacity soft correction
            ctx.grad_deconf_apply_opacity,
            ctx.grad_deconf_gamma_opacity,
            ctx.grad_deconf_eta_opacity,
            # Phase 4 medium symmetric correction
            ctx.grad_deconf_gamma_medium,
            ctx.grad_deconf_lambda_obj,
            ctx.grad_deconf_medium_bs_floor,
            ctx.grad_deconf_medium_attn_exact,
        )

        if ctx.needs_input_grad[5]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(1, 2)
            )
        else:
            v_backgrounds = None

        # When medium is not used, forward inputs were None, so gradients must also be None
        if not ctx.has_medium:
            v_medium_rgb = None
            v_medium_bs = None
            v_medium_attn = None

        # Side-channels: expose per-primitive gradient-flow diagnostics for
        # downstream consumers (wandb logging, ρ-guided MCMC, method-
        # validity analysis). Detached so they never participate in
        # autograd. None when dual-path inactive.
        def _stash(tensor):
            return tensor.detach() if (tensor is not None and tensor.numel() > 0) else None
        _RasterizeToPixels.last_rho_per_primitive      = _stash(rho_per_primitive)
        _RasterizeToPixels.last_cos_sm_per_primitive   = _stash(cos_sm_per_primitive)
        _RasterizeToPixels.last_magratio_per_primitive = _stash(magratio_per_primitive)
        _RasterizeToPixels.last_grad_vis_per_primitive = _stash(grad_vis_per_primitive)
        _RasterizeToPixels.last_T_per_view             = _stash(T_per_view)
        _RasterizeToPixels.last_exp_per_view           = _stash(exp_per_view)

        return (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_betas,
            v_backgrounds,
            None,  # masks
            v_medium_rgb,  # medium_rgb
            v_medium_bs,  # medium_bs
            v_medium_attn,  # medium_attn
            None,  # depths (no grad needed)
            None,  # width
            None,  # height
            None,  # tile_size
            None,  # isect_offsets
            None,  # flatten_ids
            None,  # grad_deconf_enabled
            None,  # grad_deconf_gamma
            None,  # grad_deconf_lambda_med
            None,  # grad_deconf_eps
            None,  # grad_deconf_residual_eta
            None,  # grad_deconf_fade
            None,  # grad_deconf_apply_means2d
            None,  # grad_deconf_apply_conic
            None,  # grad_deconf_rho_mode
            None,  # grad_deconf_apply_opacity
            None,  # grad_deconf_gamma_opacity
            None,  # grad_deconf_eta_opacity
            None,  # grad_deconf_gamma_medium
            None,  # grad_deconf_lambda_obj
            None,  # grad_deconf_medium_bs_floor
            None,  # grad_deconf_medium_attn_exact
        )
