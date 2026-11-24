#ifndef GSPLAT_CUDA_BINDINGS_H
#define GSPLAT_CUDA_BINDINGS_H

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <tuple>

#define GSPLAT_N_THREADS 256

#define GSPLAT_CHECK_CUDA(x)                                                   \
    TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define GSPLAT_CHECK_CONTIGUOUS(x)                                             \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define GSPLAT_CHECK_INPUT(x)                                                  \
    GSPLAT_CHECK_CUDA(x);                                                      \
    GSPLAT_CHECK_CONTIGUOUS(x)
#define GSPLAT_DEVICE_GUARD(_ten)                                              \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

#define GSPLAT_PRAGMA_UNROLL _Pragma("unroll")

// https://github.com/pytorch/pytorch/blob/233305a852e1cd7f319b15b5137074c9eac455f6/aten/src/ATen/cuda/cub.cuh#L38-L46
#define GSPLAT_CUB_WRAPPER(func, ...)                                          \
    do {                                                                       \
        size_t temp_storage_bytes = 0;                                         \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);                        \
        auto &caching_allocator = *::c10::cuda::CUDACachingAllocator::get();   \
        auto temp_storage = caching_allocator.allocate(temp_storage_bytes);    \
        func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);             \
    } while (false)

namespace gsplat {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cond_mean_convariance_opacity_fwd_tensor(
    const torch::Tensor& means,
    const torch::Tensor& covars,
    const torch::Tensor& opacities,
    const torch::Tensor& betas,
    const torch::Tensor& query
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> cond_mean_convariance_opacity_bwd_tensor(
    const torch::Tensor& means,
    const torch::Tensor& covars,
    const torch::Tensor& opacities,
    const torch::Tensor& betas,
    const torch::Tensor& query,
    const torch::Tensor& v_means,
    const torch::Tensor& v_covars,
    const torch::Tensor& v_opacities
);

torch::Tensor rot_scale_l_triangle_to_covar_fwd_tensor(
    const torch::Tensor& rot,
    const torch::Tensor& scale,
    const torch::Tensor& l_triangle,
    const torch::Tensor& rest_i,
    const torch::Tensor& rest_j,
    const bool spatial_block
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rot_scale_l_triangle_to_covar_bwd_tensor(
    const torch::Tensor& rot,
    const torch::Tensor& scale,
    const torch::Tensor& l_triangle,
    const torch::Tensor& rest_i,
    const torch::Tensor& rest_j,
    bool spatial_block,
    const torch::Tensor& v_covar
);

torch::Tensor l_triangle_to_rotmat_fwd_tensor(
    const torch::Tensor& l_triangle
);

torch::Tensor l_triangle_to_rotmat_bwd_tensor(
    const torch::Tensor& l_triangle,
    const torch::Tensor& v_rotmats
);

std::tuple<torch::Tensor, torch::Tensor> quat_scale_to_covar_preci_fwd_tensor(
    const torch::Tensor &quats,
    const torch::Tensor &scales,
    const bool compute_covar,
    const bool compute_preci,
    const bool triu
);

std::tuple<torch::Tensor, torch::Tensor> quat_scale_to_covar_preci_bwd_tensor(
    const torch::Tensor &quats,
    const torch::Tensor &scales,
    const at::optional<torch::Tensor> &v_covars,
    const at::optional<torch::Tensor> &v_precis,
    const bool triu
);

std::tuple<torch::Tensor, torch::Tensor> proj_fwd_tensor(
    const torch::Tensor &means,
    const torch::Tensor &covars,
    const torch::Tensor &Ks,
    const uint32_t width,
    const uint32_t height,
    const bool ortho
);

std::tuple<torch::Tensor, torch::Tensor> proj_bwd_tensor(
    const torch::Tensor &means,
    const torch::Tensor &covars,
    const torch::Tensor &Ks,
    const uint32_t width,
    const uint32_t height,
    const bool ortho,
    const torch::Tensor &v_means2d,
    const torch::Tensor &v_covars2d
);

std::tuple<torch::Tensor, torch::Tensor> world_to_cam_fwd_tensor(
    const torch::Tensor &means,
    const torch::Tensor &covars,
    const torch::Tensor &viewmats
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> world_to_cam_bwd_tensor(
    const torch::Tensor &means,
    const torch::Tensor &covars,
    const torch::Tensor &viewmats,
    const at::optional<torch::Tensor> &v_means_c,
    const at::optional<torch::Tensor> &v_covars_c,
    const bool means_requires_grad,
    const bool covars_requires_grad,
    const bool viewmats_requires_grad
);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
fully_fused_projection_fwd_tensor(
    const torch::Tensor &means,
    const at::optional<torch::Tensor> &covars,
    const at::optional<torch::Tensor> &quats,
    const at::optional<torch::Tensor> &scales,
    const torch::Tensor &viewmats,
    const torch::Tensor &Ks,
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations,
    const bool ortho
);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
fully_fused_projection_bwd_tensor(
    const torch::Tensor &means,
    const at::optional<torch::Tensor> &covars,
    const at::optional<torch::Tensor> &quats,
    const at::optional<torch::Tensor> &scales,
    const torch::Tensor &viewmats,
    const torch::Tensor &Ks,
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const bool ortho,
    const torch::Tensor &radii,
    const torch::Tensor &conics,
    const at::optional<torch::Tensor> &compensations,
    const torch::Tensor &v_means2d,
    const torch::Tensor &v_depths,
    const torch::Tensor &v_conics,
    const at::optional<torch::Tensor> &v_compensations,
    const bool viewmats_requires_grad
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> isect_tiles_tensor(
    const torch::Tensor &means2d,
    const torch::Tensor &radii,
    const torch::Tensor &depths,
    const at::optional<torch::Tensor> &camera_ids,
    const at::optional<torch::Tensor> &primitive_ids,
    const uint32_t C,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const bool sort,
    const bool double_buffer
);

torch::Tensor isect_offset_encode_tensor(
    const torch::Tensor &isect_ids,
    const uint32_t C,
    const uint32_t tile_width,
    const uint32_t tile_height
);

// Forward: returns (renders, alphas, last_ids, render_medium, render_clear)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_pixels_fwd_tensor(
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &betas,
    const at::optional<torch::Tensor> &backgrounds,
    const at::optional<torch::Tensor> &mask,
    // medium parameters (optional)
    const at::optional<torch::Tensor> &medium_rgb,
    const at::optional<torch::Tensor> &medium_bs,
    const at::optional<torch::Tensor> &medium_attn,
    const at::optional<torch::Tensor> &depths,
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets,
    const torch::Tensor &flatten_ids
);

// Backward: returns
//   (v_means2d, v_conics, v_colors, v_opacities, v_betas,
//    v_medium_rgb, v_medium_bs, v_medium_attn,
//    rho_per_primitive, cos_sm_per_primitive, magratio_per_primitive,
//    grad_vis_per_primitive, T_per_view, exp_per_view)
// Six per-primitive gradient-flow diagnostics exposed to Python for
// wandb logging, ρ-guided MCMC, and method-validity analysis:
//   rho_per_primitive       — identifiability score ρ = l1(S)/(l1(S)+λl1(M)+ε)
//   cos_sm_per_primitive    — cos(S, M) ∈ [-1, 1]; tests whether medium-path
//                             gradient is systematically direction-confounded
//                             with object-path gradient
//   magratio_per_primitive  — |M|_2 / |S|_2;      tests whether medium-path
//                             magnitude is non-negligible
//   grad_vis_per_primitive  — α-weighted mean of T_{p,i} · mean_c(exp(-β_D,c·d_i))
//                             over contributing pixels p; effective multiplier
//                             attenuating the gradient reaching primitive i
//                             (Mode #1 compound diagnostic).
//   T_per_view              — α-weighted mean of T_{p,i} only. Occlusion
//                             component of grad_vis. var(T) across views
//                             reflects primitive-ordering instability, a
//                             failure mode that exists in any 3DGS.
//   exp_per_view            — α-weighted mean of mean_c(exp(-β_D,c·d_i)) only.
//                             Medium component. var(exp) across views is the
//                             underwater-specific failure mode (Beer-Lambert
//                             amplification of per-view depth drift).
// The split lets us attribute across-view variance in grad_vis to either
// occlusion (T) or medium (exp), identifying which failure mode dominates.
// All are [C, N] when their respective pathway is active, otherwise empty.
std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor>
rasterize_to_pixels_bwd_tensor(
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &betas,
    const at::optional<torch::Tensor> &backgrounds,
    const at::optional<torch::Tensor> &mask,
    // medium parameters (optional)
    const at::optional<torch::Tensor> &medium_rgb,
    const at::optional<torch::Tensor> &medium_bs,
    const at::optional<torch::Tensor> &medium_attn,
    const at::optional<torch::Tensor> &depths,
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets,
    const torch::Tensor &flatten_ids,
    // forward outputs
    const torch::Tensor &render_alphas,
    const torch::Tensor &last_ids,
    // gradients of outputs
    const torch::Tensor &v_render_colors,
    const torch::Tensor &v_render_alphas,
    const at::optional<torch::Tensor> &v_render_medium,
    // dual-path geometry correction config
    //   enabled   — toggle; must also have medium active to take effect
    //   gamma     — cancellation-recovery coefficient γ
    //   lambda_med — medium-suppression strength λ_med in ρ denominator
    //   eps       — numerical safety floor
    //   apply_*   — per-parameter-block kill switches (default True)
    //   rho_mode  — 0=off  (base=1; at γ=0 this is the "strict M-drop"
    //                       ablation — v_means2d = S, i.e. medium-path
    //                       contribution is completely discarded),
    //               1=damp (base = ρ·|S|, slow medium-heavy),
    //               2=amp  (base = (2−ρ)·|S|, push medium-heavy),
    //               3=soft (sign(S)·(2−ρ)·|S| + ρ·M — amp on S direction
    //                       plus ρ-weighted M-direction restored; recovers
    //                       baseline at ρ=1 and amp at ρ=0).
    const bool  grad_deconf_enabled,
    const float grad_deconf_gamma,
    const float grad_deconf_lambda_med,
    const float grad_deconf_eps,
    const float grad_deconf_residual_eta,
    const float grad_deconf_fade,
    const bool  grad_deconf_apply_means2d,
    const bool  grad_deconf_apply_conic,
    const int   grad_deconf_rho_mode,
    // --- Phase 3 opacity soft correction (no-op when deconf off) ---
    const bool  grad_deconf_apply_opacity,
    const float grad_deconf_gamma_opacity,
    const float grad_deconf_eta_opacity,
    // --- Phase 4 medium symmetric correction (no-op when deconf off) ---
    const float grad_deconf_gamma_medium,
    const float grad_deconf_lambda_obj,
    const float grad_deconf_medium_bs_floor,
    const bool  grad_deconf_medium_attn_exact
);

std::tuple<torch::Tensor, torch::Tensor> rasterize_to_indices_in_range_tensor(
    const uint32_t range_start,
    const uint32_t range_end,
    const torch::Tensor transmittances,
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &opacities,
    const torch::Tensor &betas,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const torch::Tensor &tile_offsets,
    const torch::Tensor &flatten_ids
);
} // namespace gsplat

#endif // GSPLAT_CUDA_BINDINGS_H
