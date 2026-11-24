#include "bindings.h"
#include "helpers.cuh"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
// Underwater medium-splatting improvements (compile-time toggles).
//   Override via nvcc: -DGSPLAT_MEDIUM_CAP_ENABLED=0 etc.
//
//   Phase 1: medium_cap — hard per-pixel cap on accumulated backscatter.
//   Phase 3: attn_aware_alpha — scale alpha's occlusion power by mean
//            per-channel attenuation factor. Backward includes the
//            matching medium_attn chain rule.
// See rasterize_to_pixels_bwd.cu for matching backward flags and the
// orthogonal Phase 2 medium-gradient-gate logic.
// ----------------------------------------------------------------------------
#ifndef GSPLAT_MEDIUM_CAP_ENABLED
#define GSPLAT_MEDIUM_CAP_ENABLED 1
#endif
#ifndef GSPLAT_MEDIUM_CAP_VALUE
#define GSPLAT_MEDIUM_CAP_VALUE 1.0f
#endif
#ifndef GSPLAT_ATTN_AWARE_ALPHA_ENABLED
#define GSPLAT_ATTN_AWARE_ALPHA_ENABLED 1
#endif

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Rasterization to Pixels Forward Pass (with Underwater Medium)
 ****************************************************************************/

template <uint32_t COLOR_DIM, typename S>
__global__ void rasterize_to_pixels_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const vec2<S> *__restrict__ means2d, // [C, N, 2] or [nnz, 2]
    const vec3<S> *__restrict__ conics,  // [C, N, 3] or [nnz, 3]
    const S *__restrict__ colors,      // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    const S *__restrict__ opacities,   // [C, N] or [nnz]
    const S *__restrict__ betas,   // [C, N] or [nnz]
    const S *__restrict__ backgrounds, // [C, COLOR_DIM]
    const bool *__restrict__ masks,    // [C, tile_height, tile_width]
    // underwater medium parameters (per-pixel, can be nullptr)
    const S *__restrict__ medium_rgb,  // [C, image_height, image_width, 3]
    const S *__restrict__ medium_bs,   // [C, image_height, image_width, 3]
    const S *__restrict__ medium_attn, // [C, image_height, image_width, 3]
    const S *__restrict__ depths,      // [C, N] or [nnz]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    S *__restrict__ render_colors, // [C, image_height, image_width, COLOR_DIM]
    S *__restrict__ render_alphas, // [C, image_height, image_width, 1]
    S *__restrict__ render_medium, // [C, image_height, image_width, 3] (nullptr if no medium)
    S *__restrict__ render_clear,  // [C, image_height, image_width, 3] (nullptr if no medium) — object without attenuation
    int32_t *__restrict__ last_ids // [C, image_height, image_width]
) {
    auto block = cg::this_thread_block();
    int32_t camera_id = block.group_index().x;
    int32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    const uint32_t cam_pixel_offset = camera_id * image_height * image_width;

    tile_offsets += camera_id * tile_height * tile_width;
    render_colors += cam_pixel_offset * COLOR_DIM;
    render_alphas += cam_pixel_offset;
    last_ids += cam_pixel_offset;
    if (backgrounds != nullptr) {
        backgrounds += camera_id * COLOR_DIM;
    }
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
    }

    // Medium pointer offsets (per-pixel, 3 channels)
    const bool has_medium = (medium_rgb != nullptr);
    const S *med_rgb_cam = nullptr;
    const S *med_bs_cam = nullptr;
    const S *med_attn_cam = nullptr;
    S *rend_med_cam = nullptr;
    S *rend_clear_cam = nullptr;
    if (has_medium) {
        const uint32_t med_offset = cam_pixel_offset * 3;
        med_rgb_cam = medium_rgb + med_offset;
        med_bs_cam = medium_bs + med_offset;
        med_attn_cam = medium_attn + med_offset;
        rend_med_cam = render_medium + med_offset;
        rend_clear_cam = render_clear + med_offset;
    }

    S px = (S)j + 0.5f;
    S py = (S)i + 0.5f;
    int32_t pix_id = i * image_width + j;

    bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    // when the mask is provided, render the background color and return
    if (masks != nullptr && inside && !masks[tile_id]) {
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            render_colors[pix_id * COLOR_DIM + k] =
                backgrounds == nullptr ? 0.0f : backgrounds[k];
        }
        if (has_medium) {
            // When masked, medium output = medium_rgb (full scattering as background)
            rend_med_cam[pix_id * 3 + 0] = med_rgb_cam[pix_id * 3 + 0];
            rend_med_cam[pix_id * 3 + 1] = med_rgb_cam[pix_id * 3 + 1];
            rend_med_cam[pix_id * 3 + 2] = med_rgb_cam[pix_id * 3 + 2];
        }
        return;
    }

    // Load per-pixel medium parameters
    S med_rgb_pix[3], med_bs_pix[3], med_attn_pix[3];
    if (has_medium && inside) {
        GSPLAT_PRAGMA_UNROLL
        for (uint32_t c = 0; c < 3; ++c) {
            med_rgb_pix[c] = med_rgb_cam[pix_id * 3 + c];
            med_bs_pix[c] = med_bs_cam[pix_id * 3 + c];
            med_attn_pix[c] = med_attn_cam[pix_id * 3 + c];
        }
    }

    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec4<S> *xy_opacity_batch =
        reinterpret_cast<vec4<float> *>(&id_batch[block_size]); // [block_size]
    vec3<S> *conic_batch =
        reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]); // [block_size]
    // Depth batch for medium computation (shares space after conic_batch)
    S *depth_batch = reinterpret_cast<S *>(&conic_batch[block_size]); // [block_size]

    S T = 1.0f;
    uint32_t cur_idx = 0;
    uint32_t tr = block.thread_rank();

    S pix_out[COLOR_DIM] = {0.f};
    S pix_medium[3] = {0.f, 0.f, 0.f};
    S pix_clear[3] = {0.f, 0.f, 0.f};
    S prev_depth = 0.f;

    for (uint32_t b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const vec2<S> xy = means2d[g];
            const S opac = opacities[g];
            const S beta = betas[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac, beta};
            conic_batch[tr] = conics[g];
            if (has_medium) {
                depth_batch[tr] = depths[g];
            }
        }

        block.sync();

        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const vec3<S> conic = conic_batch[t];
            const vec4<S> xy_opac = xy_opacity_batch[t];
            const S opac = xy_opac.z;
            const S beta = xy_opac.w;
            const vec2<S> delta = {xy_opac.x - px, xy_opac.y - py};

            const S sigma = (conic.x * delta.x * delta.x +
                                    conic.z * delta.y * delta.y) +
                            2.f * conic.y * delta.x * delta.y;

            if (sigma < 0.f || sigma >= 1.f) {
                continue;
            }

            // Load depth early so attenuation-aware alpha (Phase 3) and
            // the medium backscatter math below can share it.
            S depth = 0.f;
            if (has_medium) {
                depth = depth_batch[t];
            }

            const S alpha_raw = min(0.999f, opac * __powf(1.f - sigma, beta));
            S alpha = alpha_raw;
#if GSPLAT_ATTN_AWARE_ALPHA_ENABLED
            if (has_medium) {
                S sum_exp_attn = 0.f;
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t c = 0; c < 3; ++c) {
                    sum_exp_attn += __expf(-med_attn_pix[c] * depth);
                }
                const S g_attn = 0.5f + 0.5f * (sum_exp_attn / 3.f);
                alpha = alpha_raw * g_attn;
            }
#endif

            const S next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4) {
                done = true;
                break;
            }

            int32_t g = id_batch[t];
            const S vis = alpha * T;
            const S *c_ptr = colors + g * COLOR_DIM;

            if (has_medium) {
                // Object color with medium attenuation (first 3 channels = RGB)
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    if (k < 3) {
                        S exp_attn = __expf(-med_attn_pix[k] * depth);
                        pix_out[k] += exp_attn * c_ptr[k] * vis;
                        // Clear object: accumulate without attenuation
                        pix_clear[k] += c_ptr[k] * vis;
                    } else {
                        // Non-RGB channels (e.g., depth) pass through unchanged
                        pix_out[k] += c_ptr[k] * vis;
                    }
                }

                // Medium backscatter accumulation
                // Integral of medium scattering between prev_depth and depth
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t c = 0; c < 3; ++c) {
                    S exp_bs_prev = __expf(-med_bs_pix[c] * prev_depth);
                    S exp_bs_cur = __expf(-med_bs_pix[c] * depth);
                    pix_medium[c] += T * (exp_bs_prev - exp_bs_cur) * med_rgb_pix[c];
                }

#if GSPLAT_MEDIUM_CAP_ENABLED
                // Phase 1: cap accumulated backscatter to prevent runaway.
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t c = 0; c < 3; ++c) {
                    pix_medium[c] =
                        fminf(pix_medium[c], (S)GSPLAT_MEDIUM_CAP_VALUE);
                }
#endif

                prev_depth = depth;
            } else {
                // No medium: standard accumulation
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    pix_out[k] += c_ptr[k] * vis;
                }
            }

            cur_idx = batch_start + t;
            T = next_T;
        }
    }

    if (inside) {
        render_alphas[pix_id] = 1.0f - T;
        GSPLAT_PRAGMA_UNROLL
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            render_colors[pix_id * COLOR_DIM + k] =
                backgrounds == nullptr ? pix_out[k]
                                       : (pix_out[k] + T * backgrounds[k]);
        }
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);

        // Finalize medium: add remaining transmittance × medium contribution
        if (has_medium) {
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t c = 0; c < 3; ++c) {
                S exp_bs_final = __expf(-med_bs_pix[c] * prev_depth);
                S final_med = pix_medium[c] + T * exp_bs_final * med_rgb_pix[c];
#if GSPLAT_MEDIUM_CAP_ENABLED
                final_med = fminf(final_med, (S)GSPLAT_MEDIUM_CAP_VALUE);
#endif
                rend_med_cam[pix_id * 3 + c] = final_med;
                // Clear object: add background contribution
                rend_clear_cam[pix_id * 3 + c] =
                    backgrounds == nullptr ? pix_clear[c]
                                           : (pix_clear[c] + T * backgrounds[c]);
            }
        }
    }
}

template <uint32_t CDIM>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> call_kernel_with_dim(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &colors,    // [C, N, channels]
    const torch::Tensor &opacities, // [C, N]
    const torch::Tensor &betas,     // [C, N]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
    // medium parameters (optional)
    const at::optional<torch::Tensor> &medium_rgb,  // [C, H, W, 3]
    const at::optional<torch::Tensor> &medium_bs,   // [C, H, W, 3]
    const at::optional<torch::Tensor> &medium_attn, // [C, H, W, 3]
    const at::optional<torch::Tensor> &depths,      // [C, N]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
) {
    GSPLAT_DEVICE_GUARD(means2d);
    GSPLAT_CHECK_INPUT(means2d);
    GSPLAT_CHECK_INPUT(conics);
    GSPLAT_CHECK_INPUT(colors);
    GSPLAT_CHECK_INPUT(opacities);
    GSPLAT_CHECK_INPUT(betas);
    GSPLAT_CHECK_INPUT(tile_offsets);
    GSPLAT_CHECK_INPUT(flatten_ids);
    if (backgrounds.has_value()) {
        GSPLAT_CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        GSPLAT_CHECK_INPUT(masks.value());
    }

    const bool has_medium = medium_rgb.has_value();
    if (has_medium) {
        GSPLAT_CHECK_INPUT(medium_rgb.value());
        GSPLAT_CHECK_INPUT(medium_bs.value());
        GSPLAT_CHECK_INPUT(medium_attn.value());
        GSPLAT_CHECK_INPUT(depths.value());
    }

    uint32_t C = tile_offsets.size(0);
    uint32_t N = means2d.size(1);
    uint32_t channels = colors.size(-1);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, tile_height, tile_width};

    torch::Tensor renders = torch::empty(
        {C, image_height, image_width, channels},
        means2d.options().dtype(torch::kFloat32)
    );
    torch::Tensor alphas = torch::empty(
        {C, image_height, image_width, 1},
        means2d.options().dtype(torch::kFloat32)
    );
    torch::Tensor last_ids = torch::empty(
        {C, image_height, image_width}, means2d.options().dtype(torch::kInt32)
    );

    // Medium output tensor
    torch::Tensor render_med;
    torch::Tensor render_clr;
    if (has_medium) {
        render_med = torch::zeros(
            {C, image_height, image_width, 3},
            means2d.options().dtype(torch::kFloat32)
        );
        render_clr = torch::zeros(
            {C, image_height, image_width, 3},
            means2d.options().dtype(torch::kFloat32)
        );
    } else {
        render_med = torch::empty({0}, means2d.options().dtype(torch::kFloat32));
        render_clr = torch::empty({0}, means2d.options().dtype(torch::kFloat32));
    }

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    // shared mem: id_batch + xy_opacity_batch + conic_batch + depth_batch
    const uint32_t shared_mem =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec4<float>) + sizeof(vec3<float>) + sizeof(float));

    if (cudaFuncSetAttribute(
            rasterize_to_pixels_fwd_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shared_mem,
            " bytes), try lowering tile_size."
        );
    }
    rasterize_to_pixels_fwd_kernel<CDIM, float>
        <<<blocks, threads, shared_mem, stream>>>(
            C,
            N,
            n_isects,
            reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
            reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            betas.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                    : nullptr,
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            has_medium ? medium_rgb.value().data_ptr<float>() : nullptr,
            has_medium ? medium_bs.value().data_ptr<float>() : nullptr,
            has_medium ? medium_attn.value().data_ptr<float>() : nullptr,
            has_medium ? depths.value().data_ptr<float>() : nullptr,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            has_medium ? render_med.data_ptr<float>() : nullptr,
            has_medium ? render_clr.data_ptr<float>() : nullptr,
            last_ids.data_ptr<int32_t>()
        );

    return std::make_tuple(renders, alphas, last_ids, render_med, render_clr);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_pixels_fwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &colors,    // [C, N, channels]
    const torch::Tensor &opacities, // [C, N]
    const torch::Tensor &betas,     // [C, N]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
    // medium parameters (optional)
    const at::optional<torch::Tensor> &medium_rgb,  // [C, H, W, 3]
    const at::optional<torch::Tensor> &medium_bs,   // [C, H, W, 3]
    const at::optional<torch::Tensor> &medium_attn, // [C, H, W, 3]
    const at::optional<torch::Tensor> &depths,      // [C, N]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
) {
    GSPLAT_CHECK_INPUT(colors);
    uint32_t channels = colors.size(-1);

#define __GS__CALL_(N)                                                         \
    case N:                                                                    \
        return call_kernel_with_dim<N>(                                        \
            means2d,                                                           \
            conics,                                                            \
            colors,                                                            \
            opacities,                                                         \
            betas,                                                             \
            backgrounds,                                                       \
            masks,                                                             \
            medium_rgb,                                                        \
            medium_bs,                                                         \
            medium_attn,                                                       \
            depths,                                                            \
            image_width,                                                       \
            image_height,                                                      \
            tile_size,                                                         \
            tile_offsets,                                                      \
            flatten_ids                                                        \
        );

    switch (channels) {
        __GS__CALL_(1)
        __GS__CALL_(2)
        __GS__CALL_(3)
        __GS__CALL_(4)
        __GS__CALL_(5)
        __GS__CALL_(8)
        __GS__CALL_(9)
        __GS__CALL_(16)
        __GS__CALL_(17)
        __GS__CALL_(32)
        __GS__CALL_(33)
        __GS__CALL_(64)
        __GS__CALL_(65)
        __GS__CALL_(128)
        __GS__CALL_(129)
        __GS__CALL_(256)
        __GS__CALL_(257)
        __GS__CALL_(512)
        __GS__CALL_(513)
    default:
        AT_ERROR("Unsupported number of channels: ", channels);
    }
}

} // namespace gsplat
