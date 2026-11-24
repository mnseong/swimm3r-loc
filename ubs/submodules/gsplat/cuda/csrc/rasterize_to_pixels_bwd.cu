#include "bindings.h"
#include "helpers.cuh"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
// Underwater medium-splatting improvements (compile-time toggles).
// Orthogonal to the dual-path correction wired through grad_deconf_*
// runtime flags. See rasterize_to_pixels_fwd.cu for matching forward flags.
//
//   Phase 2: medium_gate — at object-confident pixels (small T_final),
//            suppress the medium-induced term only inside deconf split
//            statistics. The exact alpha adjoint remains unchanged.
//   Phase 3: attn_aware_alpha — alpha_eff = alpha_raw * g, with
//            g = 0.5 + 0.5 * mean(exp_attn). Backward scales v_opacity /
//            v_beta / v_sigma by g and propagates the g -> medium_attn
//            chain rule.
// ----------------------------------------------------------------------------
#ifndef GSPLAT_MEDIUM_GATE_ENABLED
#define GSPLAT_MEDIUM_GATE_ENABLED 1
#endif
#ifndef GSPLAT_ATTN_AWARE_ALPHA_ENABLED
#define GSPLAT_ATTN_AWARE_ALPHA_ENABLED 1
#endif
// Whether the attenuation coupling (v_alpha_obj_attenuated - v_alpha_obj_pure)
// is reassigned into the deconf M-stream (the "global split", commit 9c506fa).
// =1 (current): S = clear-water gradient, M = backscatter + attenuation
//   coupling. BUT the coupling = (exp(-attn·z)-1)·clear is STRUCTURALLY anti-
//   parallel to S=clear, so M ≈ -k·S, the split is DEGENERATE (cos(S,M)≈-1,
//   measured), and full deconf → v=S = the WRONG (clear-water) target.
// =0 (backscatter-only): S = ATTENUATED object gradient (the correct underwater
//   baseline — keeps the legitimate attenuation), M = backscatter occlusion
//   ONLY (the genuine veil-chasing confound, NOT collinear with S). Full deconf
//   then removes only veil-chasing while still targeting the underwater image.
#ifndef GSPLAT_ATTN_IN_MED
#define GSPLAT_ATTN_IN_MED 1
#endif

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Rasterization to Pixels Backward Pass (with Underwater Medium)
 *
 * Forward is the standard underwater compositor (unchanged).
 * Backward splits the per-pixel alpha gradient into object-supportive
 * and medium-induced terms and accumulates three per-primitive
 * projected-spatial statistics:
 *
 *     S_i = sum_p  v_xy_obj_p         (signed object-side 2D grad)
 *     A_i = sum_p |v_xy_obj_p|        (absolute object-side magnitude)
 *     M_i = sum_p  v_xy_med_p         (signed medium-induced 2D grad)
 *
 * A post-kernel overwrites v_means2d per primitive with a geometry-only
 * correction:
 *
 *     ρ_i = l1(S_i) / (l1(S_i) + λ_med·l1(M_i) + ε)
 *     κ_i = clip((l1(A_i) − l1(S_i)) / (l1(A_i) + ε), 0, 1)
 *     G_c = sign(S_c) · (|S_c| + γ·ρ·κ·ReLU(A_c − |S_c|))   per component
 *
 * By default this correction is applied only to projected centers. Conic,
 * opacity, and medium-family corrections remain opt-in. No τ modulation,
 * no Fisher σ-gate, no PCGrad surgery — the compact dual-path projected-
 * center correction is the stable intervention path.
 ****************************************************************************/

template <uint32_t COLOR_DIM, typename S>
__global__ void rasterize_to_pixels_bwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    // fwd inputs
    const vec2<S> *__restrict__ means2d, // [C, N, 2] or [nnz, 2]
    const vec3<S> *__restrict__ conics,  // [C, N, 3] or [nnz, 3]
    const S *__restrict__ colors,      // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    const S *__restrict__ opacities,   // [C, N] or [nnz]
    const S *__restrict__ betas,   // [C, N] or [nnz]
    const S *__restrict__ backgrounds, // [C, COLOR_DIM]
    const bool *__restrict__ masks,    // [C, tile_height, tile_width]
    // medium parameters (can be nullptr)
    const S *__restrict__ medium_rgb,  // [C, H, W, 3]
    const S *__restrict__ medium_bs,   // [C, H, W, 3]
    const S *__restrict__ medium_attn, // [C, H, W, 3]
    const S *__restrict__ depths,      // [C, N] or [nnz]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // fwd outputs
    const S *__restrict__ render_alphas,  // [C, image_height, image_width, 1]
    const int32_t *__restrict__ last_ids, // [C, image_height, image_width]
    // grad outputs
    const S *__restrict__ v_render_colors, // [C, H, W, COLOR_DIM]
    const S *__restrict__ v_render_alphas, // [C, H, W, 1]
    const S *__restrict__ v_render_medium, // [C, H, W, 3] (nullptr if no medium)
    // grad inputs
    vec2<S> *__restrict__ v_means2d,     // [C, N, 2] or [nnz, 2]
    vec3<S> *__restrict__ v_conics,      // [C, N, 3] or [nnz, 3]
    S *__restrict__ v_colors,   // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    S *__restrict__ v_opacities, // [C, N] or [nnz]
    S *__restrict__ v_betas,     // [C, N] or [nnz]
    // medium gradients (can be nullptr)
    S *__restrict__ v_medium_rgb,  // [C, H, W, 3]
    S *__restrict__ v_medium_bs,   // [C, H, W, 3]
    S *__restrict__ v_medium_attn, // [C, H, W, 3]
    // --- Dual-path backward auxiliaries (nullptr => disabled) ---
    // When the buffers below are provided (and medium is active) the
    // kernel splits the per-pixel v_alpha into object-supportive and
    // medium-induced streams and accumulates per-primitive S_P / A_P / M_P
    // statistics for each corrected parameter block P ∈ {means2d, conic}.
    // The matching post-kernels consume those buffers and overwrite
    // v_means2d / v_conics with soft deconfounded updates. Optional opacity
    // and medium-family post-corrections are controlled by separate runtime
    // flags and are disabled by default.
    vec2<S> *__restrict__ v_xy_obj_signed,     // [C, N, 2]  S_i(means2d)
    vec2<S> *__restrict__ v_xy_obj_abs,        // [C, N, 2]  A_i(means2d)
    vec2<S> *__restrict__ v_xy_med_signed,     // [C, N, 2]  M_i(means2d)
    vec3<S> *__restrict__ v_conic_obj_signed,  // [C, N, 3]  S_i(conic)
    vec3<S> *__restrict__ v_conic_obj_abs,     // [C, N, 3]  A_i(conic)
    vec3<S> *__restrict__ v_conic_med_signed,  // [C, N, 3]  M_i(conic)
    // --- Gradient-visibility diagnostics (Mode #1 + mechanism split) ---
    // Three separate per-primitive alpha-weighted accumulators, each
    // normalised by the shared Σ α denominator in the post-kernel:
    //   grad_vis_accum  : Σ_p α · T · mean(exp_attn)   (combined)
    //   T_accum         : Σ_p α · T                    (occlusion only)
    //   exp_attn_accum  : Σ_p α · mean(exp_attn)       (medium only)
    // After normalisation:
    //   grad_vis_i  ≈ mean effective gradient multiplier (compound)
    //   T_i         ≈ mean transmittance (standard 3DGS occlusion)
    //   exp_i       ≈ mean Beer-Lambert attenuation (underwater specific)
    // Separating the two components lets us quantify how much of the
    // across-view variance in grad_vis comes from occlusion reordering
    // (which exists in any 3DGS) vs from underwater medium (which is the
    // method-specific contribution).
    S       *__restrict__ grad_vis_accum,      // [C, N]  numerator (compound)
    S       *__restrict__ T_accum,             // [C, N]  numerator (T only)
    S       *__restrict__ exp_attn_accum,      // [C, N]  numerator (exp only)
    S       *__restrict__ grad_vis_weight,     // [C, N]  shared denominator (Σα)
    // --- Phase 3 opacity soft deconf (nullptr ⇒ disabled) ---
    // Per-primitive S/A/M splits for the opacity adjoint, mirroring the
    // existing means2d / conic split path.
    S       *__restrict__ v_opacity_obj_signed,   // [C, N]  S_opac
    S       *__restrict__ v_opacity_obj_abs,      // [C, N]  A_opac
    S       *__restrict__ v_opacity_med_signed,   // [C, N]  M_opac
    // --- Phase 4 medium symmetric soft deconf (nullptr ⇒ disabled) ---
    // Per-pixel split stats for each medium family. Atomic-summed across
    // gaussians visiting the same pixel; baseline v_medium_* tensors
    // already carry the signed sum, so we only need the absolute |g|
    // and the object-conflict-weighted |g|·w accumulators.
    S       *__restrict__ a_medium_rgb,           // [C, H, W, 3]  A_rgb
    S       *__restrict__ c_medium_rgb,           // [C, H, W, 3]  C_rgb
    S       *__restrict__ a_medium_bs,            // [C, H, W, 3]  A_bs
    S       *__restrict__ c_medium_bs,            // [C, H, W, 3]  C_bs
    S       *__restrict__ a_medium_attn,          // [C, H, W, 3]  A_attn
    S       *__restrict__ c_medium_attn           // [C, H, W, 3]  C_attn
) {
    auto block = cg::this_thread_block();
    uint32_t camera_id = block.group_index().x;
    uint32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    const uint32_t cam_pixel_offset = camera_id * image_height * image_width;

    tile_offsets += camera_id * tile_height * tile_width;
    render_alphas += cam_pixel_offset;
    last_ids += cam_pixel_offset;
    v_render_colors += cam_pixel_offset * COLOR_DIM;
    v_render_alphas += cam_pixel_offset;
    if (backgrounds != nullptr) {
        backgrounds += camera_id * COLOR_DIM;
    }
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
    }

    const bool has_medium = (medium_rgb != nullptr);
    // Dual-path split accumulation per parameter block. Each is enabled
    // only when its corresponding split buffer is allocated (= when the
    // user requested the matching apply_* correction). This avoids paying
    // the per-pixel compute / warp-reduce / atomicAdd cost for blocks the
    // post-correction kernel won't read.
    const bool accumulate_means2d = has_medium && (v_xy_obj_signed != nullptr);
    const bool accumulate_conic   = has_medium && (v_conic_obj_signed != nullptr);
    const bool accumulate_opacity = has_medium && (v_opacity_obj_signed != nullptr);
    const bool accumulate_split   = accumulate_means2d || accumulate_conic || accumulate_opacity;

    // Medium pointer offsets
    const S *med_rgb_cam = nullptr;
    const S *med_bs_cam = nullptr;
    const S *med_attn_cam = nullptr;
    const S *v_rend_med_cam = nullptr;
    S *v_med_rgb_cam = nullptr;
    S *v_med_bs_cam = nullptr;
    S *v_med_attn_cam = nullptr;
    if (has_medium) {
        const uint32_t med_offset = cam_pixel_offset * 3;
        med_rgb_cam = medium_rgb + med_offset;
        med_bs_cam = medium_bs + med_offset;
        med_attn_cam = medium_attn + med_offset;
        v_rend_med_cam = v_render_medium + med_offset;
        v_med_rgb_cam = v_medium_rgb + med_offset;
        v_med_bs_cam = v_medium_bs + med_offset;
        v_med_attn_cam = v_medium_attn + med_offset;
    }

    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    const S px = (S)j + 0.5f;
    const S py = (S)i + 0.5f;
    const int32_t pix_id =
        min(i * image_width + j, image_width * image_height - 1);

    const bool inside = (i < image_height && j < image_width);

    const int2 range = {tile_offsets[tile_id],
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? (int32_t)n_isects
            : tile_offsets[tile_id + 1]};
    const uint32_t block_size = block.size();
    const uint32_t num_batches =
        (range.y - range.x + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;
    vec4<S> *xy_opacity_batch =
        reinterpret_cast<vec4<float> *>(&id_batch[block_size]);
    vec3<S> *conic_batch =
        reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]);
    S *rgbs_batch = (S *)&conic_batch[block_size]; // [block_size * COLOR_DIM]
    S *depth_batch = &rgbs_batch[block_size * COLOR_DIM]; // [block_size]

    S T_final = 1.0f - render_alphas[pix_id];
    S T = T_final;
    S buffer[COLOR_DIM] = {0.f};
    // Clear-water (unattenuated) color buffer — needed by the GLOBAL deconf
    // split to separate the attenuation-coupled part of the object gradient
    // into the M-stream. Accumulated only when the split is active.
    S buffer_clear[3] = {0.f};
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    S v_render_c[COLOR_DIM];
    GSPLAT_PRAGMA_UNROLL
    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * COLOR_DIM + k];
    }
    const S v_render_a = v_render_alphas[pix_id];

    // Load medium per-pixel data and gradients
    S med_rgb_pix[3] = {0.f}, med_bs_pix[3] = {0.f}, med_attn_pix[3] = {0.f};
    S v_out_med[3] = {0.f};
    S buffer_medium[3] = {0.f};
    // For tracking depth from the gaussian "behind" current one (back-to-front)
    S latter_depth = 10.f;  // large initial depth
    S latter_exp_bs[3] = {0.f, 0.f, 0.f};

    if (has_medium && inside) {
        GSPLAT_PRAGMA_UNROLL
        for (uint32_t c = 0; c < 3; ++c) {
            med_rgb_pix[c] = med_rgb_cam[pix_id * 3 + c];
            med_bs_pix[c] = med_bs_cam[pix_id * 3 + c];
            med_attn_pix[c] = med_attn_cam[pix_id * 3 + c];
            v_out_med[c] = v_rend_med_cam[pix_id * 3 + c];
        }
    }

    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final =
        cg::reduce(warp, bin_final, cg::greater<int>());

    for (uint32_t b = 0; b < num_batches; ++b) {
        block.sync();

        // Each thread fetch 1 gaussian from back to front
        const int32_t batch_end = range.y - 1 - block_size * b;
        const int32_t batch_size = min(block_size, batch_end + 1 - range.x);
        const int32_t idx = batch_end - tr;
        if (idx >= range.x) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const vec2<S> xy = means2d[g];
            const S opac = opacities[g];
            const S beta = betas[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac, beta};
            conic_batch[tr] = conics[g];
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                rgbs_batch[tr * COLOR_DIM + k] = colors[g * COLOR_DIM + k];
            }
            if (has_medium) {
                depth_batch[tr] = depths[g];
            }
        }
        block.sync();

        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size;
             ++t) {
            bool valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            S alpha;
            S opac;
            S beta;
            vec2<S> delta;
            vec3<S> conic;
            S vis;
            S sigma;
            S depth = 0.f;
            // Phase 3 scale factor; 1.0 when feature disabled or !has_medium
            S g_attn = 1.f;

            if (valid) {
                conic = conic_batch[t];
                vec4<S> xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                beta = xy_opac.w;
                delta = {xy_opac.x - px, xy_opac.y - py};

                sigma = (conic.x * delta.x * delta.x +
                                  conic.z * delta.y * delta.y) +
                          2.f * conic.y * delta.x * delta.y;

                if (sigma < 0.f || sigma >= 1.f) {
                    valid = false;
                }

                // Load depth before alpha for attenuation-aware alpha (mirrors forward)
                if (has_medium) {
                    depth = depth_batch[t];
                }

                vis = __powf(1.f - sigma, beta);
                const S alpha_raw = min(0.999f, opac * vis);
#if GSPLAT_ATTN_AWARE_ALPHA_ENABLED
                if (has_medium) {
                    S sum_exp_attn = 0.f;
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t c = 0; c < 3; ++c) {
                        sum_exp_attn += __expf(-med_attn_pix[c] * depth);
                    }
                    g_attn = 0.5f + 0.5f * (sum_exp_attn / 3.f);
                }
#endif
                alpha = alpha_raw * g_attn;
            }

            if (!warp.any(valid)) {
                continue;
            }
            S v_rgb_local[COLOR_DIM] = {0.f};
            vec3<S> v_conic_local = {0.f, 0.f, 0.f};
            vec2<S> v_xy_local = {0.f, 0.f};
            // Split contributions for the dual-path corrections.
            vec2<S> v_xy_obj_local     = {0.f, 0.f};
            vec2<S> v_xy_obj_abs_local = {0.f, 0.f};
            vec2<S> v_xy_med_local     = {0.f, 0.f};
            vec3<S> v_conic_obj_local     = {0.f, 0.f, 0.f};
            vec3<S> v_conic_obj_abs_local = {0.f, 0.f, 0.f};
            vec3<S> v_conic_med_local     = {0.f, 0.f, 0.f};
            // Phase 3 opacity split locals.
            S v_opacity_obj_local     = 0.f;
            S v_opacity_obj_abs_local = 0.f;
            S v_opacity_med_local     = 0.f;
            S v_opacity_local = 0.f;
            S v_beta_local = 0.f;
            S v_medium_rgb_pix_local[3] = {0.f};
            S v_medium_bs_pix_local[3] = {0.f};
            S v_medium_attn_pix_local[3] = {0.f};
            // Gradient-visibility diagnostic accumulators (Mode #1 +
            // mechanism split between occlusion and medium contributions).
            S grad_vis_local   = 0.f;   // Σ_p α · T · mean(exp_attn)
            S T_only_local     = 0.f;   // Σ_p α · T
            S exp_only_local   = 0.f;   // Σ_p α · mean(exp_attn)
            S grad_vis_w_local = 0.f;   // Σ_p α  (shared denominator)

            if (valid) {
                // compute the current T for this gaussian
                S ra = 1.0f / (1.0f - alpha);
                T *= ra;
                const S fac = alpha * T;

                S v_alpha = 0.f;
                // Object-supportive and medium-induced components of v_alpha.
                // v_alpha = v_alpha_obj + v_alpha_med_full is the exact
                // forward adjoint used by all baseline gradients. The gated
                // medium term is used only to populate deconf split buffers.
                S v_alpha_obj = 0.f;
                S v_alpha_obj_pure = 0.f;  // clear-water object evidence (no attn coupling)
                S v_alpha_med_full = 0.f;
                S v_alpha_med_deconf = 0.f;

                if (has_medium) {
                    // ----- Medium backward logic (adapted from WaterSplatting) -----
                    S exp_attn[3], exp_bs[3];
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t c = 0; c < 3; ++c) {
                        exp_attn[c] = __expf(-med_attn_pix[c] * depth);
                        exp_bs[c] = __expf(-med_bs_pix[c] * depth);
                    }

                    // Gradient-visibility accumulation (Mode #1 diagnostic)
                    // with separate tracks for the occlusion factor T and
                    // the medium attenuation factor mean(exp_attn), so we
                    // can decompose across-view variance of the compound
                    // quantity T·exp into its two causes.
                    if (grad_vis_accum != nullptr) {
                        const S mean_exp_attn =
                            (exp_attn[0] + exp_attn[1] + exp_attn[2]) * S(1.f/3.f);
                        // At this point T has been divided by (1-alpha) so it
                        // represents transmittance *before* this primitive,
                        // which is exactly the multiplier on its contribution.
                        grad_vis_local   += alpha * T * mean_exp_attn;
                        T_only_local     += alpha * T;
                        exp_only_local   += alpha * mean_exp_attn;
                        grad_vis_w_local += alpha;
                    }

                    // Update running medium buffer (T not yet updated in WS convention)
                    // But we process back-to-front, so T is already divided by (1-alpha)
                    // We track buffer_medium for alpha gradient contribution
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t c = 0; c < 3; ++c) {
                        buffer_medium[c] += T * med_rgb_pix[c] * exp_bs[c];
                    }

                    // Gradient for medium_rgb from backscatter
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t c = 0; c < 3; ++c) {
                        v_medium_rgb_pix_local[c] += v_out_med[c] * T * (exp_bs[c] - latter_exp_bs[c]);
                    }

                    // Gradient for medium_bs from backscatter
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t c = 0; c < 3; ++c) {
                        S v_exp_bs_c = v_out_med[c] * T * med_rgb_pix[c];
                        v_medium_bs_pix_local[c] += v_exp_bs_c * (-depth * exp_bs[c] + latter_depth * latter_exp_bs[c]);
                    }

                    // v_rgb with attenuation
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        if (k < 3) {
                            S exp_attn_fac = fac * exp_attn[k];
                            v_rgb_local[k] = v_render_c[k] * exp_attn_fac;
                        } else {
                            v_rgb_local[k] = fac * v_render_c[k];
                        }
                    }

                    // Gradient for medium_attn
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t c = 0; c < 3; ++c) {
                        S v_exp_attn_c = fac * exp_attn[c] * rgbs_batch[t * COLOR_DIM + c] * v_render_c[c];
                        v_medium_attn_pix_local[c] += -v_exp_attn_c * depth;
                    }

                    // v_alpha_obj: object RGB contribution (attenuated for
                    // colour channels, pass-through for auxiliary channels).
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        if (k < 3) {
                            v_alpha_obj += (rgbs_batch[t * COLOR_DIM + k] * T * exp_attn[k] - buffer[k] * ra) * v_render_c[k];
                            if (accumulate_split) {
                                // clear-water counterpart: same path with the
                                // attenuation stripped (exp_attn -> 1,
                                // attenuated buffer -> clear buffer).
                                v_alpha_obj_pure += (rgbs_batch[t * COLOR_DIM + k] * T - buffer_clear[k] * ra) * v_render_c[k];
                            }
                        } else {
                            v_alpha_obj += (rgbs_batch[t * COLOR_DIM + k] * T - buffer[k] * ra) * v_render_c[k];
                            if (accumulate_split) {
                                // aux channels carry no attenuation: identical.
                                v_alpha_obj_pure += (rgbs_batch[t * COLOR_DIM + k] * T - buffer[k] * ra) * v_render_c[k];
                            }
                        }
                    }
                    // Phase 2 medium gate: at object-confident pixels, suppress
                    // the medium-induced negative gradient pushing alpha down.
                    // Compile-time controlled to match origin/main exactly
                    // (default GSPLAT_MEDIUM_GATE_ENABLED=1). Applied to
                    // v_alpha_med_full so the baseline v_alpha used by
                    // opacity / conic / beta gradients is bit-identical to
                    // the pre-SGA kernel.
#if GSPLAT_MEDIUM_GATE_ENABLED
                    const S obj_conf_ = 1.f - T_final;
                    const S medium_gate_ = fmaxf(0.f, 1.f - 2.f * obj_conf_);
#else
                    const S medium_gate_ = 1.f;
#endif
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t c = 0; c < 3; ++c) {
                        const S med_term = -buffer_medium[c] * ra * v_out_med[c];
                        v_alpha_med_full   += medium_gate_ * med_term;
                        if (accumulate_split) {
                            // UNGATED for the deconf split. The gate zeroes
                            // med_term at every pixel with accumulated opacity
                            // > 0.5 (gate = max(0, 2*T_final - 1)) — i.e. every
                            // converged surface pixel — so the M-stream
                            // measured ~0, rho pinned at 1, and the deconf
                            // rewrite degenerated to the exact gradient (inert
                            // SGA; observed rho_mean 0.998 / p10 1.0). The
                            // baseline gradient keeps the gate via
                            // v_alpha_med_full (bit-identical); the split
                            // statistics and the rho-weighted M term now
                            // measure the TRUE medium-induced gradient, and
                            // the soft rewrite handles it proportionally to
                            // rho instead of a hard all-or-nothing gate.
                            v_alpha_med_deconf += med_term;
                        }
                    }

                    // render_alpha supervision and background leakage are
                    // object-supportive (they constrain visibility / geometry).
                    v_alpha_obj += T_final * ra * v_render_a;
                    if (accumulate_split) {
                        v_alpha_obj_pure += T_final * ra * v_render_a;
                    }
                    if (backgrounds != nullptr) {
                        S accum = 0.f;
                        GSPLAT_PRAGMA_UNROLL
                        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                            accum += backgrounds[k] * v_render_c[k];
                        }
                        v_alpha_obj += -T_final * ra * accum;
                        if (accumulate_split) {
                            v_alpha_obj_pure += -T_final * ra * accum;
                        }
                    }

                    v_alpha = v_alpha_obj + v_alpha_med_full;

                    // ----- GLOBAL deconf split (attenuation-aware M) -----
                    // The attenuation coupling exp(-attn*z) modulates the
                    // object gradient at EVERY veiled pixel, but it lived in
                    // the S-stream, so engagement was confined to the sparse
                    // backscatter-residual set. Reassign the attenuation-
                    // coupled part (obj - obj_pure) to the M-stream:
                    //   S := clear-water object evidence
                    //   M := backscatter occlusion + attenuation coupling
                    // S + M is unchanged, and this runs AFTER the baseline
                    // v_alpha composition above, so all baseline gradients
                    // remain bit-identical — only the split semantics widen.
                    // Where attn -> 0, obj == obj_pure and behavior reduces
                    // to the previous (backscatter-only) split.
#if GSPLAT_ATTN_IN_MED
                    if (accumulate_split) {
                        v_alpha_med_deconf += (v_alpha_obj - v_alpha_obj_pure);
                        v_alpha_obj = v_alpha_obj_pure;
                    }
#else
                    // Backscatter-only M: keep S = v_alpha_obj (ATTENUATED, the
                    // correct underwater baseline that retains the legitimate
                    // attenuation), leave M = backscatter occlusion only. No
                    // reassignment — the attenuation coupling stays in S where
                    // it belongs (it is not a confound). cos(S,M) now carries
                    // genuine directional information instead of the structural
                    // ≈-1 the global split produced. v_alpha_obj_pure unused
                    // here (kept computed for the buffer_clear bookkeeping).
                    (void)v_alpha_obj_pure;
#endif

                    // ----- Phase 2: attn-aware alpha exact chain rule -----
                    // Forward applies alpha = alpha_raw * g_attn, where
                    //   g_attn = 0.5 + 0.5 * mean(exp(-β_D · z))
                    // The standard backward only chained g_attn through
                    // ∂alpha/∂alpha_raw (handled below in v_sigma). The
                    // ∂alpha/∂g_attn · ∂g_attn/∂β_D path was missing — now
                    // added so v_medium_attn is consistent with forward.
                    // ∂g_attn/∂β_D_c = -depth · exp_attn_c / 6.
#if GSPLAT_ATTN_AWARE_ALPHA_ENABLED
                    if (has_medium && g_attn > S(0.f)) {
                        const S alpha_raw = alpha / g_attn;
                        const S chain_coef = v_alpha * alpha_raw / S(6.f);
                        GSPLAT_PRAGMA_UNROLL
                        for (uint32_t c = 0; c < 3; ++c) {
                            v_medium_attn_pix_local[c] +=
                                chain_coef * (-depth * exp_attn[c]);
                        }
                    }
#endif

                    // ----- Phase 4 medium split stats (A, C) accumulation -----
                    // S (signed) is already accumulated into v_medium_*_pix_local
                    // above; we add absolute magnitude (A) and conflict-weighted
                    // magnitude (C) here, only when the deconf path is active.
                    if (accumulate_split && a_medium_rgb != nullptr) {
                        // 0/0 guard only — per-hit |v_alpha| terms sit at
                        // ~1e-7 scale with mean-reduced losses, so a 1e-6
                        // floor dominated the denominator and squashed
                        // rho_obj_p / w_conflict toward 0 for weak hits.
                        const S _eps_p = S(1e-12f);
                        const S abs_o = fabsf(v_alpha_obj);
                        const S abs_m = fabsf(v_alpha_med_deconf);
                        const S abs_sum = fabsf(v_alpha_obj + v_alpha_med_deconf);
                        const S c_p = fmaxf(abs_o + abs_m - abs_sum, S(0.f))
                                     / (abs_o + abs_m + _eps_p);
                        const S rho_obj_p = abs_o / (abs_o + abs_m + _eps_p);
                        const S w_conflict = rho_obj_p * c_p;

                        // Re-derive this iteration's medium deltas (same
                        // formulas used to accumulate into v_medium_*_pix_local
                        // earlier in this loop body).
                        GSPLAT_PRAGMA_UNROLL
                        for (uint32_t c = 0; c < 3; ++c) {
                            const S d_rgb =
                                v_out_med[c] * T * (exp_bs[c] - latter_exp_bs[c]);
                            const S v_exp_bs_c = v_out_med[c] * T * med_rgb_pix[c];
                            const S d_bs = v_exp_bs_c *
                                (-depth * exp_bs[c] + latter_depth * latter_exp_bs[c]);
                            const S v_exp_attn_c = fac * exp_attn[c] *
                                rgbs_batch[t * COLOR_DIM + c] * v_render_c[c];
                            const S d_attn = -v_exp_attn_c * depth;

                            const S abs_rgb  = fabsf(d_rgb);
                            const S abs_bs   = fabsf(d_bs);
                            const S abs_attn = fabsf(d_attn);

                            // Pixel-level atomicAdd directly. These are
                            // tiny per-thread / per-gaussian increments.
                            const uint32_t pix_off = pix_id * 3 + c;
                            gpuAtomicAdd(a_medium_rgb  + pix_off, abs_rgb);
                            gpuAtomicAdd(a_medium_bs   + pix_off, abs_bs);
                            gpuAtomicAdd(a_medium_attn + pix_off, abs_attn);
                            gpuAtomicAdd(c_medium_rgb  + pix_off, abs_rgb  * w_conflict);
                            gpuAtomicAdd(c_medium_bs   + pix_off, abs_bs   * w_conflict);
                            gpuAtomicAdd(c_medium_attn + pix_off, abs_attn * w_conflict);
                        }
                    }

                    // Update running buffer (with attenuation)
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        if (k < 3) {
                            buffer[k] += rgbs_batch[t * COLOR_DIM + k] * fac * exp_attn[k];
                            if (accumulate_split) {
                                buffer_clear[k] += rgbs_batch[t * COLOR_DIM + k] * fac;
                            }
                        } else {
                            buffer[k] += rgbs_batch[t * COLOR_DIM + k] * fac;
                        }
                    }

                    // Update medium buffer for depth tracking (subtract contribution)
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t c = 0; c < 3; ++c) {
                        buffer_medium[c] += -T * med_rgb_pix[c] * exp_bs[c];
                    }

                    latter_depth = depth;
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t c = 0; c < 3; ++c) {
                        latter_exp_bs[c] = exp_bs[c];
                    }

                } else {
                    // ----- Standard (no medium) backward -----
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        v_rgb_local[k] = fac * v_render_c[k];
                    }
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        v_alpha +=
                            (rgbs_batch[t * COLOR_DIM + k] * T - buffer[k] * ra) *
                            v_render_c[k];
                    }
                    v_alpha += T_final * ra * v_render_a;
                    if (backgrounds != nullptr) {
                        S accum = 0.f;
                        GSPLAT_PRAGMA_UNROLL
                        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                            accum += backgrounds[k] * v_render_c[k];
                        }
                        v_alpha += -T_final * ra * accum;
                    }
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        buffer[k] += rgbs_batch[t * COLOR_DIM + k] * fac;
                    }
                }

                // Beta kernel gradients (common for both medium and non-medium).
                // We keep the legacy v_xy_local (from the full v_alpha) so that
                // opacity / conic / beta gradients are bitwise unchanged. The
                // split v_xy_{obj,med}_local below are only used to populate
                // the auxiliary buffers for post-kernel sharpening.
                // Phase 3: alpha_eff = alpha_raw * g_attn; chain rule adds
                // an extra factor of g_attn to v_opacity / v_beta / v_sigma
                // (and hence v_xy, v_conic). g_attn = 1 when feature is off
                // or medium is inactive, recovering the original formulas.
                if (opac * vis <= 0.999f) {
                    const S inv_one_minus_sigma = __powf(1.f - sigma, beta - 1.f);
                    const S v_sigma = -v_alpha * opac * beta * inv_one_minus_sigma * g_attn;
                    v_conic_local = {
                        delta.x * delta.x * v_sigma,
                        2.f * delta.x * delta.y * v_sigma,
                        delta.y * delta.y * v_sigma
                    };
                    v_xy_local = {
                        2.f * v_sigma * (conic.x * delta.x + conic.y * delta.y),
                        2.f * v_sigma * (conic.y * delta.x + conic.z * delta.y)
                    };
                    v_opacity_local = vis * v_alpha * g_attn;
                    v_beta_local = v_alpha * opac * __powf(1.f - sigma, beta) * __logf(1.f - sigma) * g_attn;

                    if (accumulate_split) {
                        // Split v_alpha into object / medium contributions.
                        // v_sigma, v_xy, v_conic and v_opacity are all linear
                        // in v_alpha, so splitting propagates by linearity.
                        // Per-block gating skips compute when its correction
                        // is disabled (saves ~9 atomicAdds + the corresponding
                        // multiplies per pixel-Gaussian intersection).
                        const S v_sigma_obj =
                            -v_alpha_obj * opac * beta * inv_one_minus_sigma * g_attn;
                        const S v_sigma_med =
                            -v_alpha_med_deconf * opac * beta * inv_one_minus_sigma * g_attn;

                        if (accumulate_means2d) {
                            v_xy_obj_local = {
                                2.f * v_sigma_obj * (conic.x * delta.x + conic.y * delta.y),
                                2.f * v_sigma_obj * (conic.y * delta.x + conic.z * delta.y)
                            };
                            v_xy_med_local = {
                                2.f * v_sigma_med * (conic.x * delta.x + conic.y * delta.y),
                                2.f * v_sigma_med * (conic.y * delta.x + conic.z * delta.y)
                            };
                            v_xy_obj_abs_local = {
                                fabsf(v_xy_obj_local.x),
                                fabsf(v_xy_obj_local.y)
                            };
                        }

                        if (accumulate_conic) {
                            v_conic_obj_local = {
                                delta.x * delta.x * v_sigma_obj,
                                2.f * delta.x * delta.y * v_sigma_obj,
                                delta.y * delta.y * v_sigma_obj
                            };
                            v_conic_med_local = {
                                delta.x * delta.x * v_sigma_med,
                                2.f * delta.x * delta.y * v_sigma_med,
                                delta.y * delta.y * v_sigma_med
                            };
                            v_conic_obj_abs_local = {
                                fabsf(v_conic_obj_local.x),
                                fabsf(v_conic_obj_local.y),
                                fabsf(v_conic_obj_local.z)
                            };
                        }

                        if (accumulate_opacity) {
                            v_opacity_obj_local = vis * v_alpha_obj * g_attn;
                            v_opacity_med_local = vis * v_alpha_med_deconf * g_attn;
                            v_opacity_obj_abs_local = fabsf(v_opacity_obj_local);
                        }
                    }
                }
            }

            // Warp reduce and atomic writes
            warpSum<COLOR_DIM, S>(v_rgb_local, warp);
            warpSum<decltype(warp), S>(v_conic_local, warp);
            warpSum<decltype(warp), S>(v_xy_local, warp);
            warpSum<decltype(warp), S>(v_opacity_local, warp);
            warpSum<decltype(warp), S>(v_beta_local, warp);
            if (accumulate_means2d) {
                warpSum<decltype(warp), S>(v_xy_obj_local, warp);
                warpSum<decltype(warp), S>(v_xy_obj_abs_local, warp);
                warpSum<decltype(warp), S>(v_xy_med_local, warp);
            }
            if (accumulate_conic) {
                warpSum<decltype(warp), S>(v_conic_obj_local, warp);
                warpSum<decltype(warp), S>(v_conic_obj_abs_local, warp);
                warpSum<decltype(warp), S>(v_conic_med_local, warp);
            }
            if (accumulate_opacity) {
                warpSum<decltype(warp), S>(v_opacity_obj_local, warp);
                warpSum<decltype(warp), S>(v_opacity_obj_abs_local, warp);
                warpSum<decltype(warp), S>(v_opacity_med_local, warp);
            }
            // Gradient-visibility diagnostic reduction (Mode #1 + split).
            if (grad_vis_accum != nullptr) {
                warpSum<decltype(warp), S>(grad_vis_local, warp);
                warpSum<decltype(warp), S>(T_only_local, warp);
                warpSum<decltype(warp), S>(exp_only_local, warp);
                warpSum<decltype(warp), S>(grad_vis_w_local, warp);
            }

            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                S *v_rgb_ptr = (S *)(v_colors) + COLOR_DIM * g;
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                S *v_conic_ptr = (S *)(v_conics) + 3 * g;
                gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
                gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);

                S *v_xy_ptr = (S *)(v_means2d) + 2 * g;
                gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

                gpuAtomicAdd(v_opacities + g, v_opacity_local);
                gpuAtomicAdd(v_betas + g, v_beta_local);

                if (accumulate_means2d) {
                    S *v_obj_ptr = (S *)(v_xy_obj_signed) + 2 * g;
                    S *v_abs_ptr = (S *)(v_xy_obj_abs) + 2 * g;
                    S *v_med_ptr = (S *)(v_xy_med_signed) + 2 * g;
                    gpuAtomicAdd(v_obj_ptr,     v_xy_obj_local.x);
                    gpuAtomicAdd(v_obj_ptr + 1, v_xy_obj_local.y);
                    gpuAtomicAdd(v_abs_ptr,     v_xy_obj_abs_local.x);
                    gpuAtomicAdd(v_abs_ptr + 1, v_xy_obj_abs_local.y);
                    gpuAtomicAdd(v_med_ptr,     v_xy_med_local.x);
                    gpuAtomicAdd(v_med_ptr + 1, v_xy_med_local.y);
                }
                if (accumulate_conic) {
                    S *c_obj_ptr = (S *)(v_conic_obj_signed) + 3 * g;
                    S *c_abs_ptr = (S *)(v_conic_obj_abs) + 3 * g;
                    S *c_med_ptr = (S *)(v_conic_med_signed) + 3 * g;
                    gpuAtomicAdd(c_obj_ptr,     v_conic_obj_local.x);
                    gpuAtomicAdd(c_obj_ptr + 1, v_conic_obj_local.y);
                    gpuAtomicAdd(c_obj_ptr + 2, v_conic_obj_local.z);
                    gpuAtomicAdd(c_abs_ptr,     v_conic_obj_abs_local.x);
                    gpuAtomicAdd(c_abs_ptr + 1, v_conic_obj_abs_local.y);
                    gpuAtomicAdd(c_abs_ptr + 2, v_conic_obj_abs_local.z);
                    gpuAtomicAdd(c_med_ptr,     v_conic_med_local.x);
                    gpuAtomicAdd(c_med_ptr + 1, v_conic_med_local.y);
                    gpuAtomicAdd(c_med_ptr + 2, v_conic_med_local.z);
                }
                if (accumulate_opacity) {
                    gpuAtomicAdd(v_opacity_obj_signed + g, v_opacity_obj_local);
                    gpuAtomicAdd(v_opacity_obj_abs    + g, v_opacity_obj_abs_local);
                    gpuAtomicAdd(v_opacity_med_signed + g, v_opacity_med_local);
                }
                // Mode #1 gradient-visibility diagnostic atomic adds
                // (combined + separated components).
                if (grad_vis_accum != nullptr) {
                    gpuAtomicAdd(grad_vis_accum  + g, grad_vis_local);
                    gpuAtomicAdd(T_accum         + g, T_only_local);
                    gpuAtomicAdd(exp_attn_accum  + g, exp_only_local);
                    gpuAtomicAdd(grad_vis_weight + g, grad_vis_w_local);
                }
            }

            // Medium per-pixel gradient reduction and write
            if (has_medium) {
                // Reduce medium gradients across warp
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t c = 0; c < 3; ++c) {
                    v_medium_rgb_pix_local[c] = cg::reduce(warp, v_medium_rgb_pix_local[c], cg::plus<S>());
                    v_medium_bs_pix_local[c] = cg::reduce(warp, v_medium_bs_pix_local[c], cg::plus<S>());
                    v_medium_attn_pix_local[c] = cg::reduce(warp, v_medium_attn_pix_local[c], cg::plus<S>());
                }
                if (warp.thread_rank() == 0) {
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t c = 0; c < 3; ++c) {
                        gpuAtomicAdd(v_med_attn_cam + pix_id * 3 + c, v_medium_attn_pix_local[c]);
                        gpuAtomicAdd(v_med_rgb_cam + pix_id * 3 + c, v_medium_rgb_pix_local[c]);
                        gpuAtomicAdd(v_med_bs_cam + pix_id * 3 + c, v_medium_bs_pix_local[c]);
                    }
                }
            }
        }
    }

    // Handle the medium segment from camera to first gaussian (back-to-front finalization)
    block.sync();
    if (has_medium && inside) {
        S v_medium_rgb_first[3] = {0.f};
        S v_medium_bs_first[3] = {0.f};
        S exp_bs_first[3] = {1.f, 1.f, 1.f}; // exp(0) = 1 at camera position (depth=0)
        S T_first = 1.f;

        GSPLAT_PRAGMA_UNROLL
        for (uint32_t c = 0; c < 3; ++c) {
            v_medium_rgb_first[c] = v_out_med[c] * T_first * (exp_bs_first[c] - latter_exp_bs[c]);
        }

        GSPLAT_PRAGMA_UNROLL
        for (uint32_t c = 0; c < 3; ++c) {
            S v_exp_bs_c = v_out_med[c] * T_first * med_rgb_pix[c];
            v_medium_bs_first[c] = v_exp_bs_c * latter_depth * latter_exp_bs[c];
        }

        GSPLAT_PRAGMA_UNROLL
        for (uint32_t c = 0; c < 3; ++c) {
            gpuAtomicAdd(v_med_rgb_cam + pix_id * 3 + c, v_medium_rgb_first[c]);
            gpuAtomicAdd(v_med_bs_cam + pix_id * 3 + c, v_medium_bs_first[c]);
        }
    }
}

/****************************************************************************
 * Soft dual-path correction — post-kernels.
 *
 * One thread per primitive in each kernel. Consumes the per-primitive
 * S/A/M buffers populated by the main backward pass and overwrites the
 * legacy gradient on the targeted parameter block. Two kernels are
 * instantiated (means2d, conic) sharing the same rule:
 *
 *     l1_S = Σ_c |S_c|
 *     l1_A = Σ_c  A_c            (already non-negative)
 *     l1_M = Σ_c |M_c|
 *     ρ   = l1_S / (l1_S + λ_med · l1_M + ε)
 *     κ   = clip((l1_A − l1_S) / (l1_A + ε), 0, 1)
 *     scale = γ · ρ · κ
 *     G_c = sign(S_c) · (|S_c| + scale · ReLU(A_c − |S_c|))
 *     v_P[g] ← G     (component-wise for vector P, scalar for opacity)
 *
 * Corner case (shared across all three): when the object-side split has
 * vanished (l1_S, l1_A ≤ ε) but the medium-side split is alive (l1_M > ε),
 * the primitive is pure medium-induced confounding — we zero that
 * parameter's update so the medium gradient cannot drift it.
 ****************************************************************************/

// --- grad-visibility normalisation post-kernel ---------------------------- //
// Divides per-primitive (Σ α·T·mean_exp_attn) by (Σ α) to produce the
// alpha-weighted mean effective gradient multiplier for each primitive.
// Runs once per backward; O(N) work.
//   grad_vis_out[i] = accum[i] / (weight[i] + ε)
// Interpretation: grad_vis_out ∈ [0, 1] where
//   1  → primitive receives full-strength gradient (no occlusion, no medium)
//   <1 → primitive's gradient is attenuated by Beer-Lambert + occlusion chain
//   ~0 → primitive is gradient-starved (deep occlusion or far + strong medium)
// When ε-guarded denominator approaches zero (primitive contributes to no
// pixel — out of frustum / too small), output a benign 1.0 so the
// diagnostic histogram isn't polluted by undefined entries.
template <typename S>
__global__ void grad_vis_normalize_kernel(
    const uint32_t total_primitives,
    const S *__restrict__ accum,
    const S *__restrict__ weight,
    S       *__restrict__ grad_vis_out,
    const S eps
) {
    const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total_primitives) return;
    const S w = weight[g];
    grad_vis_out[g] = (w > eps) ? (accum[g] / w) : S(1);
}

// --- ρ / cos(S,M) / |M|/|S| export kernel --------------------------------- //
// Computes three per-primitive gradient-flow diagnostics from the means2d
// S and M buffers and writes them to three output tensors. Runs
// independently of the correction kernels so the diagnostics are
// available even when apply_means2d / apply_conic kill switches are off.
//
// Outputs (all [C, N]):
//   rho_out       — ρ = l1(S) / (l1(S) + λ·l1(M) + ε), identifiability score
//   cos_sm_out    — cos(S, M) = <S,M> / (|S|·|M|). Tests whether the
//                   medium-path gradient is systematically misaligned with
//                   the object-path gradient (mean < 0 ⇒ real confounder;
//                   mean ≈ 0 ⇒ random noise; mean > 0 ⇒ M is helpful).
//                   Defined to 0 when |S|·|M| ≤ ε (no directional info).
//   magratio_out  — |M|_2 / (|S|_2 + ε), L2 magnitude ratio. Complementary
//                   to ρ (which uses L1 norms). Tells how large the
//                   medium-path contribution is relative to object-path in
//                   Euclidean terms — direct test of whether M is
//                   negligible.
template <typename S>
__global__ void rho_export_kernel(
    const uint32_t total_primitives,
    const vec2<S> *__restrict__ v_xy_obj_signed,
    const vec2<S> *__restrict__ v_xy_med_signed,
    S       *__restrict__ rho_out,              // [C, N]
    S       *__restrict__ cos_sm_out,           // [C, N]
    S       *__restrict__ magratio_out,         // [C, N]
    const S lambda_med,
    const S eps
) {
    const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total_primitives) return;

    const vec2<S> S_vec = v_xy_obj_signed[g];
    const vec2<S> M_vec = v_xy_med_signed[g];

    // GATE vs FLOOR separation. `eps` (runtime, 1e-12) is a pure 0/0 guard
    // so rho stays scale-invariant for every measurable splat. `stats_gate`
    // (1e-6) gates ONLY the diagnostics below: cos needs measurable
    // directions, and magratio > 0 doubles as the strong-S instrument
    // count (sga/n_engaged). It must NOT gate the correction kernels:
    // with honest rho the soft rewrite is self-protecting (M ≈ 0 → rho ≈ 1
    // → exact gradient recovered), whereas gating the rewrite at 1e-6
    // shrank/zeroed the gradients of ~99% of contributing splats (the
    // per-primitive l2_S median sits below 1e-6) — a severe quality
    // collapse, A/B-verified against the ungated kernel. The rho buffer
    // stays floor-gated so n_signal = #{rho < 1} counts global engagement
    // and the lambda controller calibrates on the full signal set.
    const S stats_gate = S(1e-6f);

    // L1 norms — for ρ (matches the correction kernels' convention).
    const S l1_S = fabsf(S_vec.x) + fabsf(S_vec.y);
    const S l1_M = fabsf(M_vec.x) + fabsf(M_vec.y);

    // L2 norms + dot — for cos(S,M) and |M|/|S|.
    const S l2_S = sqrtf(S_vec.x * S_vec.x + S_vec.y * S_vec.y);
    const S l2_M = sqrtf(M_vec.x * M_vec.x + M_vec.y * M_vec.y);
    const S dot_SM = S_vec.x * M_vec.x + S_vec.y * M_vec.y;

    // ρ: pure-medium → 0, no-signal → 1 (benign).
    if (l1_S <= eps) {
        rho_out[g] = (l1_M > eps) ? S(0) : S(1);
    } else {
        rho_out[g] = l1_S / (l1_S + lambda_med * l1_M + eps);
    }

    // cos(S,M): undefined when either magnitude ≤ ε.  Zero is the neutral
    // default that (a) doesn't skew the mean and (b) reads naturally as
    // "no directional info" in downstream histograms.
    if (l2_S > stats_gate && l2_M > stats_gate) {
        cos_sm_out[g] = dot_SM / (l2_S * l2_M);
    } else {
        cos_sm_out[g] = S(0);
    }

    // |M| / |S|: L2 magnitude ratio = the MEDIUM-MLP's influence on this
    // primitive's gradient relative to the object's. UNGATED (eps floor, not
    // stats_gate) so WEAK-S splats are included: floaters live there — object
    // says ~nothing (l2_S→0) while medium says "keep me to explain veil"
    // (l2_M>0), so |M|/|S|>1 there. That k>1 population is exactly where the
    // medium dominates the existence decision (vanilla = (1-k)·S sign-FLIPS for
    // k>1, keeping the floater); the strong-S gate hid them. Pure-medium
    // (l2_S≤eps<l2_M) → large sentinel so frac(ratio>1) counts it.
    if (l2_S > eps) {
        magratio_out[g] = l2_M / l2_S;
    } else {
        magratio_out[g] = (l2_M > eps) ? S(1e6f) : S(0);
    }
}

// OPACITY-channel diagnostic export. The opacity gradient is a SCALAR, so its
// object/medium split S_o / M_o carry SIGN directly — which the unsigned
// means2d magratio cannot. This lets us separate the two floater populations
// the opacity surgery (opacity_correction_kernel, mode 5) treats differently:
// Sign convention (Adam DESCENT on raw opacity, v=dL/d(opacity), verified):
//   v>0 -> opacity DOWN -> KILL;  v<0 -> opacity UP -> KEEP.
//   floater: object wants it gone -> S_o>0 (kill); medium holds it -> M_o<0 (keep).
//   * FLIP (surgery WIN): eps<|S_o|, OPPOSITE sign (cos=-1), |M_o|>|S_o| (k>1).
//     Vanilla v=S_o+M_o<0 (M_o<0 dominates) -> opacity UP -> keeps the floater;
//     surgery drops M -> v=S_o>0 -> opacity DOWN -> kills it. lr-impossible.
//   * ESCAPE (surgery GAP): |S_o|<=eps with M_o<0 (pure-medium keep). The
//     do-no-harm guard (abs_S<=eps -> return vanilla) lets these stay floaters.
// Writes the SAME rho/cos/magratio buffers the means2d export uses, so the
// existing sga/* logging works unchanged and the means2d-off final config is
// no longer blind. cos here is the scalar sign agreement, NOT a direction cos.
template <typename S>
__global__ void opacity_rho_export_kernel(
    const uint32_t total_primitives,
    const S *__restrict__ v_opacity_obj_signed,   // S_o (signed)
    const S *__restrict__ v_opacity_med_signed,   // M_o (signed)
    S       *__restrict__ rho_out,                // [C, N]
    S       *__restrict__ cos_sm_out,             // [C, N]  sign(S_o*M_o)
    S       *__restrict__ magratio_out,           // [C, N]  |M_o|/|S_o| + sentinel
    const S lambda_med,
    const S eps
) {
    const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total_primitives) return;

    const S S_o = v_opacity_obj_signed[g];
    const S M_o = v_opacity_med_signed[g];
    const S abs_S = fabsf(S_o);
    const S abs_M = fabsf(M_o);

    // rho: pure-medium -> 0, no-signal -> 1 (means2d convention).
    if (abs_S <= eps) {
        rho_out[g] = (abs_M > eps) ? S(0) : S(1);
    } else {
        rho_out[g] = abs_S / (abs_S + lambda_med * abs_M + eps);
    }

    // cos = scalar sign agreement. -1 = OPPOSITE sign = floater signature
    // (object wants opacity DOWN/kill, medium wants it UP/keep). +1 = aligned.
    // 0 = no anchor or no medium. NOT gated by stats_gate — a scalar's sign is
    // meaningful even for weak-S floaters below the directional measurement gate.
    if (abs_S > eps && abs_M > eps) {
        cos_sm_out[g] = (S_o * M_o > S(0.f)) ? S(1.f) : S(-1.f);
    } else {
        cos_sm_out[g] = S(0.f);
    }

    // magratio = |M_o|/|S_o| keep-floater magnitude dominance. Finite when an
    // object anchor exists. Pure-medium KEEP-floater (|S_o|<=eps, M_o<0 = medium
    // alone holds opacity UP, the floater that escapes the surgery) -> 1e6
    // sentinel = the escape population. Pure-medium with M_o>0 (medium also
    // wants it gone, harmless, dies on its own) -> 0. The sentinel has cos==0
    // (no object anchor), so Python disambiguates it from huge finite ratios.
    if (abs_S > eps) {
        magratio_out[g] = abs_M / abs_S;
    } else {
        magratio_out[g] = (abs_M > eps && M_o < S(0.f)) ? S(1e6f) : S(0.f);
    }
}

// --- means2d (2D vector) -------------------------------------------------- //
template <typename S>
__global__ void means2d_correction_kernel(
    const uint32_t total_primitives,
    const vec2<S> *__restrict__ v_xy_obj_signed,
    const vec2<S> *__restrict__ v_xy_obj_abs,
    const vec2<S> *__restrict__ v_xy_med_signed,
    vec2<S> *__restrict__ v_means2d,
    const S gamma,
    const S lambda_med,
    const S eps,
    const S residual_eta,
    const S fade,
    const int rho_mode
) {
    const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total_primitives) return;

    const vec2<S> S_vec = v_xy_obj_signed[g];
    const vec2<S> A_vec = v_xy_obj_abs[g];
    const vec2<S> M_vec = v_xy_med_signed[g];

    const S l1_S = fabsf(S_vec.x) + fabsf(S_vec.y);
    const S l1_A = A_vec.x + A_vec.y;
    const S l1_M = fabsf(M_vec.x) + fabsf(M_vec.y);
    S residual = residual_eta;
    if (residual < S(0.f)) residual = S(0.f);
    if (residual > S(1.f)) residual = S(1.f);

    // True-zero guard only (eps = runtime 1e-12 floor). With honest rho
    // the soft rewrite is self-protecting for weak splats: M ≈ 0 → rho ≈ 1
    // → exact gradient (do-no-harm). A wider 1e-6 gate here shrank the
    // means2d gradient of ~99% of contributing splats (their l2_S sits
    // below 1e-6) and collapsed quality — A/B vs the ungated kernel.
    if (l1_A <= eps && l1_S <= eps) {
        // No object-supportive center signal survived the split. Keep a
        // small exact-gradient residual so frozen primitive sets can retain
        // perceptual edge motion without fully restoring medium drift.
        v_means2d[g].x *= residual;
        v_means2d[g].y *= residual;
        return;
    }

    const S rho = l1_S / (l1_S + lambda_med * l1_M + eps);
    S kappa = (l1_A - l1_S) / (l1_A + eps);
    if (kappa < 0.f) kappa = 0.f;
    if (kappa > 1.f) kappa = 1.f;
    const S scale = gamma * rho * kappa;

    // ------------------------------------------------------------------- //
    // rho_mode = 3 (soft) — amp on S direction + ρ-weighted M direction   //
    //                                                                     //
    //     v_means2d = sign(S)·(2−ρ)·|S|  +  ρ · M                         //
    //                                                                     //
    // Boundary recovery:                                                  //
    //   ρ→1 (clean object)  ⇒  sign(S)·|S| + M  = baseline (S+M)          //
    //   ρ→0 (pure medium)   ⇒  sign(S)·2·|S|   = hard amp                 //
    // Preserves the medium-side directional signal instead of dropping it.//
    // Short-circuits before the S=0 zeroing so M can still propagate.     //
    // ------------------------------------------------------------------- //
    if (rho_mode == 3) {
        const S base_scale_soft = S(2) - rho;
        const S sign_x = (S_vec.x >= 0.f) ? S(1) : S(-1);
        const S sign_y = (S_vec.y >= 0.f) ? S(1) : S(-1);
        const S exact_x = v_means2d[g].x;
        const S exact_y = v_means2d[g].y;
        v_means2d[g].x =
            sign_x * base_scale_soft * fabsf(S_vec.x) + rho * M_vec.x +
            residual * (S(1.f) - rho) * exact_x;
        v_means2d[g].y =
            sign_y * base_scale_soft * fabsf(S_vec.y) + rho * M_vec.y +
            residual * (S(1.f) - rho) * exact_y;
        return;
    }

    // rho_mode = 4 (clean): v = S + ρ·M, the counterfactual deconfounder.
    //   "Keep the clear-water object gradient S in full; admit only an
    //    identifiability-weighted fraction ρ of the medium gradient M."
    //   ρ→1 (object-dominated) ⇒ S + M = exact (do-no-harm).
    //   ρ→0 (medium-dominated) ⇒ S, the TRUE clear-water gradient (M removed).
    // No (2−ρ) amplification (that inflated the least-identifiable splats up
    // to ~2× — the wrong counterfactual), no η residual, no κ — all vestigial
    // here. Built from S_vec/M_vec directly (NOT from `exact`, whose medium
    // term is the GATED v_alpha_med_full and so ≠ S+M exactly).
    if (rho_mode == 4) {
        // clean: v = S + ρ·M, faded to vanilla (S+M) over the refinement
        // window. f=0 -> clean deconf; f=1 -> vanilla noSGA (ρ effectively 1).
        const S f = (fade < S(0.f)) ? S(0.f) : ((fade > S(1.f)) ? S(1.f) : fade);
        const S van_x = v_means2d[g].x;   // vanilla S+M (kernel input)
        const S van_y = v_means2d[g].y;
        const S bx = S_vec.x + rho * M_vec.x;
        const S by = S_vec.y + rho * M_vec.y;
        v_means2d[g].x = (S(1.f) - f) * bx + f * van_x;
        v_means2d[g].y = (S(1.f) - f) * by + f * van_y;
        return;
    }

    // rho_mode = 5 (blend): magnitude-anchored, conflict-averse direction blend.
    //   v = ||S|| · unit( unit(S) + w·unit(M) ),   w = max(0, cos(S,M)).
    // Anchors the geometry step MAGNITUDE to ||S|| (the trustworthy primitive-
    // scale object gradient) and is EXACTLY invariant to ||M|| — the medium
    // MLP's huge gradient norm (||M|| ~ 1e3-1e4·||S||, structural: a few dense
    // MLP params vs sparse per-primitive geometry) can no longer set how far a
    // splat moves. M contributes DIRECTION only, and only insofar as it agrees
    // with S (w = relu(cos): aligned medium kept, opposing/orthogonal dropped —
    // the directional-confound test the unused cos channel was built for).
    // Single dial β = gamma (grad_deconf_gamma, default 1.0; unused elsewhere
    // in blend, so repurposed here): w = β·relu(cos). β=1 is the param-free
    // blend; β=0 → v=S (medium direction fully removed, pure deconf); β>1 →
    // admit more of the aligned medium direction (over-blend). The ONLY
    // continuous knob mean-only blend has.
    if (rho_mode == 5) {
        const S beta = gamma;
        // REFINEMENT FADE: vanilla = the kernel-input gradient (= S+M, the
        // noSGA means2d gradient). f=0 -> pure blend (the coarse/placement win,
        // medium confound live); f=1 -> vanilla noSGA. The blend wins coarse
        // placement but throttles the refinement-phase sharpening the full
        // gradient enjoys once the confound is resolved; a smooth f: 0->1 over
        // the refinement window keeps the coarse win and hands refinement back
        // to noSGA. v = (1-f)*blend + f*vanilla.
        const S f = (fade < S(0.f)) ? S(0.f) : ((fade > S(1.f)) ? S(1.f) : fade);
        const S van_x = v_means2d[g].x;   // vanilla S+M (kernel input)
        const S van_y = v_means2d[g].y;
        const S nS = sqrtf(S_vec.x * S_vec.x + S_vec.y * S_vec.y);
        const S nM = sqrtf(M_vec.x * M_vec.x + M_vec.y * M_vec.y);
        S bx, by;
        if (nS <= eps) {            // no object anchor → do-no-harm to frozen set
            bx = residual * van_x;
            by = residual * van_y;
        } else if (nM <= eps) {     // no medium term → pure object gradient
            bx = S_vec.x;
            by = S_vec.y;
        } else {
            const S c = (S_vec.x * M_vec.x + S_vec.y * M_vec.y) / (nS * nM);
            const S w = beta * ((c > 0.f) ? c : S(0.f));   // β·relu(cos)
            const S dx = S_vec.x / nS + w * M_vec.x / nM;
            const S dy = S_vec.y / nS + w * M_vec.y / nM;
            const S nd = sqrtf(dx * dx + dy * dy);
            const S sc = nS / fmaxf(nd, eps);
            bx = sc * dx;
            by = sc * dy;
        }
        v_means2d[g].x = (S(1.f) - f) * bx + f * van_x;
        v_means2d[g].y = (S(1.f) - f) * by + f * van_y;
        return;
    }

    // rho_mode:
    //   0 = off  → base_scale = 1            (pure dual-path filter)
    //   1 = damp → base_scale = ρ            ∈ [0, 1]   (slow medium-heavy)
    //   2 = amp  → base_scale = 2 − ρ        ∈ [1, 2]   (push medium-heavy)
    S base_scale;
    if (rho_mode == 1)      base_scale = rho;
    else if (rho_mode == 2) base_scale = S(2) - rho;
    else                    base_scale = S(1);

    const S extra_x = fmaxf(A_vec.x - fabsf(S_vec.x), 0.f);
    const S extra_y = fmaxf(A_vec.y - fabsf(S_vec.y), 0.f);
    const S mag_x = base_scale * fabsf(S_vec.x) + scale * extra_x;
    const S mag_y = base_scale * fabsf(S_vec.y) + scale * extra_y;
    const S sign_x = (S_vec.x >= 0.f) ? 1.f : -1.f;
    const S sign_y = (S_vec.y >= 0.f) ? 1.f : -1.f;
    const S exact_x = v_means2d[g].x;
    const S exact_y = v_means2d[g].y;
    const S deconf_x = (S_vec.x == 0.f) ? S(0.f) : sign_x * mag_x;
    const S deconf_y = (S_vec.y == 0.f) ? S(0.f) : sign_y * mag_y;
    v_means2d[g].x = deconf_x + residual * (S(1.f) - rho) * exact_x;
    v_means2d[g].y = deconf_y + residual * (S(1.f) - rho) * exact_y;
}

// --- conic (3D vector) ---------------------------------------------------- //
template <typename S>
__global__ void conic_correction_kernel(
    const uint32_t total_primitives,
    const vec3<S> *__restrict__ v_conic_obj_signed,
    const vec3<S> *__restrict__ v_conic_obj_abs,
    const vec3<S> *__restrict__ v_conic_med_signed,
    vec3<S> *__restrict__ v_conics,
    const S gamma,
    const S lambda_med,
    const S eps,
    const S fade,
    const int rho_mode
) {
    const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total_primitives) return;

    const vec3<S> S_vec = v_conic_obj_signed[g];
    const vec3<S> A_vec = v_conic_obj_abs[g];
    const vec3<S> M_vec = v_conic_med_signed[g];

    const S l1_S = fabsf(S_vec.x) + fabsf(S_vec.y) + fabsf(S_vec.z);
    const S l1_A = A_vec.x + A_vec.y + A_vec.z;
    const S l1_M = fabsf(M_vec.x) + fabsf(M_vec.y) + fabsf(M_vec.z);

    // True-zero guard only (see means2d kernel). A 1e-6 gate here ZEROED
    // the conic gradient of most contributing splats — the global M-stream
    // exceeds 1e-6 while their S sits below it — killing shape/texture
    // optimization scene-wide.
    if (l1_A <= eps && l1_S <= eps) {
        if (l1_M > eps) v_conics[g] = {0.f, 0.f, 0.f};
        return;
    }

    const S rho = l1_S / (l1_S + lambda_med * l1_M + eps);
    S kappa = (l1_A - l1_S) / (l1_A + eps);
    if (kappa < 0.f) kappa = 0.f;
    if (kappa > 1.f) kappa = 1.f;
    const S scale = gamma * rho * kappa;

    // rho_mode = 3 (soft): sign(S)·(2−ρ)·|S| + ρ·M, per component.
    if (rho_mode == 3) {
        const S base_scale_soft = S(2) - rho;
        const S sign_x = (S_vec.x >= 0.f) ? S(1) : S(-1);
        const S sign_y = (S_vec.y >= 0.f) ? S(1) : S(-1);
        const S sign_z = (S_vec.z >= 0.f) ? S(1) : S(-1);
        v_conics[g].x = sign_x * base_scale_soft * fabsf(S_vec.x) + rho * M_vec.x;
        v_conics[g].y = sign_y * base_scale_soft * fabsf(S_vec.y) + rho * M_vec.y;
        v_conics[g].z = sign_z * base_scale_soft * fabsf(S_vec.z) + rho * M_vec.z;
        return;
    }

    // rho_mode = 4 (clean): v = S + ρ·M (see means2d kernel). No amplification.
    // Faded to vanilla (S+M) over the refinement window (f: 0->1).
    if (rho_mode == 4) {
        const S f = (fade < S(0.f)) ? S(0.f) : ((fade > S(1.f)) ? S(1.f) : fade);
        const vec3<S> van = v_conics[g];   // vanilla S+M (kernel input)
        v_conics[g].x = (S(1.f) - f) * (S_vec.x + rho * M_vec.x) + f * van.x;
        v_conics[g].y = (S(1.f) - f) * (S_vec.y + rho * M_vec.y) + f * van.y;
        v_conics[g].z = (S(1.f) - f) * (S_vec.z + rho * M_vec.z) + f * van.z;
        return;
    }

    // rho_mode = 5 (blend): magnitude-anchored direction blend (see means2d).
    //   v = ||S|| · unit( unit(S) + w·unit(M) ),  w = β·relu(cos),  β=gamma.
    if (rho_mode == 5) {
        const S beta = gamma;
        const S nS = sqrtf(S_vec.x * S_vec.x + S_vec.y * S_vec.y
                           + S_vec.z * S_vec.z);
        const S nM = sqrtf(M_vec.x * M_vec.x + M_vec.y * M_vec.y
                           + M_vec.z * M_vec.z);
        if (nS <= eps) { return; }   // keep vanilla v_conics (do-no-harm)
        if (nM <= eps) {
            v_conics[g] = S_vec;
            return;
        }
        const S c = (S_vec.x * M_vec.x + S_vec.y * M_vec.y
                     + S_vec.z * M_vec.z) / (nS * nM);
        const S w = beta * ((c > 0.f) ? c : S(0.f));
        const S dx = S_vec.x / nS + w * M_vec.x / nM;
        const S dy = S_vec.y / nS + w * M_vec.y / nM;
        const S dz = S_vec.z / nS + w * M_vec.z / nM;
        const S nd = sqrtf(dx * dx + dy * dy + dz * dz);
        const S sc = nS / fmaxf(nd, eps);
        v_conics[g].x = sc * dx;
        v_conics[g].y = sc * dy;
        v_conics[g].z = sc * dz;
        return;
    }

    S base_scale;
    if (rho_mode == 1)      base_scale = rho;
    else if (rho_mode == 2) base_scale = S(2) - rho;
    else                    base_scale = S(1);

    const S extra_x = fmaxf(A_vec.x - fabsf(S_vec.x), 0.f);
    const S extra_y = fmaxf(A_vec.y - fabsf(S_vec.y), 0.f);
    const S extra_z = fmaxf(A_vec.z - fabsf(S_vec.z), 0.f);
    const S mag_x = base_scale * fabsf(S_vec.x) + scale * extra_x;
    const S mag_y = base_scale * fabsf(S_vec.y) + scale * extra_y;
    const S mag_z = base_scale * fabsf(S_vec.z) + scale * extra_z;
    const S sign_x = (S_vec.x >= 0.f) ? 1.f : -1.f;
    const S sign_y = (S_vec.y >= 0.f) ? 1.f : -1.f;
    const S sign_z = (S_vec.z >= 0.f) ? 1.f : -1.f;
    v_conics[g].x = (S_vec.x == 0.f) ? 0.f : sign_x * mag_x;
    v_conics[g].y = (S_vec.y == 0.f) ? 0.f : sign_y * mag_y;
    v_conics[g].z = (S_vec.z == 0.f) ? 0.f : sign_z * mag_z;
}

// --- Phase 3 opacity scalar soft deconf post-kernel --- //
//
//     ρ_opac     = |S_opac| / (|S_opac| + λ_med · |M_opac| + ε)
//     κ_opac     = clip((|A_opac| − |S_opac|) / (|A_opac| + ε), 0, 1)
//     G_obj_opac = sign(S_opac) · (|S_opac| + γ_opacity · ρ_opac · κ_opac
//                                  · relu(|A_opac| − |S_opac|))
//     G_opac     = G_obj_opac + η_opacity · (1 − ρ_opac) · M_opac
//
// Soft correction (not a hard overwrite of medium term). η_opacity=0
// reduces to the standard "hard suppress medium" formula.
template <typename S>
__global__ void opacity_correction_kernel(
    const uint32_t total_primitives,
    const S *__restrict__ v_opacity_obj_signed,
    const S *__restrict__ v_opacity_obj_abs,
    const S *__restrict__ v_opacity_med_signed,
    S       *__restrict__ v_opacities,
    const S gamma_opacity,
    const S eta_opacity,
    const S lambda_med,
    const S eps,
    const S fade,
    const int rho_mode
) {
    const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total_primitives) return;

    const S S_o = v_opacity_obj_signed[g];
    const S A_o = v_opacity_obj_abs[g];
    const S M_o = v_opacity_med_signed[g];

    const S abs_S = fabsf(S_o);
    const S abs_M = fabsf(M_o);

    // Refinement fade: lerp every corrected write toward vanilla (S_o+M_o,
    // the kernel-input opacity gradient) by f. f=0 -> deconf; f=1 -> noSGA.
    const S f = (fade < S(0.f)) ? S(0.f) : ((fade > S(1.f)) ? S(1.f) : fade);
    const S van_o = v_opacities[g];   // vanilla S_o + M_o (kernel input)

    // rho_mode = 5 (blend): magnitude-anchored scalar analog of the vector
    // blend. The opacity gradient is a SCALAR, so "direction" reduces to
    // sign and cos(S,M) to a same-sign test. Keep the object gradient S_o in
    // full; admit the medium gradient M_o ONLY when it AGREES in sign with
    // S_o (the relu(cos) gate), and CLAMP its magnitude to |S_o| so the
    // medium (||M|| ~ 1e3·||S|| on opacity too) can never dominate the
    // existence decision:
    //   v = S_o + [sign(S_o)==sign(M_o)] · sign(M_o) · min(|M_o|, |S_o|)
    // bounded by 2|S_o| (vs the old additive η_o·M_o which was unbounded —
    // the over-opaque/over-sharpen source measured in grad/opacity_absmean).
    if (rho_mode == 5) {
        if (abs_S <= eps) { return; }   // no object anchor → vanilla (do-no-harm)
        S add = S(0.f);
        if (S_o * M_o > S(0.f)) {       // aligned: admit capped medium
            add = ((M_o >= 0.f) ? S(1.f) : S(-1.f)) * fminf(abs_M, abs_S);
        }
        v_opacities[g] = (S(1.f) - f) * (S_o + add) + f * van_o;
        return;
    }

    // True-zero guard only (see means2d kernel).
    if (A_o <= eps && abs_S <= eps) {
        if (abs_M > eps) v_opacities[g] = (S(1.f) - f) * (eta_opacity * M_o) + f * van_o;
        return;
    }

    const S rho   = abs_S / (abs_S + lambda_med * abs_M + eps);
    S kappa = (A_o - abs_S) / (A_o + eps);
    if (kappa < 0.f) kappa = 0.f;
    if (kappa > 1.f) kappa = 1.f;
    const S extra = fmaxf(A_o - abs_S, 0.f);
    const S mag   = abs_S + gamma_opacity * rho * kappa * extra;
    const S sign_S = (S_o >= 0.f) ? S(1.f) : S(-1.f);
    const S G_obj  = (S_o == 0.f) ? S(0.f) : sign_S * mag;
    v_opacities[g] = (S(1.f) - f) * (G_obj + eta_opacity * (S(1.f) - rho) * M_o)
                     + f * van_o;
}

// --- Phase 4 medium family-aware deconf post-kernels --- //
//
//     ρ_med    = |S_med| / (|S_med| + λ_obj · |C_med| + ε)
//     κ_med    = clip((|A_med| − |S_med|) / (|A_med| + ε), 0, 1)
//     G_med    = sign(S_med) · (|S_med| + γ_medium · ρ_med · κ_med
//                               · relu(|A_med| − |S_med|))
//
// Applied per pixel × per channel. v_medium_field carries the signed
// exact gradient (= S_med); we overwrite it with G_med. γ_medium=0
// short-circuits to a no-op.
template <typename S>
__global__ void medium_correction_kernel(
    const uint32_t total_elems,        // C * H * W * 3
    const S *__restrict__ a_medium,    // [C, H, W, 3]
    const S *__restrict__ c_medium,    // [C, H, W, 3]
    S       *__restrict__ v_medium,    // [C, H, W, 3]  signed in/out
    const S gamma_medium,
    const S lambda_obj,
    const S eps
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_elems) return;

    const S S_m = v_medium[i];
    const S A_m = a_medium[i];
    const S C_m = c_medium[i];

    const S abs_S = fabsf(S_m);

    // True-zero guard only (see means2d kernel).
    if (A_m <= eps && abs_S <= eps) {
        return;  // leave v_medium as-is (≈ 0)
    }

    const S rho   = abs_S / (abs_S + lambda_obj * C_m + eps);
    S kappa = (A_m - abs_S) / (A_m + eps);
    if (kappa < 0.f) kappa = 0.f;
    if (kappa > 1.f) kappa = 1.f;
    const S extra = fmaxf(A_m - abs_S, 0.f);
    const S mag   = abs_S + gamma_medium * rho * kappa * extra;
    const S sign_S = (S_m >= 0.f) ? S(1.f) : S(-1.f);
    v_medium[i] = (S_m == 0.f) ? S(0.f) : sign_S * mag;
}

template <typename S>
__global__ void medium_backscatter_gate_kernel(
    const uint32_t total_elems,        // C * H * W * 3
    const S *__restrict__ a_medium,    // [C, H, W, 3]
    const S *__restrict__ c_medium,    // [C, H, W, 3]
    S       *__restrict__ v_medium,    // [C, H, W, 3] signed exact in/out
    const S gamma_medium,
    const S lambda_obj,
    const S bs_floor,
    const S eps
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_elems) return;

    const S A_m = a_medium[i];
    if (A_m <= eps) return;

    S conflict = lambda_obj * c_medium[i] / (A_m + eps);
    if (conflict < S(0.f)) conflict = S(0.f);
    if (conflict > S(1.f)) conflict = S(1.f);

    S floor_clamped = bs_floor;
    if (floor_clamped < S(0.f)) floor_clamped = S(0.f);
    if (floor_clamped > S(1.f)) floor_clamped = S(1.f);

    S gamma_clamped = gamma_medium;
    if (gamma_clamped < S(0.f)) gamma_clamped = S(0.f);
    if (gamma_clamped > S(1.f)) gamma_clamped = S(1.f);

    const S gate = S(1.f) - gamma_clamped * (S(1.f) - floor_clamped) * conflict;
    v_medium[i] *= gate;
}

template <uint32_t CDIM>
std::tuple<
    torch::Tensor,   // v_means2d
    torch::Tensor,   // v_conics
    torch::Tensor,   // v_colors
    torch::Tensor,   // v_opacities
    torch::Tensor,   // v_betas
    torch::Tensor,   // v_med_rgb
    torch::Tensor,   // v_med_bs
    torch::Tensor,   // v_med_attn
    torch::Tensor,   // rho_per_primitive       [C, N]; empty when do_deconf=false
    torch::Tensor,   // cos_sm_per_primitive    [C, N]; cos(S, M), direction-confound test
    torch::Tensor,   // magratio_per_primitive  [C, N]; |M|_2 / |S|_2, magnitude test
    torch::Tensor,   // grad_vis_per_primitive  [C, N]; α-weighted T·mean(exp(-β_D·d)) compound
    torch::Tensor,   // T_per_view              [C, N]; α-weighted T only (occlusion)
    torch::Tensor>   // exp_per_view            [C, N]; α-weighted mean(exp(-β_D·d)) only (medium)
call_kernel_with_dim(
    // Gaussian parameters
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &betas,
    const at::optional<torch::Tensor> &backgrounds,
    const at::optional<torch::Tensor> &masks,
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
    // --- Dual-path geometry-correction config ---
    const bool  grad_deconf_enabled,
    const float grad_deconf_gamma,
    const float grad_deconf_lambda_med,
    const float grad_deconf_eps,
    const float grad_deconf_residual_eta,
    const float grad_deconf_fade,
    const bool  grad_deconf_apply_means2d,
    const bool  grad_deconf_apply_conic,
    const int   grad_deconf_rho_mode,
    // Phase 3 opacity soft correction
    const bool  grad_deconf_apply_opacity,
    const float grad_deconf_gamma_opacity,
    const float grad_deconf_eta_opacity,
    // Phase 4 medium symmetric correction
    const float grad_deconf_gamma_medium,
    const float grad_deconf_lambda_obj,
    const float grad_deconf_medium_bs_floor,
    const bool  grad_deconf_medium_attn_exact
) {
    GSPLAT_DEVICE_GUARD(means2d);
    GSPLAT_CHECK_INPUT(means2d);
    GSPLAT_CHECK_INPUT(conics);
    GSPLAT_CHECK_INPUT(colors);
    GSPLAT_CHECK_INPUT(opacities);
    GSPLAT_CHECK_INPUT(betas);
    GSPLAT_CHECK_INPUT(tile_offsets);
    GSPLAT_CHECK_INPUT(flatten_ids);
    GSPLAT_CHECK_INPUT(render_alphas);
    GSPLAT_CHECK_INPUT(last_ids);
    GSPLAT_CHECK_INPUT(v_render_colors);
    GSPLAT_CHECK_INPUT(v_render_alphas);
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
        GSPLAT_CHECK_INPUT(v_render_medium.value());
    }

    uint32_t C = tile_offsets.size(0);
    uint32_t N = means2d.size(1);
    uint32_t n_isects = flatten_ids.size(0);
    uint32_t COLOR_DIM = colors.size(-1);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);

    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, tile_height, tile_width};

    torch::Tensor v_means2d = torch::zeros_like(means2d);
    torch::Tensor v_conics = torch::zeros_like(conics);
    torch::Tensor v_colors = torch::zeros_like(colors);
    torch::Tensor v_opacities = torch::zeros_like(opacities);
    torch::Tensor v_betas = torch::zeros_like(betas);

    // Medium gradient tensors
    torch::Tensor v_med_rgb, v_med_bs, v_med_attn;
    if (has_medium) {
        v_med_rgb = torch::zeros_like(medium_rgb.value());
        v_med_bs = torch::zeros_like(medium_bs.value());
        v_med_attn = torch::zeros_like(medium_attn.value());
    } else {
        auto opts = means2d.options().dtype(torch::kFloat32);
        v_med_rgb = torch::empty({0}, opts);
        v_med_bs = torch::empty({0}, opts);
        v_med_attn = torch::empty({0}, opts);
    }

    // Dual-path auxiliary buffers: S (signed obj), A (abs obj), M (signed
    // med) for each corrected parameter block. Only allocated when the
    // feature is active and the medium branch is in use — otherwise the
    // kernel's pointers are null and the accumulation path is fully
    // skipped.
    const bool do_deconf = grad_deconf_enabled && has_medium && n_isects > 0;
    torch::Tensor v_xy_obj_signed, v_xy_obj_abs, v_xy_med_signed;
    torch::Tensor v_conic_obj_signed, v_conic_obj_abs, v_conic_med_signed;
    // Per-primitive gradient-flow diagnostic buffers exposed to Python.
    // Allocated [C, N] when dual-path is active, otherwise empty sentinels
    // so the Python side can test `buf.numel() > 0`.
    //   rho_per_primitive      — identifiability score ρ ∈ [0, 1]
    //   cos_sm_per_prim        — cos(S, M) ∈ [-1, 1]; direction confounding
    //   magratio_per_prim      — |M|_2 / |S|_2;        magnitude confounding
    //   grad_vis_per_primitive — α-weighted T·mean(exp(-β_D·d));
    //                            transmittance-starvation diagnostic (Mode #1).
    //                            Populated only when has_medium (otherwise
    //                            left as empty sentinel).
    torch::Tensor rho_per_primitive;
    torch::Tensor cos_sm_per_prim;
    torch::Tensor magratio_per_prim;
    torch::Tensor grad_vis_per_primitive;
    torch::Tensor T_per_view;          // alpha-weighted mean transmittance
    torch::Tensor exp_per_view;        // alpha-weighted mean exp(-β_D·d)
    torch::Tensor grad_vis_accum, grad_vis_weight;
    torch::Tensor T_accum, exp_attn_accum;
    // Phase 3 opacity split buffers + Phase 4 medium A/C buffers.
    torch::Tensor v_opacity_obj_signed_t, v_opacity_obj_abs_t, v_opacity_med_signed_t;
    torch::Tensor a_med_rgb_t, c_med_rgb_t;
    torch::Tensor a_med_bs_t,  c_med_bs_t;
    torch::Tensor a_med_attn_t, c_med_attn_t;
    // Declared at outer scope so the kernel-launch site can read it.
    const bool _need_medium_ac = do_deconf && has_medium && grad_deconf_gamma_medium > 0.f;
    if (do_deconf) {
        // Per-block allocation gated on apply_* flags so disabled blocks
        // skip both buffer alloc AND per-pixel compute/atomicAdds in the
        // kernel. means2d is the primary path and (almost) always allocated
        // when SGA is on; conic / opacity allocations skipped when their
        // respective post-correction is disabled.
        if (grad_deconf_apply_means2d) {
            v_xy_obj_signed = torch::zeros_like(means2d);
            v_xy_obj_abs    = torch::zeros_like(means2d);
            v_xy_med_signed = torch::zeros_like(means2d);
        }
        // rho/cos/magratio diagnostic buffers: populated by the means2d
        // export (apply_means2d) OR the opacity export (apply_opacity — the
        // floater-surgery config, where means2d is vanilla). Allocate when
        // EITHER channel will write them so the final means2d-off run still
        // logs sga/* from the OPACITY decomposition (signed scalar, so it
        // resolves the flip-vs-escape split the means2d magratio cannot).
        if (grad_deconf_apply_means2d || grad_deconf_apply_opacity) {
            rho_per_primitive  = torch::zeros_like(opacities);  // [C, N]
            cos_sm_per_prim    = torch::zeros_like(opacities);  // [C, N]
            magratio_per_prim  = torch::zeros_like(opacities);  // [C, N]
        } else {
            auto empty_opts = means2d.options().dtype(torch::kFloat32);
            rho_per_primitive = torch::empty({0}, empty_opts);
            cos_sm_per_prim   = torch::empty({0}, empty_opts);
            magratio_per_prim = torch::empty({0}, empty_opts);
        }
        if (grad_deconf_apply_conic) {
            v_conic_obj_signed = torch::zeros_like(conics);
            v_conic_obj_abs    = torch::zeros_like(conics);
            v_conic_med_signed = torch::zeros_like(conics);
        }
        if (grad_deconf_apply_opacity) {
            v_opacity_obj_signed_t = torch::zeros_like(opacities);
            v_opacity_obj_abs_t    = torch::zeros_like(opacities);
            v_opacity_med_signed_t = torch::zeros_like(opacities);
        }
        // Phase 4 medium A/C buffers — only allocated when the medium-family
        // post-correction is enabled (gamma_medium > 0). Otherwise the kernel
        // would atomicAdd into them for every pixel × Gaussian intersection
        // with no consumer reading the result. Cuts ~18 pixel-level atomic
        // adds per iter when gamma_medium=0 (the default).
        if (_need_medium_ac) {
            a_med_rgb_t  = torch::zeros_like(medium_rgb.value());
            c_med_rgb_t  = torch::zeros_like(medium_rgb.value());
            a_med_bs_t   = torch::zeros_like(medium_bs.value());
            c_med_bs_t   = torch::zeros_like(medium_bs.value());
            a_med_attn_t = torch::zeros_like(medium_attn.value());
            c_med_attn_t = torch::zeros_like(medium_attn.value());
        }
    } else {
        auto empty_opts = means2d.options().dtype(torch::kFloat32);
        rho_per_primitive = torch::empty({0}, empty_opts);
        cos_sm_per_prim   = torch::empty({0}, empty_opts);
        magratio_per_prim = torch::empty({0}, empty_opts);
    }
    // grad_vis + split (T / exp): purely observational diagnostics.
    // Gated on grad_deconf_enabled (= do_deconf) so SGA-OFF is bit-
    // identical performance to the baseline gsplat — without this gate
    // every backward paid 4 atomicAdds/Gaussian-intersection just to
    // populate diagnostics no consumer reads when SGA is off.
    const bool do_grad_vis = do_deconf;
    if (do_grad_vis) {
        grad_vis_accum         = torch::zeros_like(opacities);
        T_accum                = torch::zeros_like(opacities);
        exp_attn_accum         = torch::zeros_like(opacities);
        grad_vis_weight        = torch::zeros_like(opacities);
        grad_vis_per_primitive = torch::zeros_like(opacities);
        T_per_view             = torch::zeros_like(opacities);
        exp_per_view           = torch::zeros_like(opacities);
    } else {
        auto empty_opts = means2d.options().dtype(torch::kFloat32);
        grad_vis_per_primitive = torch::empty({0}, empty_opts);
        T_per_view             = torch::empty({0}, empty_opts);
        exp_per_view           = torch::empty({0}, empty_opts);
    }

    if (n_isects) {
        const uint32_t shared_mem =
            tile_size * tile_size *
            (sizeof(int32_t) + sizeof(vec4<float>) + sizeof(vec3<float>) +
             sizeof(float) * COLOR_DIM + sizeof(float));  // +depth_batch
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

        if (cudaFuncSetAttribute(
                rasterize_to_pixels_bwd_kernel<CDIM, float>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shared_mem
            ) != cudaSuccess) {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ",
                shared_mem,
                " bytes), try lowering tile_size."
            );
        }
        rasterize_to_pixels_bwd_kernel<CDIM, float>
            <<<blocks, threads, shared_mem, stream>>>(
                C,
                N,
                n_isects,
                reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                betas.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
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
                render_alphas.data_ptr<float>(),
                last_ids.data_ptr<int32_t>(),
                v_render_colors.data_ptr<float>(),
                v_render_alphas.data_ptr<float>(),
                has_medium ? v_render_medium.value().data_ptr<float>() : nullptr,
                reinterpret_cast<vec2<float> *>(v_means2d.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(v_conics.data_ptr<float>()),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(),
                v_betas.data_ptr<float>(),
                has_medium ? v_med_rgb.data_ptr<float>() : nullptr,
                has_medium ? v_med_bs.data_ptr<float>() : nullptr,
                has_medium ? v_med_attn.data_ptr<float>() : nullptr,
                // Per-block gating: only forward valid pointers when the
                // corresponding apply_* flag is on, so the kernel skips
                // accumulation for unused blocks entirely.
                (do_deconf && grad_deconf_apply_means2d) ? reinterpret_cast<vec2<float> *>(v_xy_obj_signed.data_ptr<float>()) : nullptr,
                (do_deconf && grad_deconf_apply_means2d) ? reinterpret_cast<vec2<float> *>(v_xy_obj_abs.data_ptr<float>())    : nullptr,
                (do_deconf && grad_deconf_apply_means2d) ? reinterpret_cast<vec2<float> *>(v_xy_med_signed.data_ptr<float>()) : nullptr,
                (do_deconf && grad_deconf_apply_conic)   ? reinterpret_cast<vec3<float> *>(v_conic_obj_signed.data_ptr<float>()) : nullptr,
                (do_deconf && grad_deconf_apply_conic)   ? reinterpret_cast<vec3<float> *>(v_conic_obj_abs.data_ptr<float>())    : nullptr,
                (do_deconf && grad_deconf_apply_conic)   ? reinterpret_cast<vec3<float> *>(v_conic_med_signed.data_ptr<float>()) : nullptr,
                do_grad_vis ? grad_vis_accum.data_ptr<float>()  : nullptr,
                do_grad_vis ? T_accum.data_ptr<float>()         : nullptr,
                do_grad_vis ? exp_attn_accum.data_ptr<float>()  : nullptr,
                do_grad_vis ? grad_vis_weight.data_ptr<float>() : nullptr,
                // Phase 3 opacity split buffers (nullptr when apply_opacity off)
                (do_deconf && grad_deconf_apply_opacity) ? v_opacity_obj_signed_t.data_ptr<float>() : nullptr,
                (do_deconf && grad_deconf_apply_opacity) ? v_opacity_obj_abs_t.data_ptr<float>()    : nullptr,
                (do_deconf && grad_deconf_apply_opacity) ? v_opacity_med_signed_t.data_ptr<float>() : nullptr,
                // Phase 4 medium A/C buffers (nullptr unless gamma_medium > 0)
                (do_deconf && _need_medium_ac) ? a_med_rgb_t.data_ptr<float>()  : nullptr,
                (do_deconf && _need_medium_ac) ? c_med_rgb_t.data_ptr<float>()  : nullptr,
                (do_deconf && _need_medium_ac) ? a_med_bs_t.data_ptr<float>()   : nullptr,
                (do_deconf && _need_medium_ac) ? c_med_bs_t.data_ptr<float>()   : nullptr,
                (do_deconf && _need_medium_ac) ? a_med_attn_t.data_ptr<float>() : nullptr,
                (do_deconf && _need_medium_ac) ? c_med_attn_t.data_ptr<float>() : nullptr
            );

        // Grad-visibility normalisation post-kernel (Mode #1 diagnostic).
        // Runs whenever the medium branch is active, independent of the
        // dual-path correction — the diagnostic is a pure observation and
        // should be available even when no correction is applied. Three
        // separate launches share the same normalisation kernel and the
        // same denominator (Σ α); only the numerator pointer changes.
        if (do_grad_vis) {
            const uint32_t total_gv = C * N;
            const uint32_t t_per_block_gv = 256;
            const uint32_t blocks_1d_gv = (total_gv + t_per_block_gv - 1) / t_per_block_gv;
            at::cuda::CUDAStream stream_gv = at::cuda::getCurrentCUDAStream();
            // Compound quantity grad_vis = α·T·exp / α.
            grad_vis_normalize_kernel<float>
                <<<blocks_1d_gv, t_per_block_gv, 0, stream_gv>>>(
                    total_gv,
                    grad_vis_accum.data_ptr<float>(),
                    grad_vis_weight.data_ptr<float>(),
                    grad_vis_per_primitive.data_ptr<float>(),
                    grad_deconf_eps
                );
            // Occlusion-only: T_per_view = α·T / α.
            grad_vis_normalize_kernel<float>
                <<<blocks_1d_gv, t_per_block_gv, 0, stream_gv>>>(
                    total_gv,
                    T_accum.data_ptr<float>(),
                    grad_vis_weight.data_ptr<float>(),
                    T_per_view.data_ptr<float>(),
                    grad_deconf_eps
                );
            // Medium-only: exp_per_view = α·mean(exp_attn) / α.
            grad_vis_normalize_kernel<float>
                <<<blocks_1d_gv, t_per_block_gv, 0, stream_gv>>>(
                    total_gv,
                    exp_attn_accum.data_ptr<float>(),
                    grad_vis_weight.data_ptr<float>(),
                    exp_per_view.data_ptr<float>(),
                    grad_deconf_eps
                );
        }

        // Post-kernels: soft dual-path corrections for means2d / conic.
        // Two separate launches for readability; each shares the same grid
        // layout and the same hyperparameters. (opacity was removed — its
        // per-primitive correction regressed empirically.)
        if (do_deconf) {
            const uint32_t total = C * N;
            const uint32_t t_per_block = 256;
            const uint32_t blocks_1d = (total + t_per_block - 1) / t_per_block;

            // ρ / cos(S,M) / |M|/|S| export: needs the means2d split
            // buffers; only runs when apply_means2d is enabled (those buffers
            // are otherwise skipped to save backward atomicAdds).
            if (grad_deconf_apply_means2d) {
                rho_export_kernel<float>
                    <<<blocks_1d, t_per_block, 0, stream>>>(
                        total,
                        reinterpret_cast<vec2<float> *>(v_xy_obj_signed.data_ptr<float>()),
                        reinterpret_cast<vec2<float> *>(v_xy_med_signed.data_ptr<float>()),
                        rho_per_primitive.data_ptr<float>(),
                        cos_sm_per_prim.data_ptr<float>(),
                        magratio_per_prim.data_ptr<float>(),
                        grad_deconf_lambda_med,
                        grad_deconf_eps
                    );
            }

            // OPACITY-channel diagnostic export: when the floater-surgery is the
            // active path (means2d off in the final config), fill the rho/cos/
            // magratio buffers from the OPACITY S_o/M_o split instead. Runs AFTER
            // the means2d export so if both channels are on, opacity (the surgery
            // channel) takes precedence. Reads the same input buffers the
            // correction kernel does (order-independent: correction writes
            // v_opacities, a different buffer).
            if (grad_deconf_apply_opacity) {
                opacity_rho_export_kernel<float>
                    <<<blocks_1d, t_per_block, 0, stream>>>(
                        total,
                        v_opacity_obj_signed_t.data_ptr<float>(),
                        v_opacity_med_signed_t.data_ptr<float>(),
                        rho_per_primitive.data_ptr<float>(),
                        cos_sm_per_prim.data_ptr<float>(),
                        magratio_per_prim.data_ptr<float>(),
                        grad_deconf_lambda_med,
                        grad_deconf_eps
                    );
            }

            if (grad_deconf_apply_means2d) {
                means2d_correction_kernel<float>
                    <<<blocks_1d, t_per_block, 0, stream>>>(
                        total,
                        reinterpret_cast<vec2<float> *>(v_xy_obj_signed.data_ptr<float>()),
                        reinterpret_cast<vec2<float> *>(v_xy_obj_abs.data_ptr<float>()),
                        reinterpret_cast<vec2<float> *>(v_xy_med_signed.data_ptr<float>()),
                        reinterpret_cast<vec2<float> *>(v_means2d.data_ptr<float>()),
                        grad_deconf_gamma,
                        grad_deconf_lambda_med,
                        grad_deconf_eps,
                        grad_deconf_residual_eta,
                        grad_deconf_fade,
                        grad_deconf_rho_mode
                    );
            }

            if (grad_deconf_apply_conic) {
                conic_correction_kernel<float>
                    <<<blocks_1d, t_per_block, 0, stream>>>(
                        total,
                        reinterpret_cast<vec3<float> *>(v_conic_obj_signed.data_ptr<float>()),
                        reinterpret_cast<vec3<float> *>(v_conic_obj_abs.data_ptr<float>()),
                        reinterpret_cast<vec3<float> *>(v_conic_med_signed.data_ptr<float>()),
                        reinterpret_cast<vec3<float> *>(v_conics.data_ptr<float>()),
                        grad_deconf_gamma,
                        grad_deconf_lambda_med,
                        grad_deconf_eps,
                        grad_deconf_fade,
                        grad_deconf_rho_mode
                    );
            }

            // Phase 3 — opacity soft correction (scalar per-primitive).
            if (grad_deconf_apply_opacity) {
                opacity_correction_kernel<float>
                    <<<blocks_1d, t_per_block, 0, stream>>>(
                        total,
                        v_opacity_obj_signed_t.data_ptr<float>(),
                        v_opacity_obj_abs_t.data_ptr<float>(),
                        v_opacity_med_signed_t.data_ptr<float>(),
                        v_opacities.data_ptr<float>(),
                        grad_deconf_gamma_opacity,
                        grad_deconf_eta_opacity,
                        grad_deconf_lambda_med,
                        grad_deconf_eps,
                        grad_deconf_fade,
                        grad_deconf_rho_mode
                    );
            }

            // Phase 4 — medium family-aware correction (per-pixel × 3 fields).
            if (has_medium && grad_deconf_gamma_medium > 0.f) {
                const uint32_t total_pix = C * image_height * image_width * 3u;
                const uint32_t blocks_pix = (total_pix + t_per_block - 1) / t_per_block;
                // medium_rgb / medium_bs: suppress only the conflict-weighted
                // additive medium gradients, with a floor so they never vanish.
                medium_backscatter_gate_kernel<float>
                    <<<blocks_pix, t_per_block, 0, stream>>>(
                        total_pix,
                        a_med_rgb_t.data_ptr<float>(),
                        c_med_rgb_t.data_ptr<float>(),
                        v_med_rgb.data_ptr<float>(),
                        grad_deconf_gamma_medium,
                        grad_deconf_lambda_obj,
                        grad_deconf_medium_bs_floor,
                        grad_deconf_eps
                    );
                medium_backscatter_gate_kernel<float>
                    <<<blocks_pix, t_per_block, 0, stream>>>(
                        total_pix,
                        a_med_bs_t.data_ptr<float>(),
                        c_med_bs_t.data_ptr<float>(),
                        v_med_bs.data_ptr<float>(),
                        grad_deconf_gamma_medium,
                        grad_deconf_lambda_obj,
                        grad_deconf_medium_bs_floor,
                        grad_deconf_eps
                    );
                // medium_attn is exact by default: attenuation remains valid
                // on object pixels and should not be suppressed as backscatter.
                if (!grad_deconf_medium_attn_exact) {
                    medium_correction_kernel<float>
                        <<<blocks_pix, t_per_block, 0, stream>>>(
                            total_pix,
                            a_med_attn_t.data_ptr<float>(),
                            c_med_attn_t.data_ptr<float>(),
                            v_med_attn.data_ptr<float>(),
                            grad_deconf_gamma_medium,
                            grad_deconf_lambda_obj,
                            grad_deconf_eps
                        );
                }
            }

        }
    }

    return std::make_tuple(
        v_means2d, v_conics, v_colors, v_opacities, v_betas,
        v_med_rgb, v_med_bs, v_med_attn,
        rho_per_primitive, cos_sm_per_prim, magratio_per_prim,
        grad_vis_per_primitive, T_per_view, exp_per_view
    );
}

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor>
rasterize_to_pixels_bwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const torch::Tensor &betas,
    const at::optional<torch::Tensor> &backgrounds,
    const at::optional<torch::Tensor> &masks,
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
    // --- Dual-path geometry-correction config ---
    const bool  grad_deconf_enabled,
    const float grad_deconf_gamma,
    const float grad_deconf_lambda_med,
    const float grad_deconf_eps,
    const float grad_deconf_residual_eta,
    const float grad_deconf_fade,
    const bool  grad_deconf_apply_means2d,
    const bool  grad_deconf_apply_conic,
    const int   grad_deconf_rho_mode,
    // --- Phase 3 opacity soft correction ---
    const bool  grad_deconf_apply_opacity,
    const float grad_deconf_gamma_opacity,
    const float grad_deconf_eta_opacity,
    // --- Phase 4 medium symmetric correction ---
    const float grad_deconf_gamma_medium,
    const float grad_deconf_lambda_obj,
    const float grad_deconf_medium_bs_floor,
    const bool  grad_deconf_medium_attn_exact
) {
    GSPLAT_CHECK_INPUT(colors);
    uint32_t COLOR_DIM = colors.size(-1);

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
            flatten_ids,                                                       \
            render_alphas,                                                     \
            last_ids,                                                          \
            v_render_colors,                                                   \
            v_render_alphas,                                                   \
            v_render_medium,                                                   \
            grad_deconf_enabled,                                               \
            grad_deconf_gamma,                                                 \
            grad_deconf_lambda_med,                                            \
            grad_deconf_eps,                                                   \
            grad_deconf_residual_eta,                                          \
            grad_deconf_fade,                                                  \
            grad_deconf_apply_means2d,                                         \
            grad_deconf_apply_conic,                                           \
            grad_deconf_rho_mode,                                              \
            grad_deconf_apply_opacity,                                         \
            grad_deconf_gamma_opacity,                                         \
            grad_deconf_eta_opacity,                                           \
            grad_deconf_gamma_medium,                                          \
            grad_deconf_lambda_obj,                                            \
            grad_deconf_medium_bs_floor,                                       \
            grad_deconf_medium_attn_exact                                      \
        );

    switch (COLOR_DIM) {
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
        AT_ERROR("Unsupported number of channels: ", COLOR_DIM);
    }
}

} // namespace gsplat
