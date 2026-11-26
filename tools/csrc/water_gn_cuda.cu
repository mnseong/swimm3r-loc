#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include "water_gn.h"

#define WGN_THREADS 128

template <int K>
__device__ void wgn_block_add(float (*sh)[WGN_THREADS], const float* vals, double* out) {
  int tid = threadIdx.x;
  for (int k = 0; k < K; ++k) sh[k][tid] = vals[k];
  __syncthreads();
  for (int s = WGN_THREADS / 2; s > 0; s >>= 1) {
    if (tid < s)
      for (int k = 0; k < K; ++k) sh[k][tid] += sh[k][tid + s];
    __syncthreads();
  }
  if (tid == 0)
    for (int k = 0; k < K; ++k) atomicAdd(out + k, (double)sh[k][0]);
}

__global__ void wgn_pass_a_kernel(WgnParams p, const float* stack, const float* X, const float* Jv, const float* Bv, const float* bD, const float* bB,
                                  const float* z0, float* Y, float* q, float* qx, float* qy, float* m, float* dm, unsigned char* use, double* sums) {
  __shared__ float sh[1][WGN_THREADS];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float vals[1] = {0.0f};
  if (i < p.N) {
    float u, v;
    bool ok = wgn_project(p, X + 3 * i, Y + 3 * i, &u, &v);
    use[i] = ok ? 1 : 0;
    const long HW = (long)p.H * p.W;
    for (int c = 0; c < p.C; ++c) {
      long k = (long)c * p.N + i;
      if (!ok) {
        q[k] = qx[k] = qy[k] = m[k] = dm[k] = 0.0f;
        continue;
      }
      q[k] = wgn_bilinear(stack + c * HW, p.W, u, v);
      qx[k] = wgn_bilinear(stack + (p.C + c) * HW, p.W, u, v);
      qy[k] = wgn_bilinear(stack + (2 * p.C + c) * HW, p.W, u, v);
      float zm = p.transport ? Y[3 * i + 2] : z0[i];
      wgn_model(Jv[k], Bv[k], bD[k], bB[k], zm, p.transport, &m[k], &dm[k]);
    }
    vals[0] = ok ? 1.0f : 0.0f;
  }
  wgn_block_add<1>(sh, vals, sums);
}

__global__ void wgn_pass_b_kernel(WgnParams p, const float* Y, const float* q, const float* qx, const float* qy, const float* m, const float* dm,
                                  const unsigned char* use, const float* g, const float* sig, float huber_k, double* hb) {
  __shared__ float sh[WGN_NSUM][WGN_THREADS];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float acc[WGN_NSUM];
  for (int k = 0; k < WGN_NSUM; ++k) acc[k] = 0.0f;
  if (i < p.N && use[i]) {
    for (int c = 0; c < p.C; ++c) {
      long k = (long)c * p.N + i;
      float r = g[c] * q[k] - m[k];
      float w = wgn_weight(r, sig[0], huber_k);
      float row[WGN_P];
      wgn_row(p, Y + 3 * i, c, q[k], qx[k], qy[k], g[c], dm[k], row);
      wgn_accumulate(row, w, r, acc);
    }
  }
  wgn_block_add<WGN_NSUM>(sh, acc, hb);
}

void wgn_pass_a_cuda(const WgnParams& p, const float* stack, const float* X, const float* Jv, const float* Bv, const float* bD, const float* bB,
                     const float* z0, float* Y, float* q, float* qx, float* qy, float* m, float* dm, unsigned char* use, double* sums) {
  int blocks = (p.N + WGN_THREADS - 1) / WGN_THREADS;
  if (blocks == 0) return;
  wgn_pass_a_kernel<<<blocks, WGN_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(p, stack, X, Jv, Bv, bD, bB, z0, Y, q, qx, qy, m, dm, use, sums);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wgn_pass_b_cuda(const WgnParams& p, const float* Y, const float* q, const float* qx, const float* qy, const float* m, const float* dm,
                     const unsigned char* use, const float* g, const float* sig, float huber_k, double* hb) {
  int blocks = (p.N + WGN_THREADS - 1) / WGN_THREADS;
  if (blocks == 0) return;
  wgn_pass_b_kernel<<<blocks, WGN_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(p, Y, q, qx, qy, m, dm, use, g, sig, huber_k, hb);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
