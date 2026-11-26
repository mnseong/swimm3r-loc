#include <torch/extension.h>

#include <cmath>
#include <vector>

#include "water_gn.h"

#ifdef WITH_CUDA
void wgn_pass_a_cuda(const WgnParams& p, const float* stack, const float* X, const float* Iv, const float* Dv, const float* z0, float* Y, float* q,
                     float* qx, float* qy, float* m, float* dm, unsigned char* use, double* sums);
void wgn_pass_b_cuda(const WgnParams& p, const float* Y, const float* q, const float* qx, const float* qy, const float* m, const float* dm,
                     const unsigned char* use, const float* g, const float* sig, float huber_k, double* hb);
#endif

static void pass_a_cpu(const WgnParams& p, const float* stack, const float* X, const float* Iv, const float* Dv, const float* z0, float* Y, float* q,
                       float* qx, float* qy, float* m, float* dm, unsigned char* use, double* sums) {
  const int C = p.C, N = p.N;
  const long HW = (long)p.H * p.W;
  for (int i = 0; i < N; ++i) {
    float u, v;
    bool ok = wgn_project(p, X + 3 * i, Y + 3 * i, &u, &v);
    use[i] = ok ? 1 : 0;
    for (int c = 0; c < C; ++c) {
      long k = (long)c * N + i;
      if (!ok) {
        q[k] = qx[k] = qy[k] = m[k] = dm[k] = 0.0f;
        continue;
      }
      q[k] = wgn_bilinear(stack + c * HW, p.W, u, v);
      qx[k] = wgn_bilinear(stack + (C + c) * HW, p.W, u, v);
      qy[k] = wgn_bilinear(stack + (2 * C + c) * HW, p.W, u, v);
      wgn_model(Iv[k], Dv[k], Y[3 * i + 2], z0[i], p.transport, &m[k], &dm[k]);
    }
    if (ok) sums[0] += 1.0;
  }
}

static void pass_b_cpu(const WgnParams& p, const float* Y, const float* q, const float* qx, const float* qy, const float* m, const float* dm,
                       const unsigned char* use, const float* g, const float* sig, float huber_k, double* hb) {
  const int C = p.C, N = p.N;
  for (int i = 0; i < N; ++i) {
    if (!use[i]) continue;
    float acc[WGN_NSUM] = {0.0f};
    for (int c = 0; c < C; ++c) {
      long k = (long)c * N + i;
      float r = g[c] * q[k] - m[k];
      float w = wgn_weight(r, sig[0], huber_k);
      float row[WGN_P];
      wgn_row(p, Y + 3 * i, c, q[k], qx[k], qy[k], g[c], dm[k], row);
      wgn_accumulate(row, w, r, acc);
    }
    for (int k = 0; k < WGN_NSUM; ++k) hb[k] += acc[k];
  }
}

static void rodrigues(const double* w, double* R) {
  double th = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
  double K[9] = {0.0, -w[2], w[1], w[2], 0.0, -w[0], -w[1], w[0], 0.0};
  double K2[9];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      double s = 0.0;
      for (int l = 0; l < 3; ++l) s += K[3 * i + l] * K[3 * l + j];
      K2[3 * i + j] = s;
    }
  double a = th < 1e-9 ? 1.0 : std::sin(th) / th;
  double b = th < 1e-9 ? 0.0 : (1.0 - std::cos(th)) / (th * th);
  for (int i = 0; i < 9; ++i) R[i] = (i % 4 == 0 ? 1.0 : 0.0) + a * K[i] + b * K2[i];
}

static bool solve_normal(const double* hb, double* x) {
  const int P = WGN_P;
  double A[WGN_P][WGN_P + 1];
  int k = 0;
  for (int i = 0; i < P; ++i)
    for (int j = i; j < P; ++j) {
      A[i][j] = hb[k];
      A[j][i] = hb[k];
      ++k;
    }
  double tr = 0.0;
  for (int i = 0; i < P; ++i) tr += A[i][i];
  for (int i = 0; i < P; ++i) {
    A[i][i] += 1e-6 * tr / (double)P;
    A[i][P] = -hb[45 + i];
  }
  for (int c = 0; c < P; ++c) {
    int piv = c;
    for (int r = c + 1; r < P; ++r)
      if (std::fabs(A[r][c]) > std::fabs(A[piv][c])) piv = r;
    if (std::fabs(A[piv][c]) < 1e-300) return false;
    if (piv != c)
      for (int j = 0; j <= P; ++j) std::swap(A[c][j], A[piv][j]);
    for (int r = c + 1; r < P; ++r) {
      double f = A[r][c] / A[c][c];
      for (int j = c; j <= P; ++j) A[r][j] -= f * A[c][j];
    }
  }
  for (int i = P - 1; i >= 0; --i) {
    double s = A[i][P];
    for (int j = i + 1; j < P; ++j) s -= A[i][j] * x[j];
    x[i] = s / A[i][i];
  }
  for (int i = 0; i < P; ++i)
    if (!std::isfinite(x[i])) return false;
  return true;
}

static torch::Tensor mad_scale(const torch::Tensor& r) {
  auto rf = r.reshape({-1});
  int64_t n = rf.numel();
  int64_t stride = std::max<int64_t>(1, (n + 16383) / 16384);
  auto rs = rf.slice(0, 0, n, stride);
  auto med = rs.nanmedian();
  return (1.4826 * (rs - med).abs().nanmedian() + 1e-6).to(torch::kFloat32).reshape({1}).contiguous();
}

std::vector<torch::Tensor> align_level(torch::Tensor stack, torch::Tensor X, torch::Tensor Iv, torch::Tensor Dv, torch::Tensor z0, torch::Tensor St,
                                       double fx, double fy, double cx, double cy, double scale, bool transport, double huber_k, int64_t iters,
                                       double tol, int64_t min_use) {
  TORCH_CHECK(stack.dim() == 3 && stack.size(0) % 3 == 0, "stack must be (3C, H, W)");
  TORCH_CHECK(stack.scalar_type() == torch::kFloat32 && X.scalar_type() == torch::kFloat32, "float32 tensors expected");
  TORCH_CHECK(stack.size(0) / 3 <= 3, "at most 3 colour channels");
  const bool cuda = stack.is_cuda();
#ifndef WITH_CUDA
  TORCH_CHECK(!cuda, "water_gn_ext was built without CUDA");
#endif
  stack = stack.contiguous();
  X = X.contiguous();
  Iv = Iv.contiguous();
  Dv = Dv.contiguous();
  z0 = z0.contiguous();
  WgnParams p;
  p.C = (int)(stack.size(0) / 3);
  p.H = (int)stack.size(1);
  p.W = (int)stack.size(2);
  p.N = (int)X.size(0);
  p.fx = (float)fx;
  p.fy = (float)fy;
  p.cx = (float)cx;
  p.cy = (float)cy;
  p.scale = (float)scale;
  p.transport = transport ? 1 : 0;
  auto fo = stack.options();
  auto Y = torch::zeros({p.N, 3}, fo);
  auto q = torch::zeros({p.C, p.N}, fo), qx = torch::zeros({p.C, p.N}, fo), qy = torch::zeros({p.C, p.N}, fo);
  auto m = torch::zeros({p.C, p.N}, fo), dm = torch::zeros({p.C, p.N}, fo);
  auto use = torch::zeros({p.N}, fo.dtype(torch::kUInt8));
  auto sums = torch::zeros({1}, fo.dtype(torch::kFloat64));
  auto hb = torch::zeros({WGN_NSUM}, fo.dtype(torch::kFloat64));
  auto st = St.to(torch::kCPU, torch::kFloat64).contiguous();
  TORCH_CHECK(st.numel() == 15, "state must hold R (9), t (3) and the gains (3)");
  double S[15], best[15];
  for (int i = 0; i < 15; ++i) S[i] = best[i] = st.data_ptr<double>()[i];
  int steps = 0, n_use = 0;
  double last = -1.0, best_sig = -1.0;
  bool fresh = false;
  auto evaluate = [&](const double* s, bool with_b, double* hbh, double* sig_out) {
    for (int i = 0; i < 9; ++i) p.R[i] = (float)s[i];
    for (int i = 0; i < 3; ++i) p.t[i] = (float)s[9 + i];
    sums.zero_();
    hb.zero_();
    if (cuda) {
#ifdef WITH_CUDA
      wgn_pass_a_cuda(p, stack.data_ptr<float>(), X.data_ptr<float>(), Iv.data_ptr<float>(), Dv.data_ptr<float>(), z0.data_ptr<float>(),
                      Y.data_ptr<float>(), q.data_ptr<float>(), qx.data_ptr<float>(), qy.data_ptr<float>(), m.data_ptr<float>(), dm.data_ptr<float>(),
                      use.data_ptr<unsigned char>(), sums.data_ptr<double>());
#endif
    } else {
      pass_a_cpu(p, stack.data_ptr<float>(), X.data_ptr<float>(), Iv.data_ptr<float>(), Dv.data_ptr<float>(), z0.data_ptr<float>(), Y.data_ptr<float>(),
                 q.data_ptr<float>(), qx.data_ptr<float>(), qy.data_ptr<float>(), m.data_ptr<float>(), dm.data_ptr<float>(), use.data_ptr<unsigned char>(),
                 sums.data_ptr<double>());
    }
    auto g = torch::tensor({(float)s[12], (float)s[13], (float)s[14]}, torch::dtype(torch::kFloat32)).narrow(0, 0, p.C).to(stack.device()).contiguous();
    auto ub = use.to(torch::kBool).unsqueeze(0).expand({p.C, p.N});
    auto r = torch::where(ub, g.unsqueeze(1) * q - m, torch::full_like(q, NAN));
    auto sig = mad_scale(r);
    if (with_b) {
      if (cuda) {
#ifdef WITH_CUDA
        wgn_pass_b_cuda(p, Y.data_ptr<float>(), q.data_ptr<float>(), qx.data_ptr<float>(), qy.data_ptr<float>(), m.data_ptr<float>(),
                        dm.data_ptr<float>(), use.data_ptr<unsigned char>(), g.data_ptr<float>(), sig.data_ptr<float>(), (float)huber_k,
                        hb.data_ptr<double>());
#endif
      } else {
        pass_b_cpu(p, Y.data_ptr<float>(), q.data_ptr<float>(), qx.data_ptr<float>(), qy.data_ptr<float>(), m.data_ptr<float>(), dm.data_ptr<float>(),
                   use.data_ptr<unsigned char>(), g.data_ptr<float>(), sig.data_ptr<float>(), (float)huber_k, hb.data_ptr<double>());
      }
    }
    auto both = torch::cat({hb, sums, sig.to(torch::kFloat64)}).to(torch::kCPU);
    const double* h = both.data_ptr<double>();
    for (int k = 0; k < WGN_NSUM; ++k) hbh[k] = h[k];
    *sig_out = h[WGN_NSUM + 1];
    return (int)h[WGN_NSUM];
  };
  double hbh[WGN_NSUM], sig = 0.0;
  for (int64_t it = 0; it < iters; ++it) {
    n_use = evaluate(S, true, hbh, &sig);
    fresh = true;
    if (n_use < min_use) break;
    if (best_sig >= 0.0 && sig > best_sig * 1.02) {
      for (int i = 0; i < 15; ++i) S[i] = best[i];
      --steps;
      fresh = false;
      break;
    }
    best_sig = sig;
    for (int i = 0; i < 15; ++i) best[i] = S[i];
    double xi[WGN_P];
    if (!solve_normal(hbh, xi)) break;
    double Rw[9];
    rodrigues(xi + 3, Rw);
    double Rn[9], tn[3];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        double s2 = 0.0;
        for (int l = 0; l < 3; ++l) s2 += Rw[3 * i + l] * S[3 * l + j];
        Rn[3 * i + j] = s2;
      }
      tn[i] = Rw[3 * i] * S[9] + Rw[3 * i + 1] * S[10] + Rw[3 * i + 2] * S[11] + scale * xi[i];
    }
    for (int i = 0; i < 9; ++i) S[i] = Rn[i];
    for (int i = 0; i < 3; ++i) S[9 + i] = tn[i];
    for (int c = 0; c < p.C; ++c) {
      double gn = S[12 + c] + xi[6 + c];
      S[12 + c] = gn < 0.25 ? 0.25 : (gn > 4.0 ? 4.0 : gn);
    }
    fresh = false;
    ++steps;
    double nt = std::sqrt(xi[0] * xi[0] + xi[1] * xi[1] + xi[2] * xi[2]);
    double nr = std::sqrt(xi[3] * xi[3] + xi[4] * xi[4] + xi[5] * xi[5]);
    last = nt > nr ? nt : nr;
    if (last < tol) break;
  }
  if (!fresh) {
    n_use = evaluate(S, false, hbh, &sig);
    if (best_sig >= 0.0 && (n_use < min_use || sig > best_sig * 1.02)) {
      for (int i = 0; i < 15; ++i) S[i] = best[i];
      --steps;
      n_use = evaluate(S, false, hbh, &sig);
    }
  }
  if (steps < 0) steps = 0;
  auto out = torch::zeros({15}, torch::dtype(torch::kFloat64));
  for (int i = 0; i < 15; ++i) out.data_ptr<double>()[i] = S[i];
  auto info = torch::tensor({(double)steps, last, (double)n_use, sig}, torch::dtype(torch::kFloat64));
  return {out, info, q, m, use};
}

bool has_cuda() {
#ifdef WITH_CUDA
  return true;
#else
  return false;
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("align_level", &align_level, "Gauss Newton of one pyramid level, the water model inside the residual");
  mod.def("has_cuda", &has_cuda, "built with the CUDA kernels");
}
