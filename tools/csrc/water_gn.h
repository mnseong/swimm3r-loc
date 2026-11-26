#pragma once
#include <cmath>

#ifdef __CUDACC__
#define WGN_HD __host__ __device__ __forceinline__
#else
#define WGN_HD inline
#endif

#define WGN_P 9
#define WGN_NSUM 54

struct WgnParams {
  float R[9];
  float t[3];
  float fx, fy, cx, cy, scale;
  int H, W, C, N;
  int transport;
};

WGN_HD float wgn_bilinear(const float* img, int W, float u, float v) {
  int x0 = (int)floorf(u), y0 = (int)floorf(v);
  float ax = u - (float)x0, ay = v - (float)y0;
  const float* r0 = img + (long)y0 * W;
  const float* r1 = r0 + W;
  return (1.0f - ay) * ((1.0f - ax) * r0[x0] + ax * r0[x0 + 1]) + ay * ((1.0f - ax) * r1[x0] + ax * r1[x0 + 1]);
}

WGN_HD bool wgn_project(const WgnParams& p, const float* X, float* Y, float* u, float* v) {
  Y[0] = p.R[0] * X[0] + p.R[1] * X[1] + p.R[2] * X[2] + p.t[0];
  Y[1] = p.R[3] * X[0] + p.R[4] * X[1] + p.R[5] * X[2] + p.t[1];
  Y[2] = p.R[6] * X[0] + p.R[7] * X[1] + p.R[8] * X[2] + p.t[2];
  float z = Y[2];
  float zs = z > 1e-6f ? z : 1.0f;
  *u = p.fx * Y[0] / zs + p.cx;
  *v = p.fy * Y[1] / zs + p.cy;
  return z > 1e-6f && *u >= 1.0f && *u <= (float)p.W - 2.0f && *v >= 1.0f && *v <= (float)p.H - 2.0f;
}

WGN_HD void wgn_model(float J, float B, float bD, float bB, float z, int transport, float* m, float* dm) {
  float tD = expf(-bD * z), tB = expf(-bB * z);
  *m = J * tD + B * (1.0f - tB);
  *dm = transport ? (-bD * J * tD + bB * B * tB) : 0.0f;
}

WGN_HD void wgn_row(const WgnParams& p, const float* Y, int c, float q, float qx, float qy, float g, float dm, float* row) {
  float z = Y[2];
  float a0 = g * qx * p.fx / z;
  float a1 = g * qy * p.fy / z;
  float a2 = -g * (qx * p.fx * Y[0] + qy * p.fy * Y[1]) / (z * z);
  row[0] = p.scale * a0;
  row[1] = p.scale * a1;
  row[2] = p.scale * (a2 - dm);
  row[3] = Y[1] * a2 - Y[2] * a1 - dm * Y[1];
  row[4] = Y[2] * a0 - Y[0] * a2 + dm * Y[0];
  row[5] = Y[0] * a1 - Y[1] * a0;
  row[6] = c == 0 ? q : 0.0f;
  row[7] = c == 1 ? q : 0.0f;
  row[8] = c == 2 ? q : 0.0f;
}

WGN_HD float wgn_weight(float r, float sig, float k) {
  float ar = fabsf(r);
  float w = ar > 1e-12f ? k * sig / ar : 1.0f;
  return w < 1.0f ? w : 1.0f;
}

WGN_HD void wgn_accumulate(const float* row, float w, float r, float* acc) {
  int k = 0;
  for (int i = 0; i < WGN_P; ++i)
    for (int j = i; j < WGN_P; ++j) acc[k++] += w * row[i] * row[j];
  for (int i = 0; i < WGN_P; ++i) acc[45 + i] += w * row[i] * r;
}
