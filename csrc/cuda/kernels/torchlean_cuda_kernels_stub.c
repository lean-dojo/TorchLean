#include <lean/lean.h>

#include "torchlean_cuda_buffer.h"

#include <math.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

// CPU fallback for TorchLean's general tensor-kernel FFI surface.
//
// This file mirrors `torchlean_cuda_kernels.cu` symbol-for-symbol so the Lean runtime can link on
// machines without CUDA. Keep the shape checks and edge-case conventions in lockstep with the CUDA
// file; these stubs are also useful as readable reference implementations for tests and audits.

static const float kTorchLeanPiF = 3.14159265358979323846f;

static size_t checked_mul_size_stub(size_t a, size_t b, const char* msg) {
  if (a != 0 && b > SIZE_MAX / a) {
    lean_internal_panic(msg);
  }
  return a * b;
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_sum_axis0(b_lean_obj_arg BObj, uint32_t rows,
                                                               uint32_t cols) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  const size_t R = (size_t)rows;
  const size_t C = (size_t)cols;
  if (b->size != R * C) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_sum_axis0_stub: size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(C);
  for (size_t j = 0; j < C; ++j) {
    float acc = 0.0f;
    for (size_t i = 0; i < R; ++i) {
      acc += b->data[i * C + j];
    }
    out->data[j] = acc;
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_sum_axis1(b_lean_obj_arg BObj, uint32_t rows,
                                                               uint32_t cols) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  const size_t R = (size_t)rows;
  const size_t C = (size_t)cols;
  if (b->size != R * C) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_sum_axis1_stub: size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(R);
  for (size_t i = 0; i < R; ++i) {
    float acc = 0.0f;
    for (size_t j = 0; j < C; ++j) {
      acc += b->data[i * C + j];
    }
    out->data[i] = acc;
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_max_axis0(b_lean_obj_arg BObj, uint32_t rows,
                                                               uint32_t cols) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  const size_t R = (size_t)rows;
  const size_t C = (size_t)cols;
  if (b->size != R * C) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_max_axis0_stub: size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(C);
  if (R == 0) {
    for (size_t j = 0; j < C; ++j) {
      out->data[j] = 0.0f;
    }
    return torchlean_cuda_buffer_box(out);
  }
  for (size_t j = 0; j < C; ++j) {
    float m = -INFINITY;
    for (size_t i = 0; i < R; ++i) {
      m = fmaxf(m, b->data[i * C + j]);
    }
    out->data[j] = m;
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_max_axis1(b_lean_obj_arg BObj, uint32_t rows,
                                                               uint32_t cols) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  const size_t R = (size_t)rows;
  const size_t C = (size_t)cols;
  if (b->size != R * C) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_max_axis1_stub: size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(R);
  if (C == 0) {
    for (size_t i = 0; i < R; ++i) {
      out->data[i] = 0.0f;
    }
    return torchlean_cuda_buffer_box(out);
  }
  for (size_t i = 0; i < R; ++i) {
    float m = -INFINITY;
    for (size_t j = 0; j < C; ++j) {
      m = fmaxf(m, b->data[i * C + j]);
    }
    out->data[i] = m;
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_concat1d(b_lean_obj_arg AObj, b_lean_obj_arg BObj,
                                                       uint32_t n, uint32_t m) {
  torchlean_cuda_buffer* a = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  const size_t N = (size_t)n;
  const size_t M = (size_t)m;
  if (a->size != N) {
    lean_internal_panic("torchlean_cuda_buffer_concat1d_stub: a.size mismatch");
  }
  if (b->size != M) {
    lean_internal_panic("torchlean_cuda_buffer_concat1d_stub: b.size mismatch");
  }

  const size_t total = N + M;
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(total);
  for (size_t i = 0; i < N; ++i) out->data[i] = a->data[i];
  for (size_t j = 0; j < M; ++j) out->data[N + j] = b->data[j];
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_slice1d(b_lean_obj_arg BObj, uint32_t n,
                                                      uint32_t start, uint32_t len) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  const size_t N = (size_t)n;
  const size_t S = (size_t)start;
  const size_t L = (size_t)len;
  if (b->size != N) {
    lean_internal_panic("torchlean_cuda_buffer_slice1d_stub: size mismatch");
  }
  if (S > N || S + L > N) {
    lean_internal_panic("torchlean_cuda_buffer_slice1d_stub: start+len out of bounds");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(L);
  for (size_t i = 0; i < L; ++i) out->data[i] = b->data[S + i];
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_broadcast_vec_to_rows(b_lean_obj_arg VObj,
                                                                    uint32_t rows, uint32_t cols) {
  torchlean_cuda_buffer* v = torchlean_cuda_buffer_unbox(VObj);
  const size_t R = (size_t)rows;
  const size_t C = (size_t)cols;
  if (v->size != C) {
    lean_internal_panic("torchlean_cuda_buffer_broadcast_vec_to_rows_stub: vec.size mismatch");
  }

  const size_t total = R * C;
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(total);
  for (size_t i = 0; i < R; ++i) {
    for (size_t j = 0; j < C; ++j) {
      out->data[i * C + j] = v->data[j];
    }
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_broadcast_vec_to_cols(b_lean_obj_arg VObj,
                                                                    uint32_t rows, uint32_t cols) {
  torchlean_cuda_buffer* v = torchlean_cuda_buffer_unbox(VObj);
  const size_t R = (size_t)rows;
  const size_t C = (size_t)cols;
  if (v->size != R) {
    lean_internal_panic("torchlean_cuda_buffer_broadcast_vec_to_cols_stub: vec.size mismatch");
  }

  const size_t total = R * C;
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(total);
  for (size_t i = 0; i < R; ++i) {
    for (size_t j = 0; j < C; ++j) {
      out->data[i * C + j] = v->data[i];
    }
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_bmm(b_lean_obj_arg AObj, b_lean_obj_arg BObj,
                                                  uint32_t batch, uint32_t m, uint32_t n,
                                                  uint32_t p) {
  torchlean_cuda_buffer* A = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* B = torchlean_cuda_buffer_unbox(BObj);

  const size_t Batch = (size_t)batch;
  const size_t M = (size_t)m;
  const size_t N = (size_t)n;
  const size_t P = (size_t)p;

  const size_t aSz = Batch * M * N;
  const size_t bSz = Batch * N * P;
  const size_t cSz = Batch * M * P;

  if (A->size != aSz) {
    lean_internal_panic("torchlean_cuda_buffer_bmm_stub: A.size mismatch");
  }
  if (B->size != bSz) {
    lean_internal_panic("torchlean_cuda_buffer_bmm_stub: B.size mismatch");
  }

  torchlean_cuda_buffer* C = torchlean_cuda_buffer_alloc(cSz);

  for (size_t t = 0; t < Batch; ++t) {
    const float* aT = A->data + t * (M * N);
    const float* bT = B->data + t * (N * P);
    float* cT = C->data + t * (M * P);
    for (size_t i = 0; i < M; ++i) {
      for (size_t k = 0; k < P; ++k) {
        float acc = 0.0f;
        for (size_t j = 0; j < N; ++j) {
          acc += aT[i * N + j] * bT[j * P + k];
        }
        cT[i * P + k] = acc;
      }
    }
  }

  return torchlean_cuda_buffer_box(C);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_rfft1d_packed(b_lean_obj_arg XObj, uint32_t batch,
                                                            uint32_t n) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  if (n == 0) {
    lean_internal_panic("torchlean_cuda_buffer_rfft1d_packed_stub: n must be positive");
  }

  const size_t Batch = (size_t)batch;
  const size_t N = (size_t)n;
  const size_t Freq = N / 2 + 1;
  const size_t inSz =
      checked_mul_size_stub(Batch, N, "torchlean_cuda_buffer_rfft1d_packed_stub: input size overflow");
  const size_t complexSz = checked_mul_size_stub(
      Batch, Freq, "torchlean_cuda_buffer_rfft1d_packed_stub: spectrum size overflow");
  const size_t outSz = checked_mul_size_stub(
      complexSz, (size_t)2, "torchlean_cuda_buffer_rfft1d_packed_stub: packed size overflow");
  if (x->size != inSz) {
    lean_internal_panic("torchlean_cuda_buffer_rfft1d_packed_stub: size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outSz);
  for (size_t b = 0; b < Batch; ++b) {
    const float* xb = x->data + b * N;
    float* ob = out->data + b * Freq * 2;
    for (size_t k = 0; k < Freq; ++k) {
      float re = 0.0f;
      float im = 0.0f;
      for (size_t t = 0; t < N; ++t) {
        const float angle = 2.0f * kTorchLeanPiF * (float)k * (float)t / (float)N;
        re += xb[t] * cosf(angle);
        im -= xb[t] * sinf(angle);
      }
      ob[2 * k] = re;
      ob[2 * k + 1] = im;
    }
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_irfft1d_packed(b_lean_obj_arg SpecObj,
                                                             uint32_t batch, uint32_t n) {
  torchlean_cuda_buffer* spec = torchlean_cuda_buffer_unbox(SpecObj);
  if (n == 0) {
    lean_internal_panic("torchlean_cuda_buffer_irfft1d_packed_stub: n must be positive");
  }

  const size_t Batch = (size_t)batch;
  const size_t N = (size_t)n;
  const size_t Freq = N / 2 + 1;
  const size_t complexSz = checked_mul_size_stub(
      Batch, Freq, "torchlean_cuda_buffer_irfft1d_packed_stub: spectrum size overflow");
  const size_t specSz = checked_mul_size_stub(
      complexSz, (size_t)2, "torchlean_cuda_buffer_irfft1d_packed_stub: packed size overflow");
  const size_t outSz =
      checked_mul_size_stub(Batch, N, "torchlean_cuda_buffer_irfft1d_packed_stub: output size overflow");
  if (spec->size != specSz) {
    lean_internal_panic("torchlean_cuda_buffer_irfft1d_packed_stub: size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outSz);
  const int hasNyquist = (N % 2 == 0);
  const size_t nyquist = N / 2;
  for (size_t b = 0; b < Batch; ++b) {
    const float* sb = spec->data + b * Freq * 2;
    float* yb = out->data + b * N;
    for (size_t t = 0; t < N; ++t) {
      float sum = sb[0];
      const size_t lastExclusive = hasNyquist ? nyquist : Freq;
      for (size_t k = 1; k < lastExclusive; ++k) {
        const float re = sb[2 * k];
        const float im = sb[2 * k + 1];
        const float angle = 2.0f * kTorchLeanPiF * (float)k * (float)t / (float)N;
        sum += 2.0f * (re * cosf(angle) - im * sinf(angle));
      }
      if (hasNyquist) {
        const float re = sb[2 * nyquist];
        const float angle = kTorchLeanPiF * (float)t;
        sum += re * cosf(angle);
      }
      yb[t] = sum / (float)N;
    }
  }
  return torchlean_cuda_buffer_box(out);
}

static void validate_spectral_conv1d_stub(torchlean_cuda_buffer* x, torchlean_cuda_buffer* wRe,
                                          torchlean_cuda_buffer* wIm, torchlean_cuda_buffer* dY,
                                          uint32_t grid, uint32_t width, uint32_t modes,
                                          size_t* xSzOut, size_t* wSzOut, size_t* freqOut) {
  if (grid == 0 || width == 0) {
    lean_internal_panic("spectralConv1dRfft_stub: grid and width must be positive");
  }
  const size_t Freq = (size_t)grid / 2 + 1;
  if ((size_t)modes > Freq) {
    lean_internal_panic("spectralConv1dRfft_stub: modes exceeds rfft frequency count");
  }
  const size_t xSz =
      checked_mul_size_stub((size_t)grid, (size_t)width, "spectralConv1dRfft_stub: x size overflow");
  const size_t wSz = checked_mul_size_stub(
      checked_mul_size_stub((size_t)modes, (size_t)width, "spectralConv1dRfft_stub: w size overflow"),
      (size_t)width, "spectralConv1dRfft_stub: w size overflow");
  if (x->size != xSz) {
    lean_internal_panic("spectralConv1dRfft_stub: x size mismatch");
  }
  if (wRe->size != wSz || wIm->size != wSz) {
    lean_internal_panic("spectralConv1dRfft_stub: weight size mismatch");
  }
  if (dY != NULL && dY->size != xSz) {
    lean_internal_panic("spectralConv1dRfft_stub: dY size mismatch");
  }
  *xSzOut = xSz;
  *wSzOut = wSz;
  *freqOut = Freq;
}

static void spectral_conv1d_rfft_ref(const float* x, float* re, float* im, size_t grid,
                                     size_t width, size_t modes) {
  for (size_t k = 0; k < modes; ++k) {
    for (size_t c = 0; c < width; ++c) {
      float xr = 0.0f;
      float xi = 0.0f;
      for (size_t t = 0; t < grid; ++t) {
        const float angle = 2.0f * kTorchLeanPiF * (float)k * (float)t / (float)grid;
        const float v = x[t * width + c];
        xr += v * cosf(angle);
        xi -= v * sinf(angle);
      }
      re[k * width + c] = xr;
      im[k * width + c] = xi;
    }
  }
}

static void spectral_conv1d_dz_from_dy_ref(const float* dY, float* dzRe, float* dzIm, size_t grid,
                                          size_t width, size_t modes) {
  spectral_conv1d_rfft_ref(dY, dzRe, dzIm, grid, width, modes);
  const size_t nyquist = grid / 2;
  const int hasNyquist = (grid % 2 == 0);
  for (size_t k = 0; k < modes; ++k) {
    const int edge = (k == 0) || (hasNyquist && k == nyquist);
    const float scale = (edge ? 1.0f : 2.0f) / (float)grid;
    for (size_t c = 0; c < width; ++c) {
      dzRe[k * width + c] *= scale;
      dzIm[k * width + c] *= scale;
    }
  }
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_spectral_conv1d_rfft_fwd(
    b_lean_obj_arg XObj, b_lean_obj_arg WReObj, b_lean_obj_arg WImObj, uint32_t grid,
    uint32_t width, uint32_t modes) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* wRe = torchlean_cuda_buffer_unbox(WReObj);
  torchlean_cuda_buffer* wIm = torchlean_cuda_buffer_unbox(WImObj);
  size_t xSz = 0, wSz = 0, Freq = 0;
  validate_spectral_conv1d_stub(x, wRe, wIm, NULL, grid, width, modes, &xSz, &wSz, &Freq);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(xSz);
  float* xRe = (float*)calloc((size_t)modes * (size_t)width, sizeof(float));
  float* xIm = (float*)calloc((size_t)modes * (size_t)width, sizeof(float));
  float* zRe = (float*)calloc((size_t)modes * (size_t)width, sizeof(float));
  float* zIm = (float*)calloc((size_t)modes * (size_t)width, sizeof(float));
  if (!xRe || !xIm || !zRe || !zIm) lean_internal_panic_out_of_memory();
  spectral_conv1d_rfft_ref(x->data, xRe, xIm, (size_t)grid, (size_t)width, (size_t)modes);
  for (size_t k = 0; k < (size_t)modes; ++k) {
    for (size_t o = 0; o < (size_t)width; ++o) {
      float zr = 0.0f;
      float zi = 0.0f;
      for (size_t c = 0; c < (size_t)width; ++c) {
        const size_t sIdx = k * (size_t)width + c;
        const size_t wIdx = (k * (size_t)width + c) * (size_t)width + o;
        zr += xRe[sIdx] * wRe->data[wIdx] - xIm[sIdx] * wIm->data[wIdx];
        zi += xRe[sIdx] * wIm->data[wIdx] + xIm[sIdx] * wRe->data[wIdx];
      }
      zRe[k * (size_t)width + o] = zr;
      zIm[k * (size_t)width + o] = zi;
    }
  }
  for (size_t t = 0; t < (size_t)grid; ++t) {
    for (size_t o = 0; o < (size_t)width; ++o) {
      float sum = zRe[o];
      for (size_t k = 1; k < (size_t)modes; ++k) {
        const int isNyquist = ((size_t)grid % 2 == 0) && (k == (size_t)grid / 2);
        const float factor = isNyquist ? 1.0f : 2.0f;
        const float angle = 2.0f * kTorchLeanPiF * (float)k * (float)t / (float)grid;
        sum += factor * (zRe[k * (size_t)width + o] * cosf(angle) -
                         zIm[k * (size_t)width + o] * sinf(angle));
      }
      out->data[t * (size_t)width + o] = sum / (float)grid;
    }
  }
  free(xRe); free(xIm); free(zRe); free(zIm);
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_spectral_conv1d_rfft_bwd_x(
    b_lean_obj_arg XObj, b_lean_obj_arg WReObj, b_lean_obj_arg WImObj, b_lean_obj_arg DYObj,
    uint32_t grid, uint32_t width, uint32_t modes) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* wRe = torchlean_cuda_buffer_unbox(WReObj);
  torchlean_cuda_buffer* wIm = torchlean_cuda_buffer_unbox(WImObj);
  torchlean_cuda_buffer* dY = torchlean_cuda_buffer_unbox(DYObj);
  size_t xSz = 0, wSz = 0, Freq = 0;
  validate_spectral_conv1d_stub(x, wRe, wIm, dY, grid, width, modes, &xSz, &wSz, &Freq);
  torchlean_cuda_buffer* dx = torchlean_cuda_buffer_alloc(xSz);
  float* dzRe = (float*)calloc((size_t)modes * (size_t)width, sizeof(float));
  float* dzIm = (float*)calloc((size_t)modes * (size_t)width, sizeof(float));
  float* dXRe = (float*)calloc((size_t)modes * (size_t)width, sizeof(float));
  float* dXIm = (float*)calloc((size_t)modes * (size_t)width, sizeof(float));
  if (!dzRe || !dzIm || !dXRe || !dXIm) lean_internal_panic_out_of_memory();
  spectral_conv1d_dz_from_dy_ref(dY->data, dzRe, dzIm, (size_t)grid, (size_t)width, (size_t)modes);
  for (size_t k = 0; k < (size_t)modes; ++k) {
    for (size_t c = 0; c < (size_t)width; ++c) {
      float xr = 0.0f;
      float xi = 0.0f;
      for (size_t o = 0; o < (size_t)width; ++o) {
        const size_t zIdx = k * (size_t)width + o;
        const size_t wIdx = (k * (size_t)width + c) * (size_t)width + o;
        xr += dzRe[zIdx] * wRe->data[wIdx] + dzIm[zIdx] * wIm->data[wIdx];
        xi += dzIm[zIdx] * wRe->data[wIdx] - dzRe[zIdx] * wIm->data[wIdx];
      }
      dXRe[k * (size_t)width + c] = xr;
      dXIm[k * (size_t)width + c] = xi;
    }
  }
  for (size_t t = 0; t < (size_t)grid; ++t) {
    for (size_t c = 0; c < (size_t)width; ++c) {
      float acc = 0.0f;
      for (size_t k = 0; k < (size_t)modes; ++k) {
        const float angle = 2.0f * kTorchLeanPiF * (float)k * (float)t / (float)grid;
        const size_t idx = k * (size_t)width + c;
        acc += dXRe[idx] * cosf(angle) - dXIm[idx] * sinf(angle);
      }
      dx->data[t * (size_t)width + c] = acc;
    }
  }
  free(dzRe); free(dzIm); free(dXRe); free(dXIm);
  return torchlean_cuda_buffer_box(dx);
}

static lean_obj_res spectral_conv1d_bwd_w_stub_common(
    b_lean_obj_arg XObj, b_lean_obj_arg WReObj, b_lean_obj_arg WImObj, b_lean_obj_arg DYObj,
    uint32_t grid, uint32_t width, uint32_t modes, int imag) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* wRe = torchlean_cuda_buffer_unbox(WReObj);
  torchlean_cuda_buffer* wIm = torchlean_cuda_buffer_unbox(WImObj);
  torchlean_cuda_buffer* dY = torchlean_cuda_buffer_unbox(DYObj);
  size_t xSz = 0, wSz = 0, Freq = 0;
  validate_spectral_conv1d_stub(x, wRe, wIm, dY, grid, width, modes, &xSz, &wSz, &Freq);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(wSz);
  float* xRe = (float*)calloc((size_t)modes * (size_t)width, sizeof(float));
  float* xIm = (float*)calloc((size_t)modes * (size_t)width, sizeof(float));
  float* dzRe = (float*)calloc((size_t)modes * (size_t)width, sizeof(float));
  float* dzIm = (float*)calloc((size_t)modes * (size_t)width, sizeof(float));
  if (!xRe || !xIm || !dzRe || !dzIm) lean_internal_panic_out_of_memory();
  spectral_conv1d_rfft_ref(x->data, xRe, xIm, (size_t)grid, (size_t)width, (size_t)modes);
  spectral_conv1d_dz_from_dy_ref(dY->data, dzRe, dzIm, (size_t)grid, (size_t)width, (size_t)modes);
  for (size_t k = 0; k < (size_t)modes; ++k) {
    for (size_t c = 0; c < (size_t)width; ++c) {
      for (size_t o = 0; o < (size_t)width; ++o) {
        const size_t xIdx = k * (size_t)width + c;
        const size_t zIdx = k * (size_t)width + o;
        const size_t wIdx = (k * (size_t)width + c) * (size_t)width + o;
        out->data[wIdx] = imag
            ? (xRe[xIdx] * dzIm[zIdx] - xIm[xIdx] * dzRe[zIdx])
            : (xRe[xIdx] * dzRe[zIdx] + xIm[xIdx] * dzIm[zIdx]);
      }
    }
  }
  free(xRe); free(xIm); free(dzRe); free(dzIm);
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_spectral_conv1d_rfft_bwd_wre(
    b_lean_obj_arg XObj, b_lean_obj_arg WReObj, b_lean_obj_arg WImObj, b_lean_obj_arg DYObj,
    uint32_t grid, uint32_t width, uint32_t modes) {
  return spectral_conv1d_bwd_w_stub_common(XObj, WReObj, WImObj, DYObj, grid, width, modes, 0);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_spectral_conv1d_rfft_bwd_wim(
    b_lean_obj_arg XObj, b_lean_obj_arg WReObj, b_lean_obj_arg WImObj, b_lean_obj_arg DYObj,
    uint32_t grid, uint32_t width, uint32_t modes) {
  return spectral_conv1d_bwd_w_stub_common(XObj, WReObj, WImObj, DYObj, grid, width, modes, 1);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_selective_scan_diag_fwd(
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, b_lean_obj_arg XObj, b_lean_obj_arg H0Obj,
    uint32_t seqLen, uint32_t stateDim) {
  torchlean_cuda_buffer* A = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* B = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* X = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* h0 = torchlean_cuda_buffer_unbox(H0Obj);

  const size_t T = (size_t)seqLen;
  const size_t D = (size_t)stateDim;
  if (D != 0 && T > SIZE_MAX / D) {
    lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_fwd_stub: seqLen*state overflow");
  }
  const size_t total = T * D;

  if (A->size != D) {
    lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_fwd_stub: A.size mismatch");
  }
  if (B->size != D) {
    lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_fwd_stub: B.size mismatch");
  }
  if (h0->size != D) {
    lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_fwd_stub: h0.size mismatch");
  }
  if (X->size != total) {
    lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_fwd_stub: X.size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(total);
  for (size_t j = 0; j < D; ++j) {
    float h = h0->data[j];
    for (size_t t = 0; t < T; ++t) {
      const size_t idx = t * D + j;
      h = A->data[j] * h + B->data[j] * X->data[idx];
      out->data[idx] = h;
    }
  }

  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_selective_scan_diag_bwd(
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, b_lean_obj_arg XObj, b_lean_obj_arg H0Obj,
    b_lean_obj_arg OutObj, b_lean_obj_arg DYObj, uint32_t seqLen, uint32_t stateDim) {
  torchlean_cuda_buffer* A = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* B = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* X = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* h0 = torchlean_cuda_buffer_unbox(H0Obj);
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_unbox(OutObj);
  torchlean_cuda_buffer* dY = torchlean_cuda_buffer_unbox(DYObj);

  const size_t T = (size_t)seqLen;
  const size_t D = (size_t)stateDim;
  if (D != 0 && T > SIZE_MAX / D) {
    lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_bwd_stub: seqLen*state overflow");
  }
  const size_t total = T * D;

  if (A->size != D) lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_bwd_stub: A.size mismatch");
  if (B->size != D) lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_bwd_stub: B.size mismatch");
  if (h0->size != D) lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_bwd_stub: h0.size mismatch");
  if (X->size != total) lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_bwd_stub: X.size mismatch");
  if (out->size != total) lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_bwd_stub: out.size mismatch");
  if (dY->size != total) lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_bwd_stub: dY.size mismatch");

  torchlean_cuda_buffer* dA = torchlean_cuda_buffer_alloc(D);
  torchlean_cuda_buffer* dB = torchlean_cuda_buffer_alloc(D);
  torchlean_cuda_buffer* dX = torchlean_cuda_buffer_alloc(total);
  torchlean_cuda_buffer* dH0 = torchlean_cuda_buffer_alloc(D);

  for (size_t j = 0; j < D; ++j) {
    float accA = 0.0f;
    float accB = 0.0f;
    float dhNext = 0.0f;
    const float a = A->data[j];
    const float b = B->data[j];
    for (size_t tr = 0; tr < T; ++tr) {
      const size_t t = T - 1u - tr;
      const size_t idx = t * D + j;
      const float hPrev = (t == 0u) ? h0->data[j] : out->data[(t - 1u) * D + j];
      const float totalGrad = dY->data[idx] + dhNext;
      accA += totalGrad * hPrev;
      accB += totalGrad * X->data[idx];
      dX->data[idx] = totalGrad * b;
      dhNext = totalGrad * a;
    }
    dA->data[j] = accA;
    dB->data[j] = accB;
    dH0->data[j] = dhNext;
  }

  lean_object* inner2 = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(inner2, 0, torchlean_cuda_buffer_box(dX));
  lean_ctor_set(inner2, 1, torchlean_cuda_buffer_box(dH0));
  lean_object* inner1 = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(inner1, 0, torchlean_cuda_buffer_box(dB));
  lean_ctor_set(inner1, 1, inner2);
  lean_object* outer = lean_alloc_ctor(0, 2, 0);
  lean_ctor_set(outer, 0, torchlean_cuda_buffer_box(dA));
  lean_ctor_set(outer, 1, inner1);
  return outer;
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_selective_scan_diag_var_fwd(
    b_lean_obj_arg AObj, b_lean_obj_arg BObj, b_lean_obj_arg XObj, b_lean_obj_arg H0Obj,
    uint32_t seqLen, uint32_t stateDim) {
  torchlean_cuda_buffer* A = torchlean_cuda_buffer_unbox(AObj);
  torchlean_cuda_buffer* B = torchlean_cuda_buffer_unbox(BObj);
  torchlean_cuda_buffer* X = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* h0 = torchlean_cuda_buffer_unbox(H0Obj);

  const size_t T = (size_t)seqLen;
  const size_t D = (size_t)stateDim;
  if (D != 0 && T > SIZE_MAX / D) {
    lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_var_fwd_stub: seqLen*state overflow");
  }
  const size_t total = T * D;

  if (A->size != total) {
    lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_var_fwd_stub: A.size mismatch");
  }
  if (B->size != total) {
    lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_var_fwd_stub: B.size mismatch");
  }
  if (X->size != total) {
    lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_var_fwd_stub: X.size mismatch");
  }
  if (h0->size != D) {
    lean_internal_panic("torchlean_cuda_buffer_selective_scan_diag_var_fwd_stub: h0.size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(total);
  for (size_t j = 0; j < D; ++j) {
    float h = h0->data[j];
    for (size_t t = 0; t < T; ++t) {
      const size_t idx = t * D + j;
      h = A->data[idx] * h + B->data[idx] * X->data[idx];
      out->data[idx] = h;
    }
  }

  return torchlean_cuda_buffer_box(out);
}

static inline int flash_attention_allowed_stub(const torchlean_cuda_buffer* mask, uint32_t hasMask,
                                               size_t batchIdx, size_t i, size_t j, size_t n) {
  if (hasMask == 0) return 1;
  return mask->data[(batchIdx * n + i) * n + j] != 0.0f;
}

static inline float flash_attention_score_stub(const torchlean_cuda_buffer* Q,
                                               const torchlean_cuda_buffer* K,
                                               const torchlean_cuda_buffer* mask,
                                               uint32_t hasMask, size_t batchIdx, size_t i,
                                               size_t j, size_t n, size_t d, float scale) {
  if (!flash_attention_allowed_stub(mask, hasMask, batchIdx, i, j, n)) return -1000.0f;
  float dot = 0.0f;
  const size_t qBase = (batchIdx * n + i) * d;
  const size_t kBase = (batchIdx * n + j) * d;
  for (size_t k = 0; k < d; ++k) dot += Q->data[qBase + k] * K->data[kBase + k];
  return dot * scale;
}

static inline void flash_attention_row_stats_stub(const torchlean_cuda_buffer* Q,
                                                  const torchlean_cuda_buffer* K,
                                                  const torchlean_cuda_buffer* mask,
                                                  uint32_t hasMask, size_t batchIdx, size_t i,
                                                  size_t n, size_t d, float scale,
                                                  float* rowMax, float* denom) {
  float m = -INFINITY;
  for (size_t j = 0; j < n; ++j) {
    float s = flash_attention_score_stub(Q, K, mask, hasMask, batchIdx, i, j, n, d, scale);
    if (s > m) m = s;
  }
  float z = 0.0f;
  for (size_t j = 0; j < n; ++j) {
    float s = flash_attention_score_stub(Q, K, mask, hasMask, batchIdx, i, j, n, d, scale);
    z += expf(s - m);
  }
  *rowMax = m;
  *denom = z;
}

static inline float flash_attention_prob_stub(const torchlean_cuda_buffer* Q,
                                              const torchlean_cuda_buffer* K,
                                              const torchlean_cuda_buffer* mask,
                                              uint32_t hasMask, size_t batchIdx, size_t i,
                                              size_t j, size_t n, size_t d, float scale,
                                              float rowMax, float denom) {
  float s = flash_attention_score_stub(Q, K, mask, hasMask, batchIdx, i, j, n, d, scale);
  return expf(s - rowMax) / denom;
}

static inline float flash_attention_d_attn_stub(const torchlean_cuda_buffer* V,
                                                const torchlean_cuda_buffer* dOut,
                                                size_t batchIdx, size_t i, size_t j,
                                                size_t n, size_t d) {
  float acc = 0.0f;
  const size_t dOutBase = (batchIdx * n + i) * d;
  const size_t vBase = (batchIdx * n + j) * d;
  for (size_t k = 0; k < d; ++k) acc += dOut->data[dOutBase + k] * V->data[vBase + k];
  return acc;
}

static void flash_attention_check_stub(const char* label, const torchlean_cuda_buffer* Q,
                                       const torchlean_cuda_buffer* K,
                                       const torchlean_cuda_buffer* V,
                                       const torchlean_cuda_buffer* mask,
                                       const torchlean_cuda_buffer* dOut, uint32_t hasMask,
                                       uint32_t batch, uint32_t n, uint32_t d) {
  const size_t qkvSz = (size_t)batch * (size_t)n * (size_t)d;
  const size_t maskSz = (size_t)batch * (size_t)n * (size_t)n;
  if (Q->size != qkvSz || K->size != qkvSz || V->size != qkvSz) {
    lean_internal_panic(label);
  }
  if (dOut && dOut->size != qkvSz) {
    lean_internal_panic(label);
  }
  if (hasMask != 0 && mask->size != maskSz) {
    lean_internal_panic(label);
  }
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_flash_attention_fwd(
    b_lean_obj_arg QObj, b_lean_obj_arg KObj, b_lean_obj_arg VObj, b_lean_obj_arg MaskObj,
    uint32_t hasMask, uint32_t batch, uint32_t n, uint32_t d, double scaleHost) {
  torchlean_cuda_buffer* Q = torchlean_cuda_buffer_unbox(QObj);
  torchlean_cuda_buffer* K = torchlean_cuda_buffer_unbox(KObj);
  torchlean_cuda_buffer* V = torchlean_cuda_buffer_unbox(VObj);
  torchlean_cuda_buffer* mask = torchlean_cuda_buffer_unbox(MaskObj);
  flash_attention_check_stub("torchlean_cuda_buffer_flash_attention_fwd_stub: size mismatch",
                             Q, K, V, mask, NULL, hasMask, batch, n, d);
  const size_t qkvSz = (size_t)batch * (size_t)n * (size_t)d;
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(qkvSz);
  const float scale = (float)scaleHost;
  for (size_t b = 0; b < batch; ++b) {
    for (size_t i = 0; i < n; ++i) {
      float rowMax = 0.0f, denom = 0.0f;
      flash_attention_row_stats_stub(Q, K, mask, hasMask, b, i, n, d, scale, &rowMax, &denom);
      for (size_t dv = 0; dv < d; ++dv) {
        float acc = 0.0f;
        for (size_t j = 0; j < n; ++j) {
          float p = flash_attention_prob_stub(Q, K, mask, hasMask, b, i, j, n, d, scale,
                                              rowMax, denom);
          acc += p * V->data[(b * n + j) * d + dv];
        }
        out->data[(b * n + i) * d + dv] = acc;
      }
    }
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_flash_attention_bwd_q(
    b_lean_obj_arg QObj, b_lean_obj_arg KObj, b_lean_obj_arg VObj, b_lean_obj_arg MaskObj,
    b_lean_obj_arg DOutObj, uint32_t hasMask, uint32_t batch, uint32_t n, uint32_t d,
    double scaleHost) {
  torchlean_cuda_buffer* Q = torchlean_cuda_buffer_unbox(QObj);
  torchlean_cuda_buffer* K = torchlean_cuda_buffer_unbox(KObj);
  torchlean_cuda_buffer* V = torchlean_cuda_buffer_unbox(VObj);
  torchlean_cuda_buffer* mask = torchlean_cuda_buffer_unbox(MaskObj);
  torchlean_cuda_buffer* dOut = torchlean_cuda_buffer_unbox(DOutObj);
  flash_attention_check_stub("torchlean_cuda_buffer_flash_attention_bwd_q_stub: size mismatch",
                             Q, K, V, mask, dOut, hasMask, batch, n, d);
  const size_t qkvSz = (size_t)batch * (size_t)n * (size_t)d;
  torchlean_cuda_buffer* dQ = torchlean_cuda_buffer_alloc(qkvSz);
  const float scale = (float)scaleHost;
  for (size_t b = 0; b < batch; ++b) {
    for (size_t i = 0; i < n; ++i) {
      float rowMax = 0.0f, denom = 0.0f;
      flash_attention_row_stats_stub(Q, K, mask, hasMask, b, i, n, d, scale, &rowMax, &denom);
      float rowDot = 0.0f;
      for (size_t j = 0; j < n; ++j) {
        float p = flash_attention_prob_stub(Q, K, mask, hasMask, b, i, j, n, d, scale,
                                            rowMax, denom);
        rowDot += p * flash_attention_d_attn_stub(V, dOut, b, i, j, n, d);
      }
      for (size_t k = 0; k < d; ++k) {
        float acc = 0.0f;
        for (size_t j = 0; j < n; ++j) {
          if (!flash_attention_allowed_stub(mask, hasMask, b, i, j, n)) continue;
          float p = flash_attention_prob_stub(Q, K, mask, hasMask, b, i, j, n, d, scale,
                                              rowMax, denom);
          float dAttn = flash_attention_d_attn_stub(V, dOut, b, i, j, n, d);
          acc += p * (dAttn - rowDot) * scale * K->data[(b * n + j) * d + k];
        }
        dQ->data[(b * n + i) * d + k] = acc;
      }
    }
  }
  return torchlean_cuda_buffer_box(dQ);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_flash_attention_bwd_k(
    b_lean_obj_arg QObj, b_lean_obj_arg KObj, b_lean_obj_arg VObj, b_lean_obj_arg MaskObj,
    b_lean_obj_arg DOutObj, uint32_t hasMask, uint32_t batch, uint32_t n, uint32_t d,
    double scaleHost) {
  torchlean_cuda_buffer* Q = torchlean_cuda_buffer_unbox(QObj);
  torchlean_cuda_buffer* K = torchlean_cuda_buffer_unbox(KObj);
  torchlean_cuda_buffer* V = torchlean_cuda_buffer_unbox(VObj);
  torchlean_cuda_buffer* mask = torchlean_cuda_buffer_unbox(MaskObj);
  torchlean_cuda_buffer* dOut = torchlean_cuda_buffer_unbox(DOutObj);
  flash_attention_check_stub("torchlean_cuda_buffer_flash_attention_bwd_k_stub: size mismatch",
                             Q, K, V, mask, dOut, hasMask, batch, n, d);
  const size_t qkvSz = (size_t)batch * (size_t)n * (size_t)d;
  torchlean_cuda_buffer* dK = torchlean_cuda_buffer_alloc(qkvSz);
  const float scale = (float)scaleHost;
  for (size_t b = 0; b < batch; ++b) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t k = 0; k < d; ++k) {
        float acc = 0.0f;
        for (size_t i = 0; i < n; ++i) {
          float rowMax = 0.0f, denom = 0.0f;
          flash_attention_row_stats_stub(Q, K, mask, hasMask, b, i, n, d, scale, &rowMax, &denom);
          float rowDot = 0.0f;
          for (size_t t = 0; t < n; ++t) {
            float p = flash_attention_prob_stub(Q, K, mask, hasMask, b, i, t, n, d, scale,
                                                rowMax, denom);
            rowDot += p * flash_attention_d_attn_stub(V, dOut, b, i, t, n, d);
          }
          if (!flash_attention_allowed_stub(mask, hasMask, b, i, j, n)) continue;
          float p = flash_attention_prob_stub(Q, K, mask, hasMask, b, i, j, n, d, scale,
                                              rowMax, denom);
          float dAttn = flash_attention_d_attn_stub(V, dOut, b, i, j, n, d);
          acc += p * (dAttn - rowDot) * scale * Q->data[(b * n + i) * d + k];
        }
        dK->data[(b * n + j) * d + k] = acc;
      }
    }
  }
  return torchlean_cuda_buffer_box(dK);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_flash_attention_bwd_v(
    b_lean_obj_arg QObj, b_lean_obj_arg KObj, b_lean_obj_arg VObj, b_lean_obj_arg MaskObj,
    b_lean_obj_arg DOutObj, uint32_t hasMask, uint32_t batch, uint32_t n, uint32_t d,
    double scaleHost) {
  torchlean_cuda_buffer* Q = torchlean_cuda_buffer_unbox(QObj);
  torchlean_cuda_buffer* K = torchlean_cuda_buffer_unbox(KObj);
  torchlean_cuda_buffer* V = torchlean_cuda_buffer_unbox(VObj);
  torchlean_cuda_buffer* mask = torchlean_cuda_buffer_unbox(MaskObj);
  torchlean_cuda_buffer* dOut = torchlean_cuda_buffer_unbox(DOutObj);
  flash_attention_check_stub("torchlean_cuda_buffer_flash_attention_bwd_v_stub: size mismatch",
                             Q, K, V, mask, dOut, hasMask, batch, n, d);
  const size_t qkvSz = (size_t)batch * (size_t)n * (size_t)d;
  torchlean_cuda_buffer* dV = torchlean_cuda_buffer_alloc(qkvSz);
  const float scale = (float)scaleHost;
  for (size_t b = 0; b < batch; ++b) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t dv = 0; dv < d; ++dv) {
        float acc = 0.0f;
        for (size_t i = 0; i < n; ++i) {
          float rowMax = 0.0f, denom = 0.0f;
          flash_attention_row_stats_stub(Q, K, mask, hasMask, b, i, n, d, scale, &rowMax, &denom);
          float p = flash_attention_prob_stub(Q, K, mask, hasMask, b, i, j, n, d, scale,
                                              rowMax, denom);
          acc += p * dOut->data[(b * n + i) * d + dv];
        }
        dV->data[(b * n + j) * d + dv] = acc;
      }
    }
  }
  return torchlean_cuda_buffer_box(dV);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_transpose2d(b_lean_obj_arg BObj, uint32_t rows,
                                                          uint32_t cols) {
  torchlean_cuda_buffer* b = torchlean_cuda_buffer_unbox(BObj);
  const size_t R = (size_t)rows;
  const size_t C = (size_t)cols;
  if (b->size != R * C) {
    lean_internal_panic("torchlean_cuda_buffer_transpose2d_stub: size mismatch");
  }

  const size_t total = R * C;
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(total);
  for (size_t i = 0; i < R; ++i) {
    for (size_t j = 0; j < C; ++j) {
      out->data[j * R + i] = b->data[i * C + j];
    }
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_gather_vec(b_lean_obj_arg VObj, uint32_t n,
                                                         b_lean_obj_arg IdxObj, uint32_t k) {
  torchlean_cuda_buffer* v = torchlean_cuda_buffer_unbox(VObj);
  const size_t N = (size_t)n;
  const size_t K = (size_t)k;
  if (v->size != N) {
    lean_internal_panic("torchlean_cuda_buffer_gather_vec_stub: vec.size mismatch");
  }
  if (!lean_is_array((lean_object*)IdxObj)) {
    lean_internal_panic("torchlean_cuda_buffer_gather_vec_stub: expected Array Nat indices");
  }
  if (lean_array_size(IdxObj) != K) {
    lean_internal_panic("torchlean_cuda_buffer_gather_vec_stub: indices.size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(K);
  for (size_t j = 0; j < K; ++j) {
    b_lean_obj_res idxNat = lean_array_get_core(IdxObj, j);
    uint32_t i =
        nat_to_u32_or_panic(idxNat, "torchlean_cuda_buffer_gather_vec_stub: bad index Nat");
    out->data[j] = (i < n) ? v->data[(size_t)i] : 0.0f;
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_scatter_add(b_lean_obj_arg XObj,
                                                          b_lean_obj_arg ValuesObj, uint32_t n,
                                                          b_lean_obj_arg IdxObj, uint32_t k) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  torchlean_cuda_buffer* values = torchlean_cuda_buffer_unbox(ValuesObj);
  const size_t N = (size_t)n;
  const size_t K = (size_t)k;
  if (x->size != N) {
    lean_internal_panic("torchlean_cuda_buffer_scatter_add_stub: x.size mismatch");
  }
  if (values->size != K) {
    lean_internal_panic("torchlean_cuda_buffer_scatter_add_stub: values.size mismatch");
  }
  if (!lean_is_array((lean_object*)IdxObj)) {
    lean_internal_panic("torchlean_cuda_buffer_scatter_add_stub: expected Array Nat indices");
  }
  if (lean_array_size(IdxObj) != K) {
    lean_internal_panic("torchlean_cuda_buffer_scatter_add_stub: indices.size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(N);
  for (size_t i = 0; i < N; ++i) out->data[i] = x->data[i];

  for (size_t j = 0; j < K; ++j) {
    b_lean_obj_res idxNat = lean_array_get_core(IdxObj, j);
    uint32_t i =
        nat_to_u32_or_panic(idxNat, "torchlean_cuda_buffer_scatter_add_stub: bad index Nat");
    if (i < n) {
      out->data[(size_t)i] += values->data[j];
    }
  }

  return torchlean_cuda_buffer_box(out);
}

// ----------------------------
// Broadcast / reduction helpers
// ----------------------------

static inline size_t nat_to_size_or_panic(b_lean_obj_arg o, const char* msg) {
  if (!lean_is_scalar(o)) {
    lean_internal_panic(msg);
  }
  return (size_t)lean_unbox(o);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_broadcast_to(b_lean_obj_arg XObj,
                                                           b_lean_obj_arg InDimsObj,
                                                           b_lean_obj_arg OutDimsObj,
                                                           b_lean_obj_arg AxisMapObj) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  if (!lean_is_array((lean_object*)InDimsObj) || !lean_is_array((lean_object*)OutDimsObj) ||
      !lean_is_array((lean_object*)AxisMapObj)) {
    lean_internal_panic("torchlean_cuda_buffer_broadcast_to_stub: expected Array Nat dims/map");
  }

  const size_t rankIn = lean_array_size(InDimsObj);
  const size_t rankOut = lean_array_size(OutDimsObj);
  if (lean_array_size(AxisMapObj) != rankOut) {
    lean_internal_panic("torchlean_cuda_buffer_broadcast_to_stub: axisMap.size mismatch");
  }

  size_t* inDims = rankIn == 0 ? NULL : (size_t*)malloc(rankIn * sizeof(size_t));
  size_t* outDims = rankOut == 0 ? NULL : (size_t*)malloc(rankOut * sizeof(size_t));
  uint32_t* axisMap = rankOut == 0 ? NULL : (uint32_t*)malloc(rankOut * sizeof(uint32_t));
  if ((rankIn != 0 && !inDims) || (rankOut != 0 && (!outDims || !axisMap))) {
    lean_internal_panic_out_of_memory();
  }

  size_t inSize = 1;
  for (size_t i = 0; i < rankIn; ++i) {
    inDims[i] = nat_to_size_or_panic(lean_array_get_core(InDimsObj, i),
                                     "torchlean_cuda_buffer_broadcast_to_stub: bad inDims Nat");
    inSize *= inDims[i];
  }
  size_t outSize = 1;
  for (size_t i = 0; i < rankOut; ++i) {
    outDims[i] = nat_to_size_or_panic(lean_array_get_core(OutDimsObj, i),
                                      "torchlean_cuda_buffer_broadcast_to_stub: bad outDims Nat");
    axisMap[i] = nat_to_u32_or_panic(lean_array_get_core(AxisMapObj, i),
                                     "torchlean_cuda_buffer_broadcast_to_stub: bad axisMap Nat");
    outSize *= outDims[i];
  }

  if (x->size != inSize) {
    lean_internal_panic("torchlean_cuda_buffer_broadcast_to_stub: input size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outSize);
  if (outSize == 0) {
    free(inDims);
    free(outDims);
    free(axisMap);
    return torchlean_cuda_buffer_box(out);
  }

  // Pre-allocate coordinate scratch.
  size_t* inCoords = rankIn == 0 ? NULL : (size_t*)malloc(rankIn * sizeof(size_t));
  if (rankIn != 0 && !inCoords) {
    lean_internal_panic_out_of_memory();
  }

  for (size_t outIdx = 0; outIdx < outSize; ++outIdx) {
    for (size_t i = 0; i < rankIn; ++i) inCoords[i] = 0;

    size_t tmp = outIdx;
    // Decode out coords from last axis to first, but store only what we need.
    for (size_t axRev = 0; axRev < rankOut; ++axRev) {
      size_t ax = rankOut - 1 - axRev;
      const size_t od = outDims[ax];
      const size_t coord = (od == 0) ? 0 : (tmp % od);
      tmp = (od == 0) ? 0 : (tmp / od);

      const uint32_t mv = axisMap[ax];
      if (mv == 0) {
        continue;
      }
      const size_t inAx = (size_t)(mv - 1);
      if (inAx >= rankIn) {
        lean_internal_panic("torchlean_cuda_buffer_broadcast_to_stub: axisMap out of range");
      }
      const size_t id = inDims[inAx];
      if (!(id == 1 || id == od)) {
        lean_internal_panic("torchlean_cuda_buffer_broadcast_to_stub: incompatible broadcast dims");
      }
      inCoords[inAx] = (id == 1) ? 0 : coord;
    }

    size_t inIdx = 0;
    for (size_t ax = 0; ax < rankIn; ++ax) {
      inIdx = inIdx * inDims[ax] + inCoords[ax];
    }
    out->data[outIdx] = x->data[inIdx];
  }

  free(inCoords);
  free(inDims);
  free(outDims);
  free(axisMap);
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_from_broadcast(b_lean_obj_arg DOutObj,
                                                                    b_lean_obj_arg InDimsObj,
                                                                    b_lean_obj_arg OutDimsObj,
                                                                    b_lean_obj_arg AxisMapObj) {
  torchlean_cuda_buffer* dOut = torchlean_cuda_buffer_unbox(DOutObj);
  if (!lean_is_array((lean_object*)InDimsObj) || !lean_is_array((lean_object*)OutDimsObj) ||
      !lean_is_array((lean_object*)AxisMapObj)) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_from_broadcast_stub: expected Array Nat dims/map");
  }

  const size_t rankIn = lean_array_size(InDimsObj);
  const size_t rankOut = lean_array_size(OutDimsObj);
  if (lean_array_size(AxisMapObj) != rankOut) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_from_broadcast_stub: axisMap.size mismatch");
  }

  size_t* inDims = rankIn == 0 ? NULL : (size_t*)malloc(rankIn * sizeof(size_t));
  size_t* outDims = rankOut == 0 ? NULL : (size_t*)malloc(rankOut * sizeof(size_t));
  uint32_t* axisMap = rankOut == 0 ? NULL : (uint32_t*)malloc(rankOut * sizeof(uint32_t));
  if ((rankIn != 0 && !inDims) || (rankOut != 0 && (!outDims || !axisMap))) {
    lean_internal_panic_out_of_memory();
  }

  size_t inSize = 1;
  for (size_t i = 0; i < rankIn; ++i) {
    inDims[i] = nat_to_size_or_panic(lean_array_get_core(InDimsObj, i),
                                     "torchlean_cuda_buffer_reduce_from_broadcast_stub: bad inDims Nat");
    inSize *= inDims[i];
  }
  size_t outSize = 1;
  for (size_t i = 0; i < rankOut; ++i) {
    outDims[i] = nat_to_size_or_panic(lean_array_get_core(OutDimsObj, i),
                                      "torchlean_cuda_buffer_reduce_from_broadcast_stub: bad outDims Nat");
    axisMap[i] =
        nat_to_u32_or_panic(lean_array_get_core(AxisMapObj, i),
                            "torchlean_cuda_buffer_reduce_from_broadcast_stub: bad axisMap Nat");
    outSize *= outDims[i];
  }

  if (dOut->size != outSize) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_from_broadcast_stub: dOut size mismatch");
  }

  torchlean_cuda_buffer* dIn = torchlean_cuda_buffer_alloc(inSize);
  for (size_t i = 0; i < inSize; ++i) dIn->data[i] = 0.0f;
  if (outSize == 0) {
    free(inDims);
    free(outDims);
    free(axisMap);
    return torchlean_cuda_buffer_box(dIn);
  }

  size_t* inCoords = rankIn == 0 ? NULL : (size_t*)malloc(rankIn * sizeof(size_t));
  if (rankIn != 0 && !inCoords) {
    lean_internal_panic_out_of_memory();
  }

  for (size_t outIdx = 0; outIdx < outSize; ++outIdx) {
    for (size_t i = 0; i < rankIn; ++i) inCoords[i] = 0;

    size_t tmp = outIdx;
    for (size_t axRev = 0; axRev < rankOut; ++axRev) {
      size_t ax = rankOut - 1 - axRev;
      const size_t od = outDims[ax];
      const size_t coord = (od == 0) ? 0 : (tmp % od);
      tmp = (od == 0) ? 0 : (tmp / od);

      const uint32_t mv = axisMap[ax];
      if (mv == 0) continue;
      const size_t inAx = (size_t)(mv - 1);
      if (inAx >= rankIn) {
        lean_internal_panic("torchlean_cuda_buffer_reduce_from_broadcast_stub: axisMap out of range");
      }
      const size_t id = inDims[inAx];
      if (!(id == 1 || id == od)) {
        lean_internal_panic("torchlean_cuda_buffer_reduce_from_broadcast_stub: incompatible broadcast dims");
      }
      inCoords[inAx] = (id == 1) ? 0 : coord;
    }

    size_t inIdx = 0;
    for (size_t ax = 0; ax < rankIn; ++ax) {
      inIdx = inIdx * inDims[ax] + inCoords[ax];
    }
    dIn->data[inIdx] += dOut->data[outIdx];
  }

  free(inCoords);
  free(inDims);
  free(outDims);
  free(axisMap);
  return torchlean_cuda_buffer_box(dIn);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_swap_adjacent_at_depth(b_lean_obj_arg XObj,
                                                                     b_lean_obj_arg DimsObj,
                                                                     uint32_t depth) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  if (!lean_is_array((lean_object*)DimsObj)) {
    lean_internal_panic("torchlean_cuda_buffer_swap_adjacent_at_depth_stub: expected Array Nat dims");
  }
  const size_t rank = lean_array_size(DimsObj);
  if (rank < 2 || (size_t)depth + 1 >= rank) {
    lean_internal_panic("torchlean_cuda_buffer_swap_adjacent_at_depth_stub: invalid depth for rank");
  }

  size_t* dims = (size_t*)malloc(rank * sizeof(size_t));
  if (!dims) lean_internal_panic_out_of_memory();
  size_t total = 1;
  for (size_t i = 0; i < rank; ++i) {
    dims[i] = nat_to_size_or_panic(lean_array_get_core(DimsObj, i),
                                   "torchlean_cuda_buffer_swap_adjacent_at_depth_stub: bad dims Nat");
    total *= dims[i];
  }
  if (x->size != total) {
    lean_internal_panic("torchlean_cuda_buffer_swap_adjacent_at_depth_stub: input size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(total);
  if (total == 0) {
    free(dims);
    return torchlean_cuda_buffer_box(out);
  }

  size_t* coords = (size_t*)malloc(rank * sizeof(size_t));
  if (!coords) lean_internal_panic_out_of_memory();

  for (size_t outIdx = 0; outIdx < total; ++outIdx) {
    size_t tmp = outIdx;
    for (size_t axRev = 0; axRev < rank; ++axRev) {
      size_t ax = rank - 1 - axRev;
      size_t d = dims[ax];
      if (ax == (size_t)depth) {
        d = dims[(size_t)depth + 1];
      } else if (ax == (size_t)depth + 1) {
        d = dims[(size_t)depth];
      }
      coords[ax] = (d == 0) ? 0 : (tmp % d);
      tmp = (d == 0) ? 0 : (tmp / d);
    }
    size_t t = coords[depth];
    coords[depth] = coords[depth + 1];
    coords[depth + 1] = t;

    size_t inIdx = 0;
    for (size_t ax = 0; ax < rank; ++ax) {
      inIdx = inIdx * dims[ax] + coords[ax];
    }
    out->data[outIdx] = x->data[inIdx];
  }

  free(coords);
  free(dims);
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_reduce_sum_axis(b_lean_obj_arg XObj,
                                                              b_lean_obj_arg DimsObj,
                                                              uint32_t axis) {
  torchlean_cuda_buffer* x = torchlean_cuda_buffer_unbox(XObj);
  if (!lean_is_array((lean_object*)DimsObj)) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_sum_axis_stub: expected Array Nat dims");
  }
  const size_t rank = lean_array_size(DimsObj);
  if (rank == 0) {
    // Scalar: sum over any axis is identity (but axis should be invalid upstream).
    torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(1);
    out->data[0] = (x->size == 0) ? 0.0f : x->data[0];
    return torchlean_cuda_buffer_box(out);
  }
  if ((size_t)axis >= rank) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_sum_axis_stub: invalid axis");
  }

  size_t* dims = (size_t*)malloc(rank * sizeof(size_t));
  if (!dims) lean_internal_panic_out_of_memory();
  size_t inSize = 1;
  for (size_t i = 0; i < rank; ++i) {
    dims[i] = nat_to_size_or_panic(lean_array_get_core(DimsObj, i),
                                   "torchlean_cuda_buffer_reduce_sum_axis_stub: bad dims Nat");
    inSize *= dims[i];
  }
  if (x->size != inSize) {
    lean_internal_panic("torchlean_cuda_buffer_reduce_sum_axis_stub: input size mismatch");
  }

  size_t outSize = 1;
  for (size_t i = 0; i < rank; ++i) {
    if (i != (size_t)axis) {
      outSize *= dims[i];
    }
  }
  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(outSize);
  for (size_t i = 0; i < outSize; ++i) out->data[i] = 0.0f;
  if (inSize == 0) {
    free(dims);
    return torchlean_cuda_buffer_box(out);
  }

  size_t* coords = (size_t*)malloc(rank * sizeof(size_t));
  if (!coords) lean_internal_panic_out_of_memory();

  for (size_t inIdx = 0; inIdx < inSize; ++inIdx) {
    size_t tmp = inIdx;
    for (size_t axRev = 0; axRev < rank; ++axRev) {
      size_t ax = rank - 1 - axRev;
      const size_t d = dims[ax];
      coords[ax] = (d == 0) ? 0 : (tmp % d);
      tmp = (d == 0) ? 0 : (tmp / d);
    }

    size_t outIdx = 0;
    for (size_t ax = 0; ax < rank; ++ax) {
      if (ax == (size_t)axis) continue;
      outIdx = outIdx * dims[ax] + coords[ax];
    }
    if (outSize != 0) {
      out->data[outIdx] += x->data[inIdx];
    }
  }

  free(coords);
  free(dims);
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_gather_rows(b_lean_obj_arg MObj, uint32_t rows,
                                                          uint32_t cols, b_lean_obj_arg IdxObj,
                                                          uint32_t k) {
  torchlean_cuda_buffer* m = torchlean_cuda_buffer_unbox(MObj);
  const size_t R = (size_t)rows;
  const size_t C = (size_t)cols;
  const size_t K = (size_t)k;
  if (m->size != R * C) {
    lean_internal_panic("torchlean_cuda_buffer_gather_rows_stub: mat.size mismatch");
  }
  if (!lean_is_array((lean_object*)IdxObj)) {
    lean_internal_panic("torchlean_cuda_buffer_gather_rows_stub: expected Array Nat indices");
  }
  if (lean_array_size(IdxObj) != K) {
    lean_internal_panic("torchlean_cuda_buffer_gather_rows_stub: indices.size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(K * C);
  for (size_t r = 0; r < K; ++r) {
    uint32_t idx =
        nat_to_u32_or_panic(lean_array_get_core(IdxObj, r),
                            "torchlean_cuda_buffer_gather_rows_stub: bad index Nat");
    for (size_t j = 0; j < C; ++j) {
      out->data[r * C + j] = (idx < rows) ? m->data[(size_t)idx * C + j] : 0.0f;
    }
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_scatter_add_row(b_lean_obj_arg MObj,
                                                              b_lean_obj_arg RowObj, uint32_t rows,
                                                              uint32_t cols, uint32_t i) {
  torchlean_cuda_buffer* m = torchlean_cuda_buffer_unbox(MObj);
  torchlean_cuda_buffer* row = torchlean_cuda_buffer_unbox(RowObj);
  const size_t R = (size_t)rows;
  const size_t C = (size_t)cols;
  if (m->size != R * C) {
    lean_internal_panic("torchlean_cuda_buffer_scatter_add_row_stub: mat.size mismatch");
  }
  if (row->size != C) {
    lean_internal_panic("torchlean_cuda_buffer_scatter_add_row_stub: rowVec.size mismatch");
  }
  if ((size_t)i >= R) {
    lean_internal_panic("torchlean_cuda_buffer_scatter_add_row_stub: row index out of bounds");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(R * C);
  for (size_t t = 0; t < R * C; ++t) out->data[t] = m->data[t];
  for (size_t j = 0; j < C; ++j) {
    out->data[(size_t)i * C + j] += row->data[j];
  }
  return torchlean_cuda_buffer_box(out);
}

LEAN_EXPORT lean_obj_res torchlean_cuda_buffer_scatter_add_rows(b_lean_obj_arg MObj,
                                                               b_lean_obj_arg ValuesObj,
                                                               uint32_t rows, uint32_t cols,
                                                               b_lean_obj_arg IdxObj, uint32_t k) {
  torchlean_cuda_buffer* m = torchlean_cuda_buffer_unbox(MObj);
  torchlean_cuda_buffer* values = torchlean_cuda_buffer_unbox(ValuesObj);
  const size_t R = (size_t)rows;
  const size_t C = (size_t)cols;
  const size_t K = (size_t)k;
  if (m->size != R * C) {
    lean_internal_panic("torchlean_cuda_buffer_scatter_add_rows_stub: mat.size mismatch");
  }
  if (values->size != K * C) {
    lean_internal_panic("torchlean_cuda_buffer_scatter_add_rows_stub: values.size mismatch");
  }
  if (!lean_is_array((lean_object*)IdxObj)) {
    lean_internal_panic("torchlean_cuda_buffer_scatter_add_rows_stub: expected Array Nat indices");
  }
  if (lean_array_size(IdxObj) != K) {
    lean_internal_panic("torchlean_cuda_buffer_scatter_add_rows_stub: indices.size mismatch");
  }

  torchlean_cuda_buffer* out = torchlean_cuda_buffer_alloc(R * C);
  for (size_t t = 0; t < R * C; ++t) out->data[t] = m->data[t];

  for (size_t r = 0; r < K; ++r) {
    uint32_t idx =
        nat_to_u32_or_panic(lean_array_get_core(IdxObj, r),
                            "torchlean_cuda_buffer_scatter_add_rows_stub: bad index Nat");
    if (idx >= rows) continue;
    for (size_t j = 0; j < C; ++j) {
      out->data[(size_t)idx * C + j] += values->data[r * C + j];
    }
  }
  return torchlean_cuda_buffer_box(out);
}
