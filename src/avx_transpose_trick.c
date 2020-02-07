#include <lib.h>

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <immintrin.h>
#include <pmmintrin.h>
#include <xmmintrin.h>

#define AVX_LENGTH ((size_t)8)

size_t workspace_size(size_t n, size_t k) { return (k * n) + (n * n); }

static void transpose(size_t n, size_t m, const float X[n][m], float XT[m][n]);
static void symmetric_rank_k_update(size_t n, size_t m, const float X[n][m],
                                    const float XT[m][n],
                                    float XXT[restrict n][n]);

void pairwise_euclidean_distance(size_t n, size_t k, size_t l,
                                 const float X[n][k], float Z[restrict n][n],
                                 float workspace[restrict l]) {
  assert(l >= n);

  float *const XT = workspace;
  transpose(n, k, X, (float(*)[])XT);

  float *const XXT = workspace + (k * n);
  symmetric_rank_k_update(n, k, X, (const float(*)[])XT, (float(*)[])XXT);

  for (size_t i = 0; i < n; ++i) {
    Z[i][i] = 0.0f;

    for (size_t j = i + 1; j < n; ++j) {
      const float squared_distance =
          XXT[i * n + i] - 2.0f * XXT[i * n + j] + XXT[j * n + j];

      float distance;

      if (squared_distance <= 0.0f) {
        distance = 0.0f;
      } else {
        distance = sqrtf(squared_distance);
      }

      Z[i][j] = distance;
      Z[j][i] = distance;
    }
  }
}

static void transpose(size_t n, size_t m, const float X[n][m], float XT[m][n]) {
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      XT[j][i] = X[i][j];
    }
  }
}

static __m256i maskn(size_t n);
static float squared_2_norm(size_t k, const float v[k], __m256i mask);

static void symmetric_rank_k_update(size_t n, size_t m, const float X[n][m],
                                    const float XT[m][n],
                                    float XXT[restrict n][n]) {
  memset(XXT, 0, (n * n) * sizeof(float));

  const __m256i m_mask = maskn(m % AVX_LENGTH);

  for (size_t i = 0; i < n; ++i) {
    XXT[i][i] = squared_2_norm(m, X[i], m_mask);

    size_t k = 0;
    for (; k + 3 < m; k += 4) {
      const __m256 Xik0 = _mm256_set1_ps(X[i][k]);
      const __m256 Xik1 = _mm256_set1_ps(X[i][k + 1]);
      const __m256 Xik2 = _mm256_set1_ps(X[i][k + 2]);
      const __m256 Xik3 = _mm256_set1_ps(X[i][k + 3]);

      const float *XT_row_ptr_0 = &XT[k][i + 1];
      const float *XT_row_ptr_1 = &XT[k + 1][i + 1];
      const float *XT_row_ptr_2 = &XT[k + 2][i + 1];
      const float *XT_row_ptr_3 = &XT[k + 3][i + 1];

      float *XXT_row_ptr = &XXT[i][i + 1];
      size_t j_remaining = (n - (i + 1));

      for (; j_remaining >= AVX_LENGTH;
           j_remaining -= AVX_LENGTH, XT_row_ptr_0 += AVX_LENGTH,
           XT_row_ptr_1 += AVX_LENGTH, XT_row_ptr_2 += AVX_LENGTH,
           XT_row_ptr_3 += AVX_LENGTH, XXT_row_ptr += AVX_LENGTH) {
        const __m256 XXTij = _mm256_loadu_ps(XXT_row_ptr);

        const __m256 XTkj0 = _mm256_loadu_ps(XT_row_ptr_0);
        const __m256 XTkj1 = _mm256_loadu_ps(XT_row_ptr_1);
        const __m256 XTkj2 = _mm256_loadu_ps(XT_row_ptr_2);
        const __m256 XTkj3 = _mm256_loadu_ps(XT_row_ptr_3);

        const __m256 tmp0 = _mm256_fmadd_ps(Xik0, XTkj0, XXTij);
        const __m256 tmp1 = _mm256_fmadd_ps(Xik1, XTkj1, tmp0);
        const __m256 tmp2 = _mm256_fmadd_ps(Xik2, XTkj2, tmp1);
        const __m256 XXTij_new = _mm256_fmadd_ps(Xik3, XTkj3, tmp2);

        _mm256_storeu_ps(XXT_row_ptr, XXTij_new);
      }

      if (j_remaining > 0) {
        const __m256i mask = maskn(j_remaining);

        const __m256 XXTij = _mm256_maskload_ps(XXT_row_ptr, mask);

        const __m256 XTkj0 = _mm256_maskload_ps(XT_row_ptr_0, mask);
        const __m256 XTkj1 = _mm256_maskload_ps(XT_row_ptr_1, mask);
        const __m256 XTkj2 = _mm256_maskload_ps(XT_row_ptr_2, mask);
        const __m256 XTkj3 = _mm256_maskload_ps(XT_row_ptr_3, mask);

        const __m256 tmp0 = _mm256_fmadd_ps(Xik0, XTkj0, XXTij);
        const __m256 tmp1 = _mm256_fmadd_ps(Xik1, XTkj1, tmp0);
        const __m256 tmp2 = _mm256_fmadd_ps(Xik2, XTkj2, tmp1);
        const __m256 XXTij_new = _mm256_fmadd_ps(Xik3, XTkj3, tmp2);

        _mm256_maskstore_ps(XXT_row_ptr, mask, XXTij_new);
      }
    }

    if (k + 2 < m) {
      const __m256 Xik0 = _mm256_set1_ps(X[i][k]);
      const __m256 Xik1 = _mm256_set1_ps(X[i][k + 1]);
      const __m256 Xik2 = _mm256_set1_ps(X[i][k + 2]);

      const float *XT_row_ptr_0 = &XT[k][i + 1];
      const float *XT_row_ptr_1 = &XT[k + 1][i + 1];
      const float *XT_row_ptr_2 = &XT[k + 2][i + 1];

      float *XXT_row_ptr = &XXT[i][i + 1];
      size_t j_remaining = (n - (i + 1));

      for (; j_remaining >= AVX_LENGTH;
           j_remaining -= AVX_LENGTH, XT_row_ptr_0 += AVX_LENGTH,
           XT_row_ptr_1 += AVX_LENGTH, XT_row_ptr_2 += AVX_LENGTH,
           XXT_row_ptr += AVX_LENGTH) {
        const __m256 XXTij = _mm256_loadu_ps(XXT_row_ptr);

        const __m256 XTkj0 = _mm256_loadu_ps(XT_row_ptr_0);
        const __m256 XTkj1 = _mm256_loadu_ps(XT_row_ptr_1);
        const __m256 XTkj2 = _mm256_loadu_ps(XT_row_ptr_2);

        const __m256 tmp0 = _mm256_fmadd_ps(Xik0, XTkj0, XXTij);
        const __m256 tmp1 = _mm256_fmadd_ps(Xik1, XTkj1, tmp0);
        const __m256 XXTij_new = _mm256_fmadd_ps(Xik2, XTkj2, tmp1);

        _mm256_storeu_ps(XXT_row_ptr, XXTij_new);
      }

      if (j_remaining > 0) {
        const __m256i mask = maskn(j_remaining);

        const __m256 XXTij = _mm256_maskload_ps(XXT_row_ptr, mask);

        const __m256 XTkj0 = _mm256_maskload_ps(XT_row_ptr_0, mask);
        const __m256 XTkj1 = _mm256_maskload_ps(XT_row_ptr_1, mask);
        const __m256 XTkj2 = _mm256_maskload_ps(XT_row_ptr_2, mask);

        const __m256 tmp0 = _mm256_fmadd_ps(Xik0, XTkj0, XXTij);
        const __m256 tmp1 = _mm256_fmadd_ps(Xik1, XTkj1, tmp0);
        const __m256 XXTij_new = _mm256_fmadd_ps(Xik2, XTkj2, tmp1);

        _mm256_maskstore_ps(XXT_row_ptr, mask, XXTij_new);
      }
    } else if (k + 1 < m) {
      const __m256 Xik0 = _mm256_set1_ps(X[i][k]);
      const __m256 Xik1 = _mm256_set1_ps(X[i][k + 1]);

      const float *XT_row_ptr_0 = &XT[k][i + 1];
      const float *XT_row_ptr_1 = &XT[k + 1][i + 1];

      float *XXT_row_ptr = &XXT[i][i + 1];
      size_t j_remaining = (n - (i + 1));

      for (; j_remaining >= AVX_LENGTH;
           j_remaining -= AVX_LENGTH, XT_row_ptr_0 += AVX_LENGTH,
           XT_row_ptr_1 += AVX_LENGTH, XXT_row_ptr += AVX_LENGTH) {
        const __m256 XXTij = _mm256_loadu_ps(XXT_row_ptr);

        const __m256 XTkj0 = _mm256_loadu_ps(XT_row_ptr_0);
        const __m256 XTkj1 = _mm256_loadu_ps(XT_row_ptr_1);

        const __m256 tmp0 = _mm256_fmadd_ps(Xik0, XTkj0, XXTij);
        const __m256 XXTij_new = _mm256_fmadd_ps(Xik1, XTkj1, tmp0);

        _mm256_storeu_ps(XXT_row_ptr, XXTij_new);
      }

      if (j_remaining > 0) {
        const __m256i mask = maskn(j_remaining);

        const __m256 XXTij = _mm256_maskload_ps(XXT_row_ptr, mask);

        const __m256 XTkj0 = _mm256_maskload_ps(XT_row_ptr_0, mask);
        const __m256 XTkj1 = _mm256_maskload_ps(XT_row_ptr_1, mask);

        const __m256 tmp0 = _mm256_fmadd_ps(Xik0, XTkj0, XXTij);
        const __m256 XXTij_new = _mm256_fmadd_ps(Xik1, XTkj1, tmp0);

        _mm256_maskstore_ps(XXT_row_ptr, mask, XXTij_new);
      }
    } else if (k < m) {
      const __m256 Xik = _mm256_set1_ps(X[i][k]);

      const float *XT_row_ptr = &XT[k][i + 1];

      float *XXT_row_ptr = &XXT[i][i + 1];
      size_t j_remaining = (n - (i + 1));

      for (; j_remaining >= AVX_LENGTH; j_remaining -= AVX_LENGTH,
                                        XT_row_ptr += AVX_LENGTH,
                                        XXT_row_ptr += AVX_LENGTH) {
        const __m256 XXTij = _mm256_loadu_ps(XXT_row_ptr);
        const __m256 XTkj = _mm256_loadu_ps(XT_row_ptr);

        const __m256 XXTij_new = _mm256_fmadd_ps(Xik, XTkj, XXTij);

        _mm256_storeu_ps(XXT_row_ptr, XXTij_new);
      }

      if (j_remaining > 0) {
        const __m256i mask = maskn(j_remaining);

        const __m256 XXTij = _mm256_maskload_ps(XXT_row_ptr, mask);
        const __m256 XTkj = _mm256_maskload_ps(XT_row_ptr, mask);

        const __m256 XXTij_new = _mm256_fmadd_ps(Xik, XTkj, XXTij);

        _mm256_maskstore_ps(XXT_row_ptr, mask, XXTij_new);
      }
    }
  }
}

static float horizontal_sum(__m256 v);

static float squared_2_norm(size_t k, const float v[k], __m256i mask) {
  __m256 squared_norm_accumulators = _mm256_setzero_ps();

  for (; k >= AVX_LENGTH; k -= AVX_LENGTH, v += AVX_LENGTH) {
    const __m256 elems = _mm256_loadu_ps(v);

    squared_norm_accumulators =
        _mm256_fmadd_ps(elems, elems, squared_norm_accumulators);
  }

  if (k > 0) {
    const __m256 elems = _mm256_maskload_ps(v, mask);

    squared_norm_accumulators =
        _mm256_fmadd_ps(elems, elems, squared_norm_accumulators);
  }

  const float squared_norm = horizontal_sum(squared_norm_accumulators);

  return squared_norm;
}

static __m256i maskn(size_t n) {
  assert(n <= 8);

  __m256i mask;

  switch (n) {
  default: // saturate n at 8
    mask = _mm256_set1_epi32(INT_MIN);
    break;
  case 7:
    mask = _mm256_set_epi32(0, INT_MIN, INT_MIN, INT_MIN, INT_MIN, INT_MIN,
                            INT_MIN, INT_MIN);
    break;
  case 6:
    mask = _mm256_set_epi32(0, 0, INT_MIN, INT_MIN, INT_MIN, INT_MIN, INT_MIN,
                            INT_MIN);
    break;
  case 5:
    mask =
        _mm256_set_epi32(0, 0, 0, INT_MIN, INT_MIN, INT_MIN, INT_MIN, INT_MIN);
    break;
  case 4:
    mask = _mm256_set_epi32(0, 0, 0, 0, INT_MIN, INT_MIN, INT_MIN, INT_MIN);
    break;
  case 3:
    mask = _mm256_set_epi32(0, 0, 0, 0, 0, INT_MIN, INT_MIN, INT_MIN);
    break;
  case 2:
    mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, INT_MIN, INT_MIN);
    break;
  case 1:
    mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, INT_MIN);
    break;
  case 0:
    mask = _mm256_setzero_si256();
    break;
  }

  return mask;
}

static float horizontal_sum(__m256 v) {
  __m128 vlow = _mm256_castps256_ps128(v);
  __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
  vlow = _mm_add_ps(vlow, vhigh);             // add the low 128

  __m128 shuf = _mm_movehdup_ps(vlow); // broadcast elements 3,1 to 2,0
  __m128 sums = _mm_add_ps(vlow, shuf);
  shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
  sums = _mm_add_ss(sums, shuf);

  return _mm_cvtss_f32(sums);
}
