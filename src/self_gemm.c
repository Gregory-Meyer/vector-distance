#include <lib.h>

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>

#include <immintrin.h>
#include <pmmintrin.h>
#include <xmmintrin.h>

#define AVX_LENGTH ((size_t)8)

size_t workspace_size(size_t n, size_t k) {
  (void)n;
  (void)k;

  return n + k * n;
}

static __m256i maskn(size_t n);
static float self_inner_product(size_t n, const float v[n], __m256i load_mask);
static void do_pairwise_euclidean_distance(size_t n, size_t m,
                                           const float X[n * m],
                                           float Z[restrict n * n],
                                           float squared_norms[restrict n],
                                           float XT[restrict m * n]);

void pairwise_euclidean_distance(size_t n, size_t m, size_t p,
                                 const float X[n][m], float Z[restrict n][n],
                                 float workspace[restrict p]) {
  assert(p >= n);

  float *const squared_norms = workspace; // n
  float *const XT = workspace + n;        // k x n

  do_pairwise_euclidean_distance(n, m, &X[0][0], &Z[0][0], squared_norms, XT);
}

static void do_pairwise_euclidean_distance(size_t n, size_t m,
                                           const float X[n * m],
                                           float Z[restrict n * n],
                                           float squared_norms[restrict n],
                                           float XT[restrict m * n]) {
  const __m256i load_mask = maskn(m % AVX_LENGTH);

  for (size_t i = 0; i < n; ++i) {
    const float squared_norm = self_inner_product(m, &X[i * m], load_mask);

    squared_norms[i] = squared_norm;
  }

  for (size_t i = 0; i < n; ++i) {
    for (size_t k = 0; k < m; ++k) {
      XT[k * n + i] = X[i * m + k];
    }
  }

  for (size_t i = 0; i < n; ++i) {
    Z[i * n + i] = 0.0f;

    const float squared_norm_i = squared_norms[i];
    for (size_t j = i + 1; j < n; ++j) {
      Z[i * n + j] = -0.5f * (squared_norm_i + squared_norms[j]);
    }

    for (size_t k = 0; k < m; k += 4) {
      const __m256 X_ik0 = _mm256_set1_ps(X[i * m + k]);

      const bool load1 = k + 1 < m;
      const bool load2 = k + 2 < m;
      const bool load3 = k + 3 < m;

      __m256 X_ik1;
      __m256 X_ik2;
      __m256 X_ik3;

      if (load3) {
        X_ik1 = _mm256_set1_ps(X[i * m + k + 1]);
        X_ik2 = _mm256_set1_ps(X[i * m + k + 2]);
        X_ik3 = _mm256_set1_ps(X[i * m + k + 3]);
      } else if (load2) {
        X_ik1 = _mm256_set1_ps(X[i * m + k + 1]);
        X_ik2 = _mm256_set1_ps(X[i * m + k + 2]);
        X_ik3 = _mm256_setzero_ps();
      } else if (load1) {
        X_ik1 = _mm256_set1_ps(X[i * m + k + 1]);
        X_ik2 = X_ik3 = _mm256_setzero_ps();
      } else {
        X_ik1 = X_ik2 = X_ik3 = _mm256_setzero_ps();
      }

      size_t num_loads = n - (i + 1);
      const size_t Z_base_index = i * n + i + 1;
      const size_t XT_base_index = k * n + i + 1;
      size_t offset = 0;

      if (load3) {
        for (; num_loads >= AVX_LENGTH;
             offset += AVX_LENGTH, num_loads -= AVX_LENGTH) {
          const __m256 Z_elems = _mm256_loadu_ps(Z + Z_base_index + offset);

          const __m256 XT_elems0 = _mm256_loadu_ps(XT + XT_base_index + offset);
          const __m256 XT_elems1 =
              _mm256_loadu_ps(XT + XT_base_index + offset + n);
          const __m256 XT_elems2 =
              _mm256_loadu_ps(XT + XT_base_index + offset + 2 * n);
          const __m256 XT_elems3 =
              _mm256_loadu_ps(XT + XT_base_index + offset + 3 * n);

          const __m256 prod0 = _mm256_mul_ps(X_ik0, XT_elems0);
          const __m256 prod1 = _mm256_mul_ps(X_ik1, XT_elems1);
          const __m256 prod2 = _mm256_mul_ps(X_ik2, XT_elems2);
          const __m256 prod3 = _mm256_mul_ps(X_ik3, XT_elems3);

          const __m256 prod01 = _mm256_add_ps(prod0, prod1);
          const __m256 prod23 = _mm256_add_ps(prod2, prod3);

          const __m256 prod0123 = _mm256_add_ps(prod01, prod23);
          const __m256 accumulated = _mm256_add_ps(prod0123, Z_elems);

          _mm256_storeu_ps(Z + Z_base_index + offset, accumulated);
        }

        if (num_loads > 0) {
          const __m256i iomask = maskn(num_loads);

          const __m256 Z_elems =
              _mm256_maskload_ps(Z + Z_base_index + offset, iomask);

          const __m256 XT_elems0 =
              _mm256_maskload_ps(XT + XT_base_index + offset, iomask);
          const __m256 XT_elems1 =
              _mm256_maskload_ps(XT + XT_base_index + offset + n, iomask);
          const __m256 XT_elems2 =
              _mm256_maskload_ps(XT + XT_base_index + offset + 2 * n, iomask);
          const __m256 XT_elems3 =
              _mm256_maskload_ps(XT + XT_base_index + offset + 3 * n, iomask);

          const __m256 prod0 = _mm256_mul_ps(X_ik0, XT_elems0);
          const __m256 prod1 = _mm256_mul_ps(X_ik1, XT_elems1);
          const __m256 prod2 = _mm256_mul_ps(X_ik2, XT_elems2);
          const __m256 prod3 = _mm256_mul_ps(X_ik3, XT_elems3);

          const __m256 prod01 = _mm256_add_ps(prod0, prod1);
          const __m256 prod23 = _mm256_add_ps(prod2, prod3);

          const __m256 prod0123 = _mm256_add_ps(prod01, prod23);
          const __m256 accumulated = _mm256_add_ps(prod0123, Z_elems);

          _mm256_maskstore_ps(Z + Z_base_index + offset, iomask, accumulated);
        }
      } else if (load2) {
        for (; num_loads >= AVX_LENGTH;
             offset += AVX_LENGTH, num_loads -= AVX_LENGTH) {
          const __m256 Z_elems = _mm256_loadu_ps(Z + Z_base_index + offset);

          const __m256 XT_elems0 = _mm256_loadu_ps(XT + XT_base_index + offset);
          const __m256 XT_elems1 =
              _mm256_loadu_ps(XT + XT_base_index + offset + n);
          const __m256 XT_elems2 =
              _mm256_loadu_ps(XT + XT_base_index + offset + 2 * n);

          const __m256 prod0 = _mm256_mul_ps(X_ik0, XT_elems0);
          const __m256 prod1 = _mm256_mul_ps(X_ik1, XT_elems1);
          const __m256 prod2 = _mm256_mul_ps(X_ik2, XT_elems2);

          const __m256 prod01 = _mm256_add_ps(prod0, prod1);

          const __m256 prod012 = _mm256_add_ps(prod01, prod2);
          const __m256 accumulated = _mm256_add_ps(prod012, Z_elems);

          _mm256_storeu_ps(Z + Z_base_index + offset, accumulated);
        }

        if (num_loads > 0) {
          const __m256i iomask = maskn(num_loads);

          const __m256 Z_elems =
              _mm256_maskload_ps(Z + Z_base_index + offset, iomask);

          const __m256 XT_elems0 =
              _mm256_maskload_ps(XT + XT_base_index + offset, iomask);
          const __m256 XT_elems1 =
              _mm256_maskload_ps(XT + XT_base_index + offset + n, iomask);
          const __m256 XT_elems2 =
              _mm256_maskload_ps(XT + XT_base_index + offset + 2 * n, iomask);

          const __m256 prod0 = _mm256_mul_ps(X_ik0, XT_elems0);
          const __m256 prod1 = _mm256_mul_ps(X_ik1, XT_elems1);
          const __m256 prod2 = _mm256_mul_ps(X_ik2, XT_elems2);

          const __m256 prod01 = _mm256_add_ps(prod0, prod1);

          const __m256 prod012 = _mm256_add_ps(prod01, prod2);
          const __m256 accumulated = _mm256_add_ps(prod012, Z_elems);

          _mm256_maskstore_ps(Z + Z_base_index + offset, iomask, accumulated);
        }
      } else if (load1) {
        for (; num_loads >= AVX_LENGTH;
             offset += AVX_LENGTH, num_loads -= AVX_LENGTH) {
          const __m256 Z_elems = _mm256_loadu_ps(Z + Z_base_index + offset);

          const __m256 XT_elems0 = _mm256_loadu_ps(XT + XT_base_index + offset);
          const __m256 XT_elems1 =
              _mm256_loadu_ps(XT + XT_base_index + offset + n);

          const __m256 prod0 = _mm256_mul_ps(X_ik0, XT_elems0);
          const __m256 prod1 = _mm256_mul_ps(X_ik1, XT_elems1);

          const __m256 prod01 = _mm256_add_ps(prod0, prod1);

          const __m256 accumulated = _mm256_add_ps(prod01, Z_elems);

          _mm256_storeu_ps(Z + Z_base_index + offset, accumulated);
        }

        if (num_loads > 0) {
          const __m256i iomask = maskn(num_loads);

          const __m256 Z_elems =
              _mm256_maskload_ps(Z + Z_base_index + offset, iomask);

          const __m256 XT_elems0 =
              _mm256_maskload_ps(XT + XT_base_index + offset, iomask);
          const __m256 XT_elems1 =
              _mm256_maskload_ps(XT + XT_base_index + offset + n, iomask);

          const __m256 prod0 = _mm256_mul_ps(X_ik0, XT_elems0);
          const __m256 prod1 = _mm256_mul_ps(X_ik1, XT_elems1);

          const __m256 prod01 = _mm256_add_ps(prod0, prod1);

          const __m256 accumulated = _mm256_add_ps(prod01, Z_elems);

          _mm256_maskstore_ps(Z + Z_base_index + offset, iomask, accumulated);
        }
      } else {
        for (; num_loads >= AVX_LENGTH;
             offset += AVX_LENGTH, num_loads -= AVX_LENGTH) {
          const __m256 Z_elems = _mm256_loadu_ps(Z + Z_base_index + offset);

          const __m256 XT_elems0 = _mm256_loadu_ps(XT + XT_base_index + offset);

          const __m256 prod0 = _mm256_mul_ps(X_ik0, XT_elems0);

          const __m256 accumulated = _mm256_add_ps(prod0, Z_elems);

          _mm256_storeu_ps(Z + Z_base_index + offset, accumulated);
        }

        if (num_loads > 0) {
          const __m256i iomask = maskn(num_loads);

          const __m256 Z_elems =
              _mm256_maskload_ps(Z + Z_base_index + offset, iomask);

          const __m256 XT_elems0 =
              _mm256_maskload_ps(XT + XT_base_index + offset, iomask);

          const __m256 prod0 = _mm256_mul_ps(X_ik0, XT_elems0);

          const __m256 accumulated = _mm256_add_ps(prod0, Z_elems);

          _mm256_maskstore_ps(Z + Z_base_index + offset, iomask, accumulated);
        }
      }
    }

    for (size_t j = i + 1; j < n; ++j) {
      const float squared_distance = -2.0f * Z[i * n + j];

      float distance;

      if (squared_distance <= 0.0f) {
        distance = 0.0f;
      } else {
        distance = sqrtf(squared_distance);
      }

      Z[i * n + j] = distance;
      Z[j * n + i] = distance;
    }
  }
}

static float horizontal_sum(__m256 v);

static float self_inner_product(size_t n, const float v[n], __m256i load_mask) {
  __m256 inner_product_accumulators = _mm256_setzero_ps();

  for (; n >= AVX_LENGTH; n -= AVX_LENGTH, v += AVX_LENGTH) {
    const __m256 elems = _mm256_loadu_ps(v);

    inner_product_accumulators =
        _mm256_fmadd_ps(elems, elems, inner_product_accumulators);
  }

  if (n > 0) {
    const __m256 elems = _mm256_maskload_ps(v, load_mask);

    inner_product_accumulators =
        _mm256_fmadd_ps(elems, elems, inner_product_accumulators);
  }

  const float inner_product = horizontal_sum(inner_product_accumulators);

  return inner_product;
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
