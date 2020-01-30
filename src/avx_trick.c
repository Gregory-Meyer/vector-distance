#include <lib.h>

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>

#include <immintrin.h>
#include <pmmintrin.h>
#include <xmmintrin.h>

#define AVX_LENGTH ((size_t)8)

size_t workspace_size(size_t n, size_t k) {
  (void)k;

  return n;
}

static void squared_2_norm_for_each(size_t n, size_t k, const float X[n][k],
                                    float squared_2_norms[restrict n]);
static float squared_2_norm(size_t k, const float v[k]);
static float euclidean_distance(size_t k, const float v[k], const float u[k],
                                float u_squared_2_norm, float v_squared_2_norm);

void pairwise_euclidean_distance(size_t n, size_t k, size_t l,
                                 const float X[n][k], float Z[restrict n][n],
                                 float workspace[restrict l]) {
  assert(l >= n);

  float *const squared_2_norms = workspace;
  squared_2_norm_for_each(n, k, X, squared_2_norms);

  for (size_t i = 0; i < n; ++i) {
    Z[i][i] = 0.0f;

    for (size_t j = i + 1; j < n; ++j) {
      const float distance = euclidean_distance(
          k, X[i], X[j], squared_2_norms[i], squared_2_norms[j]);

      Z[i][j] = distance;
      Z[j][i] = distance;
    }
  }
}

static __m256i maskn(size_t n);

static void squared_2_norm_for_each(size_t n, size_t k, const float X[n][k],
                                    float squared_2_norms[restrict n]) {
  for (; n >= AVX_LENGTH;
       n -= AVX_LENGTH, X += AVX_LENGTH, squared_2_norms += AVX_LENGTH) {
    __m256 squared_norms;

    for (size_t i = 0; i < AVX_LENGTH; ++i) {
      squared_norms[i] = squared_2_norm(k, X[i]);
    }

    _mm256_storeu_ps(squared_2_norms, squared_norms);
  }

  if (n > 0) {
    __m256 squared_norms = _mm256_setzero_ps();

    for (size_t i = 0; i < n; ++i) {
      squared_norms[i] = squared_2_norm(k, X[i]);
    }

    _mm256_maskstore_ps(squared_2_norms, maskn(n), squared_norms);
  }
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

static float horizontal_sum(__m256 v);

static float squared_2_norm(size_t k, const float v[k]) {
  __m256 squared_norm_accumulators = _mm256_setzero_ps();

  for (; k >= AVX_LENGTH; k -= AVX_LENGTH, v += AVX_LENGTH) {
    const __m256 elems = _mm256_loadu_ps(v);

    squared_norm_accumulators =
        _mm256_fmadd_ps(elems, elems, squared_norm_accumulators);
  }

  if (k > 0) {
    const __m256 elems = _mm256_maskload_ps(v, maskn(k));

    squared_norm_accumulators =
        _mm256_fmadd_ps(elems, elems, squared_norm_accumulators);
  }

  const float squared_norm = horizontal_sum(squared_norm_accumulators);

  return squared_norm;
}

static float euclidean_distance(size_t k, const float v[k], const float u[k],
                                float v_squared_2_norm,
                                float u_squared_2_norm) {
  __m256 inner_product_accumulators = _mm256_setzero_ps();

  for (; k >= AVX_LENGTH; k -= AVX_LENGTH, v += AVX_LENGTH, u += AVX_LENGTH) {
    const __m256 v_elems = _mm256_loadu_ps(v);
    const __m256 u_elems = _mm256_loadu_ps(u);

    inner_product_accumulators =
        _mm256_fmadd_ps(v_elems, u_elems, inner_product_accumulators);
  }

  if (k > 0) {
    const __m256i load_mask = maskn(k);

    const __m256 v_elems = _mm256_maskload_ps(v, load_mask);
    const __m256 u_elems = _mm256_maskload_ps(u, load_mask);

    inner_product_accumulators =
        _mm256_fmadd_ps(v_elems, u_elems, inner_product_accumulators);
  }

  const float inner_product = horizontal_sum(inner_product_accumulators);
  const float squared_distance =
      u_squared_2_norm + v_squared_2_norm - 2.0f * inner_product;

  if (squared_distance <= 0.0f) {
    return 0.0f;
  }

  const float distance = sqrtf(squared_distance);

  return distance;
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
