#include <lib.h>

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include <immintrin.h>
#include <pmmintrin.h>
#include <xmmintrin.h>

#define AVX_LENGTH ((size_t)8)

size_t workspace_size(size_t n, size_t k) {
  (void)k;

  return n * (n - 1) / 2 + n;
}

static void upper_triangular_all_pairs_inner_product(
    size_t n, size_t k, const float X[n][k], float Y[restrict n * (n - 1) / 2],
    float D[restrict n]);

void pairwise_euclidean_distance(size_t n, size_t k, size_t p,
                                 const float X[n][k], float Z[restrict n][n],
                                 float workspace[restrict p]) {
  assert(p >= workspace_size(n, k));

  const size_t num_triangular_elements = n * (n - 1) / 2;

  float *const inner_products = workspace;
  float *const squared_distances = workspace + num_triangular_elements;
  upper_triangular_all_pairs_inner_product(n, k, X, inner_products,
                                           squared_distances);

  Z[0][0] = 0.0f;

  size_t i = 0;
  size_t j = 1;
  for (size_t triangular_index = 0; triangular_index < num_triangular_elements;
       ++triangular_index) {
    const float squared_distance = squared_distances[i] + squared_distances[j] -
                                   2.0f * inner_products[triangular_index];

    float distance;

    if (squared_distance <= 0.0f) {
      distance = 0.0f;
    } else {
      distance = sqrtf(squared_distance);
    }

    Z[i][j] = distance;
    Z[j][i] = distance;

    if (j == n - 1) {
      i += 1;
      j = i + 1;

      Z[i][i] = 0.0f;
    } else {
      ++j;
    }
  }
}

static __m256i maskn(size_t n);
static float self_inner_product(size_t k, const float v[k], __m256i load_mask);
static float inner_product(size_t k, const float v[k], const float u[k],
                           __m256i load_mask);

static void upper_triangular_all_pairs_inner_product(
    size_t n, size_t k, const float X[n][k], float Y[restrict n * (n - 1) / 2],
    float D[restrict n]) {
  const size_t num_triangular_elements = n * (n - 1) / 2;

  const __m256i load_mask = maskn(k % AVX_LENGTH);
  D[0] = self_inner_product(k, X[0], load_mask);

  size_t i = 0;
  size_t j = 1;
  for (size_t triangular_index = 0; triangular_index < num_triangular_elements;
       ++triangular_index) {
    Y[triangular_index] = inner_product(k, X[i], X[j], load_mask);

    if (j == n - 1) {
      i += 1;
      j = i + 1;

      D[i] = self_inner_product(k, X[i], load_mask);
    } else {
      ++j;
    }
  }
}

static float horizontal_sum(__m256 v);

static float self_inner_product(size_t k, const float v[k], __m256i load_mask) {
  __m256 inner_product_accumulators = _mm256_setzero_ps();

  for (; k >= AVX_LENGTH; k -= AVX_LENGTH, v += AVX_LENGTH) {
    const __m256 elems = _mm256_loadu_ps(v);

    inner_product_accumulators =
        _mm256_fmadd_ps(elems, elems, inner_product_accumulators);
  }

  if (k > 0) {
    const __m256 elems = _mm256_maskload_ps(v, load_mask);

    inner_product_accumulators =
        _mm256_fmadd_ps(elems, elems, inner_product_accumulators);
  }

  const float inner_product = horizontal_sum(inner_product_accumulators);

  return inner_product;
}

static float inner_product(size_t k, const float v[k], const float u[k],
                           __m256i load_mask) {
  __m256 inner_product_accumulators = _mm256_setzero_ps();

  for (; k >= AVX_LENGTH; k -= AVX_LENGTH, v += AVX_LENGTH, u += AVX_LENGTH) {
    const __m256 v_elems = _mm256_loadu_ps(v);
    const __m256 u_elems = _mm256_loadu_ps(u);

    inner_product_accumulators =
        _mm256_fmadd_ps(v_elems, u_elems, inner_product_accumulators);
  }

  if (k > 0) {
    const __m256 v_elems = _mm256_maskload_ps(v, load_mask);
    const __m256 u_elems = _mm256_maskload_ps(u, load_mask);

    inner_product_accumulators =
        _mm256_fmadd_ps(v_elems, u_elems, inner_product_accumulators);
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
