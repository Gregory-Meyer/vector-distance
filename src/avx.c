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
  (void)n;
  (void)k;

  return 0;
}

static float euclidean_distance(size_t k, const float v[k], const float u[k]);

void pairwise_euclidean_distance(size_t n, size_t k, size_t l,
                                 const float X[n][k], float Z[restrict n][n],
                                 float workspace[restrict l]) {
  (void)l;
  (void)workspace;

  for (size_t i = 0; i < n; ++i) {
    Z[i][i] = 0.0f;

    for (size_t j = i + 1; j < n; ++j) {
      const float distance = euclidean_distance(k, X[i], X[j]);

      Z[i][j] = distance;
      Z[j][i] = distance;
    }
  }
}

static __m256i maskn(size_t n);
static float horizontal_sum(__m256 v);

static float euclidean_distance(size_t k, const float v[k], const float u[k]) {
  __m256 squared_distance_accumulators = _mm256_setzero_ps();

  for (; k >= AVX_LENGTH; k -= AVX_LENGTH, v += AVX_LENGTH, u += AVX_LENGTH) {
    const __m256 v_elems = _mm256_loadu_ps(v);
    const __m256 u_elems = _mm256_loadu_ps(u);

    const __m256 difference = _mm256_sub_ps(v_elems, u_elems);
    squared_distance_accumulators =
        _mm256_fmadd_ps(difference, difference, squared_distance_accumulators);
  }

  if (k > 0) {
    const __m256i load_mask = maskn(k);

    const __m256 v_elems = _mm256_maskload_ps(v, load_mask);
    const __m256 u_elems = _mm256_maskload_ps(u, load_mask);

    const __m256 difference = _mm256_sub_ps(v_elems, u_elems);
    squared_distance_accumulators =
        _mm256_fmadd_ps(difference, difference, squared_distance_accumulators);
  }

  const float squared_distance = horizontal_sum(squared_distance_accumulators);
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
