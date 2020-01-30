#include <lib.h>

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>

#include <emmintrin.h>
#include <immintrin.h>
#include <pmmintrin.h>
#include <xmmintrin.h>

#define SSE_LENGTH ((size_t)4)

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

static float horizontal_sum(__m128 v);
static __m128i maskn(size_t n);

static float euclidean_distance(size_t k, const float v[k], const float u[k]) {
  __m128 squared_distance_accumulators = _mm_setzero_ps();

  for (; k >= SSE_LENGTH; k -= SSE_LENGTH, v += SSE_LENGTH, u += SSE_LENGTH) {
    const __m128 v_elems = _mm_loadu_ps(v);
    const __m128 u_elems = _mm_loadu_ps(u);

    const __m128 difference = _mm_sub_ps(v_elems, u_elems);
    squared_distance_accumulators =
        _mm_fmadd_ps(difference, difference, squared_distance_accumulators);
  }

  if (k > 0) {
    const __m128i load_mask = maskn(k);

    const __m128 v_elems = _mm_maskload_ps(v, load_mask);
    const __m128 u_elems = _mm_maskload_ps(u, load_mask);

    const __m128 difference = _mm_sub_ps(v_elems, u_elems);
    squared_distance_accumulators =
        _mm_fmadd_ps(difference, difference, squared_distance_accumulators);
  }

  const float squared_distance = horizontal_sum(squared_distance_accumulators);
  const float distance = sqrtf(squared_distance);

  return distance;
}

static float horizontal_sum(__m128 v) {
  __m128 shuf = _mm_movehdup_ps(v); // broadcast elements 3,1 to 2,0
  __m128 sums = _mm_add_ps(v, shuf);
  shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
  sums = _mm_add_ss(sums, shuf);

  return _mm_cvtss_f32(sums);
}

static __m128i maskn(size_t n) {
  assert(n <= 4);

  __m128i mask;

  switch (n) {
  default: // saturate n at 4
    mask = _mm_set1_epi32(INT_MIN);
    break;
  case 3:
    mask = _mm_set_epi32(0, INT_MIN, INT_MIN, INT_MIN);
    break;
  case 2:
    mask = _mm_set_epi32(0, 0, INT_MIN, INT_MIN);
    break;
  case 1:
    mask = _mm_set_epi32(0, 0, 0, INT_MIN);
    break;
  case 0:
    mask = _mm_setzero_si128();
    break;
  }

  return mask;
}
