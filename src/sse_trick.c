#include <lib.h>

#include <assert.h>
#include <limits.h>
#include <math.h>

#include <emmintrin.h>
#include <immintrin.h>
#include <pmmintrin.h>
#include <xmmintrin.h>

#define SSE_LENGTH ((size_t)4)

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

static __m128i maskn(size_t n);

static void squared_2_norm_for_each(size_t n, size_t k, const float X[n][k],
                                    float squared_2_norms[restrict n]) {
  for (; n >= SSE_LENGTH;
       n -= SSE_LENGTH, X += SSE_LENGTH, squared_2_norms += SSE_LENGTH) {
    __m128 squared_norms;

    for (size_t i = 0; i < SSE_LENGTH; ++i) {
      squared_norms[i] = squared_2_norm(k, X[i]);
    }

    _mm_storeu_ps(squared_2_norms, squared_norms);
  }

  if (n > 0) {
    __m128 squared_norms = _mm_setzero_ps();

    for (size_t i = 0; i < n; ++i) {
      squared_norms[i] = squared_2_norm(k, X[i]);
    }

    _mm_maskstore_ps(squared_2_norms, maskn(n), squared_norms);
  }
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

static float horizontal_sum(__m128 v);

static float squared_2_norm(size_t k, const float v[k]) {
  __m128 squared_norm_accumulators = _mm_setzero_ps();

  for (; k >= SSE_LENGTH; k -= SSE_LENGTH, v += SSE_LENGTH) {
    const __m128 elems = _mm_loadu_ps(v);

    squared_norm_accumulators =
        _mm_fmadd_ps(elems, elems, squared_norm_accumulators);
  }

  if (k > 0) {
    const __m128 elems = _mm_maskload_ps(v, maskn(k));

    squared_norm_accumulators =
        _mm_fmadd_ps(elems, elems, squared_norm_accumulators);
  }

  const float squared_norm = horizontal_sum(squared_norm_accumulators);

  return squared_norm;
}

static float euclidean_distance(size_t k, const float v[k], const float u[k],
                                float u_squared_2_norm,
                                float v_squared_2_norm) {
  __m128 inner_product_accumulators = _mm_setzero_ps();

  for (; k >= SSE_LENGTH; k -= SSE_LENGTH, v += SSE_LENGTH, u += SSE_LENGTH) {
    const __m128 v_elems = _mm_loadu_ps(v);
    const __m128 u_elems = _mm_loadu_ps(u);

    inner_product_accumulators =
        _mm_fmadd_ps(v_elems, u_elems, inner_product_accumulators);
  }

  if (k > 0) {
    const __m128i mask = maskn(k);

    const __m128 v_elems = _mm_maskload_ps(v, mask);
    const __m128 u_elems = _mm_maskload_ps(u, mask);

    inner_product_accumulators =
        _mm_fmadd_ps(v_elems, u_elems, inner_product_accumulators);
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

static float horizontal_sum(__m128 v) {
  __m128 shuf = _mm_movehdup_ps(v); // broadcast elements 3,1 to 2,0
  __m128 sums = _mm_add_ps(v, shuf);
  shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
  sums = _mm_add_ss(sums, shuf);

  return _mm_cvtss_f32(sums);
}
