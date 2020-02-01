#include <lib.h>

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <string.h>

#include <immintrin.h>
#include <pmmintrin.h>
#include <xmmintrin.h>

#define AVX_LENGTH ((size_t)8)

size_t workspace_size(size_t n, size_t m) {
  (void)n;
  (void)m;

  return n + (m * n) + (n * (n - 1) / 2);
}

static void
do_pairwise_euclidean_distance(size_t n, size_t m, const float X[restrict n][m],
                               float Z[restrict n][n], float XT[restrict m][n],
                               float squared_norms[restrict n],
                               float inner_products[restrict n * (n - 1) / 2]);

void pairwise_euclidean_distance(size_t n, size_t m, size_t p,
                                 const float X[n][m], float Z[restrict n][n],
                                 float workspace[restrict p]) {
  assert(p >= n);

  float *restrict const squared_norms = workspace; // n
  float *restrict const XT = squared_norms + n;    // m x n
  float *restrict const inner_products = XT + m * n;

  do_pairwise_euclidean_distance(n, m, X, Z, (float(*)[])XT, squared_norms,
                                 inner_products);
}

static void transpose(size_t n, size_t m, const float X[restrict n][m],
                      float XT[restrict m][n]);
static void
compute_inner_products(size_t n, size_t m, const float X[restrict n][m],
                       const float XT[restrict m][n],
                       float inner_products[restrict n * (n - 1) / 2],
                       float squared_norms[restrict n]);
static void compute_distances(
    size_t n, const float inner_products[restrict n * (n - 1) / 2],
    const float squared_norms[restrict n], float Z[restrict n][n]);

static void
do_pairwise_euclidean_distance(size_t n, size_t m, const float X[restrict n][m],
                               float Z[restrict n][n], float XT[restrict m][n],
                               float squared_norms[restrict n],
                               float inner_products[restrict n * (n - 1) / 2]) {
  transpose(n, m, X, XT);
  compute_inner_products(n, m, X, (const float(*)[])XT, inner_products,
                         squared_norms);
  compute_distances(n, inner_products, squared_norms, Z);
}

static void transpose(size_t n, size_t m, const float X[restrict n][m],
                      float XT[restrict m][n]) {
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      XT[j][i] = X[i][j];
    }
  }
}

static float squared_euclidean_norm(size_t n, const float v[n]);
static void
compute_inner_products_for_row(size_t n, size_t m, const float X[restrict n][m],
                               const float XT[restrict m][n],
                               float inner_products[restrict n * (n - 1) / 2],
                               size_t i, size_t t0, size_t t1);

static void
compute_inner_products(size_t n, size_t m, const float X[restrict n][m],
                       const float XT[restrict m][n],
                       float inner_products[restrict n * (n - 1) / 2],
                       float squared_norms[restrict n]) {
  const size_t num_triangle_indices = n * (n - 1) / 2;
  memset(inner_products, 0, num_triangle_indices * sizeof(float));

  size_t i = 0;
  for (size_t triangle_index = 0; triangle_index < num_triangle_indices; ++i) {
    squared_norms[i] = squared_euclidean_norm(m, X[i]);

    const size_t j_offset = i + 1;
    const size_t columns_to_compute = n - j_offset;

    const size_t t0 = triangle_index;
    const size_t t1 = t0 + columns_to_compute;

    compute_inner_products_for_row(n, m, X, XT, inner_products, i, t0, t1);

    triangle_index = t1;
  }

  squared_norms[n - 1] = squared_euclidean_norm(m, X[n - 1]);
}

static void compute_distances(
    size_t n, const float inner_products[restrict n * (n - 1) / 2],
    const float squared_norms[restrict n], float Z[restrict n][n]) {
  const size_t num_triangle_elements = n * (n - 1) / 2;

  Z[0][0] = 0.0f;

  size_t i = 0;
  size_t j = 1;
  for (size_t triangle_index = 0; triangle_index < num_triangle_elements;
       ++triangle_index) {
    const float squared_distance = squared_norms[i] -
                                   2.0f * inner_products[triangle_index] +
                                   squared_norms[j];

    float distance;

    if (squared_distance <= 0.0f) {
      distance = 0.0f;
    } else {
      distance = sqrtf(squared_distance);
    }

    Z[i][j] = distance;
    Z[j][i] = distance;

    if (j == n - 1) {
      ++i;
      j = i + 1;

      Z[i][i] = 0.0f;
    } else {
      ++j;
    }
  }
}

static __m256i maskn(size_t n);
static float horizontal_sum(__m256 v);

static float squared_euclidean_norm(size_t n, const float v[n]) {
  __m256 squared_norm_accumulators = _mm256_setzero_ps();

  for (; n >= AVX_LENGTH; n -= AVX_LENGTH, v += AVX_LENGTH) {
    const __m256 v_elems = _mm256_loadu_ps(v);
    squared_norm_accumulators =
        _mm256_fmadd_ps(v_elems, v_elems, squared_norm_accumulators);
  }

  if (n > 0) {
    const __m256i mask = maskn(n);

    const __m256 v_elems = _mm256_maskload_ps(v, mask);
    squared_norm_accumulators =
        _mm256_fmadd_ps(v_elems, v_elems, squared_norm_accumulators);
  }

  const float squared_norm = horizontal_sum(squared_norm_accumulators);

  return squared_norm;
}

static void
compute_inner_products_for_row(size_t n, size_t m, const float X[restrict n][m],
                               const float XT[restrict m][n],
                               float inner_products[restrict n * (n - 1) / 2],
                               size_t i, size_t t0, size_t t1) {
  const size_t num_columns = t1 - t0;

  for (size_t k = 0; k < m; ++k) {
    const __m256 Xik = _mm256_set1_ps(X[i][k]);
    const float *offset_XT = &XT[k][i + 1];
    float *offset_products = inner_products + t0;

    size_t offset_index = num_columns;
    for (; offset_index >= AVX_LENGTH; offset_index -= AVX_LENGTH,
                                       offset_products += AVX_LENGTH,
                                       offset_XT += AVX_LENGTH) {
      const __m256 current = _mm256_loadu_ps(offset_products);
      const __m256 XTkj = _mm256_loadu_ps(offset_XT);
      const __m256 updated = _mm256_fmadd_ps(Xik, XTkj, current);
      _mm256_storeu_ps(offset_products, updated);
    }

    if (offset_index > 0) {
      const __m256i mask = maskn(offset_index);
      const __m256 current = _mm256_maskload_ps(offset_products, mask);
      const __m256 XTkj = _mm256_maskload_ps(offset_XT, mask);
      const __m256 updated = _mm256_fmadd_ps(Xik, XTkj, current);
      _mm256_maskstore_ps(offset_products, mask, updated);
    }
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
