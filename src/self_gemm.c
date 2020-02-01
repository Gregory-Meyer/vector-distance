#include <lib.h>

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>

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
compute_inner_products(size_t n, size_t m, const float X[restrict n][m],
                       const float XT[restrict m][n],
                       float inner_products[restrict n * (n - 1) / 2],
                       float squared_norms[restrict n]) {
  const size_t num_triangle_indices = n * (n - 1) / 2;

  size_t i = 0;
  for (size_t triangle_index = 0; triangle_index < num_triangle_indices; ++i) {
    squared_norms[i] = squared_euclidean_norm(m, X[i]);

    const size_t j_offset = i + 1;
    const size_t columns_to_compute = n - j_offset;

    const size_t first_triangle_index = triangle_index;
    const size_t last_triangle_index =
        first_triangle_index + columns_to_compute;

    for (size_t t = first_triangle_index; t < last_triangle_index; ++t) {
      inner_products[t] = 0.0f;
    }

    for (size_t k = 0; k < m; ++k) {
      size_t j = j_offset;
      for (size_t t = first_triangle_index; t < last_triangle_index; ++t, ++j) {
        inner_products[t] += X[i][k] * XT[k][j];
      }
    }

    triangle_index = last_triangle_index;
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

static float squared_euclidean_norm(size_t n, const float v[n]) {
  float squared_norm_accumulator = 0.0f;

  for (size_t i = 0; i < n; ++i) {
    squared_norm_accumulator += v[i] * v[i];
  }

  return squared_norm_accumulator;
}
