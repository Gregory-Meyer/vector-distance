#include <lib.h>

#include <assert.h>
#include <math.h>

size_t workspace_size(size_t n, size_t k) {
  (void)k;

  return n;
}

static void squared_2_norm_for_each(size_t n, size_t k, const float X[n][k],
                                    float squared_2_norms[restrict n]);
static float squared_2_norm(size_t k, const float v[k]);
static float euclidean_distance(size_t k, const float v[k], const float u[k],
                                float v_squared_2_norm, float u_squared_2_norm);

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

static void squared_2_norm_for_each(size_t n, size_t k, const float X[n][k],
                                    float squared_2_norms[restrict n]) {
  for (size_t i = 0; i < n; ++i) {
    squared_2_norms[i] = squared_2_norm(k, X[i]);
  }
}

static float squared_2_norm(size_t k, const float v[k]) {
  float squared_norm_accumulator = 0.0f;

  for (size_t i = 0; i < k; ++i) {
    squared_norm_accumulator = fmaf(v[i], v[i], squared_norm_accumulator);
  }

  return squared_norm_accumulator;
}

static float euclidean_distance(size_t k, const float v[k], const float u[k],
                                float v_squared_2_norm,
                                float u_squared_2_norm) {
  float dot_product_accumulator = 0.0f;

  for (size_t i = 0; i < k; ++i) {
    dot_product_accumulator = fmaf(v[i], u[i], dot_product_accumulator);
  }

  const float squared_distance =
      v_squared_2_norm + u_squared_2_norm - 2.0f * dot_product_accumulator;

  if (squared_distance <= 0.0f) {
    return 0.0f;
  }

  const float distance = sqrtf(squared_distance);

  return distance;
}
