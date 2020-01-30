#include <lib.h>

#include <math.h>

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

static float euclidean_distance(size_t k, const float v[k], const float u[k]) {
  float squared_distance_accumulator = 0.0f;

  for (size_t i = 0; i < k; ++i) {
    const float element_difference = v[i] - u[i];

    squared_distance_accumulator = fmaf(element_difference, element_difference,
                                        squared_distance_accumulator);
  }

  const float distance = sqrtf(squared_distance_accumulator);

  return distance;
}
