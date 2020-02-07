#include <lib.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

size_t workspace_size(size_t n, size_t k) { return (k * n) + (n * n); }

static void transpose(size_t n, size_t m, const float X[n][m], float XT[m][n]);
static void symmetric_rank_k_update(size_t n, size_t m, const float X[n][m],
                                    const float XT[m][n],
                                    float XXT[restrict n][n]);

void pairwise_euclidean_distance(size_t n, size_t k, size_t l,
                                 const float X[n][k], float Z[restrict n][n],
                                 float workspace[restrict l]) {
  assert(l >= n);

  float *const XT = workspace;
  transpose(n, k, X, (float(*)[])XT);

  float *const XXT = workspace + (k * n);
  symmetric_rank_k_update(n, k, X, (const float(*)[])XT, (float(*)[])XXT);

  for (size_t i = 0; i < n; ++i) {
    Z[i][i] = 0.0f;

    for (size_t j = i + 1; j < n; ++j) {
      const float squared_distance =
          XXT[i * n + i] - 2.0f * XXT[i * n + j] + XXT[j * n + j];

      float distance;

      if (squared_distance <= 0.0f) {
        distance = 0.0f;
      } else {
        distance = sqrtf(squared_distance);
      }

      Z[i][j] = distance;
      Z[j][i] = distance;
    }
  }
}

static void transpose(size_t n, size_t m, const float X[n][m], float XT[m][n]) {
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      XT[j][i] = X[i][j];
    }
  }
}

static float squared_2_norm(size_t k, const float v[k]);

static void symmetric_rank_k_update(size_t n, size_t m, const float X[n][m],
                                    const float XT[m][n],
                                    float XXT[restrict n][n]) {
  memset(XXT, 0, (n * n) * sizeof(float));

  for (size_t i = 0; i < n; ++i) {
    XXT[i][i] = squared_2_norm(m, X[i]);

    for (size_t k = 0; k < m; ++k) {
      for (size_t j = i + 1; j < n; ++j) {
        XXT[i][j] = fmaf(X[i][k], XT[k][j], XXT[i][j]);
      }
    }
  }
}

static float squared_2_norm(size_t k, const float v[k]) {
  float squared_norm_accumulator = 0.0f;

  for (size_t i = 0; i < k; ++i) {
    squared_norm_accumulator = fmaf(v[i], v[i], squared_norm_accumulator);
  }

  return squared_norm_accumulator;
}
