#include <lib.h>

#include <assert.h>
#include <limits.h>
#include <math.h>

#include <emmintrin.h>
#include <immintrin.h>
#include <pmmintrin.h>
#include <xmmintrin.h>

#include <cblas.h>

#define SSE_LENGTH ((size_t)4)

size_t workspace_size(size_t n, size_t k) {
  (void)k;

  return n * n;
}

void pairwise_euclidean_distance(size_t n, size_t k, size_t l,
                                 const float X[n][k], float Z[restrict n][n],
                                 float workspace[restrict l]) {
  assert(l >= workspace_size(n, k));

  float *const inner_products = workspace;
  cblas_ssyrk(CblasRowMajor, CblasUpper, CblasNoTrans, (int)n, (int)k, 1.0f,
              (const float *)X, (int)k, 0.0f, inner_products, (int)n);

  for (size_t i = 0; i < n; ++i) {
    Z[i][i] = 0.0f;

    for (size_t j = i + 1; j < n; ++j) {
      const float squared_distance = inner_products[i * n + i] +
                                     inner_products[j * n + j] -
                                     2.0f * inner_products[i * n + j];

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
