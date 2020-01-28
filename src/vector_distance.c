#include <assert.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <immintrin.h>

#include <pcg_variants.h>

#define AVX_SIZE sizeof(__m256)
#define AVX_NUM_SINGLES (sizeof(__m256) / sizeof(float))

static void fill_random(pcg32_random_t *rng, size_t n, size_t m, float X[n][m]);
static void vector_distances(size_t n, size_t m, size_t k, const float X[n][k],
                             const float Y[m][k], float Z[restrict n][m],
                             float xn[restrict n], float yn[restrict m]);
static void print_matrix(size_t n, size_t m, const float A[n][m]);

int main(int argc, const char *const argv[argc]) {
  if (argc < 2) {
    fputs("error: missing argument N\n", stderr);

    return EXIT_FAILURE;
  } else if (argc < 3) {
    fputs("error: missing argument M\n", stderr);

    return EXIT_FAILURE;
  } else if (argc < 4) {
    fputs("error: missing argument K\n", stderr);

    return EXIT_FAILURE;
  }

  const char *const n_str = argv[1];
  size_t n;

  if (sscanf(n_str, "%zu", &n) != 1) {
    fprintf(stderr, "error: couldn't parse N (\"%s\") as an integer\n", n_str);

    return EXIT_FAILURE;
  }

  const char *const m_str = argv[2];
  size_t m;

  if (sscanf(m_str, "%zu", &m) != 1) {
    fprintf(stderr, "error: couldn't parse M (\"%s\") as an integer\n", m_str);

    return EXIT_FAILURE;
  }

  const char *const k_str = argv[3];
  size_t k;

  if (sscanf(k_str, "%zu", &k) != 1) {
    fprintf(stderr, "error: couldn't parse K (\"%s\") as an integer\n", k_str);

    return EXIT_FAILURE;
  }

  float *const X = aligned_alloc(AVX_SIZE, n * k * sizeof(float));

  if (!X) {
    fprintf(stderr, "error: couldn't allocate memory for X (%zu x %zu)\n", n,
            k);

    return EXIT_FAILURE;
  }

  float *const Y = aligned_alloc(AVX_SIZE, m * k * sizeof(float));

  if (!Y) {
    free(X);

    fprintf(stderr, "error: couldn't allocate memory for Y (%zu x %zu)\n", m,
            k);

    return EXIT_FAILURE;
  }

  float *const Z = malloc(n * m * sizeof(float));

  if (!Z) {
    free(Y);
    free(X);

    fprintf(stderr, "error: couldn't allocate memory for Z (%zu x %zu)\n", n,
            m);

    return EXIT_FAILURE;
  }

  float *const xn = malloc(n * sizeof(float));
  float *const yn = malloc(m * sizeof(float));

  pcg32_random_t rng = PCG32_INITIALIZER;
  fill_random(&rng, n, k, (float(*)[])X);
  fill_random(&rng, m, k, (float(*)[])Y);

  puts("X =");
  print_matrix(n, k, (const float(*)[])X);

  puts("Y =");
  print_matrix(m, k, (const float(*)[])Y);

  vector_distances(n, m, k, (const float(*)[])X, (const float(*)[])Y,
                   (float(*)[])Z, xn, yn);

  puts("Z =");
  print_matrix(n, m, (const float(*)[])Z);

  free(Z);
  free(Y);
  free(X);
}

static void fill_random(pcg32_random_t *rng, size_t n, size_t m,
                        float X[n][m]) {
  _Static_assert(
      FLT_RADIX == 2 && sizeof(float) == sizeof(uint32_t),
      "float must be an IEEE-754 single precision floating point number");

  static const uint32_t ZERO_EXPONENT = 0x3e800000;

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      const uint32_t randomness = pcg32_random_r(rng);
      const uint32_t fraction_bits = randomness & ((1 << 24) - 1);
      const uint32_t as_integer = ZERO_EXPONENT | fraction_bits;

      memcpy(&X[i][j], &as_integer, sizeof(as_integer));
    }
  }
}

static float euclidean_distance2(size_t k, const float v[k], const float u[k],
                                 float vn, float un);
static void squared_norms(size_t n, size_t k, const float X[n][k],
                          float xn[restrict n]);

static void vector_distances(size_t n, size_t m, size_t k, const float X[n][k],
                             const float Y[m][k], float Z[restrict n][m],
                             float xn[restrict n], float yn[restrict m]) {
  squared_norms(n, k, X, xn);
  squared_norms(m, k, Y, yn);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i; j < m; ++j) {
      const float distance2 = euclidean_distance2(k, X[i], Y[j], xn[i], yn[j]);

      if (distance2 <= 0.0f) {
        Z[i][j] = Z[j][i] = 0.0f;
      } else {
        Z[i][j] = Z[j][i] = sqrtf(distance2);
      }
    }
  }
}

static void print_matrix(size_t n, size_t m, const float A[n][m]) {
  putchar('[');

  for (size_t i = 0; i < n; ++i) {
    if (i > 0) {
      fputs(",[", stdout);
    } else {
      putchar('[');
    }

    for (size_t j = 0; j < m; ++j) {
      if (j > 0) {
        fputs(",", stdout);
      }

      printf("%.9f", A[i][j]);
    }

    putchar(']');
  }

  puts("]");
}

static float squared_norm(size_t k, const float x[k]);
static __m256i maskn(size_t n);

static void squared_norms(size_t n, size_t k, const float X[n][k],
                          float xn[restrict n]) {
  const size_t num_vector_stores = n / AVX_NUM_SINGLES;

  for (size_t i = 0; i < num_vector_stores; ++i) {
    __m256 to_store;

    for (size_t j = 0; j < AVX_NUM_SINGLES; ++j) {
      to_store[j] = squared_norm(k, X[j]);
    }

    _mm256_storeu_ps(xn, to_store);

    n -= AVX_NUM_SINGLES;
    X += AVX_NUM_SINGLES;
    xn += AVX_NUM_SINGLES;
  }

  if (n > 0) {
    __m256 to_store = _mm256_setzero_ps();

    for (size_t i = 0; i < n; ++i) {
      to_store[i] = squared_norm(k, X[i]);
    }

    _mm256_maskstore_ps(xn, maskn(n), to_store);
  }
}

static float avx_horizontal_sum(__m256 x);

static float euclidean_distance2(size_t k, const float v[k], const float u[k],
                                 float vn, float un) {
  const size_t num_vector_loads = k / (sizeof(__m256) / sizeof(float));
  __m256 inner_product_accumulator = _mm256_setzero_ps();

  for (size_t i = 0; i < num_vector_loads; ++i) {
    const __m256 v_elems = _mm256_loadu_ps(v);
    const __m256 u_elems = _mm256_loadu_ps(u);

    k -= AVX_NUM_SINGLES;
    v += AVX_NUM_SINGLES;
    u += AVX_NUM_SINGLES;

    const __m256 inner_product_elems = _mm256_mul_ps(v_elems, u_elems);
    inner_product_accumulator =
        _mm256_add_ps(inner_product_accumulator, inner_product_elems);
  }

  if (k > 0) {
    const __m256i mask = maskn(k);

    const __m256 v_elems = _mm256_maskload_ps(v, mask);
    const __m256 u_elems = _mm256_maskload_ps(u, mask);

    const __m256 inner_product_elems = _mm256_mul_ps(v_elems, u_elems);
    inner_product_accumulator =
        _mm256_add_ps(inner_product_accumulator, inner_product_elems);
  }

  const float inner_product = avx_horizontal_sum(inner_product_accumulator);

  return vn + un - 2.0f * inner_product;
}

static float squared_norm(size_t k, const float x[k]) {
  const size_t num_vector_loads = k / AVX_NUM_SINGLES;
  __m256 norm_components = _mm256_setzero_ps();

  for (size_t i = 0; i < num_vector_loads; ++i) {
    const __m256 elems = _mm256_loadu_ps(x);

    k -= AVX_NUM_SINGLES;
    x += AVX_NUM_SINGLES;

    const __m256 squared = _mm256_mul_ps(elems, elems);
    norm_components = _mm256_add_ps(norm_components, squared);
  }

  if (k > 0) {
    const __m256i mask = maskn(k);

    const __m256 elems = _mm256_maskload_ps(x, mask);

    const __m256 squared = _mm256_mul_ps(elems, elems);
    norm_components = _mm256_add_ps(norm_components, squared);
  }

  return avx_horizontal_sum(norm_components);
}

static __m256i maskn(size_t n) {
  assert(n < 8);

  __m256i mask = _mm256_setzero_si256();
  memset(&mask, 0xff, sizeof(uint32_t) * n);

  return mask;
}

static float sse_horizontal_sum(__m128 v);

static float avx_horizontal_sum(__m256 v) {
  __m128 vlow = _mm256_castps256_ps128(v);
  __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
  vlow = _mm_add_ps(vlow, vhigh);             // add the low 128

  return sse_horizontal_sum(
      vlow); // and inline the sse3 version, which is optimal for AVX
             // (no wasted instructions, and all of them are the 4B minimum)
}

static float sse_horizontal_sum(__m128 v) {
  __m128 shuf = _mm_movehdup_ps(v); // broadcast elements 3,1 to 2,0
  __m128 sums = _mm_add_ps(v, shuf);
  shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}
