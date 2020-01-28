#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <immintrin.h>

#include <pcg_variants.h>

static void fill_random(pcg32_random_t *rng, size_t n, size_t m, float X[n][m]);
static void vector_distances(size_t n, size_t m, size_t k, const float X[n][k],
                             const float Y[m][k], float Z[restrict n][m]);
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

  float *const X = malloc(n * k * sizeof(float));

  if (!X) {
    fprintf(stderr, "error: couldn't allocate memory for X (%zu x %zu)\n", n,
            k);

    return EXIT_FAILURE;
  }

  float *const Y = malloc(m * k * sizeof(float));

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

  pcg32_random_t rng = PCG32_INITIALIZER;
  fill_random(&rng, n, k, (float(*)[])X);
  fill_random(&rng, m, k, (float(*)[])Y);

  puts("X =");
  print_matrix(n, k, (const float(*)[])X);

  puts("Y =");
  print_matrix(m, k, (const float(*)[])Y);

  vector_distances(n, m, k, (const float(*)[])X, (const float(*)[])Y,
                   (float(*)[])Z);

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

static float euclidean_distance(size_t k, const float v[k], const float u[k]);

static void vector_distances(size_t n, size_t m, size_t k, const float X[n][k],
                             const float Y[m][k], float Z[restrict n][m]) {
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i; j < m; ++j) {
      const float distance = euclidean_distance(k, X[i], Y[j]);

      Z[i][j] = Z[j][i] = distance;
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

static float horizontal_sum(__m256 x);

static float euclidean_distance(size_t k, const float v[k], const float u[k]) {
  __m256 squared_distances = _mm256_setzero_ps();

  const size_t num_vector_loads = k / 8;

  const float *v_load_ptr = v;
  const float *u_load_ptr = u;

  for (size_t i = 0; i < num_vector_loads; ++i) {
    const __m256 v_elems = _mm256_loadu_ps(v_load_ptr);
    const __m256 u_elems = _mm256_loadu_ps(u_load_ptr);

    v_load_ptr += 8;
    u_load_ptr += 8;

    const __m256 offsets = _mm256_sub_ps(v_elems, u_elems);
    const __m256 squared_offsets = _mm256_mul_ps(offsets, offsets);
    squared_distances = _mm256_add_ps(squared_distances, squared_offsets);
  }

  if (k % 8 != 0) {
    __m256i mask = _mm256_setzero_si256();

    for (size_t i = 0; i < 8 - (k % 8); ++i) {
      ((uint32_t *)&mask)[7 - i] = 0xffffffff;
    }

    const __m256 v_elems = _mm256_maskload_ps(v_load_ptr, mask);
    const __m256 u_elems = _mm256_maskload_ps(u_load_ptr, mask);

    const __m256 offsets = _mm256_sub_ps(v_elems, u_elems);
    const __m256 squared_offsets = _mm256_mul_ps(offsets, offsets);
    squared_distances = _mm256_add_ps(squared_distances, squared_offsets);
  }

  return sqrtf(horizontal_sum(squared_distances));
}

static float horizontal_sum(__m256 x) {
  // https://stackoverflow.com/questions/13219146/how-to-sum-m256-horizontally/13222410#13222410

  // hiQuad = ( x7, x6, x5, x4 )
  const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
  // loQuad = ( x3, x2, x1, x0 )
  const __m128 loQuad = _mm256_castps256_ps128(x);
  // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
  const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
  // loDual = ( -, -, x1 + x5, x0 + x4 )
  const __m128 loDual = sumQuad;
  // hiDual = ( -, -, x3 + x7, x2 + x6 )
  const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
  // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
  const __m128 sumDual = _mm_add_ps(loDual, hiDual);
  // lo = ( -, -, -, x0 + x2 + x4 + x6 )
  const __m128 lo = sumDual;
  // hi = ( -, -, -, x1 + x3 + x5 + x7 )
  const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
  // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
  const __m128 sum = _mm_add_ss(lo, hi);

  return _mm_cvtss_f32(sum);
}
