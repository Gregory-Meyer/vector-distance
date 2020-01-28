#include <assert.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <immintrin.h>

#include <pcg_variants.h>

#define X_OFFSET 1
#define Y_OFFSET 2

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

  float *X = malloc((n * k + X_OFFSET) * sizeof(float));

  if (!X) {
    fprintf(stderr, "error: couldn't allocate memory for X (%zu x %zu)\n", n,
            k);

    return EXIT_FAILURE;
  }

  X += X_OFFSET;

  float *Y = malloc((m * k + Y_OFFSET) * sizeof(float));

  if (!Y) {
    free(X - X_OFFSET);

    fprintf(stderr, "error: couldn't allocate memory for Y (%zu x %zu)\n", m,
            k);

    return EXIT_FAILURE;
  }

  Y += Y_OFFSET;

  float *const Z = malloc(n * m * sizeof(float));

  if (!Z) {
    free(Y - Y_OFFSET);
    free(X - X_OFFSET);

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
  free(Y - Y_OFFSET);
  free(X - X_OFFSET);
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

#define AVX_SIZE sizeof(__m256)
#define AVX_NUM_SINGLES (sizeof(__m256) / sizeof(float))

static float avx_horizontal_sum(__m256 x);
static __m256i loadn_mask(size_t n);

static float euclidean_distance_unaligned(size_t k, const float v[k],
                                          const float u[k]);
static float euclidean_distance_aligned(size_t k, const float v[k],
                                        const float u[k],
                                        __m256 squared_distances);

static float euclidean_distance(size_t k, const float v[k], const float u[k]) {

  const uintptr_t v_offset = (uintptr_t)v % AVX_SIZE;
  const uintptr_t u_offset = (uintptr_t)u % AVX_SIZE;

  if (v_offset == u_offset) {
    __m256 squared_distances;

    if (v_offset != 0) {
      const size_t num_to_load = (AVX_SIZE - v_offset) / sizeof(float);
      const __m256i mask = loadn_mask(num_to_load);

      const __m256 v_elems = _mm256_maskload_ps(v, mask);
      const __m256 u_elems = _mm256_maskload_ps(u, mask);

      const __m256 offsets = _mm256_sub_ps(v_elems, u_elems);
      squared_distances = _mm256_mul_ps(offsets, offsets);

      k -= num_to_load;
      v += num_to_load;
      u += num_to_load;
    } else {
      squared_distances = _mm256_setzero_ps();
    }

    return euclidean_distance_aligned(k, v, u, squared_distances);
  } else {
    return euclidean_distance_unaligned(k, v, u);
  }
}

static float euclidean_distance_unaligned(size_t k, const float v[k],
                                          const float u[k]) {
  assert((uintptr_t)v % AVX_SIZE != 0 || (uintptr_t)u % AVX_SIZE != 0);

  const size_t num_vector_loads = k / (sizeof(__m256) / sizeof(float));
  __m256 squared_distances = _mm256_setzero_ps();

  for (size_t i = 0; i < num_vector_loads; ++i) {
    const __m256 v_elems = _mm256_loadu_ps(v);
    const __m256 u_elems = _mm256_loadu_ps(u);

    v += AVX_NUM_SINGLES;
    u += AVX_NUM_SINGLES;

    const __m256 offsets = _mm256_sub_ps(v_elems, u_elems);
    const __m256 squared_offsets = _mm256_mul_ps(offsets, offsets);
    squared_distances = _mm256_add_ps(squared_distances, squared_offsets);
  }

  if (k % AVX_NUM_SINGLES != 0) {
    const __m256i mask = loadn_mask(AVX_NUM_SINGLES - k);

    const __m256 v_elems = _mm256_maskload_ps(v, mask);
    const __m256 u_elems = _mm256_maskload_ps(u, mask);

    const __m256 offsets = _mm256_sub_ps(v_elems, u_elems);
    const __m256 squared_offsets = _mm256_mul_ps(offsets, offsets);
    squared_distances = _mm256_add_ps(squared_distances, squared_offsets);
  }

  return sqrtf(avx_horizontal_sum(squared_distances));
}

static float euclidean_distance_aligned(size_t k, const float v[k],
                                        const float u[k],
                                        __m256 squared_distances) {
  assert((uintptr_t)v % AVX_SIZE == 0);
  assert((uintptr_t)u % AVX_SIZE == 0);

  const size_t num_vector_loads = k / AVX_NUM_SINGLES;

  for (size_t i = 0; i < num_vector_loads; ++i) {
    const __m256 v_elems = _mm256_load_ps(v);
    const __m256 u_elems = _mm256_load_ps(u);

    v += AVX_NUM_SINGLES;
    u += AVX_NUM_SINGLES;

    const __m256 offsets = _mm256_sub_ps(v_elems, u_elems);
    const __m256 squared_offsets = _mm256_mul_ps(offsets, offsets);
    squared_distances = _mm256_add_ps(squared_distances, squared_offsets);
  }

  if (k % 8 != 0) {
    const __m256i mask = loadn_mask(AVX_NUM_SINGLES - k);

    const __m256 v_elems = _mm256_maskload_ps(v, mask);
    const __m256 u_elems = _mm256_maskload_ps(u, mask);

    const __m256 offsets = _mm256_sub_ps(v_elems, u_elems);
    const __m256 squared_offsets = _mm256_mul_ps(offsets, offsets);
    squared_distances = _mm256_add_ps(squared_distances, squared_offsets);
  }

  return sqrtf(avx_horizontal_sum(squared_distances));
}

static __m256i loadn_mask(size_t n) {
  assert(n < 8);

  __m256i mask = _mm256_setzero_si256();

  for (size_t i = 0; i < n; ++i) {
    ((uint32_t *)&mask)[i] = 0xffffffff;
  }

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
