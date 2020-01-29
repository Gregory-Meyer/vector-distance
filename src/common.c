#include <common.h>

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

void fill_with_randomness(pcg32_random_t *rng, size_t n, size_t m,
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

void print_matrix(size_t n, size_t m, const float A[n][m]) {
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
