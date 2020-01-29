#ifndef COMMON_H
#define COMMON_H

#include <stddef.h>

#include <pcg_variants.h>

void fill_with_randomness(pcg32_random_t *rng, size_t n, size_t m,
                          float X[n][m]);
void print_matrix(size_t n, size_t m, const float A[n][m]);

#endif
