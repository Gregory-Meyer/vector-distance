#ifndef LIB_H
#define LIB_H

#include <stddef.h>

size_t workspace_size(size_t n, size_t k);
void pairwise_euclidean_distance(size_t n, size_t k, size_t l,
                                 const float X[n][k], float Z[restrict n][n],
                                 float workspace[restrict l]);

#endif
