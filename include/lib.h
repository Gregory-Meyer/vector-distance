#ifndef LIB_H
#define LIB_H

#include <stddef.h>

size_t workspace_size(size_t n, size_t m, size_t k);
void pairwise_euclidean_distance(size_t n, size_t m, size_t k, size_t l,
                                 const float X[n][k], const float Y[m][k],
                                 float Z[restrict n][m],
                                 float workspace[restrict l]);

#endif
