#include <common.h>
#include <lib.h>

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <dlfcn.h>

#include <pcg_variants.h>

typedef size_t(WorkspaceSizeFunction)(size_t, size_t, size_t);
typedef void(PairwiseEuclideanDistanceFunction)(size_t n, size_t m, size_t k,
                                                size_t l, const float X[n][k],
                                                const float Y[m][k],
                                                float Z[restrict n][m],
                                                float workspace[restrict l]);

int main(int argc, const char *const argv[argc]) {
  if (argc < 2) {
    fputs("error: missing argument LIB\n", stderr);

    return EXIT_FAILURE;
  } else if (argc < 3) {
    fputs("error: missing argument N\n", stderr);

    return EXIT_FAILURE;
  } else if (argc < 4) {
    fputs("error: missing argument M\n", stderr);

    return EXIT_FAILURE;
  } else if (argc < 5) {
    fputs("error: missing argument K\n", stderr);

    return EXIT_FAILURE;
  }

  int errc = EXIT_SUCCESS;

  const char *const library_name = argv[1];
  void *const lib = dlopen(library_name, RTLD_NOW);

  if (!lib) {
    fprintf(stderr, "error: couldn't open shared library '%s': %s\n",
            library_name, dlerror());

    return EXIT_FAILURE;
  }

  WorkspaceSizeFunction *const workspace_size_fn =
      (WorkspaceSizeFunction *)dlsym(lib, "workspace_size");

  if (!workspace_size_fn) {
    fprintf(stderr,
            "error: couldn't resolve symbol 'workspace_size' from shared "
            "library '%s': %s\n",
            library_name, dlerror());

    errc = EXIT_FAILURE;
    goto cleanup_lib;
  }

  PairwiseEuclideanDistanceFunction *const pairwise_euclidean_distance_fn =
      (PairwiseEuclideanDistanceFunction *)dlsym(lib,
                                                 "pairwise_euclidean_distance");

  if (!pairwise_euclidean_distance_fn) {
    fprintf(stderr,
            "error: couldn't resolve symbol 'pairwise_euclidean_distance' from "
            "shared library '%s': %s\n",
            library_name, dlerror());

    errc = EXIT_FAILURE;
    goto cleanup_lib;
  }

  const char *const n_str = argv[2];
  size_t n;

  if (sscanf(n_str, "%zu", &n) != 1) {
    fprintf(stderr, "error: couldn't parse N (\"%s\") as an integer\n", n_str);

    errc = EXIT_FAILURE;
    goto cleanup_lib;
  }

  const char *const m_str = argv[3];
  size_t m;

  if (sscanf(m_str, "%zu", &m) != 1) {
    fprintf(stderr, "error: couldn't parse M (\"%s\") as an integer\n", m_str);

    errc = EXIT_FAILURE;
    goto cleanup_lib;
  }

  const char *const k_str = argv[4];
  size_t k;

  if (sscanf(k_str, "%zu", &k) != 1) {
    fprintf(stderr, "error: couldn't parse K (\"%s\") as an integer\n", k_str);

    errc = EXIT_FAILURE;
    goto cleanup_lib;
  }

  float *const X = malloc(n * k * sizeof(float));

  if (!X) {
    fprintf(stderr, "error: couldn't allocate memory for X (%zu x %zu)\n", n,
            k);

    errc = EXIT_FAILURE;
    goto cleanup_lib;
  }

  float *const Y = malloc(m * k * sizeof(float));

  if (!Y) {
    fprintf(stderr, "error: couldn't allocate memory for Y (%zu x %zu)\n", m,
            k);

    errc = EXIT_FAILURE;
    goto cleanup_X;
  }

  float *const Z = malloc(n * m * sizeof(float));

  if (!Z) {
    fprintf(stderr, "error: couldn't allocate memory for Z (%zu x %zu)\n", n,
            m);

    errc = EXIT_FAILURE;
    goto cleanup_Y;
  }

  const size_t l = workspace_size_fn(n, m, k);

  float *workspace = NULL;

  if (l > 0) {
    workspace = malloc(l * sizeof(float));

    if (!workspace) {
      fprintf(stderr, "error: couldn't allocate memory for workspace (%zu)\n",
              l);

      errc = EXIT_FAILURE;
      goto cleanup_Z;
    }
  }

  pcg32_random_t rng = PCG32_INITIALIZER;
  fill_with_randomness(&rng, n, k, (float(*)[])X);
  fill_with_randomness(&rng, m, k, (float(*)[])Y);

  puts("X =");
  print_matrix(n, k, (const float(*)[])X);

  puts("Y =");
  print_matrix(m, k, (const float(*)[])Y);

  pairwise_euclidean_distance_fn(n, m, k, l, (const float(*)[])X,
                                 (const float(*)[])Y, (float(*)[])Z, workspace);

  puts("Z =");
  print_matrix(n, m, (const float(*)[])Z);

  free(workspace);

cleanup_Z:
  free(Z);

cleanup_Y:
  free(Y);

cleanup_X:
  free(X);

cleanup_lib:
  dlclose(lib);

  return errc;
}
