#include <common.h>
#include <lib.h>

#include <dlfcn.h>
#include <getopt.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <pcg_variants.h>

typedef size_t(WorkspaceSizeFunction)(size_t n, size_t k);
typedef void(PairwiseEuclideanDistanceFunction)(size_t n, size_t k, size_t l,
                                                const float X[n][k],
                                                float Z[restrict n][n],
                                                float workspace[restrict l]);

static const struct option LONG_OPTIONS[] = {{"quiet", no_argument, NULL, 'q'},
                                             {NULL, 0, NULL, 0}};

int main(int argc, char *argv[argc]) {
  if (argc < 2) {
    fputs("error: missing argument LIB\n", stderr);

    return EXIT_FAILURE;
  } else if (argc < 3) {
    fputs("error: missing argument N\n", stderr);

    return EXIT_FAILURE;
  } else if (argc < 4) {
    fputs("error: missing argument K\n", stderr);

    return EXIT_FAILURE;
  }

  bool quiet = false;

  while (true) {
    const int ch = getopt_long(argc, argv, "q", LONG_OPTIONS, NULL);

    if (ch == -1) {
      break;
    }

    switch (ch) {
    case 'q':
      quiet = true;
      break;

    case '?':
    default:
      __builtin_unreachable();
    }
  }

  int errc = EXIT_SUCCESS;

  const char *const library_name = argv[optind];
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

  const char *const n_str = argv[optind + 1];
  size_t n;

  if (sscanf(n_str, "%zu", &n) != 1) {
    fprintf(stderr, "error: couldn't parse N (\"%s\") as an integer\n", n_str);

    errc = EXIT_FAILURE;
    goto cleanup_lib;
  }

  const char *const k_str = argv[optind + 2];
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

  float *const Z = malloc(n * n * sizeof(float));

  if (!Z) {
    fprintf(stderr, "error: couldn't allocate memory for Z (%zu x %zu)\n", n,
            n);

    errc = EXIT_FAILURE;
    goto cleanup_X;
  }

  const size_t l = workspace_size_fn(n, k);

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

  if (!quiet) {
    puts("X =");
    print_matrix(n, k, (const float(*)[])X);
  }

  pairwise_euclidean_distance_fn(n, k, l, (const float(*)[])X, (float(*)[])Z,
                                 workspace);

  if (!quiet) {
    puts("Z =");
    print_matrix(n, n, (const float(*)[])Z);
  }

  free(workspace);

cleanup_Z:
  free(Z);

cleanup_X:
  free(X);

cleanup_lib:
  dlclose(lib);

  return errc;
}
