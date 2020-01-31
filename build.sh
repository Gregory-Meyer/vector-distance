#!/usr/bin/env sh

set -e

THIS_SCRIPT_PARENT=$(realpath "$0")
THIS_SCRIPT_PARENT=$(dirname "${THIS_SCRIPT_PARENT}")

COMMON_FLAGS='-pipe -march=native -pedantic -Wall -Wcast-qual -Wconversion -Wextra -Wshadow -Wmissing-prototypes'

case "$1" in
    debug)
        BUILD_TYPE='Debug'
        C_FLAGS="${COMMON_FLAGS} -O0 -g -DDEBUG -march=native -fsanitize=address"
        BUILD_DIR='debug'
        ;;

    release)
        BUILD_TYPE='Release'
        C_FLAGS="${COMMON_FLAGS} -O3 -g -DNDEBUG -march=native"
        BUILD_DIR='release'
        ;;

    clean)
        rm -rf "${THIS_SCRIPT_PARENT}/debug" "${THIS_SCRIPT_PARENT}/release"
        exit 0
        ;;

    *)
        echo "unrecognized build mode '$1'" >&2
        exit 1
        ;;
esac

cmake \
    -S "${THIS_SCRIPT_PARENT}" \
    -B "${THIS_SCRIPT_PARENT}/${BUILD_DIR}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_C_FLAGS="${C_FLAGS}"
cmake --build "${THIS_SCRIPT_PARENT}/${BUILD_DIR}" -j $(nproc)
