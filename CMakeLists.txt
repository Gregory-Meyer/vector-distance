cmake_minimum_required(VERSION 3.11)

project(vector-distance VERSION 0.1.0 LANGUAGES C)

find_package(BLAS REQUIRED)

add_executable(vector-distance src/vector_distance.c)
target_compile_features(vector-distance PRIVATE c_std_11)
target_link_libraries(vector-distance PRIVATE dl common)

add_library(scalar SHARED src/scalar.c)
target_compile_features(scalar PUBLIC c_std_11)
target_include_directories(scalar PUBLIC include)
target_link_libraries(scalar PRIVATE m)

add_library(scalar-trick SHARED src/scalar_trick.c)
target_compile_features(scalar-trick PUBLIC c_std_11)
target_include_directories(scalar-trick PUBLIC include)
target_link_libraries(scalar-trick PRIVATE m)

add_library(sse SHARED src/sse.c)
target_compile_features(sse PUBLIC c_std_11)
target_include_directories(sse PUBLIC include)
target_link_libraries(sse PRIVATE m)

add_library(sse-trick SHARED src/sse_trick.c)
target_compile_features(sse-trick PUBLIC c_std_11)
target_include_directories(sse-trick PUBLIC include)
target_link_libraries(sse-trick PRIVATE m)

add_library(avx SHARED src/avx.c)
target_compile_features(avx PUBLIC c_std_11)
target_include_directories(avx PUBLIC include)
target_link_libraries(avx PRIVATE m)

add_library(avx-trick SHARED src/avx_trick.c)
target_compile_features(avx-trick PUBLIC c_std_11)
target_include_directories(avx-trick PUBLIC include)
target_link_libraries(avx-trick PRIVATE m)

add_library(sgemm SHARED src/sgemm.c)
target_compile_features(sgemm PUBLIC c_std_11)
target_include_directories(sgemm PUBLIC include)
target_link_libraries(sgemm PRIVATE m openblas)

add_library(self-gemm SHARED src/self_gemm.c)
target_compile_features(self-gemm PUBLIC c_std_11)
target_include_directories(self-gemm PUBLIC include)
target_link_libraries(self-gemm PRIVATE m)

add_library(common STATIC src/common.c)
target_compile_features(common PUBLIC c_std_11)
target_include_directories(common PUBLIC include)
target_link_libraries(common PUBLIC pcg_random)
