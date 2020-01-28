cmake_minimum_required(VERSION 3.11)

project(vector-distance VERSION 0.1.0 LANGUAGES C)

add_executable(vector-distance src/vector_distance.c)
target_compile_features(vector-distance PRIVATE c_std_11)
target_link_libraries(vector-distance PRIVATE m pcg_random)
