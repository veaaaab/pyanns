cmake_minimum_required (VERSION 3.1)
project(helpa LANGUAGES CXX)

add_library(helpa INTERFACE)
target_include_directories(helpa INTERFACE .) 

set(CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

SET(CMAKE_CXX_FLAGS  "-Wall -Wextra -O3 -std=c++20 -march=native -fpic -fopenmp -ftree-vectorize -fno-exceptions -fno-rtti" )

OPTION(USE_AVX512BF16 "Use avx512_bf16 instruction set" ON)
OPTION(USE_AVX512FP16 "Use avx512_fp16 instruction set" OFF)

if(USE_AVX512BF16)
    add_definitions(-DUSE_AVX512BF16)
endif(USE_AVX512BF16)

if(USE_AVX512FP16)
    add_definitions(-DUSE_AVX512FP16)
endif(USE_AVX512FP16)

add_executable(main main.cc)
target_link_libraries(main helpa)

option(WITH_BENCH "Build benchmark" ON)
if (WITH_BENCH)
    add_subdirectory(bench)
endif()
