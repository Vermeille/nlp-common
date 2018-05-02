cmake_minimum_required(VERSION 2.8)

project(NLP-COMMON)
SET(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g3 -Wall -Wextra -Wno-deprecated-declarations")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

add_subdirectory(autodiff/src)
add_subdirectory(src)
add_subdirectory(tests)


