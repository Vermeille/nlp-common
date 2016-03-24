cmake_minimum_required(VERSION 2.8)

project(libad CXX)

find_package(CUDA REQUIRED)

cuda_add_library(ad
    ad/nn.cpp
    ad/graph.cpp
    ad/helpers.cpp
    ad/hashtable.cpp
    ad/nn/fullyconn.cpp
    ad/nn/gru.cpp
    ad/nn/rnn.cpp
    ad/nn/mrnn.cpp
    ad/nn/lstm.cpp
    ad/nn/highway-layer.cpp
    ad/optimizers/momentum.cpp

    ad/ad.h
    ad/nn.h
    ad/graph.h
    ad/helpers.h
    ad/operators.h
    ad/hashtable.h
    ad/trainers.h
    ad/initializers.h
    ad/nn/fullyconn.h
    ad/nn/rnn.h
    ad/nn/gru.h
    ad/nn/mrnn.h
    ad/nn/lstm.h
    ad/nn/scrn.h
    ad/nn/highway-layer.h
    ad/optimizer.h
    ad/optimizers/sgd.h
    ad/optimizers/adam.h
    ad/optimizers/adagrad.h
    ad/optimizers/adadelta.h
    ad/optimizers/momentum.h

    ad/operators.cu
    ad/matrix.cu
    ad/initializers.cu
    ad/cuda/helpers.cu
    ad/kernels/identity.cu
    ad/kernels/kernels.h
    ad/kernels/relu.cu
    ad/optimizers/adam.cu
    ad/optimizers/adadelta.cu
    ad/optimizers/adagrad.cu
    ad/optimizers/sgd.cu
)

target_include_directories(ad PUBLIC
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(ad ${CUDA_curand_LIBRARY})
cuda_add_cublas_to_target(ad)

