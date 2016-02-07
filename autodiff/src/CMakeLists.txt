cmake_minimum_required(VERSION 2.8)

project(libad CXX)

find_package(Eigen3)

add_library(ad
    ad/nn.cpp
    ad/graph.cpp
    ad/helpers.cpp
    ad/operators.cpp
    ad/nn/fullyconn.cpp
    ad/nn/rnn1.cpp
    ad/nn/rnn2.cpp
    ad/nn/rnn-encoder.cpp
    ad/nn/highway-layer.cpp

    ad/ad.h
    ad/nn.h
    ad/graph.h
    ad/helpers.h
    ad/operators.h
    ad/nn/fullyconn.h
    ad/nn/input-layer.h
    ad/nn/neural-output.h
    ad/nn/rnn1.h
    ad/nn/rnn2.h
    ad/nn/rnn-encoder.h
    ad/nn/highway-layer.h
    ad/optimizer.h
    ad/optimizers/sgd.h
)

target_include_directories(ad PUBLIC
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})

