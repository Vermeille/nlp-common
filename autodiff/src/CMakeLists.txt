cmake_minimum_required(VERSION 2.8)

project(libad CXX)

find_package(Eigen3)

add_library(ad
    ad/nn.cpp
    ad/graph.cpp
    ad/helpers.cpp
    ad/operators.cpp
    ad/hashtable.cpp
    ad/initializers.cpp
    ad/nn/fullyconn.cpp
    ad/nn/gru.cpp
    ad/nn/rnn.cpp
    ad/nn/mrnn.cpp
    ad/nn/lstm.cpp
    ad/nn/highway-layer.cpp
    ad/optimizers/adam.cpp
    ad/optimizers/adagrad.cpp
    ad/optimizers/adadelta.cpp
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
)

target_include_directories(ad PUBLIC
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})

