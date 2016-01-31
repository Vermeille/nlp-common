cmake_minimum_required(VERSION 2.8)

project(AD-MAIN CXX)


add_library(ad
    ad/ad.h
    ad/graph.h
    ad/graph.cpp
    ad/operators.h
    ad/operators.cpp
    ad/optimizer.h
    ad/optimizers/sgd.h
)

install(FILES ${libad} DESTINATION include/ad)
install(TARGETS ad
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

set_target_properties(ad PROPERTIES LINKER_LANGUAGE CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -g3 -Wall -Wextra -Wno-deprecated-declarations")

add_executable(main main.cpp)
FIND_PACKAGE(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})
target_include_directories(main PUBLIC .)
target_link_libraries(main LINK_PUBLIC ad)

