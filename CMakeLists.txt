cmake_minimum_required(VERSION 2.8)

project(NLP-COMMON CXX)

add_library(nlp-common
    document.h
    rule.h
    rule.cpp
    rules-matcher.h
    rules-matcher.cpp
    tokenizer.h
    tokenizer.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g3")

target_link_libraries(nlp-common LINK_PUBLIC glog)

install(TARGETS nlp-common
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

file(GLOB headers *.h)
install(FILES ${headers} DESTINATION include/nlp-common)

