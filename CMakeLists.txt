cmake_minimum_required(VERSION 2.8)

project(NLP-COMMON CXX)

add_library(nlp-common
    document.h
    rule.h
    rule.cpp
    rules-matcher.h
    rules-matcher.cpp
    tokenizer.h
    tokenizer.cpp
    featurizer.h
    featurizer.cpp
    dict.h
    dict.cpp
    bow.h
    bow.cpp
    sequence-tagger.cpp
    sequence-tagger.h
    )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g3 -Wall -Wextra")

target_link_libraries(nlp-common LINK_PUBLIC glog)

add_executable(tests tests.cpp)
target_link_libraries(tests LINK_PUBLIC nlp-common glog)

install(TARGETS nlp-common
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

file(GLOB headers *.h)
install(FILES ${headers} DESTINATION include/nlp-common)

