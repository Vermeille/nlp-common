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

target_link_libraries(nlp-common LINK_PUBLIC glog ad)
target_include_directories(nlp-common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

