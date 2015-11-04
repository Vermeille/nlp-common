add_library(nlp-common
    document.h)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

install(TARGETS httpi
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

file(GLOB headers *.h)
install(FILES ${headers} DESTINATION include/nlp-common)

