cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

find_package(ZLIB)

add_library(cnpy STATIC cnpy.cpp)
target_include_directories(cnpy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cnpy PRIVATE ZLIB::ZLIB)
