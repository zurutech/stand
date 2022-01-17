cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(
  solvers
  VERSION 0.1
  LANGUAGES C CXX CUDA)

# Create compile_commands.json in build dir while compiling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Policy about option behavior. This is needed as we are setting a variable
# created by option() inside SPDLOG. Using the new behavior the warning is not
# raised and the option() command does not do anything in case the variable is
# already present. Basically in this way the already existing variable is
# considered. Notice: we cannot do simply cmake_policy(SET CMP0077 NEW) as
# spdlog is inside another cmake project. In this way we are setting the default
# value for this policy for the current project and all other projects
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# Clang tidy setup: add option for enabling clang tidy checks and search for the
# exe
option(ENABLE_CLANG_TIDY
       "Output clang-tidy suggestions during the building process" OFF)
if(ENABLE_CLANG_TIDY)
  find_program(
    CLANG_TIDY_EXE
    NAMES "clang-tidy"
    DOC "Path to clang-tidy executable")

  if(NOT CLANG_TIDY_EXE)
    message(STATUS "clang-tidy not found.")
  else()
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")

    # Copy .clang-tidy config file
    file(COPY ${PROJECT_SOURCE_DIR}/.clang-tidy
         DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

    option(CLANG_TIDY_FIX "Perform fixes for Clang-Tidy" OFF)
    if(CLANG_TIDY_FIX)
      # uses .clang-tidy file in project root
      set(DO_CLANG_TIDY
          "${CLANG_TIDY_EXE}" "--config-file=${PROJECT_SOURCE_DIR}/.clang-tidy"
          "-fix")
    else()
      set(DO_CLANG_TIDY "${CLANG_TIDY_EXE}")
    endif()
  endif()
endif()

find_package(OpenMP REQUIRED)

if(OpenMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

set(CMAKE_LIBRARY_PATH "${PROJECT_SOURCE_DIR}/libs")
add_subdirectory("${CMAKE_LIBRARY_PATH}/cnpy")

add_definitions(-DDATASETS_DIR="${PROJECT_SOURCE_DIR}/../datasets")

# Build project
add_subdirectory(src)

option(BUILD_TESTS "Build the tests" OFF)
if(BUILD_TESTS)
  message(STATUS "Building tests")
  enable_testing()
  add_subdirectory(test)
endif()

option(BUILD_BENCHMARK "Build the benchmark" OFF)
if(BUILD_BENCHMARK)
  message(STATUS "Building benchmark")
  add_subdirectory(benchmark)
endif()

# copy compile commands from build dir to project dir once compiled
add_custom_target(
  do_always ALL
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
    ${PROJECT_SOURCE_DIR}/compile_commands.json)
