cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(solvers_tests LANGUAGES C CXX CUDA)

enable_testing()

find_package(GTest REQUIRED)

# PETSc
find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(PETSC PETSc)
endif()

# Define test source files
set(TEST_SOURCE_FILES
    main.cpp
    solvers/SparseSystemTest.cpp
    solvers/SolverTest.cpp
    solvers/CuSparseSolverTest.cpp
    solvers/PetscSolverTest.cpp
    solvers/SuiteSparseSolverTest.cpp
    solvers/SuperLUSolverTest.cpp
    solvers/ViennaCLSolverTest.cu)


# Create test target
add_executable(solvers_tests ${TEST_SOURCE_FILES})

target_include_directories(solvers_tests PRIVATE "${PETSC_INCLUDE_DIRS}")

target_link_libraries(
  solvers_tests
  PRIVATE solvers GTest::gtest)

# Define compile features for this target: c++17 for tests
target_compile_features(solvers_tests PRIVATE cxx_std_17)

# Disable CXX extensions (it's better to do not rely upon compiler extensions)
set_target_properties(solvers_tests PROPERTIES CXX_EXTENSIONS OFF)

gtest_discover_tests(solvers_tests WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
