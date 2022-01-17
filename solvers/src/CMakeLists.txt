cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(SOURCES
    solvers/CuSparseSolver.cpp
    solvers/PetscSolver.cpp
    solvers/SparseSystem.cpp
    solvers/SuiteSparseSolver.cpp
    solvers/SuperLUSolver.cpp
    solvers/ViennaCLSolver.cu)

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(fmt REQUIRED)
find_package(OpenCL REQUIRED)

# PETSc
find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(PETSC PETSc)
endif()

find_library(SUPERLU_LIB superlu_dist REQUIRED)
find_library(SUITESPARSE_UMFPACK_LIB umfpack REQUIRED)
find_library(SUITESPARSE_CHOLMOD_LIB cholmod REQUIRED)
find_library(SUITESPARSE_KLU_LIB klu REQUIRED)

add_library(solvers "${SOURCES}")

target_include_directories(solvers PUBLIC "${PROJECT_SOURCE_DIR}/include" "${CUDA_INCLUDE_DIRS}" "${PETSC_INCLUDE_DIRS}")
# libs folder considered as system header
target_include_directories(solvers SYSTEM PUBLIC "${CMAKE_LIBRARY_PATH}")

target_link_libraries(
  solvers
  PUBLIC Eigen3::Eigen
         fmt::fmt-header-only
         OpenMP::OpenMP_CXX
         MPI::MPI_CXX
         cnpy
         "${CUDA_cusparse_LIBRARY}"
         "${CUDA_cusolver_LIBRARY}"
         "${SUITESPARSE_UMFPACK_LIB}"
         "${SUITESPARSE_CHOLMOD_LIB}"
         "${SUITESPARSE_KLU_LIB}"
         "${SUPERLU_LIB}"
         "${PETSC_LINK_LIBRARIES}")

target_compile_features(solvers PRIVATE cxx_std_17)
set_target_properties(solvers PROPERTIES CXX_EXTENSIONS OFF)
