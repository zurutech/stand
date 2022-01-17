/* Copyright 2022 Zuru Tech HK Limited.
 *
 * Licensed under the Apache License, Version 2.0(the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

#include <Eigen/Dense>

#include <petscsys.h>

#include <solvers/CuSparseSolver.hpp>
#include <solvers/PetscSolver.hpp>
#include <solvers/Solver.hpp>
#include <solvers/SparseSystem.hpp>
#include <solvers/SuiteSparseSolver.hpp>
#include <solvers/SuperLUSolver.hpp>
#include <solvers/ViennaCLSolver.hpp>

namespace fs = std::filesystem;

std::vector<fs::path> getDatasetPaths(const fs::path& dataset_dir)
{
    std::vector<fs::path> paths;
    for (const fs::path& entry : fs::directory_iterator(dataset_dir)) {
        if (entry.extension() == ".npz") {
            paths.emplace_back(entry);
        }
    }
    return paths;
}

bool endsWith(std::string const& value, std::string const& ending)
{
    if (ending.size() > value.size()) {
        return false;
    }
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

void benchmarkSolver(benchmark::State& state,
                     const solvers::Solver& solver,
                     const std::vector<fs::path>& dataset)
{
    double error_sum = 0.;
    for (auto _ : state) {
        int64_t it = state.items_processed();
        solvers::SparseSystem system =
            solvers::SparseSystem::loadNpz(dataset[it]);
        double duration;
        Eigen::VectorXd result = solver.solve(system, duration);
        state.SetIterationTime(duration);
        Eigen::VectorXd solution = system.getEigenSolution();
        double error = (result - solution).norm() / solution.norm();
        error_sum += error;
        state.SetItemsProcessed(it + 1);
    }
    state.counters["Average Relative Error"] =
        benchmark::Counter(error_sum, benchmark::Counter::kAvgIterations);
}

std::vector<benchmark::internal::Benchmark*> registerCuSparseBenchmarks(
    const std::vector<fs::path>& dataset)
{
    std::map<std::string,
             std::pair<solvers::CuSparseMethod, solvers::CuSparseReorder>>
        configs = {
            {"GPUCuSparse/Qr",
             {solvers::CuSparseMethod::QR, solvers::CuSparseReorder::None}},
            {"GPUCuSparse/QrSymRCM",
             {solvers::CuSparseMethod::QR, solvers::CuSparseReorder::SymRCM}},
            {"GPUCuSparse/QrSymAMD",
             {solvers::CuSparseMethod::QR, solvers::CuSparseReorder::SymAMD}},
            {"GPUCuSparse/QrMetis",
             {solvers::CuSparseMethod::QR, solvers::CuSparseReorder::METIS}},
            {"GPUCuSparse/Cholesky",
             {solvers::CuSparseMethod::Cholesky,
              solvers::CuSparseReorder::None}},
            {"GPUCuSparse/CholeskySymRCM",
             {solvers::CuSparseMethod::Cholesky,
              solvers::CuSparseReorder::SymRCM}},
            {"GPUCuSparse/CholeskySymAMD",
             {solvers::CuSparseMethod::Cholesky,
              solvers::CuSparseReorder::SymAMD}},
            {"GPUCuSparse/CholeskyMetis",
             {solvers::CuSparseMethod::Cholesky,
              solvers::CuSparseReorder::METIS}}};
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    benchmarks.reserve(configs.size());
    for (const auto& [name, config] : configs) {
        solvers::CuSparseSolver solver(config.first, config.second);
        benchmarks.emplace_back(benchmark::RegisterBenchmark(
            name.c_str(), benchmarkSolver, solver, dataset));
    }
    return benchmarks;
}

std::vector<benchmark::internal::Benchmark*> registerPetscBenchmarks(
    const std::vector<fs::path>& dataset)
{
    std::map<std::string,
             std::pair<solvers::PetscMethod, solvers::PetscPreconditioner>>
        configs = {
            {"Petsc/Cg",
             {solvers::PetscMethod::CG, solvers::PetscPreconditioner::None}},
            {"Petsc/CgJacobi",
             {solvers::PetscMethod::CG, solvers::PetscPreconditioner::Jacobi}},
            {"Petsc/CgSor",
             {solvers::PetscMethod::CG, solvers::PetscPreconditioner::SOR}},
            {"Petsc/CgEisenstat",
             {solvers::PetscMethod::CG,
              solvers::PetscPreconditioner::Eisenstat}},
            {"Petsc/CgIlu",
             {solvers::PetscMethod::CG, solvers::PetscPreconditioner::ILU}},
            {"Petsc/CgIcc",
             {solvers::PetscMethod::CG, solvers::PetscPreconditioner::ICC}},
            {"Petsc/Gmres",
             {solvers::PetscMethod::GMRES, solvers::PetscPreconditioner::None}},
            {"Petsc/GmresJacobi",
             {solvers::PetscMethod::GMRES,
              solvers::PetscPreconditioner::Jacobi}},
            {"Petsc/GmresSor",
             {solvers::PetscMethod::GMRES, solvers::PetscPreconditioner::SOR}},
            {"Petsc/GmresEisenstat",
             {solvers::PetscMethod::GMRES,
              solvers::PetscPreconditioner::Eisenstat}},
            {"Petsc/GmresIlu",
             {solvers::PetscMethod::GMRES, solvers::PetscPreconditioner::ILU}},
            {"Petsc/GmresIcc",
             {solvers::PetscMethod::GMRES, solvers::PetscPreconditioner::ICC}},
            {"Petsc/FlexibleGmres",
             {solvers::PetscMethod::FlexibleGMRES,
              solvers::PetscPreconditioner::None}},
            {"Petsc/FlexibleGmresJacobi",
             {solvers::PetscMethod::FlexibleGMRES,
              solvers::PetscPreconditioner::Jacobi}},
            {"Petsc/FlexibleGmresSor",
             {solvers::PetscMethod::FlexibleGMRES,
              solvers::PetscPreconditioner::SOR}},
            {"Petsc/FlexibleGmresEisenstat",
             {solvers::PetscMethod::FlexibleGMRES,
              solvers::PetscPreconditioner::Eisenstat}},
            {"Petsc/FlexibleGmresIlu",
             {solvers::PetscMethod::FlexibleGMRES,
              solvers::PetscPreconditioner::ILU}},
            {"Petsc/FlexibleGmresIcc",
             {solvers::PetscMethod::FlexibleGMRES,
              solvers::PetscPreconditioner::ICC}},
            {"Petsc/Qmr",
             {solvers::PetscMethod::QMR, solvers::PetscPreconditioner::None}},
            {"Petsc/QmrJacobi",
             {solvers::PetscMethod::QMR, solvers::PetscPreconditioner::Jacobi}},
            {"Petsc/QmrSor",
             {solvers::PetscMethod::QMR, solvers::PetscPreconditioner::SOR}},
            {"Petsc/QmrEisenstat",
             {solvers::PetscMethod::QMR,
              solvers::PetscPreconditioner::Eisenstat}},
            {"Petsc/QmrIlu",
             {solvers::PetscMethod::QMR, solvers::PetscPreconditioner::ILU}},
            {"Petsc/QmrIcc",
             {solvers::PetscMethod::QMR, solvers::PetscPreconditioner::ICC}},
            {"Petsc/Cr",
             {solvers::PetscMethod::CR, solvers::PetscPreconditioner::None}},
            {"Petsc/CrJacobi",
             {solvers::PetscMethod::CR, solvers::PetscPreconditioner::Jacobi}},
            {"Petsc/CrSor",
             {solvers::PetscMethod::CR, solvers::PetscPreconditioner::SOR}},
            {"Petsc/CrEisenstat",
             {solvers::PetscMethod::CR,
              solvers::PetscPreconditioner::Eisenstat}},
            {"Petsc/CrIlu",
             {solvers::PetscMethod::CR, solvers::PetscPreconditioner::ILU}},
            {"Petsc/CrIcc",
             {solvers::PetscMethod::CR, solvers::PetscPreconditioner::ICC}},
            {"Petsc/Minres",
             {solvers::PetscMethod::MinRes,
              solvers::PetscPreconditioner::None}},
            {"Petsc/MinresJacobi",
             {solvers::PetscMethod::MinRes,
              solvers::PetscPreconditioner::Jacobi}},
            {"Petsc/MinresSor",
             {solvers::PetscMethod::MinRes, solvers::PetscPreconditioner::SOR}},
            {"Petsc/MinresEisenstat",
             {solvers::PetscMethod::MinRes,
              solvers::PetscPreconditioner::Eisenstat}},
            {"Petsc/MinresIlu",
             {solvers::PetscMethod::MinRes, solvers::PetscPreconditioner::ILU}},
            {"Petsc/MinresIcc",
             {solvers::PetscMethod::MinRes, solvers::PetscPreconditioner::ICC}},
            {"Petsc/Symmlq",
             {solvers::PetscMethod::SymmLQ,
              solvers::PetscPreconditioner::None}},
            {"Petsc/SymmlqJacobi",
             {solvers::PetscMethod::SymmLQ,
              solvers::PetscPreconditioner::Jacobi}},
            {"Petsc/SymmlqSor",
             {solvers::PetscMethod::SymmLQ, solvers::PetscPreconditioner::SOR}},
            {"Petsc/SymmlqEisenstat",
             {solvers::PetscMethod::SymmLQ,
              solvers::PetscPreconditioner::Eisenstat}},
            {"Petsc/SymmlqIlu",
             {solvers::PetscMethod::SymmLQ, solvers::PetscPreconditioner::ILU}},
            {"Petsc/SymmlqIcc",
             {solvers::PetscMethod::SymmLQ, solvers::PetscPreconditioner::ICC}},
            {"Petsc/Cholesky",
             {solvers::PetscMethod::Cholesky,
              solvers::PetscPreconditioner::None}},
            {"Petsc/Lu",
             {solvers::PetscMethod::LU, solvers::PetscPreconditioner::None}},
            {"Petsc/Qr",
             {solvers::PetscMethod::QR, solvers::PetscPreconditioner::None}}};
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    benchmarks.reserve(configs.size());
    for (const auto& [name, config] : configs) {
        solvers::PetscSolver solver(config.first, config.second);
        benchmarks.emplace_back(benchmark::RegisterBenchmark(
            name.c_str(), benchmarkSolver, solver, dataset));
    }
    return benchmarks;
}

std::vector<benchmark::internal::Benchmark*> registerSuiteSparseBenchmarks(
    const std::vector<fs::path>& dataset)
{
    std::map<std::string, solvers::SuiteSparseMethod> configs = {
        {"SuiteSparse/SimplicialLLT",
         solvers::SuiteSparseMethod::SimplicialLLT},
        {"SuiteSparse/SimplicialLDLT",
         solvers::SuiteSparseMethod::SimplicialLDLT},
        {"SuiteSparse/SupernodalLLT",
         solvers::SuiteSparseMethod::SupernodalLLT},
        {"SuiteSparse/LU", solvers::SuiteSparseMethod::LU},
        {"SuiteSparse/KLU", solvers::SuiteSparseMethod::KLU}};
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    benchmarks.reserve(configs.size());
    constexpr bool gpu = true;
    for (const auto& [name, config] : configs) {
        solvers::SuiteSparseSolver solver(config);
        benchmarks.emplace_back(benchmark::RegisterBenchmark(
            name.c_str(), benchmarkSolver, solver, dataset));
        if (!endsWith(name, "LU")) {
            solvers::SuiteSparseSolver gpu_solver(config, gpu);
            benchmarks.emplace_back(benchmark::RegisterBenchmark(
                ("GPU" + name).c_str(), benchmarkSolver, solver, dataset));
        }
    }
    return benchmarks;
}

std::vector<benchmark::internal::Benchmark*> registerSuperLUBenchmarks(
    const std::vector<fs::path>& dataset)
{
    std::map<std::string, solvers::SuperLUReorder> configs = {
        {"GPUSuperLU/Lu", solvers::SuperLUReorder::None},
        {"GPUSuperLU/LuMinimumDegree", solvers::SuperLUReorder::MinimumDegree},
        {"GPUSuperLU/LuColamd", solvers::SuperLUReorder::ColAMD},
        {"GPUSuperLU/LuMetis", solvers::SuperLUReorder::METIS}};
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    benchmarks.reserve(configs.size());
    for (const auto& [name, config] : configs) {
        solvers::SuperLUSolver solver(config);
        benchmarks.emplace_back(benchmark::RegisterBenchmark(
            name.c_str(), benchmarkSolver, solver, dataset));
    }
    return benchmarks;
}

std::vector<benchmark::internal::Benchmark*> registerViennaCLBenchmarks(
    const std::vector<fs::path>& dataset,
    const double tolerance = 1e-6,
    const int iterations = 1000)
{
    std::map<std::string, std::pair<solvers::ViennaCLMethod,
                                    solvers::ViennaCLPreconditioner>>
        configs = {
            {"GPUViennaCL/Cg",
             {solvers::ViennaCLMethod::CG,
              solvers::ViennaCLPreconditioner::None}},
            {"GPUViennaCL/CgChowPatel",
             {solvers::ViennaCLMethod::CG,
              solvers::ViennaCLPreconditioner::ChowPatel}},
            {"GPUViennaCL/CgIlu0",
             {solvers::ViennaCLMethod::CG,
              solvers::ViennaCLPreconditioner::ILU0}},
            {"GPUViennaCL/CgIchol0",
             {solvers::ViennaCLMethod::CG,
              solvers::ViennaCLPreconditioner::IChol0}},
            {"GPUViennaCL/CgBlockIlu0",
             {solvers::ViennaCLMethod::CG,
              solvers::ViennaCLPreconditioner::BlockILU0}},
            {"GPUViennaCL/CgJacobi",
             {solvers::ViennaCLMethod::CG,
              solvers::ViennaCLPreconditioner::Jacobi}},
            {"GPUViennaCL/CgRowScaling",
             {solvers::ViennaCLMethod::CG,
              solvers::ViennaCLPreconditioner::RowScaling}},
            {"GPUViennaCL/Gmres",
             {solvers::ViennaCLMethod::GMRES,
              solvers::ViennaCLPreconditioner::None}},
            {"GPUViennaCL/GmresChowPatel",
             {solvers::ViennaCLMethod::GMRES,
              solvers::ViennaCLPreconditioner::ChowPatel}},
            {"GPUViennaCL/GmresIlu0",
             {solvers::ViennaCLMethod::GMRES,
              solvers::ViennaCLPreconditioner::ILU0}},
            {"GPUViennaCL/GmresIchol0",
             {solvers::ViennaCLMethod::GMRES,
              solvers::ViennaCLPreconditioner::IChol0}},
            {"GPUViennaCL/GmresBlockIlu0",
             {solvers::ViennaCLMethod::GMRES,
              solvers::ViennaCLPreconditioner::BlockILU0}},
            {"GPUViennaCL/GmresJacobi",
             {solvers::ViennaCLMethod::GMRES,
              solvers::ViennaCLPreconditioner::Jacobi}},
            {"GPUViennaCL/GmresRowScaling",
             {solvers::ViennaCLMethod::GMRES,
              solvers::ViennaCLPreconditioner::RowScaling}},
        };
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    benchmarks.reserve(configs.size());
    for (const auto& [name, config] : configs) {
        solvers::ViennaCLSolver solver(config.first, config.second, tolerance,
                                       iterations);
        benchmarks.emplace_back(benchmark::RegisterBenchmark(
            name.c_str(), benchmarkSolver, solver, dataset));
    }
    return benchmarks;
}

int main(int argc, char** argv)
{
    fs::path dataset_dir;

    const char* dataset_option = "--dataset=";
    std::vector<char*> benchmark_arguments;
    benchmark_arguments.reserve(argc);
    for (int i = 0; i < argc; ++i) {
        if (strncmp(dataset_option, argv[i], strlen(dataset_option)) == 0) {
            dataset_dir.assign(argv[i] + strlen(dataset_option));
        }
        else {
            benchmark_arguments.emplace_back(argv[i]);
        }
    }
    if (dataset_dir.empty()) {
        std::cout << "The required argument --dataset is missing, use "
                     "--dataset=<dataset_directory>"
                  << std::endl;
        return 1;
    }
    if (!fs::exists(dataset_dir)) {
        std::cout << "No such dataset directory: " << dataset_dir << std::endl;
        return 1;
    }

    PetscInitialize(nullptr, nullptr, PETSC_CONFIG, nullptr);
    std::vector<fs::path> dataset = getDatasetPaths(dataset_dir);

    std::vector<std::vector<benchmark::internal::Benchmark*>> benchmarks = {
        registerCuSparseBenchmarks(dataset), registerPetscBenchmarks(dataset),
        registerSuiteSparseBenchmarks(dataset),
        registerViennaCLBenchmarks(dataset),
        registerSuperLUBenchmarks(dataset)};
    for (const auto& lib_benchmarks : benchmarks) {
        for (const auto& benchmark : lib_benchmarks) {
            benchmark->UseManualTime()
                ->Unit(benchmark::kMillisecond)
                ->Iterations(dataset.size());
        }
    }

    argc = static_cast<int>(benchmark_arguments.size());
    argv = benchmark_arguments.data();
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    PetscFinalize();
    return 0;
}