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

#include <chrono>
#include <stdexcept>

#include <Eigen/Dense>

#include <superlu_dist/superlu_ddefs.h>

#include <solvers/SparseSystem.hpp>
#include <solvers/SuperLUSolver.hpp>

namespace solvers {

SuperLUSolver::SuperLUSolver(const SuperLUReorder reorder)
{
    /* Set the default input options:
        options.Fact = DOFACT;
        options.Equil = YES;
        options.ColPerm = METIS_AT_PLUS_A;
        options.RowPerm = LargeDiag_MC64;
        options.ReplaceTinyPivot = YES;
        options.Trans = NOTRANS;
        options.IterRefine = DOUBLE;
        options.SolveInitialized = NO;
        options.RefineInitialized = NO;
        options.PrintStat = YES;
     */
    set_default_options_dist(&_options);
    _options.PrintStat = yes_no_t::NO;
    _options.ColPerm = static_cast<colperm_t>(reorder);
    _options.IterRefine = IterRefine_t::NOREFINE;
    _options.SymPattern = yes_no_t::YES;
}

Eigen::VectorXd SuperLUSolver::solve(const SparseSystem& system,
                                     double& duration) const
{
    auto [ccol_index, row_index, values, b] = system.toStdCSR();
    // Transfer ownership from vectors to pointers
    auto* ccol_index_ptr = new int[ccol_index.size()];
    auto* row_index_ptr = new int[row_index.size()];
    auto* values_ptr = new double[values.size()];
    std::copy(ccol_index.begin(), ccol_index.end(), ccol_index_ptr);
    std::copy(row_index.begin(), row_index.end(), row_index_ptr);
    std::copy(values.begin(), values.end(), values_ptr);

    const auto n = static_cast<int>(system.dim());
    const auto nnz = static_cast<int>(system.nnz());

    // Choose a GPU device
    int rank;
    int devs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cudaGetDeviceCount(&devs);
    cudaSetDevice(rank % devs);

    gridinfo_t grid;
    constexpr int nprow = 1;
    constexpr int npcol = 1;
    superlu_gridinit(MPI_COMM_WORLD, nprow, npcol, &grid);

    // Create A in CSC format
    SuperMatrix A;
    dCreate_CompCol_Matrix_dist(&A, n, n, nnz, values_ptr, row_index_ptr,
                                ccol_index_ptr, Stype_t::SLU_NC, Dtype_t::SLU_D,
                                Mtype_t::SLU_GE);

    // Initialize variables
    dScalePermstruct_t scale_perm;
    dScalePermstructInit(n, n, &scale_perm);
    dLUstruct_t lu;
    dLUstructInit(n, &lu);
    SuperLUStat_t stat;
    PStatInit(&stat);

    // Call the linear equation solver
    double error;
    int info;
    superlu_dist_options_t options(_options);
    Eigen::VectorXd result = Eigen::Map<Eigen::VectorXd>(b.data(), n);
    pdgssvx_ABglobal(&options, &A, &scale_perm, result.data(), n, 1, &grid, &lu,
                     &error, &stat, &info);

    // Measure execution time in seconds
    duration = 0.;
    for (size_t phase = 0; phase < PhaseType::NPHASES; ++phase) {
        duration += stat.utime[phase];
    }

    // Destroy MPI objects
    PStatFree(&stat);
    Destroy_CompCol_Matrix_dist(&A);
    dDestroy_LU(n, &grid, &lu);
    dScalePermstructFree(&scale_perm);
    dLUstructFree(&lu);
    superlu_gridexit(&grid);
    cudaDeviceReset();
    return result;
}

}    // namespace solvers