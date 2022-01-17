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

#ifndef SOLVERS_PETSCSOLVER_HPP
#define SOLVERS_PETSCSOLVER_HPP

#include <Eigen/Dense>

#include <petscksp.h>

#include <solvers/Solver.hpp>
#include <solvers/SparseSystem.hpp>

namespace solvers {

enum class PetscMethod {
    CG,
    FlexibleCG,
    GMRES,
    FlexibleGMRES,
    QMR,
    CR,
    MinRes,
    SymmLQ,
    Cholesky,
    LU,
    QR
};

enum class PetscPreconditioner { None, Jacobi, SOR, Eisenstat, ILU, ICC };

/**
 * \brief PetscSolver Solver based on PETSc library.
 */
class PetscSolver : public Solver {
private:
    /**
     * \brief _ksp Resolution algorithm.
     */
    KSP _ksp{};

public:
    explicit PetscSolver(
        PetscMethod method = PetscMethod::CG,
        PetscPreconditioner preconditioner = PetscPreconditioner::None);

    Eigen::VectorXd solve(const SparseSystem& system,
                          double& duration) const override;
};

}    // namespace solvers

#endif    // SOLVERS_PETSCSOLVER_HPP