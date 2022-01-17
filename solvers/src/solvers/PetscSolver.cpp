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

#include <cstring>
#include <stdexcept>

#include <Eigen/Dense>

#include <petscksp.h>

#include <solvers/PetscSolver.hpp>
#include <solvers/SparseSystem.hpp>

namespace solvers {

PetscSolver::PetscSolver(const PetscMethod method,
                         const PetscPreconditioner preconditioner)
{
    KSPCreate(PETSC_COMM_SELF, &_ksp);
    KSPType ksp_type;
    PCType pc_type = "";
    switch (method) {
        case PetscMethod::CG:
            ksp_type = KSPCG;
            break;
        case PetscMethod::FlexibleCG:
            ksp_type = KSPFCG;
            break;
        case PetscMethod::GMRES:
            ksp_type = KSPGMRES;
            break;
        case PetscMethod::FlexibleGMRES:
            ksp_type = KSPFGMRES;
            break;
        case PetscMethod::QMR:
            ksp_type = KSPTCQMR;
            break;
        case PetscMethod::CR:
            ksp_type = KSPCR;
            break;
        case PetscMethod::MinRes:
            ksp_type = KSPMINRES;
            break;
        case PetscMethod::SymmLQ:
            ksp_type = KSPSYMMLQ;
            break;
        case PetscMethod::Cholesky:
            ksp_type = KSPPREONLY;
            pc_type = PCCHOLESKY;
            break;
        case PetscMethod::LU:
            ksp_type = KSPPREONLY;
            pc_type = PCLU;
            break;
        case PetscMethod::QR:
            ksp_type = KSPPREONLY;
            pc_type = PCQR;
            break;
        default:
            throw std::logic_error("Invalid solving method");
            break;
    }
    KSPSetType(_ksp, ksp_type);
    PC pc;
    KSPGetPC(_ksp, &pc);
    if (strlen(pc_type) != 0) {
        if (preconditioner != PetscPreconditioner::None) {
            throw std::logic_error(
                "The preconditioner cannot be used with direct methods");
        }
    }
    else {
        switch (preconditioner) {
            case PetscPreconditioner::None:
                pc_type = PCNONE;
                break;
            case PetscPreconditioner::Jacobi:
                pc_type = PCJACOBI;
                break;
            case PetscPreconditioner::SOR:
                pc_type = PCSOR;
                break;
            case PetscPreconditioner::Eisenstat:
                pc_type = PCEISENSTAT;
                break;
            case PetscPreconditioner::ILU:
                pc_type = PCILU;
                break;
            case PetscPreconditioner::ICC:
                pc_type = PCICC;
                break;
            default:
                throw std::logic_error("Invalid preconditioner");
                break;
        }
    }
    PCSetType(pc, pc_type);
    KSPSetFromOptions(_ksp);
}

Eigen::VectorXd PetscSolver::solve(const SparseSystem& system,
                                   double& duration) const
{
    auto [A, b] = system.toPetscCSR();
    KSPReset(_ksp);
    KSPSetOperators(_ksp, A, A);
    double start;
    double end;
    PetscTime(&start);
    KSPSolve(_ksp, b, b);
    PetscTime(&end);
    double* result_data;
    VecGetArray(b, &result_data);
    Eigen::VectorXd result = Eigen::Map<Eigen::VectorXd>(
        result_data, static_cast<int64_t>(system.dim()));
    duration = end - start;
    return result;
}

}    // namespace solvers