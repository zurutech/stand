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

#include <Eigen/CholmodSupport>
#include <Eigen/Dense>
#include <Eigen/KLUSupport>
#include <Eigen/Sparse>
#include <Eigen/UmfPackSupport>

#include <solvers/SparseSystem.hpp>
#include <solvers/SuiteSparseSolver.hpp>

namespace solvers {

Eigen::VectorXd SuiteSparseSolver::_cholmod_solve(const SparseSystem& system,
                                                  double& duration) const
{
    auto [A, b] = system.toEigenUpperCSR();
    Eigen::VectorXd result;
    std::chrono::time_point start = std::chrono::high_resolution_clock::now();
    switch (_method) {
        case SuiteSparseMethod::SimplicialLLT: {
            Eigen::CholmodSimplicialLLT<
                Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::Upper>
                simplicial_llt;
            simplicial_llt.compute(A);
            result = simplicial_llt.solve(b);
            break;
        }
        case SuiteSparseMethod::SimplicialLDLT: {
            Eigen::CholmodSimplicialLDLT<
                Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::Upper>
                simplicial_ldlt;
            simplicial_ldlt.compute(A);
            result = simplicial_ldlt.solve(b);
            break;
        }
        case SuiteSparseMethod::SupernodalLLT: {
            Eigen::CholmodSupernodalLLT<
                Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::Upper>
                supernodal_llt;
            supernodal_llt.compute(A);
            result = supernodal_llt.solve(b);
            break;
        }
        default:
            throw std::logic_error("Invalid solving method");
            break;
    }
    std::chrono::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_difference = end - start;
    duration = time_difference.count();
    return result;
}

Eigen::VectorXd SuiteSparseSolver::_cholmod_gpu_solve(
    const SparseSystem& system,
    double& duration) const
{
    auto [A, b] = system.toEigenCSR();
    Eigen::SparseMatrix<double, Eigen::RowMajor, SuiteSparse_long> A_long(A);
    cholmod_sparse cholmod_A = viewAsCholmod(A_long);
    cholmod_A.stype = 1;
    cholmod_dense cholmod_b = viewAsCholmod(b);
    cholmod_dense* cholmod_x;
    auto* c = new cholmod_common();
    cholmod_factor* L;
    cholmod_l_start(c);
    c->useGPU = 1;
    switch (_method) {
        case SuiteSparseMethod::SimplicialLLT: {
            c->final_asis = 0;
            c->supernodal = CHOLMOD_SIMPLICIAL;
            c->final_ll = 1;
            break;
        }
        case SuiteSparseMethod::SimplicialLDLT: {
            c->final_asis = 1;
            c->supernodal = CHOLMOD_SIMPLICIAL;
            break;
        }
        case SuiteSparseMethod::SupernodalLLT: {
            c->final_asis = 1;
            c->supernodal = CHOLMOD_SUPERNODAL;
            break;
        }
        default:
            throw std::logic_error("Invalid solving method");
            break;
    }
    std::chrono::time_point start = std::chrono::high_resolution_clock::now();
    L = cholmod_l_analyze(&cholmod_A, c);
    cholmod_l_factorize(&cholmod_A, L, c);
    cholmod_x = cholmod_l_solve(CHOLMOD_A, L, &cholmod_b, c);
    std::chrono::time_point end = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd result = Eigen::Map<Eigen::VectorXd>(
        static_cast<double*>(cholmod_x->x), static_cast<int64_t>(system.dim()));
    cholmod_l_free_factor(&L, c);
    cholmod_l_free_dense(&cholmod_x, c);
    cholmod_l_finish(c);
    std::chrono::duration<double> time_difference = end - start;
    duration = time_difference.count();
    return result;
}

Eigen::VectorXd SuiteSparseSolver::_lu_solve(const SparseSystem& system,
                                             double& duration) const
{
    auto [A, b] = system.toEigenCSC();
    Eigen::VectorXd result;

    std::chrono::time_point start = std::chrono::high_resolution_clock::now();
    if (_method == SuiteSparseMethod::LU) {
        Eigen::UmfPackLU<Eigen::SparseMatrix<double, Eigen::ColMajor>>
            umfpacklu;
        umfpacklu.compute(A);
        result = umfpacklu.solve(b);
    }
    else {
        Eigen::KLU<Eigen::SparseMatrix<double, Eigen::ColMajor>> klu;
        klu.compute(A);
        result = klu.solve(b);
    }
    std::chrono::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_difference = end - start;
    duration = time_difference.count();
    return result;
}

Eigen::VectorXd SuiteSparseSolver::solve(const SparseSystem& system,
                                         double& duration) const
{
    Eigen::VectorXd result;

    switch (_method) {
        case SuiteSparseMethod::SimplicialLLT:
        case SuiteSparseMethod::SimplicialLDLT:
        case SuiteSparseMethod::SupernodalLLT:
            if (_gpu) {
                result = _cholmod_gpu_solve(system, duration);
            }
            else {
                result = _cholmod_solve(system, duration);
            }
            break;
        case SuiteSparseMethod::LU:
        case SuiteSparseMethod::KLU:
            result = _lu_solve(system, duration);
            break;
        default:
            throw std::logic_error("Invalid solving method");
            break;
    }
    return result;
}

}    // namespace solvers