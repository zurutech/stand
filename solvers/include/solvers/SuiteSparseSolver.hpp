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

#ifndef SOLVERS_SUITESPARSESOLVER_HPP
#define SOLVERS_SUITESPARSESOLVER_HPP

#include <Eigen/CholmodSupport>
#include <Eigen/Dense>
#include <Eigen/KLUSupport>
#include <Eigen/UmfPackSupport>

#include <solvers/Solver.hpp>
#include <solvers/SparseSystem.hpp>

namespace solvers {

enum class SuiteSparseMethod {
    SimplicialLLT,
    SimplicialLDLT,
    SupernodalLLT,
    LU,
    KLU
};

/**
 * \brief SuiteSparseSolver Direct solver based on SuiteSparse library.
 */
class SuiteSparseSolver : public Solver {
private:
    /**
     * \brief _method Resolution algorithm.
     */
    SuiteSparseMethod _method;

    /**
     * \brief _gpu Whether to use GPU (even if GPU support is very limited)
     */
    bool _gpu;

    /**
     * \brief _cholmod_solve Solve a sparse linear system using CHOLMOD module.
     * 
     * \param[in] system Sparse linear system.
     * \param[out] duration Running time of the resolution in seconds.
     * \return Solution vector of the sparse linear system in Eigen format.
     */
    Eigen::VectorXd _cholmod_solve(const SparseSystem& system,
                                   double& duration) const;

    /**
     * \brief _cholmod_gpu_solve Solve a sparse linear system using CHOLMOD module on GPU.
     * 
     * \param[in] system Sparse linear system.
     * \param[out] duration Running time of the resolution in seconds.
     * \return Solution vector of the sparse linear system in Eigen format.
     */
    Eigen::VectorXd _cholmod_gpu_solve(const SparseSystem& system,
                                       double& duration) const;

    /**
     * \brief _lu_solve Solve a sparse linear system using LU or KLU modules.
     * 
     * \param[in] system Sparse linear system.
     * \param[out] duration Running time of the resolution in seconds.
     * \return Solution vector of the sparse linear system in Eigen format.
     */
    Eigen::VectorXd _lu_solve(const SparseSystem& system,
                              double& duration) const;

public:
    explicit SuiteSparseSolver(
        const SuiteSparseMethod method = SuiteSparseMethod::SupernodalLLT,
        const bool gpu = false)
        : _method(method), _gpu(gpu){};

    Eigen::VectorXd solve(const SparseSystem& system,
                          double& duration) const override;
};

}    // namespace solvers

#endif    // SOLVERS_SUITESPARSESOLVER_HPP