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

#ifndef SOLVERS_CUSPARSESOLVER_HPP
#define SOLVERS_CUSPARSESOLVER_HPP

#include <Eigen/Dense>

#include <solvers/Solver.hpp>
#include <solvers/SparseSystem.hpp>

namespace solvers {

enum class CuSparseMethod { QR, Cholesky };
enum class CuSparseReorder { None, SymRCM, SymAMD, METIS };

/**
 * \brief CuSparseSolver Direct solver based on cuSPARSE library.
 */
class CuSparseSolver : public Solver {
private:
    /**
     * \brief _method Resolution algorithm.
     */
    CuSparseMethod _method;

    /**
     * \brief _reorder Algorithm of node reordering used to reduce fill-in.
     */
    CuSparseReorder _reorder;

public:
    explicit CuSparseSolver(
        const CuSparseMethod method = CuSparseMethod::Cholesky,
        const CuSparseReorder reorder = CuSparseReorder::METIS)
        : _method(method), _reorder(reorder){};

    Eigen::VectorXd solve(const SparseSystem& system,
                          double& duration) const override;
};

}    // namespace solvers

#endif    // SOLVERS_CUSPARSESOLVER_HPP