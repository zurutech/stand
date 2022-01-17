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

#ifndef SOLVERS_SOLVER_HPP
#define SOLVERS_SOLVER_HPP

#include <Eigen/Dense>

#include <solvers/SparseSystem.hpp>

namespace solvers {

/**
 * \brief Solver Base class for linear system solvers.
 */
class Solver {
public:
    /**
     * \brief solve Solve a sparse linear system.
     * 
     * \param[in] system Sparse linear system.
     * \param[out] duration Running time of the resolution in seconds.
     * \return Solution vector of the sparse linear system in Eigen format.
     */
    virtual Eigen::VectorXd solve(const SparseSystem& system,
                                  double& duration) const = 0;
};

}    // namespace solvers

#endif    // SOLVERS_SOLVER_HPP