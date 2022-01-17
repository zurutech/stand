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

#ifndef SOLVERS_SUPERLUSOLVER_HPP
#define SOLVERS_SUPERLUSOLVER_HPP

#include <Eigen/Dense>

#include <superlu_dist/superlu_ddefs.h>

#include <solvers/Solver.hpp>
#include <solvers/SparseSystem.hpp>

namespace solvers {

enum class SuperLUReorder {
    None = colperm_t::NATURAL,
    MinimumDegree = colperm_t::MMD_AT_PLUS_A,
    ColAMD = colperm_t::COLAMD,
    METIS = colperm_t::METIS_AT_PLUS_A,
};

/**
 * \brief SuperLUSolver Direct solver based on SuperLU_DIST library.
 */
class SuperLUSolver : public Solver {
private:
    /**
     * \brief _options Parameters of the resolution algorithm.
     */
    superlu_dist_options_t _options{};

public:
    explicit SuperLUSolver(SuperLUReorder reorder = SuperLUReorder::METIS);

    Eigen::VectorXd solve(const SparseSystem& system,
                          double& duration) const override;
};

}    // namespace solvers

#endif    // SOLVERS_SUPERLUSOLVER_HPP