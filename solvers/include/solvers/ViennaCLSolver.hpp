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

#ifndef SOLVERS_VIENNACLSOLVER_HPP
#define SOLVERS_VIENNACLSOLVER_HPP

#include <Eigen/Dense>

#include <viennacl/linalg/cg.hpp>
#include <viennacl/linalg/gmres.hpp>
#include <viennacl/linalg/ichol.hpp>
#include <viennacl/linalg/ilu.hpp>
#include <viennacl/linalg/jacobi_precond.hpp>

#include <solvers/Solver.hpp>
#include <solvers/SparseSystem.hpp>

namespace solvers {

enum class ViennaCLMethod { CG, GMRES };
enum class ViennaCLPreconditioner {
    None,
    ChowPatel,
    ILU0,
    IChol0,
    BlockILU0,
    Jacobi,
    RowScaling
};

/**
 * \brief ViennaCLSolver Iterative solver based on ViennaCL library.
 */
class ViennaCLSolver : public Solver {
private:
    /**
     * \brief _method Resolution algorithm.
     */
    ViennaCLMethod _method;

    /**
     * \brief _preconditioner Preconditioning algorithm.
     */
    ViennaCLPreconditioner _preconditioner;

    /**
     * \brief _cg_config Parameters for conjugate gradient algorithm.
     */
    viennacl::linalg::cg_tag _cg_config;

    /**
     * \brief _gmres_config Parameters for GMRES algorithm.
     */
    viennacl::linalg::gmres_tag _gmres_config;

    /**
     * \brief _chow_patel_config Parameters for Chow-Patel preconditioner.
     */
    viennacl::linalg::chow_patel_tag _chow_patel_config;

    /**
     * \brief _ilu0_config Parameters for ILU(0) preconditioner.
     */
    viennacl::linalg::ilu0_tag _ilu0_config;

    /**
     * \brief _ichol0_config Parameters for ICC preconditioner.
     */
    viennacl::linalg::ichol0_tag _ichol0_config;

    /**
     * \brief _jacobi_config Parameters for Jacobi preconditioner.
     */
    viennacl::linalg::jacobi_tag _jacobi_config;

    /**
     * \brief _row_scaling_config Parameters for row scaling preconditioner.
     */
    viennacl::linalg::row_scaling_tag _row_scaling_config;

public:
    explicit ViennaCLSolver(
        ViennaCLMethod method = ViennaCLMethod::CG,
        ViennaCLPreconditioner preconditioner = ViennaCLPreconditioner::None,
        double tolerance = 1e-8,
        int iterations = 300);

    Eigen::VectorXd solve(const SparseSystem& system,
                          double& duration) const override;
};

}    // namespace solvers

#endif    // SOLVERS_VIENNACLSOLVER_HPP