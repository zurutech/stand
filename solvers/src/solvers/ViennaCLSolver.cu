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

#ifndef VIENNACL_WITH_CUDA
#define VIENNACL_WITH_CUDA
#endif

#ifndef VIENNACL_HAVE_EIGEN
#define VIENNACL_HAVE_EIGEN
#endif

#include <chrono>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <viennacl/compressed_matrix.hpp>
#include <viennacl/linalg/cg.hpp>
#include <viennacl/linalg/gmres.hpp>
#include <viennacl/linalg/ichol.hpp>
#include <viennacl/linalg/ilu.hpp>
#include <viennacl/linalg/jacobi_precond.hpp>
#include <viennacl/vector.hpp>

#include <solvers/SparseSystem.hpp>
#include <solvers/ViennaCLSolver.hpp>

namespace solvers {

ViennaCLSolver::ViennaCLSolver(const ViennaCLMethod method,
                               const ViennaCLPreconditioner preconditioner,
                               const double tolerance,
                               const int iterations)
    : _method(method), _preconditioner(preconditioner)
{
    _cg_config = viennacl::linalg::cg_tag(tolerance, iterations);
    _gmres_config =
        viennacl::linalg::gmres_tag(tolerance, iterations, iterations / 10);
}

Eigen::VectorXd ViennaCLSolver::solve(const SparseSystem& system,
                                      double& duration) const
{
    auto [A, b] = system.toEigenCSR();
    Eigen::SparseMatrix<float, Eigen::RowMajor> A_float = A.cast<float>();
    Eigen::VectorXf b_float = b.cast<float>();
    const size_t dim = b.size();
    viennacl::compressed_matrix<float> vcl_A(dim, dim);
    viennacl::vector<float> vcl_b(dim);
    viennacl::copy(A_float, vcl_A);
    viennacl::copy(b_float, vcl_b);
    viennacl::vector<float> viennacl_result;

    switch (_method) {
        case ViennaCLMethod::CG:
            break;
        case ViennaCLMethod::GMRES:
            break;
        default:
            throw std::logic_error("Invalid solving method");
            break;
    }

    std::chrono::time_point start = std::chrono::high_resolution_clock::now();
    switch (_preconditioner) {
        case ViennaCLPreconditioner::None:
            if (_method == ViennaCLMethod::CG) {
                viennacl_result =
                    viennacl::linalg::solve(vcl_A, vcl_b, _cg_config);
            }
            else {
                viennacl_result =
                    viennacl::linalg::solve(vcl_A, vcl_b, _gmres_config);
            }
            break;
        case ViennaCLPreconditioner::ChowPatel: {
            viennacl::linalg::chow_patel_icc_precond<
                viennacl::compressed_matrix<float>>
                chow_patel(vcl_A, _chow_patel_config);
            if (_method == ViennaCLMethod::CG) {
                viennacl_result = viennacl::linalg::solve(
                    vcl_A, vcl_b, _cg_config, chow_patel);
            }
            else {
                viennacl_result = viennacl::linalg::solve(
                    vcl_A, vcl_b, _gmres_config, chow_patel);
            }
            break;
        }
        case ViennaCLPreconditioner::ILU0: {
            viennacl::linalg::ilu0_precond<viennacl::compressed_matrix<float>>
                ilu0(vcl_A, _ilu0_config);
            if (_method == ViennaCLMethod::CG) {
                viennacl_result =
                    viennacl::linalg::solve(vcl_A, vcl_b, _cg_config, ilu0);
            }
            else {
                viennacl_result =
                    viennacl::linalg::solve(vcl_A, vcl_b, _gmres_config, ilu0);
            }
            break;
        }
        case ViennaCLPreconditioner::IChol0: {
            viennacl::linalg::ichol0_precond<viennacl::compressed_matrix<float>>
                ichol0(vcl_A, _ichol0_config);
            if (_method == ViennaCLMethod::CG) {
                viennacl_result =
                    viennacl::linalg::solve(vcl_A, vcl_b, _cg_config, ichol0);
            }
            else {
                viennacl_result = viennacl::linalg::solve(
                    vcl_A, vcl_b, _gmres_config, ichol0);
            }
            break;
        }
        case ViennaCLPreconditioner::BlockILU0: {
            viennacl::linalg::block_ilu_precond<
                viennacl::compressed_matrix<float>, viennacl::linalg::ilu0_tag>
                block_ilu0(vcl_A, _ilu0_config);
            if (_method == ViennaCLMethod::CG) {
                viennacl_result = viennacl::linalg::solve(
                    vcl_A, vcl_b, _cg_config, block_ilu0);
            }
            else {
                viennacl_result = viennacl::linalg::solve(
                    vcl_A, vcl_b, _gmres_config, block_ilu0);
            }
            break;
        }
        case ViennaCLPreconditioner::Jacobi: {
            viennacl::linalg::jacobi_precond<viennacl::compressed_matrix<float>>
                jacobi(vcl_A, _jacobi_config);
            if (_method == ViennaCLMethod::CG) {
                viennacl_result =
                    viennacl::linalg::solve(vcl_A, vcl_b, _cg_config, jacobi);
            }
            else {
                viennacl_result = viennacl::linalg::solve(
                    vcl_A, vcl_b, _gmres_config, jacobi);
            }
            break;
        }
        case ViennaCLPreconditioner::RowScaling: {
            viennacl::linalg::row_scaling<viennacl::compressed_matrix<float>>
                row_scaling(vcl_A, _row_scaling_config);
            if (_method == ViennaCLMethod::CG) {
                viennacl_result = viennacl::linalg::solve(
                    vcl_A, vcl_b, _cg_config, row_scaling);
            }
            else {
                viennacl_result = viennacl::linalg::solve(
                    vcl_A, vcl_b, _gmres_config, row_scaling);
            }
            break;
        }
        default:
            throw std::logic_error("Invalid preconditioner method");
            break;
    }
    std::chrono::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_difference = end - start;
    duration = time_difference.count();
    Eigen::VectorXf result_float(static_cast<int>(dim));
    viennacl::copy(viennacl_result, result_float);
    Eigen::VectorXd result = result_float.cast<double>();
    return result;
}

}    // namespace solvers