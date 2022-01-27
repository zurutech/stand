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

#ifndef SOLVERS_SPARSESYSTEM_HPP
#define SOLVERS_SPARSESYSTEM_HPP

#include <filesystem>
#include <tuple>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <petscmat.h>
#include <petscvec.h>

namespace solvers {

/**
 * \brief SparseSystem Represent a solved sparse symmetric linear system.
 */
class SparseSystem {
private:
    /**
     * \brief _upper_A Triangular upper part of the sparse symmetric matrix of the system.
     */
    Eigen::SparseMatrix<double, Eigen::RowMajor> _upper_A;

    /**
     * \brief _b Constant term of the system.
     */
    Eigen::VectorXd _b;

    /**
     * \brief _x Solution of the system.
     */
    Eigen::VectorXd _x;

public:
    SparseSystem(const Eigen::SparseMatrix<double, Eigen::RowMajor>& upper_A,
                 Eigen::VectorXd b,
                 Eigen::VectorXd x);

    /**
     * \brief loadNpz Load a sparse system from a .npz file in StAnD format.
     * 
     * \param[in] npz_path Path to the input .npz file.
     * \return Solved sparse symmetric linear system.
     */
    [[nodiscard]] static SparseSystem loadNpz(
        const std::filesystem::path& npz_path);

    /**
     * \brief saveNps Save a sparse system in a .npz file in StAnD format.
     * 
     * \param[in] npz_path Path to the output .npz file.
     */
    void saveNpz(const std::filesystem::path& npz_path) const;

    /**
     * \brief removeNullEntries Remove null entries from the sparse matrix.
     * 
     * \param[in] tolerance Maximum absolute value for a matrix coefficient to be considered null.
     */
    void removeNullEntries(double tolerance = 0.);

    /**
     * \brief dim Get the number of rows and columns of the sparse matrix.
     * 
     * \return Number of rows and columns of the sparse matrix.
     */
    [[nodiscard]] size_t dim() const;

    /**
     * \brief nnz Get the number of non-null coefficients of the sparse matrix.
     * 
     * \return Number of non-null coefficients of the sparse matrix.
     */
    [[nodiscard]] size_t nnz() const;

    /**
     * \brief getEigenSolution Get the solution vector in Eigen format.
     * 
     * \return Solution vector.
     */
    [[nodiscard]] Eigen::VectorXd getEigenSolution() const;

    /**
     * \brief getPetscSolution Get the solution vector in PETSc format.
     * 
     * \return Solution vector.
     */
    [[nodiscard]] Vec getPetscSolution() const;

    /**
     * \brief toEigenCSC Get the coefficient matrix and the constant term in Eigen format.
     *        The sparse coefficient matrix is stored in CSC format.
     * 
     * \return Pair of coefficient matrix and constant term in Eigen format.
     */
    [[nodiscard]] std::tuple<Eigen::SparseMatrix<double, Eigen::ColMajor>,
                             Eigen::VectorXd>
    toEigenCSC() const;

    /**
     * \brief toEigenCSR Get the coefficient matrix and the constant term in Eigen format.
     *        The sparse coefficient matrix is stored in CSR format.
     * 
     * \return Pair of coefficient matrix and constant term in Eigen format.
     */
    [[nodiscard]] std::tuple<Eigen::SparseMatrix<double, Eigen::RowMajor>,
                             Eigen::VectorXd>
    toEigenCSR() const;

    /**
     * \brief toEigenUpperCSR Get the upper part of coefficient matrix and
     *        the constant term in Eigen format.
     *        The upper part of the sparse coefficient matrix is stored in CSR format.
     * 
     * \return Pair of upper part of coefficient matrix and constant term in Eigen format.
     */
    [[nodiscard]] std::tuple<Eigen::SparseMatrix<double, Eigen::RowMajor>,
                             Eigen::VectorXd>
    toEigenUpperCSR() const;

    /**
     * \brief toPetscCSR Get the coefficient matrix and the constant term in PETSc format.
     *        The sparse coefficient matrix is stored in CSR format.
     * 
     * \return Pair of coefficient matrix and constant term in Eigen format.
     */
    [[nodiscard]] std::tuple<Mat, Vec> toPetscCSR() const;

    /**
     * \brief toStdCSR Get the coefficient matrix and the constant term in std::vector format.
     *        The sparse coefficient matrix is stored in CSR format and it is
     *        represented by three vectors: compressed row indices, column indices and values.
     * 
     * \return Tuple of vectors representing:
     *         - compressed row indices of the coefficient matrix;
     *         - column indices of the coefficient matrix;
     *         - values of the coefficients;
     *         - constant term.
     */
    [[nodiscard]] std::tuple<std::vector<int>,
                             std::vector<int>,
                             std::vector<double>,
                             std::vector<double>>
    toStdCSR() const;
};

}    // namespace solvers

#endif    // SOLVERS_SPARSESYSTEM_HPP