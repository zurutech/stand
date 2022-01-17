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

#include <cmath>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <cnpy/cnpy.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <fmt/core.h>

#include <petscmat.h>
#include <petscvec.h>

#include <solvers/SparseSystem.hpp>

namespace solvers {

SparseSystem::SparseSystem(
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& upper_A,
    Eigen::VectorXd b,
    Eigen::VectorXd x)
    : _upper_A(upper_A), _b(std::move(b)), _x(std::move(x))
{
    _upper_A.makeCompressed();
}

SparseSystem SparseSystem::loadNpz(const std::filesystem::path& npz_path)
{
    cnpy::npz_t system = cnpy::npz_load(npz_path.string());
    cnpy::NpyArray A_indices_array = system["A_indices"];
    cnpy::NpyArray A_values_array = system["A_values"];
    cnpy::NpyArray b_array = system["b"];
    cnpy::NpyArray x_array = system["x"];

    const auto n = static_cast<int64_t>(b_array.shape[0]);
    const auto nnz = static_cast<int64_t>(A_values_array.shape[0]);

    Eigen::VectorXd b = Eigen::Map<Eigen::VectorXd>(b_array.data<double>(), n);
    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(x_array.data<double>(), n);
    Eigen::MatrixX2<int> indices =
        Eigen::Map<Eigen::MatrixX2<int>>(A_indices_array.data<int>(), nnz, 2);
    Eigen::VectorXd values =
        Eigen::Map<Eigen::VectorXd>(A_values_array.data<double>(), nnz);

    std::vector<Eigen::Triplet<double>> upper_A_entries;
    upper_A_entries.reserve(nnz / 2 + n);
    for (int k = 0; k < nnz; ++k) {
        if (indices(k, 0) <= indices(k, 1)) {
            upper_A_entries.emplace_back(Eigen::Triplet<double>(
                indices(k, 0), indices(k, 1), values(k)));
        }
    }
    Eigen::SparseMatrix<double, Eigen::RowMajor> upper_A(n, n);
    upper_A.setFromTriplets(upper_A_entries.begin(), upper_A_entries.end());
    upper_A.makeCompressed();
    return SparseSystem{upper_A, b, x};
}

void SparseSystem::saveNpz(const std::filesystem::path& npz_path) const
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(
        _upper_A.selfadjointView<Eigen::Upper>());
    const size_t n = A.rows();
    const size_t nnz = A.nonZeros();

    using InnerIterator =
        Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator;
    std::vector<int> indices_vector;
    indices_vector.reserve(2 * nnz);
    for (Eigen::Index row = 0; row < static_cast<Eigen::Index>(n); ++row) {
        for (InnerIterator it(A, row); it; ++it) {
            indices_vector.emplace_back(it.row());
        }
    }
    for (size_t c = 0; c < nnz; ++c) {
        indices_vector.emplace_back(*(A.innerIndexPtr() + c));
    }

    cnpy::npz_save<int>(npz_path.string(), "A_indices", indices_vector.data(),
                        {2, nnz}, "a");
    cnpy::npz_save<double>(npz_path.string(), "A_values", A.valuePtr(), {nnz},
                           "a");
    cnpy::npz_save<double>(npz_path.string(), "b", _b.data(), {n}, "a");
    cnpy::npz_save<double>(npz_path.string(), "x", _x.data(), {n}, "a");
}

void SparseSystem::removeNullEntries(double tolerance)
{
    _upper_A.prune([tolerance](const Eigen::Index row, const Eigen::Index col,
                               const double value) {
        return std::abs(value) > tolerance;
    });
}

size_t SparseSystem::dim() const
{
    return static_cast<size_t>(_b.size());
}

size_t SparseSystem::nnz() const
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(
        _upper_A.selfadjointView<Eigen::Upper>());
    return static_cast<size_t>(A.nonZeros());
}

Eigen::VectorXd SparseSystem::getEigenSolution() const
{
    return _x;
}

Vec SparseSystem::getPetscSolution() const
{
    auto n = static_cast<int>(_b.size());
    Vec x;
    VecCreate(PETSC_COMM_WORLD, &x);
    VecSetSizes(x, PETSC_DECIDE, n);
    VecSetFromOptions(x);
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    VecSetValues(x, n, indices.data(), _x.data(), INSERT_VALUES);
    VecAssemblyBegin(x);
    VecAssemblyEnd(x);
    return x;
}

std::tuple<Eigen::SparseMatrix<double, Eigen::ColMajor>, Eigen::VectorXd>
SparseSystem::toEigenCSC() const
{
    Eigen::SparseMatrix<double, Eigen::ColMajor> A(
        _upper_A.selfadjointView<Eigen::Upper>());
    return std::make_tuple(A, _b);
}

std::tuple<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::VectorXd>
SparseSystem::toEigenCSR() const
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(
        _upper_A.selfadjointView<Eigen::Upper>());
    return std::make_tuple(A, _b);
}

std::tuple<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::VectorXd>
SparseSystem::toEigenUpperCSR() const
{
    return std::make_tuple(_upper_A, _b);
}

std::tuple<Mat, Vec> SparseSystem::toPetscCSR() const
{
    auto n = static_cast<int>(_b.size());

    Mat upper_A;
    Eigen::SparseMatrix<double, Eigen::RowMajor> eigen_upper_A = _upper_A;
    MatCreateSeqSBAIJWithArrays(
        PETSC_COMM_SELF, 1, n, n, eigen_upper_A.outerIndexPtr(),
        eigen_upper_A.innerIndexPtr(), eigen_upper_A.valuePtr(), &upper_A);
    Mat A;
    MatConvert(upper_A, MATSEQAIJ, MAT_INITIAL_MATRIX, &A);

    Vec b;
    VecCreate(PETSC_COMM_WORLD, &b);
    VecSetSizes(b, PETSC_DECIDE, n);
    VecSetFromOptions(b);
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    VecSetValues(b, n, indices.data(), _b.data(), INSERT_VALUES);
    VecAssemblyBegin(b);
    VecAssemblyEnd(b);

    return std::make_tuple(A, b);
}

std::tuple<std::vector<int>,
           std::vector<int>,
           std::vector<double>,
           std::vector<double>>
SparseSystem::toStdCSR() const
{
    const Eigen::SparseMatrix<double, Eigen::RowMajor> A(
        _upper_A.selfadjointView<Eigen::Upper>());
    const size_t n = A.rows();
    const size_t nnz = A.nonZeros();
    std::vector<int> crow_index(A.outerIndexPtr(), A.outerIndexPtr() + n + 1);
    std::vector<int> col_index(A.innerIndexPtr(), A.innerIndexPtr() + nnz);
    std::vector<double> values(A.valuePtr(), A.valuePtr() + nnz);
    std::vector<double> b(_b.data(), _b.data() + n);
    return std::make_tuple(crow_index, col_index, values, b);
}

}    // namespace solvers
