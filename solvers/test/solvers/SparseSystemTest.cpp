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

#include <filesystem>

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <petscmat.h>
#include <petscvec.h>

#include <solvers/SparseSystem.hpp>

namespace {

namespace fs = std::filesystem;

TEST(SparseSystemTest, TestLoadNpz)
{
    const fs::path datasets_dir(DATASETS_DIR);
    const fs::path dataset_dir = datasets_dir / "stand_small_test";
    fs::path npz_path;
    for (const fs::path& entry : fs::directory_iterator(dataset_dir)) {
        if (entry.extension() == ".npz") {
            npz_path = entry;
            break;
        }
    }
    solvers::SparseSystem system = solvers::SparseSystem::loadNpz(npz_path);
    auto [A, b] = system.toEigenCSR();
    Eigen::VectorXd x = system.getEigenSolution();
    constexpr double tolerance = 1e-6;
    ASSERT_TRUE(b.isApprox(A * x, tolerance));
}

TEST(SparseSystemTest, TestSaveNpz)
{
    constexpr size_t n = 4;
    std::vector<Eigen::Triplet<double>> upper_A_entries = {
        Eigen::Triplet<double>(0, 0, 1.),  Eigen::Triplet<double>(0, 1, -2.),
        Eigen::Triplet<double>(0, 3, 1.5), Eigen::Triplet<double>(1, 1, 4.),
        Eigen::Triplet<double>(1, 2, -1.), Eigen::Triplet<double>(2, 3, -3.),
        Eigen::Triplet<double>(3, 3, 0.5),
    };
    Eigen::SparseMatrix<double, Eigen::RowMajor> upper_A(n, n);
    upper_A.setFromTriplets(upper_A_entries.begin(), upper_A_entries.end());
    upper_A.makeCompressed();
    Eigen::Vector4d x(-0.5, 1., 2., -2.);
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(
        upper_A.selfadjointView<Eigen::Upper>());
    Eigen::Vector4d b = A * x;
    solvers::SparseSystem system = solvers::SparseSystem(upper_A, b, x);

    fs::path npz_path = fs::temp_directory_path() / "test.npz";
    system.saveNpz(npz_path);
    solvers::SparseSystem loaded_system =
        solvers::SparseSystem::loadNpz(npz_path);
    auto [loaded_upper_A, loaded_b] = system.toEigenUpperCSR();
    Eigen::VectorXd loaded_x = system.getEigenSolution();
    ASSERT_TRUE(b.isApprox(loaded_b));
    ASSERT_TRUE(x.isApprox(loaded_x));
    ASSERT_TRUE(upper_A.isApprox(loaded_upper_A));
}

TEST(SparseSystemTest, TestToPetsc)
{
    constexpr double tolerance = 1e-6;

    fs::path datasets_dir(DATASETS_DIR);
    fs::path dataset_dir = datasets_dir / "stand_small_test";
    fs::path npz_path;
    for (const std::filesystem::path& entry :
         std::filesystem::directory_iterator(dataset_dir)) {
        if (entry.extension() == ".npz") {
            npz_path = entry;
            break;
        }
    }
    solvers::SparseSystem system = solvers::SparseSystem::loadNpz(npz_path);

    auto n = static_cast<int>(system.dim());
    auto [A, b] = system.toPetscCSR();
    double b_norm = 1.;
    VecNorm(b, NormType::NORM_2, &b_norm);
    Vec x = system.getPetscSolution();
    Vec r;
    double r_norm = b_norm;
    VecCreateSeq(PETSC_COMM_SELF, n, &r);
    MatMult(A, x, r);
    VecAXPY(r, -1, b);
    VecNorm(r, NormType::NORM_2, &r_norm);
    ASSERT_TRUE(r_norm < tolerance * b_norm);
}

}    // namespace
