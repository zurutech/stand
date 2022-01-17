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

#include <gtest/gtest.h>

#include <solvers/CuSparseSolver.hpp>

#include "SolverTest.cpp"

namespace {

class CuSparseSolverTest : public SolverTest {
};

TEST_F(CuSparseSolverTest, TestQr)
{
    solvers::CuSparseSolver solver(solvers::CuSparseMethod::QR,
                                   solvers::CuSparseReorder::None);
    ASSERT_TRUE(testSolver(solver));
}

TEST_F(CuSparseSolverTest, TestQrSymRCM)
{
    solvers::CuSparseSolver solver(solvers::CuSparseMethod::QR,
                                   solvers::CuSparseReorder::SymRCM);
    ASSERT_TRUE(testSolver(solver));
}

TEST_F(CuSparseSolverTest, TestQrSymAMD)
{
    solvers::CuSparseSolver solver(solvers::CuSparseMethod::QR,
                                   solvers::CuSparseReorder::SymAMD);
    ASSERT_TRUE(testSolver(solver));
}

TEST_F(CuSparseSolverTest, TestQrMetis)
{
    solvers::CuSparseSolver solver(solvers::CuSparseMethod::QR,
                                   solvers::CuSparseReorder::METIS);
    ASSERT_TRUE(testSolver(solver));
}

TEST_F(CuSparseSolverTest, TestCholesky)
{
    solvers::CuSparseSolver solver(solvers::CuSparseMethod::Cholesky,
                                   solvers::CuSparseReorder::None);
    ASSERT_TRUE(testSolver(solver));
}

TEST_F(CuSparseSolverTest, TestCholeskySymRCM)
{
    solvers::CuSparseSolver solver(solvers::CuSparseMethod::Cholesky,
                                   solvers::CuSparseReorder::SymRCM);
    ASSERT_TRUE(testSolver(solver));
}

TEST_F(CuSparseSolverTest, TestCholeskySymAMD)
{
    solvers::CuSparseSolver solver(solvers::CuSparseMethod::Cholesky,
                                   solvers::CuSparseReorder::SymAMD);
    ASSERT_TRUE(testSolver(solver));
}

TEST_F(CuSparseSolverTest, TestCholeskyMetis)
{
    solvers::CuSparseSolver solver(solvers::CuSparseMethod::Cholesky,
                                   solvers::CuSparseReorder::METIS);
    ASSERT_TRUE(testSolver(solver));
}

}    // namespace
