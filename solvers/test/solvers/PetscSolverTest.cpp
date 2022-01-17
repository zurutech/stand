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

#include <solvers/PetscSolver.hpp>

#include "SolverTest.cpp"

namespace {

class PetscSolverTest : public SolverTest {
public:
    const double test_tolerance = 1e-2;
};

TEST_F(PetscSolverTest, TestCg)
{
    solvers::PetscSolver solver(solvers::PetscMethod::CG);
    ASSERT_TRUE(testSolver(solver, test_tolerance));
}

TEST_F(PetscSolverTest, TestFlexibleCg)
{
    solvers::PetscSolver solver(solvers::PetscMethod::FlexibleCG);
    ASSERT_TRUE(testSolver(solver, test_tolerance));
}

TEST_F(PetscSolverTest, TestGmres)
{
    solvers::PetscSolver solver(solvers::PetscMethod::GMRES);
    ASSERT_TRUE(testSolver(solver, test_tolerance));
}

TEST_F(PetscSolverTest, TestFlexibleGmres)
{
    solvers::PetscSolver solver(solvers::PetscMethod::FlexibleGMRES);
    ASSERT_TRUE(testSolver(solver, test_tolerance));
}

TEST_F(PetscSolverTest, TestQmr)
{
    solvers::PetscSolver solver(solvers::PetscMethod::QMR);
    ASSERT_TRUE(testSolver(solver, test_tolerance));
}

TEST_F(PetscSolverTest, TestCr)
{
    solvers::PetscSolver solver(solvers::PetscMethod::CR);
    ASSERT_TRUE(testSolver(solver, test_tolerance));
}

TEST_F(PetscSolverTest, TestMinres)
{
    solvers::PetscSolver solver(solvers::PetscMethod::MinRes);
    ASSERT_TRUE(testSolver(solver, test_tolerance));
}

TEST_F(PetscSolverTest, TestSymmlq)
{
    solvers::PetscSolver solver(solvers::PetscMethod::SymmLQ);
    ASSERT_TRUE(testSolver(solver, test_tolerance));
}

TEST_F(PetscSolverTest, TestCholesky)
{
    solvers::PetscSolver solver(solvers::PetscMethod::Cholesky);
    ASSERT_TRUE(testSolver(solver, test_tolerance));
}

TEST_F(PetscSolverTest, TestLu)
{
    solvers::PetscSolver solver(solvers::PetscMethod::LU);
    ASSERT_TRUE(testSolver(solver, test_tolerance));
}

TEST_F(PetscSolverTest, TestQr)
{
    solvers::PetscSolver solver(solvers::PetscMethod::QR);
    ASSERT_TRUE(testSolver(solver, test_tolerance));
}

TEST_F(PetscSolverTest, TestCgJacobi)
{
    solvers::PetscSolver solver(solvers::PetscMethod::CG,
                                solvers::PetscPreconditioner::Jacobi);
    ASSERT_TRUE(testSolver(solver, test_tolerance));
}

TEST_F(PetscSolverTest, TestCgSor)
{
    solvers::PetscSolver solver(solvers::PetscMethod::CG,
                                solvers::PetscPreconditioner::SOR);
    ASSERT_TRUE(testSolver(solver, test_tolerance));
}

TEST_F(PetscSolverTest, TestCgEisenstat)
{
    solvers::PetscSolver solver(solvers::PetscMethod::CG,
                                solvers::PetscPreconditioner::Eisenstat);
    ASSERT_TRUE(testSolver(solver, test_tolerance));
}

TEST_F(PetscSolverTest, TestCgIlu)
{
    solvers::PetscSolver solver(solvers::PetscMethod::CG,
                                solvers::PetscPreconditioner::ILU);
    ASSERT_TRUE(testSolver(solver, test_tolerance));
}

TEST_F(PetscSolverTest, TestCgIcc)
{
    solvers::PetscSolver solver(solvers::PetscMethod::CG,
                                solvers::PetscPreconditioner::ICC);
    ASSERT_TRUE(testSolver(solver, test_tolerance));
}

}    // namespace
