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

#include <solvers/ViennaCLSolver.hpp>

#include "SolverTest.cpp"

namespace {

class ViennaCLSolverTest : public SolverTest {
public:
    const double tolerance = 1e-5;
    const int iterations = 1000;
    const double test_tolerance = 1e-2;
};

TEST_F(ViennaCLSolverTest, TestCg)
{
    solvers::ViennaCLSolver solver(solvers::ViennaCLMethod::CG,
                                   solvers::ViennaCLPreconditioner::None,
                                   tolerance, iterations);
    ASSERT_TRUE(testSolver(solver, test_tolerance));
}

TEST_F(ViennaCLSolverTest, TestGmres)
{
    solvers::ViennaCLSolver solver(solvers::ViennaCLMethod::GMRES,
                                   solvers::ViennaCLPreconditioner::None,
                                   tolerance, iterations);
    ASSERT_TRUE(testSolver(solver, test_tolerance));
}

TEST_F(ViennaCLSolverTest, TestCgChowPatel)
{
    solvers::ViennaCLSolver solver(solvers::ViennaCLMethod::CG,
                                   solvers::ViennaCLPreconditioner::ChowPatel,
                                   tolerance, iterations);
    ASSERT_TRUE(testSolver(solver, test_tolerance));
}

TEST_F(ViennaCLSolverTest, TestCgIlu0)
{
    solvers::ViennaCLSolver solver(solvers::ViennaCLMethod::CG,
                                   solvers::ViennaCLPreconditioner::ILU0,
                                   tolerance, iterations);
    ASSERT_TRUE(testSolver(solver, test_tolerance));
}

TEST_F(ViennaCLSolverTest, TestCgIchol0)
{
    solvers::ViennaCLSolver solver(solvers::ViennaCLMethod::CG,
                                   solvers::ViennaCLPreconditioner::IChol0,
                                   tolerance, iterations);
    ASSERT_TRUE(testSolver(solver, test_tolerance));
}

TEST_F(ViennaCLSolverTest, TestCgJacobi)
{
    solvers::ViennaCLSolver solver(solvers::ViennaCLMethod::CG,
                                   solvers::ViennaCLPreconditioner::Jacobi,
                                   tolerance, iterations);
    ASSERT_TRUE(testSolver(solver, test_tolerance));
}

TEST_F(ViennaCLSolverTest, TestCgRowScaling)
{
    solvers::ViennaCLSolver solver(solvers::ViennaCLMethod::CG,
                                   solvers::ViennaCLPreconditioner::RowScaling,
                                   tolerance, iterations);
    ASSERT_TRUE(testSolver(solver, test_tolerance));
}

}    // namespace
