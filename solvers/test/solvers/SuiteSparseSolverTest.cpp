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

#include <solvers/SuiteSparseSolver.hpp>

#include "SolverTest.cpp"

namespace {

class SuiteSparseSolverTest : public SolverTest {
};

TEST_F(SuiteSparseSolverTest, TestSimplicialLLT)
{
    solvers::SuiteSparseSolver solver(
        solvers::SuiteSparseMethod::SimplicialLLT);
    ASSERT_TRUE(testSolver(solver));

    constexpr bool gpu = true;
    solvers::SuiteSparseSolver gpu_solver(
        solvers::SuiteSparseMethod::SimplicialLLT, gpu);
    ASSERT_TRUE(testSolver(gpu_solver));
}

TEST_F(SuiteSparseSolverTest, TestSimplicialLDLT)
{
    solvers::SuiteSparseSolver solver(
        solvers::SuiteSparseMethod::SimplicialLDLT);
    ASSERT_TRUE(testSolver(solver));

    constexpr bool gpu = true;
    solvers::SuiteSparseSolver gpu_solver(
        solvers::SuiteSparseMethod::SimplicialLDLT, gpu);
    ASSERT_TRUE(testSolver(gpu_solver));
}

TEST_F(SuiteSparseSolverTest, TestSupernodalLLT)
{
    solvers::SuiteSparseSolver solver(
        solvers::SuiteSparseMethod::SupernodalLLT);
    ASSERT_TRUE(testSolver(solver));

    constexpr bool gpu = true;
    solvers::SuiteSparseSolver gpu_solver(
        solvers::SuiteSparseMethod::SupernodalLLT, gpu);
    ASSERT_TRUE(testSolver(gpu_solver));
}

TEST_F(SuiteSparseSolverTest, TestLU)
{
    solvers::SuiteSparseSolver solver(solvers::SuiteSparseMethod::LU);
    ASSERT_TRUE(testSolver(solver));
}

TEST_F(SuiteSparseSolverTest, TestKLU)
{
    solvers::SuiteSparseSolver solver(solvers::SuiteSparseMethod::KLU);
    ASSERT_TRUE(testSolver(solver));
}

}    // namespace
