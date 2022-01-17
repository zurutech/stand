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

#include <solvers/SuperLUSolver.hpp>

#include "SolverTest.cpp"

namespace {

class SuperLUSolverTest : public SolverTest {
};

TEST_F(SuperLUSolverTest, TestLu)
{
    solvers::SuperLUSolver solver(solvers::SuperLUReorder::None);
    ASSERT_TRUE(testSolver(solver));
}

TEST_F(SuperLUSolverTest, TestLuMinimumDegree)
{
    solvers::SuperLUSolver solver(solvers::SuperLUReorder::MinimumDegree);
    ASSERT_TRUE(testSolver(solver));
}

TEST_F(SuperLUSolverTest, TestLuColamd)
{
    solvers::SuperLUSolver solver(solvers::SuperLUReorder::ColAMD);
    ASSERT_TRUE(testSolver(solver));
}

TEST_F(SuperLUSolverTest, TestLuMetis)
{
    solvers::SuperLUSolver solver(solvers::SuperLUReorder::METIS);
    ASSERT_TRUE(testSolver(solver));
}

}    // namespace
