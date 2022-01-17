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
#include <memory>

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <solvers/Solver.hpp>
#include <solvers/SparseSystem.hpp>

namespace {

namespace fs = std::filesystem;

class SolverTest : public ::testing::Test {
protected:
    std::unique_ptr<solvers::SparseSystem> _system;

    SolverTest() : _system(nullptr)
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
        _system = std::make_unique<solvers::SparseSystem>(system);
    }

    bool testSolver(const solvers::Solver& solver,
                    const double tolerance = 1e-8)
    {
        double duration;
        Eigen::VectorXd result = solver.solve(*_system, duration);
        Eigen::VectorXd x = _system->getEigenSolution();
        return result.isApprox(x, tolerance);
    }
};

}    // namespace
