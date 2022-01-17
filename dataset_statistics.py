#!/usr/bin/env python3

# Copyright 2022 Zuru Tech HK Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Script to compute some dataset statistics on StAnD."""

import argparse
import sys
from pathlib import Path
import numpy as np


def cli():
    parser = argparse.ArgumentParser(
        description="StAnD statistics script",
        usage=("python3 dataset_statistics.py dataset-dir"),
    )
    parser.add_argument(
        "dataset_dir",
        metavar="dataset-dir",
        help="Path to dataset directory",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser()
    problem_paths = list(dataset_dir.glob("*.npz"))
    dofs_sum = 0
    dofs_max = 0
    for p in problem_paths:
        x = np.load(p)["x"]
        if x.size > dofs_max:
            dofs_max = x.size
        dofs_sum += x.size
    print(
        f"Average number of DOFs for dataset {dataset_dir.stem}: {round(dofs_sum / len(problem_paths)):d}"
    )
    print(f"Maximum number of DOFs for dataset {dataset_dir.stem}: {dofs_max:d}")


if __name__ == "__main__":
    sys.exit(cli())
