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
"""Command line interface for stand"""

import argparse
import sys
from pathlib import Path
import multiprocessing

import yaml

from .static_analysis_generator import StaticAnalysisGenerator


def cli():
    parser = argparse.ArgumentParser(
        description="Command line interface for StAnD",
        usage=("python3 -m stand config-path"),
    )
    parser.add_argument(
        "config_path",
        metavar="config-path",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel processes",
    )
    args = parser.parse_args(sys.argv[1:])

    config_path = Path(args.config_path).expanduser()
    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    params = {k.lower(): v for k, v in config["PARAMETERS"].items()}
    generator = StaticAnalysisGenerator(**params)
    output_dir = Path(config["DIRECTORY"]).expanduser()
    generator.export_problems(
        output_dir,
        config["NUM_STRUCTURES"],
        config["NUM_PROBLEMS_PER_STRUCTURE"],
        processes=args.processes,
    )


if __name__ == "__main__":
    sys.exit(cli())
