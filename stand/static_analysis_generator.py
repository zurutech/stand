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
"""Generate random solved linear static analysis problems on frame structures."""

from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Union, Tuple, Iterator

import numpy as np
import networkx as nx
from tqdm import tqdm

from .random_value import RandomValue
from .beam import Beam
from .frame_structure import FrameStructure, StaticAnalysisProblem


class StaticAnalysisGenerator:
    def __init__(
        self,
        beam_length: Union[float, Tuple[float, float]],
        beam_width: Union[float, Tuple[float, float]],
        beam_height: Union[float, Tuple[float, float]],
        grid_depth: Union[int, Tuple[int, int]],
        grid_width: Union[int, Tuple[int, int]],
        grid_height: Union[int, Tuple[int, int]],
        young_modulus: Union[float, Tuple[float, float]] = 3.0e10,
        density: Union[float, Tuple[float, float]] = 2.5e3,
        pressure: Union[float, Tuple[float, float]] = 2.0e3,
        wind_speed: Union[float, Tuple[float, float]] = (0.25, 0.31),
        snow_pressure: Union[float, Tuple[float, float]] = (6e2, 5.6e3),
        seed: Optional[int] = None,
    ):
        """Build a StaticAnalysisGenerator.

        Args:
            beam_length (Union[float, Tuple[float, float]]):
                Boundaries of the interval used to sample beam lengths (m).
            beam_width (Union[float, Tuple[float, float]]):
                Boundaries of the interval used to sample the beam section width (m).
            beam_height (Union[float, Tuple[float, float]]):
                Boundaries of the interval used to sample the beam section height (m).
            grid_depth (Union[int, Tuple[int, int]]): Boundaries of the interval used
                to sample the number of beams along the depth of the base grid structure.
            grid_width (Union[int, Tuple[int, int]]): Boundaries of the interval used
                to sample the number of beams along the width of the base grid structure.
            grid_height (Union[int, Tuple[int, int]]): Boundaries of the interval used
                to sample the number of beams along the height of the base grid structure.
            young_modulus (Union[float, Tuple[float, float]], optional):
                Boundaries of the interval used to sample the beam Young's modulus (Pa).
                Defaults to 3.0e10.
            density (Union[float, Tuple[float, float]], optional):
                Boundaries of the interval used to sample the beam density (kg/m^3).
                Defaults to 2.5e3.
            pressure (Union[float, Tuple[float, float]], optional):
                Boundaries of the interval used to sample the vertical frame pressure applied (Pa).
                Defaults to 2.0e3.
            wind_speed (Union[float, Tuple[float, float]], optional):
                Boundaries of the interval used to sample the speed of wind (m/s).
                Defaults to (0.25, 0.31).
            snow_pressure (Union[float, Tuple[float, float]], optional):
                Boundaries of the interval used to sample the pressure of snow on the roof (Pa).
                Defaults to (6e2, 5.6e3).
            seed (Optional[int], optional): Random seed. Defaults to None.
        """
        super().__init__()
        self.beam_length = RandomValue(beam_length)
        self.beam_width = RandomValue(beam_width)
        self.beam_height = RandomValue(beam_height)
        self.grid_depth = RandomValue(grid_depth)
        self.grid_width = RandomValue(grid_width)
        self.grid_height = RandomValue(grid_height)
        self.young_modulus = RandomValue(young_modulus)
        self.density = RandomValue(density)
        self.pressure = RandomValue(pressure)
        self.wind_speed = RandomValue(wind_speed)
        self.snow_pressure = RandomValue(snow_pressure)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(beam_length={self.beam_length}, "
            f"beam_width={self.beam_width}, beam_height={self.beam_height}, "
            f"grid_depth={self.grid_depth}, grid_width={self.grid_width}, "
            f"grid_height={self.grid_height}, young_modulus={self.young_modulus}, "
            f"density={self.density}, pressure={self.pressure}, wind_speed={self.wind_speed}, "
            f"snow_pressure={self.snow_pressure})"
        )

    def _sample_beam(self) -> Beam:
        """Sample beam parameters.

        Returns:
            Beam: Description of beam parameters.
        """
        beam = Beam(
            width=self.beam_width.get(rng=self.rng),
            height=self.beam_height.get(rng=self.rng),
            young_modulus=self.young_modulus.get(rng=self.rng),
            density=self.density.get(rng=self.rng),
        )
        return beam

    def _sample_grid(self) -> nx.Graph:
        """Sample structure nodes and connectivity.

        Returns:
            nx.Graph: Graph describing the structure.
        """
        grid_depth = self.grid_depth.get(rng=self.rng)
        grid_width = self.grid_width.get(rng=self.rng)
        grid_height = self.grid_height.get(rng=self.rng)
        x_spacing = self.beam_length.get(size=grid_depth, rng=self.rng)
        y_spacing = self.beam_length.get(size=grid_width, rng=self.rng)
        z_spacing = self.beam_length.get(size=grid_height, rng=self.rng)
        x_coordinates = np.insert(np.cumsum(x_spacing), 0, 0.0)
        y_coordinates = np.insert(np.cumsum(y_spacing), 0, 0.0)
        z_coordinates = np.insert(np.cumsum(z_spacing), 0, 0.0)
        grid = nx.grid_graph((z_coordinates, y_coordinates, x_coordinates))
        levels = self.rng.integers(
            z_spacing.size, endpoint=True, size=(x_spacing.size, y_spacing.size)
        )
        heights = z_coordinates[levels]
        max_heights = np.pad(heights, ((0, 1), (0, 1)))
        max_heights[1:, :-1] = np.maximum(max_heights[1:, :-1], heights)
        max_heights[:-1, 1:] = np.maximum(max_heights[:-1, 1:], heights)
        max_heights[1:, 1:] = np.maximum(max_heights[1:, 1:], heights)
        x_base_coordinates, y_base_coordinates = np.meshgrid(
            x_coordinates, y_coordinates, indexing="ij"
        )
        max_heights_dict = {
            (x, y): h
            for x, y, h in zip(
                x_base_coordinates.flat, y_base_coordinates.flat, max_heights.flat
            )
        }
        removable_nodes = [
            n for n in grid.nodes() if n[2] > max_heights_dict[(n[0], n[1])]
        ]
        grid.remove_nodes_from(removable_nodes)
        node_indices = {n: i for i, n in enumerate(grid.nodes)}
        nx.set_node_attributes(grid, node_indices, name="index")
        edge_indices = {
            e: {"index": i, "axis": np.argmin(np.isclose(e[0], e[1]))}
            for i, e in enumerate(grid.edges)
        }
        nx.set_edge_attributes(grid, edge_indices)
        return grid

    def set_seed(self, seed: int) -> None:
        """Set the seed of the random generator.

        Args:
            seed (int): Random seed.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def sample_problems(self, n_problems: int) -> Iterator[StaticAnalysisProblem]:
        """Sample several solved linear static analysis problems
        applying variable random loads to the structure at hand.

        Args:
            n_problems (int): Number of problems to be sampled.

        Yields:
            Iterator[StaticAnalysisProblem]: Iterator over sampled
                linear static analysis problems.
        """
        structure = FrameStructure(self._sample_beam(), self._sample_grid())
        stiffness = structure.build_model()

        n_load_types = 4
        weight_prob = 1 - ((n_load_types - 1) / n_load_types) ** n_problems
        if self.rng.binomial(1, weight_prob) == 1:
            structure.weight_load()
            yield structure.deform(stiffness)
            structure.reset_loads()
            n_problems -= 1
        sampled_load_types = self.rng.integers(3, size=n_problems)
        for l in sampled_load_types:
            if l == 0:
                structure.pressure_load(self.pressure.get(rng=self.rng))
            elif l == 1:
                structure.snow_load(self.snow_pressure.get(rng=self.rng))
            else:
                wind_speed = self.wind_speed.get(rng=self.rng)
                direction = ("-x", "+x", "-y", "+y")[self.rng.integers(4)]
                exposition = self.rng.integers(5)
                structure.wind_load(wind_speed, direction, exposition)
            yield structure.deform(stiffness)
            structure.reset_loads()

    @staticmethod
    def _export_helper(
        generator: "StaticAnalysisGenerator",
        fixed_args: Tuple[int, Path],
    ) -> int:
        """Helper function to support multiprocessing problem generation.

        Args:
            generator (StaticAnalysisGenerator): Static analysis problem generator.
            fixed_args (Tuple[int, Path]): Tuple composed of:
                - number of problems to be sampled for every structure;
                - path to the output directory.

        Returns:
            int: Number of problems sampled.
        """
        n_problems, output_dir = fixed_args
        for problem in generator.sample_problems(n_problems):
            output_path = output_dir / f"{generator.seed}_{problem.sha1()}.npz"
            problem.to_npz(output_path)
        return n_problems

    def export_problems(
        self,
        output_dir: Union[Path, str],
        n_structures: int,
        n_problems_per_structure: int,
        processes: int = 1,
    ) -> None:
        """Generate and export static analysis problems.

        Args:
            output_dir (Union[Path, str]): Path to the output directory.
            n_structures (int): Number of structures to be sampled.
            n_problems_per_structure (int): Number of problems
                to be sampled for every structure.
            processes (int, optional): Number of parallel processes. Defaults to 1.
        """
        output_dir = Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        seeds = self.rng.integers(np.iinfo(np.int64).max, size=n_structures)
        existing_seeds = np.array(
            [int(p.stem.split("_", 1)[0]) for p in output_dir.glob("*.npz")]
        )
        unique_seeds, seed_count = np.unique(existing_seeds, return_counts=True)
        seeds = np.setdiff1d(
            seeds, unique_seeds[seed_count == n_problems_per_structure]
        )
        total_problems = len(seeds) * n_problems_per_structure
        with tqdm(total=total_problems, desc="Export") as progress_bar:
            if processes > 1:
                generators = [deepcopy(self) for _ in range(len(seeds))]
                for seed, generator in zip(seeds, generators):
                    generator.set_seed(seed)
                helper = partial(
                    StaticAnalysisGenerator._export_helper,
                    fixed_args=(n_problems_per_structure, output_dir),
                )
                with Pool(processes) as pool:
                    for c in pool.imap_unordered(helper, generators):
                        progress_bar.update(c)
                    pool.close()
                    pool.join()
            else:
                for seed in seeds:
                    self.set_seed(seed)
                    for problem in self.sample_problems(n_problems_per_structure):
                        output_path = output_dir / f"{self.seed}_{problem.sha1()}.npz"
                        problem.to_npz(output_path)
                        progress_bar.update()
