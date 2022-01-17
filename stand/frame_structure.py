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
"""Dataclass describing a frame structure."""

import itertools
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, ClassVar, Tuple, List, Set, Dict

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from openseespy import opensees
import opsvis
import scipy.sparse

from .beam import Beam
from .frame import Frame
from .static_analysis_problem import StaticAnalysisProblem


@dataclass(eq=False)
class FrameStructure:
    """
    Describe a frame structure.

    Object Attributes:
        beam (Beam): Beam parameters shared by all beams in the structure.
        grid (nx.Graph): Graph describing beam connectivity.
        _pattern_id (int): Identifier of current load pattern object.

    Attributes:
        WIND_K (tuple[float, ...]): Scaling parameters of the coefficient of
            wind exposition, one for each category of exposition.
        WIND_Z_MIN (tuple[float, ...]): Height thresholds (m) used in the
            calculation of the coefficient of wind exposition,
            one for each category of exposition.
        WIND_Z0 (tuple[float, ...]): Height scaling factors (m) used in
            the calculation of the coefficient of wind exposition,
            one for each category of exposition.
    """

    beam: Beam
    grid: nx.Graph
    _pattern_id: int = 0
    WIND_K: ClassVar[Tuple[float, ...]] = (0.17, 0.19, 0.20, 0.22, 0.23)
    WIND_Z_MIN: ClassVar[Tuple[float, ...]] = (2.0, 4.0, 5.0, 8.0, 12.0)
    WIND_Z0: ClassVar[Tuple[float, ...]] = (0.01, 0.05, 0.10, 0.30, 0.70)

    @staticmethod
    def _local_beam_vector(
        beam_axis: int, global_vector_axis: int, magnitude: float
    ) -> np.ndarray:
        """Convert a coordinate vector expressed in global coordinates
        to a coordinate vector expressed in beam local coordinates.

        Args:
            beam_axis (int): Index of the axis of the beam in global coordinates.
            global_vector_axis (int): Index of the axis of the vector in global coordinates.
            magnitude (float): Magnitude of the vector.

        Returns:
            np.ndarray: Local vector coordinates.
        """
        axis_map = np.array(
            [
                [0, 1, 2],
                [1, 0, 2],
                [2, 1, 0],
            ]
        )
        sign_map = np.ones_like(axis_map)
        sign_map[1, 0] = sign_map[2, 1] = -1
        local_vector = np.zeros(3)
        local_vector[axis_map[beam_axis, global_vector_axis]] = (
            sign_map[beam_axis, global_vector_axis] * magnitude
        )
        return local_vector

    @staticmethod
    def _distribute_pressure(frame: Frame, pressure: float) -> None:
        """Distribute a uniform pression on a frame on supporting beams.

        Args:
            frame (Frame): Description of the frame.
            pressure (float): Pressure on the frame (Pa).
        """
        loads = frame.load_distribution * pressure
        load_vector1 = FrameStructure._local_beam_vector(
            frame.edge_axes[0], frame.orthogonal_axis, loads[0]
        )
        load_vector2 = FrameStructure._local_beam_vector(
            frame.edge_axes[1], frame.orthogonal_axis, loads[1]
        )
        opensees.eleLoad(
            "-ele",
            *frame.edge_indices[::2],
            "-type",
            "-beamUniform",
            load_vector1[1],
            load_vector1[2],
            load_vector1[0],
        )
        opensees.eleLoad(
            "-ele",
            *frame.edge_indices[1::2],
            "-type",
            "-beamUniform",
            load_vector2[1],
            load_vector2[2],
            load_vector2[0],
        )

    @cached_property
    def size(self) -> np.ndarray:
        """Get the size of the bounding box around the structure.

        Returns:
            np.ndarray: Size of the structure bounding box.
        """
        grid_coords = np.array(self.grid.nodes)
        return np.max(grid_coords, axis=0) - np.min(grid_coords, axis=0)

    @cached_property
    def frames(self) -> Set[Frame]:
        """Get all frames in the structure.

        Returns:
            Set[Frame]: Set of frames in the structure.
        """
        frames = set()
        for v in self.grid:
            for u, w in itertools.combinations(self.grid[v], 2):
                for z in (set(self.grid[u]) & set(self.grid[w])) - {v}:
                    frames.add(Frame.from_grid_nodes(self.grid, (v, u, z, w)))
        return frames

    @cached_property
    def perimeter_frames(self) -> Dict[str, Set[Frame]]:
        """Get the perimeter frames of the structure divided
        by their outward orthogonal direction.

        Returns:
            Dict[str, Set[Frame]]: Dictionary of frames indexed by their
                outward orthogonal direction with keys
                ("-x", "+x", "-y", "+y", "-z", "+z").
        """
        aligned_groups: Dict[Tuple[int, ...], List[Frame]] = {}
        for r in self.frames:
            key = r.parallel_key
            if key not in aligned_groups:
                aligned_groups[key] = [r, r]
                continue
            group_lowest, group_highest = aligned_groups[key]
            if r.plane_coordinate < group_lowest.plane_coordinate:
                aligned_groups[key][0] = r
            if r.plane_coordinate > group_highest.plane_coordinate:
                aligned_groups[key][1] = r
        axis_names = ("x", "y", "z")
        result = {
            "-x": set(),
            "+x": set(),
            "-y": set(),
            "+y": set(),
            "-z": set(),
            "+z": set(),
        }
        for group_lowest, group_highest in aligned_groups.values():
            result[f"-{axis_names[group_lowest.orthogonal_axis]}"].add(group_lowest)
            result[f"+{axis_names[group_highest.orthogonal_axis]}"].add(group_highest)
        return result

    def build_model(self) -> scipy.sparse.coo_matrix:
        """Build OpenSees model of the structure.

        Returns:
            scipy.sparse.coo_matrix: Stiffness matrix of the structure in COO format.
        """
        opensees.wipe()
        opensees.model("basic", "-ndm", 3)
        for node, node_index in self.grid.nodes().data("index"):
            opensees.node(node_index, node[0], node[1], node[2])
            if np.isclose(node[2], 0.0):
                opensees.fix(node_index, 1, 1, 1, 1, 1, 1)

        # Geometric transformation for horizontal beams
        horizontal_transf_id = 0
        opensees.geomTransf("Linear", horizontal_transf_id, 0.0, 0.0, 1.0)
        # Geometric transformation for vertical beams (columns)
        vertical_transf_id = 1
        opensees.geomTransf("Linear", vertical_transf_id, 1.0, 0.0, 0.0)

        node_indices = nx.get_node_attributes(self.grid, "index")
        for u, v, edge_data in self.grid.edges(data=True):
            opensees.element(
                "ElasticTimoshenkoBeam",
                edge_data["index"],
                node_indices[u],
                node_indices[v],
                self.beam.young_modulus,
                self.beam.shear_modulus,
                self.beam.section_x,
                self.beam.inertia_xx,
                self.beam.inertia_yy,
                self.beam.inertia_zz,
                self.beam.section_y,
                self.beam.section_z,
                vertical_transf_id if edge_data["axis"] == 2 else horizontal_transf_id,
            )
        opensees.numberer("Plain")
        opensees.constraints("Plain")
        opensees.system("FullGeneral")
        opensees.algorithm("Linear")
        opensees.integrator("GimmeMCK", 0.0, 0.0, 1.0)
        opensees.analysis("Transient")
        opensees.analyze(1, 0.0)
        system_size = opensees.systemSize()
        stiffness = np.array(opensees.printA("-ret"))
        stiffness = np.reshape(stiffness, (system_size, system_size))
        stiffness = scipy.sparse.coo_matrix(stiffness)

        opensees.wipeAnalysis()
        time_series_id = 1
        opensees.timeSeries("Constant", time_series_id)
        opensees.pattern("Plain", self._pattern_id, time_series_id)
        opensees.numberer("Plain")
        opensees.constraints("Plain")
        opensees.system("SparseSPD")
        opensees.algorithm("Linear")
        opensees.integrator("LoadControl", 1.0)
        opensees.analysis("Static")
        return stiffness

    def weight_load(self) -> None:
        """Apply load associated to the proper weight of the structure."""
        horizontal_elements = []
        vertical_elements = []
        for _, __, edge_data in self.grid.edges(data=True):
            if edge_data["axis"] == 2:
                vertical_elements.append(edge_data["index"])
            else:
                horizontal_elements.append(edge_data["index"])
        opensees.eleLoad(
            "-ele",
            *horizontal_elements,
            "-type",
            "-beamUniform",
            0.0,
            -self.beam.weight_uniform_load,
        )
        opensees.eleLoad(
            "-ele",
            *vertical_elements,
            "-type",
            "-beamUniform",
            0.0,
            0.0,
            -self.beam.weight_uniform_load,
        )

    def pressure_load(self, pressure: float) -> None:
        """Apply a vertical uniform pressure on all horizontal frames.

        Args:
            pressure (float): Uniform pressure (Pa).
        """
        for frame in self.frames:
            if frame.orthogonal_axis != 2:
                continue
            self._distribute_pressure(frame, -pressure)

    def snow_load(self, pressure: float) -> None:
        """Apply snow pressure on frames at the top of the structure."""
        shape_coefficient = 0.8
        for r in self.perimeter_frames["+z"]:
            self._distribute_pressure(r, -pressure * shape_coefficient)

    def wind_load(self, wind_speed: float, direction: str, exposition: int) -> None:
        """Apply wind load on perimeter frames.

        Args:
            wind_speed (float): Speed of wind (m/s).
            direction (str): String indicating wind direction.
                Valid choices are: "-x", "+x", "-y", "+y".
            exposition (int): Index of class of exposition. Valid choiches are: 0, 1, 2, 3, 4.

        Raises:
            ValueError: If `direction` is not one of the accepted values.
            ValueError: If `exposition` is not one of the accepted values.
        """
        if direction not in ("-x", "+x", "-y", "+y"):
            raise ValueError(f"invalid wind direction {direction}")
        if not 0 <= exposition < 5:
            raise ValueError(f"invalid wind exposition category {exposition}")
        sign = direction[0]
        direction_axis = direction[1]
        opposite_direction = ("-" if sign == "+" else "+") + direction_axis
        lateral_axis = "y" if direction_axis == "x" else "x"
        windward_frames = self.perimeter_frames[opposite_direction]
        leeward_frames = self.perimeter_frames[direction]
        negative_lateral_frames = self.perimeter_frames[f"-{lateral_axis}"]
        positive_lateral_frames = self.perimeter_frames[f"+{lateral_axis}"]

        wind_pressure = 0.5 * 1.25 * wind_speed ** 2

        k = self.WIND_K[exposition]
        z_min = self.WIND_Z_MIN[exposition]
        z_0 = self.WIND_Z0[exposition]

        b, d, h = self.size
        if direction_axis == "x":
            b, d = d, b
        ratio = h / d
        windward_cp = 0.7 + 0.1 * ratio if ratio < 1 else 0.8
        leeward_cp = 0.3 + 0.2 * ratio if ratio < 1 else 0.5 + 0.05 * (ratio - 1)
        lateral_cp = 0.5 + 0.8 * ratio if ratio < 0.5 else 0.9
        if sign == "-":
            windward_cp *= -1
            leeward_cp *= -1
        negative_lateral_cp = -lateral_cp
        positive_lateral_cp = lateral_cp
        for frames, cp in (
            (windward_frames, windward_cp),
            (leeward_frames, leeward_cp),
            (negative_lateral_frames, negative_lateral_cp),
            (positive_lateral_frames, positive_lateral_cp),
        ):
            for r in frames:
                pressure = (
                    wind_pressure * cp * r.exposition_coefficient(b, k, z_min, z_0)
                )
                self._distribute_pressure(r, pressure)

    def reset_loads(self) -> None:
        """Clear loads applied to the structure."""
        opensees.remove("loadPattern", self._pattern_id)
        self._pattern_id += 1
        opensees.pattern("Plain", self._pattern_id, 1)

    @staticmethod
    def plot(output_path: str) -> None:
        """Export 3D plot of the structure.

        Args:
            output_path (str): Path of output image.
        """
        opsvis.plot_model(node_labels=0, element_labels=0, local_axes=True)
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def deform(
        stiffness: scipy.sparse.coo_matrix, output_path: Optional[str] = None
    ) -> StaticAnalysisProblem:
        """Perform linear static analysis to get the deformation of
        the structure under the applied loads.
        Optionally export the 3D plot of the deformed structure.

        Args:
            stiffness (scipy.sparse.coo_matrix): Stiffness matrix.
            output_path (Optional[str], optional): Path of output image.
                Defaults to None, in that case no plot is exported.

        Returns:
            StaticAnalysisProblem: Solved linear static analysis problem.
        """
        analysis_steps = 1
        opensees.analyze(analysis_steps)

        if output_path:
            opsvis.plot_defo()
            plt.savefig(output_path)
            plt.close()

        system_size = opensees.systemSize()
        displacement = np.empty(system_size)
        node_tags = opensees.getNodeTags()
        for node_tag in node_tags:
            dofs = np.array(opensees.nodeDOFs(node_tag))
            node_displacement = np.array(opensees.nodeDisp(node_tag))
            dofs_mask = dofs != -1
            displacement[dofs[dofs_mask]] = node_displacement[dofs_mask]
        load = stiffness.dot(displacement)

        return StaticAnalysisProblem(stiffness, displacement, load)
