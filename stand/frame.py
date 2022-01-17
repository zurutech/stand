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
"""Dataclass describing a frame, i.e. a rectangle of four connected beams."""

import operator
from dataclasses import dataclass
from functools import cached_property
from typing import Tuple

import numpy as np
import networkx as nx

Node = Tuple[float, float, float]


@dataclass(eq=False, frozen=True)
class Frame:
    """
    Describe a frame.

    Object Attributes:
        node_indices (tuple[int, int, int, int]): Ordered tuple of frame node indices.
            Node indices come from the structure where the frame is inserted.
        edge_indices (tuple[int, int, int, int]): Ordered tuple of frame beam indices.
            Beam indices come from the structure where the frame is inserted.
        node_coords (np.ndarray): Ordered 3D coordinates of the frame nodes.
    """

    node_indices: Tuple[int, int, int, int]
    edge_indices: Tuple[int, int, int, int]
    node_coords: np.ndarray

    def __post_init__(self) -> None:
        if self.node_coords.shape != (4, 3):
            raise ValueError(
                f"the coordinate array should have shape (4,3) but its shape is {self.node_coords.shape}"
            )

    def __hash__(self):
        return hash(
            self.node_indices + self.edge_indices + tuple(self.node_coords.flat)
        )

    @staticmethod
    def from_grid_nodes(
        grid: nx.Graph, nodes: Tuple[Node, Node, Node, Node]
    ) -> "Frame":
        """Build a frame from a structure graph and four nodes.

        Returns:
            Frame: frame composed of the given four nodes in canonical position.
        """
        node_coords = np.array(nodes)
        node_indices = [grid.nodes[n]["index"] for n in nodes]
        source_nodes = nodes
        target_nodes = nodes[1:] + nodes[:1]
        edge_indices = [grid[s][t]["index"] for s, t in zip(source_nodes, target_nodes)]
        # Fix a canonical order where the first edge has the lowest index
        # and the second edge is the one with the lowest index between the
        # two edges adjacent to the first edge.
        min_index, _ = min(enumerate(edge_indices), key=operator.itemgetter(1))
        edge_indices = edge_indices[min_index:] + edge_indices[:min_index]
        if edge_indices[1] > edge_indices[3]:
            start_index = (min_index + 1) % 4
            node_indices = node_indices[start_index::-1] + node_indices[:start_index:-1]
            edge_indices[1], edge_indices[3] = edge_indices[3], edge_indices[1]
            node_coords = np.row_stack(
                (node_coords[start_index::-1], node_coords[:start_index:-1])
            )
        else:
            node_indices = node_indices[min_index:] + node_indices[:min_index]
            node_coords = np.roll(node_coords, min_index, axis=0)
        node_indices = tuple(node_indices)
        edge_indices = tuple(edge_indices)
        return Frame(node_indices, edge_indices, node_coords)

    @cached_property
    def orthogonal_axis(self) -> int:
        """Get the axis orthogonal to the frame.

        Raises:
            ValueError: If the frame is not orthogonal to a coordinate axis.

        Returns:
            int: Index of the axis orthogonal to the frame.
        """
        equal_coordinates = np.all(
            np.isclose(self.node_coords, self.node_coords[0]), axis=0
        )
        if np.sum(equal_coordinates) != 1:
            raise ValueError(
                "the nodes do not lie on a plane orthogonal to a coordinate axis"
            )
        return np.argmax(equal_coordinates)

    @cached_property
    def plane_coordinate(self) -> float:
        """Get the constant coordinate of the coordinate plane containing the frame.

        Returns:
            float: Coordinate of the coordinate plane containing the frame.
        """
        return self.node_coords[0, self.orthogonal_axis]

    @cached_property
    def edge_axes(self) -> Tuple[int, int, int, int]:
        """Get the indices of the axes of frame beams.

        Raises:
            ValueError: If the frame is not a rectangle.

        Returns:
            tuple[int, int, int, int]: Ordered indices of the axes of frame beams.
        """
        edge_equal_coordinates = np.isclose(
            np.roll(self.node_coords, -1, axis=0), self.node_coords
        )
        if np.any(np.sum(edge_equal_coordinates, axis=1) != 2):
            raise ValueError(
                "the quadrilateral is not a rectangle because some sides are not parallel to a coordinate axis"
            )
        edge_axes = np.argmin(edge_equal_coordinates, axis=1)
        if np.any(edge_axes[:2] != edge_axes[2:]):
            raise ValueError(
                "the quadrilateral is not a rectangle because some opposite sides are not parallel"
            )
        return tuple(edge_axes)

    @cached_property
    def edge_lengths(self) -> np.ndarray:
        """Get the lengths of frame beams.

        Raises:
            ValueError: If the frame is not a rectangle.

        Returns:
            np.ndarray: Ordered lengths of frame beams (m).
        """
        lengths = np.linalg.norm(
            np.roll(self.node_coords, -1, axis=0) - self.node_coords, axis=1
        )
        if not np.allclose(lengths[:2], lengths[2:]):
            raise ValueError(
                "the quadrilateral is not a rectangle because opposite sides have different lengths"
            )
        return lengths

    @cached_property
    def area(self) -> float:
        """Get the area of frame surface.

        Returns:
            float: Area of frame surface (m^2).
        """
        length1, length2 = 0.5 * (self.edge_lengths[:2] + self.edge_lengths[2:])
        return length1 * length2

    @cached_property
    def center(self) -> np.ndarray:
        """Get the center of the frame.

        Returns:
            np.ndarray: Coordinates of the center of the frame.
        """
        return np.mean(self.node_coords, axis=0)

    def exposition_coefficient(
        self, width: float, k_r: float, z_min: float, z_0: float
    ) -> float:
        """Compute the coefficient of wind exposition.

        Args:
            width (float): Width of the face subject to the wind.
            k_r (float): Scaling parameter of the coefficient of wind exposition.
            z_min (float): Height threshold (m).
            z_0 (float): Height scaling factor (m).

        Returns:
            float: Coefficient of wind exposition for the frame.
        """
        z_e = np.max(self.node_coords[:, 2])
        if z_e <= width:
            z_e = width
        if z_e <= z_min:
            z_e = z_min
        return k_r ** 2 * np.log(z_e / z_0) * (7 + np.log(z_e / z_0))

    @cached_property
    def parallel_key(self) -> Tuple[int, ...]:
        """Get a representation of the frame identical for all parallel overlying frames.
        This representation is used to find perimeter frames.

        Returns:
            Tuple[int, ...]: Frame representation independent of the coordinate of the frame plane.
        """
        center_key = tuple(
            np.delete((self.center * 1000).astype(np.int32), self.orthogonal_axis)
        )
        sizes = np.max(self.node_coords, axis=0) - np.min(self.node_coords, axis=0)
        size_key = tuple(
            np.delete((sizes * 1000).astype(np.int32), self.orthogonal_axis)
        )
        return center_key + size_key + (self.orthogonal_axis,)

    @cached_property
    def load_distribution(self) -> np.ndarray:
        """Get the distribution on beams of a uniform pressure applied on the frame.

        Returns:
            np.ndarray: Ordered fractions of pressure applied on beams.
        """
        length1, length2 = 0.5 * (self.edge_lengths[:2] + self.edge_lengths[2:])
        load1 = 0.25 * length1
        load2 = (0.5 * length1 * (length2 - 0.5 * length1)) / length2
        if length1 > length2:
            load1, load2 = load2, load1
        return np.array([load1, load2, load1, load2])
