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
"""Dataclass containing a solved linear static analysis problem."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import scipy.sparse
import numpy as np


@dataclass(frozen=True)
class StaticAnalysisProblem:
    """
    A solved linear static analysis problem.

    Object Attributes:
        K (scipy.sparse.coo_matrix): Sparse stiffness matrix in COO format.
        u (np.ndarray): Displacement vector, i.e. the solution of the linear system.
        f (np.ndarray): Load vector, i.e. the constant term of the linear system.
    """

    K: scipy.sparse.coo_matrix
    u: np.ndarray
    f: np.ndarray

    def to_npz(self, output_path: Union[str, Path]) -> None:
        """Save the linear system in a .npz file.

        Args:
            output_path (Union[str, Path]): Path to the output .npz file.
        """
        output_path = Path(output_path).expanduser()
        np.savez_compressed(
            output_path,
            **{
                "A_indices": np.row_stack([self.K.row, self.K.col]),
                "A_values": self.K.data,
                "x": self.u,
                "b": self.f,
            },
        )

    def sha1(self) -> str:
        """Represent the static analysis problem with a unique hash.

        Returns:
            str: Concatenation of the SHA1 hash of the stiffness matrix and
                of the SHA1 hash of the constant term and the solution.
        """
        sha = hashlib.sha1()
        sha.update(np.concatenate([self.K.row, self.K.col, self.K.data]))
        matrix_hash = sha.hexdigest()
        sha.update(np.concatenate([self.u, self.f]))
        problem_hash = sha.hexdigest()
        return f"{matrix_hash}_{problem_hash}"
