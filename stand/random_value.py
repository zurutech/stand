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
"""Class to sample random values uniformly from an interval of integers or reals."""

import numbers
from typing import Optional, Union, Sequence, Tuple

import numpy as np


class RandomValue:
    """Sample random values uniformly from an interval of integers or reals."""

    def __init__(
        self,
        value: Union[int, float, Tuple[int, int], Tuple[float, float]],
    ):
        """Build a RandomValue.

        Args:
            value (Union[int, float, Tuple[int, int], Tuple[float, float]]):
                Interval boundaries or a single number if the interval is a single value.
                If both boundaries are integers the sampled values are integers,
                otherwise the sampled values are decimal numbers.
        """
        self.value = value
        if isinstance(value, numbers.Real):
            self.value_type = "value"
        else:
            try:
                real_range = isinstance(value[0], numbers.Real) and isinstance(
                    value[1], numbers.Real
                )
            except TypeError:
                raise (f"The provided range of values {value} is not valid")
            if not real_range:
                raise (f"The provided range of values {value} is not valid")
            integer_range = isinstance(value[0], numbers.Integral) and isinstance(
                value[1], numbers.Integral
            )
            self.value_type = "int_range" if integer_range else "float_range"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value})"

    def get(
        self,
        size: Optional[Sequence[int]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Union[int, float, np.ndarray]:
        """Sample a value in the interval at build time.

        Args:
            size (Optional[Sequence[int]], optional): Size of the array of sampled values.
                Defaults to None, in that case a single number is sampled.
            rng (Optional[np.random.Generator], optional): Random generator.
                Defaults to None, in that case the default random generator is used.

        Raises:
            ValueError: If self.value_type is not "value", "int_range" or "float_range".

        Returns:
            Union[int, float, np.ndarray]: Sampled value or array of sampled values.
        """
        rng = np.random.default_rng() if rng is None else rng
        if self.value_type == "value":
            if size is None:
                value = self.value
            else:
                value = np.full(size, self.value)
        elif self.value_type == "int_range":
            value = rng.integers(self.value[0], self.value[1], size=size)
        elif self.value_type == "float_range":
            value = rng.uniform(self.value[0], self.value[1], size=size)
        else:
            raise ValueError(f"The value_type {self.value_type} is not valid")
        return value
