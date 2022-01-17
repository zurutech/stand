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
"""Dataclass containing beam parameters."""

from dataclasses import dataclass
from functools import cached_property

import scipy.constants


@dataclass(frozen=True)
class Beam:
    """
    Describe a beam and its parameters.

    Object Attributes:
        width (float): Width of the rectangular section of the beam (m).
        height (float): Height of the rectangular section of the beam (m).
        young_modulus (float): Young's modulus of beam material (Pa).
        density (float): Density of beam material (kg/m^3).
        nu (float): Poisson's ratio.
    """

    width: float
    height: float
    young_modulus: float = 3.0e10
    density: float = 2.5e3
    nu: float = 0.2

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(f"The provided width {self.width} is not positive")
        if self.height <= 0:
            raise ValueError(f"The provided height {self.height} is not positive")
        if self.young_modulus <= 0:
            raise ValueError(
                f"The provided Young's modulus {self.young_modulus} is not positive"
            )
        if self.density <= 0:
            raise ValueError(f"The provided density {self.density} is not positive")

    @cached_property
    def shear_modulus(self) -> float:
        """Get the shear modulus of the beam.

        Returns:
            float: Shear modulus of the beam.
        """
        return 0.5 * self.young_modulus / (1 + self.nu)

    @cached_property
    def section_x(self) -> float:
        """Get the area of the rectangular cross section.

        Returns:
            float: Area of the section perpendicular to the local x-axis.
        """
        return self.width * self.height

    @cached_property
    def section_y(self) -> float:
        """Get the area of the rectangular section perpendicular to the y-axis.

        Returns:
            float: Area of the section perpendicular to the local y-axis.
        """
        return self.section_x / 1.2

    @cached_property
    def section_z(self) -> float:
        """Get the area of the rectangular section perpendicular to the z-axis.

        Returns:
            float: Area of the section perpendicular to the local z-axis.
        """
        return self.section_x / 1.2

    @cached_property
    def inertia_yy(self) -> float:
        """Get the second moment of inertia about the local y-axis.

        Returns:
            float: Second moment of inertia about the local y-axis.
        """
        return self.width * self.height ** 3 / 12.0

    @cached_property
    def inertia_zz(self) -> float:
        """Get the second moment of inertia about the local z-axis.

        Returns:
            float: Second moment of inertia about the local z-axis.
        """
        return self.width ** 3 * self.height / 12.0

    @cached_property
    def inertia_xx(self) -> float:
        """Get the torsional moment of inertia of the cross section.

        Returns:
            float: Torsional moment of inertia of the cross section.
        """
        return self.inertia_yy + self.inertia_zz

    @cached_property
    def weight_uniform_load(self) -> float:
        """Get the pressure of the dead weight of the beam on the beam itself.

        Returns:
            float: Pressure of the dead weight of the beam on the beam itself (N/m).
        """
        return scipy.constants.g * self.density * self.section_x
