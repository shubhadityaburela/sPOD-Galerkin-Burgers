"""Uniform 1D grid on which the Burgers' equation is discretized."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """A uniform 1D finite-volume grid on [x_min, x_max] with n_cells cells."""

    x_min: float
    x_max: float
    n_cells: int

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.n_cells

    @property
    def cell_centers(self) -> np.ndarray:
        return self.x_min + (np.arange(self.n_cells) + 0.5) * self.dx
