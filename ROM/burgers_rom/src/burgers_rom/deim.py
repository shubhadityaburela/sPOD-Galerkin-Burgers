"""Standard (non-shifted) DEIM: SVD-based basis construction and truncation."""

from dataclasses import dataclass


import numpy as np


@dataclass
class DEIMBasis:
    """Thin-SVD decomposition of a flux snapshot matrix F (n_faces, n_snapshots) = modes @ diag(singular_values) @ coeffs.

    modes: DEIM modes (spatial basis), shape (n_faces, n_modes)
    singular_values: singular values in descending order, shape (n_modes,)
    coeffs: temporal coefficients, shape (n_modes, n_snapshots)
    """

    modes: np.ndarray
    singular_values: np.ndarray
    coeffs: np.ndarray

    def truncate(self, r: int) -> "DEIMBasis":
        """Keep only the first r modes."""
        return DEIMBasis(self.modes[:, :r], self.singular_values[:r], self.coeffs[:r, :])

    def normalized_singular_values(self) -> np.ndarray:
        """Singular values scaled by sigma_0, so the decay is comparable across snapshot sets."""
        return self.singular_values / self.singular_values[0]

    def energy_content(self) -> np.ndarray:
        """Cumulative fraction of total singular-value-squared energy captured by the first r modes."""
        energy = self.singular_values**2
        return np.cumsum(energy) / np.sum(energy)

    def modes_for_energy(self, fraction: float) -> int:
        """Smallest number of modes whose cumulative energy reaches `fraction` (e.g. 0.99)."""
        return int(np.searchsorted(self.energy_content(), fraction) + 1)

