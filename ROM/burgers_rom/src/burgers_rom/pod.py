"""Standard (non-shifted) POD: SVD-based basis construction and truncation."""

from dataclasses import dataclass

import numpy as np


@dataclass
class PODBasis:
    """Thin-SVD decomposition of a snapshot matrix X (n_dof, n_snapshots) = modes @ diag(singular_values) @ coeffs.

    modes: POD modes (spatial basis), shape (n_dof, n_modes)
    singular_values: singular values in descending order, shape (n_modes,)
    coeffs: temporal coefficients, shape (n_modes, n_snapshots)
    """

    modes: np.ndarray
    singular_values: np.ndarray
    coeffs: np.ndarray

    def truncate(self, r: int) -> "PODBasis":
        """Keep only the first r modes."""
        return PODBasis(self.modes[:, :r], self.singular_values[:r], self.coeffs[:r, :])

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


def compute_pod_basis(snapshot_matrix: np.ndarray) -> PODBasis:
    """Thin SVD of a snapshot matrix with one snapshot per column."""
    modes, singular_values, coeffs = np.linalg.svd(snapshot_matrix, full_matrices=False)
    return PODBasis(modes, singular_values, coeffs)
