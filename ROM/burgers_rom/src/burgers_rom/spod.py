"""Shifted POD (co-moving frame) basis construction, and the shifted-mode
lookup tables (V, W) the sPOD-Galerkin online stage needs.

compute_spod_basis builds the open-spline coefficient matrices (A1, D1, D2,
R) for the grid exactly once and stores them on the returned sPODBasis
alongside Nx/dx, so nothing downstream (make_V_W, or a fresh call to
shift_snapshots) has to rebuild them or thread them through by hand.
"""

from dataclasses import dataclass

import numpy as np

from burgers_rom.transforms import (
    construct_open_spline_coeffs_multiple,
    first_derivative_shifted_U_open,
    give_open_spline_coefficient_matrices,
    shift_open_matrix_precomputed_coeffs_multiple,
    shifted_U_open,
)

DEFAULT_NUM_SAMPLE = 10


@dataclass
class sPODBasis:
    """Thin-SVD decomposition of the co-moving-frame snapshot matrix, plus
    the grid/spline bookkeeping make_V_W needs.

    modes, singular_values, coeffs: as PODBasis, but modes live in the
        co-moving frame (compute_spod_basis's de-shifted snapshot matrix),
        not the lab frame.
    Nx, dx: grid size/spacing the basis was built on.
    A1, D1, D2: open/natural spline coefficient matrices for this grid
        (give_open_spline_coefficient_matrices(Nx)), built once by
        compute_spod_basis and carried along rather than rebuilt.
    R: LU factorization (scipy.linalg.lu_factor's (lu, piv) pair) of the
        natural-spline system matrix M, from the same call.
    V, W: shifted-mode / shift-derivative lookup tables, filled in by
        make_V_W (None until then).
    """

    modes: np.ndarray
    singular_values: np.ndarray
    coeffs: np.ndarray
    Nx: int
    dx: float
    A1: np.ndarray
    D1: np.ndarray
    D2: np.ndarray
    R: tuple[np.ndarray, np.ndarray]
    shift: np.ndarray
    n_modes: int | None = None
    V: np.ndarray | None = None
    W: np.ndarray | None = None


    def truncate(self, r: int) -> "sPODBasis":
        """Keep only the first r modes; carries the grid/spline bookkeeping along."""
        return sPODBasis(self.modes[:, :r], self.singular_values[:r], self.coeffs[:r, :],
                          self.Nx, self.dx, self.A1, self.D1, self.D2, self.R, self.shift, r)

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

    def make_V_W(self, delta_s: np.ndarray) -> None:
        """Tabulate the (truncated) modes shifted by each delta_s[it] (V) and
        minus their derivative w.r.t. the shift amount (W) -- the lookup the
        sPOD-Galerkin online stage indexes into instead of re-solving the
        spline shift on every rhs call (see spod_galerkin.py).

        Sets V, W in place, each shape (len(delta_s), Nx, n_modes).
        """
        n_modes = self.modes.shape[1]
        self.V = np.empty((len(delta_s), self.Nx, n_modes))
        self.W = np.empty((len(delta_s), self.Nx, n_modes))

        b, c, d = construct_open_spline_coeffs_multiple(self.modes, self.A1, self.D1, self.D2, self.R, self.dx)
        for it, s in enumerate(delta_s):
            self.V[it] = shifted_U_open(self.modes, s, b, c, d, self.Nx, self.dx)
            self.W[it] = -first_derivative_shifted_U_open(s, b, c, d, self.Nx, self.dx)



def compute_shifts(snapshot_matrix: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute the shift amount for each snapshot (e.g., based on the location of the maximum value)."""
    return -np.array([x[np.argmin(np.diff(snapshot_matrix[:, i]))] - x[np.argmin(np.diff(snapshot_matrix[:, 0]))]
                    for i in range(snapshot_matrix.shape[1])])


def shift_snapshots(snapshot_matrix: np.ndarray, x: np.ndarray, cs: np.ndarray,
                     spline_matrices: tuple | None = None) -> np.ndarray:
    """Shift a snapshot matrix to a co-moving frame using cubic spline interpolation.

    Inputs:
    - snapshot_matrix: array of size mxn where m is the number of grid points and n the number of snapshots
    - x: array of size m containing the spatial grid points
    - cs: array of size n containing the shift amount for each snapshot
    - spline_matrices: optional pre-built (A1, D1, D2, R) from
      give_open_spline_coefficient_matrices(m) -- pass an sPODBasis's stored
      matrices here to avoid rebuilding them; built fresh if omitted.

    Outputs:
    - shifted_snapshots: shifted version of original snapshot matrix (array of size mxn)
    """

    Nx = snapshot_matrix.shape[0]
    dx = x[1] - x[0]

    if spline_matrices is None:
        spline_matrices = give_open_spline_coefficient_matrices(Nx)
    A1, D1, D2, R = spline_matrices

    spline_coeffs_b, spline_coeffs_c, spline_coeffs_d = construct_open_spline_coeffs_multiple(
        snapshot_matrix, A1, D1, D2, R, dx)
    shifted_snapshots = shift_open_matrix_precomputed_coeffs_multiple(
        snapshot_matrix, cs, spline_coeffs_b, spline_coeffs_c, spline_coeffs_d, Nx, dx)

    return shifted_snapshots


def compute_spod_basis(snapshot_matrix: np.ndarray, x: np.ndarray) -> sPODBasis:
    """Co-moving-frame POD: de-shift every snapshot (compute_shifts +
    shift_snapshots), then an ordinary thin SVD of the result.
    """
    Nx = snapshot_matrix.shape[0]
    dx = float(x[1] - x[0])
    spline_matrices = give_open_spline_coefficient_matrices(Nx)

    shift = compute_shifts(snapshot_matrix, x)
    shifted_snapshots = shift_snapshots(snapshot_matrix, x, shift, spline_matrices)

    modes, singular_values, coeffs = np.linalg.svd(shifted_snapshots, full_matrices=False)
    A1, D1, D2, R = spline_matrices
    return sPODBasis(modes, singular_values, coeffs, Nx, dx, A1, D1, D2, R, shift)


def subsample(x: np.ndarray, num_samples: int = DEFAULT_NUM_SAMPLE) -> np.ndarray:
    """Subsample the spatial grid points, for use as make_V_W's delta_s lookup grid."""
    return np.linspace(x[0], x[-1], num_samples)


def findIntervalAndGiveInterpolationWeight_1D(xPoints: np.ndarray, xStar: float) -> tuple[int, float]:
    """Bracketing interval and linear-interpolation weight for xStar in xPoints.

    Returns (intervalIdx, alpha) such that xStar ~= alpha*xPoints[intervalIdx]
    + (1-alpha)*xPoints[intervalIdx+1] -- callers use alpha the same way to
    blend the matching table rows (e.g. V[intervalIdx], V[intervalIdx+1]).

    xStar outside [xPoints[0], xPoints[-1]] is CLAMPED to the nearest edge
    (alpha pinned to 1 or 0) rather than extrapolated: intervalIdx alone was
    already being clamped to the boundary interval, but alpha was not, so a
    query even slightly out of range produced a linearly extrapolated (and
    unboundedly worse the further out) weight -- e.g. xStar = 2.0 against a
    table spanning [0, 1] previously returned alpha = -9.0, not a clamp to
    the closest table entry. That fed directly into matrices_sPOD_galerkin_online
    and sPODROMSolver.project/reconstruct, whose queried shift is not
    guaranteed to stay in-range.
    """
    intervalIdx = np.searchsorted(xPoints, xStar) - 1
    intervalIdx = max(0, min(intervalIdx, len(xPoints) - 2))
    x1, x2 = xPoints[intervalIdx], xPoints[intervalIdx + 1]
    alpha = (x2 - xStar) / (x2 - x1)
    alpha = max(0.0, min(1.0, alpha))

    return intervalIdx, alpha # type: ignore

 