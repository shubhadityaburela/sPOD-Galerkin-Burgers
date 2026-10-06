"""FOM-vs-ROM reconstruction and prediction error metrics."""

import numpy as np


def relative_l2_error(u_fom: np.ndarray, u_rom: np.ndarray) -> float:
    """Relative L2 error between two same-shaped fields."""
    return np.linalg.norm(u_fom - u_rom) / np.linalg.norm(u_fom)


def relative_l2_error_over_time(U_fom: np.ndarray, U_rom: np.ndarray) -> np.ndarray:
    """Relative L2 error at each time instance for trajectories shaped (n, m).

    One snapshot per column, so the norm runs over axis 0 (space) and the
    result has one entry per time instance.
    """
    numerator = np.linalg.norm(U_fom - U_rom, axis=0)
    denominator = np.linalg.norm(U_fom, axis=0)
    return numerator / denominator


def frobenius_relative_error(Q: np.ndarray, Q_tilde: np.ndarray) -> float:
    """||Q - Q~||_F / ||Q||_F over the whole trajectory matrix at once.

    Q, Q_tilde: (n, m) snapshot matrices, one snapshot per column.
    np.linalg.norm on a 2D array with no `ord` given is the Frobenius norm.
    """
    return np.linalg.norm(Q - Q_tilde) / np.linalg.norm(Q)
