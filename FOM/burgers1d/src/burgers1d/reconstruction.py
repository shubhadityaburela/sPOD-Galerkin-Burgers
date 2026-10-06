"""Piecewise-linear MUSCL reconstruction with a TVD slope limiter."""

import numpy as np


def _minmod3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Elementwise minmod of three arrays.

    Returns the smallest-magnitude value when all three share a sign, else 0.
    This is what keeps a limited slope from creating a new local extremum.
    """
    same_sign = (np.sign(a) == np.sign(b)) & (np.sign(b) == np.sign(c))
    magnitude = np.minimum(np.minimum(np.abs(a), np.abs(b)), np.abs(c))
    return np.where(same_sign, np.sign(a) * magnitude, 0.0)


def mc_slopes(u_padded: np.ndarray) -> np.ndarray:
    """Monotonized-central (MC) limited slope for every cell in ``u_padded[1:-1]``.

    ``u_padded`` must carry at least 2 ghost cells on each side. The MC limiter
    is the minmod of the one-sided slopes doubled and their central average,
    which is more compressive than minmod while remaining TVD:

        slope = minmod(2*(u_i - u_{i-1}), (u_{i+1} - u_{i-1}) / 2, 2*(u_{i+1} - u_i))
    """
    d_minus = u_padded[1:-1] - u_padded[:-2]
    d_plus = u_padded[2:] - u_padded[1:-1]
    return _minmod3(2.0 * d_minus, 0.5 * (d_minus + d_plus), 2.0 * d_plus)


def muscl_interface_states(u_padded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Limited linear reconstruction of the left/right state at every cell face.

    ``u_padded`` must carry 2 ghost cells on each side (as produced by a
    boundary condition called with ``n_ghost=2``). Returns ``(u_left, u_right)``,
    one entry per face from x_min to x_max, in the same layout the first-order
    stencil produces from a single-ghost padding.
    """
    slopes = mc_slopes(u_padded)
    cells = u_padded[1:-1]

    right_edge = cells + 0.5 * slopes
    left_edge = cells - 0.5 * slopes

    u_left = right_edge[:-1]
    u_right = left_edge[1:]
    return u_left, u_right
