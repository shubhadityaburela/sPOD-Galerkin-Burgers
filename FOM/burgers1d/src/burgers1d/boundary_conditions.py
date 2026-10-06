"""Boundary condition strategies.

Each function takes the interior state ``u`` and returns a padded array with
``n_ghost`` ghost cells on each side. The first-order stencil needs one ghost
cell per side; the second-order MUSCL reconstruction needs two, so it can
compute a limited slope for the outermost real cells as well.
"""

import numpy as np


def periodic(u: np.ndarray, n_ghost: int = 1) -> np.ndarray:
    return np.concatenate((u[-n_ghost:], u, u[:n_ghost]))


def dirichlet(u: np.ndarray, u_left: float = 0.0, u_right: float = 0.0, n_ghost: int = 1) -> np.ndarray:
    return np.concatenate((np.full(n_ghost, u_left), u, np.full(n_ghost, u_right)))


def outflow(u: np.ndarray, n_ghost: int = 1) -> np.ndarray:
    """Zero-gradient (transmissive) boundaries."""
    return np.concatenate((np.full(n_ghost, u[0]), u, np.full(n_ghost, u[-1])))


BOUNDARY_CONDITIONS = {
    "periodic": periodic,
    "dirichlet": dirichlet,
    "outflow": outflow,
}
