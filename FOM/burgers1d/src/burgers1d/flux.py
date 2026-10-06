"""Numerical flux functions for the Burgers' equation f(u) = u^2 / 2."""

import numpy as np


def physical_flux(u: np.ndarray) -> np.ndarray:
    return 0.5 * u**2


def lax_friedrichs(u_left: np.ndarray, u_right: np.ndarray, dx: float, dt: float) -> np.ndarray:
    """Global Lax-Friedrichs numerical flux at a cell interface (more diffusive than LLF and Godunov).

    F(u_l, u_r) = 0.5 * (f(u_l) + f(u_r)) - 0.5 * (dx/dt) * (u_r - u_l)
    """
    alpha = dx / dt
    return 0.5 * (physical_flux(u_left) + physical_flux(u_right)) - 0.5 * alpha * (u_right - u_left)


def godunov(u_left: np.ndarray, u_right: np.ndarray, dx: float | None = None, dt: float | None = None) -> np.ndarray:
    """Godunov (exact Riemann solver) numerical flux at a cell interface.

    f(u) = u^2/2 is convex, so the exact Riemann solution reduces to:
      - shock (u_l >= u_r):     F = max(f(u_l), f(u_r))
      - rarefaction (u_l < u_r): F = f(u_l) if u_l >= 0
                                     f(u_r) if u_r <= 0
                                     0      otherwise (sonic point in the fan)
    dx and dt are accepted but unused, to keep a uniform flux signature.
    """
    f_left = physical_flux(u_left)
    f_right = physical_flux(u_right)

    shock = np.maximum(f_left, f_right)
    rarefaction = np.where(u_left >= 0, f_left, np.where(u_right <= 0, f_right, 0.0))

    return np.where(u_left >= u_right, shock, rarefaction)


FLUXES = {
    "lax_friedrichs": lax_friedrichs,
    "godunov": godunov,
}
