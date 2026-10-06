"""Explicit finite-volume time-stepping solver for the 1D Burgers' equation."""

from dataclasses import dataclass

import numpy as np

from burgers1d.boundary_conditions import BOUNDARY_CONDITIONS
from burgers1d.flux import FLUXES
from burgers1d.grid import Grid
from burgers1d.reconstruction import muscl_interface_states

SPATIAL_ORDERS = ("first", "second")
TIME_INTEGRATORS = ("forward_euler", "ssp_rk2")


@dataclass
class BurgersSolver:
    """Solves u_t + (u^2/2)_x = 0 on a uniform grid via finite volumes.

    Two independent choices control accuracy:
      - ``spatial_order``: "first" reads raw cell values at each interface;
        "second" reconstructs a piecewise-linear, MC-limited (TVD) profile
        per cell first (see burgers1d.reconstruction).
      - ``time_integrator``: "forward_euler" is a single-stage explicit step;
        "ssp_rk2" is the 2-stage strong-stability-preserving Runge-Kutta
        method, which preserves the TVD property of the spatial scheme.

    Either axis can be combined with any numerical flux (see burgers1d.flux.FLUXES),
    which is used as the Riemann solver at each (possibly reconstructed) interface.
    """

    grid: Grid
    cfl: float = 0.5
    boundary: str = "periodic"
    flux: str = "lax_friedrichs"
    spatial_order: str = "first"
    time_integrator: str = "forward_euler"

    def __post_init__(self) -> None:
        if self.boundary not in BOUNDARY_CONDITIONS:
            raise ValueError(f"Unknown boundary condition '{self.boundary}'. "
                              f"Choose one of {list(BOUNDARY_CONDITIONS)}.")
        if self.flux not in FLUXES:
            raise ValueError(f"Unknown flux '{self.flux}'. Choose one of {list(FLUXES)}.")
        if self.spatial_order not in SPATIAL_ORDERS:
            raise ValueError(f"Unknown spatial order '{self.spatial_order}'. "
                              f"Choose one of {list(SPATIAL_ORDERS)}.")
        if self.time_integrator not in TIME_INTEGRATORS:
            raise ValueError(f"Unknown time integrator '{self.time_integrator}'. "
                              f"Choose one of {list(TIME_INTEGRATORS)}.")

    def time_step(self, u: np.ndarray) -> float:
        """Largest stable dt for the current state, from the CFL condition."""
        max_speed = np.max(np.abs(u))
        if max_speed == 0.0:
            return np.inf
        return self.cfl * self.grid.dx / max_speed

    def rhs(self, u: np.ndarray, dt: float) -> np.ndarray:
        """Spatial update: -1/dx * (F_{i+1/2} - F_{i-1/2}) for every cell."""
        apply_bc = BOUNDARY_CONDITIONS[self.boundary]

        if self.spatial_order == "first":
            u_padded = apply_bc(u, n_ghost=1)
            u_left = u_padded[:-1]
            u_right = u_padded[1:]
        else:
            u_padded = apply_bc(u, n_ghost=2)
            u_left, u_right = muscl_interface_states(u_padded)

        fluxes = FLUXES[self.flux](u_left, u_right, self.grid.dx, dt)

        return -(fluxes[1:] - fluxes[:-1]) / self.grid.dx

    def step(self, u: np.ndarray, dt: float) -> np.ndarray:
        """Advance the state by one step of size dt with the chosen time integrator."""
        if self.time_integrator == "forward_euler":
            return u + dt * self.rhs(u, dt)

        # ssp_rk2 (Heun / trapezoidal): a convex combination of forward-Euler
        # stages, which is what keeps it strong-stability-preserving (TVD)
        # whenever the underlying spatial scheme is.
        u1 = u + dt * self.rhs(u, dt)
        return 0.5 * u + 0.5 * (u1 + dt * self.rhs(u1, dt))

    def solve(self, u0: np.ndarray, t_end: float) -> tuple[np.ndarray, float]:
        """Integrate from u0 at t=0 until t_end. Returns (u_final, t_reached)."""
        u = u0.copy()
        t = 0.0
        while t < t_end:
            dt = min(self.time_step(u), t_end - t)
            u = self.step(u, dt)
            t += dt
        return u, t
