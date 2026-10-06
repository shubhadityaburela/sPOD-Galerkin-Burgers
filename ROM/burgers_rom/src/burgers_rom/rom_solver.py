"""Time integration of the reduced-order system.

Only needs the grid spacing dx (a scalar, read off the snapshot grid once) --
no Grid, no BurgersSolver instance. The spatial operator is either the plain
Galerkin projection of FOM/burgers1d's operator (galerkin.py) or, when a
DEIMOperator is supplied, its hyper-reduced surrogate (hyperreduction.py).
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from burgers_rom.galerkin import PHASE_CONDITIONS, reduced_rhs_POD, reduced_rhs_sPOD
from burgers_rom.hyperreduction import DEIMOperator
from burgers_rom.spod import findIntervalAndGiveInterpolationWeight_1D, sPODBasis
from burgers_rom.spod_hyperreduction import SPODDEIMOperator

TIME_INTEGRATORS = ("forward_euler", "ssp_rk2")


@dataclass
class PODROMSolver:
    """Time-steps the reduced system da/dt = rhs(a).

    With deim=None (default), rhs is the plain Galerkin projection
    modes.T @ L(modes @ a) -- exact but O(n_dof) per evaluation. Passing a
    DEIMOperator switches rhs to its hyper-reduced counterpart, which touches
    only the sampled stencil cells and faces.
    """

    modes: np.ndarray            # POD basis, shape (n_dof, n_modes)
    dx: float
    cfl: float = 0.5
    time_integrator: str = "ssp_rk2"
    deim: DEIMOperator | None = None

    def __post_init__(self) -> None:
        if self.time_integrator not in TIME_INTEGRATORS:
            raise ValueError(f"Unknown time integrator '{self.time_integrator}'. "
                              f"Choose one of {list(TIME_INTEGRATORS)}.")

    def project(self, u: np.ndarray) -> np.ndarray:
        """Reduced coordinates of a full-order state: a = modes.T @ u."""
        return self.modes.T @ u

    def reconstruct(self, a: np.ndarray) -> np.ndarray:
        """Full-order state from reduced coordinates: u = modes @ a."""
        return self.modes @ a

    def time_step(self, a: np.ndarray) -> float:
        """Largest stable dt, from the CFL condition on the reconstructed field.

        Note: this reconstructs the full n_dof field, so it stays O(n_dof) even
        in DEIM mode -- an acceptable start, but replace with an estimate from
        the sampled stencil cells if the CFL check ever dominates the runtime.
        """
        speed = np.max(np.abs(self.reconstruct(a)))
        if speed == 0.0:
            return np.inf
        return self.cfl * self.dx / speed

    def rhs(self, a: np.ndarray) -> np.ndarray:
        if self.deim is not None:
            return self.deim.reduced_rhs(a)
        return reduced_rhs_POD(a, self.modes, self.dx)

    def step(self, a: np.ndarray, dt: float) -> np.ndarray:
        """Advance the reduced state by one step of size dt."""
        if self.time_integrator == "forward_euler":
            return a + dt * self.rhs(a)

        a1 = a + dt * self.rhs(a)
        return 0.5 * a + 0.5 * (a1 + dt * self.rhs(a1))

    def solve(self, a0: np.ndarray, t_end: float) -> tuple[np.ndarray, float]:
        """Integrate from a0 at t=0 until t_end. Returns (a_final, t_reached)."""
        a = a0.copy()
        t = 0.0
        while t < t_end:
            dt = min(self.time_step(a), t_end - t)
            a = self.step(a, dt)
            t += dt
        return a, t


@dataclass
class sPODROMSolver:
    """Time-steps the augmented sPOD-Galerkin reduced system d/dt[a; z] = rhs(a, z).

    Unlike PODROMSolver (and unlike a fixed-velocity co-moving frame), the
    frame shift z is not prescribed -- it's part of the reduced state, tracked
    alongside the mode coefficients a and evolved by the SAME Galerkin
    projection (see galerkin.py's module docstring for the [a; z]
    normal-equations derivation). The lab-frame reconstruction at shift z is
    V(z) = T^{-z}[basis.modes], approximated by linearly interpolating the
    offline lookup tables basis.V/basis.W (built at the sample shifts
    delta_s, see spod.sPODBasis.make_V_W) rather than re-solving the spline
    shift on every call.

    Every method that touches the reduced state takes (or returns) the state
    as (a, z) or the concatenated az = [a; z] rather than just a, since z is
    now a genuine dynamical variable, not a fixed function of t.
    """

    basis: sPODBasis                         # truncated sPOD basis (co-moving modes + V/W lookup tables)
    lhs_matrices: np.ndarray                 # offline Gram-matrix tables (V^T V, V^T W, W^T W) at each delta_s
    delta_s: np.ndarray                      # shift values the V/W/lhs_matrices lookup tables are sampled at
    cfl: float = 0.5
    time_integrator: str = "ssp_rk2"
    deim: "SPODDEIMOperator | None" = None    # burgers_rom.spod_hyperreduction.SPODDEIMOperator
    precondition: bool = False                # Jacobi-precondition the augmented (a,z) mass matrix each rhs call (galerkin.solve_lin_system)
    phase_condition: str = "residual"         # "residual" (Psi_Res), "freeze", "freeze_tilde", "template", or "freeze_tilde_weighted" -- see galerkin.py
    template_q: np.ndarray | None = None      # fixed test direction, required iff phase_condition == "template" -- galerkin.compute_template_direction
    sigma: np.ndarray | None = None           # weighted-inner-product matrix, required iff phase_condition == "freeze_tilde_weighted" -- galerkin.compute_energy_weighted_sigma

    def __post_init__(self) -> None:
        if self.time_integrator not in TIME_INTEGRATORS:
            raise ValueError(f"Unknown time integrator '{self.time_integrator}'. "
                              f"Choose one of {list(TIME_INTEGRATORS)}.")
        if self.phase_condition not in PHASE_CONDITIONS:
            raise ValueError(f"Unknown phase_condition '{self.phase_condition}'. "
                              f"Choose one of {list(PHASE_CONDITIONS)}.")
        if self.phase_condition == "template" and self.template_q is None:
            raise ValueError("phase_condition='template' requires template_q -- see galerkin.compute_template_direction.")
        if self.phase_condition == "freeze_tilde_weighted" and self.sigma is None:
            raise ValueError("phase_condition='freeze_tilde_weighted' requires sigma -- "
                              "see galerkin.compute_energy_weighted_sigma.")

    def project(self, u: np.ndarray, shift: float) -> np.ndarray:
        """Reduced coordinates a from a lab-frame state u at a known shift:
        solve V(shift) @ a ~= u in the least-squares sense, a = pinv(V) @ u.

        V(shift) is basis.V/basis.W's lookup table (sPODBasis.make_V_W)
        linearly interpolated to `shift` -- unlike the raw SVD modes it is
        generally NOT orthonormal (cubic-spline shifting doesn't preserve
        orthonormality), so a plain V.T @ u is not the correct projector.
        pinv(V): pinv(V) @ V = I for full-column-rank V, so this is the
        direct analogue of PODROMSolver.project for a non-orthonormal basis.
        Used once, to turn the initial FOM snapshot into a starting (a, z).
        """
        if self.basis.V is None:
            raise ValueError("basis.V is not set -- call basis.make_V_W(delta_s) before project().")

        intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(self.delta_s, -shift)
        V = np.add(weight * self.basis.V[intervalIdx], (1 - weight) * self.basis.V[intervalIdx + 1])

        return np.linalg.pinv(V) @ u

    def reconstruct(self, a: np.ndarray, shift: float) -> np.ndarray:
        """Lab-frame state from reduced coordinates at a given shift: u = V(shift) @ a.

        V(shift) is the same table-interpolated, shift-dependent basis as
        project() -- this IS the lab-frame reconstruction (unlike
        PODROMSolver.reconstruct's fixed basis, there is no separate
        co-moving-frame-only step here, since V already bakes the current
        shift in).
        """
        if self.basis.V is None:
            raise ValueError("basis.V is not set -- call basis.make_V_W(delta_s) before reconstruct().")

        intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(self.delta_s, -shift)
        V = np.add(weight * self.basis.V[intervalIdx], (1 - weight) * self.basis.V[intervalIdx + 1])

        return V @ a

    def time_step(self, a: np.ndarray, z: float) -> float:
        """Largest stable dt, from the CFL condition on the reconstructed field.

        Note: this reconstructs the full n_dof field, so it stays O(n_dof) even
        in DEIM mode -- an acceptable start, but replace with an estimate from
        the sampled stencil cells if the CFL check ever dominates the runtime.
        """
        speed = np.max(np.abs(self.reconstruct(a, z)))
        if speed == 0.0:
            return np.inf
        return self.cfl * self.basis.dx / speed

    def rhs(self, az: np.ndarray) -> np.ndarray:
        """d/dt[a; z] as one (n_modes+1,) vector -- unpacks az into (a, z) and
        delegates to galerkin.reduced_rhs_sPOD, which solves the Galerkin
        normal equations for da/dt and dz/dt jointly (see galerkin.py).

        With deim=None (default), the nonlinearity is the full O(n_dof)
        evaluation. Passing a SPODDEIMOperator switches T1/T2 (the two
        projections of the nonlinearity M's mass-matrix assembly does NOT
        need) to their hyper-reduced, O(n_dof)-independent counterpart --
        see spod_hyperreduction.py.
        """
        a = az[:-1]
        z = az[-1]
        return reduced_rhs_sPOD(self.lhs_matrices, self.basis.V, self.basis.W, a, z, self.delta_s, self.basis.n_modes,
                                 self.basis.dx, precondition=self.precondition, phase_condition=self.phase_condition,
                                 sdeim=self.deim, template_q=self.template_q, sigma=self.sigma)

    def step(self, az: np.ndarray, dt: float) -> np.ndarray:
        """Advance the augmented state az = [a; z] by one step of size dt."""
        if self.time_integrator == "forward_euler":
            return az + dt * self.rhs(az)

        a1 = az + dt * self.rhs(az)
        return 0.5 * az + 0.5 * (a1 + dt * self.rhs(a1))

    def solve(self, a0: np.ndarray, z0: float, t_end: float) -> tuple[np.ndarray, float, float]:
        """Integrate (a, z) from (a0, z0) at t=0 until t_end.

        Returns (a_final, z_final, t_reached) -- z_final is the ROM's own
        online estimate of the frame shift at t_reached, distinct from (and
        not required to match) basis.shift's offline, training-data-derived
        trajectory.
        """
        a = a0.copy()
        z = z0
        az = np.concatenate((a, [z]))
        t = 0.0
        while t < t_end:
            dt = min(self.time_step(a, z), t_end - t)
            az = self.step(az, dt)
            a, z = az[:-1], az[-1]  # keep the CFL check current -- it was using the t=0 state every iteration
            t += dt
        return az[:-1], az[-1], t
