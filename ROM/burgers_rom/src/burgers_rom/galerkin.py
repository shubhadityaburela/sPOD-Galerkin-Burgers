"""Galerkin projection of u_t + (u^2/2)_x = 0 onto a reduced basis."""

from burgers_rom.spod import findIntervalAndGiveInterpolationWeight_1D, sPODBasis
import numpy as np
import opt_einsum as oe

from burgers1d.boundary_conditions import outflow
from burgers1d.flux import godunov
from burgers1d.reconstruction import muscl_interface_states


def burgers_rhs(u: np.ndarray, dx: float) -> np.ndarray:
    """L(u) for outflow BCs, MC-limited MUSCL reconstruction, and the Godunov flux.

    Calls FOM/burgers1d's own flux and reconstruction functions directly rather
    than a hand-synced copy, so this always matches the discretization that
    generated the training snapshots by construction. Those functions are
    already local/stencil-based -- they operate on whatever array they're
    given, not "the whole domain" specifically -- which is also what keeps
    them reusable for a future hyper-reduced (DEIM/EIM) evaluator that would
    call them on small windows instead of the full state (see hyperreduction.py).
    """
    u_padded = outflow(u, n_ghost=2)
    u_left, u_right = muscl_interface_states(u_padded)
    fluxes = godunov(u_left, u_right)
    return -(fluxes[1:] - fluxes[:-1]) / dx


def reduced_rhs_POD(a: np.ndarray, modes: np.ndarray, dx: float) -> np.ndarray:
    """Project the full-order spatial operator onto the reduced basis.

    Reconstructs u = modes @ a, evaluates the *unmodified* full-order right-hand
    side L(u) = burgers_rhs(u, dx), and projects it back with modes.T (valid
    since POD modes are orthonormal, modes.T @ modes = I):

        da/dt = modes.T @ L(modes @ a)

    No hyper-reduction: burgers_rhs still touches every one of the n_dof cells
    on every call, so this costs the same O(n_dof) as the FOM's rhs regardless
    of how small the reduced dimension r is. Reducing the *state* to r does not
    by itself reduce the *work* -- that only happens once burgers_rhs is
    replaced by a DEIM/EIM-evaluated surrogate (see hyperreduction.py).
    """
    u = modes @ a
    return modes.T @ burgers_rhs(u, dx)


r"""sPOD-Galerkin with an online-adapted frame shift.

Unlike a fixed-velocity co-moving frame (a single precomputed s(t)), the
reduced state here is the AUGMENTED vector [a; z]: the usual mode
coefficients a *plus* the frame shift z itself, both evolved jointly. The
ansatz is

    u(x, t) ~= V(z(t)) @ a(t),      V(z) := T^{-z}[Phi]

where Phi = basis.modes are the co-moving (stationary-shock) sPOD modes and
T^{-z} shifts them forward by z to reconstruct the lab-frame field -- see
transforms.shifted_U_open. Differentiating in time:

    du/dt ~= V(z) da/dt + (dV/dz @ a) dz/dt = V(z) da/dt + (W(z) @ a) dz/dt

using dV/dz = W(z) (basis.W is built from first_derivative_shifted_U_open,
which is exactly this derivative -- see spod.py's make_V_W). So at any
instant the tangent space the state can move in is spanned by the l columns
of V(z) plus the single direction W(z) @ a: an (n, l+1) matrix

    Psi(a, z) = [ V(z) | W(z) @ a ]

Requiring the PDE residual du/dt - burgers_rhs(u) to be L2-orthogonal to
Psi's column space (standard Galerkin) gives the normal equations solved
below:

    Psi^T Psi @ [da/dt; dz/dt] = Psi^T burgers_rhs(u)

Psi^T Psi expands into the four blocks built in
matrices_sPOD_galerkin_online:

    [ V^T V         V^T W a  ]     [ V^T r ]
    [ a^T W^T V     a^T W^T W a ]  [ a^T W^T r ]     r := burgers_rhs(V(z) a, dx)

V(z), W(z), and the three Gram matrices V^T V, V^T W, W^T W all depend on
the continuous shift z, but are only ever needed at the discrete lookup
grid basis.delta_s (spod.subsample) -- matrices_sPOD_galerkin tabulates them
there once, offline; the online rhs (matrices_sPOD_galerkin_online) just
linearly interpolates between the two bracketing table entries
(findIntervalAndGiveInterpolationWeight_1D) instead of rebuilding V(z)/W(z)
from the spline machinery on every call.

phase_condition: "residual" vs. "freeze" vs. "freeze_tilde"
-------------------------------------------------------------
The row above for dz/dt (the bottom row of Psi^T Psi) is the "residual"
phase condition Psi_Res: it comes from jointly minimizing the PDE residual
over da/dt AND dz/dt at once (Black/Schulze/Unger, ESAIM:M2AN 54(6):2011-2043,
2020, eq. (6.2b)/"Psi_Res"). It is coupled to da/dt through the cross block
a^T W^T V. When W(z) @ a is nearly redundant with span(V(z)) -- exactly what
was measured on the shock dataset (93-97% of W(z)@a explained by V(z)) --
Psi_Res = Psi_freeze - Psi_freeze_tilde (their eq. (6.13)) becomes a
near-cancellation of two large, nearly-equal terms, which is exactly why the
full (l+1, l+1) system above becomes ill-conditioned as |z| grows.

Table 6.1 of the same paper lists two alternative phase conditions from the
classical Beyn-Thuemmler "freezing"/symmetry-reduction literature (Section 6),
both of which drop the da/dt coupling entirely, so dz/dt becomes a single
well-posed SCALAR equation, solved independently of da/dt (da/dt is then
still solved from the SAME amplitude row as always -- eq. (6.2a) is identical
across all three phase conditions, only the dz/dt row changes):

  "freeze" (eq. 6.11): the first-order optimality condition for
  min_dz/dt [ (1/2)||residual||^2 + (1/2)||v_dot||^2 ], i.e. residual
  matching regularized by how fast the co-moving profile itself changes:

      [a^T W(z)^T W(z) a] dz/dt = a^T W(z)^T r

  "freeze_tilde" (eq. 6.12): the first-order optimality condition for
  min_dz/dt (1/2)||v_dot||^2 ALONE, i.e. choosing dz/dt purely to keep the
  co-moving profile as stationary as possible, ignoring the PDE residual for
  this row entirely:

      ||V(z)^T W(z) a||^2 dz/dt = [V(z)^T W(z) a]^T (V(z)^T r)

  Writing Na := V(z)^T W(z) @ a (already computed below as the M[:modes,
  modes:] block), this is just ||Na||^2 dz/dt = Na^T A[:modes] -- reusing
  Na and A[:modes] directly, no new quantities needed.

Both "freeze" and "freeze_tilde" degrade only if their own scalar divisor
vanishes (||W(z) a||^2 or ||Na||^2 respectively -- the shift direction
having genuinely zero local effect, or zero overlap with V(z)'s span), a
different and much rarer failure than Psi_Res's near-cancellation.

Implemented here by zeroing the M[modes:, :modes] cross block for both
"freeze" and "freeze_tilde": with that block zero, M becomes block upper
triangular in (da/dt, dz/dt), so np.linalg.solve(M, A) automatically
back-solves dz/dt from the now-decoupled bottom row first, then da/dt from
the top row using that dz/dt -- the two-stage solve described above, without
hand-rolling it.

phase_condition: "template"
----------------------------
All three conditions above test the residual against W(z) @ a -- the tangent
direction of the ansatz manifold AT THE CURRENT (a, z), which is why they
degrade in lockstep: that direction is self-referential (quadratic in a,
built from whatever the trajectory currently is), so it can become nearly
redundant with span(V(z)) precisely when the trajectory drifts somewhere
degenerate. "template" (Rowley & Marsden 2000, eq. (2.16)) breaks that
self-reference by testing against a FIXED direction instead, chosen once
offline from a reference state (a0, z0) -- by default the training
trajectory's own t=0 snapshot (see compute_template_direction):

    q := W(z0) @ a0     (fixed, computed once -- never depends on t)

Following the SAME reduce-then-project construction as Psi_Res (project the
ROM's own residual Psi(a,z)[da/dt; dz/dt] - r, r := burgers_rhs(V(z)a, dx),
onto a chosen direction and require the projection to vanish) but with q in
place of the current-state W(z) @ a:

    <q, V(z) da/dt + W(z) a dz/dt - r> = 0
    q^T V(z) da/dt + (q^T W(z) a) dz/dt = q^T r

This is deliberately NOT the same as differentiating a kinematic alignment
constraint <u, q> = const with the RHS set to zero (that alternative, closer
to Rowley's own infinite-dimensional derivation, would drop the q^T r term
entirely and let z track pure shape-alignment with the template, ignoring
the PDE). Keeping q^T r on the right-hand side -- i.e. evaluating the FOM's
own right-hand side at the CURRENT reconstruction, exactly like every other
phase condition here does -- keeps this consistent with Psi_Res/Psi_freeze/
Psi_freeze_tilde's shared philosophy: z should track how the solution is
ACTUALLY evolving, not just stay shape-aligned with a fixed reference.

Unlike the other three, M's cross blocks are no longer transposes of each
other (M[:modes, modes:] = V^T W a from the amplitude row, M[modes:, :modes]
= q^T V(z) here) -- this is a genuine Petrov-Galerkin closure (test space !=
trial space for the phase-condition row), not the symmetric normal-equations
structure Psi_Res/freeze/freeze_tilde share. Its conditioning is governed by
whether q stays non-orthogonal to the ACTUAL trajectory's W(z) @ a, not by
W(z) @ a's self-overlap with V(z) -- a different, hopefully much rarer,
failure mode. sdeim hyperreduction is not yet supported for this condition
(see the NotImplementedError in matrices_sPOD_galerkin_online).

phase_condition: "freeze_tilde_weighted"
------------------------------------------
Unlike "template" (which changed the TEST DIRECTION), this changes the INNER
PRODUCT "freeze_tilde" is derived from -- a spatial reweighting, not a
state-space one, motivated by Beyn & Thuemmler's finding (SIAM J. Appl. Dyn.
Syst. 3(2):85-116, 2004, Section 4.3.3, Barkley spiral example) that the
choice of inner product used to define the phase condition can by itself
determine stability.

"freeze_tilde" minimizes ||a_dot||_2^2, which is secretly an L2(space) norm
in disguise: since basis.modes = Phi has orthonormal columns,
||v_dot||_{L2}^2 = ||Phi @ a_dot||_{L2}^2 = a_dot^T (Phi^T Phi) a_dot =
a_dot^T a_dot exactly. Swapping in a WEIGHTED L2_eta inner product
<f, g>_eta := f^T diag(eta) g, eta(x) > 0 a chosen spatial weight, means
instead minimizing a_dot^T Sigma a_dot with

    Sigma := Phi^T diag(eta) Phi     in R^{modes x modes}, FIXED, precomputed
                                      once offline (see compute_energy_weighted_sigma)

Re-deriving with the SAME reduced kinematics a_dot = T1 - Na * dz/dt as
"freeze_tilde" (row 1/the amplitude equation is untouched -- weighting only
changes this one scalar row) and minimizing (1/2) a_dot^T Sigma a_dot over
dz/dt gives

    [Na^T Sigma Na] dz/dt = Na^T Sigma T1

which reduces EXACTLY to "freeze_tilde"'s own formula when Sigma = I (eta =
1 everywhere) -- a strict generalization, not a different mechanism.

eta is chosen here as the co-moving basis's own local energy,
eta(x) := sum_i phi_i(x)^2 (= diag(Phi @ Phi^T)) -- near-zero in the flat
far field where every mode vanishes, large right at the shock where the
modes actually carry structure, needing no new offline machinery beyond Phi
itself. Because Na = V(z)^T W(z) @ a still updates every step from the
CURRENT (a, z) (only the WEIGHT is fixed, not a test direction evaluated at
a stale reference state), this doesn't inherit "template"'s go-stale-as-z-
drifts failure mode -- it's a purely spatial re-emphasis layered on top of
an otherwise fully current-state-dependent phase condition. Compatible with
sdeim hyperreduction (Na and T1 come from the same place "freeze_tilde"
already gets them from, hyper-reduced or not).
"""

PHASE_CONDITIONS = ("residual", "freeze", "freeze_tilde", "template", "freeze_tilde_weighted")


def solve_lin_system(M, A, precondition: bool = False):
    """Solve M @ [da/dt; dz/dt] = A, kept as its own function so the
    augmented (a, z) system's assembly (matrices_sPOD_galerkin_online) and
    its solve are visually separate steps in reduced_rhs_sPOD.

    precondition: if True, apply symmetric diagonal (Jacobi/equilibration)
    scaling before solving. M's diagonal spans wildly different scales --
    the a-block is ~O(1) (V(z) is close to orthonormal) while the z-block
    a^T W(z)^T W(z) a is ~O(1e5) (W is a spatial derivative of a
    near-discontinuous shock profile, hence large-norm) -- which by itself
    was measured to account for most of M's raw condition number. Rescaling
    to a unit diagonal,

        d_i = 1/sqrt(M_ii),  D = diag(d),  M~ = D M D,  A~ = D A

    solves the mathematically IDENTICAL system (D is invertible whenever
    every M_ii > 0) at a far better condition number: van der Sluis's result
    says unit-diagonal scaling is within a factor of (l+1) of the best
    possible diagonal preconditioner for a symmetric positive (semi)definite
    matrix. Recomputed fresh from the CURRENT M every call, since M's scale
    can drift over the trajectory (V(z) loses orthonormality with |z|), not
    a fixed scaling decided once.
    """
    if not precondition:
        return np.linalg.solve(M, A)

    diag_M = np.diag(M)
    d = 1.0 / np.sqrt(diag_M)
    M_scaled = (d[:, None] * M) * d[None, :]
    A_scaled = d * A

    y = np.linalg.solve(M_scaled, A_scaled)
    return d * y


def matrices_sPOD_galerkin_online(lhs: np.ndarray, V: np.ndarray, W: np.ndarray, a: np.ndarray, z: float, ds: np.ndarray,
                                  modes: int, dx: float, phase_condition: str = "residual",
                                  sdeim: "object | None" = None,
                                  template_q: np.ndarray | None = None,
                                  sigma: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the (modes+1, modes+1) normal-equations system for [da/dt; dz/dt]
    at the current reduced state (a, z) -- see this module's docstring for the
    Psi = [V(z) | W(z) a] derivation. lhs is matrices_sPOD_galerkin's offline
    table of (V^T V, V^T W, W^T W) at each of the len(ds) sample shifts; V, W
    are basis.V/basis.W, the matching shifted-mode / shift-derivative tables.

    phase_condition: "residual" (default) uses the joint-residual-minimizing
    row (Psi_Res); "freeze" and "freeze_tilde" drop that row's da/dt coupling
    (the two Beyn-Thuemmler freezing conditions); "template" tests against a
    FIXED direction instead of the current-state W(z) @ a -- see this
    module's docstring.

    sdeim: if given (a spod_hyperreduction.SPODDEIMOperator), T1 = V(z)^T r
    and T2 = a^T W(z)^T r are obtained from its hyper-reduced approximation
    (O(p, r, modes), independent of n) instead of the full O(n) nonlinearity
    evaluation below -- see spod_hyperreduction.py's module docstring. M's
    assembly and the phase_condition branching are completely unaffected:
    both M1/N/M2 (hence M itself) come from the offline Gram-matrix table
    lhs, never from a full nonlinearity evaluation, hyper-reduced or not.
    Not yet supported together with phase_condition="template" (raises
    NotImplementedError) -- that would need a third hyper-reduced factor
    (q^T instead of V(z)^T/a^T W(z)^T) that spod_hyperreduction.py doesn't
    build yet.

    template_q: required when phase_condition="template" -- the fixed test
    direction, see compute_template_direction and this module's docstring.

    sigma: required when phase_condition="freeze_tilde_weighted" -- the fixed
    (modes, modes) weighted-inner-product matrix, see
    compute_energy_weighted_sigma and this module's docstring.

    Returns (M, A) with M @ [da/dt; dz/dt] = A the linear system to solve
    (reduced_rhs_sPOD does that solve).
    """
    if phase_condition not in PHASE_CONDITIONS:
        raise ValueError(f"Unknown phase_condition '{phase_condition}'. Choose one of {PHASE_CONDITIONS}.")
    if phase_condition == "template":
        if template_q is None:
            raise ValueError("phase_condition='template' requires template_q -- see compute_template_direction.")
        if sdeim is not None:
            raise NotImplementedError("phase_condition='template' does not yet support sdeim hyperreduction "
                                       "-- it needs the full nonlinearity to project onto template_q.")
    if phase_condition == "freeze_tilde_weighted" and sigma is None:
        raise ValueError("phase_condition='freeze_tilde_weighted' requires sigma -- "
                          "see compute_energy_weighted_sigma.")

    M = np.empty((modes + 1, modes + 1))
    A = np.empty(modes + 1)

    # z is the CURRENT frame shift; V/W's table index is built in cs = "shift
    # applied to the stationary modes to reach the lab frame" = -z (see
    # rom_solver.sPODROMSolver.project/reconstruct), so look up -z here too.
    intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(ds, -z)

    Da = a.reshape(-1, 1)

    # Linearly interpolate the three offline Gram-matrix tables to the
    # current shift z, rather than recomputing V(z)^T V(z) etc. from scratch.
    M1 = np.add(weight * lhs[0, intervalIdx], (1 - weight) * lhs[0, intervalIdx + 1])  # ~= V(z)^T V(z)
    N = np.add(weight * lhs[1, intervalIdx], (1 - weight) * lhs[1, intervalIdx + 1])   # ~= V(z)^T W(z)
    M2 = np.add(weight * lhs[2, intervalIdx], (1 - weight) * lhs[2, intervalIdx + 1])  # ~= W(z)^T W(z)

    # V(z)^T W(z) a, the tangent-space cross block -- also exactly the
    # quantity "freeze_tilde" below needs (see module docstring).
    Na = N @ Da

    # Psi^T Psi's amplitude row (a-block), identical across all three phase
    # conditions -- only the dz/dt row (below) differs.
    M[:modes, :modes] = M1.copy()
    M[:modes, modes:] = Na

    if sdeim is None:
        # V(z), W(z) themselves, interpolated the same way, needed for the
        # right-hand side (Psi^T r below) rather than just their Gram matrices.
        VT = np.add(weight * V[intervalIdx], (1 - weight) * V[intervalIdx + 1]).T
        WT = np.add(weight * W[intervalIdx], (1 - weight) * W[intervalIdx + 1]).T

        # r = burgers_rhs(u), the *same* FOM flux-divergence operator as plain
        # POD-Galerkin (reduced_rhs_POD) -- the nonlinearity itself doesn't know
        # about the moving frame, only the basis it's projected against does.
        nonlinearity = burgers_rhs(VT.T @ a, dx)
        T1 = VT @ nonlinearity                  # V(z)^T r
        T2 = float(a @ (WT @ nonlinearity))     # a^T W^T r
    else:
        T1, T2 = sdeim.nonlinear_terms(a, z)

    A[:modes] = T1

    if phase_condition == "residual":
        M[modes:, :modes] = Na.T
        M[modes:, modes:] = Da.T @ (M2 @ Da)          # a^T W^T W a
        A[modes:] = T2
    elif phase_condition == "freeze":
        M[modes:, :modes] = 0.0
        M[modes:, modes:] = Da.T @ (M2 @ Da)          # a^T W^T W a
        A[modes:] = T2
    elif phase_condition == "freeze_tilde":
        M[modes:, :modes] = 0.0
        M[modes:, modes:] = Na.T @ Na                  # ||V^T W a||^2
        A[modes:] = Na.T @ A[:modes]                   # (V^T W a)^T (V^T r)
    elif phase_condition == "template":
        # sdeim is None here, enforced above, so VT/WT/nonlinearity exist
        M[modes:, :modes] = VT @ template_q             # q^T V(z), as a (modes,) row
        M[modes:, modes:] = float((WT @ template_q) @ a)  # q^T W(z) a
        A[modes:] = float(template_q @ nonlinearity)     # q^T r
    else:  # "freeze_tilde_weighted": freeze_tilde's own formula with Sigma in
           # place of the identity -- reduces to it exactly when sigma == I.
        NaT_Sigma = Na.T @ sigma                        # (Na^T Sigma), shape (1, modes)
        M[modes:, :modes] = 0.0
        M[modes:, modes:] = NaT_Sigma @ Na               # Na^T Sigma Na
        A[modes:] = NaT_Sigma @ A[:modes]                # Na^T Sigma T1

    return M, A


def reduced_rhs_sPOD(lhs: np.ndarray, V: np.ndarray, W: np.ndarray, a: np.ndarray,
                     z: float, ds: np.ndarray, modes: int, dx: float, precondition: bool = False,
                     phase_condition: str = "residual", sdeim: "object | None" = None,
                     template_q: np.ndarray | None = None, sigma: np.ndarray | None = None) -> np.ndarray:
    """da/dt and dz/dt together, as one (modes+1,) vector: solve the normal
    equations M @ [da/dt; dz/dt] = A assembled by matrices_sPOD_galerkin_online.
    Callers (sPODROMSolver.rhs) split the result back into da/dt = result[:-1]
    and dz/dt = result[-1].

    precondition: forwarded to solve_lin_system -- see its docstring.
    phase_condition: forwarded to matrices_sPOD_galerkin_online -- "residual"
    (default, Psi_Res), "freeze" (Psi_freeze), "freeze_tilde"
    (Psi_freeze_tilde), "template", or "freeze_tilde_weighted" -- see this
    module's docstring.
    sdeim, template_q, sigma: forwarded to matrices_sPOD_galerkin_online -- see its docstring.
    """

    M, A = matrices_sPOD_galerkin_online(lhs, V, W, a, z, ds, modes, dx, phase_condition=phase_condition, sdeim=sdeim,
                                          template_q=template_q, sigma=sigma)

    return solve_lin_system(M, A, precondition=precondition)


def compute_template_direction(basis: sPODBasis, delta_s: np.ndarray, u0: np.ndarray,
                                z0: float | None = None) -> np.ndarray:
    """The fixed test direction q for phase_condition="template": W(z0) @ a0,
    the shift-tangent direction at a chosen REFERENCE state (a0, z0) -- see
    this module's docstring for why a fixed direction (rather than the
    current-state W(z) @ a used by "residual"/"freeze"/"freeze_tilde") trades
    local optimality for robustness against their shared near-cancellation
    failure mode.

    Defaults to the training trajectory's own t=0 snapshot: z0 = basis.shift[0],
    which is exactly 0 by construction (spod.compute_shifts always measures
    shift relative to the first snapshot) -- picked simply because it's
    already available and, being the very start of the trajectory, hasn't had
    a chance to drift into whatever configuration eventually causes trouble.

    a0 is obtained the same way sPODROMSolver.project() would (pinv(V(z0)) @
    u0), computed directly here since no solver instance exists yet at this
    point in the offline setup.
    """
    if basis.V is None or basis.W is None:
        raise ValueError("basis.V/basis.W are not set -- call basis.make_V_W(delta_s) before "
                          "compute_template_direction().")

    if z0 is None:
        z0 = float(basis.shift[0])

    intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(delta_s, -z0)
    V0 = np.add(weight * basis.V[intervalIdx], (1 - weight) * basis.V[intervalIdx + 1])
    W0 = np.add(weight * basis.W[intervalIdx], (1 - weight) * basis.W[intervalIdx + 1])

    a0 = np.linalg.pinv(V0) @ u0
    return W0 @ a0


def compute_energy_weighted_sigma(basis: sPODBasis) -> np.ndarray:
    """Sigma := Phi^T diag(eta) Phi for phase_condition="freeze_tilde_weighted",
    the fixed (modes, modes) matrix "freeze_tilde"'s own ||a_dot||_2^2
    generalizes to under the weighted inner product <f,g>_eta := f^T diag(eta) g
    -- see this module's docstring for the derivation.

    eta(x) := sum_i phi_i(x)^2 (= diag(Phi @ Phi^T)), the co-moving basis's own
    local energy: near-zero in the flat far field where every mode vanishes,
    large right at the shock where the modes carry structure. Needs nothing
    beyond basis.modes itself -- computed once, offline.
    """
    Phi = basis.modes
    eta = np.sum(Phi**2, axis=1)
    return Phi.T @ (eta[:, None] * Phi)


def matrices_sPOD_galerkin(basis: sPODBasis, num_samples: int, modes: int) -> np.ndarray:
    """Offline precomputation: the three Gram matrices matrices_sPOD_galerkin_online
    needs, evaluated at every one of the num_samples tabulated shift values in
    basis.delta_s (i.e. at each basis.V[it]/basis.W[it] pair), so the online
    rhs only has to interpolate between two rows of a small lookup table
    instead of recomputing V(z)^T V(z) etc. (each an (n, modes) @ (modes, n)
    contraction) on every time step.

    Returns LHS_mat, shape (3, num_samples, modes, modes):
        LHS_mat[0, it] = V[it]^T @ V[it]
        LHS_mat[1, it] = V[it]^T @ W[it]
        LHS_mat[2, it] = W[it]^T @ W[it]
    """
    LHS_mat = np.zeros((3, num_samples, modes, modes))
    for it in range(num_samples):
        LHS_mat[0, it, ...] = oe.contract('ij,jk->ik', basis.V[it].T, basis.V[it])
        LHS_mat[1, it, ...] = oe.contract('ij,jk->ik', basis.V[it].T, basis.W[it])
        LHS_mat[2, it, ...] = oe.contract('ij,jk->ik', basis.W[it].T, basis.W[it])

    return LHS_mat
