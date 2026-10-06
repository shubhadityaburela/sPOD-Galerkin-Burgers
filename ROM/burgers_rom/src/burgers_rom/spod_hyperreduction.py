r"""Shifted DEIM (sDEIM) hyper-reduction for the sPOD-Galerkin ROM's nonlinear
flux term -- the sPOD analogue of hyperreduction.py's (plain POD) DEIM.

Recap of the full (non-hyper-reduced) sPOD-Galerkin online rhs (galerkin.py):
Psi^T r splits into two terms, both requiring the FULL O(n) nonlinearity
r = burgers_rhs(V(z) a, dx):

    T1 = V(z)^T r                (modes,)   -- feeds A[:modes]
    T2 = a^T W(z)^T r            scalar     -- feeds A[modes:]

sDEIM approximates r itself via a SHIFT-DEPENDENT DEIM basis V_nl(z), the
same way plain DEIM approximates it via a fixed basis Phi_nl:

    r = burgers_rhs(V(z) a, dx) = (-1/dx) D f_num(E V(z) a)
      ~= (-1/dx) D V_nl(z) [S^T V_nl(z)]^{-1} S^T f_num(E V(z) a)

Substituting into T1, T2 and precomputing everything that doesn't depend on
the online (a, z):

    T1 ~= [(-1/dx) V(z)^T D V_nl(z) (S^T V_nl(z))^{-1}] @ S^T f_num(E V(z) a)
           \_______________________ factor_a(z) ________________________/
    T2 ~= a^T [(-1/dx) W(z)^T D V_nl(z) (S^T V_nl(z))^{-1}] @ S^T f_num(E V(z) a)
              \_______________________ factor_b(z) ________________________/

factor_a(z), factor_b(z) are each (modes, r); tabulated offline at every
shift sample delta_s[it] (same lookup grid as basis.V/basis.W) and linearly
interpolated online, exactly like galerkin.py's Gram-matrix table lhs.

CRITICAL: unlike V_nl(z)/W(z) themselves, the selection indices S (which r
of the n+1 faces are sampled) must be the SAME at every table entry, fixed
once offline -- NOT re-selected per shift sample. If S varied with z,
"column k" of factor_a(z_it) and factor_a(z_{it+1}) would refer to different
physical faces, and linearly interpolating between them would blend
unrelated quantities. Only the *values* in factor_a/factor_b/stencil_basis
vary with z; their shape and the physical meaning of each row/column stay
fixed across the whole table -- the same invariant basis.V/basis.W already
rely on.

Picking S from a SINGLE reference shift (S = select_deim_points(Phi_nl), i.e.
V_nl(0) = Phi_nl directly) and then reusing those same *physical* row indices
at every table entry is NOT robust: Phi_nl's modes are strongly localized
(large values concentrated in a narrow window, near-zero elsewhere -- typical
DEIM behavior for a flux field dominated by a moving shock), so pivoted QR
clusters the selected faces tightly around wherever that informative window
sits AT z=0. The moment z moves that window away from the fixed indices
(verified: even a ONE-grid-cell shift is enough), V_nl(z)[S] collapses
towards singular -- cond(V_nl(z)[S]) was measured to jump from ~8 to ~1e228
between the very first two table entries. Re-selecting S from scratch at
every shift (fresh QR pivoting on V_nl(z_it)) doesn't work either: nothing
guarantees "index j" in S_it and S_{it+1} refer to the same physical feature,
so factor_a(z_it) and factor_a(z_{it+1}) would no longer be interpolatable.

The fix implemented below: pick S0 once from Phi_nl (=V_nl(0)), then TRACK it
through the shift table by rigid translation rather than re-selecting it.
V_nl(z) is itself just Phi_nl translated by z (shift_open_matrix_...), so

    V_nl(z_it)[S0 + k_it, :]  ~=  Phi_nl[S0, :],   k_it = round(delta_s[it] / dx)

i.e. sampling S0's rows shifted along with the profile lands back on the same
well-conditioned submatrix at every interior table entry, because it's always
the same *relative* location on the (rigidly translating) informative window.
"Index j" still means the same tracked point at every it (it's always "the
row that was S0[j] at the reference shift"), so factor_a/factor_b/stencil_basis
stay linearly interpolatable across the table exactly like basis.V/basis.W
already are -- only now that guarantee holds simultaneously with good
conditioning, instead of trading one for the other.

Near the domain edges, S0 + k_it runs past the grid; those indices are
CLAMPED (np.clip), not periodically wrapped -- the FOM's outflow/transmissive
BC and the open (non-periodic) spline shift mean nothing re-enters from the
opposite side, so wrapping would sample physically unrelated content. This
degrades accuracy only in the last ~1% of the shift range, coinciding with
the already-known, unrelated boundary effect where V_nl(z) itself loses rank
as the co-moving profile exits the domain -- an inherent limit of a bounded,
non-periodic domain, not a fixable artifact of this selection scheme.
"""

from dataclasses import dataclass

import numpy as np

from burgers_rom.deim import DEIMBasis
from burgers_rom.hyperreduction import (
    build_flux_snapshot_matrix,
    build_stencil_maps,
    sampled_face_fluxes,
    select_deim_points,
)
from burgers_rom.spod import findIntervalAndGiveInterpolationWeight_1D, sPODBasis
from burgers_rom.transforms import (
    construct_open_spline_coeffs_multiple,
    give_open_spline_coefficient_matrices,
    shift_open_matrix_precomputed_coeffs_multiple,
    shifted_U_open,
)


def compute_snl_basis(U: np.ndarray, shift: np.ndarray, dx: float, r: int) -> tuple[np.ndarray, tuple]:
    """Phi_nl in R^{(n+1) x r}: the co-moving-frame DEIM basis of the flux
    snapshots, de-shifted using the SAME shift trajectory as the state
    (basis.shift) rather than a freshly computed one -- the modeling
    assumption that the nonlinearity travels along the same path as the
    state itself (see module docstring).

    Returns (Phi_nl, spline_matrices_nl); spline_matrices_nl = (A1, D1, D2,
    M_lu) for the (n+1)-sized flux grid, needed again by make_Vnl_table
    (built once here, not rebuilt per shift sample).
    """
    F_snap = build_flux_snapshot_matrix(U)  # (n+1, m)
    n_faces = F_snap.shape[0]

    spline_matrices_nl = give_open_spline_coefficient_matrices(n_faces)
    A1, D1, D2, M_lu = spline_matrices_nl
    b, c, d = construct_open_spline_coeffs_multiple(F_snap, A1, D1, D2, M_lu, dx)
    F_snap_shifted = shift_open_matrix_precomputed_coeffs_multiple(F_snap, shift, b, c, d, n_faces, dx)

    modes, singular_values, coeffs = np.linalg.svd(F_snap_shifted, full_matrices=False)
    full_basis = DEIMBasis(modes, singular_values, coeffs)
    Phi_nl = full_basis.truncate(r).modes

    return Phi_nl, spline_matrices_nl


def make_Vnl_table(Phi_nl: np.ndarray, delta_s: np.ndarray, spline_matrices_nl: tuple, dx: float) -> np.ndarray:
    """V_nl(z) lookup table: Phi_nl shifted to each sample in delta_s, the
    sPODBasis.make_V_W of the flux-nonlinearity basis (value only, no
    derivative table needed -- see module docstring, V_nl(z) is never
    differentiated in the sDEIM formulas).

    Returns V_nl_table, shape (len(delta_s), n+1, r).
    """
    A1, D1, D2, M_lu = spline_matrices_nl
    n_faces = Phi_nl.shape[0]
    b, c, d = construct_open_spline_coeffs_multiple(Phi_nl, A1, D1, D2, M_lu, dx)

    V_nl_table = np.empty((len(delta_s), n_faces, Phi_nl.shape[1]))
    for it, s in enumerate(delta_s):
        V_nl_table[it] = shifted_U_open(Phi_nl, s, b, c, d, n_faces, dx)

    return V_nl_table


@dataclass
class SPODDEIMOperator:
    """Everything the online phase needs, fully assembled offline.

    factor_a, factor_b: (num_samples, modes, r)  interpolation tables for
        T1's and T2's projectors (see module docstring)
    stencil_basis:      (num_samples, p, modes)  interpolation table, the p
        stencil rows of (E @ V(z)) at each shift sample
    local_stencil:      (r, 4)  FIXED across all shift samples (S is fixed --
        see module docstring), per-face window positions into the p sampled
        cells, identical role to hyperreduction.DEIMOperator.local_stencil
    delta_s:            (num_samples,)  the shift lookup grid all three
        tables above share with basis.V/basis.W/galerkin's lhs table
    """

    factor_a: np.ndarray
    factor_b: np.ndarray
    stencil_basis: np.ndarray
    local_stencil: np.ndarray
    delta_s: np.ndarray

    def nonlinear_terms(self, a: np.ndarray, z: float) -> tuple[np.ndarray, float]:
        """(T1, T2), the sDEIM approximation of (V(z)^T r, a^T W(z)^T r) --
        the two quantities galerkin.matrices_sPOD_galerkin_online needs from
        the nonlinearity, computed here at cost O(p*modes + r*modes),
        independent of n.
        """
        intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(self.delta_s, -z)

        factor_a_z = np.add(weight * self.factor_a[intervalIdx], (1 - weight) * self.factor_a[intervalIdx + 1])
        factor_b_z = np.add(weight * self.factor_b[intervalIdx], (1 - weight) * self.factor_b[intervalIdx + 1])
        stencil_basis_z = np.add(weight * self.stencil_basis[intervalIdx],
                                  (1 - weight) * self.stencil_basis[intervalIdx + 1])

        u_stencil = stencil_basis_z @ a
        f_hat = sampled_face_fluxes(u_stencil, self.local_stencil)

        T1 = factor_a_z @ f_hat
        T2 = float(a @ (factor_b_z @ f_hat))
        return T1, T2


def build_spod_deim_operator(basis: sPODBasis, U: np.ndarray, delta_s: np.ndarray, r: int) -> SPODDEIMOperator:
    """Offline assembly: run the whole sDEIM pipeline and return the operator.

    basis must already have V/W populated (basis.make_V_W(delta_s), same
    delta_s passed here) -- factor_a/factor_b/stencil_basis are built from
    basis.V/basis.W directly, not recomputed.

    Steps:
      1. Phi_nl, spline_matrices_nl = compute_snl_basis(U, basis.shift, basis.dx, r)   (n+1, r)
      2. V_nl_table = make_Vnl_table(Phi_nl, delta_s, spline_matrices_nl, basis.dx)     (num_samples, n+1, r)
      3. S0 = select_deim_points(Phi_nl, r)   -- reference selection, at the true zero shift
      4. stencil_cells0, local_stencil = build_stencil_maps(S0, basis.Nx)   -- reference window
         structure; local_stencil (which of the p stencil slots each face's 4-cell window
         occupies) stays fixed across the table -- a rigid translation of S0 preserves the
         relative gaps between selected faces, so the dedup/position pattern doesn't change,
         only the physical cells being dedup'd. Only stencil_cells0 needs the per-it offset.
      5. per shift sample it:
           k_it   = round(delta_s[it] / dx)              -- tracked integer-cell offset
           S_it   = clip(S0 + k_it, 0, n_faces - 1)       -- S0, translated to follow the shock
           inv_S  = pinv(V_nl_table[it][S_it])                                    (r, r)
           factor_a[it] = (-1/dx) * basis.V[it].T @ D @ V_nl_table[it] @ inv_S   (modes, r)
           factor_b[it] = (-1/dx) * basis.W[it].T @ D @ V_nl_table[it] @ inv_S   (modes, r)
           stencil_basis[it] = (E @ basis.V[it])[clip(stencil_cells0 + k_it, ...)]  (p, modes)

    Step 5's last two lines are computed as a direct O(p*modes) row gather
    rather than the O(n^2*modes) dense products E @ basis.V[it] / D @ V_nl_table[it]
    -- E only duplicates the first/last rows into ghost cells and passes interior
    rows through unchanged (build_extension_matrix), so its rows are recovered by
    simply clamping each padded-space index to the corresponding basis.V[it] row
    (see padded_row_idx below), and D is just the forward difference
    (build_difference_matrix); done once per shift sample (up to a few thousand
    times here), the dense matmuls would dominate this whole function's runtime
    for no benefit. pinv (not inv) is used for inv_S since S_it can pick up
    duplicate entries where clipping saturates near the domain edges -- see
    module docstring on why that's an accepted, bounded degradation there
    rather than something to special-case away.
    """
    if basis.V is None or basis.W is None:
        raise ValueError("basis.V/basis.W are not set -- call basis.make_V_W(delta_s) before build_spod_deim_operator().")

    n = basis.Nx
    dx = basis.dx
    modes = basis.modes.shape[1]

    Phi_nl, spline_matrices_nl = compute_snl_basis(U, basis.shift, dx, r)
    V_nl_table = make_Vnl_table(Phi_nl, delta_s, spline_matrices_nl, dx)
    n_faces = Phi_nl.shape[0]

    S0 = select_deim_points(Phi_nl, r)
    stencil_cells0, local_stencil = build_stencil_maps(S0, n)

    num_samples = len(delta_s)
    factor_a = np.empty((num_samples, modes, r))
    factor_b = np.empty((num_samples, modes, r))
    p = len(stencil_cells0)
    stencil_basis = np.empty((num_samples, p, modes))

    for it in range(num_samples):
        k_it = int(round(delta_s[it] / dx))
        S_it = np.clip(S0 + k_it, 0, n_faces - 1)

        Vnl_it = V_nl_table[it]
        inv_S = np.linalg.pinv(Vnl_it[S_it])
        # D @ Vnl_it, done as the plain forward difference build_difference_matrix
        # encodes ((Df)_i = f_{i+1} - f_i) rather than a dense (n, n+1) matmul,
        # since this runs once per shift sample (see docstring above).
        DVnl = Vnl_it[1:] - Vnl_it[:-1]
        DVnl_invS = DVnl @ inv_S  # (n, r), shared by both factors

        factor_a[it] = (-1 / dx) * basis.V[it].T @ DVnl_invS
        factor_b[it] = (-1 / dx) * basis.W[it].T @ DVnl_invS

        # E's rows: [row 0, row 0, row 0, ..., row n-1, row n-1, row n-1] (two
        # ghosts per side, build_extension_matrix) -- so (E @ V)[stencil_cells]
        # is just V at these row indices, clamped into [0, n-1].
        padded_row_idx = np.clip(stencil_cells0 + k_it - 2, 0, n - 1)
        stencil_basis[it] = basis.V[it][padded_row_idx]

    return SPODDEIMOperator(factor_a, factor_b, stencil_basis, local_stencil, delta_s)
