import matplotlib.pyplot as plt
from scipy.linalg import qr

from burgers_rom.deim import DEIMBasis

r"""DEIM hyper-reduction for the nonlinear flux term of the POD-Galerkin ROM.

Semi-discrete hyper-reduced system:

    da/dt = (-1/dx) * Phi^T D Phi_nl (S^T Phi_nl)^{-1} S^T f_num(E Phi a(t))
            \_________________________________________/     \______________/
                offline: B, an (l x r) matrix                online: r sampled
                                                             face fluxes

Spaces and shapes (n = number of interior cells):

    cell space (interior):  n        Phi  (n, l)      POD basis, truncation rank l
    face space:             n + 1    D    (n, n+1)    difference matrix, rows [-1, 1]
                                     Phi_nl (n+1, r)  DEIM basis of the flux snapshots
                                     S    (r,)        DEIM face indices (sorted)
    padded cell space:      n + 2    E    (n+2, n)    transmissive-BC extension matrix

    f_num(.) = [F_god(u^L_{-1/2}, u^R_{-1/2}), ..., F_god(u^L_{n-1/2}, u^R_{n-1/2})]
             in R^{n+1}: the Godunov flux at every face, with the left/right face
             states u^L, u^R coming from the MC-limited MUSCL reconstruction.

    E Phi a lives in padded cell space (n+2: interior + 2 ghost cells),
    f_num and S live in face space (n+1). D maps face space back to cell space.

Offline phase precomputes:
  1) B = (-1/dx) * Phi^T @ D @ Phi_nl @ inv(Phi_nl[S])            -> (l, r)
  2) the stencil machinery for S^T f_num(E Phi a): each selected face i+1/2
     needs the four cells u_{i-1}, u_i, u_{i+1}, u_{i+2} for the MC limiter.
     The union of those cells over the r faces (duplicates stored ONCE) gives
     p unique padded-cell indices, and selecting those p rows of (E @ Phi)
     gives the (p, l) stencil basis.

Online phase per rhs evaluation, all costs independent of n:
     u_st  = stencil_basis @ a            (p,)   p sampled padded-cell values
     f_hat = f_num evaluated face-locally (r,)   p cell values in, r fluxes out
     da/dt = B @ f_hat                    (l,)
"""

from dataclasses import dataclass

from burgers1d.boundary_conditions import outflow
from burgers1d.flux import godunov
from burgers1d.reconstruction import muscl_interface_states
import numpy as np
from scipy.linalg import qr


def build_difference_matrix(n_cells: int) -> np.ndarray:
    """The (n, n+1) matrix D mapping face fluxes to the cell-wise flux difference.

    Row i is [0, ..., 0, -1, 1, 0, ..., 0] with the -1 in column i, so that
    (D f)_i = f_{i+1/2} - f_{i-1/2}. Together with the -1/dx prefactor absorbed
    into B, this is exactly the finite-volume divergence in burgers_rhs.
    """
    return -np.eye(n_cells, n_cells + 1) + np.eye(n_cells, n_cells + 1, k=1)


def build_extension_matrix(n_cells: int) -> np.ndarray:
    """The (n+4, n) transmissive (outflow/zero-gradient) extension matrix E.

    E = [e_0; e_0; I_n; e_{n-1}; e_{n-1}] as rows: the first and last interior
    cells are each duplicated TWICE into ghost cells, so E @ u is exactly the
    padded state burgers1d.boundary_conditions.outflow(u, n_ghost=2) would
    produce -- [u_0, u_0, u_1, ..., u_{n-1}, u_{n-1}, u_{n-1}]. That's 2 ghosts
    per side (n+4 total) rather than 1 (n+2): muscl_interface_states needs 2
    ghosts per side to reconstruct every face, so E @ Phi @ a can be fed to it
    (and then to godunov) with no extra repadding step in between. Sizing it
    this way up front costs nothing, since only p << n+4 rows of E @ Phi are
    ever actually selected (see build_stencil_maps).
    """
    E = np.zeros((n_cells + 4, n_cells))
    E[0, 0] = 1.0
    E[1, 0] = 1.0
    E[2:n_cells + 2, :] = np.eye(n_cells)
    E[n_cells + 2, n_cells - 1] = 1.0
    E[n_cells + 3, n_cells - 1] = 1.0
    return E


def face_flux_vector(u_padded: np.ndarray) -> np.ndarray:
    """f_num: all n+1 face fluxes of one padded state u_padded in R^{n+2}.

    MC-limited MUSCL reconstruction of the left/right states at every face,
    then the Godunov flux -- the same physics as galerkin.burgers_rhs but
    returning the raw face-flux vector instead of its divergence. Used offline
    to build the flux snapshot matrix; never called online.

    """
    u_left, u_right = muscl_interface_states(u_padded)
    fluxes = godunov(u_left, u_right)

    return fluxes


def build_flux_snapshot_matrix(U: np.ndarray) -> np.ndarray:
    """The matrix the DEIM SVD is computed on: fluxes of the training snapshots.

        F_snap = [f_num(E u(t_0)), ..., f_num(E u(t_m))]  in R^{(n+1) x m}

    U is the FOM snapshot trajectory shaped (n, m) as loaded by snapshots.py;
    each snapshot is extended to padded cell space and pushed through
    face_flux_vector, one column per time instance.
    """
    flux_snapshots = np.empty((U.shape[0] + 1, U.shape[1]))
    for i in range(U.shape[1]):
        u_padded = outflow(U[:, i], n_ghost=2)
        flux_snapshots[:, i] = face_flux_vector(u_padded)

    return flux_snapshots



def compute_deim_basis(flux_snapshots: np.ndarray, r: int) -> np.ndarray:
    """Phi_nl in R^{(n+1) x r}: first r left singular vectors of F_snap.

    Thin SVD of the flux snapshot matrix, truncated to the prescribed number
    of DEIM points r (basis size and point count coincide in standard DEIM).
    """

    Pmodes, singular_values, coeffs = np.linalg.svd(flux_snapshots, full_matrices=False)
    full_basis = DEIMBasis(Pmodes, singular_values, coeffs)
    basis = full_basis.truncate(r)

    return basis.modes


def select_deim_points(deim_basis: np.ndarray, r: int) -> np.ndarray:
    """The r sampled face indices S, via pivoted QR on Phi_nl^T (Q-DEIM).

        _, _, piv = qr(Phi_nl.T, pivoting=True)
        S = np.sort(piv[:r])

    S lives in face space: each entry is the index of one of the n+1 faces.
    S^T Phi_nl in the formula is then simply the row selection Phi_nl[S],
    an (r, r) matrix that must be invertible (guaranteed by the pivoting).
    """
    _, _, piv = qr(deim_basis.T, pivoting=True)
    S = np.sort(piv[:r])

    return S


def build_stencil_maps(deim_faces: np.ndarray, n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Offline part 2): which padded cells feed the r selected faces, and how.

    Face k (in face-space indexing 0..n) is reconstructed from the 4 padded
    cells [k, k+1, k+2, k+3] of the (n+4)-length E @ u (2 ghosts per side, see
    build_extension_matrix) -- verified directly against
    burgers1d.reconstruction.muscl_interface_states on a full padded array.
    k ranges 0..n_cells, so k+3 tops out at n_cells+3, the last valid padded
    index; no clipping is needed.

    Returns:
      stencil_cells: (p,) sorted UNIQUE padded-cell indices over all r faces
        (cells shared by neighboring faces are stored once -- no double storage).
        Selecting these p rows of (E @ Phi) yields the (p, l) stencil basis.
      local_stencil: (r, 4) positions into stencil_cells, so that online,
        u_st[local_stencil[j]] recovers the 4-cell window of face j without
        touching anything of size n.
    """

    local_stencil = np.empty((len(deim_faces), 4), dtype=int)
    for j, face in enumerate(deim_faces):
        local_stencil[j] = np.arange(face, face + 4)

    stencil_cells = np.unique(local_stencil.flatten())
    local_stencil = np.searchsorted(stencil_cells, local_stencil)

    return stencil_cells, local_stencil


def sampled_face_fluxes(u_stencil: np.ndarray, local_stencil: np.ndarray) -> np.ndarray:
    """The online nonlinearity f_hat_r: p sampled cell values in, r face fluxes out.

    Gathers all r 4-cell windows [u_{i-1}, u_i, u_{i+1}, u_{i+2}] from
    u_stencil via local_stencil in one shot, as columns of a (4, r) array
    (window position on axis 0, face index on axis 1). burgers1d's
    muscl_interface_states/godunov slice/broadcast along axis 0, so calling
    them once on this batch reproduces the same MC-limited reconstruction and
    Godunov flux as the FOM for each face -- no separate implementation, no
    per-face loop. Cost O(r), never O(n).
    """

    windows = u_stencil[local_stencil].T           # (r, 4) -> (4, r): window position x face
    u_left, u_right = muscl_interface_states(windows)  # each (1, r): one face per column
    return godunov(u_left, u_right).ravel()         # (r,)


@dataclass
class DEIMOperator:
    """Everything the online phase needs, fully assembled offline.

    projector:     (l, r)  B = (-1/dx) * Phi^T @ D @ Phi_nl @ inv(Phi_nl[S])
    stencil_basis: (p, l)  the p stencil rows of (E @ Phi)
    local_stencil: (r, 4)  per-face window positions into the p sampled cells
    """

    projector: np.ndarray
    stencil_basis: np.ndarray
    local_stencil: np.ndarray

    def reduced_rhs(self, a: np.ndarray) -> np.ndarray:
        """da/dt = B @ f_num(stencil_basis @ a), evaluated at r faces only.

        The hyper-reduced counterpart of galerkin.reduced_rhs: reconstructs
        only the p sampled cells (not the full n-cell state), evaluates only
        the r selected face fluxes, and projects with the precomputed B --
        cost O(p*l + r*l), independent of n.
        """

        u_stencil = self.stencil_basis @ a
        f_hat = sampled_face_fluxes(u_stencil, self.local_stencil)
        return self.projector @ f_hat


def build_deim_operator(modes: np.ndarray, U: np.ndarray, dx: float, r: int) -> DEIMOperator:
    """Offline assembly: run the whole DEIM pipeline and return the operator.

    Steps (each one of the functions above):
      1. F_snap  = build_flux_snapshot_matrix(U)            (n+1, m)
      2. Phi_nl  = compute_deim_basis(F_snap, r)            (n+1, r)
      3. S       = select_deim_points(Phi_nl)               (r,)
      4. B       = (-1/dx) * modes.T @ D @ Phi_nl @ inv(Phi_nl[S])   (l, r)
      5. stencil_cells, local_stencil = build_stencil_maps(S, n)
      6. stencil_basis = (E @ modes)[stencil_cells]          (p, l)

    modes is the truncated POD basis Phi (n, l); U the training snapshots
    (n, m); r the prescribed number of DEIM points.
    """

    # 1) flux snapshot matrix
    F_snap = build_flux_snapshot_matrix(U)

    # 2) DEIM basis
    Phi_nl = compute_deim_basis(F_snap, r)

    # 3) DEIM points
    S = select_deim_points(Phi_nl, r)

    # 4) DEIM projector
    D = build_difference_matrix(U.shape[0])
    B = (-1 / dx) * modes.T @ D @ Phi_nl @ np.linalg.inv(Phi_nl[S])

    # 5) stencil maps
    stencil_cells, local_stencil = build_stencil_maps(S, U.shape[0])

    # 6) stencil basis
    E = build_extension_matrix(U.shape[0])
    stencil_basis = (E @ modes)[stencil_cells]

    return DEIMOperator(B, stencil_basis, local_stencil)
