import numpy as np
from scipy.sparse import spdiags, diags, linalg, csc_matrix
from scipy.linalg import cholesky, lu_factor, lu_solve


def give_spline_coefficient_matrices(Nxi):
    """
    This function returns the coefficient matrices needed for determining the
    spline coefficients for a cubic spline with periodic boundary conditions.

    On each sub-interval [x_{i-1},x_i] the spline has the form
          f_i(x) = a_i+b_i(x-x_i)+c_i(x-x_i)^2+d_i(x-x_i)^3.
    Given a vector y which contains sample data at the grid points with
    uniform grid size h, the spline coefficients may be computed as follows:

    - Solve the linear system M*c = (3/h^2)*D2*y for c
    - Set a=y
    - Set d=(1/(3h))*D1*c
    - Set b=(1/h)*D1*y+(h/3)*A1*c,

    cf. Appendix D.1.1 of the PhD thesis "Energy-based Model Reduction of
    Transport-dominated phenomena".

    Inputs:
    - Nxi: number of grid points

    Outputs:
    - A1: coefficient matrix of size Nxi x Nxi
    - D1: coefficient matrix of size Nxi x Nxi
    - D2: coefficient matrix of size Nxi x Nxi
    - R: Cholesky factor of the matrix M (size Nxi x Nxi)
    """

    # Initialize matrices
    # Create the A1 matrix
    diagonals = np.ones((Nxi, 1)) * np.array([1, 2, 1])
    offsets = [-1, 0, Nxi - 1]
    A1 = spdiags(diagonals.T, offsets, Nxi, Nxi).tocsc()

    # Create the D1 matrix
    diagonals = np.ones((Nxi, 1)) * np.array([-1, 1, -1])
    D1 = spdiags(diagonals.T, offsets, Nxi, Nxi).tocsc()

    # Create the D2 matrix
    diagonals = np.ones((Nxi, 1)) * np.array([1, 1, -2, 1, 1])
    offsets = [-(Nxi - 1), -1, 0, 1, Nxi - 1]
    D2 = spdiags(diagonals.T, offsets, Nxi, Nxi).tocsc()

    # Create the M matrix and compute its Cholesky decomposition
    diagonals = np.ones((Nxi, 1)) * np.array([1, 1, 4, 1, 1])
    M = diags(diagonals.T, offsets, shape=(Nxi, Nxi)).toarray()
    R = csc_matrix(cholesky(M))

    return A1, D1, D2, R


def construct_spline_coeffs_multiple(qs, A1, D1, D2, R, dxi):
    """
    This function determines the spline coefficients for a given vector or
    matrix of sample data qs, where a cubic spline with periodic boundary
    conditions is assumed.

    On each sub-interval [x_{i-1},x_i] the spline has the form
          f_i(x) = a_i+b_i(x-x_i)+c_i(x-x_i)^2+d_i(x-x_i)^3.
    Given a vector y which contains sample data at the grid points with
    uniform grid size h, the spline coefficients may be computed as follows:

    - Solve the linear system M*c = (3/h^2)*D2*y for c
    - Set a=y
    - Set d=(1/(3h))*D1*c
    - Set b=(1/h)*D1*y+(h/3)*A1*c,

    cf. Appendix D.1.1 of the PhD thesis "Energy-based Model Reduction of
    Transport-dominated phenomena".

    Inputs:
    - qs: array of size mxn where m is the number of grid points and n the number of data sets
    - A1: coefficient matrix of size Nxi x Nxi
    - D1: coefficient matrix of size Nxi x Nxi
    - D2: coefficient matrix of size Nxi x Nxi
    - R: Cholesky factor of the matrix M (size Nxi x Nxi)
    - dxi: grid size

    Outputs:
    - splineCoeffs: struct array containing the spline coefficients for each
    data set: splineCoeffs_b, splineCoeffs_c, splineCoeffs_d
    (each of them has the same dimension as qs)
    """

    n = qs.shape[1]
    spline_coeffs_b = np.zeros_like(qs)
    spline_coeffs_c = np.zeros_like(qs)
    spline_coeffs_d = np.zeros_like(qs)

    c_fac = (3 / dxi ** 2)
    b1_fac = (1 / dxi)
    b2_fac = (dxi / 3)
    d_fac = (1 / (3 * dxi))

    for i in range(n):
        # Assign the solved result to spline_coeffs
        aa = D2 @ qs[:, i]
        aa = linalg.spsolve(R.T, aa)
        spline_coeffs_c[:, i] = c_fac * linalg.spsolve(R, aa)
        spline_coeffs_b[:, i] = b1_fac * (D1 @ qs[:, i]) + b2_fac * (A1 @ spline_coeffs_c[:, i])
        spline_coeffs_d[:, i] = d_fac * (D1 @ spline_coeffs_c[:, i])

    return spline_coeffs_b, spline_coeffs_c, spline_coeffs_d



def shift_matrix_precomputed_coeffs_multiple(qs, cs, spline_coeffs_b, spline_coeffs_c, spline_coeffs_d, Nxi, dxi):
    """
    This function determines a shifted version of a data set using cubic
    spline interpolation with periodic boundary conditions.

    On each sub-interval [x_{i-1},x_i] the spline has the form
          f_i(x) = a_i+b_i(x-x_i)+c_i(x-x_i)^2+d_i(x-x_i)^3.

    Inputs:
    - qs: array of size mxn where m is the number of grid points and n the number of data sets
    - cs: scalar or array of size p or array of size n containing the shift amount(s)
    - splineCoeffs_b, splineCoeffs_c, splineCoeffs_d: containing the spline coefficients for
    each data set: splineCoeffs.b, splineCoeffs.c, splineCoeffs.d
    (each of them has size mxn where m is the number of grid
    points and n the number of data sets)
    - Nxi: number of grid points
    - dxi: grid size

    Outputs:
    - qsShifted: shifted version of original data set(s) (if there are
    multiple data sets, i.e. n>1, then qsShifted is an array of size mxn;
    otherwise qsShifted is an array of size mxp where p is the number of
    shift amounts)
    """

    n = qs.shape[1]

    # Length of computational domain
    L = Nxi * dxi

    # Compute csTilde = cs modulo L
    q1 = np.floor(cs / L)
    csTilde = cs - q1 * L
    # Compute zeta = csTilde modulo dxi
    q2 = np.floor(csTilde / dxi)

    zeta = csTilde - q2 * dxi
    zeta2 = zeta ** 2
    zeta3 = zeta ** 3

    qsShifted = np.zeros((Nxi, n))

    for i in range(n):
        # Compute interpolated data set
        qsInterpolated = qs[:, i] - zeta[i] * spline_coeffs_b[:, i] + zeta2[i] * spline_coeffs_c[:, i] \
                         - zeta3[i] * spline_coeffs_d[:, i]
        # Compute shifted data set
        qsShifted[:, i] = np.concatenate((qsInterpolated[Nxi - int(q2[i]):], qsInterpolated[:Nxi - int(q2[i])]), axis=0)

    return qsShifted



def shifted_U(U, cs, spline_coeffs_b, spline_coeffs_c, spline_coeffs_d, Nxi, dxi):
    """
    This function determines a shifted version of a data set using cubic
    spline interpolation with periodic boundary conditions.

    On each sub-interval [x_{i-1},x_i] the spline has the form
          f_i(x) = a_i+b_i(x-x_i)+c_i(x-x_i)^2+d_i(x-x_i)^3.

    Inputs:
    - U: array of size mxr where m is the number of grid points and r the number of modes
    - cs: scalar the shift amount
    - splineCoeffs_b, splineCoeffs_c, splineCoeffs_d: containing the spline coefficients for
    each data set: splineCoeffs.b, splineCoeffs.c, splineCoeffs.d
    (each of them has size mxr where m is the number of grid
    points and r the number of modes)
    - Nxi: number of grid points
    - dxi: grid size

    Outputs:
    - T^z(U): shifted version of stationary modes U (if there are
    multiple modes, i.e. r>1, then T^z(U) is an array of size mxr;)
    """

    r = U.shape[1]

    # Length of computational domain
    L = Nxi * dxi

    # Compute csTilde = cs modulo L
    q1 = np.floor(cs / L)
    csTilde = cs - q1 * L
    # Compute zeta = csTilde modulo dxi
    q2 = np.floor(csTilde / dxi)
    zeta = csTilde - q2 * dxi
    zeta2 = zeta ** 2
    zeta3 = zeta ** 3

    UShifted = np.zeros((Nxi, r))

    for i in range(r):
        # Compute interpolated data set
        UInterpolated = U[:, i] - zeta * spline_coeffs_b[:, i] + zeta2 * spline_coeffs_c[:, i] \
                         - zeta3 * spline_coeffs_d[:, i]
        # Compute shifted data set
        UShifted[:, i] = np.concatenate((UInterpolated[Nxi - int(q2):], UInterpolated[:Nxi - int(q2)]), axis=0)

    return UShifted


# -----------------------------------------------------------------------
# Open-domain (transmissive) cubic spline shift.
#
# The four functions above are all periodic: give_spline_coefficient_matrices
# builds A1/D1/D2/M as CIRCULANT matrices (the offset "Nxi-1" diagonals
# couple knot 0 to knot Nxi-1), and shift_matrix_precomputed_coeffs_multiple/
# shifted_U realize the integer part of the shift with np.concatenate, i.e.
# a circular roll. Both steps assume the field is periodic; for outflow/
# transmissive data (e.g. a step function) this wraps whatever exits one
# edge back in at the other, reintroducing a discontinuity that was never
# physically there.
#
# The functions below are the open-domain analogue, built the same way but
# with the periodicity removed at its two points of entry:
#   1) give_open_spline_coefficient_matrices uses a NATURAL spline (zero
#      curvature at both ends, c_0 = c_{Nxi-1} = 0) instead of periodic
#      wrap, so the boundary rows never reference the opposite edge of the
#      domain. The natural-BC system matrix is no longer symmetric (its two
#      boundary rows are plain identity rows), so it is solved with a plain
#      LU factorization instead of the periodic version's Cholesky trick.
#   2) shift_open_matrix_precomputed_coeffs_multiple / shifted_U_open
#      replace the circular roll with explicit clamping to the boundary
#      VALUE (qs[0]/qs[-1]) for any query point that falls outside the
#      domain -- the direct spline analogue of the FOM's outflow BC.
# -----------------------------------------------------------------------


def give_open_spline_coefficient_matrices(Nxi):
    """
    Coefficient matrices for a NATURAL (non-periodic) cubic spline -- the
    open-domain analogue of give_spline_coefficient_matrices.

    Interior rows (i=1,...,Nxi-2) use the identical recurrence as the
    periodic version:
        c_{i-1} + 4c_i + c_{i+1} = (3/h^2)(y_{i-1} - 2y_i + y_{i+1})
        d_i = (1/(3h))(c_i - c_{i-1})
        b_i = (1/h)(y_i - y_{i-1}) + (h/3)(c_{i-1} + 2c_i)
    The two boundary rows impose zero curvature at the ends instead of the
    periodic wrap: c_0 = 0, c_{Nxi-1} = 0. Row 0 of A1/D1 is left as zero
    and simply unused -- there is no spline piece to the left of x_0 on an
    open domain (piece i covers [x_{i-1}, x_i], and there is no x_{-1}).

    Inputs:
    - Nxi: number of grid points

    Outputs:
    - A1, D1: (Nxi, Nxi) coefficient matrices (row 0 unused)
    - D2: (Nxi, Nxi) coefficient matrix building the RHS of the c system
      (rows 0 and Nxi-1 are zero -- c_0, c_{Nxi-1} are fixed directly by
      the natural BC, not by the 2nd-difference recurrence)
    - M_lu: LU factorization (scipy.linalg.lu_factor) of the natural-spline
      system matrix M, reused for every right-hand side
    """
    ones_full = np.ones(Nxi)
    ones_off = np.ones(Nxi - 1)

    M = np.diag(4.0 * ones_full) + np.diag(ones_off, 1) + np.diag(ones_off, -1)
    M[0, :] = 0.0
    M[0, 0] = 1.0
    M[-1, :] = 0.0
    M[-1, -1] = 1.0

    D2 = np.diag(-2.0 * ones_full) + np.diag(ones_off, 1) + np.diag(ones_off, -1)
    D2[0, :] = 0.0
    D2[-1, :] = 0.0

    A1 = np.diag(2.0 * ones_full) + np.diag(ones_off, -1)
    A1[0, :] = 0.0

    D1 = np.diag(ones_full) + np.diag(-ones_off, -1)
    D1[0, :] = 0.0

    M_lu = lu_factor(M)

    return A1, D1, D2, M_lu


def construct_open_spline_coeffs_multiple(qs, A1, D1, D2, M_lu, dxi):
    """
    Natural-spline coefficients for qs -- open-domain analogue of
    construct_spline_coeffs_multiple.

    Row 0 of the returned b, c, d is unused (there is no spline piece to
    the left of x_0); shift_open_matrix_precomputed_coeffs_multiple /
    shifted_U_open fill the corresponding sample with the boundary value
    qs[0] directly instead.

    Inputs:
    - qs: array of size Nxi x n, n data sets
    - A1, D1, D2, M_lu: from give_open_spline_coefficient_matrices
    - dxi: grid size

    Outputs:
    - spline_coeffs_b, spline_coeffs_c, spline_coeffs_d: each Nxi x n
    """
    c_fac = 3 / dxi ** 2
    b1_fac = 1 / dxi
    b2_fac = dxi / 3
    d_fac = 1 / (3 * dxi)

    rhs = c_fac * (D2 @ qs)
    spline_coeffs_c = lu_solve(M_lu, rhs)  # all columns solved in one shot
    spline_coeffs_b = b1_fac * (D1 @ qs) + b2_fac * (A1 @ spline_coeffs_c)
    spline_coeffs_d = d_fac * (D1 @ spline_coeffs_c)

    return spline_coeffs_b, spline_coeffs_c, spline_coeffs_d


def shift_open_matrix_precomputed_coeffs_multiple(qs, cs, spline_coeffs_b, spline_coeffs_c, spline_coeffs_d, Nxi, dxi):
    """
    Shift each column of qs by its own amount cs[i], with transmissive
    (constant-extrapolation) boundaries instead of the periodic wrap of
    shift_matrix_precomputed_coeffs_multiple.

    There is no domain length to reduce cs modulo -- an open domain has no
    periodicity -- so cs is used directly: q2 = floor(cs/h) is the integer
    number of cells shifted, zeta = cs - q2*h the sub-cell remainder. Each
    knot's sub-cell-shifted value is built with the same cubic formula as
    the periodic version, except:
      - knot 0 has no spline piece to its left, so its sub-cell value is
        just qs[:, i][0] -- the left boundary VALUE (flat extrapolation);
      - after applying the integer shift q2 (by index arithmetic instead of
        an array roll), any index that lands beyond the right edge is set
        to the right boundary value qs[:, i][-1] rather than wrapping back
        in from the left.

    Inputs / outputs: identical shapes/meaning to
    shift_matrix_precomputed_coeffs_multiple.
    """
    n = qs.shape[1]
    q2 = np.floor(cs / dxi).astype(int)
    zeta = cs - q2 * dxi
    zeta2 = zeta ** 2
    zeta3 = zeta ** 3

    qsShifted = np.zeros((Nxi, n))
    base_idx = np.arange(Nxi)

    for i in range(n):
        qsInterpolated = (qs[:, i] - zeta[i] * spline_coeffs_b[:, i]
                           + zeta2[i] * spline_coeffs_c[:, i] - zeta3[i] * spline_coeffs_d[:, i])
        qsInterpolated[0] = qs[0, i]  # no piece 0: clamp to left boundary value

        idx = base_idx - q2[i]
        shifted = qsInterpolated[np.clip(idx, 0, Nxi - 1)]
        shifted[idx > Nxi - 1] = qs[-1, i]  # off the right edge: clamp to right boundary value
        qsShifted[:, i] = shifted

    return qsShifted


def shifted_U_open(U, cs, spline_coeffs_b, spline_coeffs_c, spline_coeffs_d, Nxi, dxi):
    """
    Open-domain / transmissive analogue of shifted_U: shifts every column
    (mode) of U by the same scalar amount cs, clamping to the boundary
    VALUE of that column instead of wrapping the array. See
    shift_open_matrix_precomputed_coeffs_multiple for the full reasoning.

    Inputs / outputs: identical shapes/meaning to shifted_U.
    """
    r = U.shape[1]
    q2 = int(np.floor(cs / dxi))
    zeta = cs - q2 * dxi
    zeta2 = zeta ** 2
    zeta3 = zeta ** 3

    idx = np.arange(Nxi) - q2
    idx_clipped = np.clip(idx, 0, Nxi - 1)
    off_right = idx > Nxi - 1

    UShifted = np.zeros((Nxi, r))
    for i in range(r):
        UInterpolated = (U[:, i] - zeta * spline_coeffs_b[:, i]
                          + zeta2 * spline_coeffs_c[:, i] - zeta3 * spline_coeffs_d[:, i])
        UInterpolated[0] = U[0, i]  # no piece 0: clamp to left boundary value

        shifted = UInterpolated[idx_clipped]
        shifted[off_right] = U[-1, i]  # off the right edge: clamp to right boundary value
        UShifted[:, i] = shifted

    return UShifted


def first_derivative_shifted_U_open(cs, spline_coeffs_b, spline_coeffs_c, spline_coeffs_d, Nxi, dxi):
    """
    Open-domain / transmissive analogue of first_derivative_shifted_U:
    the first derivative (w.r.t. the shift amount, equivalently minus the
    spatial derivative of the shifted field -- d/dcs[u(x-cs)] = -u'(x-cs))
    of T^cs[U], with the same constant-extrapolation boundaries as
    shifted_U_open instead of the periodic wrap.

    On each sub-interval [x_{i-1},x_i] the spline has the form
          f_i(x) = a_i+b_i(x-x_i)+c_i(x-x_i)^2+d_i(x-x_i)^3
    and its derivative is
          f_i'(x) = b_i+2*c_i(x-x_i)+3*d_i(x-x_i)^2.

    There is no domain length to reduce cs modulo -- see
    shift_open_matrix_precomputed_coeffs_multiple -- so cs is used directly.
    Beyond either edge the field is flat (clamped to its boundary VALUE, per
    shifted_U_open), and the derivative of a constant is zero: unlike
    shifted_U_open (which clamps to U[0]/U[-1]), both boundary regions here
    are set to 0 -- knot 0 has no spline piece to draw a derivative from in
    the first place, and any index shifted past the right edge sits in the
    same flat-extrapolated region. See second_derivative_shifted_U_open for
    the same fix applied to the second derivative.

    Inputs:
    - cs: scalar containing the shift amount
    - splineCoeffs_b, splineCoeffs_c, splineCoeffs_d: containing the spline coefficients for
    each data set: splineCoeffs.b, splineCoeffs.c, splineCoeffs.d
    (each of them has size mxr where m is the number of grid
    points and r the number of modes)
    - Nxi: number of grid points
    - dxi: grid size

    Outputs:
    - (T^cs)' U: derivative of shifted data set(s) with respect to the shift
    amount, shape (Nxi, r)
    """

    r = spline_coeffs_b.shape[1]

    q2 = int(np.floor(cs / dxi))
    zeta = cs - q2 * dxi
    zeta2 = zeta ** 2

    idx = np.arange(Nxi) - q2
    idx_clipped = np.clip(idx, 0, Nxi - 1)
    off_right = idx > Nxi - 1

    dUShifted = np.zeros((Nxi, r))
    for i in range(r):
        dUInterpolated = (- spline_coeffs_b[:, i] + 2 * zeta * spline_coeffs_c[:, i]
                           - 3 * zeta2 * spline_coeffs_d[:, i])
        dUInterpolated[0] = 0.0  # no piece 0: flat-extrapolated region, zero derivative

        shifted = dUInterpolated[idx_clipped]
        shifted[off_right] = 0.0  # off the right edge: flat-extrapolated region, zero derivative
        dUShifted[:, i] = shifted

    return dUShifted


def second_derivative_shifted_U_open(cs, spline_coeffs_b, spline_coeffs_c, spline_coeffs_d, Nxi, dxi):
    """
    Open-domain / transmissive analogue of second_derivative_shifted_U:
    the second derivative (w.r.t. the shift amount, equivalently the second
    spatial derivative of the shifted field -- d^2/dcs^2[u(x-cs)] = u''(x-cs))
    of T^cs[U], with the same constant-extrapolation boundaries as
    shifted_U_open instead of the periodic wrap.

    On each sub-interval [x_{i-1},x_i] the spline has the form
          f_i(x) = a_i+b_i(x-x_i)+c_i(x-x_i)^2+d_i(x-x_i)^3
    and its second derivative is
          f_i"(x) = 2*c_i + 6*d_i(x-x_i).

    There is no domain length to reduce cs modulo -- see
    shift_open_matrix_precomputed_coeffs_multiple -- so cs is used directly.
    Beyond either edge the field is flat (clamped to its boundary VALUE, per
    shifted_U_open), and the derivative of a constant is zero: unlike
    shifted_U_open (which clamps to U[0]/U[-1]), both boundary regions here
    are set to 0 -- knot 0 has no spline piece to draw a second derivative
    from in the first place, and any index shifted past the right edge sits
    in the same flat-extrapolated region.

    Inputs:
    - cs: scalar containing the shift amount
    - splineCoeffs_b, splineCoeffs_c, splineCoeffs_d: containing the spline coefficients for
    each data set: splineCoeffs.b, splineCoeffs.c, splineCoeffs.d
    (each of them has size mxr where m is the number of grid
    points and r the number of modes)
    - Nxi: number of grid points
    - dxi: grid size

    Outputs:
    - (T^cs)" U: second derivative of shifted data set(s) with respect to the
    shift amount, shape (Nxi, r)
    """

    r = spline_coeffs_b.shape[1]

    q2 = int(np.floor(cs / dxi))
    zeta = cs - q2 * dxi

    idx = np.arange(Nxi) - q2
    idx_clipped = np.clip(idx, 0, Nxi - 1)
    off_right = idx > Nxi - 1

    ddUShifted = np.zeros((Nxi, r))
    for i in range(r):
        ddUInterpolated = 2 * spline_coeffs_c[:, i] - 6 * zeta * spline_coeffs_d[:, i]
        ddUInterpolated[0] = 0.0  # no piece 0: flat-extrapolated region, zero derivative

        shifted = ddUInterpolated[idx_clipped]
        shifted[off_right] = 0.0  # off the right edge: flat-extrapolated region, zero derivative
        ddUShifted[:, i] = shifted

    return ddUShifted