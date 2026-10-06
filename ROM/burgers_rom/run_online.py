"""Online stage: march a POD/sPOD-Galerkin ROM forward in time and compare it to the FOM.

Set HYPERREDUCED below to choose the online operator:
  False -> plain Galerkin: modes.T @ L(modes @ a), exact but O(n_dof) per rhs
           evaluation (galerkin.py).
  True  -> DEIM: the hyper-reduced surrogate assembled offline by
           build_deim_operator (hyperreduction.py), touching only the sampled
           stencil cells and faces per evaluation.

Either way the spatial operator is FOM/burgers1d's flux and reconstruction
functions, so it always matches whatever discretization generated
FOM/burgers1d/data/burgers1d_snapshots.npz -- this requires burgers1d to be
installed in the same environment (pip install -e ../../FOM/burgers1d).
"""

from pathlib import Path

from burgers_rom.galerkin import compute_energy_weighted_sigma, compute_template_direction, matrices_sPOD_galerkin
from burgers_rom.spod import compute_spod_basis, subsample
import matplotlib.pyplot as plt
import numpy as np

from burgers_rom.error_metrics import frobenius_relative_error
from burgers_rom.hyperreduction import build_deim_operator
from burgers_rom.pod import compute_pod_basis
from burgers_rom.rom_solver import PODROMSolver, sPODROMSolver
from burgers_rom.snapshots import load_snapshots
from burgers_rom.spod_hyperreduction import build_spod_deim_operator

ROM = "sPOD" # POD or sPOD

if ROM == "POD":
    N_MODES = 300      # reduced dimension l, chosen directly
    HYPERREDUCED = True  # False: plain Galerkin rhs; True: DEIM-sampled rhs
    N_DEIM = 300        # number of DEIM points r (only used when HYPERREDUCED)
elif ROM == "sPOD":
    N_MODES = 5      # reduced dimension l, chosen directly
    HYPERREDUCED = False  # False: plain Galerkin rhs; True: DEIM-sampled rhs ("template" doesn't support this yet)
    N_DEIM = 15        # number of DEIM points r (only used when HYPERREDUCED)
    PRECONDITION_SPOD = False  # Jacobi-precondition the augmented (a,z) mass matrix each rhs call (galerkin.solve_lin_system)
    PHASE_CONDITION_SPOD = "freeze_tilde_weighted"  # "residual" (Psi_Res), "freeze", "freeze_tilde", "template" (fixed test direction),
                                                     # or "freeze_tilde_weighted" (weighted inner product) -- see galerkin.py
else:
    raise ValueError(f"Unknown ROM type: {ROM}")

repo_root = Path(__file__).parents[2]
snapshot_path = repo_root / "FOM" / "burgers1d" / "data" / "burgers1d_snapshots.npz"
snapshots = load_snapshots(snapshot_path)
dx = float(snapshots.x[1] - snapshots.x[0])
t_values = snapshots.t
U_fom = snapshots.U            # (n, m), one snapshot per column

# Both branches below feed into these two: tag distinguishes saved filenames,
# variant labels plot titles/legends -- computed once, here, so POD and sPOD
# (and Galerkin vs DEIM within each) are always named the same way and never
# silently overwrite each other's plots.
method_label = "Galerkin-DEIM" if HYPERREDUCED else "Galerkin"
tag = f"{ROM}_{'deim' if HYPERREDUCED else 'galerkin'}"
variant = f"{ROM}-{method_label} (l = {N_MODES}, r = {N_DEIM})" if HYPERREDUCED else f"{ROM}-{method_label} (l = {N_MODES})"

if ROM == "POD":
    full_basis = compute_pod_basis(snapshots.U)
    basis = full_basis.truncate(N_MODES)
    print(f"l = {N_MODES} modes (captures {full_basis.energy_content()[N_MODES - 1]:.2%} of total energy, informational only)")

    if HYPERREDUCED:
        deim = build_deim_operator(basis.modes, snapshots.U, dx, N_DEIM)
        print(f"DEIM: r = {N_DEIM} points, {deim.stencil_basis.shape[0]} unique stencil cells")
    else:
        deim = None

    rom_solver = PODROMSolver(modes=basis.modes, dx=dx, cfl=0.5, time_integrator="ssp_rk2", deim=deim)

    # Offline (projection-only) error: how well the truncated basis alone
    # represents every FOM snapshot, with no time integration involved --
    # the best-case error the online ROM below could ever achieve.
    U_offline = basis.modes @ (basis.modes.T @ U_fom)

    U_rom = np.empty_like(U_fom)
    a = rom_solver.project(U_fom[:, 0])
    U_rom[:, 0] = rom_solver.reconstruct(a)

    t_current = 0.0
    for i, t_next in enumerate(t_values[1:], start=1):
        a, dt_reached = rom_solver.solve(a, t_next - t_current)
        t_current += dt_reached
        U_rom[:, i] = rom_solver.reconstruct(a)

elif ROM == "sPOD":
    full_basis = compute_spod_basis(snapshots.U, snapshots.x)
    basis = full_basis.truncate(N_MODES)
    print(f"l = {N_MODES} sPOD modes (captures {full_basis.energy_content()[N_MODES - 1]:.2%} "
          f"of co-moving-frame energy, informational only)")
    
    # Subsample the shifts to match the number of snapshots for the online stage
    delta_s = subsample(snapshots.x, num_samples=basis.Nx)

    # Compute shifted basis V and W for the sPOD-Galerkin operator (see spod_galerkin.py)
    basis.make_V_W(delta_s)

    # Compute the precomputed matrices
    lhs_matrices = matrices_sPOD_galerkin(basis=basis, num_samples=len(delta_s), modes=N_MODES)    

    if HYPERREDUCED:
        deim = build_spod_deim_operator(basis, U_fom, delta_s, N_DEIM)
        print(f"sDEIM: r = {N_DEIM} points, {deim.stencil_basis.shape[1]} unique stencil cells")
    else:
        deim = None

    # "template" needs a fixed test direction, computed once offline from the
    # training trajectory's own t=0 snapshot -- see galerkin.compute_template_direction.
    if PHASE_CONDITION_SPOD == "template":
        template_q = compute_template_direction(basis, delta_s, U_fom[:, 0])
    else:
        template_q = None

    # "freeze_tilde_weighted" needs the fixed weighted-inner-product matrix --
    # see galerkin.compute_energy_weighted_sigma.
    if PHASE_CONDITION_SPOD == "freeze_tilde_weighted":
        sigma = compute_energy_weighted_sigma(basis)
    else:
        sigma = None

    rom_solver = sPODROMSolver(basis=basis, lhs_matrices=lhs_matrices, delta_s=delta_s, cfl=0.5, time_integrator="ssp_rk2",
                                precondition=PRECONDITION_SPOD, phase_condition=PHASE_CONDITION_SPOD, deim=deim,
                                template_q=template_q, sigma=sigma)

    # Offline (projection-only) error: least-squares-project every FOM
    # snapshot onto the co-moving basis at its OWN known/training shift
    # (basis.shift), with no online (a, z) time integration involved -- the
    # best-case error the online ROM below could ever achieve.
    U_offline = np.empty_like(U_fom)
    for i in range(U_fom.shape[1]):
        a_i = rom_solver.project(U_fom[:, i], basis.shift[i])
        U_offline[:, i] = rom_solver.reconstruct(a_i, basis.shift[i])

    U_rom = np.empty_like(U_fom)
    a = rom_solver.project(U_fom[:, 0], basis.shift[0])
    U_rom[:, 0] = rom_solver.reconstruct(a, basis.shift[0])
    z = basis.shift[0]

    t_current = 0.0
    for i, t_next in enumerate(t_values[1:], start=1):
        a, z, dt_reached = rom_solver.solve(a, z, t_next - t_current)
        t_current += dt_reached
        U_rom[:, i] = rom_solver.reconstruct(a, z)

offline_error = frobenius_relative_error(U_fom, U_offline)
online_error = frobenius_relative_error(U_fom, U_rom)
print(f"offline ||Q - Q~||_F / ||Q||_F = {offline_error:.4e}")
print(f"online  ||Q - Q~||_F / ||Q||_F = {online_error:.4e}")

plotpath = Path(__file__).parent / "plots"
plotpath.mkdir(exist_ok=True)

fig2, ax2 = plt.subplots(figsize=(6, 4.5))
ax2.plot(snapshots.x, U_fom[:, -1], label="FOM", linewidth=2)
ax2.plot(snapshots.x, U_rom[:, -1], "--", label="ROM", linewidth=2)
ax2.set_xlabel("x")
ax2.set_ylabel("u")
ax2.set_title(f"FOM vs {variant} at t = {t_values[-1]:.2f}")
ax2.legend()
fig2.tight_layout()
fig2.savefig(plotpath / f"rom_vs_fom_final_time_{tag}.png", dpi=150)
print(f"Saved rom_vs_fom_final_time_{tag}.png")

# --- x-t pcolormesh comparison: FOM vs ROM, same color scale ---
vmin = min(U_fom.min(), U_rom.min())
vmax = max(U_fom.max(), U_rom.max())

fig3, (ax3_fom, ax3_rom) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

mesh_fom = ax3_fom.pcolormesh(snapshots.x, t_values, U_fom.T, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
ax3_fom.set_xlabel("x")
ax3_fom.set_ylabel("t")
ax3_fom.set_title("FOM")

ax3_rom.pcolormesh(snapshots.x, t_values, U_rom.T, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
ax3_rom.set_xlabel("x")
ax3_rom.set_title(variant)

fig3.colorbar(mesh_fom, ax=[ax3_fom, ax3_rom], label="u")
fig3.suptitle("FOM vs ROM - x-t solution")
fig3.savefig(plotpath / f"rom_vs_fom_xt_{tag}.png", dpi=150)
print(f"Saved rom_vs_fom_xt_{tag}.png")
