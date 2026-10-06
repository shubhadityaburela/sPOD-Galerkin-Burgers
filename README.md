# sPOD-Galerkin-Burgers

Shifted-POD (sPOD) Galerkin reduced-order modeling of the 1D Burgers' equation, with a focus on diagnosing and fixing the ROM's online divergence for a traveling-shock test problem.

```
u_t + (u^2 / 2)_x = 0,   x in [0, 1],   outflow boundary conditions
```

## Motivation

Transport-dominated problems (moving shocks, traveling waves) have slowly decaying Kolmogorov n-widths, so ordinary POD-Galerkin needs many modes to represent a translating profile. sPOD factors the translation out explicitly:

```
u(t) ≈ V(z(t)) a(t) = T(z(t)) Φ a(t)
```

- `Φ`: fixed, orthonormal co-moving-frame modes (SVD of the de-shifted snapshots)
- `T(z)`: shift operator (cubic-spline interpolation, so non-integer and non-periodic shifts work)
- `a(t)`: reduced amplitudes; `z(t)`: shift, evolved as a dynamical unknown jointly with `a`

The amplitude equation gives `l` equations for `l+1` unknowns, so a **phase condition** is needed to close the system for `dz/dt`.

## Repository layout

| Path | Contents |
|---|---|
| [FOM/burgers1d/](FOM/burgers1d/) | Full-order model: finite-volume solver (MC-limited MUSCL + Godunov flux, SSP-RK2), Riemann-step initial condition |
| [ROM/burgers_rom/](ROM/burgers_rom/) | Reduced-order models: POD, sPOD, Galerkin projection, DEIM / shifted DEIM hyperreduction, error metrics |
| [sPOD/](sPOD/) | sPOD library, examples and notebooks |

## Installation

```bash
conda env create -f environment.yml
conda activate spod-galerkin-burgers
```

This installs the dependencies and both packages (`burgers1d` and `burgers_rom`) in editable mode. The sPOD library's own environment is described in [sPOD/sPOD-env.yml](sPOD/sPOD-env.yml).

## Usage

```bash
# 1. Generate FOM snapshots
python FOM/burgers1d/run_example.py

# 2. Offline stage (basis construction)
python ROM/burgers_rom/run_offline.py

# 3. Online stage (ROM time integration and error evaluation)
python ROM/burgers_rom/run_online.py
```

## Phase conditions studied

All share the same amplitude row `Vᵀ V ȧ + Vᵀ W a ż = Vᵀ F(V a)` with `W = dV/dz`, and differ only in the `ż` equation.

| Phase condition | Idea | Outcome |
|---|---|---|
| Residual | Jointly minimize the residual over `(ȧ, ż)` | Fragile: `Wa` is 93–97% redundant with `span(V)`, so the system nearly loses rank. Worst performer |
| Freeze | Minimize physical drift `‖v̇‖²` of the co-moving profile over `ż` (Beyn–Thümmler) | Good at moderate `l` |
| Freeze-tilde | Minimize the reduced rate `‖ȧ‖²` over `ż` | **Best found** (error 0.052 at `l=10`) |
| Template | Fixed test direction `q = W(z₀)a₀` (Rowley–Marsden) | Fails: `q` is only representative near `z=0`, but the shock traverses the whole domain |
| Freeze-tilde, weighted | Weighted inner product `diag(η)` | No effect: the weight carries no information beyond `Φ` |

### Online relative error vs. number of modes `l`

(1000×1000 FOM, no hyperreduction)

| l | residual | freeze | freeze-tilde | template | freeze-tilde, weighted |
|---|---|---|---|---|---|
| 3  | 0.395 | 0.416 | 1.519 | inf | 1.389 |
| 5  | 0.531 | 0.278 | 0.260 | inf | 0.261 |
| 8  | 0.705 | 0.139 | 0.154 | inf | 0.154 |
| 10 | 0.761 | 0.463 | **0.052** | 0.953 | **0.052** |
| 15 | 0.790 | 0.666 | 0.771 | inf | 0.769 |
| 20 | 0.792 | 0.671 | 0.779 | inf | 0.781 |
| 30 | 0.781 | 0.688 | 0.811 | 0.978 | 0.810 |

## Hyperreduction (shifted DEIM)

The nonlinear terms are approximated with a shift-dependent DEIM basis and offline-precomputed interpolation tables, so the online cost is independent of the full grid size.

A fixed set of sample points selected at `z=0` becomes ill-conditioned within one shift-grid step (condition number from ~8 to 1e228+), because the shock structure translates with `z`. The fix is to translate the selected indices rigidly with the shift, clamping at the domain edges instead of wrapping periodically. With `l=5`, all phase conditions diverge for ≤20 sample points and match or beat the non-hyperreduced baseline from ≥30 points.

## Mass-conservation diagnostic

Max deviation of total mass from the FOM:

| | Deviation |
|---|---|
| Offline reconstruction (true shift) | 2.2e-5 |
| Online, residual | 5.9e-1 |
| Online, freeze-tilde | 1.2e-2 |

The ansatz is not the bottleneck. The online dynamics lose conservation, since nothing in the Galerkin projection enforces it.

## Current direction and open problems

1. **Port-Hamiltonian (structure-preserving) MOR.** Use the entropy pair `H(u) = ∫u²/2 dx` (flux `u³/3`) so that outflow boundaries become ports and shock entropy production becomes the dissipative term. Prerequisites:
   - a continuous power balance for the outflow boundary conditions;
   - a split-form discretization with the entropy-conservative flux `F*(uL,uR) = (uL² + uL·uR + uR²)/6` plus a provably PSD dissipative correction (the current MUSCL/Godunov scheme likely does not admit this split);
   - a structure-preserving integrator, e.g. implicit midpoint instead of SSP-RK2.
2. **Richer group action.** Pure translation cannot represent rarefaction fans or shock formation. A translation-plus-scaling action (Rowley, Kevrekidis, Marsden, Lust 2003) is a complementary open direction.

## References

- Beyn, Thümmler (2004), freezing / phase conditions for traveling waves
- Rowley, Marsden (2000), template fitting
- Rowley, Kevrekidis, Marsden, Lust (2003), reduction of self-similar dynamics
