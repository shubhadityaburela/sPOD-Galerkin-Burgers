# burgers1d

A small, self-contained finite-volume solver for the 1D inviscid Burgers' equation

    u_t + (u^2 / 2)_x = 0

## Layout

```
FOM/burgers1d/
├── src/
│   └── burgers1d/
│       ├── __init__.py
│       ├── grid.py            # uniform 1D grid
│       ├── initial_conditions.py
│       ├── boundary_conditions.py
│       ├── flux.py              # numerical flux functions (Lax-Friedrichs, Godunov)
│       ├── reconstruction.py    # MUSCL reconstruction with the MC limiter
│       ├── solver.py            # time-stepping solver
│       └── cli.py               # command-line entry point
├── run_example.py          # example usage / plot
└── pyproject.toml
```

## Install (editable, for development)

```bash
pip install -e .
```

## Run

```bash
python run_example.py
```

or via the installed CLI:

```bash
burgers1d --n 200 --t-end 0.5 --ic sine
burgers1d --n 200 --t-end 0.5 --ic sine --flux godunov --spatial-order second --time-integrator ssp_rk2
```

## Method

- Finite-volume discretization on a uniform grid.
- Numerical flux is either Lax-Friedrichs (simple, monotone) or Godunov (exact
  Riemann solver, see `flux.py`).
- Spatial accuracy is either first order (raw cell values at each interface) or
  second order: piecewise-linear reconstruction per cell with the MC
  (monotonized-central) TVD limiter (see `reconstruction.py`), which avoids the
  new local extrema an unlimited linear reconstruction would create near shocks.
- Time stepping is either explicit forward-Euler (first order) or the 2-stage
  SSP (strong-stability-preserving) Runge-Kutta method `ssp_rk2`, both with a
  CFL-limited step size. Pairing `spatial_order="second"` with
  `time_integrator="ssp_rk2"` gives a scheme that is second order in both space
  and time while remaining TVD.
- Periodic, Dirichlet, or outflow boundary conditions.

`flux`, `spatial_order`, and `time_integrator` are independent `BurgersSolver`
options — any flux can be paired with either spatial order or time integrator.

This is intentionally minimal — no adaptive meshing, no implicit time stepping.
It is meant to be easy to read and extend.
