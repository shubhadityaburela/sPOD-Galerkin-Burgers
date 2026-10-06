"""Command-line entry point: run the solver and plot the result."""

import argparse

import numpy as np

from burgers1d.flux import FLUXES
from burgers1d.grid import Grid
from burgers1d.initial_conditions import INITIAL_CONDITIONS
from burgers1d.solver import SPATIAL_ORDERS, TIME_INTEGRATORS, BurgersSolver


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve the 1D Burgers' equation.")
    parser.add_argument("--n", type=int, default=200, help="Number of grid cells.")
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=1.0)
    parser.add_argument("--t-end", type=float, default=0.3, help="Final simulation time.")
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--ic", choices=list(INITIAL_CONDITIONS), default="sine",
                         help="Initial condition.")
    parser.add_argument("--boundary", choices=["periodic", "dirichlet", "outflow"],
                         default="periodic")
    parser.add_argument("--flux", choices=list(FLUXES), default="lax_friedrichs",
                         help="Numerical flux.")
    parser.add_argument("--spatial-order", choices=list(SPATIAL_ORDERS), default="first",
                         help="'first' reads raw cell values; 'second' uses MC-limited MUSCL reconstruction.")
    parser.add_argument("--time-integrator", choices=list(TIME_INTEGRATORS), default="forward_euler",
                         help="'forward_euler' or the 2-stage SSP Runge-Kutta method 'ssp_rk2'.")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting the result.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    grid = Grid(x_min=args.x_min, x_max=args.x_max, n_cells=args.n)
    u0 = INITIAL_CONDITIONS[args.ic](grid.cell_centers)
    solver = BurgersSolver(grid=grid, cfl=args.cfl, boundary=args.boundary, flux=args.flux,
                            spatial_order=args.spatial_order, time_integrator=args.time_integrator)

    u_final, t_reached = solver.solve(u0, args.t_end)
    print(f"Reached t = {t_reached:.4f} (target {args.t_end}) with {args.n} cells.")

    if not args.no_plot:
        import matplotlib.pyplot as plt

        plt.plot(grid.cell_centers, u0, "--", label="initial")
        plt.plot(grid.cell_centers, u_final, label=f"t = {t_reached:.3f}")
        plt.xlabel("x")
        plt.ylabel("u")
        plt.legend()
        plt.title("1D Burgers' equation")
        plt.show()


if __name__ == "__main__":
    main()
