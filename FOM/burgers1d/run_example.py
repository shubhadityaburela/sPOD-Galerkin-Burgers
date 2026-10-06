"""Minimal example: solve Burgers' equation from a sine wave and plot the shock forming."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from burgers1d import Grid, BurgersSolver
from burgers1d.initial_conditions import gaussian, riemann_step, sine, smoothed_step

grid = Grid(x_min=0.0, x_max=1.0, n_cells=1000)
solver = BurgersSolver(grid=grid, cfl=0.5, boundary="outflow", flux="godunov", spatial_order="second", time_integrator="ssp_rk2")

u0 = riemann_step(grid.cell_centers, x_jump=0.05)

impath = Path(__file__).parent / "plots"
impath.mkdir(exist_ok=True)

datapath = Path(__file__).parent / "data"
datapath.mkdir(exist_ok=True)

# --- x-t pcolormesh over 1000 time instances in [0, 1] ---
# March the state forward in time, reusing it as the starting point for the
# next snapshot instead of re-solving from u0 every time.
t_values = np.linspace(0.0, 1.8, 1000)
U = np.empty((t_values.size, grid.n_cells))
U[0] = u0

u = u0
t_current = 0.0
for i, t_next in enumerate(t_values[1:], start=1):
    u, dt_reached = solver.solve(u, t_next - t_current)
    t_current += dt_reached
    U[i] = u

np.savez(datapath / "burgers1d_snapshots.npz", x=grid.cell_centers, t=t_values, U=U)
print("Saved burgers1d_snapshots.npz")

fig, ax = plt.subplots(figsize=(7, 5))
mesh = ax.pcolormesh(grid.cell_centers, t_values, U, shading="auto", cmap="viridis")
fig.colorbar(mesh, ax=ax, label="u")
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_title("1D Burgers' equation - x-t solution")
fig.tight_layout()

fig.savefig(impath / "burgers1d_xt.png", dpi=150)
print("Saved burgers1d_xt.png")

# --- animation of the wave evolving over the same 500 time instances ---
anim_fig, anim_ax = plt.subplots(figsize=(7, 5))
line, = anim_ax.plot(grid.cell_centers, U[0], color="tab:blue")
anim_ax.set_xlim(grid.x_min, grid.x_max)
anim_ax.set_ylim(U.min() - 0.05 * abs(U.min()), U.max() + 0.05 * abs(U.max()))
anim_ax.set_xlabel("x")
anim_ax.set_ylabel("u")
title = anim_ax.set_title(f"1D Burgers' equation - t = {t_values[0]:.3f}")


def update(frame):
    line.set_ydata(U[frame])
    title.set_text(f"1D Burgers' equation - t = {t_values[frame]:.3f}")
    return line, title


anim = animation.FuncAnimation(anim_fig, update, frames=t_values.size, interval=30, blit=False)
anim.save(impath / "burgers1d_evolution.mp4", writer="ffmpeg", dpi=150)
print("Saved burgers1d_evolution.mp4")
