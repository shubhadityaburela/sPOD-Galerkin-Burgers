#!/usr/bin/env python3
"""
1D wave equation (explicit second-order) with PERIODIC boundary conditions.
PDE: u_tt = c^2 u_xx, periodic on x in [0, L].
Initial condition: Gaussian pulse (u(x,0) = exp(-(x-x0)^2 / sigma^2)), initial velocity = 0.

Usage: run the script (it will run with default parameters). Modify parameters in main() if desired.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib


def gaussian_ic(x, x0, sigma):
    return np.exp(-((x - x0) / sigma) ** 2)


def run_wave_periodic(L=1.0,
                      c_right=1.0,
                      c_left=0.8,
                      nx=401,
                      T=2.0,
                      x0=None,
                      sigma=0.05,
                      r_target=0.6,
                      v_coeff=0.5,
                      snapshot_frames=200):
    """
    Two-component periodic 1D wave simulation where the right-traveling component
    moves with speed `c_right` and the left-traveling component moves with speed
    `c_left`. The initial displacement is a single Gaussian u0, and the initial
    velocity is split so the amplitudes of right/left components follow v_coeff:
        amplitude_right  = (1 + v_coeff)/2
        amplitude_left   = (1 - v_coeff)/2
    (v_coeff=0 -> symmetric; v_coeff=1 -> purely right-moving; -1 -> purely left-moving)

    Returns: x, frames (list of combined u arrays), energies (array of total energy),
             times (array), params dict
    """
    x = np.linspace(0.0, L, nx, endpoint=False)  # periodic grid
    dx = x[1] - x[0]

    # choose dt based on the max wave speed for CFL
    max_c = max(abs(c_right), abs(c_left))
    dt = r_target * dx / max_c
    nt = int(np.ceil(T / dt))
    dt = T / nt
    r_r = c_right * dt / dx
    r_l = c_left * dt / dx
    r2_r = r_r * r_r
    r2_l = r_l * r_l

    # initial condition (single Gaussian)
    if x0 is None:
        x0 = L / 2.0
    u0 = gaussian_ic(x, x0, sigma)

    # directional amplitudes
    a_r = 0.5 * (1.0 + v_coeff)
    a_l = 0.5 * (1.0 - v_coeff)

    # split initial displacement
    u0_r = a_r * u0
    u0_l = a_l * u0

    # derivative of u0 for directional velocity construction
    du0dx = (np.roll(u0, -1) - np.roll(u0, 1)) / (2.0 * dx)

    # initial velocities for each component (pure traveling choices at their speeds)
    v0_r = -c_right * du0dx * a_r   # right-moving contribution
    v0_l = +c_left * du0dx * a_l    # left-moving contribution

    # time levels for each component
    u_prev_r = u0_r.copy()
    u_prev_l = u0_l.copy()
    u_r = np.zeros_like(u_prev_r)
    u_l = np.zeros_like(u_prev_l)

    # first time step (Taylor / central) for each component
    lap_r0 = np.roll(u_prev_r, -1) - 2.0 * u_prev_r + np.roll(u_prev_r, 1)
    lap_l0 = np.roll(u_prev_l, -1) - 2.0 * u_prev_l + np.roll(u_prev_l, 1)
    u_r[:] = u_prev_r + dt * v0_r + 0.5 * r2_r * lap_r0
    u_l[:] = u_prev_l + dt * v0_l + 0.5 * r2_l * lap_l0

    # storage (store combined field for visualization)
    frames = [ (u_prev_r + u_prev_l).copy() ]  # include t=0
    energies = []
    times = []

    # compute initial energy (sum of component energies)
    vel_r = (u_r - u_prev_r) / dt
    vel_l = (u_l - u_prev_l) / dt
    ux_r = (np.roll(u_prev_r, -1) - np.roll(u_prev_r, 1)) / (2.0 * dx)
    ux_l = (np.roll(u_prev_l, -1) - np.roll(u_prev_l, 1)) / (2.0 * dx)
    ed_r = 0.5 * (vel_r ** 2 + (c_right ** 2) * (ux_r ** 2))
    ed_l = 0.5 * (vel_l ** 2 + (c_left  ** 2) * (ux_l ** 2))
    energies.append(np.trapz(ed_r + ed_l, x))
    times.append(0.0)

    # time stepping loop: evolve both components independently and sum for output
    for n in range(1, nt + 1):
        # right component update
        lap_r = np.roll(u_r, -1) - 2.0 * u_r + np.roll(u_r, 1)
        u_next_r = 2.0 * u_r - u_prev_r + r2_r * lap_r

        # left component update
        lap_l = np.roll(u_l, -1) - 2.0 * u_l + np.roll(u_l, 1)
        u_next_l = 2.0 * u_l - u_prev_l + r2_l * lap_l

        # energies: compute each component's contribution and sum
        vel_r = (u_next_r - u_r) / dt
        vel_l = (u_next_l - u_l) / dt
        ux_next_r = (np.roll(u_next_r, -1) - np.roll(u_next_r, 1)) / (2.0 * dx)
        ux_next_l = (np.roll(u_next_l, -1) - np.roll(u_next_l, 1)) / (2.0 * dx)
        ed_r = 0.5 * (vel_r ** 2 + (c_right ** 2) * (ux_next_r ** 2))
        ed_l = 0.5 * (vel_l ** 2 + (c_left  ** 2) * (ux_next_l ** 2))
        energies.append(np.trapz(ed_r + ed_l, x))
        times.append(n * dt)

        frames.append((u_next_r + u_next_l).copy())

        # rotate time levels for each component
        u_prev_r, u_r = u_r, u_next_r
        u_prev_l, u_l = u_l, u_next_l

    # ensure final combined state included
    if not np.allclose(frames[-1], (u_r + u_l)):
        frames.append((u_r + u_l).copy())

    params = {
        'L': L, 'c_right': c_right, 'c_left': c_left, 'nx': nx,
        'dt': dt, 'nt': nt, 'r_right': r_r, 'r_left': r_l,
        'sigma': sigma, 'x0': x0, 'v_coeff': v_coeff
    }
    return x, frames, np.array(energies), np.array(times), params



def plot_and_animate(x, frames, energies, times, params, animation_interval=30):

    # animation
    fig, ax = plt.subplots()
    line, = ax.plot(x, frames[0])
    ax.set_xlim(x[0], x[-1])
    # dynamic y-limits smaller than full range for nicer view
    all_vals = np.concatenate(frames)
    yr = max(abs(all_vals.min()), abs(all_vals.max()))
    ax.set_ylim(-0.1 * yr, 1.1 * yr)
    ax.set_xlabel("x")
    ax.set_ylabel("t")

    def update(u_frame):
        line.set_ydata(u_frame)
        return (line,)

    anim = FuncAnimation(fig, update, frames=frames, interval=animation_interval, blit=True)
    plt.show()

    return anim


def provide_wave_equation_data(c_right, c_left, v_coeff, x0_factor, plot_at_all=False):
    # default parameters (change if you want)
    L = 1.0
    c_r = c_right
    c_l = c_left
    nx = 501
    T = 3.0
    sigma = 0.05
    x0 = x0_factor * L
    r_target = 0.6

    x, frames, energies, times, params = run_wave_periodic(
        L=L, c_right=c_r, c_left=c_l, nx=nx, T=T, x0=x0, sigma=sigma, r_target=r_target, snapshot_frames=300,
        v_coeff=v_coeff
    )

    print("Wave equation parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    if plot_at_all:
        # animation
        fig, ax = plt.subplots()
        ax.pcolormesh(np.asarray(frames))
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        plt.show()

        # # plot & animate (will open windows / block until closed)
        # anim = plot_and_animate(x, frames, energies, times, params)


    shift_l = (c_l * times)[::-1]
    shift_r = (c_r * times)[::-1]
    dt = times[1] - times[0]
    dx = x[1] - x[0]

    return np.array(frames).T, shift_l, shift_r, dx, dt, x, times
