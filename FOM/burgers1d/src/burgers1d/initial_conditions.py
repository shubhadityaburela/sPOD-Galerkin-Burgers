"""Initial conditions for the Burgers' equation."""

import numpy as np


def sine(x: np.ndarray) -> np.ndarray:
    """Smooth sine profile on x in [0, 1], useful for observing shock formation."""
    return 0.5 + np.sin(2 * np.pi * x)


def riemann_step(x: np.ndarray, x_jump: float = 0.1, u_left: float = 1.0, u_right: float = 0.0) -> np.ndarray:
    """A single downward or upward step, i.e. a Riemann problem."""
    return np.where(x < x_jump, u_left, u_right)


def smoothed_step(x: np.ndarray, x_jump: float = 0.1, u_left: float = 0.8, u_right: float = 0.7,
                   width: float = 0.05) -> np.ndarray:
    """riemann_step's jump, smoothed into a tanh transition of the given width
    (width -> 0 recovers the discontinuous step).

    For the default (downward, u_left > u_right) case this is still
    compressive -- characteristics converge and it shocks eventually -- but
    since it STARTS smooth rather than already discontinuous at t=0, there is
    a finite pre-shock window. The transition midpoint (and, by Rankine-
    Hugoniot, the shock once it forms) moves at the constant average speed
    ubar = (u_left + u_right) / 2, so:

        t_shock  = 2 * width / (u_left - u_right)
        distance = ubar * t_shock = width * (u_left + u_right) / (u_left - u_right)

    Getting a long pre-shock travel distance from a NARROW width therefore
    needs a small jump (u_left - u_right) riding on a large mean speed
    (u_left + u_right) / 2, rather than simply widening the transition (which
    would just erase the flat plateaus and stop looking like a step at all).
    These defaults (a 0.1 jump on a 0.75 mean speed, width=0.05) give
    t_shock = 1.0 and a pre-shock travel distance of 0.75 -- three quarters
    of the [0, 1] domain -- while keeping a narrow, visibly step-like profile.
    """
    return u_right + (u_left - u_right) * 0.5 * (1.0 - np.tanh((x - x_jump) / width))


def gaussian(x: np.ndarray, amplitude: float = 0.5, x0: float = 0.1, width: float = 0.18) -> np.ndarray:
    """A smooth, amplitude-scaled Gaussian bump.

    For u_t + (u^2/2)_x = 0, characteristics leave x0 at speed u0(x0), so the
    shock formation time is t_shock = width * sqrt(e) / amplitude, and the
    peak (which moves at constant speed = amplitude until the shock forms)
    travels a distance of width * sqrt(e) ~= 1.6487 * width before shocking --
    notably independent of amplitude. With these defaults (x0=0.1, width=0.18)
    that's a ~0.30 pre-shock travel distance (t_shock ~= 0.59), a middle
    ground between a narrow pulse and one that stays shock-free across the
    whole domain (which would need width ~= 0.5-0.6, i.e. not narrow at all).
    Useful for sanity checks that specifically require a smooth,
    discontinuity-free profile for at least part of the run -- see
    ROM/burgers_rom/tests/investigate_spod_smooth_gaussian_sanity_check.py.
    """
    return amplitude * np.exp(-((x - x0) ** 2) / (2 * width**2))


INITIAL_CONDITIONS = {
    "sine": sine,
    "riemann": riemann_step,
    "smoothed_step": smoothed_step,
    "gaussian": gaussian,
}
