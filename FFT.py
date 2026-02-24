import numpy as np
from scipy.fft import fft

np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(
    threshold=np.inf,   # print ALL elements, no truncation
    linewidth=np.inf    # do not wrap lines
)


def fft_luminosity(luminosity, theta, theta_pos):
    theta_pos_rad = np.deg2rad(theta_pos)
    theta_pos_rad_idx = np.argmin(np.abs(theta - theta_pos_rad))
    luminosity_per_theta = luminosity[theta_pos_rad_idx, :]

    return fft(luminosity_per_theta)


def find_frequency(freq, ampl, test_scenario):
    if test_scenario == "2CR":
        max_indices = np.argpartition(ampl, -2)[-2:]
        max_idx = np.max(max_indices)
        return freq[max_idx], None, None
    elif test_scenario == "2CRT":
        max_indices = np.argpartition(ampl, -5)[-5:]
        return freq[max_indices[2]], freq[max_indices[0]], None
    elif test_scenario == "DS2":
        max_indices = np.argpartition(ampl, -3)[-3:]
        return freq[max_indices[1]], freq[max_indices[0]], freq[max_indices[0]]
