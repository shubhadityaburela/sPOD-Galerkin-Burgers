import numpy as np
from scipy.fft import fft



def fft_luminosity(luminosity, theta, theta_pos):
    theta_pos_rad = np.deg2rad(theta_pos)
    theta_pos_rad_idx = np.argmin(np.abs(theta - theta_pos_rad))
    luminosity_per_theta = luminosity[theta_pos_rad_idx, :]

    return fft(luminosity_per_theta)


def find_frequency(freq, ampl):
    max_indices = np.argpartition(ampl, -2)[-2:]
    max_idx = np.max(max_indices)
    return freq[max_idx]