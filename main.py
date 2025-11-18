import argparse
import os
import sys

import numpy as np

from FFT import fft_luminosity, find_frequency
from Plots import plot_luminosity_1d, plot_fft_luminosity, plot_fft_pressure
from read_matlab_files import read_matlab, data_luminosity, data_pressure


def parse_arguments():
    p = argparse.ArgumentParser(description="Input the variables for running the script.")
    p.add_argument("dir_prefix", type=str, help="Directory prefix for MATLAB files input")
    p.add_argument("which_variable", type=str, choices=["Luminosity", "Pressure"],
                   help="Select either the luminosity or the pressure data")
    p.add_argument("test_scenario", type=str, choices=["2CR", "2CRT", "DS2"],
                   help="Input test scenario")
    return p.parse_args()


def select_variable(which_variable):
    if which_variable == "Luminosity":
        return "00_hsv"
    elif which_variable == "Pressure":
        return "01_pressure"
    else:
        return "00_hsv"


def select_filename(test_scenario):
    if test_scenario == "2CR":
        print("Test case: Two counter rotating waves with equal speed")
        return "BD0041.mat"
    elif test_scenario == "2CRT":
        print(
            "Test case: Two counter rotating waves traveling at different speed, where the faster one is larger in amplitude")
        return "BD0038.mat"
    elif test_scenario == "DS2":
        print("Test case: One dominant wave and a set of two counter rotating traveling waves.")
        return "CE2029.mat"
    else:
        print("Default test case: Two counter rotating waves with equal speed")
        return "BD0041.mat"


if __name__ == "__main__":
    args = parse_arguments()

    dir = args.dir_prefix
    variable_folder = select_variable(args.which_variable)
    file_name = select_filename(args.test_scenario)

    # Read the MATLAB files
    matlab_files = read_matlab(dir, variable_folder, file_name)

    if args.which_variable == "Luminosity":
        RDC_data = data_luminosity(matlab_files)

        # Plot the luminosity snapshot data
        plot_luminosity_1d(RDC_data.lumCenter.T, time_window_length=200, plot_at_all=False)

        # Perform FFT on the data
        luminosity_fft = fft_luminosity(RDC_data.lumCenter.T, theta=RDC_data.theta, theta_pos=60)

        # Plot the FFT luminosity
        freq, ampl = plot_fft_luminosity(luminosity_fft, RDC_data.m, RDC_data.dt)

        # Finding the most occurring Frequency
        most_occur_freq = find_frequency(freq, ampl)

        # Compute the velocity of the wave
        ang_velocity = 2 * np.pi * most_occur_freq * 1000  # Multiply 1000 to convert to Hz
        velocity = RDC_data.RA * ang_velocity / 1000   # Divide by 1000 for mm/sec -> m/sec

        shift = velocity * (RDC_data.dt * np.arange(RDC_data.m))

    elif args.which_variable == "Pressure":
        RDC_data = data_pressure(matlab_files)

        # Plot the FFT pressure
        freq, ampl = plot_fft_pressure(RDC_data.pressureProbeFreqMat.T, RDC_data.pressureProbeAmpMat.T,
                                       RDC_data.ProbePos,
                                       desired_probe_pos=60, freq_cut_off=45)

        # Finding the most occurring Frequency
        most_occur_freq = find_frequency(freq, ampl)

        # Compute the velocity of the wave
        ang_velocity = 2 * np.pi * most_occur_freq * 1000  # Multiply 1000 to convert to Hz
        velocity = RDC_data.RA * ang_velocity / 1000   # Divide by 1000 for mm/sec -> m/sec







