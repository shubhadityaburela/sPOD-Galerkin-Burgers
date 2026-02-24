import argparse
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

from FFT import fft_luminosity, find_frequency
from Plots import plot_luminosity_1d, plot_fft_luminosity, plot_fft_pressure, plot_sPOD_1D_frames, PlotFOM2D
from read_matlab_files import read_matlab, data_luminosity, data_pressure
from sPOD_calls import sPOD_1D, sPOD_2D
from wave_equation import provide_wave_equation_data


def parse_arguments():
    p = argparse.ArgumentParser(description="Input the variables for running the script.")
    p.add_argument("dir_prefix", type=str, help="Directory prefix for MATLAB files input")
    p.add_argument("which_variable", type=str, choices=["Luminosity", "Pressure", "Custom"],
                   help="Select either the luminosity or the pressure data")
    p.add_argument("test_scenario", type=str, choices=["2CR", "2CRT", "DS2", ""],
                   help="Input test scenario")
    p.add_argument("sPOD_dim", type=str, choices=["1D", "2D"],
                   help="Input the sPOD dimension")
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
        print("Default test case: Custom wave equation")
        return "BD0041.mat"


if __name__ == "__main__":
    args = parse_arguments()

    dir = args.dir_prefix
    variable_folder = select_variable(args.which_variable)
    file_name = select_filename(args.test_scenario)

    if args.which_variable == "Luminosity":
        # Read the MATLAB files
        matlab_files = read_matlab(dir, variable_folder, file_name)

        RDC_data = data_luminosity(matlab_files)

        # Plot the luminosity snapshot data
        plot_luminosity_1d(RDC_data.lumCenter.T, time_window_length=200, plot_at_all=False)

        # Perform FFT on the data
        if args.test_scenario == "2CR" or args.test_scenario == "2CRT":
            luminosity_fft = fft_luminosity(RDC_data.lumCenter.T, theta=RDC_data.theta, theta_pos=60)
        else:
            luminosity_fft = fft_luminosity(RDC_data.lumCenter.T, theta=RDC_data.theta, theta_pos=300)

        # Plot the FFT luminosity
        freq, ampl = plot_fft_luminosity(luminosity_fft, RDC_data.m, RDC_data.dt, plot_at_all=False)

        # Finding the most occurring Frequency
        first_freq, second_freq, third_freq = find_frequency(freq, ampl, args.test_scenario)

        # Compute the velocity of the wave
        if args.test_scenario == "2CR":
            ang_velocity = 2 * np.pi * first_freq * 1000  # Multiply 1000 to convert to Hz
            velocity = RDC_data.RA * ang_velocity / 1000  # Divide by 1000 for mm/sec -> m/sec
            t = np.arange(RDC_data.m)

            if args.sPOD_dim == "1D":
                shift_1 = (ang_velocity * (RDC_data.dt * t))[::-1]
                shift_2 = (ang_velocity * (RDC_data.dt * t))[::-1]
            else:
                # Plot the 2D polar snapshot video
                PlotFOM2D(RDC_data.annPolMat, RDC_data.Xtilde, RDC_data.Ytilde, t, plot_every=50,
                          plot_at_all=False)

                # Compute the shifts
                shift_1_per_R = (ang_velocity * (RDC_data.dt * t))[::-1]
                shift_2_per_R = (ang_velocity * (RDC_data.dt * t))[::-1]

        elif args.test_scenario == "2CRT":
            ang_velocity_1 = 2 * np.pi * first_freq * 1000  # Multiply 1000 to convert to Hz
            ang_velocity_2 = 2 * np.pi * second_freq * 1000  # Multiply 1000 to convert to Hz
            velocity_1 = RDC_data.RA * ang_velocity_1 / 1000  # Divide by 1000 for mm/sec -> m/sec
            velocity_2 = RDC_data.RA * ang_velocity_2 / 1000  # Divide by 1000 for mm/sec -> m/sec
            t = np.arange(RDC_data.m)

            if args.sPOD_dim == "1D":
                shift_1 = (ang_velocity_1 * (RDC_data.dt * t))[::-1]
                shift_2 = (ang_velocity_2 * (RDC_data.dt * t))[::-1]
            else:
                pass
        elif args.test_scenario == "DS2":
            ang_velocity_1 = 2 * np.pi * first_freq * 1000  # Multiply 1000 to convert to Hz
            ang_velocity_2 = 2 * np.pi * second_freq * 1000  # Multiply 1000 to convert to Hz
            ang_velocity_3 = 2 * np.pi * third_freq * 1000  # Multiply 1000 to convert to Hz
            velocity_1 = RDC_data.RA * ang_velocity_1 / 1000  # Divide by 1000 for mm/sec -> m/sec
            velocity_2 = RDC_data.RA * ang_velocity_2 / 1000  # Divide by 1000 for mm/sec -> m/sec
            velocity_3 = RDC_data.RA * ang_velocity_3 / 1000  # Divide by 1000 for mm/sec -> m/sec
            t = np.arange(RDC_data.m)

            if args.sPOD_dim == "1D":
                shift_1 = (ang_velocity_1 * (RDC_data.dt * t))[::-1]
                shift_2 = (ang_velocity_2 * (RDC_data.dt * t))[::-1]
                shift_3 = (ang_velocity_3 * (RDC_data.dt * t) + np.pi)[::-1]
            else:
                pass

        # Applying sPOD on the 1D snapshot data
        print("#############################################")
        print("sPOD run started....")
        time_window_length = 500
        trim_first_few = 0

        if args.sPOD_dim == "1D":
            shift_left = shift_1 - shift_1[time_window_length - 1]
            shift_right = shift_2 - shift_2[time_window_length - 1]
            if args.test_scenario == "DS2":
                shift_right_right = shift_3 - shift_3[time_window_length - 1]
            else:
                shift_right_right = None
            Q_sPOD = sPOD_1D(RDC_data.lumCenter.T, RDC_data.theta, RDC_data.dt * t, 2 * np.pi,
                             len(RDC_data.theta), -shift_left, shift_right,
                             trim_first_few=trim_first_few,
                             time_window_length=time_window_length,
                             sPOD_type="New",
                             spod_iter=100,
                             shifts_right_right=shift_right_right
                             )

            # Plot the sPOD results for the 1D data
            plot_sPOD_1D_frames(Q_sPOD, np.squeeze(RDC_data.theta), t, trim_first_few, time_window_length)
        else:
            shifts = np.zeros((2, 2, len(t)))
            shifts[0, 0, :] = 0.0  # Radial direction (Frame 1)
            shifts[0, 1, :] = - (shift_1_per_R - shift_1_per_R[time_window_length - 1])   # Angular direction (Frame 1)
            shifts[1, 0, :] = 0.0  # Radial direction (Frame 2)
            shifts[1, 1, :] = shift_2_per_R - shift_2_per_R[time_window_length - 1]   # Angular direction (Frame 2)

            Q_sPOD = sPOD_2D(RDC_data.annPolCropMat, RDC_data.theta, RDC_data.r, t, shifts,
                             trim_first_few, time_window_length, RDC_data.Xtilde, RDC_data.Ytilde,
                             spod_iter=100, shifts_right_right=None)

    elif args.which_variable == "Pressure":
        # Read the MATLAB files
        matlab_files = read_matlab(dir, variable_folder, file_name)

        RDC_data = data_pressure(matlab_files)

        # Plot the FFT pressure
        freq, ampl = plot_fft_pressure(RDC_data.pressureProbeFreqMat.T, RDC_data.pressureProbeAmpMat.T,
                                       RDC_data.ProbePos,
                                       desired_probe_pos=60, freq_cut_off=45)

        # Finding the most occurring Frequency
        first_freq, second_freq, third_freq = find_frequency(freq, ampl, args.test_scenario)

        # Compute the velocity of the wave
        if args.test_scenario == "2CR":
            ang_velocity = 2 * np.pi * first_freq * 1000  # Multiply 1000 to convert to Hz
            velocity = RDC_data.RA * ang_velocity / 1000  # Divide by 1000 for mm/sec -> m/sec
        elif args.test_scenario == "2CRT":
            ang_velocity_1 = 2 * np.pi * first_freq * 1000  # Multiply 1000 to convert to Hz
            ang_velocity_2 = 2 * np.pi * second_freq * 1000  # Multiply 1000 to convert to Hz
            velocity_1 = RDC_data.RA * ang_velocity_1 / 1000  # Divide by 1000 for mm/sec -> m/sec
            velocity_2 = RDC_data.RA * ang_velocity_2 / 1000  # Divide by 1000 for mm/sec -> m/sec
        elif args.test_scenario == "DS2":
            ang_velocity_1 = 2 * np.pi * first_freq * 1000  # Multiply 1000 to convert to Hz
            ang_velocity_2 = 2 * np.pi * second_freq * 1000  # Multiply 1000 to convert to Hz
            ang_velocity_3 = 2 * np.pi * third_freq * 1000  # Multiply 1000 to convert to Hz
            velocity_1 = RDC_data.RA * ang_velocity_1 / 1000  # Divide by 1000 for mm/sec -> m/sec
            velocity_2 = RDC_data.RA * ang_velocity_2 / 1000  # Divide by 1000 for mm/sec -> m/sec
            velocity_3 = RDC_data.RA * ang_velocity_3 / 1000  # Divide by 1000 for mm/sec -> m/sec

    elif args.which_variable == "Custom":

        Q, shift_l, shift_r, dx, dt, X, T = provide_wave_equation_data(c_left=1.0, c_right=1.0,
                                                                       v_coeff=0.0, x0_factor=0.5,
                                                                       plot_at_all=False)

        # Applying sPOD on the 1D snapshot data
        print("#############################################")
        print("sPOD run started....")
        time_window_length = len(T)
        trim_first_few = 0
        shift_left = shift_l - shift_l[time_window_length - 1]
        shift_right = shift_r - shift_r[time_window_length - 1]
        Q_sPOD = sPOD_1D(Q, X, T, X[-1],
                         len(X), -shift_left, shift_right,
                         trim_first_few=trim_first_few,
                         time_window_length=time_window_length,
                         sPOD_type="New",
                         spod_iter=30)

        # Plot the sPOD results for the 1D data
        plot_sPOD_1D_frames(Q_sPOD, X, T, trim_first_few, time_window_length)