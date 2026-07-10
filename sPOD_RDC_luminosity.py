import argparse
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

from FFT import fft_luminosity, find_frequency, windowed_fft_luminosity
from Plots import plot_luminosity_1d, plot_fft_luminosity, plot_fft_pressure, plot_sPOD_1D_frames, PlotFOM2D
from read_matlab_files import read_matlab, data_luminosity, data_pressure
from sPOD_calls import sPOD_1D, sPOD_2D
from wave_equation import provide_wave_equation_data


def parse_arguments():
    p = argparse.ArgumentParser(description="Input the variables for running the script.")
    p.add_argument("dir_prefix", type=str, help="Directory prefix for MATLAB files input")
    p.add_argument("which_variable", type=str, choices=["Luminosity", "Pressure", "Custom"],
                   help="Select either the luminosity or the pressure data")
    p.add_argument("test_scenario", type=str, choices=["2CR", "2CRT", "DS2", ""], help="Input test scenario")
    p.add_argument("sPOD_dim", type=str, choices=["1D", "2D"], help="Input the sPOD dimension")
    p.add_argument("--windowing", action="store_true", help="Enable windowed FFT")
    p.add_argument("alpha", type=float, help="Input lagrange parameter for data fitting")
    p.add_argument("beta", type=float, help="Input lagrange parameter for nonlinearity equality")
    p.add_argument("lamda", type=float, nargs=3, help="Enter the nuclear norm factor (frame wise)")
    p.add_argument("gamma", type=float, nargs=3, help="Enter the TV factor (frame wise)")
    p.add_argument("tau", type=float, help="Input the nonlinearity nuclear norm factor")
    p.add_argument("eta", type=float, help="Input the nonlinearity sparsity factor")
    p.add_argument("omega", type=float, help="Input the noise factor")

    p.add_argument("nmodes_max", type=int, nargs=4, help="Enter the number of modes allowed for each frame (first "
                                                         "value for the nonlinear and next k values for traveling "
                                                         "frames)")
    p.add_argument("dir_prefix_output", type=str, help="Directory prefix for Output")

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


def build_dirs(prefix, which_variable, test_scenario, sPOD_dim, alpha, beta, lamda, gamma, tau, eta, omega, nmodes_max):
    var_str = which_variable
    scen_str = test_scenario
    dim_str = sPOD_dim
    reg_str = f"alpha={alpha}_beta={beta}_lamda={lamda}_gamma={gamma}_tau={tau}_eta={eta}_omega={omega}_nmodes={nmodes_max}"
    data_dir = os.path.join(prefix, "data", var_str, scen_str, dim_str, reg_str)
    plot_dir = os.path.join(prefix, "plots", var_str, scen_str, dim_str, reg_str)
    # os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    return data_dir, plot_dir


if __name__ == "__main__":
    args = parse_arguments()

    dir = args.dir_prefix
    variable_folder = select_variable(args.which_variable)
    file_name = select_filename(args.test_scenario)
    windowing = args.windowing

    # Instantiate the arguments
    alpha = args.alpha
    beta = args.beta
    lamda = args.lamda
    gamma = args.gamma
    tau = args.tau
    eta = args.eta
    omega = args.omega

    nmodes_max = args.nmodes_max

    print(
        f"alpha={alpha}, beta={beta}, lamda={lamda}, gamma={gamma}, tau={tau}, eta={eta}, omega={omega}, nmodes_max={nmodes_max}")

    # Create the desired directories
    data_dir, plot_dir = build_dirs(args.dir_prefix_output, args.which_variable, args.test_scenario,
                                    args.sPOD_dim, alpha, beta, lamda, gamma, tau, eta, omega, nmodes_max)

    if args.which_variable == "Luminosity":
        # Read the MATLAB files
        matlab_files = read_matlab(dir, variable_folder, file_name)

        RDC_data = data_luminosity(matlab_files)

        # Plot the luminosity snapshot data
        plot_luminosity_1d(RDC_data.lumCenter.T, time_window_length=250, plot_at_all=False)

        # Perform FFT on the data
        if windowing is False:
            if args.test_scenario == "2CR" or args.test_scenario == "2CRT":
                luminosity_fft = fft_luminosity(RDC_data.lumCenter.T, theta=RDC_data.theta, theta_pos=60)
            else:
                luminosity_fft = fft_luminosity(RDC_data.lumCenter.T, theta=RDC_data.theta, theta_pos=300)

            # Plot the FFT luminosity
            freq, ampl = plot_fft_luminosity(luminosity_fft, RDC_data.m, RDC_data.dt, plot_at_all=True)

            # Finding the most occurring Frequency
            first_freq, second_freq, third_freq = find_frequency(freq, ampl, args.test_scenario)
        else:
            if args.test_scenario == "2CR" or args.test_scenario == "2CRT":
                luminosity_freq, t_spec = windowed_fft_luminosity(RDC_data.lumCenter.T, theta=RDC_data.theta,
                                                                  theta_pos=60,
                                                                  window=512, dt=RDC_data.dt,
                                                                  scenario=args.test_scenario, plot_at_all=False)
                # print(luminosity_freq)
                # exit()
        # Compute the velocity of the wave
        if args.test_scenario == "2CR":
            t = np.arange(RDC_data.m)
            if windowing is False:
                ang_velocity = 2 * np.pi * first_freq * 1000  # Multiply 1000 to convert to Hz
                velocity = RDC_data.RA * ang_velocity / 1000  # Divide by 1000 for mm/sec -> m/sec
                shift_1 = (ang_velocity * (RDC_data.dt * t))[::-1]
                shift_2 = (ang_velocity * (RDC_data.dt * t))[::-1]
            else:
                freq_interp = np.interp(t, t_spec, luminosity_freq)
                ang_velocity = 2 * np.pi * freq_interp * 1000
                shift_1 = np.cumsum(ang_velocity * RDC_data.dt)[::-1]
                shift_2 = np.cumsum(ang_velocity * RDC_data.dt)[::-1]


        elif args.test_scenario == "2CRT":
            ang_velocity_1 = 2 * np.pi * first_freq * 1000  # Multiply 1000 to convert to Hz
            ang_velocity_2 = 2 * np.pi * second_freq * 1000  # Multiply 1000 to convert to Hz
            velocity_1 = RDC_data.RA * ang_velocity_1 / 1000  # Divide by 1000 for mm/sec -> m/sec
            velocity_2 = RDC_data.RA * ang_velocity_2 / 1000  # Divide by 1000 for mm/sec -> m/sec
            t = np.arange(RDC_data.m)

            shift_1 = (ang_velocity_1 * (RDC_data.dt * t))[::-1]
            shift_2 = (ang_velocity_2 * (RDC_data.dt * t))[::-1]

        elif args.test_scenario == "DS2":
            ang_velocity_1 = 2 * np.pi * first_freq * 1000  # Multiply 1000 to convert to Hz
            ang_velocity_2 = 2 * np.pi * second_freq * 1000  # Multiply 1000 to convert to Hz
            ang_velocity_3 = 2 * np.pi * third_freq * 1000  # Multiply 1000 to convert to Hz
            velocity_1 = RDC_data.RA * ang_velocity_1 / 1000  # Divide by 1000 for mm/sec -> m/sec
            velocity_2 = RDC_data.RA * ang_velocity_2 / 1000  # Divide by 1000 for mm/sec -> m/sec
            velocity_3 = RDC_data.RA * ang_velocity_3 / 1000  # Divide by 1000 for mm/sec -> m/sec
            t = np.arange(RDC_data.m)

            shift_1 = (ang_velocity_1 * (RDC_data.dt * t))[::-1]
            shift_2 = (ang_velocity_2 * (RDC_data.dt * t))[::-1]
            shift_3 = (ang_velocity_3 * (RDC_data.dt * t) + np.pi)[::-1]

        # Applying sPOD on the 1D snapshot data
        print("#############################################")
        print("sPOD run started....")
        time_window_length = 250
        trim_first_few = 0

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
                         alpha=alpha, beta=beta, lamda=lamda, gamma=gamma, tau=tau, eta=eta, omega=omega,
                         nmodes=nmodes_max,
                         spod_iter=139,
                         shifts_right_right=shift_right_right
                         )

        # Plot the sPOD results for the 1D data
        plot_sPOD_1D_frames(Q_sPOD, np.squeeze(RDC_data.theta), t, trim_first_few, time_window_length, plot_dir)

    elif args.which_variable == "Pressure":
        pass
