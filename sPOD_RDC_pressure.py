import argparse
import gc
import os
import sys
import time

import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter, find_peaks

from matplotlib import pyplot as plt, ticker, animation
from skimage.filters import frangi
from sklearn import linear_model
from sklearn.linear_model import RANSACRegressor, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.utils.extmath import randomized_svd

from Helper import window_cut_pressure_data, extract_matrix_from_jpg, extract_matrix_from_matlab_file
from Plots import plot_raw_pressure, plot_phase_avg_pressure, plot_phase_avg_animation
from pressure_map_gui import butter_bandpass_filter
from pressure_window_segment import estimate_det_freq, pick_phase_locked_peaks, build_grad_aligned_segments
from read_matlab_files import new_read_matlab, new_raw_data_pressure, new_processed_data_pressure

sys.path.append('./sPOD/lib/')

from sPOD_tools import shifted_rPCA, shifted_POD_nl, shifted_POD, give_interpolation_error, build_all_frames
from transforms import transforms


def parse_arguments():
    p = argparse.ArgumentParser(description="Input the variables for running the script.")
    p.add_argument("dir_prefix", type=str, help="Directory prefix for MATLAB files input")
    p.add_argument("type", type=str, choices=["Raw", "Processed"], help="Raw data or the processed and cut data")
    p.add_argument("filename", type=str, help="Input the name of the .mat file")
    p.add_argument("dir_prefix_output", type=str, help="Directory prefix for Output")

    return p.parse_args()


def build_dirs(prefix, filename):
    data_dir = os.path.join(prefix, "data" + "/" + filename)
    plot_dir = os.path.join(prefix, "plots" + "/" + filename)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    return data_dir, plot_dir


if __name__ == "__main__":
    args = parse_arguments()

    dir = args.dir_prefix
    type = args.type
    numerical_data = False
    numerical_data_matlab = True
    experimental_data = False

    if type == "Raw":
        folder = "Rawest_data"
        filename = args.filename + ".mat"
    elif type == "Processed":
        folder = "Cut_phase_avg_data"
        filename = args.filename + "_processed.mat"

    # Create the desired directories
    data_dir, plot_dir = build_dirs(args.dir_prefix_output, args.filename)

    # Read the MATLAB files
    matlab_files = new_read_matlab(dir, folder, filename)
    if type == "Raw":
        RDC_pressure = new_raw_data_pressure(matlab_files)
        # plot_raw_pressure(RDC_pressure.t, RDC_pressure.pcb01, plot_dir)

        num_sensors = 15
        num_axial_sensors = 13
        num_radial_sensors = 2
        arrays = [getattr(RDC_pressure, f"pcb{i:02d}") for i in range(1, num_sensors + 1)]
        array_axial_sensors = ["pcb" + f"{i:02d}" for i in range(1, num_axial_sensors + 1)]
        array_radial_sensors = ["pcb" + f"{i:02d}" for i in range(1, num_radial_sensors + 1)]
        pressure = np.vstack(arrays)

        # Band pass filter
        for k in range(num_sensors):
            pressure[k, :] = butter_bandpass_filter(RDC_pressure.t, pressure[k, :], f1=100.0, f2=100000.0, order=4)

        # Window the data between 27 ms - 32 ms
        window_pressure_axial, window_pressure_radial, window_time = window_cut_pressure_data(t_start=0.27, t_end=0.32,
                                                                                              pressure_sensor_array=pressure,
                                                                                              t=RDC_pressure.t)

        # Estimate the detonation frequency based on the radial sensors
        f_det = estimate_det_freq(window_time, window_pressure_radial.tolist())

        # Phase locking to the detonation frequency (Compute the peaks)
        pk_idx = pick_phase_locked_peaks(
            window_time,
            window_pressure_axial[0],
            f_det,
            last_win_s=0.050,
            height_frac=0.2,
            min_sep_frac=0.6,
        )

        # Chop the axial signals into 2pi intervals and align them at the start of the gradient curve
        seg_bank, X_rev, samples_per_rev = build_grad_aligned_segments(
            window_time,
            {k: window_pressure_axial.tolist()[k] for k in range(num_axial_sensors)},
            ref_key=0,  # 0 stands for "pcb01"
            pk_idx=pk_idx,
            fs=RDC_pressure.fs,
            f_det=f_det,
            waves_per_rev=1,
            revs=2,
            grad_left_frac=0.2,
            require_positive_grad=True,
        )
        n_segments = 0
        for k in seg_bank:
            n_segments = max(n_segments, seg_bank[k].shape[0])
        seg_len = X_rev.shape[0]
        seg_stack = np.full((n_segments, num_axial_sensors, seg_len), np.nan, dtype=float)
        for i in range(num_axial_sensors):
            segs = seg_bank[i]
            seg_stack[:segs.shape[0], i, :] = segs

        # Phase average all the segments for all the sensors
        phase_avg = np.nanmean(seg_stack, axis=0) if seg_stack.size else np.empty((num_axial_sensors, seg_len))

        # -------------------------- Populate the domain with interpolation --------------------------------- #
        # Interpolate along the axial direction
        interp_points_axial = 1000
        phase_avg_interpolated = np.zeros((interp_points_axial, seg_len))
        axial_location_measurement = np.arange(5, 126, 10)
        axial_location = np.linspace(5, 125, interp_points_axial)
        for i in range(seg_len):
            cubic_interpolant = interp1d(axial_location_measurement, phase_avg[:, i], kind='cubic')
            phase_avg_interpolated[:, i] = cubic_interpolant(axial_location)

        # Interpolate along the theta direction
        interp_points_radial = 1000
        phase_avg_interpolated_interpolated = np.zeros((interp_points_axial, interp_points_radial))
        radial_location = np.linspace(0, X_rev[-1], interp_points_radial)
        for i in range(interp_points_axial):
            cubic_interpolant = interp1d(X_rev, phase_avg_interpolated[i, :])
            phase_avg_interpolated_interpolated[i, :] = cubic_interpolant(radial_location)

        # # Plot the interpolated snapshot
        # plot_phase_avg_pressure(radial_location, axial_location, phase_avg_interpolated_interpolated, plot_dir,
        #                         key="full")

        # ------------------------------ Extract a single front for analysis ------------------------------------ #
        idx_radial_interesting = np.where((radial_location >= 1.98 * np.pi) & (radial_location <= 3.98 * np.pi))
        idx_axial_interesting = np.where(axial_location <= 90)
        phase_avg_interpolated_interpolated_interesting = \
            phase_avg_interpolated_interpolated[np.ix_(idx_axial_interesting[0], idx_radial_interesting[0])]
        radial_location_interesting = radial_location[idx_radial_interesting[0]]
        axial_location_interesting = axial_location[idx_axial_interesting[0]]

        # ------------------------------ Compute the profiles for 3 waves ------------------------------------- #
        if numerical_data:
            # Extract the image
            image = extract_matrix_from_jpg()

            # Gaussian blurring and thresholding to get rid of turbulence region
            image = cv2.GaussianBlur(image.astype(np.float32), (31, 31), 0)
            image = np.where(image > 0.20, image, 0.0)

            # Downsample for size
            image = cv2.resize(image, (700, 1820), interpolation=cv2.INTER_AREA)  # 600 in the radial direction and 1560 in the axial direction

            axial_location_interesting = np.linspace(0, 125, image.shape[0])
            radial_location_interesting = np.linspace(-np.pi, np.pi, image.shape[1])
            phase_avg_interpolated_interpolated_interesting = (np.flipud(image)).copy()
            phase_avg_interpolated_interpolated_interesting = np.roll(phase_avg_interpolated_interpolated_interesting, -300, axis=1)

            # Further thresholding
            im = phase_avg_interpolated_interpolated_interesting[1740:, 180:].copy()
            im = np.where(im > 0.9, im, 0.0)
            phase_avg_interpolated_interpolated_interesting[1740:, 180:] = im.copy()

            # import matplotlib
            # matplotlib.use('TkAgg')
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(figsize=(10, 6))
            # theta, AX = np.meshgrid(radial_location_interesting, axial_location_interesting)
            # c = ax.pcolormesh(theta, AX, phase_avg_interpolated_interpolated_interesting, shading='auto', cmap="jet")
            # fig.colorbar(c, ax=ax, label="Pressure")
            # plt.show()

            # Get the peaks of the profiles to get a structure of the traveling fronts
            peaks = []
            for i in range(len(axial_location_interesting)):
                pk, properties = find_peaks(phase_avg_interpolated_interpolated_interesting[i, :],
                                            prominence=0, height=0, distance=200)
                prominences = properties['prominences']
                height = properties['peak_heights']
                top_n_idx = np.argsort(-height)[:2]  # Take only the top two peaks
                # if axial_location_interesting[i] < 35:
                #     top_n_idx = [top_n_idx[0]]
                top_n_most_prominent_peaks = pk[top_n_idx]
                peaks.append(top_n_most_prominent_peaks)

            # Fix the stationary frame and calculate the shifts based on that stationary front
            ref_radial = radial_location_interesting[peaks[0][0]]  # Fixed based on the detonation front
            profile_detonation = ref_radial * np.ones(len(axial_location_interesting))
            profile_oblique = np.zeros_like(profile_detonation)
            profile_oblique_refl = np.zeros_like(profile_detonation)

            for i in range(len(axial_location_interesting)):
                big_peak_pos = radial_location_interesting[peaks[i]]
                if len(peaks[i]) < 2:
                    profile_oblique[i] = big_peak_pos
                    profile_oblique_refl[i] = ref_radial
                else:
                    profile_oblique[i] = big_peak_pos[0]
                    profile_oblique_refl[i] = big_peak_pos[1]

            # Smooth the profiles
            # 1. Oblique shock
            axial_hinge = 35  # The height of the detonation front
            obl_hinge = ref_radial
            axial_shifted_full = axial_location_interesting - axial_hinge
            obl_shifted_full = profile_oblique - obl_hinge
            valid_mask = profile_oblique != ref_radial
            axial_shifted_valid = axial_shifted_full[valid_mask]
            obl_shifted_valid = obl_shifted_full[valid_mask]
            constrained_estimator = linear_model.LinearRegression(fit_intercept=False)
            ransac = linear_model.RANSACRegressor(
                estimator=constrained_estimator,
                min_samples=10
            )
            ransac.fit(axial_shifted_valid[:, np.newaxis], obl_shifted_valid)
            obl_pred_shifted = ransac.predict(axial_shifted_full[:, np.newaxis])
            profile_oblique = obl_pred_shifted + obl_hinge


            # 2. Reflected oblique shock
            axial_hinge = axial_location_interesting[-1]
            obl_hinge = profile_oblique[-1]
            axial_shifted_full = axial_location_interesting - axial_hinge
            obl_shifted_full = profile_oblique_refl - obl_hinge
            valid_mask = profile_oblique_refl != ref_radial
            axial_shifted_valid = axial_shifted_full[valid_mask]
            obl_shifted_valid = obl_shifted_full[valid_mask]
            constrained_estimator = linear_model.LinearRegression(fit_intercept=False)
            ransac = linear_model.RANSACRegressor(
                estimator=constrained_estimator,
                min_samples=10
            )
            ransac.fit(axial_shifted_valid[:, np.newaxis], obl_shifted_valid)
            obl_pred_shifted = ransac.predict(axial_shifted_full[:, np.newaxis])
            profile_oblique_refl = obl_pred_shifted + obl_hinge

            # Now we couple the detonation front and the oblique shock wave into a single frame
            profile_detonation_plus_oblique = np.maximum(profile_detonation, profile_oblique)

            # # Plot the animation of the interesting section and the peaks
            # plot_phase_avg_pressure(radial_location_interesting, axial_location_interesting,
            #                         phase_avg_interpolated_interpolated_interesting, plot_dir,
            #                         key="interesting", plot_smooth=True, pk_idx=peaks,
            #                         detonation_plus_shock=profile_detonation_plus_oblique,
            #                         reflection=profile_oblique_refl)
            # plot_phase_avg_animation(radial_location_interesting, axial_location_interesting,
            #                          phase_avg_interpolated_interpolated_interesting, peaks)
            # ---------------------------------------- Compute the shifts --------------------------------------- #
            shift_detonation_plus_oblique = profile_detonation_plus_oblique - ref_radial
            shift_oblique_refl = profile_oblique_refl - ref_radial

            # --------------------------------------- Apply sPOD on the data ----------------------------------------- #
            Nradial = len(radial_location_interesting)
            Naxial = len(axial_location_interesting)

            dradial = radial_location_interesting[1] - radial_location_interesting[0]
            L_radial = [2.0 * np.pi]
            data_shape = [Nradial, 1, 1, Naxial]
            Q_tmp = np.reshape(phase_avg_interpolated_interpolated_interesting.T, data_shape)

            trafo_1 = transforms(data_shape, L_radial, shifts=-shift_detonation_plus_oblique,
                                 dx=[dradial],
                                 use_scipy_transform=False,
                                 use_cubic_spline=False,
                                 interp_order=5)
            trafo_2 = transforms(data_shape, L_radial, shifts=-shift_oblique_refl,
                                 dx=[dradial],
                                 use_scipy_transform=False,
                                 use_cubic_spline=False,
                                 interp_order=5)

            interp_err = give_interpolation_error(Q_tmp, trafo_2)
            print("Transformation interpolation error =  %4.4e " % interp_err)

            # import matplotlib
            # matplotlib.use('TkAgg')
            # fig, axs = plt.subplots(1, 3, num=3, sharey=True, figsize=(12, 5))
            # bottom, top = 0.15, 0.9
            # left, right = 0.1, 0.85
            # fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.1, wspace=0.2)
            # # Original
            # axs[0].pcolormesh(phase_avg_interpolated_interpolated_interesting, cmap='jet')
            # axs[0].axis('auto')
            # axs[0].set_yticks([], [])
            # axs[0].set_xticks([], [])
            # axs[0].set_title(r"$Q$")
            #
            # axs[1].pcolormesh(np.reshape(trafo_1.reverse(Q_tmp), newshape=[-1, Naxial]).T, cmap='jet')
            # axs[1].axis('auto')
            # axs[1].set_yticks([], [])
            # axs[1].set_xticks([], [])
            # axs[1].set_title(r"$\mathcal{T}^{-1}_1 Q$")
            #
            # axs[2].pcolormesh(np.reshape(trafo_1.apply(trafo_1.reverse(Q_tmp)), newshape=[-1, Naxial]).T, cmap='jet')
            # axs[2].axis('auto')
            # axs[2].set_yticks([], [])
            # axs[2].set_xticks([], [])
            # axs[2].set_title(r"$\mathcal{T}_1 \mathcal{T}^{-1}_1 Q$")
            # plt.show()
            # exit()

            # # ---------------------------------------------------------------------------------------------------- #
            # # ---------------------------------------------------------------------------------------------------- #
            # # ---------------------------------------------------------------------------------------------------- #
            # # FIGURE OUT A WAY TO FILTER OUT THE HIGH ENERGY COMPOENETS OR LIKE A revrese SVT.
            # U, S, VT = randomized_svd(np.reshape(trafo_1.reverse(Q_tmp), newshape=[-1, Naxial]).T,
            #                           n_components=30, random_state=42)
            # print(S)
            # S = np.sign(S) * np.maximum(np.abs(S) - (0.2 / 0.0289), 0)
            # print(S)
            # qmat_POD = (U @ np.diag(S)) @ VT
            # import matplotlib
            # matplotlib.use('TkAgg')
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(figsize=(10, 6))
            # c = ax.pcolormesh(qmat_POD, shading='auto', cmap="jet")
            # fig.colorbar(c, ax=ax, label="Pressure")
            # plt.show()
            # exit()
            # # ---------------------------------------------------------------------------------------------------- #
            # # ---------------------------------------------------------------------------------------------------- #
            # # ---------------------------------------------------------------------------------------------------- #

            trafos = [trafo_1, trafo_2]
            qmat = phase_avg_interpolated_interpolated_interesting.T
            [N, M] = np.shape(qmat)
            daxial = axial_location_interesting[1] - axial_location_interesting[0]

            Q = (qmat - np.min(qmat)) / (np.max(qmat) - np.min(qmat))

            # Constant parameters
            alpha0 = N * M / (4 * np.sum(np.abs(qmat))) * 0.01  # Parameter for data fitting dual
            beta0 = N * M / (4 * np.sum(np.abs(qmat))) * 0.005  # Parameter for nonlinearity equality dual
            lamda0 = [1.0, 1.0]  # Parameter for nuclear norm weighing of the traveling frames
            gamma0 = [1e-5, 1e-5]  # TV regularization of traveling wave time amplitudes
            tau0 = 1e8  # Low rankness in nonlinearity
            eta0 = 1e8  # Sparsity in nonlinearity
            omega0 = 1e8

            ret = shifted_POD_nl(Q, trafos, nmodes_max=np.array([1, 25, 3]), eps=1e-16,
                                 Niter=4, use_rSVD=True,
                                 alpha=alpha0, beta=beta0, lamda=lamda0, gamma=gamma0, tau=tau0, eta=eta0, omega=omega0,
                                 dt=daxial, dtol=1e-5)

            sPOD_frames, Qtilde, Q_nl, E, ranks = ret.frames, ret.data_approx, ret.nonlinear_matrix, ret.noise_matrix, ret.ranks

            Q1 = sPOD_frames[0].build_field()
            Q2 = sPOD_frames[1].build_field()

            T1Q1 = trafo_1.apply(Q1)
            T2Q2 = trafo_2.apply(Q2)

            Qtilde = Qtilde + Q_nl
            total_ranks = int(np.sum(ranks))
            U, S, VT = randomized_svd(qmat, n_components=total_ranks, random_state=42)
            qmat_POD = (U @ np.diag(S)) @ VT
            print(
                f"POD relative error with {total_ranks} modes is {np.linalg.norm(qmat - qmat_POD) / np.linalg.norm(qmat)}")

            import matplotlib
            matplotlib.use('TkAgg')
            cmap = "jet"
            qmin = np.min(Q)
            qmax = np.max(Q)
            fig, axs = plt.subplots(2, 4, num=3, sharey=True, figsize=(16, 16))
            bottom, top = 0.15, 0.9
            left, right = 0.1, 0.85
            fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.3, wspace=0.2)
            # Original
            im = axs[0, 0].pcolormesh(Q.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[0, 0].axis('auto')
            axs[0, 0].set_title(r"$Q$")
            axs[0, 0].set_yticks([], [])
            axs[0, 0].set_xticks([], [])
            # Reconstruction
            axs[1, 0].pcolormesh(Qtilde.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[1, 0].axis('auto')
            axs[1, 0].set_title(r"$\tilde{Q}$")
            axs[1, 0].set_yticks([], [])
            axs[1, 0].set_xticks([], [])

            # 1. 1st Shifted frame
            axs[0, 1].pcolormesh(T1Q1.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[0, 1].axis('auto')
            axs[0, 1].set_title(r"$T^1Q^1$")
            axs[0, 1].set_yticks([], [])
            axs[0, 1].set_xticks([], [])
            # 1. 1st Unshifted frame
            axs[1, 1].pcolormesh(Q1.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[1, 1].axis('auto')
            axs[1, 1].set_title(r"$Q^1$")
            axs[1, 1].set_yticks([], [])
            axs[1, 1].set_xticks([], [])

            # 2. 2nd Shifted frame
            axs[0, 2].pcolormesh(T2Q2.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[0, 2].axis('auto')
            axs[0, 2].set_title(r"$T^2Q^2$")
            axs[0, 2].set_yticks([], [])
            axs[0, 2].set_xticks([], [])
            # 2. 2nd Unshifted frame
            axs[1, 2].pcolormesh(Q2.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[1, 2].axis('auto')
            axs[1, 2].set_title(r"$Q^2$")
            axs[1, 2].set_yticks([], [])
            axs[1, 2].set_xticks([], [])

            # 4. Nonlinear frame
            axs[0, 3].pcolormesh(Q_nl.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[0, 3].axis('auto')
            axs[0, 3].set_title(r"$Q_{nl}$")
            axs[0, 3].set_yticks([], [])
            axs[0, 3].set_xticks([], [])
            # 4. Noise
            axs[1, 3].pcolormesh(E.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[1, 3].axis('auto')
            axs[1, 3].set_title(r"$E$")
            axs[1, 3].set_yticks([], [])
            axs[1, 3].set_xticks([], [])
            plt.show()

        elif numerical_data_matlab:
            # Extract the image
            image = extract_matrix_from_matlab_file()

            phase_avg_interpolated_interpolated_interesting = (np.fliplr(np.flipud(image))).copy()
            phase_avg_interpolated_interpolated_interesting = np.roll(
                phase_avg_interpolated_interpolated_interesting, -500, axis=1)
            phase_avg_interpolated_interpolated_interesting = phase_avg_interpolated_interpolated_interesting[6:, :]
            axial_location_interesting = np.linspace(0, 125, phase_avg_interpolated_interpolated_interesting.shape[0])
            radial_location_interesting = np.linspace(-np.pi, np.pi, phase_avg_interpolated_interpolated_interesting.shape[1])



            # import matplotlib
            # matplotlib.use('TkAgg')
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(figsize=(10, 6))
            # theta, AX = np.meshgrid(radial_location_interesting, axial_location_interesting)
            # c = ax.pcolormesh(theta, AX, phase_avg_interpolated_interpolated_interesting, shading='auto', cmap="jet")
            # fig.colorbar(c, ax=ax, label="Pressure")
            # plt.show()
            #
            # import matplotlib
            # matplotlib.use('TkAgg')
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(figsize=(18, 12), subplot_kw={"projection": "3d"})
            # theta, AX = np.meshgrid(radial_location_interesting, axial_location_interesting)
            # surf = ax.plot_surface(
            #     theta,
            #     AX,
            #     phase_avg_interpolated_interpolated_interesting,
            #     cmap="jet",
            #     linewidth=0,
            #     antialiased=False,
            #     rstride=4,
            #     cstride=4
            # )
            # ax.set_xlabel(r'$\theta$ (radians)', labelpad=10)
            # ax.set_ylabel('Axial Location (mm)', labelpad=10)
            # ax.set_zlabel('Temperature', labelpad=10)
            # ax.set_title('3D Temperature Surface Profile')
            # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Temperature")
            # plt.show()
            #
            # exit()

            peaks = []
            matrix_width = phase_avg_interpolated_interpolated_interesting.shape[1]

            for i in range(len(axial_location_interesting)):
                slice_data = phase_avg_interpolated_interpolated_interesting[i, :]

                # Pre-calculate derivatives
                grad1 = np.diff(slice_data)
                grad1 = np.append(grad1, 0)
                grad2 = np.diff(grad1)
                grad2 = np.append(grad2, 0)

                black_peak_coord = None
                red_peak_coord = None
                orange_peak_coord = None

                # ---------------------------------------------------------------------
                # 1. DETONATION / OBLIQUE SHOCK (Black) -> Second Derivative
                # ---------------------------------------------------------------------
                pk_black, props_black = find_peaks(grad2, prominence=2, height=0, distance=100)
                black_pixel_idx = None
                if len(pk_black) > 0:
                    top_black_idx = np.argsort(-props_black['peak_heights'])[0]
                    black_pixel_idx = pk_black[top_black_idx]
                    black_peak_coord = radial_location_interesting[black_pixel_idx]

                    # ---------------------------------------------------------------------
                    # 2. PARASITIC DEFLAGRATION (Red) -> Raw Slice with Dynamic Masking
                    # ---------------------------------------------------------------------
                    deflagration_slice = slice_data.copy()

                    if black_pixel_idx is not None:
                        current_y = axial_location_interesting[i]

                        # 1. Calculate a dynamic buffer that increases linearly above 30 mm
                        if current_y < 30.0:
                            buffer_pixels = 0
                        else:
                            # Scale linearly from a buffer of ~20 pixels at 30mm
                            # up to ~90 pixels at 120mm.
                            # (You can tweak these numbers based on your matrix grid size)
                            y_fraction = (current_y - 35.0) / (120.0 - 35.0)
                            buffer_pixels = int(8 + (110 * y_fraction))

                        # 2. Apply the dynamic geometric mask
                        shock_footprint_end = min(matrix_width, black_pixel_idx + buffer_pixels)
                        deflagration_slice[:shock_footprint_end] = 0

                    pk_red, props_red = find_peaks(deflagration_slice, prominence=5, height=0, distance=100)
                    if len(pk_red) > 0:
                        top_red_idx = np.argsort(-props_red['peak_heights'])[0]
                        red_pixel_idx = pk_red[top_red_idx]

                        if deflagration_slice[red_pixel_idx] > 0:
                            red_peak_coord = radial_location_interesting[red_pixel_idx]

                # ---------------------------------------------------------------------
                # 3. REFLECTED SHOCK (Orange) -> Your Proven Raw Pressure Logic
                # ---------------------------------------------------------------------
                pk_raw, properties_raw = find_peaks(slice_data, prominence=5, height=0, distance=100)
                if len(pk_raw) > 0:
                    top_idx_raw = np.argsort(-properties_raw['peak_heights'])[:3]
                    strongest_peaks_raw = np.sort(pk_raw[top_idx_raw])

                    if len(strongest_peaks_raw) == 3:
                        orange_peak_coord = radial_location_interesting[strongest_peaks_raw[2]]

                    elif len(strongest_peaks_raw) == 2:
                        black_raw_idx = strongest_peaks_raw[0]
                        second_raw_idx = strongest_peaks_raw[1]
                        pixel_separation = second_raw_idx - black_raw_idx

                        if pixel_separation > (matrix_width * 0.22):
                            orange_peak_coord = radial_location_interesting[second_raw_idx]

                # Append all three structured elements
                peaks.append([black_peak_coord, red_peak_coord, orange_peak_coord])

            # Fix the stationary frame and calculate the shifts based on that stationary front
            ref_radial = peaks[0][0]  # Fixed based on the detonation front
            # Assuming 'peaks' now contains 3 items per slice: [black_coord, red_coord, orange_coord]
            profile_detonation = ref_radial * np.ones(len(axial_location_interesting))
            profile_oblique = np.zeros_like(profile_detonation)
            profile_deflagration = np.zeros_like(profile_detonation)  # NEW
            profile_oblique_refl = np.zeros_like(profile_detonation)

            for i in range(len(axial_location_interesting)):
                slice_peaks = peaks[i]  # [black, red, orange]

                # Safely assign Black
                profile_oblique[i] = slice_peaks[0] if slice_peaks[0] is not None else ref_radial

                # Safely assign Red (Deflagration) - NEW
                profile_deflagration[i] = slice_peaks[1] if slice_peaks[1] is not None else None

                # Safely assign Orange (Reflection)
                profile_oblique_refl[i] = slice_peaks[2] if slice_peaks[2] is not None else None

            # ---------------------------------------------------------------------
            # 1. Oblique Shock (Kept with Hinge constraint at y = 35mm)
            # ---------------------------------------------------------------------
            axial_hinge = 30
            obl_hinge = ref_radial
            axial_shifted_full = axial_location_interesting - axial_hinge
            obl_shifted_full = profile_oblique - obl_hinge
            valid_mask = profile_oblique != ref_radial

            axial_shifted_valid = axial_shifted_full[valid_mask]
            obl_shifted_valid = obl_shifted_full[valid_mask]

            constrained_estimator = linear_model.LinearRegression(fit_intercept=False)
            ransac_obl = linear_model.RANSACRegressor(estimator=constrained_estimator, min_samples=10)
            ransac_obl.fit(axial_shifted_valid[:, np.newaxis], obl_shifted_valid)
            profile_oblique = ransac_obl.predict(axial_shifted_full[:, np.newaxis]) + obl_hinge

            # ---------------------------------------------------------------------
            # 2. Deflagration Front (NEW: Free RANSAC Fit without Hinge Constraints)
            # ---------------------------------------------------------------------
            # 1. Filter out None/NaN values
            valid_def_mask = (profile_deflagration != None) & (~np.isnan(profile_deflagration.astype(float)))

            # 2. ADDITIONAL CONSTRAINT: Ignore the straight detonation section below 35 mm
            # This mirrors your oblique shock logic, making sure these points are completely ignored by RANSAC
            valid_def_mask = valid_def_mask & (axial_location_interesting > 30)

            if np.sum(valid_def_mask) > 10:  # Ensure we have enough sample points to fit
                axial_def_valid = axial_location_interesting[valid_def_mask]
                def_valid = profile_deflagration[valid_def_mask].astype(float)

                # Use standard LinearRegression (fit_intercept=True) because there is no hard-fixed origin point
                free_estimator_def = linear_model.LinearRegression(fit_intercept=True)
                ransac_def = linear_model.RANSACRegressor(estimator=free_estimator_def, min_samples=15,
                                                          residual_threshold=0.09, max_trials=200)

                ransac_def.fit(axial_def_valid[:, np.newaxis], def_valid)

                # Predict over the entire axial range for a smooth continuous line
                profile_deflagration = ransac_def.predict(axial_location_interesting[:, np.newaxis])
            else:
                # Fallback if detection failed on too many rows
                profile_deflagration = np.zeros_like(axial_location_interesting)

            # ---------------------------------------------------------------------
            # 3. Reflected Oblique Shock (MODIFIED: Cleared Hinge Constraints)
            # ---------------------------------------------------------------------
            valid_refl_mask = (profile_oblique_refl != None) & (~np.isnan(profile_oblique_refl.astype(float)))

            if np.sum(valid_refl_mask) > 10:
                axial_refl_valid = axial_location_interesting[valid_refl_mask]
                refl_valid = profile_oblique_refl[valid_refl_mask].astype(float)

                # Use standard LinearRegression with intercept enabled to freely capture slope/position
                free_estimator_refl = linear_model.LinearRegression(fit_intercept=True)
                ransac_refl = linear_model.RANSACRegressor(estimator=free_estimator_refl, min_samples=10)

                ransac_refl.fit(axial_refl_valid[:, np.newaxis], refl_valid)
                profile_oblique_refl = ransac_refl.predict(axial_location_interesting[:, np.newaxis])
            else:
                profile_oblique_refl = np.zeros_like(axial_location_interesting)

            # Combine primary detonation and oblique shock wave into a single frame
            profile_detonation_plus_oblique = np.maximum(profile_detonation, profile_oblique)

            # # Plot the animation of the interesting section and the peaks
            # plot_phase_avg_pressure(radial_location_interesting, axial_location_interesting,
            #                         phase_avg_interpolated_interpolated_interesting, plot_dir,
            #                         key="interesting", plot_smooth=True, pk_idx=peaks,
            #                         detonation=profile_detonation,
            #                         deflagration=profile_deflagration,
            #                         shock=profile_oblique,
            #                         reflection=profile_oblique_refl)

            # ---------------------------------------- Compute the shifts --------------------------------------- #
            shift_detonation = profile_detonation - ref_radial
            shift_deflagration = profile_deflagration - ref_radial
            shift_oblique = profile_oblique - ref_radial
            shift_oblique_refl = profile_oblique_refl - ref_radial

            # --------------------------------------- Apply sPOD on the data ----------------------------------------- #
            Nradial = len(radial_location_interesting)
            Naxial = len(axial_location_interesting)

            dradial = radial_location_interesting[1] - radial_location_interesting[0]
            L_radial = [2.0 * np.pi]
            data_shape = [Nradial, 1, 1, Naxial]
            Q_tmp = np.reshape(phase_avg_interpolated_interpolated_interesting.T, data_shape)

            trafo_1 = transforms(data_shape, L_radial, shifts=-shift_detonation,
                                 dx=[dradial],
                                 use_scipy_transform=False,
                                 use_cubic_spline=False,
                                 interp_order=5)
            trafo_2 = transforms(data_shape, L_radial, shifts=-shift_deflagration,
                                 dx=[dradial],
                                 use_scipy_transform=False,
                                 use_cubic_spline=False,
                                 interp_order=5)
            trafo_3 = transforms(data_shape, L_radial, shifts=-shift_oblique,
                                 dx=[dradial],
                                 use_scipy_transform=False,
                                 use_cubic_spline=False,
                                 interp_order=5)
            trafo_4 = transforms(data_shape, L_radial, shifts=-shift_oblique_refl,
                                 dx=[dradial],
                                 use_scipy_transform=False,
                                 use_cubic_spline=False,
                                 interp_order=5)

            interp_err = give_interpolation_error(Q_tmp, trafo_4)
            print("Transformation interpolation error =  %4.4e " % interp_err)

            # import matplotlib
            # matplotlib.use('TkAgg')
            # fig, axs = plt.subplots(1, 3, num=3, sharey=True, figsize=(12, 5))
            # bottom, top = 0.15, 0.9
            # left, right = 0.1, 0.85
            # fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.1, wspace=0.2)
            # # Original
            # axs[0].pcolormesh(phase_avg_interpolated_interpolated_interesting, cmap='hot')
            # axs[0].axis('auto')
            # axs[0].set_yticks([], [])
            # axs[0].set_xticks([], [])
            # axs[0].set_title(r"$Q$")
            #
            # axs[1].pcolormesh(np.reshape(trafo_4.reverse(Q_tmp), newshape=[-1, Naxial]).T, cmap='hot')
            # axs[1].axis('auto')
            # axs[1].set_yticks([], [])
            # axs[1].set_xticks([], [])
            # axs[1].set_title(r"$\mathcal{T}^{-1}_x Q$")
            #
            # axs[2].pcolormesh(np.reshape(trafo_4.apply(trafo_4.reverse(Q_tmp)), newshape=[-1, Naxial]).T, cmap='hot')
            # axs[2].axis('auto')
            # axs[2].set_yticks([], [])
            # axs[2].set_xticks([], [])
            # axs[2].set_title(r"$\mathcal{T}_x \mathcal{T}^{-1}_x Q$")
            # plt.show()
            # exit()


            trafos = [trafo_1, trafo_2, trafo_3, trafo_4]
            qmat = phase_avg_interpolated_interpolated_interesting.T
            [N, M] = np.shape(qmat)
            daxial = axial_location_interesting[1] - axial_location_interesting[0]

            Q = (qmat - np.min(qmat)) / (np.max(qmat) - np.min(qmat))

            # Constant parameters
            alpha0 = N * M / (4 * np.sum(np.abs(qmat))) * 0.7  # Parameter for data fitting dual
            beta0 = N * M / (4 * np.sum(np.abs(qmat))) * 0.7  # Parameter for nonlinearity equality dual
            lamda0 = [1.1, 1.1, 1.2, 1.1]  # Parameter for nuclear norm weighing of the traveling frames
            gamma0 = [1e-5, 1e-5, 1e-5, 1e-5]  # TV regularization of traveling wave time amplitudes
            tau0 = 1e8  # Low rankness in nonlinearity
            eta0 = 1e8  # Sparsity in nonlinearity
            omega0 = 1e8

            ret = shifted_POD_nl(Q, trafos, nmodes_max=np.array([1, 3, 3, 3, 3]), eps=1e-16,
                                 Niter=58, use_rSVD=True,
                                 alpha=alpha0, beta=beta0, lamda=lamda0, gamma=gamma0, tau=tau0, eta=eta0,
                                 omega=omega0,
                                 dt=daxial, dtol=1e-8)

            sPOD_frames, Qtilde, Q_nl, E, ranks = ret.frames, ret.data_approx, ret.nonlinear_matrix, ret.noise_matrix, ret.ranks

            Q1 = sPOD_frames[0].build_field()
            Q2 = sPOD_frames[1].build_field()
            Q3 = sPOD_frames[2].build_field()
            Q4 = sPOD_frames[3].build_field()

            T1Q1 = trafo_1.apply(Q1)
            T2Q2 = trafo_2.apply(Q2)
            T3Q3 = trafo_3.apply(Q3)
            T4Q4 = trafo_4.apply(Q4)

            Qtilde = Qtilde + Q_nl
            total_ranks = int(np.sum(ranks))
            U, S, VT = randomized_svd(qmat, n_components=total_ranks, random_state=42)
            qmat_POD = (U @ np.diag(S)) @ VT
            print(
                f"POD relative error with {total_ranks} modes is {np.linalg.norm(qmat - qmat_POD) / np.linalg.norm(qmat)}")

            import matplotlib

            matplotlib.use('TkAgg')
            cmap = "hot"
            qmin = np.min(Q)
            qmax = np.max(Q)
            fig, axs = plt.subplots(2, 5, num=3, sharey=True, figsize=(20, 16))
            bottom, top = 0.15, 0.9
            left, right = 0.1, 0.85
            fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.3, wspace=0.2)
            # Original
            im = axs[0, 0].pcolormesh(Q.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[0, 0].axis('auto')
            axs[0, 0].set_title(r"$Q$")
            axs[0, 0].set_yticks([], [])
            axs[0, 0].set_xticks([], [])
            # Reconstruction
            axs[1, 0].pcolormesh(Qtilde.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[1, 0].axis('auto')
            axs[1, 0].set_title(r"$\tilde{Q}$")
            axs[1, 0].set_yticks([], [])
            axs[1, 0].set_xticks([], [])

            axs[0, 1].pcolormesh(T1Q1.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[0, 1].axis('auto')
            axs[0, 1].set_title(r"$T_1Q_1$")
            axs[0, 1].set_yticks([], [])
            axs[0, 1].set_xticks([], [])

            axs[1, 1].pcolormesh(Q1.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[1, 1].axis('auto')
            axs[1, 1].set_title(r"$Q_1$")
            axs[1, 1].set_yticks([], [])
            axs[1, 1].set_xticks([], [])

            # 2. 2nd Shifted frame
            axs[0, 2].pcolormesh(T2Q2.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[0, 2].axis('auto')
            axs[0, 2].set_title(r"$T_2Q_2$")
            axs[0, 2].set_yticks([], [])
            axs[0, 2].set_xticks([], [])
            # 2. 2nd Unshifted frame
            axs[1, 2].pcolormesh(Q2.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[1, 2].axis('auto')
            axs[1, 2].set_title(r"$Q_2$")
            axs[1, 2].set_yticks([], [])
            axs[1, 2].set_xticks([], [])

            axs[0, 3].pcolormesh(T3Q3.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[0, 3].axis('auto')
            axs[0, 3].set_title(r"$T_3Q_3$")
            axs[0, 3].set_yticks([], [])
            axs[0, 3].set_xticks([], [])

            axs[1, 3].pcolormesh(Q3.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[1, 3].axis('auto')
            axs[1, 3].set_title(r"$Q_3$")
            axs[1, 3].set_yticks([], [])
            axs[1, 3].set_xticks([], [])

            axs[0, 4].pcolormesh(T4Q4.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[0, 4].axis('auto')
            axs[0, 4].set_title(r"$T_4Q_4$")
            axs[0, 4].set_yticks([], [])
            axs[0, 4].set_xticks([], [])

            axs[1, 4].pcolormesh(Q4.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[1, 4].axis('auto')
            axs[1, 4].set_title(r"$Q_4$")
            axs[1, 4].set_yticks([], [])
            axs[1, 4].set_xticks([], [])
            plt.show()

        elif experimental_data:
            # Get the peaks of the profiles to get a structure of the traveling fronts
            peaks = []
            for i in range(len(axial_location_interesting)):
                pk, properties = find_peaks(phase_avg_interpolated_interpolated_interesting[i, :],
                                            prominence=0, height=0, distance=np.pi)
                prominences = properties['prominences']
                top_n_idx = np.argsort(-prominences)[:2]  # Take only the top two peaks
                top_n_most_prominent_peaks = pk[top_n_idx]
                peaks.append(top_n_most_prominent_peaks)

            # Fix the stationary frame and calculate the shifts based on that stationary front
            ref_radial = radial_location_interesting[peaks[0][0]]  # Fixed based on the detonation front
            profile_detonation = ref_radial * np.ones(len(axial_location_interesting))
            profile_oblique = np.zeros_like(profile_detonation)
            profile_oblique_refl = np.zeros_like(profile_detonation)
            points_dormant_oblique = len([
                i for i in range(len(peaks))
                if np.any(peaks[i] == peaks[0])
            ])

            for i in range(len(axial_location_interesting)):
                big_peak_pos = radial_location_interesting[peaks[i]]
                if len(peaks[i]) < 2:
                    profile_oblique[i] = big_peak_pos
                    profile_oblique_refl[i] = ref_radial
                else:
                    profile_oblique[i] = big_peak_pos[0]
                    profile_oblique_refl[i] = big_peak_pos[1]

            # Smooth the profiles
            # 1. Oblique shock
            ransac = linear_model.RANSACRegressor()
            ransac.fit(axial_location_interesting[:, np.newaxis], profile_oblique)
            profile_oblique = ransac.predict(axial_location_interesting[:, np.newaxis])

            # 2. Reflected oblique shock
            axial_hinge = axial_location_interesting[-1]
            obl_hinge = profile_oblique[-1]
            axial_shifted_full = axial_location_interesting - axial_hinge
            obl_shifted_full = profile_oblique_refl - obl_hinge
            valid_mask = profile_oblique_refl != ref_radial
            axial_shifted_valid = axial_shifted_full[valid_mask]
            obl_shifted_valid = obl_shifted_full[valid_mask]
            constrained_estimator = linear_model.LinearRegression(fit_intercept=False)
            ransac = linear_model.RANSACRegressor(
                estimator=constrained_estimator,
                min_samples=10
            )
            ransac.fit(axial_shifted_valid[:, np.newaxis], obl_shifted_valid)
            obl_pred_shifted = ransac.predict(axial_shifted_full[:, np.newaxis])
            profile_oblique_refl = obl_pred_shifted + obl_hinge

            # # Plot the animation of the interesting section and the peaks
            # plot_phase_avg_pressure(radial_location_interesting, axial_location_interesting,
            #                         phase_avg_interpolated_interpolated_interesting, plot_dir,
            #                         key="interesting", plot_smooth=True, pk_idx=peaks, detonation=profile_detonation,
            #                         shock=profile_oblique, reflection=profile_oblique_refl)
            # plot_phase_avg_animation(radial_location_interesting, axial_location_interesting,
            #                          phase_avg_interpolated_interpolated_interesting, peaks)

            # ---------------------------------------- Compute the shifts --------------------------------------- #
            shift_detonation = profile_detonation - ref_radial
            shift_oblique = profile_oblique - ref_radial
            shift_oblique_refl = profile_oblique_refl - ref_radial

            # --------------------------------------- Apply sPOD on the data ----------------------------------------- #
            Nradial = len(radial_location_interesting)
            Naxial = len(axial_location_interesting)

            # # --- 2. Find the global boundaries ---
            # # We need to know the min and max radial values across ALL profiles
            # # to scale them into the 0 to Nradial row indices.
            # all_profiles = np.concatenate([profile_oblique, profile_oblique_refl])
            # val_min = np.nanmin(all_profiles)
            # val_max = np.nanmax(all_profiles)
            #
            # # Create the blank Nradial x Naxial matrix
            # wave_matrix = np.zeros((Nradial, Naxial))
            #
            # # --- 3. Helper function to rasterize a 1D line into the 2D matrix ---
            # def draw_front(matrix, profile_array, amplitude=1.0):
            #     # Mask out any NaNs (ghost data or jump artifacts)
            #     valid_mask = ~np.isnan(profile_array)
            #
            #     # The columns map perfectly to the valid axial indices
            #     col_indices = np.arange(Naxial)[valid_mask]
            #     valid_vals = profile_array[valid_mask]
            #
            #     # Scale the real radial values into integer row indices (0 to Nradial-1)
            #     normalized_vals = (valid_vals - val_min) / (val_max - val_min)
            #     row_indices = np.round(normalized_vals * (Nradial - 1)).astype(int)
            #
            #     # Clip just to be safe against rounding errors
            #     row_indices = np.clip(row_indices, 0, Nradial - 1)
            #
            #     # "Draw" the line into the matrix: matrix[row, column]
            #     matrix[row_indices, col_indices] = amplitude
            #
            #
            # # --- 4. Draw all three fronts into the matrix ---
            # draw_front(wave_matrix, profile_oblique, amplitude=1.3)
            # draw_front(wave_matrix, profile_oblique_refl, amplitude=0.8)
            #
            # # --- 5. Make them look like physical traveling fronts ---
            # # Apply a Gaussian filter. Since dimensions are swapped, sigma is (row_smear, col_smear)
            # # We smear more heavily in the radial direction (rows) to make the waves thick
            # phase_avg_interpolated_interpolated_interesting = gaussian_filter(wave_matrix, sigma=(5, 2)).T

            dradial = radial_location_interesting[1] - radial_location_interesting[0]
            L_radial = [2 * np.pi]
            data_shape = [Nradial, 1, 1, Naxial]
            Q_tmp = np.reshape(phase_avg_interpolated_interpolated_interesting.T, data_shape)

            trafo_1 = transforms(data_shape, L_radial, shifts=-shift_oblique,
                                 dx=[dradial],
                                 use_scipy_transform=False,
                                 use_cubic_spline=True,
                                 interp_order=5)
            trafo_2 = transforms(data_shape, L_radial, shifts=-shift_oblique_refl,
                                 dx=[dradial],
                                 use_scipy_transform=False,
                                 use_cubic_spline=True,
                                 interp_order=5)

            interp_err = give_interpolation_error(Q_tmp, trafo_2)
            print("Transformation interpolation error =  %4.4e " % interp_err)

            # import matplotlib
            # matplotlib.use('TkAgg')
            # fig, axs = plt.subplots(1, 3, num=3, sharey=True, figsize=(12, 5))
            # bottom, top = 0.15, 0.9
            # left, right = 0.1, 0.85
            # fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.1, wspace=0.2)
            # # Original
            # axs[0].pcolormesh(phase_avg_interpolated_interpolated_interesting)
            # axs[0].axis('auto')
            # axs[0].set_yticks([], [])
            # axs[0].set_xticks([], [])
            # axs[0].set_title(r"$Q$")
            #
            # axs[1].pcolormesh(np.reshape(trafo_2.reverse(Q_tmp), newshape=[-1, Naxial]).T)
            # axs[1].axis('auto')
            # axs[1].set_yticks([], [])
            # axs[1].set_xticks([], [])
            # axs[1].set_title(r"$\mathcal{T}_{-1} Q$")
            #
            # axs[2].pcolormesh(np.reshape(trafo_2.apply(trafo_2.reverse(Q_tmp)), newshape=[-1, Naxial]).T)
            # axs[2].axis('auto')
            # axs[2].set_yticks([], [])
            # axs[2].set_xticks([], [])
            # axs[2].set_title(r"$\mathcal{T}_1 \mathcal{T}_{-1} Q$")
            # plt.show()
            # exit()

            trafos = [trafo_1, trafo_2]
            qmat = phase_avg_interpolated_interpolated_interesting.T
            [N, M] = np.shape(qmat)
            daxial = axial_location_interesting[1] - axial_location_interesting[0]

            Q = (qmat - np.min(qmat)) / (np.max(qmat) - np.min(qmat))

            # Constant parameters
            alpha0 = N * M / (4 * np.sum(np.abs(qmat))) * 0.05  # Parameter for data fitting dual
            beta0 = N * M / (4 * np.sum(np.abs(qmat))) * 0.5  # Parameter for nonlinearity equality dual
            lamda0 = [1.0, 1.0]  # Parameter for nuclear norm weighing of the traveling frames
            gamma0 = [0, 0]  # TV regularization of traveling wave time amplitudes
            tau0 = 0.5
            eta0 = 0.001
            omega0 = 0.33

            ret = shifted_POD_nl(Q, trafos, nmodes_max=np.array([3, 2, 2]), eps=1e-16,
                                 Niter=99, use_rSVD=True,
                                 alpha=alpha0, beta=beta0, lamda=lamda0, gamma=gamma0, tau=tau0, eta=eta0, omega=omega0,
                                 dt=daxial, dtol=1e-5)

            sPOD_frames, Qtilde, Q_nl, E, ranks = ret.frames, ret.data_approx, ret.nonlinear_matrix, ret.noise_matrix, ret.ranks

            Q1 = sPOD_frames[0].build_field()
            Q2 = sPOD_frames[1].build_field()

            T1Q1 = trafo_1.apply(Q1)
            T2Q2 = trafo_2.apply(Q2)

            Qtilde = Qtilde + Q_nl
            total_ranks = int(np.sum(ranks))
            U, S, VT = randomized_svd(qmat, n_components=total_ranks, random_state=42)
            qmat_POD = (U @ np.diag(S)) @ VT
            print(
                f"POD relative error with {total_ranks} modes is {np.linalg.norm(qmat - qmat_POD) / np.linalg.norm(qmat)}")

            import matplotlib

            matplotlib.use('TkAgg')
            cmap = "jet"
            qmin = np.min(Q)
            qmax = np.max(Q)
            fig, axs = plt.subplots(2, 4, num=3, sharey=True, figsize=(16, 8))
            bottom, top = 0.15, 0.9
            left, right = 0.1, 0.85
            fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.1, wspace=0.1)
            # Original
            f = 14
            im = axs[0, 0].pcolormesh(Q.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[0, 0].axis('auto')
            axs[0, 0].set_title(r"$\mathbf{Q}$", fontsize=f)
            axs[0, 0].set_yticks([], [])
            axs[0, 0].set_xticks([], [])
            # Reconstruction
            axs[1, 0].pcolormesh(Qtilde.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[1, 0].axis('auto')
            axs[1, 0].set_title(r"$\tilde{\mathbf{Q}}$", fontsize=f)
            axs[1, 0].set_yticks([], [])
            axs[1, 0].set_xticks([], [])

            # 1. 1st Shifted frame
            axs[0, 1].pcolormesh(T1Q1.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[0, 1].axis('auto')
            axs[0, 1].set_title(r"$\mathbf{T}_1\mathbf{Q}_1$", fontsize=f)
            axs[0, 1].set_yticks([], [])
            axs[0, 1].set_xticks([], [])
            # 1. 1st Unshifted frame
            axs[1, 1].pcolormesh(Q1.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[1, 1].axis('auto')
            axs[1, 1].set_title(r"$\mathbf{Q}_1$", fontsize=f)
            axs[1, 1].set_yticks([], [])
            axs[1, 1].set_xticks([], [])

            # 2. 2nd Shifted frame
            axs[0, 2].pcolormesh(T2Q2.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[0, 2].axis('auto')
            axs[0, 2].set_title(r"$\mathbf{T}_2\mathbf{Q}_2$", fontsize=f)
            axs[0, 2].set_yticks([], [])
            axs[0, 2].set_xticks([], [])
            # 2. 2nd Unshifted frame
            axs[1, 2].pcolormesh(Q2.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[1, 2].axis('auto')
            axs[1, 2].set_title(r"$\mathbf{Q}_2$", fontsize=f)
            axs[1, 2].set_yticks([], [])
            axs[1, 2].set_xticks([], [])

            # 4. Nonlinear frame
            axs[0, 3].pcolormesh(Q_nl.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[0, 3].axis('auto')
            axs[0, 3].set_title(r"$\mathbf{Q}_{nl}$", fontsize=f)
            axs[0, 3].set_yticks([], [])
            axs[0, 3].set_xticks([], [])
            # 4. Noise
            axs[1, 3].pcolormesh(E.T, cmap=cmap, vmin=qmin, vmax=qmax)
            axs[1, 3].axis('auto')
            axs[1, 3].set_title(r"$\mathbf{E}$", fontsize=f)
            axs[1, 3].set_yticks([], [])
            axs[1, 3].set_xticks([], [])

            fig.savefig("RDC_sPOD_pressure.png", dpi=300, transparent=True)
            plt.show()





    elif type == "Processed":
        pass
