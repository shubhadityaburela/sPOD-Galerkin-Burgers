import matplotlib.pyplot as plt
import numpy as np
import sys

from sklearn.utils.extmath import randomized_svd

from Helper import retain_two_peaks
from Plots import PlotFOM2D, PlotPolar2D

sys.path.append('./sPOD/lib/')

from sPOD_tools import shifted_rPCA, shifted_POD_nl, shifted_POD, give_interpolation_error, build_all_frames
from transforms import transforms
from scipy.signal import savgol_filter


def sPOD_1D(Q, theta, t, L_thet, Ntheta, shifts_left, shifts_right, trim_first_few,
            time_window_length, sPOD_type, spod_iter=100, shifts_right_right=None):
    # Trim the data according to the time_window_length
    Q_trim = Q[:, trim_first_few:time_window_length].copy()
    t_trim = t[trim_first_few:time_window_length].copy()
    Nt = time_window_length - trim_first_few
    shifts_left_trim = shifts_left[trim_first_few:time_window_length].copy()
    shifts_right_trim = shifts_right[trim_first_few:time_window_length].copy()
    if shifts_right_right is not None:
        shifts_right_right_trim = shifts_right_right[trim_first_few:time_window_length].copy()

    theta = np.squeeze(theta)

    # Normalize the data
    Q_trim = (Q_trim - np.min(Q_trim)) / (np.max(Q_trim) - np.min(Q_trim))
    Q_trim = savgol_filter(Q_trim, window_length=100, polyorder=3, axis=0)

    # plt.ion()
    # fig, ax = plt.subplots(1, 1)
    # for i in range(Q_trim.shape[1]):
    #     ax.plot(theta, Q_trim[:, i])
    #     ax.plot(theta, Q_trim_smooth[:, i])
    #     ax.set_ylim(0.0, 1.0)
    #     plt.draw()
    #     plt.pause(0.50)
    #     ax.cla()
    # exit()

    dtheta = theta[1] - theta[0]
    L_theta = [L_thet]
    data_shape = [Ntheta, 1, 1, Nt]
    Q_tmp = np.reshape(Q_trim, data_shape)

    if sPOD_type == "Legacy":
        print("Choosing Legacy sPOD.............")
        trafo_1 = transforms(data_shape, L_theta, shifts=np.zeros_like(shifts_right_trim),
                             trafo_type="identity",
                             dx=[dtheta],
                             use_scipy_transform=False,
                             use_cubic_spline=True,
                             interp_order=5)
        trafo_2 = transforms(data_shape, L_theta, shifts=shifts_left_trim,
                             dx=[dtheta],
                             use_scipy_transform=False,
                             use_cubic_spline=True,
                             interp_order=5)
        trafo_3 = transforms(data_shape, L_theta, shifts=shifts_right_trim,
                             dx=[dtheta],
                             use_scipy_transform=False,
                             use_cubic_spline=True,
                             interp_order=5)

        interp_err = give_interpolation_error(Q_tmp, trafo_2)
        print("Transformation interpolation error =  %4.4e " % interp_err)

        trafos = [trafo_1, trafo_2, trafo_3]
        qmat = np.reshape(Q_trim, [-1, Nt])
        [N, M] = np.shape(qmat)
        mu0 = N * M / (4 * np.sum(np.abs(qmat))) * 0.0015
        lambd0 = 1 / np.sqrt(np.maximum(M, N)) * 1.5
        ret = shifted_rPCA(Q_trim, trafos, nmodes_max=np.array([5, 3, 3]), eps=1e-16, Niter=spod_iter, use_rSVD=True,
                           mu=mu0, lambd=lambd0, dtol=1e-5, total_variation_iterations=100)
        sPOD_frames, Qtilde, rel_err, ranks = ret.frames, ret.data_approx, ret.rel_err_hist, ret.ranks

        Q1 = sPOD_frames[0].build_field()
        Q2 = sPOD_frames[1].build_field()
        Q3 = sPOD_frames[2].build_field()

        T1Q1 = trafo_1.apply(Q1)
        T2Q2 = trafo_2.apply(Q2)
        T3Q3 = trafo_3.apply(Q3)

        total_ranks = int(np.sum(ranks) + 2)
        U, S, VT = randomized_svd(qmat, n_components=total_ranks, random_state=42)
        qmat_POD = (U @ np.diag(S)) @ VT
        print(
            f"POD relative error with {total_ranks} modes is {np.linalg.norm(qmat - qmat_POD) / np.linalg.norm(qmat)}")

    else:
        trafo_1 = transforms(data_shape, L_theta, shifts=shifts_left_trim,
                             dx=[dtheta],
                             use_scipy_transform=False,
                             use_cubic_spline=True,
                             interp_order=5)
        trafo_2 = transforms(data_shape, L_theta, shifts=shifts_right_trim,
                             dx=[dtheta],
                             use_scipy_transform=False,
                             use_cubic_spline=True,
                             interp_order=5)

        if shifts_right_right is not None:
            trafo_3 = transforms(data_shape, L_theta, shifts=shifts_right_right_trim,
                                 dx=[dtheta],
                                 use_scipy_transform=False,
                                 use_cubic_spline=True,
                                 interp_order=5)

        interp_err = give_interpolation_error(Q_tmp, trafo_2)
        print("Transformation interpolation error =  %4.4e " % interp_err)

        # fig, axs = plt.subplots(1, 3, num=3, sharey=True, figsize=(12, 5))
        # bottom, top = 0.15, 0.9
        # left, right = 0.1, 0.85
        # fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.1, wspace=0.2)
        # # Original
        # axs[0].pcolormesh(Q_trim.T)
        # axs[0].axis('auto')
        # axs[0].set_yticks([], [])
        # axs[0].set_xticks([], [])
        # axs[0].set_title(r"$Q$")
        #
        # axs[1].pcolormesh(np.reshape(trafo_2.reverse(Q_tmp), newshape=[-1, Nt]).T)
        # axs[1].axis('auto')
        # axs[1].set_yticks([], [])
        # axs[1].set_xticks([], [])
        # axs[1].set_title(r"$\mathcal{T}^{-1}_2 Q$")
        #
        # axs[2].pcolormesh(np.reshape(trafo_2.apply(trafo_2.reverse(Q_tmp)), newshape=[-1, Nt]).T)
        # axs[2].axis('auto')
        # axs[2].set_yticks([], [])
        # axs[2].set_xticks([], [])
        # axs[2].set_title(r"$\mathcal{T}_2 \mathcal{T}^{-1}_2 Q$")
        # plt.show()
        # exit()

        if shifts_right_right is None:
            # 2CR : mu0 factor = 0.0015, tau0 factor = 0.3, gamma0 = [dt^2, dt^2], nmodes_max = [3, 3]
            # 2CRT : mu0 factor = 0.0015, tau0 factor = 0.3, gamma0 = [10^2 * dt^2, 10^6 * dt^2], nmodes_max = [3, 3]

            trafos = [trafo_1, trafo_2]
            qmat = np.reshape(Q_trim, [-1, Nt])
            [N, M] = np.shape(qmat)
            mu0 = N * M / (4 * np.sum(np.abs(qmat))) * 0.0015
            tau0 = 1 / np.sqrt(np.maximum(M, N)) * 0.3  # Tune this for playing around
            dt = t[1] - t[0]
            gamma0 = [dt ** 2, dt ** 2]  # Tune this for playing around

            ret = shifted_POD_nl(Q_trim, trafos, nmodes_max=np.array([3, 3]), eps=1e-16, Niter=spod_iter, use_rSVD=True,
                                 mu=mu0, tau=tau0, gamma=gamma0, dt=dt, dtol=1e-5)

            sPOD_frames, Qtilde, Q_nl, ranks = ret.frames, ret.data_approx, ret.nonlinear_matrix, ret.ranks

            Q1 = Q_nl.copy()
            Q2 = sPOD_frames[0].build_field()
            Q3 = sPOD_frames[1].build_field()

            T1Q1 = Q1.copy()
            T2Q2 = trafo_1.apply(Q2)
            T3Q3 = trafo_2.apply(Q3)

            Q4 = np.zeros_like(Q3)
            T4Q4 = np.zeros_like(T3Q3)
        else:
            trafos = [trafo_1, trafo_2, trafo_3]
            qmat = np.reshape(Q_trim, [-1, Nt])
            [N, M] = np.shape(qmat)
            mu0 = N * M / (4 * np.sum(np.abs(qmat))) * 0.009
            tau0 = 1 / np.sqrt(np.maximum(M, N)) * 0.30  # Tune this for playing around
            dt = t[1] - t[0]
            gamma0 = [1000 * dt ** 2, 1.0 * dt ** 2, 1.0 * dt ** 2]  # Tune this for playing around

            ret = shifted_POD_nl(Q_trim, trafos, nmodes_max=np.array([3, 1, 1]), eps=1e-16, Niter=spod_iter,
                                 use_rSVD=True,
                                 mu=mu0, tau=tau0, gamma=gamma0, dt=dt, dtol=1e-5)

            sPOD_frames, Qtilde, Q_nl, ranks = ret.frames, ret.data_approx, ret.nonlinear_matrix, ret.ranks

            Q1 = Q_nl.copy()
            Q2 = sPOD_frames[0].build_field()
            Q3 = sPOD_frames[1].build_field()
            Q4 = sPOD_frames[2].build_field()

            T1Q1 = Q1.copy()
            T2Q2 = trafo_1.apply(Q2)
            T3Q3 = trafo_2.apply(Q3)
            T4Q4 = trafo_3.apply(Q4)

        Qtilde = Qtilde + Q_nl

        total_ranks = int(np.sum(ranks) + 2)
        U, S, VT = randomized_svd(qmat, n_components=total_ranks, random_state=42)
        qmat_POD = (U @ np.diag(S)) @ VT
        print(
            f"POD relative error with {total_ranks} modes is {np.linalg.norm(qmat - qmat_POD) / np.linalg.norm(qmat)}")

    # for i in range(ranks[0]):
    #     plt.plot(t_trim, sPOD_frames[0].modal_system['VT'][i], label=f"{i}")
    # plt.legend()
    # plt.show()
    # for i in range(ranks[1]):
    #     plt.plot(t_trim, sPOD_frames[1].modal_system['VT'][i], label=f"{i}")
    # plt.legend()
    # plt.show()

    return [Q_trim, T1Q1, T2Q2, T3Q3, T4Q4, Q1, Q2, Q3, Q4, Qtilde]


def sPOD_2D(Q, theta, R, t, shifts, trim_first_few, time_window_length, Xtilde, Ytilde,
            spod_iter=100, shifts_right_right=None):
    # Trim the data according to the time_window_length
    Q_trim = Q[trim_first_few:time_window_length, :, :].copy()
    t_trim = t[trim_first_few:time_window_length].copy()
    shifts_trim = shifts[..., trim_first_few:time_window_length].copy()
    theta = np.squeeze(theta)
    R = np.squeeze(R)

    Ntheta = len(theta)
    NR = 15  # len(R)
    Nt = time_window_length - trim_first_few

    # Smooth the polar data
    Q_polar = (Q_trim - np.min(Q_trim)) / (np.max(Q_trim) - np.min(Q_trim))
    Q_polar_smooth = np.reshape(savgol_filter(Q_polar, window_length=100, polyorder=3, axis=1),
                                newshape=[NR, Ntheta, 1, Nt])

    dR = R[1] - R[0]
    dtheta = theta[1] - theta[0]
    data_shape = [NR, Ntheta, 1, Nt]
    d_del = np.asarray([dR, dtheta])
    L = np.asarray([R[-1] - R[0], 2 * np.pi])

    # Construct the transformation operator
    trafo_1 = transforms(data_shape, L, shifts=shifts_trim[0],
                         dx=d_del,
                         use_scipy_transform=True)
    trafo_2 = transforms(data_shape, L, shifts=shifts_trim[1],
                         dx=d_del,
                         use_scipy_transform=True)

    # Check the transformation interpolation error
    err = give_interpolation_error(Q_polar_smooth, trafo_1)
    print("Transformation interpolation error =  %4.4e " % err)

    # PlotFOM2D(np.reshape(Q_polar_smooth, newshape=[Nt, Ntheta, NR]), Xtilde, Ytilde,
    #           t_trim, plot_every=1,
    #           plot_at_all=True)
    #
    # PlotFOM2D(np.reshape(trafo_1.reverse(Q_polar_smooth), newshape=[Nt, Ntheta, NR]), Xtilde, Ytilde,
    #           t_trim, plot_every=1,
    #           plot_at_all=True)
    # exit()

    trafos = [trafo_1, trafo_2]
    qmat = np.reshape(Q_polar_smooth, [-1, Nt])
    [N, M] = np.shape(qmat)
    mu0 = N * M / (4 * np.sum(np.abs(qmat))) * 0.001
    tau0 = 1 / np.sqrt(np.maximum(M, N)) * 0.35
    dt = t[1] - t[0]
    gamma0 = [dt ** 2, dt ** 2]

    ret = shifted_POD_nl(qmat, trafos, nmodes_max=np.array([10, 10]), eps=1e-16, Niter=spod_iter, use_rSVD=True,
                         mu=mu0, tau=tau0, gamma=gamma0, dt=dt, dtol=1e-5)

    sPOD_frames, Qtilde, Q_nl, ranks = ret.frames, ret.data_approx, ret.nonlinear_matrix, ret.ranks

    Q1 = np.reshape(Q_nl.copy(), newshape=[Nt, Ntheta, NR])
    Q2 = np.reshape(sPOD_frames[0].build_field(), newshape=[Nt, Ntheta, NR])
    Q3 = np.reshape(sPOD_frames[1].build_field(), newshape=[Nt, Ntheta, NR])

    T1Q1 = Q1.copy()
    T2Q2 = np.reshape(trafo_1.apply(np.reshape(Q2, newshape=data_shape)), newshape=[Nt, Ntheta, NR])
    T3Q3 = np.reshape(trafo_2.apply(np.reshape(Q3, newshape=data_shape)), newshape=[Nt, Ntheta, NR])

    Q4 = np.zeros_like(Q3)
    T4Q4 = np.zeros_like(T3Q3)

    PlotPolar2D(Q_polar_smooth, T1Q1, T2Q2, T3Q3, R, theta, t_trim, plot_every=1, plot_at_all=True)