import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.fft import fftfreq

import matplotlib
matplotlib.use('TkAgg')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"]})

VERY_SMALL_SIZE = 12
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=VERY_SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=VERY_SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

cmap = 'hot'


def plot_luminosity_1d(luminosity, time_window_length, plot_at_all=False):
    if plot_at_all:
        fig = plt.figure(figsize=(5, 5))
        ax1 = fig.add_subplot(111)
        im1 = ax1.pcolormesh(luminosity[:, :time_window_length].T, cmap='YlOrRd')
        ax1.axis('off')
        ax1.axis('auto')
        ax1.set_title(r"Luminosity")
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='10%', pad=0.08)
        fig.colorbar(im1, cax=cax, orientation='vertical')
        fig.supylabel(r"$t$")
        fig.supxlabel(r"$\theta$")
        plt.show()


def plot_fft_luminosity(luminosity_fft, Nt, dt, plot_at_all=False):
    # Frequency bins
    # Convert to KHz
    xf = fftfreq(Nt, dt)[:Nt // 2] / 1000
    amp = 2.0 / Nt * np.abs(luminosity_fft[0:Nt // 2])
    amp = amp / np.max(amp)

    tick_spacing_khz = 1

    if plot_at_all:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xf, amp)
        ax.set_xlim(left=0, right=xf.max())
        ax.margins(x=0)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_khz))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
        ax.set_xlabel(r'$f$ (kHz)')
        ax.set_ylabel('Amplitude')
        ax.set_title('FFT luminosity')
        ax.grid(True)
        plt.show()

    return xf, amp


def plot_fft_pressure(pressure_freq, pressure_amp, probe_pos, desired_probe_pos, freq_cut_off):
    probe_pos_idx = np.argmin(np.abs(probe_pos - desired_probe_pos))

    # Convert to KHz
    pressure_freq = pressure_freq / 1000

    # Subset of full based on cutoff
    subset_indices = np.where(pressure_freq[probe_pos_idx, :] < freq_cut_off)

    freq = pressure_freq[probe_pos_idx, subset_indices[0]]
    ampl = pressure_amp[probe_pos_idx, subset_indices[0]]

    tick_spacing_khz = 1
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(freq, ampl)
    ax.set_xlim(left=0, right=freq.max())
    ax.margins(x=0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_khz))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    ax.set_xlabel(r'$f$ (kHz)')
    ax.set_ylabel('Amplitude')
    ax.set_title('FFT pressure')
    ax.grid(True)
    plt.show()

    return freq, ampl


def plot_sPOD_1D_frames(Q_sPOD, theta, t, trim_first_few, time_window_length, immpath):
    Q = Q_sPOD[0]
    T1Q1 = Q_sPOD[1]
    T2Q2 = Q_sPOD[2]
    T3Q3 = Q_sPOD[3]
    T4Q4 = Q_sPOD[4]
    E = Q_sPOD[5]
    Q2 = Q_sPOD[6]
    Q3 = Q_sPOD[7]
    Q4 = Q_sPOD[8]
    Qtilde = Q_sPOD[9]

    [theta_grid, t_grid] = np.meshgrid(theta, t[trim_first_few:time_window_length])
    theta_grid = theta_grid.T
    t_grid = t_grid.T

    qmin = np.min(Q)
    qmax = np.max(Q)
    fig, axs = plt.subplots(2, 6, num=3, sharey=True, figsize=(24, 16))
    bottom, top = 0.15, 0.9
    left, right = 0.1, 0.85
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.3, wspace=0.2)
    # Original
    im = axs[0, 0].pcolormesh(theta_grid, t_grid, Q, cmap=cmap, vmin=qmin, vmax=qmax)
    axs[0, 0].axis('auto')
    axs[0, 0].set_title(r"$Q$")
    axs[0, 0].set_yticks([], [])
    axs[0, 0].set_xticks([], [])
    # Reconstruction
    axs[1, 0].pcolormesh(theta_grid, t_grid, Qtilde, cmap=cmap, vmin=qmin, vmax=qmax)
    axs[1, 0].axis('auto')
    axs[1, 0].set_title(r"$\tilde{Q}$")
    axs[1, 0].set_yticks([], [])
    axs[1, 0].set_xticks([], [])

    # 1. 1st Shifted frame
    axs[0, 1].pcolormesh(theta_grid, t_grid, T1Q1, cmap=cmap, vmin=qmin, vmax=qmax)
    axs[0, 1].axis('auto')
    axs[0, 1].set_title(r"$T^1Q^1$")
    axs[0, 1].set_yticks([], [])
    axs[0, 1].set_xticks([], [])
    # 1. 1st Unshifted frame
    axs[1, 1].pcolormesh(theta_grid, t_grid, T1Q1, cmap=cmap, vmin=qmin, vmax=qmax)
    axs[1, 1].axis('auto')
    axs[1, 1].set_title(r"$Q^1$")
    axs[1, 1].set_yticks([], [])
    axs[1, 1].set_xticks([], [])

    # 2. 2nd Shifted frame
    axs[0, 2].pcolormesh(theta_grid, t_grid, T2Q2, cmap=cmap, vmin=qmin, vmax=qmax)
    axs[0, 2].axis('auto')
    axs[0, 2].set_title(r"$T^2Q^2$")
    axs[0, 2].set_yticks([], [])
    axs[0, 2].set_xticks([], [])
    # 2. 2nd Unshifted frame
    axs[1, 2].pcolormesh(theta_grid, t_grid, Q2, cmap=cmap, vmin=qmin, vmax=qmax)
    axs[1, 2].axis('auto')
    axs[1, 2].set_title(r"$Q^2$")
    axs[1, 2].set_yticks([], [])
    axs[1, 2].set_xticks([], [])

    # 3. 3rd Shifted frame
    axs[0, 3].pcolormesh(theta_grid, t_grid, T3Q3, cmap=cmap, vmin=qmin, vmax=qmax)
    axs[0, 3].axis('auto')
    axs[0, 3].set_title(r"$T^3Q^3$")
    axs[0, 3].set_yticks([], [])
    axs[0, 3].set_xticks([], [])
    # 3. 3rd Unshifted frame
    axs[1, 3].pcolormesh(theta_grid, t_grid, Q3, cmap=cmap, vmin=qmin, vmax=qmax)
    axs[1, 3].axis('auto')
    axs[1, 3].set_title(r"$Q^3$")
    axs[1, 3].set_yticks([], [])
    axs[1, 3].set_xticks([], [])

    # 4. 4th Shifted frame
    axs[0, 4].pcolormesh(theta_grid, t_grid, T4Q4, cmap=cmap, vmin=qmin, vmax=qmax)
    axs[0, 4].axis('auto')
    axs[0, 4].set_title(r"$T^4Q^4$")
    axs[0, 4].set_yticks([], [])
    axs[0, 4].set_xticks([], [])
    # 4. 4th Unshifted frame
    axs[1, 4].pcolormesh(theta_grid, t_grid, Q4, cmap=cmap, vmin=qmin, vmax=qmax)
    axs[1, 4].axis('auto')
    axs[1, 4].set_title(r"$Q^4$")
    axs[1, 4].set_yticks([], [])
    axs[1, 4].set_xticks([], [])

    # 5. Noise frame
    axs[0, 5].pcolormesh(theta_grid, t_grid, E, cmap=cmap, vmin=qmin, vmax=qmax)
    axs[0, 5].axis('auto')
    axs[0, 5].set_title(r"$E$")
    axs[0, 5].set_yticks([], [])
    axs[0, 5].set_xticks([], [])
    # 5. Noise frame
    axs[1, 5].pcolormesh(theta_grid, t_grid, E, cmap=cmap, vmin=qmin, vmax=qmax)
    axs[1, 5].axis('auto')
    axs[1, 5].set_title(r"$E$")
    axs[1, 5].set_yticks([], [])
    axs[1, 5].set_xticks([], [])

    cbar_ax = fig.add_axes([0.90, bottom, 0.01, top - bottom])
    fig.colorbar(im, cax=cbar_ax)

    fig.supylabel(r"time $t$")
    fig.supxlabel(r"space $x$")

    out_file = os.path.join(immpath, f"sPOD_decomposition.png")
    fig.savefig(out_file, dpi=300, transparent=True)

    # qmin = np.min(Q)
    # qmax = np.max(Q)
    #
    # VERY_SMALL_SIZE = 8
    # SMALL_SIZE = 10
    # MEDIUM_SIZE = 10
    # BIGGER_SIZE = 10
    #
    # plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    # plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=VERY_SMALL_SIZE)  # fontsize of the tick labels
    # plt.rc('ytick', labelsize=VERY_SMALL_SIZE)  # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    #
    # fig, axs = plt.subplots(1, 1, num=1, sharey=True, figsize=(4, 5))
    # bottom, top = 0.15, 0.9
    # left, right = 0.1, 0.85
    # fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.3, wspace=0.2)
    # im = axs.pcolormesh(theta_grid, t_grid, T1Q1, cmap=cmap, vmin=qmin, vmax=qmax)
    # axs.axis('auto')
    # axs.set_yticks([], [])
    # axs.set_xticks([], [])
    # out_file = os.path.join(immpath, f"sPOD_decomposition_1.png")
    # fig.savefig(out_file, dpi=300, transparent=True)
    #
    # fig, axs = plt.subplots(1, 1, num=2, sharey=True, figsize=(4, 5))
    # bottom, top = 0.15, 0.9
    # left, right = 0.1, 0.85
    # fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.3, wspace=0.2)
    # im = axs.pcolormesh(theta_grid, t_grid, T2Q2, cmap=cmap, vmin=qmin, vmax=qmax)
    # axs.axis('auto')
    # axs.set_yticks([], [])
    # axs.set_xticks([], [])
    # out_file = os.path.join(immpath, f"sPOD_decomposition_2.png")
    # fig.savefig(out_file, dpi=300, transparent=True)
    #
    # fig, axs = plt.subplots(1, 1, num=3, sharey=True, figsize=(4, 5))
    # bottom, top = 0.15, 0.9
    # left, right = 0.1, 0.85
    # fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.3, wspace=0.2)
    # im = axs.pcolormesh(theta_grid, t_grid, T3Q3, cmap=cmap, vmin=qmin, vmax=qmax)
    # axs.axis('auto')
    # axs.set_yticks([], [])
    # axs.set_xticks([], [])
    # out_file = os.path.join(immpath, f"sPOD_decomposition_3.png")
    # fig.savefig(out_file, dpi=300, transparent=True)

    # s_1 = np.linalg.svd(Q, compute_uv=False)
    # s_2 = np.linalg.svd(Q2, compute_uv=False)
    # s_3 = np.linalg.svd(Q3, compute_uv=False)
    # s_1 = s_1[:100]
    # s_2 = s_2[:100]
    # s_3 = s_3[:100]
    # s_norm_1 = s_1 / s_1[0]  # normalize by largest singular value
    # s_norm_2 = s_2 / s_2[0]  # normalize by largest singular value
    # s_norm_3 = s_3 / s_3[0]  # normalize by largest singular value
    # idx = np.arange(1, len(s_norm_1) + 1)
    #
    # fig, axs = plt.subplots(1, 1, num=4, sharey=True, figsize=(4, 5))
    # bottom, top = 0.15, 0.9
    # left, right = 0.1, 0.85
    # fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.3, wspace=0.2)
    # axs.semilogy(idx, s_norm_1, marker="+", linestyle='None', markersize=3, label=r"$Q$")
    # axs.semilogy(idx, s_norm_2, marker="o", linestyle='None', markersize=3, label=r"$T^1Q^1$")
    # axs.semilogy(idx, s_norm_3, marker="x", linestyle='None', markersize=3, label=r"$T^2Q^2$")
    # # axs.set_ylabel(r"$\sigma_k/\sigma_0$")
    # # axs.set_title("Singular value decay")
    # axs.legend(fontsize=10)
    # axs.grid(True)
    # out_file = os.path.join(immpath, "MOR4.png")
    # fig.savefig(out_file, dpi=300, transparent=True)



def PlotFOM2D(SnapMat, R, theta, t, plot_every=9, plot_at_all=False):
    Nt = int(len(t))
    if plot_at_all:
        plt.ion()  # interactive mode on (so plot updates)
        fig, ax = plt.subplots(figsize=(10, 3))

        # initial frame (use first snapshot)
        n0 = 0
        vmin = np.min(SnapMat)
        vmax = np.max(SnapMat)
        im = ax.imshow(SnapMat[n0, :, :].T, origin='lower', vmin=vmin, vmax=vmax,
                       extent=[theta.min(), theta.max(), R.min(), R.max()], aspect="auto", cmap='YlOrRd')

        ax.set_title(r"$Luminosity$")
        ax.set_yticks([])
        ax.set_xticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='10%', pad=0.08)
        cb = fig.colorbar(im, cax=cax, orientation='vertical')

        fig.supylabel(r"$R$")
        fig.supxlabel(r"$\theta$")

        plt.show(block=False)  # show once (non-blocking)

        for n in range(Nt):
            if n % plot_every == 0:
                frame = SnapMat[n, :, :].T
                # update data and color limits
                im.set_data(frame)
                im.set_clim(vmin, vmax)
                # redraw
                fig.canvas.draw_idle()
                plt.pause(0.01)  # small pause to allow GUI event loop to update

        plt.ioff()  # optional: turn interactive mode off
        plt.show()  # show final frame (blocks)


def PlotPolar2D(Q, T1Q1, T2Q2, T3Q3, R, theta, t, plot_every=9, plot_at_all=False):
    Nt = int(len(t))
    cmap = "viridis"
    if plot_at_all:
        plt.ion()  # interactive mode on (so plot updates)
        # compute global vmin/vmax for consistent color scaling (optional)
        vmin = np.min(Q)
        vmax = np.max(Q)

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)

        # create the three initial images (transpose if you want the same orientation as before)
        n0 = 0
        im1 = axes[0].imshow(T1Q1[n0, :, :].T, origin='lower',
                             vmin=vmin, vmax=vmax,
                             extent=[theta.min(), theta.max(), R.min(), R.max()],
                             aspect='auto', cmap=cmap)
        axes[0].set_title(r"$T_1 Q_1$")
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        im2 = axes[1].imshow(T2Q2[n0, :, :].T, origin='lower',
                             vmin=vmin, vmax=vmax,
                             extent=[theta.min(), theta.max(), R.min(), R.max()],
                             aspect='auto', cmap=cmap)
        axes[1].set_title(r"$T_2 Q_2$")
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        im3 = axes[2].imshow(T3Q3[n0, :, :].T, origin='lower',
                             vmin=vmin, vmax=vmax,
                             extent=[theta.min(), theta.max(), R.min(), R.max()],
                             aspect='auto', cmap=cmap)
        axes[2].set_title(r"$T_3 Q_3$")
        axes[2].set_xticks([])
        axes[2].set_yticks([])

        # single shared colorbar on the right for all three axes
        cbar = fig.colorbar(im3, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.04, pad=0.02)
        cbar.set_label("value")

        fig.supylabel(r"$R$")
        fig.supxlabel(r"$\theta$")

        plt.show(block=False)  # show once, non-blocking

        # update loop: update all three images per time-step
        for n in range(Nt):
            if n % plot_every == 0:
                frame1 = T1Q1[n, :, :].T
                frame2 = T2Q2[n, :, :].T
                frame3 = T3Q3[n, :, :].T

                im1.set_data(frame1)
                im2.set_data(frame2)
                im3.set_data(frame3)

                # if using global vmin/vmax above, you only need to set_clim once (or omit)
                im1.set_clim(vmin, vmax)
                im2.set_clim(vmin, vmax)
                im3.set_clim(vmin, vmax)

                # trigger redraw
                fig.canvas.draw_idle()
                plt.pause(0.1)  # tune pause for desired playback speed

        plt.ioff()
        plt.show()  # final blocking show (optional)
