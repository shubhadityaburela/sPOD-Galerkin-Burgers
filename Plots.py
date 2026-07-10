import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.fft import fftfreq
from sklearn import linear_model
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"]})

VERY_SMALL_SIZE = 4
SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=VERY_SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=VERY_SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

cmap = 'hot'   # 'YlOrRd'


def save_fig(filepath, figure=None, **kwargs):
    import tikzplotlib
    import os
    import matplotlib.pyplot as plt

    ## split extension
    fpath = os.path.splitext(filepath)[0]
    ## get figure handle
    if figure is None:
        figure = plt.gcf()
    figure.savefig(fpath + ".png", dpi=200, transparent=True)
    tikzplotlib.save(
        figure=figure,
        filepath=fpath + ".tex",
        axis_height='\\figureheight',
        axis_width='\\figurewidth',
        override_externals=True,
        **kwargs
    )



def plot_luminosity_1d(luminosity, time_window_length, plot_at_all=False):
    if plot_at_all:
        fig = plt.figure(figsize=(5, 5))
        ax1 = fig.add_subplot(111)
        im1 = ax1.pcolormesh(luminosity[:, :time_window_length].T, cmap='hot')
        ax1.axis('off')
        ax1.axis('auto')
        ax1.set_title(r"Luminosity")
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='10%', pad=0.08)
        fig.colorbar(im1, cax=cax, orientation='vertical')
        fig.supylabel(r"$t$")
        fig.supxlabel(r"$\theta$")
        plt.show()

        # save_fig('RDC_polar_snapshot', fig)


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


    # fig = plt.figure(figsize=(20, 10))
    # ax1 = fig.add_subplot(2, 4, 1)
    # im1 = ax1.pcolormesh(theta_grid, t_grid, Qtilde, cmap=cmap, vmin=qmin, vmax=qmax)
    # ax1.axis('off')
    # ax1.axis('auto')
    # ax1.set_title(r"$\tilde{Q}$")
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes('right', size='10%', pad=0.08)
    # fig.colorbar(im1, cax=cax, orientation='vertical')
    #
    # ax2 = fig.add_subplot(2, 4, 2)
    # im2 = ax2.pcolormesh(theta_grid, t_grid, T2Q2, cmap=cmap, vmin=qmin, vmax=qmax)
    # ax2.axis('off')
    # ax2.axis('auto')
    # ax2.set_title(r"$T^1 Q^1$")
    # divider = make_axes_locatable(ax2)
    # cax = divider.append_axes('right', size='10%', pad=0.08)
    # fig.colorbar(im2, cax=cax, orientation='vertical')
    #
    # ax3 = fig.add_subplot(2, 4, 3)
    # im3 = ax3.pcolormesh(theta_grid, t_grid, T3Q3, cmap=cmap, vmin=qmin, vmax=qmax)
    # ax3.axis('off')
    # ax3.axis('auto')
    # ax3.set_title(r"$T^2 Q^2$")
    # divider = make_axes_locatable(ax3)
    # cax = divider.append_axes('right', size='10%', pad=0.08)
    # fig.colorbar(im3, cax=cax, orientation='vertical')
    #
    # ax4 = fig.add_subplot(2, 4, 4)
    # im4 = ax4.pcolormesh(theta_grid, t_grid, T1Q1, cmap=cmap, vmin=qmin, vmax=qmax)
    # ax4.axis('off')
    # ax4.axis('auto')
    # ax4.set_title(r"$T^3 Q^3$")
    # divider = make_axes_locatable(ax4)
    # cax = divider.append_axes('right', size='10%', pad=0.08)
    # fig.colorbar(im4, cax=cax, orientation='vertical')
    #
    # ##############################################################################################################
    # ax5 = fig.add_subplot(2, 4, 5)
    # im5 = ax5.pcolormesh(theta_grid, t_grid, T2Q2, cmap=cmap, vmin=qmin, vmax=qmax)
    # ax5.axis('off')
    # ax5.axis('auto')
    # ax5.set_title(r"$T^1 Q^1$")
    # divider = make_axes_locatable(ax5)
    # cax = divider.append_axes('right', size='10%', pad=0.08)
    # fig.colorbar(im5, cax=cax, orientation='vertical')
    #
    # ax6 = fig.add_subplot(2, 4, 6)
    # im6 = ax6.pcolormesh(theta_grid, t_grid, Q2, cmap=cmap, vmin=qmin, vmax=qmax)
    # ax6.axis('off')
    # ax6.axis('auto')
    # ax6.set_title(r"$Q^1$")
    # divider = make_axes_locatable(ax6)
    # cax = divider.append_axes('right', size='10%', pad=0.08)
    # fig.colorbar(im6, cax=cax, orientation='vertical')
    #
    # ax7 = fig.add_subplot(2, 4, 7)
    # im7 = ax7.pcolormesh(theta_grid, t_grid, T3Q3, cmap=cmap, vmin=qmin, vmax=qmax)
    # ax7.axis('off')
    # ax7.axis('auto')
    # ax7.set_title(r"$T^2 Q^2$")
    # divider = make_axes_locatable(ax7)
    # cax = divider.append_axes('right', size='10%', pad=0.08)
    # fig.colorbar(im7, cax=cax, orientation='vertical')
    #
    # ax8 = fig.add_subplot(2, 4, 8)
    # im8 = ax8.pcolormesh(theta_grid, t_grid, Q3, cmap=cmap, vmin=qmin, vmax=qmax)
    # ax8.axis('off')
    # ax8.axis('auto')
    # ax8.set_title(r"$Q^2$")
    # divider = make_axes_locatable(ax8)
    # cax = divider.append_axes('right', size='10%', pad=0.08)
    # fig.colorbar(im8, cax=cax, orientation='vertical')
    #
    # fig.supylabel(r"$t$")
    # fig.supxlabel(r"$\theta$")
    #
    # save_fig('RDC_sPOD_frames', fig)


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



def plot_raw_pressure(t, pcbxx, plot_dir):

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(t, pcbxx)
    ax.set_xlim(left=0, right=t.max()+0.01)
    ax.margins(x=0)
    ax.set_xlabel(r'$t$ (sec)')
    ax.set_ylabel('Pressure')
    ax.set_title('Pressure measurements')
    ax.grid(True)

    out_file = os.path.join(plot_dir, "raw_pressure.png")
    fig.savefig(out_file, dpi=300, transparent=True)


def plot_phase_avg_pressure(rad, axi, phase_avg, plot_dir, key, plot_smooth=True, pk_idx=None,
                            detonation=None, shock=None, deflagration=None,
                            reflection=None):
    peak_colors = ['black', 'red', 'orange', 'magenta', 'cyan']
    theta, AX = np.meshgrid(rad, axi)
    fig, ax = plt.subplots(figsize=(10, 6))
    c = ax.pcolormesh(theta, AX, phase_avg, shading='auto', cmap="jet")
    fig.colorbar(c, ax=ax, label="Pressure")
    if key == "interesting":
        for i in range(len(axi)):
            peaks_for_this_slice = pk_idx[i]
            if plot_smooth:
                if detonation is not None:
                    ax.scatter(detonation[i], np.full_like(peaks_for_this_slice[0], axi[i]),
                               color='black', s=10, zorder=5)
                if deflagration is not None:
                    ax.scatter(deflagration[i], np.full_like(peaks_for_this_slice[0], axi[i]),
                               color='red', s=10, zorder=5)
                if shock is not None:
                    ax.scatter(shock[i], np.full_like(peaks_for_this_slice[0], axi[i]),
                               color='blue', s=10, zorder=5)
                if reflection is not None:
                    ax.scatter(reflection[i], np.full_like(peaks_for_this_slice[0], axi[i]),
                               color='magenta', s=10, zorder=5)

            # ax.scatter(peaks_for_this_slice[0], np.full_like(peaks_for_this_slice[0], axi[i]),
            #            color='black', s=3, zorder=5)
            # if len(peaks_for_this_slice) >= 2:
            #     # Loop over every peak found in this slice
            #     for peak_idx, peak_coord in enumerate(peaks_for_this_slice):
            #         # Pick a color based on the peak's index (loops back around if you have tons of peaks)
            #         color = peak_colors[peak_idx % len(peak_colors)]
            #
            #         ax.scatter(peak_coord, axi[i],
            #                    color=color, s=3, zorder=5)

            # Plot Detonation / Oblique shock
            if peaks_for_this_slice[0] is not None:
                ax.scatter(peaks_for_this_slice[0], axi[i], color='black', s=3, zorder=5)

            # Plot Parasitic Deflagration Front (NEW!)
            if peaks_for_this_slice[1] is not None:
                ax.scatter(peaks_for_this_slice[1], axi[i], color='red', s=3, zorder=5)

            # Plot Reflected Oblique Shock
            if peaks_for_this_slice[2] is not None:
                ax.scatter(peaks_for_this_slice[2], axi[i], color='orange', s=3, zorder=5)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=np.pi))

    def pi_formatter(x, pos):
        """Format ticks as multiples of pi."""
        m = np.round(x / np.pi)
        if m == 0:
            return "0"
        elif m == 1:
            return r"$\pi$"
        elif m == -1:
            return r"$-\pi$"
        else:
            return rf"${int(m)}\pi$"

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(pi_formatter))
    ax.set_xlabel(r"$\theta$ (radians)")
    ax.set_ylabel("Axial Location (mm)")
    out_file = os.path.join(plot_dir, "phase_avg_pressure_" + key + ".png")
    fig.savefig(out_file, dpi=300, transparent=True)


def plot_phase_avg_animation(rad, axi, phase_avg, pk_idx):
    import matplotlib
    matplotlib.use('TkAgg')
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))

    # Calculate limits once
    y_min = np.min(phase_avg)
    y_max = np.max(phase_avg)

    # Draw the first line once
    line1, = ax.plot(
        rad,
        phase_avg[0, :],
        color='blue'
    )

    # FIX 1: Use ax.plot with markers instead of ax.scatter
    line2, = ax.plot(
        rad[pk_idx[0]],
        phase_avg[0, pk_idx[0]],
        marker='o',  # Draw a circle
        color='red',  # Make it stand out
        linestyle='None'  # Do not connect the dots with a line
    )

    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$\theta$ (radians)")
    ax.set_ylabel("Pressure")

    for i in range(len(axi)):
        if i % 5 == 0:
            # Update line data
            line1.set_ydata(phase_avg[i, :])

            # FIX 2: Wrap scalar values in brackets [ ] to pass them as sequences
            line2.set_xdata([rad[pk_idx[i]]])
            line2.set_ydata([phase_avg[i, pk_idx[i]]])

            ax.set_title(f"Axial Location Index: {i}")

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.1)

    plt.ioff()
    plt.close(fig)