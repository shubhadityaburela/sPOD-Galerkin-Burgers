import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.fft import fftfreq

matplotlib.use("TkAgg")

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


def plot_fft_luminosity(luminosity_fft, Nt, dt):

    # Frequency bins
    # Convert to KHz
    xf = fftfreq(Nt, dt)[:Nt // 2] / 1000
    amp = 2.0 / Nt * np.abs(luminosity_fft[0:Nt // 2])
    amp = amp / np.max(amp)

    tick_spacing_khz = 1
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
