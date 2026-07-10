import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft
from scipy.signal import spectrogram, find_peaks, medfilt

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



def windowed_fft_luminosity(luminosity, theta, theta_pos, window=None, dt=None, scenario=None, plot_at_all=False):

    theta_pos_rad = np.deg2rad(theta_pos)
    theta_pos_rad_idx = np.argmin(np.abs(theta - theta_pos_rad))
    luminosity_per_theta = luminosity[theta_pos_rad_idx, :]

    if scenario == "2CR":
        f, t_spec, Sxx = spectrogram(luminosity_per_theta, fs=(1 / dt), window='hann',
                                     nperseg=window, noverlap=int(0.75 * window), nfft=1024)

        freq_vs_time = np.zeros(len(t_spec))
        for i in range(len(t_spec)):
            column = Sxx[:, i]

            # Find peaks that are at least 20% of the maximum intensity in this window
            # Adjust 'height' if your signal is very noisy
            peaks, props = find_peaks(column, height=np.max(column) * 0.2)

            if len(peaks) > 0:
                # CRITICAL: Pick the peak with the LOWEST frequency index (the fundamental)
                # Even if the 2nd harmonic is taller, peaks[0] is the leftmost peak.
                fundamental_idx = peaks[0]
                freq_vs_time[i] = f[fundamental_idx]
            else:
                # Fallback if no peaks found (optional)
                freq_vs_time[i] = freq_vs_time[i - 1] if i > 0 else 0
    elif scenario == "2CRT":
        # 1. Use a larger nfft to "zoom in" on the frequency bins
        # 8192 or 16384 provides much smoother interpolation for close peaks
        f, t_spec, Sxx = spectrogram(luminosity_per_theta, fs=(1 / dt), window='hann',
                                     nperseg=1024, noverlap=448*2, nfft=16384)

        # 2. Set your "Anchor" frequencies from your global FFT
        anchor1 = 4111.677
        anchor2 = 3919.216
        search_width = 30  # Hz (Search +/- 150Hz around the anchors)

        f1_tracked = np.zeros(len(t_spec))
        f2_tracked = np.zeros(len(t_spec))

        for i in range(len(t_spec)):
            column = Sxx[:, i]

            # Define the search zones
            zone1 = (f > anchor1 - search_width) & (f < anchor1 + search_width)
            zone2 = (f > anchor2 - search_width) & (f < anchor2 + search_width)

            # Find the peak index WITHIN each zone
            if np.any(zone1):
                f1_tracked[i] = f[zone1][np.argmax(column[zone1])]

            if np.any(zone2):
                f2_tracked[i] = f[zone2][np.argmax(column[zone2])]

        # 3. Clean up any zeros with interpolation if a window failed
        f1_tracked[f1_tracked == 0] = anchor1
        f2_tracked[f2_tracked == 0] = anchor2

    # print(f1_tracked)
    # print(f2_tracked)


    if plot_at_all:
        plt.figure(figsize=(12, 6))
        plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud', cmap='magma')
        plt.title('RDE Wave Spectrogram: Frequency Evolution')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(label='Intensity [dB]')
        plt.tight_layout()
        plt.show()

    return freq_vs_time / 1000, t_spec


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
