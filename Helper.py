import numpy as np
from scipy.signal import find_peaks


def _two_peak_indices(col, prominence=None, distance=None):
    peaks, _ = find_peaks(col, prominence=prominence, distance=distance)
    if peaks.size >= 2:
        sel = peaks[np.argsort(col[peaks])[-2:]][::-1]
    elif peaks.size == 1:
        best = peaks[0]
        other = np.argmax(np.where(np.arange(col.size)==best, -np.inf, col))
        sel = np.array([best, other])
    else:
        sel = np.argpartition(col, -2)[-2:]
        sel = sel[np.argsort(col[sel])[::-1]]
    return sel


def retain_two_peaks(snapshot, prominence=None, distance=None):
    """
    snapshot: ndarray (n_x, n_t)
    returns: cleaned ndarray with only two peaks per column (others zero)
    """
    cleaned = np.zeros_like(snapshot, dtype=snapshot.dtype)
    n_x, n_t = snapshot.shape
    for j in range(n_t):
        sel = _two_peak_indices(snapshot[:, j], prominence=prominence, distance=distance)
        cleaned[sel, j] = snapshot[sel, j]
    return cleaned
