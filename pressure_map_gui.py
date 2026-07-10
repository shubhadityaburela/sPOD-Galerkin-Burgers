"""
Pressure Map GUI V02
--------------------
Minimal GUI that reproduces the core outputs from pressure_map_gui.py with
just five figures:
1. Reference PCB spaghetti plot aligned at ~2*pi with the phase-averaged mean.
2. Phase-averaged traces stacked for PCB01-PCB13.
3. Phase-averaged traces with primary (>=2*pi) and secondary (2*pi..4*pi) peaks.
4. Discrete sensor map (scatter, no interpolation) over 0..4*pi.
5. Phase-averaged pressure map with peak markers (primary + secondary).
"""
from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from itertools import combinations

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib import colors as mcolors

from scipy.io import loadmat
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter, welch, get_window
from scipy.interpolate import CubicHermiteSpline, CubicSpline, RectBivariateSpline, griddata

# Consistent journal-ready typography
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------


def pi_formatter(x, _pos):
    n = x / np.pi
    if np.isclose(n, 0):
        return "0"
    if np.isclose(n, 1):
        return r"$\pi$"
    if np.isclose(n % 1, 0):
        return r"${:.0f}\pi$".format(n)
    return r"${:.1f}\pi$".format(n)


def pi_formatter_plain(x, _pos):
    """Plain-text pi tick labels (uniform font, no mathtext)."""
    n = float(x / np.pi)
    if np.isclose(n, 0.0):
        return "0"
    if np.isclose(n, 1.0):
        return "π"
    if np.isclose(n, -1.0):
        return "-π"
    if np.isclose(n % 1, 0.0):
        return f"{int(round(n))}π"
    return f"{n:.1f}π"


LENGTH_TOKEN_RE = re.compile(r"L\s*(\d{2,4})(?!\d)", flags=re.IGNORECASE)


def _parse_length_token(text: str) -> float | None:
    """Extract L### token from arbitrary text."""
    if not text:
        return None
    try:
        match = LENGTH_TOKEN_RE.search(text)
        if match:
            return float(match.group(1))
    except Exception:
        return None
    return None


def _coerce_to_str(value) -> str | None:
    """Convert MATLAB-loaded values (nested arrays/bytes) into a plain string."""
    try:
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            return _coerce_to_str(value.flat[0])
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except Exception:
                return value.decode(errors="ignore")
        if isinstance(value, str):
            return value
        return str(value)
    except Exception:
        return None


def parse_length_mm_from_path(path: str) -> float | None:
    """Extract Lxxx token from any path or filename string."""
    return _parse_length_token(path)


def parse_length_mm_from_metadata(mat_struct, mat_path: str | None = None) -> float | None:
    """Extract combustor length from MAT metadata (fileName/pathName) with path fallback."""
    for field in ("fileName", "pathName"):
        try:
            raw_val = mat_struct[field]
        except Exception:
            continue
        text = _coerce_to_str(raw_val)
        length = _parse_length_token(text) if text else None
        if length is not None:
            return length
    return parse_length_mm_from_path(mat_path) if mat_path else None


PCB_LABELS = [f"pcb{ii:02d}" for ii in range(1, 14)]
DEFAULT_POS_MM = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125]
COMBUSTOR_RADIUS_MM = 45.0
COMBUSTOR_CIRCUMFERENCE_MM = 2.0 * np.pi * COMBUSTOR_RADIUS_MM
PHASE_PER_MM = (2.0 * np.pi) / COMBUSTOR_CIRCUMFERENCE_MM
REFLECTION_BETA_MIN_DEG = 10.0
REFLECTION_BETA_MAX_DEG = 80.0
SECONDARY_WINDOW_EXTRA_MM = 10.0
DEFAULT_SECONDARY_WINDOW_GAIN = 1.0
FLAT_GRADIENT_THRESHOLD = 0.18
PRIMARY_POST_2PI_MIN_PRESSURE = 0.5  # enforce a real peak after 2π
PHASE_LIMIT_DISCRETE = 4.0 * np.pi
PHASE_LIMIT_MAP = 4.0 * np.pi
PRIMARY_COLOR = "#ff9800"
SECONDARY_COLOR = "#26c6da"
PRIMARY_POLY_COLOR = "#6a1b9a"
PRIMARY_HERMITE_COLOR = "#4fc3f7"
SECOND_DERIV_COLOR = "#d81b60"
GRADIENT_TRACE_COLOR = "#1e88e5"
SECOND_GRADIENT_COLOR = "#bcaaa4"
RIDGE_COLOR = "#ff7043"
OBLIQUE_RANSAC_COLOR = "#2e7d32"
SECONDARY_WINDOW_COLOR = "#8d6e63"
DEFAULT_OBLIQUE_TOL_DEG = 5.0
ANNOTATION_DARK_ORANGE = "#bf360c"
ANNOTATION_DARK_BLUE = "#0d47a1"
ALPHA_H_FONT_SIZE = 10
BETA_YR_FONT_SIZE = 10
PRIMARY_MIN_REL_AMP = 0.5  # keep primary peaks above 50% of the available amplitude
PRIMARY_TOP_PERCENTILE = 95.0  # only consider the strongest ~5% portion of the profile
THETA_TICK_STEP = 0.25 * np.pi  # show quarter-pi ticks on phase plots
SEED_WINDOW_COLOR = "#1e88e5"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = BASE_DIR / "processed_full_data_set" / "L100" / "OR25"
DEFAULT_DATASET_FILE = "CO1083_L100_OR25_A0358_P0983_20251013.mat"
DEFAULT_SECONDARY_SEED_SPAN = 1.2 * np.pi
DEFAULT_DPDY_SMOOTH_FRAC = 0.15
DP_DY_SPLINE_SMOOTHING = 0.02  # scaling factor for smoothing spline tolerance


def _norm_key(name: str) -> str | None:
    name = name.strip().lower()
    if not name.startswith("pcb"):
        return None
    digits = "".join(ch for ch in name if ch.isdigit())
    if not digits:
        return None
    return f"pcb{int(digits):02d}"


def load_dataset(path: str):
    mat = loadmat(path)
    D = mat["D"][0, 0]
    t = D["time"][0, 0][0].astype(float).flatten()
    pcb = {}
    for fld in D.dtype.names:
        nk = _norm_key(fld)
        if nk is None:
            continue
        try:
            arr = D[fld][0, 0]["data"]
            data = np.array(arr).astype(float).flatten()
        except Exception:
            arr = D[fld]
            if hasattr(arr, "dtype") and arr.dtype.fields is None:
                data = np.array(arr).astype(float).flatten()
            else:
                continue
        pcb[nk] = data
    keys = sorted(pcb.keys(), key=lambda k: int(k[-2:]))
    return t, {k: pcb[k] for k in keys}, D


def _extract_soft_dtw_phase_average(mat_struct):
    """Return (phase, barycenter, labels) from D.window.soft_dtw when available."""
    try:
        if not hasattr(mat_struct, "dtype") or mat_struct.dtype.names is None:
            return None
        if "window" not in mat_struct.dtype.names:
            return None
        win = mat_struct["window"]
        if getattr(win, "size", 0) == 0:
            return None
        win0 = win[0, 0] if np.ndim(win) >= 2 else win.item()
        if not hasattr(win0, "dtype") or win0.dtype.names is None:
            return None
        if "soft_dtw" not in win0.dtype.names:
            return None

        soft = win0["soft_dtw"]
        if getattr(soft, "size", 0) == 0:
            return None
        soft0 = soft[0, 0] if np.ndim(soft) >= 2 else soft.item()
        if not hasattr(soft0, "dtype") or soft0.dtype.names is None:
            return None
        needed = {"phase", "barycenter", "pcb_labels"}
        if not needed.issubset(set(soft0.dtype.names)):
            return None

        phase = np.asarray(soft0["phase"], dtype=float).reshape(-1)
        bary = np.asarray(soft0["barycenter"], dtype=float)
        if phase.size == 0 or bary.size == 0 or bary.ndim != 2:
            return None
        if bary.shape[1] != phase.size and bary.shape[0] == phase.size:
            bary = bary.T
        if bary.shape[1] != phase.size:
            return None

        raw_labels = np.asarray(soft0["pcb_labels"]).reshape(-1)
        labels = []
        for raw in raw_labels:
            val = raw
            if isinstance(val, np.ndarray):
                if val.size == 0:
                    continue
                val = val.flat[0]
            nk = _norm_key(str(val))
            if nk is not None:
                labels.append(nk)

        if len(labels) != bary.shape[0]:
            labels = [f"pcb{idx + 1:02d}" for idx in range(bary.shape[0])]
        return phase, bary, labels
    except Exception:
        return None


def butter_bandpass_filter(t, x, f1, f2, order=4):
    dt = float(np.mean(np.diff(t)))
    fs = 1.0 / dt
    wn = [2.0 * f1 / fs, 2.0 * f2 / fs]
    b, a = butter(order, wn, btype="bandpass")
    return filtfilt(b, a, x)


def estimate_det_freq(time, signals, fmin=3000.0, fmax=20000.0):
    dt = float(np.mean(np.diff(time)))
    n = len(time)
    win = np.hanning(n)
    freqs = np.fft.rfftfreq(n, d=dt)
    f_est = []
    for sig in signals:
        x = sig - np.mean(sig)
        spec = np.fft.rfft(win * x)
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(mask):
            continue
        k = np.argmax(np.abs(spec[mask]))
        f_est.append(freqs[mask][k])
    return float(np.mean(f_est)) if f_est else None


def estimate_ref_freq_welch(time, signal):
    dt = float(np.mean(np.diff(time)))
    if dt <= 0:
        return None
    fs = 1.0 / dt
    n = len(time)
    if n < 8:
        return None
    win_len = min(2**15, n)
    if win_len < 64:
        return None
    nover = win_len // 2
    win = get_window('hann', win_len, fftbins=True)
    f, Pxx = welch(signal, fs=fs, window=win, nperseg=win_len, noverlap=nover,
                   detrend='constant', scaling='density')
    lo = np.searchsorted(f, 1000.0)
    hi = np.searchsorted(f, fs / 4.0)
    if hi <= lo:
        return None
    idx = int(np.argmax(Pxx[lo:hi]) + lo)
    return float(f[idx])


def pick_phase_locked_peaks(time, signal, f_guess, last_win_s=0.050, height_frac=0.2, min_sep_frac=0.6):
    dt = float(np.mean(np.diff(time)))
    fs = 1.0 / dt
    T = 1.0 / max(f_guess, 1e-9)
    min_dist = max(1, int(min_sep_frac * T / dt))
    start_idx = max(0, len(time) - int(last_win_s * fs))
    s_win = signal[start_idx:]
    thr = np.median(s_win) + height_frac * (np.max(s_win) - np.median(s_win))
    peaks, _ = find_peaks(s_win, distance=min_dist, height=thr)
    return peaks + start_idx


def build_grad_aligned_segments(time, signals_dict, ref_key, pk_idx, fs, f_det, waves_per_rev, grad_left_frac=0.2):
    f_rev = f_det / float(max(waves_per_rev, 1))
    samples_per_rev = int(round(fs / max(f_rev, 1e-9)))
    seg_len = 3 * samples_per_rev
    X_rev = np.linspace(0.0, 2 * np.pi * 3, seg_len)
    ref = signals_dict[ref_key]
    dref = np.gradient(ref)
    w_left = max(1, int(round(grad_left_frac * samples_per_rev)))
    g_idx = []
    for p in pk_idx:
        a = max(0, p - w_left)
        b = p + 1
        if b <= a + 1:
            continue
        loc = np.argmax(dref[a:b])
        g = a + loc
        start = g - samples_per_rev
        end = g + 2 * samples_per_rev
        if start >= 0 and end <= len(ref):
            g_idx.append((start, end))
    seg_bank = {}
    for k, sig in signals_dict.items():
        segs = [sig[s:e] for (s, e) in g_idx]
        seg_bank[k] = np.stack(segs, axis=0) if segs else np.empty((0, seg_len))
    return seg_bank, X_rev, samples_per_rev


def _peak_candidates(x_vals, y_vals):
    mask = np.isfinite(y_vals)
    if np.count_nonzero(mask) < 3:
        return None, None
    x = x_vals[mask]
    y = y_vals[mask]
    span = float(np.nanmax(y) - np.nanmin(y))
    if span <= 1e-6:
        return None, None
    prominence = max(0.02, 0.08 * span)
    distance = max(1, int(round(0.03 * len(y))))
    idxs, _ = find_peaks(y, prominence=prominence, distance=distance)
    if idxs.size == 0:
        idxs = np.array([np.argmax(y)], dtype=int)
    x_peaks = x[idxs]
    y_peaks = y[idxs]
    return x_peaks, y_peaks


def detect_primary_windows(x_vals, y_vals, start_phases, window_span=2.0 * np.pi):
    x_peaks, y_peaks = _peak_candidates(x_vals, y_vals)
    if x_peaks is None:
        return []
    y_finite = y_vals[np.isfinite(y_vals)]
    if y_finite.size == 0:
        return []
    amp_max = float(np.nanmax(y_finite))
    if not np.isfinite(amp_max) or amp_max <= 1e-9:
        return []
    perc_thresh = float(np.nanpercentile(y_finite, PRIMARY_TOP_PERCENTILE))
    amp_thresh = max(PRIMARY_MIN_REL_AMP * amp_max, perc_thresh)
    peaks = []
    for start in start_phases:
        end = start + window_span
        mask = (x_peaks >= start) & (x_peaks <= end)
        if not np.any(mask):
            mask = x_peaks >= start
        if not np.any(mask):
            continue
        idx_candidates = np.where(mask)[0]
        if idx_candidates.size == 0:
            continue
        thr = amp_thresh
        if start >= 2.0 * np.pi:
            thr = max(thr, PRIMARY_POST_2PI_MIN_PRESSURE)
        idx_candidates = idx_candidates[y_peaks[idx_candidates] >= thr]
        if idx_candidates.size == 0:
            continue
        idx = idx_candidates[np.argmin(x_peaks[idx_candidates])]
        peaks.append(
            {
                "phase": float(x_peaks[idx]),
                "pressure": float(y_peaks[idx]),
                "window_start": float(start),
            }
        )
    return peaks


def summarize_peaks(
    X_rev,
    Z,
    Y_ax,
    length_limit_mm,
    secondary_seed_offset=0,
    secondary_seed_span=DEFAULT_SECONDARY_SEED_SPAN,
    secondary_seed_bounds=None,
    secondary_window_gain=DEFAULT_SECONDARY_WINDOW_GAIN,
    secondary_allow_wrap_once=False,
):
    summary = []
    enabled_indices = []
    start_phases = [0.0, 2.0 * np.pi, 4.0 * np.pi]
    for idx, name in enumerate(PCB_LABELS):
        row = Z[idx, :] if idx < Z.shape[0] else np.full_like(X_rev, np.nan)
        within_limit = (
            length_limit_mm is None
            or not math.isfinite(length_limit_mm)
            or (idx < len(Y_ax) and float(Y_ax[idx]) <= length_limit_mm + 1e-6)
        )
        if within_limit:
            enabled_indices.append(idx)
            valid_starts = [st for st in start_phases if st < float(X_rev[-1]) - 1e-9]
            primary_list = detect_primary_windows(X_rev, row, valid_starts)
            primary_ref = None
            for peak in primary_list:
                if 2.0 * np.pi <= peak["phase"] <= 3.0 * np.pi:
                    primary_ref = peak
                    break
            if primary_ref is None and primary_list:
                primary_ref = primary_list[0]
        else:
            primary_ref = None
            primary_list = []
        summary.append(
            {
                "pcb": name.upper(),
                "primary": primary_ref,
                "secondary": None,
                "primary_list": primary_list,
                "secondary_window": None,
                "enabled": within_limit,
            }
    )
    _assign_outlet_secondary(
        summary,
        Z,
        X_rev,
        Y_ax,
        enabled_indices,
        length_limit_mm,
        secondary_seed_offset=secondary_seed_offset,
        secondary_seed_span=secondary_seed_span,
        secondary_seed_bounds=secondary_seed_bounds,
        secondary_window_gain=secondary_window_gain,
        secondary_allow_wrap_once=bool(secondary_allow_wrap_once),
    )
    return summary


def wrap_phase(value, phase_min, phase_max):
    span = 2.0 * np.pi
    if not np.isfinite(value):
        return None
    if phase_max <= phase_min:
        return value
    while value < phase_min:
        value += span
    while value > phase_max:
        value -= span
    return value


def _primary_phase_in_window(entry, phase_min, phase_max):
    plist = entry.get("primary_list") or []
    for peak in plist:
        if phase_min <= peak["phase"] <= phase_max:
            return peak["phase"]
    if plist:
        return plist[0]["phase"]
    primary = entry.get("primary")
    if primary is None:
        return None
    phase = primary.get("phase")
    if phase is None:
        return None
    return phase if phase_min <= phase <= phase_max else phase


def _assign_outlet_secondary(
    summary,
    Z,
    X_rev,
    Y_ax,
    enabled_indices,
    length_mm,
    secondary_seed_offset=0,
    secondary_seed_span=DEFAULT_SECONDARY_SEED_SPAN,
    secondary_seed_bounds=None,
    secondary_window_gain=DEFAULT_SECONDARY_WINDOW_GAIN,
    secondary_allow_wrap_once=False,
):
    if not enabled_indices:
        return
    outlet_idx = _determine_outlet_index(enabled_indices, Y_ax, length_mm)
    seed_idx = _determine_secondary_seed_index(enabled_indices, outlet_idx, offset=2)
    if seed_idx is None:
        return
    seed_shift = max(0, int(secondary_seed_offset or 0))
    seed_span = float(secondary_seed_span) if secondary_seed_span else DEFAULT_SECONDARY_SEED_SPAN
    seed_span = max(seed_span, np.pi * 0.5)
    window_gain = float(secondary_window_gain) if secondary_window_gain else DEFAULT_SECONDARY_WINDOW_GAIN
    window_gain = max(0.5, min(5.0, window_gain))
    entry = summary[seed_idx]
    row = Z[seed_idx, :]
    window_used = None
    fallback_window = None
    if secondary_seed_bounds and len(secondary_seed_bounds) == 2:
        lo, hi = sorted((float(secondary_seed_bounds[0]), float(secondary_seed_bounds[1])))
        window_used = (max(0.0, lo), min(hi, float(X_rev[-1])))
        summary[seed_idx]["secondary_seed_window"] = window_used
    candidate = _find_seed_secondary(row, X_rev, entry, offset=seed_shift, span_limit=seed_span, bounds=secondary_seed_bounds)
    if candidate:
        summary[seed_idx]["secondary"] = candidate
        if window_used:
            summary[seed_idx]["secondary_window"] = window_used
            summary[seed_idx]["secondary_seed_window"] = window_used
        _propagate_secondary_chain(
            summary,
            Z,
            X_rev,
            Y_ax,
            enabled_indices,
            seed_idx,
            candidate["phase"],
            window_gain=window_gain,
            allow_wrap_once=bool(secondary_allow_wrap_once),
        )
        return
    primary_phase = _primary_phase_in_window(entry, 0.0, 2.0 * np.pi)
    if primary_phase is None:
        return
    row = Z[seed_idx, :]
    phase_start = primary_phase + 1e-6
    phase_end = 2.0 * np.pi
    if phase_end <= phase_start:
        phase_end = min(primary_phase + np.pi, float(X_rev[-1]))
    fallback_window = (phase_start, phase_end)
    if window_used:
        fallback_window = (max(window_used[0], phase_start), min(window_used[1], phase_end))
    if fallback_window[1] <= fallback_window[0]:
        return
    candidate = _peak_in_window(row, X_rev, fallback_window, rank=seed_shift)
    if not candidate:
        neighbor_idx = max((idx for idx in enabled_indices if idx < seed_idx), default=None)
        if neighbor_idx is not None:
            fallback_window = _fallback_secondary_window(float(Y_ax[seed_idx]), float(Y_ax[neighbor_idx]), gain=window_gain)
            if fallback_window:
                if window_used:
                    fallback_window = (
                        max(window_used[0], fallback_window[0]),
                        min(window_used[1], fallback_window[1]),
                    )
                if fallback_window[1] > fallback_window[0]:
                    candidate = _peak_in_window(row, X_rev, fallback_window, rank=seed_shift)
                    if candidate is None:
                        candidate = _peak_in_window_relaxed(row, X_rev, fallback_window, rank=seed_shift)
    if not candidate:
        return
    summary[seed_idx]["secondary"] = candidate
    if window_used:
        summary[seed_idx]["secondary_window"] = window_used
        summary[seed_idx]["secondary_seed_window"] = window_used
    elif fallback_window:
        summary[seed_idx]["secondary_window"] = fallback_window
        summary[seed_idx]["secondary_seed_window"] = fallback_window
    _propagate_secondary_chain(
        summary,
        Z,
        X_rev,
        Y_ax,
        enabled_indices,
        seed_idx,
        candidate["phase"],
        window_gain=window_gain,
        allow_wrap_once=bool(secondary_allow_wrap_once),
    )
    return
    summary[seed_idx]["secondary"] = candidate
    _propagate_secondary_chain(
        summary,
        Z,
        X_rev,
        Y_ax,
        enabled_indices,
        seed_idx,
        candidate["phase"],
        window_gain=window_gain,
    )


def _propagate_secondary_chain(
    summary,
    Z,
    X_rev,
    Y_ax,
    enabled_indices,
    start_idx,
    start_phase,
    window_gain=1.0,
    allow_wrap_once=False,
):
    current_idx = start_idx
    current_phase = start_phase
    wrap_used = False
    two_pi = 2.0 * np.pi
    while True:
        target_idx = max((idx for idx in enabled_indices if idx < current_idx), default=None)
        if target_idx is None:
            break
        window = _phase_window_from_beta(
            current_phase,
            float(Y_ax[current_idx]),
            float(Y_ax[target_idx]),
            gain=window_gain,
            allow_over_2pi=bool(allow_wrap_once and not wrap_used),
        )
        if window is None:
            break
        is_wrap_window = bool(window[1] > two_pi + 1e-12)
        summary[target_idx]["secondary_window"] = window
        peak = _peak_in_window(Z[target_idx, :], X_rev, window)
        if not peak:
            peak = _peak_in_window_relaxed(Z[target_idx, :], X_rev, window)
        if not peak:
            break
        summary[target_idx]["secondary"] = peak
        if is_wrap_window:
            # Allow one boundary-crossing search window, then stop.
            wrap_used = True
            break
        current_idx = target_idx
        current_phase = peak["phase"]


def _find_seed_secondary(row, X_rev, entry, offset=0, span_limit=DEFAULT_SECONDARY_SEED_SPAN, bounds=None):
    if bounds and len(bounds) == 2:
        lo, hi = sorted((float(bounds[0]), float(bounds[1])))
        window = (max(0.0, lo), min(hi, float(X_rev[-1])))
    else:
        window = (0.0, min(float(span_limit), float(X_rev[-1])))
    candidate = _peak_in_window(row, X_rev, window, rank=max(0, int(offset)))
    if candidate is None:
        candidate = _peak_in_window_relaxed(row, X_rev, window, rank=max(0, int(offset)))
    if candidate:
        return candidate
    # Fallback to primary peaks, but still respect the requested window
    primary = entry.get("primary")
    if primary:
        phase = float(primary["phase"])
        if window[0] <= phase <= window[1]:
            return {"phase": phase, "pressure": primary["pressure"]}
    primary_list = entry.get("primary_list") or []
    if primary_list:
        peak = primary_list[0]
        phase = float(peak["phase"])
        if window[0] <= phase <= window[1]:
            return {"phase": phase, "pressure": peak["pressure"]}
    return None


def _beta_phase_offsets(delta_y):
    if delta_y <= 1e-6:
        return None
    beta_min = math.radians(REFLECTION_BETA_MIN_DEG)
    beta_max = math.radians(REFLECTION_BETA_MAX_DEG)
    offset_min = delta_y * PHASE_PER_MM * math.tan(beta_min)
    offset_max = delta_y * PHASE_PER_MM * math.tan(beta_max)
    if not np.isfinite(offset_min) or not np.isfinite(offset_max):
        return None
    return (min(offset_min, offset_max), max(offset_min, offset_max))


def _phase_window_from_beta(seed_phase, y_seed, y_target, gain=1.0, allow_over_2pi=False):
    offsets = _beta_phase_offsets(abs(y_seed - y_target))
    if offsets is None:
        return None
    offset_min, offset_max = offsets
    gain = max(0.5, min(5.0, float(gain)))
    phase_min = seed_phase + offset_min
    phase_max = seed_phase + offset_max * gain + SECONDARY_WINDOW_EXTRA_MM * PHASE_PER_MM
    two_pi = 2.0 * np.pi
    if phase_min >= two_pi:
        return None
    if (phase_max > two_pi) and (not allow_over_2pi):
        # Terminal no-wrap search: clip the final window at 2*pi so one last
        # reflected-shock candidate can still be evaluated before stopping.
        phase_max = two_pi
        if phase_max <= phase_min:
            return None
    return phase_min, phase_max


def _fallback_secondary_window(y_seed, neighbor_y, gain=1.0):
    offsets = _beta_phase_offsets(abs(y_seed - neighbor_y))
    if offsets is None:
        return None
    gain = max(0.5, min(5.0, float(gain)))
    half_span = gain * max(abs(offsets[0]), abs(offsets[1])) + SECONDARY_WINDOW_EXTRA_MM * PHASE_PER_MM
    if half_span <= 1e-6:
        return None
    center = 0.9 * np.pi
    phase_min = max(0.0, center - half_span)
    phase_max = min(2.0 * np.pi, center + half_span)
    if phase_max <= phase_min:
        return None
    return phase_min, phase_max


def _angle_from_slope(slope):
    if abs(slope) < 1e-12:
        return float("nan")
    dy_dx = (2.0 * np.pi) / (COMBUSTOR_CIRCUMFERENCE_MM * slope)
    return float(np.degrees(np.arctan(dy_dx)))


def _fit_line(points):
    y = np.array([p[1] for p in points], dtype=float)
    phase = np.array([p[0] for p in points], dtype=float)
    slope, intercept = np.polyfit(y, phase, 1)
    return float(slope), float(intercept)


def fit_oblique_from_primary(points, length_mm, tol_deg, phase_min, phase_max):
    if not points or len(points) < 3:
        return None
    pts_sorted = sorted(points, key=lambda p: p[1])
    if length_mm is not None and np.isfinite(length_mm):
        target_y = 0.5 * float(length_mm) + 1.0
    else:
        target_y = 0.5 * (pts_sorted[-1][1] + pts_sorted[0][1])
    idx_center = int(np.argmin([abs(y - target_y) for _, y in pts_sorted]))
    base = pts_sorted[idx_center]
    # find immediate neighbors above/below within the phase window
    lower = None
    upper = None
    lower_idx_val = None
    upper_idx_val = None
    for j in range(idx_center - 1, -1, -1):
        candidate = pts_sorted[j]
        if candidate[0] >= phase_min:
            lower = candidate
            lower_idx_val = j
            break
    for j in range(idx_center + 1, len(pts_sorted)):
        candidate = pts_sorted[j]
        if candidate[0] <= phase_max:
            upper = candidate
            upper_idx_val = j
            break
    if lower is None or upper is None:
        return None
    selected = [lower, base, upper]
    slope, intercept = _fit_line(selected)
    angle = _angle_from_slope(slope)
    if not np.isfinite(angle):
        return None
    tol = max(0.1, abs(tol_deg))

    def try_add(idx, insert_front):
        nonlocal selected, slope, intercept, angle
        candidate = pts_sorted[idx]
        new_points = selected + [candidate] if not insert_front else [candidate] + selected
        new_points_sorted = sorted(new_points, key=lambda p: p[1])
        slope_new, intercept_new = _fit_line(new_points_sorted)
        angle_new = _angle_from_slope(slope_new)
        if not np.isfinite(angle_new) or abs(angle_new - angle) > tol:
            return False
        selected = new_points_sorted
        slope = slope_new
        intercept = intercept_new
        angle = angle_new
        return True

    upper_idx = (upper_idx_val if upper_idx_val is not None else idx_center) + 1
    while upper_idx < len(pts_sorted) and try_add(upper_idx, False):
        upper_idx += 1
    lower_idx = (lower_idx_val if lower_idx_val is not None else idx_center) - 1
    while lower_idx >= 0 and try_add(lower_idx, True):
        lower_idx -= 1

    if len(selected) < 3:
        return None
    seed_point = pts_sorted[idx_center]
    return {
        "slope": slope,
        "intercept": intercept,
        "points": selected,
        "seed_point": seed_point,
    }


def fit_oblique_ransac(
    points,
    length_mm,
    tol_deg,
    phase_min,
    phase_max,
    angle_min_deg=50.0,
    angle_max_deg=80.0,
    iterations=80,
    phase_tol=0.12 * np.pi,
    min_inlier_fraction=0.6,
    verbose=True,
    pcb_ids=None,
):
    if not points or len(points) < 3:
        return None
    pts = np.array(points, dtype=float)
    y_vals = pts[:, 1]
    phase_vals = pts[:, 0]
    total_candidates = len(points)
    min_inliers = max(3, int(math.ceil(min_inlier_fraction * total_candidates)))
    if total_candidates <= 5:
        combos = list(combinations(range(total_candidates), 3))
    else:
        rng = np.random.default_rng(12345)
        combo_set = set()
        max_unique = math.comb(total_candidates, 3)
        target = min(max(10, iterations), max_unique)
        while len(combo_set) < target:
            idx = tuple(sorted(rng.choice(total_candidates, 3, replace=False)))
            combo_set.add(idx)
            if len(combo_set) == max_unique:
                break
        combos = list(combo_set)

    if verbose:
        print(
            f"[RANSAC] candidates={total_candidates}, combos={len(combos)}, "
            f"min_inliers={min_inliers}, angle_window=({angle_min_deg},{angle_max_deg})"
        )
        if pcb_ids:
            print(f"[RANSAC] PCB candidates: {pcb_ids}")

    best = None
    best_inliers = None
    best_angle = None
    for combo in combos:
        sample = [points[i] for i in combo]
        slope, intercept = _fit_line(sample)
        angle = abs(_angle_from_slope(slope))
        if not np.isfinite(angle) or not (angle_min_deg <= angle <= angle_max_deg):
            continue
        phase_pred = slope * y_vals + intercept
        residual = np.abs(phase_vals - phase_pred)
        inliers = np.where(residual <= phase_tol)[0]
        if inliers.size < min_inliers:
            continue
        if best is None or inliers.size > best or (
            inliers.size == best and residual[inliers].sum() < residual[best_inliers].sum()
        ):
            best = inliers.size
            best_inliers = inliers
            best_angle = angle

    if best_inliers is None:
        if verbose:
            print("[RANSAC] No valid fit found.")
        return None
    inlier_points = [points[i] for i in best_inliers]
    slope, intercept = _fit_line(inlier_points)
    angle = _angle_from_slope(slope)
    if not np.isfinite(angle):
        return None
    if verbose:
        inlier_pcbs = [pcb_ids[i] if pcb_ids else i for i in best_inliers]
        print(
            f"[RANSAC] best_inliers={len(best_inliers)}/{total_candidates}, "
            f"angle={angle:.1f} deg, inlier_pcbs={inlier_pcbs}"
        )

    return {
        "slope": slope,
        "intercept": intercept,
        "points": inlier_points,
        "seed_point": inlier_points[0],
        "method": "ransac",
    }


def _peak_in_window(row, X_rev, window, rank=0):
    x_peaks, y_peaks = _peak_candidates(X_rev, row)
    y_finite = row[np.isfinite(row)]
    amp_thresh = None
    if y_finite.size:
        amp_max = float(np.nanmax(y_finite))
        perc = float(np.nanpercentile(y_finite, PRIMARY_TOP_PERCENTILE))
        amp_thresh = max(PRIMARY_MIN_REL_AMP * amp_max, perc)
    start, end = sorted(window)
    if x_peaks is not None:
        mask = (x_peaks >= start) & (x_peaks <= end)
        if np.any(mask):
            idxs = np.where(mask)[0]
            if idxs.size == 0:
                return None
            if amp_thresh is not None:
                idxs = idxs[y_peaks[idxs] >= amp_thresh]
                if idxs.size == 0:
                    return None
            best_idx = idxs[np.argmax(y_peaks[idxs])]
            rank = max(0, int(rank))
            if rank == 0:
                chosen = best_idx
            else:
                phase_sorted = idxs[np.argsort(x_peaks[idxs])]
                target_pos = min(rank, len(phase_sorted) - 1)
                chosen = phase_sorted[target_pos]
            return {"phase": float(x_peaks[chosen]), "pressure": float(y_peaks[chosen])}
    return _fallback_peak_by_gradient(row, X_rev, start, end)


def _peak_in_window_relaxed(row, X_rev, window, rank=0):
    """Pick a peak in a window without percentile/relative thresholding."""
    x_peaks, y_peaks = _peak_candidates(X_rev, row)
    start, end = sorted(window)
    if x_peaks is not None:
        mask = (x_peaks >= start) & (x_peaks <= end)
        if np.any(mask):
            idxs = np.where(mask)[0]
            if idxs.size == 0:
                return None
            best_idx = idxs[np.argmax(y_peaks[idxs])]
            rank = max(0, int(rank))
            if rank == 0:
                chosen = best_idx
            else:
                phase_sorted = idxs[np.argsort(x_peaks[idxs])]
                target_pos = min(rank, len(phase_sorted) - 1)
                chosen = phase_sorted[target_pos]
            return {"phase": float(x_peaks[chosen]), "pressure": float(y_peaks[chosen])}
    return None


def _fallback_peak_by_gradient(row, X_rev, start, end):
    mask = (X_rev >= start) & (X_rev <= end) & np.isfinite(row)
    idx = np.where(mask)[0]
    if idx.size < 3:
        return None
    x = X_rev[idx]
    y = row[idx]
    grad = np.gradient(y, x)
    pos_to_neg = (grad[:-1] > 0.0) & (grad[1:] <= 0.0)
    candidates = idx[:-1][pos_to_neg]
    if candidates.size == 0:
        rel_idx = int(np.argmax(y))
        grad_val = abs(grad[rel_idx])
        amp_range = float(np.max(y) - np.min(y)) if np.size(y) else 0.0
        amp_range = max(amp_range, 1e-6)
        window_width = max(end - start, 1e-6)
        norm_grad = grad_val * window_width / amp_range
        if norm_grad > FLAT_GRADIENT_THRESHOLD:
            return None
        peak_idx = idx[rel_idx]
    else:
        peak_idx = candidates[np.argmax(row[candidates])]
    return {"phase": float(X_rev[peak_idx]), "pressure": float(row[peak_idx])}


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


def build_figures(data, show_peak_gradient=False, show_ridge_detect=False):
    figures = []
    figspec = [
        ("Fig 1 - Ref Alignment", figure_phase_reference),
        ("Fig 2 - Stacked Traces", figure_stacked_traces),
        ("Fig 3 - Peaks", figure_peaks_panel),
    ]
    if show_peak_gradient:
        figspec.append(("Fig 3a - PGF", figure_peak_gradient_finder))
    figspec.extend(
        [
            ("Fig 4 - Discrete Map", figure_discrete_map),
            ("Fig 5a - Phase-averaged Map", figure_pressure_map),
            ("Fig 5b - Phase-averaged Map (journal aspect)", figure_pressure_map_journal),
        ]
    )
    if show_ridge_detect:
        figspec.append(("Fig 5c - ride detect", figure_ridge_detect))
    figspec.append(("Fig 6 - Primary Polynomial Map", figure_primary_polynomial_map))
    figspec.append(("Fig 7 - Speed (PCB03-08)", figure_speed_primary))
    for label, builder in figspec:
        fig = builder(data)
        figures.append((label, fig))
    return figures


def figure_phase_reference(data):
    X_rev = data["X_rev"]
    ref_key = data["ref_key"]
    segs = data["seg_bank"].get(ref_key, np.empty((0, len(X_rev))))
    avg_ref = segs.mean(axis=0) if segs.size else np.zeros_like(X_rev)
    fig, ax = plt.subplots(figsize=(6.5, 3.25))
    for seg in segs:
        ax.plot(X_rev, seg, color="0.75", alpha=0.3, linewidth=0.8)
    ax.plot(X_rev, avg_ref, color="tab:red", linewidth=2.2, label="Phase-averaged")
    ax.set_xlim(data["xlim"])
    ax.set_xticks(data["theta_marks"])
    ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
    ax.set_xlabel("Phase [rad]")
    ax.set_ylabel("Pressure [bar]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def figure_stacked_traces(data):
    X_rev = data["X_rev"]
    Z = data["avg_matrix"]
    fig, axes = plt.subplots(13, 1, figsize=(11, 1.7 * 13 + 1.2), sharex=True)
    for i in range(13):
        sensor_idx = 12 - i
        key = PCB_LABELS[sensor_idx]
        trace = Z[sensor_idx, :] if sensor_idx < Z.shape[0] else np.full_like(X_rev, np.nan)
        ax = axes[i]
        if np.all(~np.isfinite(trace)):
            ax.text(0.02, 0.5, f"{key.upper()} missing", transform=ax.transAxes, color="0.4")
            ax.set_ylim(0, 1)
            ax.set_yticks([])
        else:
            ax.plot(X_rev, trace, color="tab:red", linewidth=1.8)
            ax.grid(True, alpha=0.25)
        ax.set_ylabel(key.upper())
        ax.set_xlim(data["xlim"])
        xticks = np.arange(data["xlim"][0], data["xlim"][1] + 1e-6, THETA_TICK_STEP)
        if xticks.size == 0:
            xticks = data["theta_marks"]
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
    axes[-1].set_xlabel("Phase [rad]")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def figure_peaks_panel(data):
    X_rev = data["X_rev"]
    Z = data["avg_matrix"]
    summary = data["peak_summary"]
    fig, axes = plt.subplots(13, 1, figsize=(11, 1.7 * 13 + 1.5), sharex=True)
    legend_handles = [
        Line2D([], [], marker="o", markersize=8, color="none", markerfacecolor=PRIMARY_COLOR, markeredgecolor="black", label="Primary peaks (0/2π/4π)"),
        Line2D([], [], marker="D", markersize=8, color="none", markerfacecolor=SECONDARY_COLOR, markeredgecolor="black", label="Secondary peak"),
    ]
    window_legend_added = False
    for i in range(13):
        sensor_idx = 12 - i
        key = PCB_LABELS[sensor_idx]
        trace = Z[sensor_idx, :] if sensor_idx < Z.shape[0] else np.full_like(X_rev, np.nan)
        ax = axes[i]
        if np.all(~np.isfinite(trace)):
            ax.text(0.02, 0.5, f"{key.upper()} missing", transform=ax.transAxes, color="0.4")
            ax.set_ylim(0, 1)
            ax.set_yticks([])
        else:
            ax.plot(X_rev, trace, color="tab:red", linewidth=1.4)
            ax.grid(True, alpha=0.25)
            entry = summary[sensor_idx]
            if not entry.get("enabled", True):
                ax.text(0.02, 0.1, "outside L", transform=ax.transAxes, color="0.45", fontsize=9)
            for peak in entry.get("primary_list", []):
                phase = float(peak["phase"])
                if data["xlim"][0] <= phase <= data["phase_max_map"]:
                    ax.scatter(
                        [phase],
                        [peak["pressure"]],
                        color=PRIMARY_COLOR,
                        edgecolors="black",
                        linewidths=0.6,
                        s=52,
                        marker="o",
                        zorder=4,
                    )
            window = entry.get("secondary_window")
            if window:
                phase_min, phase_max = window
                for phase in (phase_min, phase_max):
                    if data["xlim"][0] <= phase <= data["phase_max_map"]:
                        label = "Secondary search window" if not window_legend_added else None
                        ax.axvline(
                            phase,
                            color=SECONDARY_WINDOW_COLOR,
                            linestyle="--",
                            linewidth=1.0,
                            alpha=0.7,
                            label=label,
                        )
                        window_legend_added = True
            secondary = entry.get("secondary")
            if secondary:
                phase = float(secondary["phase"])
                if data["xlim"][0] <= phase <= data["phase_max_map"]:
                    ax.scatter(
                        [phase],
                        [secondary["pressure"]],
                        color=SECONDARY_COLOR,
                        edgecolors="black",
                        linewidths=0.6,
                        s=46,
                        marker="D",
                        zorder=4,
                    )
            # legacy support if primary_list empty but single primary exists
            if not entry.get("primary_list"):
                peak = entry.get("primary")
                if peak:
                    phase = float(peak["phase"])
                    if data["xlim"][0] <= phase <= data["phase_max_map"]:
                        ax.scatter(
                            [phase],
                            [peak["pressure"]],
                            color=PRIMARY_COLOR,
                            edgecolors="black",
                            linewidths=0.6,
                            s=52,
                            marker="o",
                            zorder=4,
                        )
        ax.set_ylabel(key.upper())
        ax.set_xlim(data["xlim"])
        xticks = np.arange(data["xlim"][0], data["xlim"][1] + 1e-6, THETA_TICK_STEP)
        if xticks.size == 0:
            xticks = data["theta_marks"]
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
        seed_window = entry.get("secondary_seed_window")
        if seed_window:
            seed_label = "Seed bounds" if not window_legend_added else None
            for phase in seed_window:
                if data["xlim"][0] <= phase <= data["phase_max_map"]:
                    ax.axvline(
                        phase,
                        color=SEED_WINDOW_COLOR,
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.9,
                        label=seed_label,
                    )
                    window_legend_added = True
    axes[-1].set_xlabel("Phase [rad]")
    fig.legend(handles=legend_handles, loc="upper right", frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def figure_peak_gradient_finder(data):
    X_rev = data["X_rev"]
    Z = data["avg_matrix"]
    summary = data.get("peak_summary", [])
    phase_max = data.get("phase_max_map", float(X_rev[-1]) if X_rev.size else 0.0)
    fig, axes = plt.subplots(13, 1, figsize=(11, 1.7 * 13 + 1.5), sharex=True)
    legend_handles = [
        Line2D([], [], color="0.6", linewidth=1.4, label="Phase-averaged signal"),
        Line2D([], [], color=GRADIENT_TRACE_COLOR, linewidth=1.4, label="dP/dtheta (gradient)"),
        Line2D([], [], color=SECOND_GRADIENT_COLOR, linewidth=1.4, label="d²P/dtheta²"),
        Line2D([], [], marker="o", markersize=8, color="none", markerfacecolor=PRIMARY_COLOR, markeredgecolor="black", label="Primary peaks"),
        Line2D([], [], marker="D", markersize=8, color="none", markerfacecolor=SECONDARY_COLOR, markeredgecolor="black", label="Secondary peak"),
    ]
    window_legend_added = False

    def _sample_series(x_vals, series_vals, phase):
        mask_valid = np.isfinite(series_vals)
        if np.count_nonzero(mask_valid) < 2:
            return float("nan")
        x_valid = x_vals[mask_valid]
        y_valid = series_vals[mask_valid]
        if phase <= x_valid[0]:
            return float(y_valid[0])
        if phase >= x_valid[-1]:
            return float(y_valid[-1])
        return float(np.interp(phase, x_valid, y_valid))

    for i in range(13):
        sensor_idx = 12 - i
        key = PCB_LABELS[sensor_idx]
        trace = Z[sensor_idx, :] if sensor_idx < Z.shape[0] else np.full_like(X_rev, np.nan)
        ax = axes[i]
        if np.all(~np.isfinite(trace)):
            ax.text(0.02, 0.5, f"{key.upper()} missing", transform=ax.transAxes, color="0.4")
            ax.set_ylim(0, 1)
            ax.set_yticks([])
        else:
            ax.plot(X_rev, trace, color="0.6", linewidth=1.2)
            grad_trace = np.full_like(trace, np.nan, dtype=float)
            second_trace = np.full_like(trace, np.nan, dtype=float)
            mask = np.isfinite(trace)
            if np.count_nonzero(mask) >= 3:
                x_valid = X_rev[mask]
                grad_vals = np.gradient(trace[mask], x_valid)
                grad_trace[mask] = grad_vals
                if np.count_nonzero(mask) >= 4:
                    second_vals = np.gradient(grad_vals, x_valid)
                    second_trace[mask] = second_vals
            if np.any(np.isfinite(grad_trace)):
                ax.plot(X_rev, grad_trace, color=GRADIENT_TRACE_COLOR, linewidth=1.2)
            if np.any(np.isfinite(second_trace)):
                ax.plot(X_rev, second_trace, color=SECOND_GRADIENT_COLOR, linewidth=1.1)
            entry = summary[sensor_idx] if sensor_idx < len(summary) else {}
            if entry:
                if not entry.get("enabled", True):
                    ax.text(0.02, 0.1, "outside L", transform=ax.transAxes, color="0.45", fontsize=9)
                for peak in entry.get("primary_list", []):
                    phase = float(peak["phase"])
                    if data["xlim"][0] <= phase <= phase_max:
                        ax.scatter(
                            [phase],
                            [peak["pressure"]],
                            color=PRIMARY_COLOR,
                            edgecolors="black",
                            linewidths=0.6,
                            s=48,
                            marker="o",
                            zorder=4,
                        )
                window = entry.get("secondary_window")
                if window:
                    phase_min, phase_max_window = window
                    for phase in (phase_min, phase_max_window):
                        if data["xlim"][0] <= phase <= phase_max:
                            label = "Secondary search window" if not window_legend_added else None
                            ax.axvline(
                                phase,
                                color=SECONDARY_WINDOW_COLOR,
                                linestyle="--",
                                linewidth=1.0,
                                alpha=0.7,
                                label=label,
                            )
                            window_legend_added = True
                seed_window = entry.get("secondary_seed_window")
                if seed_window:
                    seed_label = "Seed bounds" if not window_legend_added else None
                    for phase in seed_window:
                        if data["xlim"][0] <= phase <= phase_max:
                            ax.axvline(
                                phase,
                                color=SEED_WINDOW_COLOR,
                                linestyle="--",
                                linewidth=1.0,
                                alpha=0.9,
                                label=seed_label,
                            )
                            window_legend_added = True
                secondary = entry.get("secondary")
                if secondary:
                    phase = float(secondary["phase"])
                    if data["xlim"][0] <= phase <= phase_max:
                        grad_val = _sample_series(X_rev, grad_trace, phase)
                        if not np.isfinite(grad_val):
                            grad_val = grad_trace[np.abs(X_rev - phase).argmin()]
                        ax.scatter(
                            [phase],
                            [grad_val],
                            color=SECONDARY_COLOR,
                            edgecolors="black",
                            linewidths=0.6,
                            s=44,
                            marker="D",
                            zorder=4,
                        )
                if not entry.get("primary_list"):
                    peak = entry.get("primary")
                    if peak:
                        phase = float(peak["phase"])
                        if data["xlim"][0] <= phase <= phase_max:
                            ax.scatter(
                                [phase],
                                [peak["pressure"]],
                                color=PRIMARY_COLOR,
                                edgecolors="black",
                                linewidths=0.6,
                                s=48,
                                marker="o",
                                zorder=4,
                            )
            else:
                ax.text(0.02, 0.1, "outside L", transform=ax.transAxes, color="0.45", fontsize=9)
            ax.grid(True, alpha=0.25)
        ax.set_ylabel(key.upper())
        ax.set_xlim(data["xlim"])
        xticks = np.arange(data["xlim"][0], data["xlim"][1] + 1e-6, THETA_TICK_STEP)
        if xticks.size == 0:
            xticks = data["theta_marks"]
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
    axes[-1].set_xlabel("Phase [rad]")
    fig.legend(handles=legend_handles, loc="upper right", frameon=True)
    fig.suptitle("Figure 3a: PGF (peak gradient finder)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def figure_discrete_map(data):
    X_rev = data["X_rev"]
    Z = data["avg_matrix"]
    Y_ax = data["Y_ax"]
    phase_max = data["phase_max_discrete"]
    mask = (X_rev >= 0.0) & (X_rev <= phase_max)
    if not np.any(mask):
        mask = X_rev >= 0.0
    phase_raw = X_rev[mask]
    Z_raw = Z[:, mask]
    if phase_raw.size == 0:
        phase_raw = np.array([0.0])
        Z_raw = np.ma.masked_all((Z.shape[0], 1))
    else:
        Z_raw = np.ma.masked_invalid(Z_raw)
    if phase_raw[0] > 0.0:
        phase_raw = np.insert(phase_raw, 0, 0.0)
        Z_start = Z_raw[:, 0][:, None]
        Z_raw = np.ma.hstack([Z_start, Z_raw])
    if phase_raw[-1] < phase_max:
        phase_raw = np.append(phase_raw, phase_max)
        Z_end = Z_raw[:, -1][:, None]
        Z_raw = np.ma.hstack([Z_raw, Z_end])
    fig, ax = plt.subplots(figsize=(10, 5.2))
    phase_grid, y_grid = np.meshgrid(phase_raw, Y_ax, indexing="xy")
    values = Z_raw.filled(np.nan)
    valid = np.isfinite(values)

    vmin = -1.0
    vmax = 4.5

    # Build an opaque interpolated background
    background = np.zeros_like(values, dtype=float)
    background[:] = np.nan
    if np.count_nonzero(valid) >= 6:
        xi = np.linspace(0.0, phase_max, 400)
        yi = np.linspace(Y_ax[0], Y_ax[-1], 400)
        Xi, Yi = np.meshgrid(xi, yi)
        pts = np.column_stack((phase_grid[valid], y_grid[valid]))
        vals = values[valid]
        Zi_cubic = griddata(pts, vals, (Xi, Yi), method="cubic")
        Zi_lin = griddata(pts, vals, (Xi, Yi), method="linear")
        Zi_near = griddata(pts, vals, (Xi, Yi), method="nearest")
        Zi = np.where(np.isfinite(Zi_cubic), Zi_cubic, np.where(np.isfinite(Zi_lin), Zi_lin, Zi_near))
        Zi = np.nan_to_num(Zi, nan=np.nanmean(vals) if vals.size else 0.0)
        ax.imshow(
            Zi,
            extent=(0.0, phase_max, Y_ax[0], Y_ax[-1]),
            origin="lower",
            cmap="plasma",
            alpha=0.8,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            zorder=0,
        )

    scatter = ax.scatter(
        phase_grid[valid],
        y_grid[valid],
        c=values[valid],
        cmap="plasma",
        vmin=vmin,
        vmax=vmax,
        s=36,
        edgecolors="k",
        linewidths=0.3,
        zorder=1,
    )
    ax.set_xlabel("Phase [rad]")
    ax.set_ylabel("Axial location [mm]")
    ax.set_xlim(0.0, phase_max)
    xticks = np.arange(0.0, phase_max + 1e-6, np.pi)
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
    ax.set_ylim(Y_ax[0] - 5, Y_ax[-1] + 5)
    ax.set_yticks(Y_ax)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    length_mm = data.get("combustor_length_mm")
    if length_mm is not None and np.isfinite(length_mm):
        ax.axhline(float(length_mm), color="red", linewidth=1.2, linestyle="--", alpha=0.9)

    # Overlay peak markers identical to Figures 3/5
    primary_points = []
    secondary_points = []
    summary = data["peak_summary"]
    for entry in summary:
        try:
            idx = int(entry["pcb"][-2:]) - 1
        except Exception:
            continue
        if idx < 0 or idx >= len(Y_ax):
            continue
        y_val = float(Y_ax[idx])
        for peak in entry.get("primary_list", []):
            if not peak:
                continue
            phase_val = float(peak["phase"])
            if not (0.0 <= phase_val <= phase_max + 1e-6):
                continue
            primary_points.append((phase_val, y_val))
        secondary = entry.get("secondary")
        if secondary:
            phase_val = float(secondary["phase"])
            if 0.0 <= phase_val <= phase_max + 1e-6:
                secondary_points.append((phase_val, y_val))
    if primary_points:
        xp, yp = zip(*primary_points)
        ax.scatter(xp, yp, color=PRIMARY_COLOR, marker="o", edgecolors="black", linewidths=0.6, s=54, label="Primary peaks")
    if secondary_points:
        xs, ys = zip(*secondary_points)
        ax.scatter(xs, ys, color=SECONDARY_COLOR, marker="D", edgecolors="black", linewidths=0.6, s=48, label="Secondary peaks")
    if primary_points or secondary_points:
        ax.legend(loc="upper right", frameon=True)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Pressure [bar]")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def _build_phase_map_grid(data, nx=400, ny=400):
    X_rev = data["X_rev"]
    Z = data["avg_matrix"]
    Y_ax = data["Y_ax"]
    phase_max = data["phase_max_map"]
    mask_phase = X_rev <= phase_max
    if not np.any(mask_phase):
        mask_phase = X_rev >= 0.0
    phase_samples = np.tile(X_rev[mask_phase], len(Y_ax))
    axial_samples = np.repeat(Y_ax, np.count_nonzero(mask_phase))
    values = Z[:, mask_phase].reshape(-1)
    finite_mask = np.isfinite(values)
    points = np.column_stack((phase_samples[finite_mask], axial_samples[finite_mask]))
    vals = values[finite_mask]
    xi = np.linspace(0.0, phase_max, max(50, int(nx)))
    yi = np.linspace(Y_ax[0], Y_ax[-1], max(50, int(ny)))
    Xi, Yi = np.meshgrid(xi, yi)
    if points.size == 0:
        Zi = np.zeros_like(Xi)
    else:
        try:
            Zi_cubic = griddata(points, vals, (Xi, Yi), method="cubic")
            Zi_lin = griddata(points, vals, (Xi, Yi), method="linear")
            Zi_near = griddata(points, vals, (Xi, Yi), method="nearest")
            Zi = np.where(np.isfinite(Zi_cubic), Zi_cubic, np.where(np.isfinite(Zi_lin), Zi_lin, Zi_near))
            Zi = np.nan_to_num(Zi, nan=np.nanmean(vals) if vals.size else 0.0)
        except Exception:
            Zi = np.zeros_like(Xi)
    return Xi, Yi, Zi, phase_max


def _figure_map_griddata(data, add_secondary_fit=True):
    X_rev = data["X_rev"]
    Y_ax = data["Y_ax"]
    Xi, Yi, Zi, phase_max = _build_phase_map_grid(data)
    theta_marks = np.arange(0.0, phase_max + 1e-6, np.pi)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    cf = ax.contourf(Xi, Yi, Zi, levels=50, cmap="plasma", vmin=-1.0, vmax=4.5)
    ax.contour(Xi, Yi, Zi, levels=20, colors="k", linewidths=0.4, alpha=0.4)
    ax.set_xlabel("Phase [rad]")
    ax.set_ylabel("Axial location [mm]")
    ax.set_xlim(0.0, phase_max)
    ax.set_xticks(theta_marks)
    ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
    ax.set_ylim(Y_ax[0] - 5, Y_ax[-1] + 5)
    ax.set_yticks(Y_ax)
    length_mm = data.get("combustor_length_mm")
    if length_mm is not None and np.isfinite(length_mm):
        ax.axhline(float(length_mm), color="red", linewidth=1.3, linestyle="--", alpha=0.9, label="Combustor length")
    summary = data["peak_summary"]
    pcb01_phase = determine_pcb01_phase(summary, 0.0, phase_max)
    points = collect_peak_points(summary, Y_ax, phase_max)
    fit_info = None
    if points:
        primary_points = [(p["phase"], p["y"]) for p in points if p["type"] == "primary"]
        secondary_points = [(p["phase"], p["y"]) for p in points if p["type"] == "secondary"]
        if primary_points:
            xp, yp = zip(*primary_points)
            ax.scatter(xp, yp, color=PRIMARY_COLOR, marker="o", edgecolors="black", linewidths=0.6, s=54, label="Primary peak")
        if secondary_points:
            xs, ys = zip(*secondary_points)
            ax.scatter(xs, ys, color=SECONDARY_COLOR, marker="D", edgecolors="black", linewidths=0.6, s=48, label="Secondary peak")
            if add_secondary_fit:
                fit_info = plot_secondary_regression(ax, secondary_points, pcb01_phase)
    if add_secondary_fit and fit_info and length_mm is not None and np.isfinite(length_mm):
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        annotate_beta_angle(ax, fit_info.get("slope"), length_mm, x_lim[0], x_lim[1], y_lim[0], y_lim[1])
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Pressure [bar]")
    cbar.set_ticks(np.arange(-1.0, 4.5 + 1e-6, 0.5))
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    return fig


def figure_ridge_detect(data):
    Xi, Yi, Zi, phase_max = _build_phase_map_grid(data)
    if Xi is None or np.size(Xi) == 0:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.text(0.5, 0.5, "Insufficient data for ridge detection", transform=ax.transAxes, ha="center", va="center")
        fig.suptitle("Figure 5a: ride detect (insufficient data)", fontsize=13)
        fig.tight_layout()
        return fig
    phi_coords = Xi[0, :]
    y_coords = Yi[:, 0]
    finite_vals = Zi[np.isfinite(Zi)]
    if finite_vals.size == 0:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.text(0.5, 0.5, "Unable to process map intensities for ridge detection", transform=ax.transAxes, ha="center", va="center")
        fig.suptitle("Figure 5a: ride detect (no finite data)", fontsize=13)
        fig.tight_layout()
        return fig
    map_median = float(np.median(finite_vals))
    map_p90 = float(np.percentile(finite_vals, 90))
    intensity_threshold = map_median + 0.15 * max(map_p90 - map_median, 1e-6)
    phase_min = max(data["xlim"][0], 0.5 * np.pi)
    phase_max_window = min(phase_max, 3.2 * np.pi)
    pcb01_phase = determine_pcb01_phase(data["peak_summary"], 0.0, phase_max)
    anchor_phase = float(np.clip(pcb01_phase, phase_min, phase_max_window))
    focus_left = max(phase_min, anchor_phase - 0.6 * np.pi)
    focus_right = min(phase_max_window, anchor_phase + 0.6 * np.pi)
    phase_mask = (phi_coords >= focus_left) & (phi_coords <= focus_right)
    ridge_points = []
    prev_phase = float("nan")
    phase_sigma = 0.18 * np.pi
    length_mm = data.get("combustor_length_mm")
    y_max_allowed = length_mm if length_mm and np.isfinite(length_mm) else y_coords[-1]
    start_row_idx = next(
        (idx for idx, y_val in enumerate(reversed(y_coords)) if y_val <= y_max_allowed + 2.0),
        None,
    )
    if start_row_idx is None:
        start_row_idx = 0
    start_row = len(y_coords) - 1 - start_row_idx
    for row_idx in range(start_row, -1, -1):
        y_val = float(y_coords[row_idx])
        if y_val > y_max_allowed + 5.0:
            continue
        row_values = Zi[row_idx, :]
        if not np.any(np.isfinite(row_values)):
            continue
        row_focus = row_values[phase_mask]
        phase_focus = phi_coords[phase_mask]
        if row_focus.size < 4 or not np.any(np.isfinite(row_focus)):
            continue
        finite_mask = np.isfinite(row_focus)
        if np.count_nonzero(finite_mask) < 3:
            continue
        if not np.all(finite_mask):
            valid_phase = phase_focus[finite_mask]
            valid_vals = row_focus[finite_mask]
            row_focus = np.interp(phase_focus, valid_phase, valid_vals)
        window_len = max(5, int(round(0.08 * row_focus.size)))
        if window_len % 2 == 0:
            window_len += 1
        if window_len >= row_focus.size:
            window_len = row_focus.size - (1 - row_focus.size % 2)
        if window_len >= 5:
            poly_order = 3 if window_len > 5 else 2
            try:
                smooth_row = savgol_filter(row_focus, window_length=window_len, polyorder=min(poly_order, window_len - 1), mode="interp")
            except ValueError:
                smooth_row = row_focus
        else:
            smooth_row = row_focus
        if smooth_row.size == 0 or not np.any(np.isfinite(smooth_row)):
            continue
        smooth_row = np.asarray(smooth_row, dtype=float)
        span = float(np.nanmax(smooth_row) - np.nanmin(smooth_row))
        prominence = max(1e-6, 0.08 * span)
        peaks, properties = find_peaks(smooth_row, prominence=prominence)
        if peaks.size == 0:
            candidate_indices = np.array([int(np.nanargmax(smooth_row))])
            candidate_scores = smooth_row[candidate_indices]
        else:
            candidate_indices = peaks
            candidate_scores = smooth_row[candidate_indices]
        candidate_phases = phase_focus[candidate_indices]
        if np.isnan(prev_phase):
            weights = candidate_scores
        else:
            proximity = np.exp(-0.5 * ((candidate_phases - prev_phase) / max(phase_sigma, 1e-6)) ** 2)
            weights = candidate_scores * proximity
        best_idx = int(np.argmax(weights))
        peak_score = float(candidate_scores[best_idx])
        if peak_score < intensity_threshold:
            continue
        phase_val = float(candidate_phases[best_idx])
        ridge_points.append((phase_val, y_val, float(weights[best_idx])))
        prev_phase = 0.6 * prev_phase + 0.4 * phase_val if np.isfinite(prev_phase) else phase_val
    fig, ax = plt.subplots(figsize=(12, 5.5))
    cf = ax.contourf(Xi, Yi, Zi, levels=50, cmap="plasma")
    ax.contour(Xi, Yi, Zi, levels=20, colors="k", linewidths=0.4, alpha=0.4)
    ax.set_xlabel("Phase [rad]")
    ax.set_ylabel("Axial location [mm]")
    ax.set_xlim(phase_min, phase_max_window)
    xticks = np.arange(0.0, phase_max + 1e-6, np.pi)
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
    ax.set_ylim(y_coords[0], y_coords[-1])
    length_mm = data.get("combustor_length_mm")
    if length_mm is not None and np.isfinite(length_mm):
        ax.axhline(float(length_mm), color="red", linewidth=1.3, linestyle="--", alpha=0.9, label="Combustor length")
    ridge_handle = None
    fit_info = None
    if ridge_points:
        ridge_points_sorted = sorted(ridge_points, key=lambda p: p[1])
        xs = [p[0] for p in ridge_points_sorted]
        ys = [p[1] for p in ridge_points_sorted]
        weights = [p[2] for p in ridge_points_sorted]
        ax.scatter(xs, ys, color=RIDGE_COLOR, edgecolors="black", linewidths=0.5, s=30, label="Ridge candidates")
        if len(ridge_points_sorted) >= 3:
            weights_arr = np.asarray(weights, dtype=float)
            if np.nanmax(weights_arr) > 0:
                weights_arr = np.clip(weights_arr / np.nanmax(weights_arr), 0.1, 1.0)
            else:
                weights_arr = np.full_like(weights_arr, 0.5)
            y_vals = np.array(ys, dtype=float)
            x_vals = np.array(xs, dtype=float)
            try:
                coeffs = np.polyfit(y_vals, x_vals, 1, w=weights_arr)
                slope, intercept = float(coeffs[0]), float(coeffs[1])
                fit_info = {"slope": slope, "intercept": intercept}
                y_line = np.linspace(min(y_vals), max(y_vals), 200)
                x_line = slope * y_line + intercept
                ridge_handle = ax.plot(x_line, y_line, color=RIDGE_COLOR, linewidth=2.2, label="Ridge fit")[0]
            except Exception:
                fit_info = None
    if length_mm is not None and fit_info:
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        annotate_beta_angle(ax, fit_info.get("slope"), length_mm, x_lim[0], x_lim[1], y_lim[0], y_lim[1])
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Pressure [bar]")
    legend_handles = []
    if length_mm is not None:
        legend_handles.append(Line2D([], [], color="red", linestyle="--", linewidth=1.3, label="Combustor length"))
    if ridge_handle:
        legend_handles.append(ridge_handle)
    if ridge_points:
        legend_handles.append(Line2D([], [], marker="o", color="none", markerfacecolor=RIDGE_COLOR, markeredgecolor="black", label="Ridge candidates"))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", frameon=True)
    ax.set_title("Figure 5a: ride detect (map ridge inference)")
    fig.tight_layout()
    return fig


def _setup_phase_map_axes(
    data,
    fig=None,
    ax=None,
    phase_ref_rad=None,
    x_min_override=None,
    x_max_override=None,
    pressure_vmin=-1.0,
    pressure_vmax=4.5,
    pressure_extend="neither",
    pressure_gamma=1.0,
    x_label="Phase [rad]",
    fixed_pcb01_phase=None,
    use_plain_pi_labels=False,
):
    X_rev = data["X_rev"]
    Z = data["avg_matrix"]
    Y_ax = data["Y_ax"]
    phase_max = data["phase_max_map"]
    phase_cap = float(phase_max)
    if x_max_override is not None and np.isfinite(float(x_max_override)):
        phase_cap = max(phase_cap, float(x_max_override))
    if X_rev.size:
        phase_cap = min(phase_cap, float(np.nanmax(X_rev)))

    mask_phase = X_rev <= phase_cap
    if not np.any(mask_phase):
        mask_phase = X_rev >= 0.0
    X_phase = X_rev[mask_phase]
    if X_phase.size < 4:
        X_phase = np.linspace(0.0, phase_max, max(4, len(X_phase)))

    Z_subset = Z[:, mask_phase]
    finite_mask = np.isfinite(Z_subset)
    if np.count_nonzero(finite_mask) < 8:
        return None

    for row_idx in range(Z_subset.shape[0]):
        row = Z_subset[row_idx, :]
        if np.all(~np.isfinite(row)):
            Z_subset[row_idx, :] = np.nanmean(Z_subset)
        else:
            nans = ~np.isfinite(row)
            if np.any(nans):
                row_interp = np.interp(X_phase[nans], X_phase[~nans], row[~nans])
                row[nans] = row_interp
            Z_subset[row_idx, :] = row

    spline = RectBivariateSpline(Y_ax, X_phase, Z_subset, kx=3, ky=1, s=0.0)
    x_min = float(x_min_override) if x_min_override is not None else 0.5 * np.pi
    x_max = float(x_max_override) if x_max_override is not None else min(6.0 * np.pi, phase_cap)
    if not np.isfinite(x_max) or x_max <= x_min:
        x_max = max(x_min + 0.5 * np.pi, float(np.nanmax(X_phase) if X_phase.size else (x_min + 0.5 * np.pi)))
    xi = np.linspace(x_min, x_max, 500)
    yi = np.linspace(5.0, 125.0, 400)
    Zi = spline(yi, xi)

    axial_range = 125.0 - 5.0
    phase_range_mm = (x_max - x_min) * COMBUSTOR_RADIUS_MM
    aspect_ratio = phase_range_mm / axial_range
    height_in = 5.5
    width_in = height_in * aspect_ratio

    owns_axes = fig is None or ax is None
    if owns_axes:
        fig, ax = plt.subplots(figsize=(width_in, height_in))
    else:
        ax.clear()
        fig.set_constrained_layout(False)
    levels = np.linspace(float(pressure_vmin), float(pressure_vmax), 50)
    gamma_val = float(pressure_gamma) if np.isfinite(float(pressure_gamma)) else 1.0
    gamma_val = max(0.2, min(3.0, gamma_val))
    norm = None
    if not np.isclose(gamma_val, 1.0):
        norm = mcolors.PowerNorm(
            gamma=gamma_val,
            vmin=float(pressure_vmin),
            vmax=float(pressure_vmax),
        )
    cf = ax.contourf(
        xi,
        yi,
        Zi,
        levels=levels,
        cmap="plasma",
        vmin=float(pressure_vmin) if norm is None else None,
        vmax=float(pressure_vmax) if norm is None else None,
        norm=norm,
        extend=str(pressure_extend),
    )
    ax.contour(xi, yi, Zi, levels=20, colors="k", linewidths=0.4, alpha=0.4)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Axial location [mm]")
    ax.set_xlim(x_min, x_max)
    step = 0.5 * np.pi
    t0 = np.ceil((x_min - 1e-12) / step) * step
    t1 = np.floor((x_max + 1e-12) / step) * step
    xticks = np.arange(t0, t1 + 0.25 * step, step)
    if xticks.size < 3:
        xticks = np.array([x_min, 0.5 * (x_min + x_max), x_max], dtype=float)
    ax.set_xticks(xticks.tolist())
    if phase_ref_rad is None:
        ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter_plain if use_plain_pi_labels else pi_formatter))
    else:
        base_formatter = pi_formatter_plain if use_plain_pi_labels else pi_formatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: base_formatter(x - float(phase_ref_rad), pos)))
    ax.set_ylim(5.0, 125.0)
    ax.set_yticks(np.arange(5, 125 + 1e-6, 10.0))
    ax.tick_params(labelsize=10)

    length_mm = data.get("combustor_length_mm")
    if length_mm is not None and np.isfinite(length_mm):
        ax.axhline(float(length_mm), color="red", linewidth=1.3, linestyle="--", alpha=0.9, label="Combustor length")
    summary = data["peak_summary"]
    pcb01_phase = determine_pcb01_phase(summary, x_min, x_max)
    if fixed_pcb01_phase is not None and np.isfinite(float(fixed_pcb01_phase)):
        pcb01_phase = float(np.clip(float(fixed_pcb01_phase), x_min, x_max))
    ax.axvline(
        pcb01_phase,
        color="black",
        linewidth=1.1,
        linestyle="--",
        alpha=0.85,
        label="PCB01 ref",
    )
    return {
        "fig": fig,
        "ax": ax,
        "cf": cf,
        "x_limits": (x_min, x_max),
        "y_limits": (5.0, 125.0),
        "pcb01_phase": pcb01_phase,
        "length_mm": length_mm,
        "spline": spline,
        "surface": (xi, yi, Zi),
    }


def figure_pressure_map(data):
    return _pressure_map_render(data)


def figure_pressure_map_journal(data):
    height_in = 3.25
    width_in = 8.5
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    return _pressure_map_render(
        data,
        fig=fig,
        ax=ax,
        legend_loc="upper right",
        legend_fontsize=10,
        ann_fontsize=10,
        beta_fontsize=10,
        phase_ref_rad=2.0 * np.pi,
        x_min_override=0.5 * np.pi,
        x_max_override=4.5 * np.pi,
        cbar_vmin=-1.0,
        cbar_vmax=4.5,
        cbar_extend="both",
        cbar_gamma=0.8,
        cbar_ticks=[0.0, 1.5, 3.0, 4.5],
        x_label="Phase [rad]",
        round_annotation_int=True,
        oblique_fit_y_min=15.0,
        show_oblique_seed_marker=False,
        use_plain_pi_labels=True,
        alpha_label_x_rel=0.0,
        beta_label_x_rel=-1.0 * np.pi,
    )


def figure_primary_polynomial_map(data):
    summary = data["peak_summary"]
    Y_ax = data["Y_ax"]
    phase_max = data["phase_max_map"]
    length_mm = data.get("combustor_length_mm")
    max_sensor_idx = data.get("polyfit_last_idx", _third_last_sensor_idx(length_mm, Y_ax))
    fig = plt.figure(figsize=(13.5, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.25, bottom=0.18, top=0.9)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_profile = fig.add_subplot(gs[0, 1])
    base = _setup_phase_map_axes(data, fig=fig, ax=ax_map)
    poly_fit = None
    hermite_fit = None

    if base is None:
        plt.close(fig)
        return _figure_map_griddata(data, add_secondary_fit=False)

    cf = base["cf"]
    x_limits = base["x_limits"]
    points = collect_peak_points(summary, Y_ax, phase_max)
    if points:
        primary_points = [(p["phase"], p["y"]) for p in points if p["type"] == "primary"]
        secondary_points = [(p["phase"], p["y"]) for p in points if p["type"] == "secondary"]
        if primary_points:
            xp, yp = zip(*primary_points)
            ax_map.scatter(xp, yp, color=PRIMARY_COLOR, marker="o", edgecolors="black", linewidths=0.6, s=54, label="Primary peak")
        if secondary_points:
            xs, ys = zip(*secondary_points)
            ax_map.scatter(xs, ys, color=SECONDARY_COLOR, marker="D", edgecolors="black", linewidths=0.6, s=48, label="Secondary peak")
    poly_fit = plot_primary_polynomial(ax_map, summary, Y_ax, x_limits, max_sensor_idx)
    hermite_fit = None
    if data.get("use_hermite_fit"):
        hermite_fit = plot_primary_hermite(ax_map, summary, Y_ax, x_limits, max_sensor_idx)

    fallback_idx = max_sensor_idx if max_sensor_idx is not None else _third_last_sensor_idx(length_mm, Y_ax)
    pcb_idx = poly_fit["limit_idx"] if poly_fit else (hermite_fit["limit_idx"] if hermite_fit else fallback_idx)
    ax_map.legend(loc="upper right", frameon=True)

    ax_profile.set_xlabel("Axial location [mm]")
    ax_profile.set_ylabel("Pressure [bar]")
    ax_profile.set_xlim(ax_map.get_ylim())
    ax_profile.set_xticks(ax_map.get_yticks())
    ax_profile_secondary = ax_profile.twinx()
    ax_profile_secondary.set_ylabel("dP/dy [bar/mm]")
    ax_profile_secondary.grid(False)
    ax_profile.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)

    metrics = data.get("metrics", {})
    spline = base.get("spline")
    profile_drawn = False
    h_line = None
    yR_line = None
    first_min_marker = None
    second_deriv_marker = None
    second_min_marker = None
    ransac_line = None
    x_bounds = []
    annotation_entries = []
    smooth_frac = float(data.get("dpdy_smooth_frac", DEFAULT_DPDY_SMOOTH_FRAC))
    dpdy_max_rank = int(data.get("dpdy_max_rank", 0))
    if poly_fit and spline is not None:
        pressures = spline.ev(poly_fit["y_samples"], poly_fit["phase_samples"])
        y_vals = np.array(poly_fit["y_samples"], dtype=float)
        p_vals = np.array(pressures, dtype=float)
        derivative_line = None
        if y_vals.size > 1:
            dy = np.gradient(y_vals)
            dp = np.gradient(p_vals)
            derivative = np.divide(dp, dy, out=np.zeros_like(dp), where=dy != 0)
        else:
            derivative = np.zeros_like(p_vals)
        pressure_line = ax_profile.scatter(
            y_vals,
            p_vals,
            color="#7b1fa2",
            s=10,
            label="Polynomial pressure samples",
        )
        derivative_y = y_vals
        derivative_vals = derivative
        derivative_label = "dP/dy (gradient)"
        derivative_spline = None
        if y_vals.size >= 2:
            try:
                unique_y, unique_idx = np.unique(y_vals, return_index=True)
                order = np.argsort(unique_y)
                unique_y = unique_y[order]
                unique_p = p_vals[unique_idx][order]
                if unique_y.size >= 2:
                    cs = CubicSpline(unique_y, unique_p, bc_type="natural")
                    derivative_spline = cs.derivative()
                    derivative_y = np.linspace(unique_y.min(), unique_y.max(), 400)
                    derivative_vals = derivative_spline(derivative_y)
                    derivative_label = "dP/dy (cubic spline)"
            except (np.linalg.LinAlgError, ValueError):
                derivative_y = y_vals
                derivative_vals = derivative
        plot_derivative_vals = _smooth_series(derivative_vals, frac=smooth_frac)
        derivative_line, = ax_profile_secondary.plot(
            derivative_y,
            plot_derivative_vals,
            color=PRIMARY_HERMITE_COLOR,
            linewidth=1.3,
            linestyle="-",
            label=derivative_label,
        )
        first_min_x = None
        detection_vals = plot_derivative_vals

        first_min = _first_derivative_minimum(derivative_y, detection_vals)
        if first_min:
            min_x = first_min[0]
            min_val = float(np.interp(min_x, derivative_y, plot_derivative_vals))
            first_min_x = min_x
            first_min_marker = ax_profile_secondary.scatter(
                [min_x],
                [min_val],
                color=PRIMARY_HERMITE_COLOR,
                edgecolors="black",
                linewidths=0.7,
                s=45,
                zorder=7,
                label="dP/dy first min",
            )
            x_bounds.append(min_x)
            annotation_entries.append(f"dP/dy min: {min_x:.1f} mm")
        second_min = _first_derivative_maximum(derivative_y, detection_vals, first_min_x, rank=dpdy_max_rank)
        if second_min:
            second_min_marker = ax_profile_secondary.scatter(
                [second_min[0]],
                [float(np.interp(second_min[0], derivative_y, plot_derivative_vals))],
                color="#d4af37",
                edgecolors="black",
                linewidths=0.7,
                s=45,
                zorder=7,
                label="dP/dy max",
            )
            x_bounds.append(second_min[0])
            annotation_entries.append(f"dP/dy max: {second_min[0]:.1f} mm")

        steepest = _max_second_derivative(derivative_y, detection_vals, first_min_x)
        if data.get("use_second_deriv_max"):
            sx, _, s2 = steepest
            sy = float(np.interp(sx, derivative_y, plot_derivative_vals))
            second_deriv_marker = ax_profile_secondary.scatter(
                [sx],
                [sy],
                color=SECOND_DERIV_COLOR,
                edgecolors="black",
                linewidths=0.7,
                s=45,
                zorder=7,
                label="d²P/dy² max",
            )
            x_bounds.append(sx)
            annotation_entries.append(f"d²P/dy² max: {sx:.1f} mm")
        p_lower = np.min(p_vals)
        p_upper = np.max(p_vals)
        p_margin = 0.1 * (p_upper - p_lower if p_upper > p_lower else 1.0)
        ax_profile.set_ylim(p_lower - p_margin, p_upper + p_margin)
        d_lower, d_upper = np.min(derivative_vals), np.max(derivative_vals)
        d_margin = 0.1 * (d_upper - d_lower if d_upper > d_lower else 1.0)
        ax_profile_secondary.set_ylim(d_lower - d_margin, d_upper + d_margin)
        x_bounds.extend([y_vals.min(), y_vals.max()])
        if "H_axial" in metrics and np.isfinite(metrics["H_axial"]):
            h_line = ax_profile.axvline(
                metrics["H_axial"],
                color=ANNOTATION_DARK_ORANGE,
                linestyle="--",
                linewidth=1.1,
                label="H axial",
            )
            x_bounds.append(metrics["H_axial"])
            annotation_entries.append(f"H axial: {metrics['H_axial']:.1f} mm")
        if "y_R_axial" in metrics and np.isfinite(metrics["y_R_axial"]):
            y_r_val = float(metrics["y_R_axial"])
            if y_r_val >= 0.0:
                yR_line = ax_profile.axvline(
                    y_r_val,
                    color=ANNOTATION_DARK_BLUE,
                    linestyle="--",
                    linewidth=1.1,
                    label="y_R axial",
                )
                x_bounds.append(y_r_val)
            annotation_entries.append(f"y_R axial: {y_r_val:.1f} mm")
        if x_bounds:
            x_min = min(x_bounds)
            x_max = max(x_bounds)
            margin = 0.03 * (x_max - x_min if x_max > x_min else 1.0)
            ax_profile.set_xlim(x_min - margin, x_max + margin)
        profile_drawn = True
    else:
        ax_profile.text(
            0.5,
            0.5,
            "Polynomial fit unavailable",
            transform=ax_profile.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="0.4",
        )
    if profile_drawn:
        legend_handles = [pressure_line]
        if h_line:
            legend_handles.append(h_line)
        if yR_line:
            legend_handles.append(yR_line)
        if ransac_line:
            legend_handles.append(ransac_line)
        ax_profile.legend(handles=legend_handles, loc="upper left", frameon=True)
        secondary_handles = [derivative_line]
        if first_min_marker:
            secondary_handles.append(first_min_marker)
        if second_min_marker:
            secondary_handles.append(second_min_marker)
        if second_deriv_marker:
            secondary_handles.append(second_deriv_marker)
        ax_profile_secondary.legend(handles=secondary_handles, loc="upper right", frameon=True)
        if annotation_entries:
            ax_profile_secondary.text(
                0.5,
                0.05,
                "\n".join(annotation_entries),
                transform=ax_profile_secondary.transAxes,
                ha="center",
                va="bottom",
                fontsize=10,
                color="0.2",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.5", linewidth=0.8),
            )

    fig.tight_layout()
    return fig


def figure_speed_primary(data):
    X_rev = data.get("X_rev", np.array([]))
    Z = data.get("avg_matrix", np.empty((0, 0)))
    f_det = float(data.get("f_det", 0.0) or 0.0)
    waves_per_rev = int(data.get("waves_per_rev", 1) or 1)
    f_rev = f_det / float(max(waves_per_rev, 1))
    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    speed_pcbs = list(range(3, 8))
    times_ms = []
    distances_mm = []
    labels = []

    phase_min = 2.0 * np.pi
    phase_max = 2.5 * np.pi
    if X_rev.size:
        phase_mask = (X_rev >= phase_min) & (X_rev <= phase_max)
    else:
        phase_mask = np.array([], dtype=bool)
    if not np.any(phase_mask):
        phase_mask = np.isfinite(X_rev)
    phase_axis = X_rev[phase_mask] if X_rev.size else np.array([])

    def _primary_peak_phase(row: np.ndarray) -> float:
        if row.size != X_rev.size or phase_axis.size == 0:
            return float("nan")
        y = np.asarray(row, dtype=float)[phase_mask]
        if y.size == 0:
            return float("nan")
        finite = np.isfinite(y)
        if np.count_nonzero(finite) < 3:
            return float("nan")
        if not np.all(finite):
            y = np.interp(np.arange(len(y)), np.flatnonzero(finite), y[finite])
        if len(y) >= 7:
            win = min(31, len(y) if len(y) % 2 == 1 else len(y) - 1)
            if win < 5:
                win = 5 if len(y) >= 5 else (len(y) | 1)
            y = savgol_filter(y, window_length=max(3, win), polyorder=3)
        idx_peak = int(np.nanargmax(y))
        return float(phase_axis[idx_peak])

    if f_rev > 0.0 and np.isfinite(f_rev):
        for idx in speed_pcbs:
            row_idx = idx - 1
            if row_idx < 0 or row_idx >= Z.shape[0]:
                continue
            phase_val = _primary_peak_phase(Z[row_idx, :])
            if not np.isfinite(phase_val):
                continue
            time_ms = (phase_val / (2.0 * np.pi * f_rev)) * 1e3
            dist_mm = 25.0 + 10.0 * (idx - 3)
            times_ms.append(float(time_ms))
            distances_mm.append(float(dist_mm))
            labels.append(f"PCB{idx:02d}")

    if times_ms:
        if "PCB03" in labels:
            t0 = times_ms[labels.index("PCB03")]
        else:
            t0 = float(np.nanmin(times_ms))
        times_arr = np.array([t - t0 for t in times_ms], dtype=float)
        dist_arr = np.array(distances_mm, dtype=float)
        order = np.argsort(dist_arr)
        times_arr = times_arr[order]
        dist_arr = dist_arr[order]
        labels = [labels[i] for i in order.tolist()]

        ax.scatter(times_arr, dist_arr, s=60, color=RIDGE_COLOR, edgecolors="black", linewidths=0.6, zorder=3)
        for t, y, label in zip(times_arr, dist_arr, labels):
            ax.text(t, y + 1.0, label, fontsize=9, ha="center", va="bottom")

        if len(times_arr) >= 2 and np.ptp(times_arr) > 1e-9:
            coeff = np.polyfit(times_arr, dist_arr, 1)
            slope = float(coeff[0])
            intercept = float(coeff[1])
            t_line = np.linspace(times_arr.min(), times_arr.max(), 200)
            y_line = slope * t_line + intercept
            ax.plot(t_line, y_line, color="black", linewidth=1.2, linestyle="--", zorder=2)
            ax.text(
                0.02,
                0.98,
                f"Speed ~ {slope:.1f} m/s",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="0.4"),
            )

        x_pad = 0.05 * (times_arr.max() - times_arr.min()) if times_arr.size > 1 else 0.05
        if not np.isfinite(x_pad) or x_pad <= 0.0:
            x_pad = 0.05
        ax.set_xlim(times_arr.min() - x_pad, times_arr.max() + x_pad)
        ax.set_ylim(20.0, 80.0)
    else:
        ax.text(
            0.5,
            0.5,
            "Not enough primary peak data for PCB03-08",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="0.3",
        )

    ax.set_xlabel("Time from PCB03 peak [ms]")
    ax.set_ylabel("Axial distance [mm]")
    ax.set_title("Figure 7: Primary Peak Travel Time (PCB03-08)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def collect_peak_points(summary, Y_ax, phase_max):
    points = []
    for entry in summary:
        try:
            idx = int(entry["pcb"][-2:]) - 1
        except Exception:
            continue
        if idx < 0 or idx >= len(Y_ax):
            continue
        y_val = float(Y_ax[idx])
        primary_peaks = entry.get("primary_list") or ([entry.get("primary")] if entry.get("primary") else [])
        for peak in primary_peaks:
            if not peak:
                continue
            phase_val = float(peak["phase"])
            if not (0.0 <= phase_val <= phase_max + 1e-6):
                continue
            points.append(
                {
                    "phase": phase_val,
                    "y": y_val,
                    "pressure": peak["pressure"],
                    "type": "primary",
                }
            )
        secondary = entry.get("secondary")
        if secondary:
            phase_val = float(secondary["phase"])
            if 0.0 <= phase_val <= phase_max + 1e-6:
                points.append(
                    {
                        "phase": phase_val,
                        "y": y_val,
                        "pressure": secondary["pressure"],
                        "type": "secondary",
                    }
                )
    return points


def _pcb_label_from_idx(idx):
    if 0 <= idx < len(PCB_LABELS):
        return PCB_LABELS[idx].upper()
    return f"PCB{idx + 1:02d}"


def _max_sensor_idx_within_length(length_mm, Y_ax):
    if not len(Y_ax):
        return -1
    if length_mm is None or not np.isfinite(length_mm):
        return len(Y_ax) - 1
    valid = [i for i, y in enumerate(Y_ax) if y <= float(length_mm) + 1e-6]
    return valid[-1] if valid else 0


def _resolve_polyfit_limit_idx(length_mm, Y_ax, last_pcb_setting):
    base_max = _max_sensor_idx_within_length(length_mm, Y_ax)
    if base_max < 0:
        return -1
    if last_pcb_setting is None:
        return max(base_max - 2, 0)
    idx = None
    try:
        if isinstance(last_pcb_setting, str):
            digits = "".join(ch for ch in last_pcb_setting if ch.isdigit())
            if digits:
                idx = int(digits) - 1
        else:
            idx = int(last_pcb_setting) - 1
    except Exception:
        idx = None
    if idx is None:
        return max(base_max - 2, 0)
    idx = max(0, min(idx, len(Y_ax) - 1))
    idx = min(idx, base_max)
    return idx


def _third_last_sensor_idx(length_mm, Y_ax):
    max_idx = _max_sensor_idx_within_length(length_mm, Y_ax)
    if max_idx < 0:
        return -1
    return max(max_idx - 2, 0)


def _determine_outlet_index(enabled_indices, Y_ax, length_mm):
    if not enabled_indices:
        return None
    if length_mm is None or not np.isfinite(length_mm):
        return enabled_indices[-1]
    within_length = [idx for idx in enabled_indices if float(Y_ax[idx]) <= float(length_mm) + 1e-6]
    if within_length:
        return within_length[-1]
    return enabled_indices[-1]


def _determine_secondary_seed_index(enabled_indices, outlet_idx, offset=2):
    if not enabled_indices or outlet_idx is None:
        return None
    if outlet_idx not in enabled_indices:
        outlet_idx = enabled_indices[-1]
    outlet_pos = enabled_indices.index(outlet_idx)
    seed_pos = outlet_pos - max(offset, 0)
    if seed_pos < 0:
        seed_pos = 0
    return enabled_indices[seed_pos]


def _collect_primary_fit_points(summary, Y_ax, max_sensor_idx, x_limits):
    if not len(Y_ax):
        return None
    idx_limit = min(max(max_sensor_idx, 0), len(Y_ax) - 1)
    pairs = []
    for idx in range(idx_limit + 1):
        entry = summary[idx]
        primary = entry.get("primary")
        if not primary:
            continue
        phase_val = float(primary["phase"])
        if not (min(x_limits) <= phase_val <= max(x_limits)):
            continue
        pairs.append((float(Y_ax[idx]), phase_val))
    if len(pairs) < 2:
        return None
    pairs.sort(key=lambda t: t[0])
    ys = np.array([p[0] for p in pairs], dtype=float)
    phases = np.array([p[1] for p in pairs], dtype=float)
    return {
        "ys": ys,
        "phases": phases,
        "limit_idx": idx_limit,
        "points": pairs,
    }


def fit_primary_polynomial(summary, Y_ax, x_limits, max_sensor_idx, degree=5):
    data = _collect_primary_fit_points(summary, Y_ax, max_sensor_idx, x_limits)
    if not data:
        return None
    ys = data["ys"]
    phases = data["phases"]
    deg = min(degree, len(ys) - 1)
    coeffs = np.polyfit(ys, phases, deg=deg)
    y_samples = np.linspace(ys.min(), ys.max(), 400)
    phase_samples = np.polyval(coeffs, y_samples)
    mask = (phase_samples >= min(x_limits)) & (phase_samples <= max(x_limits))
    if not np.any(mask):
        return None
    fit = dict(data)
    fit.update(
        {
            "coeffs": coeffs,
            "y_samples": y_samples[mask],
            "phase_samples": phase_samples[mask],
            "degree": deg,
            "fit_points": data["points"],
        }
    )
    return fit


def fit_primary_hermite(summary, Y_ax, x_limits, max_sensor_idx):
    data = _collect_primary_fit_points(summary, Y_ax, max_sensor_idx, x_limits)
    if not data:
        return None
    ys = data["ys"]
    phases = data["phases"]
    if len(ys) < 2:
        return None
    dy = np.gradient(phases, ys)
    try:
        spline = CubicHermiteSpline(ys, phases, dy)
    except Exception:
        return None
    sample_y = np.linspace(ys.min(), ys.max(), 400)
    sample_phase = spline(sample_y)
    mask = (sample_phase >= min(x_limits)) & (sample_phase <= max(x_limits))
    if not np.any(mask):
        return None
    return {
        "y_samples": sample_y[mask],
        "phase_samples": sample_phase[mask],
        "points": data["points"],
        "limit_idx": data["limit_idx"],
    }


def plot_primary_polynomial(ax, summary, Y_ax, x_limits, max_sensor_idx):
    if not len(Y_ax) or max_sensor_idx < 0:
        return None
    fit = fit_primary_polynomial(summary, Y_ax, x_limits, max_sensor_idx)
    if not fit:
        return None
    pcb_label = _pcb_label_from_idx(fit["limit_idx"])
    ax.plot(
        fit["phase_samples"],
        fit["y_samples"],
        color=PRIMARY_POLY_COLOR,
        linewidth=2.2,
        linestyle="-",
        label=f"Primary polynomial fit (PCB01–{pcb_label})",
    )
    return fit


def plot_primary_hermite(ax, summary, Y_ax, x_limits, max_sensor_idx):
    if not len(Y_ax) or max_sensor_idx < 0:
        return None
    fit = fit_primary_hermite(summary, Y_ax, x_limits, max_sensor_idx)
    if not fit:
        return None
    pcb_label = _pcb_label_from_idx(fit["limit_idx"])
    ax.plot(
        fit["phase_samples"],
        fit["y_samples"],
        color=PRIMARY_HERMITE_COLOR,
        linewidth=2.0,
        linestyle="--",
        label=f"Hermite spline fit (PCB01–{pcb_label})",
    )
    return fit


def _first_derivative_minimum(xs, ys):
    """Return the first local minimum (left-most) in the derivative profile."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size == 0:
        return None
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    if ys.size >= 3:
        for idx in range(1, ys.size - 1):
            if ys[idx] <= ys[idx - 1] and ys[idx] < ys[idx + 1]:
                return float(xs[idx]), float(ys[idx])
    min_idx = int(np.argmin(ys))
    return float(xs[min_idx]), float(ys[min_idx])


def _first_derivative_maximum(xs, ys, min_x=None, rank=0):
    """Return the ranked local maximum after the given minimum location."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 2:
        return None
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    start = 0
    if min_x is not None and np.isfinite(min_x):
        start = int(np.searchsorted(xs, float(min_x), side="right"))
    maxima = []
    for idx in range(max(start, 1), ys.size - 1):
        if ys[idx] >= ys[idx - 1] and ys[idx] > ys[idx + 1]:
            maxima.append((float(xs[idx]), float(ys[idx])))
            if len(maxima) > rank:
                return maxima[rank]
    if not maxima:
        max_idx = np.argmax(ys[start:])
        return float(xs[start + max_idx]), float(ys[start + max_idx])
    return maxima[-1]


def _max_second_derivative(xs, ys, min_x):
    """Return the first local maximum of d²P/dy² after the dP/dy minimum."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 3:
        return None
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    tol = 1e-6
    if min_x is not None and np.isfinite(min_x):
        start_idx = int(np.searchsorted(xs, float(min_x) + tol, side="left"))
    else:
        start_idx = 0
    start_idx = max(1, min(start_idx, xs.size - 2))
    second = np.gradient(ys, xs)
    for idx in range(start_idx, xs.size - 1):
        if second[idx] >= second[idx - 1] and second[idx] > second[idx + 1]:
            return float(xs[idx]), float(ys[idx]), float(second[idx])
    if start_idx >= xs.size - 1:
        return None
    tail = second[start_idx:]
    idx = start_idx + int(np.argmax(tail))
    idx = min(idx, xs.size - 1)
    return float(xs[idx]), float(ys[idx]), float(second[idx])


def _smooth_series(values, frac=0.15, min_window=5, max_window=81):
    """Return Savitzky-Golay smoothed copy of the provided series."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 5:
        return arr
    target = max(min_window, min(max_window, int(round(frac * arr.size))))
    if target % 2 == 0:
        target += 1
    target = min(target, arr.size if arr.size % 2 == 1 else arr.size - 1)
    target = max(5, target)
    poly_order = 3 if target > 3 else 2
    try:
        return savgol_filter(arr, target, polyorder=poly_order, mode="interp")
    except ValueError:
        window = max(3, min(arr.size // 2 * 2 + 1, target))
        if window < 3:
            return arr
        kernel = np.hanning(window)
        kernel = kernel / np.sum(kernel)
        return np.convolve(arr, kernel, mode="same")


def annotate_beta_angle(
    ax,
    slope,
    length_mm,
    x_min,
    x_max,
    y_min,
    y_max,
    fontsize=BETA_YR_FONT_SIZE,
    round_to_int=False,
    anchor_x_override=None,
):
    """Annotate beta angle near the combustor length line."""
    if length_mm is None or not np.isfinite(length_mm):
        return
    if slope is None or not np.isfinite(slope):
        return
    horizontal_scale_mm = COMBUSTOR_CIRCUMFERENCE_MM / (2.0 * np.pi)
    if horizontal_scale_mm <= 0.0:
        return
    dy_dx = slope / horizontal_scale_mm
    beta_deg = float(np.degrees(np.arctan(abs(dy_dx))))
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    if anchor_x_override is not None and np.isfinite(float(anchor_x_override)):
        anchor_x = float(np.clip(float(anchor_x_override), x_min + 0.05 * x_span, x_max - 0.05 * x_span))
    else:
        anchor_x = np.clip(x_min + 0.25 * x_span, x_min + 0.05 * x_span, x_max - 0.05 * x_span)
    preferred_y = float(length_mm) - 10.0
    anchor_y = np.clip(preferred_y, y_min + 0.2 * y_span, y_max - 0.05 * y_span)
    beta_txt = f"{int(np.rint(beta_deg))}" if round_to_int else f"{beta_deg:.1f}"
    ax.annotate(
        rf"$\beta \approx {beta_txt}^\circ$",
        xy=(anchor_x, anchor_y),
        xytext=(0.0, 0.0),
        textcoords="offset points",
        ha="center",
        va="center",
        color=ANNOTATION_DARK_BLUE,
        fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=ANNOTATION_DARK_BLUE, linewidth=0.9),
        zorder=8,
    )


def _plot_oblique_overlay(ax, fit, x_min, x_max, color, label, linestyle=":", linewidth=1.8):
    y_line = np.linspace(5.0, 125.0, 200)
    phase_line = fit["slope"] * y_line + fit["intercept"]
    mask = (phase_line >= x_min) & (phase_line <= x_max)
    if not np.any(mask):
        return None
    handle = ax.plot(
        phase_line[mask],
        y_line[mask],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )[0]
    return handle


def determine_pcb01_phase(summary, x_min, x_max):
    """Return PCB01 reference phase within plotting range, preferring the ~2π cycle."""
    axis_min = min(x_min, x_max)
    axis_max = max(x_min, x_max)
    preferred_phase = 2.0 * np.pi
    pcb01_phase = None
    for entry in summary:
        if entry.get("pcb", "").lower() != "pcb01":
            continue
        primary_list = entry.get("primary_list") or []
        candidates = []
        for peak in primary_list:
            if not peak:
                continue
            phase_val = float(peak["phase"])
            if axis_min <= phase_val <= axis_max:
                candidates.append(phase_val)
        if candidates:
            pcb01_phase = min(candidates, key=lambda ph: abs(ph - preferred_phase))
        if pcb01_phase is None and entry.get("primary"):
            phase_val = float(entry["primary"]["phase"])
            if axis_min <= phase_val <= axis_max:
                pcb01_phase = phase_val
        break
    if pcb01_phase is None:
        default_phase = preferred_phase
        pcb01_phase = min(max(default_phase, axis_min), axis_max)
    return float(pcb01_phase)


def _beta_deg_from_secondary_slope(slope):
    if slope is None or not np.isfinite(slope):
        return float("nan")
    horizontal_scale_mm = COMBUSTOR_CIRCUMFERENCE_MM / (2.0 * np.pi)
    if horizontal_scale_mm <= 0.0:
        return float("nan")
    return float(np.degrees(np.arctan(abs(float(slope) / horizontal_scale_mm))))


def _fit_y_vs_phase(points):
    xs = np.array([float(p["phase"]) for p in points], dtype=float)
    ys = np.array([float(p["y"]) for p in points], dtype=float)
    if xs.size < 2:
        return None, None
    slope, intercept = np.polyfit(xs, ys, 1)
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None, None
    return float(slope), float(intercept)


def plot_secondary_regression(
    ax,
    secondary_points,
    extend_to=None,
    fontsize=BETA_YR_FONT_SIZE,
    tol_deg=None,
    enforce_windows=False,
    round_to_int=False,
):
    """Plot reflected-shock fit through secondary points with optional angle tolerance/window gating."""
    if not secondary_points or len(secondary_points) < 2:
        return None

    parsed = []
    for p in secondary_points:
        if isinstance(p, dict):
            phase_val = p.get("phase")
            y_val = p.get("y")
            win = p.get("window")
        else:
            try:
                phase_val, y_val = p[0], p[1]
            except Exception:
                continue
            win = None
        try:
            phase = float(phase_val)
            y_val = float(y_val)
        except Exception:
            continue
        if not (np.isfinite(phase) and np.isfinite(y_val)):
            continue
        if win is not None:
            try:
                lo, hi = sorted((float(win[0]), float(win[1])))
                if enforce_windows and not (lo <= phase <= hi):
                    continue
                win = (lo, hi)
            except Exception:
                win = None
        parsed.append({"phase": phase, "y": y_val, "window": win})

    if len(parsed) < 2:
        return None

    # Default behaviour: plain least-squares on all points.
    used = list(parsed)
    slope, intercept = _fit_y_vs_phase(used)
    if slope is None:
        return None

    # Optional robust grow-from-inlet fit: keep line anchored to initial (high-y) points.
    if tol_deg is not None and np.isfinite(float(tol_deg)) and float(tol_deg) > 0.0 and len(parsed) >= 3:
        tol = abs(float(tol_deg))
        pts_sorted = sorted(parsed, key=lambda p: p["y"], reverse=True)
        used = pts_sorted[:3]
        slope, intercept = _fit_y_vs_phase(used)
        if slope is None:
            return None
        beta_ref = _beta_deg_from_secondary_slope(slope)
        for candidate in pts_sorted[3:]:
            trial = used + [candidate]
            slope_new, intercept_new = _fit_y_vs_phase(trial)
            if slope_new is None:
                continue
            beta_new = _beta_deg_from_secondary_slope(slope_new)
            if not np.isfinite(beta_new):
                continue
            if abs(beta_new - beta_ref) <= tol:
                used = trial
                slope = slope_new
                intercept = intercept_new
                beta_ref = beta_new

    xs = np.array([float(p["phase"]) for p in used], dtype=float)
    ys = np.array([float(p["y"]) for p in used], dtype=float)
    x_lim = ax.get_xlim()
    axis_min = min(x_lim)
    axis_max = max(x_lim)
    x_start = np.clip(xs.min(), axis_min, axis_max)
    x_end = np.clip(xs.max(), axis_min, axis_max)
    extend_target = None
    if extend_to is not None and np.isfinite(extend_to):
        extend_target = float(np.clip(extend_to, axis_min, axis_max))
        x_start = min(x_start, extend_target)
        x_end = max(x_end, extend_target)
    if x_end - x_start <= 1e-6:
        return None
    line_x = np.linspace(x_start, x_end, 200)
    line_y = slope * line_x + intercept
    y_min, y_max = ax.get_ylim()
    line_mask = (line_y >= min(y_min, y_max)) & (line_y <= max(y_min, y_max))
    if not np.any(line_mask):
        return None
    ax.plot(
        line_x[line_mask],
        line_y[line_mask],
        color="#00838f",
        linewidth=2.0,
        linestyle="-.",
        label="Reflected shock fit",
    )
    info = {"slope": slope, "intercept": intercept, "extend_point": None, "fit_points": used}
    if extend_target is None:
        return info
    y_point = slope * extend_target + intercept
    info["extend_point"] = (extend_target, y_point)
    y_min, y_max = ax.get_ylim()
    if min(y_min, y_max) <= y_point <= max(y_min, y_max):
        marker_kwargs = dict(
            color="#0d47a1",
            marker="X",
            s=80,
            linewidths=1.1,
            edgecolors="black",
            zorder=8,
            label="_nolegend_",
        )
        ax.scatter([extend_target], [y_point], **marker_kwargs)
        spacing = 8.0
        ax.annotate(
            (rf"$y_R \approx {int(np.rint(y_point))}\ \mathrm{{mm}}$" if round_to_int else rf"$y_R \approx {y_point:.1f}\ \mathrm{{mm}}$"),
            xy=(extend_target, y_point),
            xytext=(-spacing, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            color=ANNOTATION_DARK_BLUE,
            fontsize=fontsize,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor=ANNOTATION_DARK_BLUE,
                linewidth=0.8,
            ),
        )
    return info


def _pressure_map_render(
    data,
    fig=None,
    ax=None,
    legend_loc="upper right",
    legend_fontsize=10,
    ann_fontsize=10,
    beta_fontsize=10,
    phase_ref_rad=None,
    x_min_override=None,
    x_max_override=None,
    cbar_vmin=-1.0,
    cbar_vmax=4.5,
    cbar_extend="neither",
    cbar_gamma=1.0,
    cbar_ticks=None,
    x_label="Phase [rad]",
    round_annotation_int=False,
    oblique_fit_y_min=None,
    show_oblique_seed_marker=True,
    fixed_pcb01_phase=None,
    use_plain_pi_labels=False,
    alpha_label_x_rel=None,
    beta_label_x_rel=None,
):
    base = _setup_phase_map_axes(
        data,
        fig=fig,
        ax=ax,
        phase_ref_rad=phase_ref_rad,
        x_min_override=x_min_override,
        x_max_override=x_max_override,
        pressure_vmin=cbar_vmin,
        pressure_vmax=cbar_vmax,
        pressure_extend=cbar_extend,
        pressure_gamma=cbar_gamma,
        x_label=x_label,
        fixed_pcb01_phase=fixed_pcb01_phase,
        use_plain_pi_labels=bool(use_plain_pi_labels),
    )
    if base is None:
        if fig is not None:
            plt.close(fig)
        return _figure_map_griddata(data)

    fig = base["fig"]
    ax = base["ax"]
    cf = base["cf"]
    x_min, x_max = base["x_limits"]
    y_limits = base["y_limits"]
    pcb01_phase = base["pcb01_phase"]
    length_mm = base["length_mm"]
    summary = data["peak_summary"]
    Y_ax = data["Y_ax"]
    phase_max = data["phase_max_map"]
    points = collect_peak_points(summary, Y_ax, phase_max)
    metrics = data.setdefault("metrics", {})
    fit_info = None
    if points:
        primary_points = [(p["phase"], p["y"]) for p in points if p["type"] == "primary"]
        secondary_points = [(p["phase"], p["y"]) for p in points if p["type"] == "secondary"]
        tol_deg = float(data.get("obl_tol_deg", DEFAULT_OBLIQUE_TOL_DEG))
        if primary_points:
            xp, yp = zip(*primary_points)
            ax.scatter(xp, yp, color=PRIMARY_COLOR, marker="o", edgecolors="black", linewidths=0.6, s=54, label="Primary peak")
        if secondary_points:
            xs, ys = zip(*secondary_points)
            ax.scatter(xs, ys, color=SECONDARY_COLOR, marker="D", edgecolors="black", linewidths=0.6, s=48, label="Secondary peak")
            secondary_fit_points = []
            for idx, entry in enumerate(summary):
                secondary = entry.get("secondary")
                if not secondary:
                    continue
                try:
                    phase_val = float(secondary["phase"])
                    y_val = float(Y_ax[idx])
                except Exception:
                    continue
                secondary_fit_points.append(
                    {
                        "phase": phase_val,
                        "y": y_val,
                        "window": entry.get("secondary_window"),
                    }
                )
            fit_info = plot_secondary_regression(
                ax,
                secondary_fit_points if secondary_fit_points else secondary_points,
                pcb01_phase,
                fontsize=beta_fontsize,
                tol_deg=tol_deg,
                enforce_windows=True,
                round_to_int=bool(round_annotation_int),
            )
            if fit_info and fit_info.get("extend_point"):
                metrics["y_R_axial"] = float(fit_info["extend_point"][1])

        phase_window_min = max(2.0 * np.pi, x_min)
        phase_window_max = min(3.0 * np.pi, x_max)
        combustor_length_mm = data.get("combustor_length_mm")
        y_min_allowed = float(Y_ax[0]) + 1e-9
        if oblique_fit_y_min is not None and np.isfinite(float(oblique_fit_y_min)):
            y_min_allowed = max(y_min_allowed, float(oblique_fit_y_min))
        y_max_allowed = float(combustor_length_mm) - 5.0 if combustor_length_mm and np.isfinite(combustor_length_mm) else float(Y_ax[-1])
        last_pcb_y = None
        if combustor_length_mm and np.isfinite(combustor_length_mm):
            cutoff = float(combustor_length_mm) - 5.0
            valid_positions = [float(y_val) for y_val in Y_ax if float(y_val) <= cutoff + 1e-9]
            if valid_positions:
                last_pcb_y = max(valid_positions)
        oblique_candidates = []
        for phase, y in primary_points:
            if not (phase_window_min <= phase <= phase_window_max):
                continue
            if not (y_min_allowed <= y <= y_max_allowed):
                continue
            oblique_candidates.append((phase, y))
        if last_pcb_y is None and oblique_candidates:
            last_pcb_y = max(y for _, y in oblique_candidates)
        if last_pcb_y is not None:
            threshold = last_pcb_y - 1e-9
            oblique_candidates = [(phase, y) for phase, y in oblique_candidates if y < threshold]
        oblique_fit = fit_oblique_from_primary(
            oblique_candidates,
            data.get("combustor_length_mm"),
            tol_deg,
            phase_window_min,
            phase_window_max,
        )
        if oblique_fit:
            y_line = np.linspace(5.0, 125.0, 200)
            phase_line = oblique_fit["slope"] * y_line + oblique_fit["intercept"]
            mask = (phase_line >= x_min) & (phase_line <= x_max)
            if np.any(mask):
                ax.plot(
                    phase_line[mask],
                    y_line[mask],
                    color="#ffb74d",
                    linewidth=2.0,
                    linestyle="--",
                    label="Oblique shock fit",
                )
                fit_points = oblique_fit.get("points", [])
                if fit_points:
                    fit_phases = [float(ph) for ph, _ in fit_points]
                    fit_ys = [float(y) for _, y in fit_points]
                    ax.scatter(
                        fit_phases,
                        fit_ys,
                        facecolors="none",
                        edgecolors="white",
                        linewidths=1.4,
                        s=85,
                        zorder=6,
                    )
                seed_phase, seed_y = oblique_fit.get("seed_point", (None, None))
                start_phase = float(seed_phase) if seed_phase is not None else float(phase_line[mask][0])
                start_y = float(seed_y) if seed_y is not None else float(y_line[mask][0])
                if show_oblique_seed_marker:
                    ax.scatter(
                        [start_phase],
                        [start_y],
                        facecolors="none",
                        edgecolors="red",
                        linewidths=1.4,
                        s=130,
                        zorder=7,
                    )
                slope = oblique_fit["slope"]
                if abs(slope) > 1e-9:
                    y_intercept = float((pcb01_phase - oblique_fit["intercept"]) / slope)
                    if 5.0 <= y_intercept <= 125.0:
                        metrics["H_axial"] = y_intercept
                        ax.scatter(
                            [pcb01_phase],
                            [y_intercept],
                            facecolors="#e65100",
                            edgecolors="black",
                            linewidths=0.8,
                            s=75,
                            zorder=7,
                            label="_nolegend_",
                        )
                        ax.annotate(
                            (rf"$H \approx {int(np.rint(y_intercept))}\ \mathrm{{mm}}$" if round_annotation_int else rf"$H \approx {y_intercept:.1f}\ \mathrm{{mm}}$"),
                            xy=(pcb01_phase, y_intercept),
                            xytext=(8, 0),
                            textcoords="offset points",
                            color=ANNOTATION_DARK_ORANGE,
                            fontsize=ann_fontsize,
                            va="center",
                            bbox=dict(
                                boxstyle="round,pad=0.2",
                                facecolor="white",
                                edgecolor=ANNOTATION_DARK_ORANGE,
                                linewidth=0.8,
                            ),
                        )
                    alpha_point = fit_points[min(len(fit_points) - 1, 1)] if fit_points else (phase_line[mask][0], y_line[mask][0])
                    x_alpha = float(alpha_point[0])
                    y_alpha = float(alpha_point[1])
                    if alpha_label_x_rel is not None and phase_ref_rad is not None and np.isfinite(float(alpha_label_x_rel)):
                        x_alpha = float(phase_ref_rad) + float(alpha_label_x_rel)
                    if length_mm is not None and np.isfinite(length_mm):
                        y_alpha = float(length_mm) - 3.0
                    y_alpha = np.clip(y_alpha, 5.0, 125.0)
                    x_alpha = float(np.clip(x_alpha, x_min + 0.02 * (x_max - x_min), x_max - 0.02 * (x_max - x_min)))
                    if alpha_label_x_rel is not None and phase_ref_rad is not None:
                        alpha_offset_x = 0
                        alpha_ha = "center"
                    else:
                        alpha_offset_x = -8 if x_alpha > (x_min + x_max) / 2 else 8
                        alpha_ha = "right" if alpha_offset_x < 0 else "left"
                    ax.annotate(
                        (
                            rf"$\alpha \approx {int(np.rint(abs(np.degrees(np.arctan((2.0 * np.pi) / (COMBUSTOR_CIRCUMFERENCE_MM * slope))))))}^\circ$"
                            if round_annotation_int
                            else rf"$\alpha \approx {abs(np.degrees(np.arctan((2.0 * np.pi) / (COMBUSTOR_CIRCUMFERENCE_MM * slope)))):.1f}^\circ$"
                        ),
                        xy=(x_alpha, y_alpha),
                        xytext=(alpha_offset_x, -6),
                        textcoords="offset points",
                        color=ANNOTATION_DARK_ORANGE,
                        fontsize=ann_fontsize,
                        ha=alpha_ha,
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            facecolor="white",
                            edgecolor=ANNOTATION_DARK_ORANGE,
                            linewidth=0.8,
                        ),
                    )

    if fit_info and length_mm is not None and np.isfinite(length_mm):
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        annotate_beta_angle(
            ax,
            fit_info.get("slope"),
            length_mm,
            x_lim[0],
            x_lim[1],
            y_lim[0],
            y_lim[1],
            fontsize=beta_fontsize,
            round_to_int=bool(round_annotation_int),
            anchor_x_override=(float(phase_ref_rad) + float(beta_label_x_rel))
            if (beta_label_x_rel is not None and phase_ref_rad is not None and np.isfinite(float(beta_label_x_rel)))
            else None,
        )

    cbar = fig.colorbar(cf, ax=ax, extend=str(cbar_extend))
    cbar.set_label("Pressure [bar]")
    if cbar_ticks is None:
        cbar_ticks = np.arange(float(cbar_vmin), float(cbar_vmax) + 1e-6, 0.5)
    cbar.set_ticks(np.asarray(cbar_ticks, dtype=float))
    ax.legend(loc=legend_loc, frameon=True, fontsize=legend_fontsize)
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Compute pipeline
# ---------------------------------------------------------------------------


def compute_figures(
    mat_path,
    ref_key,
    revs,
    f_det_in,
    use_filter=False,
    filter_low_hz=100.0,
    filter_high_hz=100000.0,
    pcb_pos=None,
    obl_tol_deg=DEFAULT_OBLIQUE_TOL_DEG,
    secondary_seed_offset=0,
    secondary_seed_span=DEFAULT_SECONDARY_SEED_SPAN,
    secondary_seed_bounds=None,
    secondary_allow_wrap_once=False,
    dpdy_smooth_frac=DEFAULT_DPDY_SMOOTH_FRAC,
    dpdy_max_rank=0,
    secondary_window_gain=DEFAULT_SECONDARY_WINDOW_GAIN,
    use_hermite_fit=False,
    use_second_deriv_max=False,
    polyfit_last_pcb=None,
    time_start=None,
    time_end=None,
    show_peak_gradient=False,
    show_ridge_detect=False,
    use_soft_dtw_phase_average=False,
):
    if pcb_pos is None:
        pcb_pos = DEFAULT_POS_MM
    time, pcb_raw, mat_struct = load_dataset(mat_path)
    if time_start is not None or time_end is not None:
        t_min = float(time_start) if time_start is not None else float(time[0])
        t_max = float(time_end) if time_end is not None else float(time[-1])
        if t_min >= t_max:
            raise ValueError("Time window start must be less than end.")
        mask = (time >= t_min) & (time <= t_max)
        if mask.sum() < 100:
            raise ValueError("Time window too short to compute phase averages.")
        time = time[mask]
        for k in list(pcb_raw.keys()):
            pcb_raw[k] = pcb_raw[k][mask]
    ref_key_norm = _norm_key(ref_key) or "pcb01"
    if ref_key_norm not in pcb_raw:
        ref_key_norm = sorted(pcb_raw.keys(), key=lambda k: int(k[-2:]))[0]
    dt = float(np.mean(np.diff(time)))
    fs = 1.0 / dt
    pcb = {}
    for k, v in pcb_raw.items():
        if use_filter:
            try:
                pcb[k] = butter_bandpass_filter(time, v, float(filter_low_hz), float(filter_high_hz), order=4)
            except Exception:
                pcb[k] = v.copy()
        else:
            pcb[k] = v.copy()
    all_keys = sorted(pcb.keys(), key=lambda k: int(k[-2:]))
    axial_keys = [k for k in all_keys if 1 <= int(k[-2:]) <= 13]
    radial_keys = [k for k in all_keys if int(k[-2:]) > 13]
    ref_candidates = radial_keys if radial_keys else [ref_key_norm]
    if f_det_in and f_det_in > 0:
        f_det = f_det_in
    else:
        f_det = estimate_ref_freq_welch(time, pcb[ref_key_norm])
        if not f_det:
            f_det = estimate_det_freq(time, [pcb[k] for k in ref_candidates])
    if not f_det:
        f_det = 6200.0
    waves_per_rev = 1 if 4000.0 <= f_det <= 7000.0 else (2 if f_det > 7000.0 else 1)
    pk_global = pick_phase_locked_peaks(time, pcb[ref_key_norm], f_det)
    samples_per_rev_guess = int(round(fs / max(f_det / max(waves_per_rev, 1), 1e-6)))
    if samples_per_rev_guess <= 0:
        samples_per_rev_guess = 1
    if pk_global.size == 0:
        pk_global = np.arange(samples_per_rev_guess, len(time) - 2 * samples_per_rev_guess, samples_per_rev_guess)
    signals_dict = {}
    for key in axial_keys:
        signals_dict[key] = pcb[key]
    if ref_key_norm not in signals_dict:
        signals_dict[ref_key_norm] = pcb[ref_key_norm]
    seg_bank, X_rev, _ = build_grad_aligned_segments(
        time,
        signals_dict,
        ref_key_norm,
        pk_global,
        fs,
        f_det,
        waves_per_rev,
    )
    Z = np.full((13, len(X_rev)), np.nan, dtype=float)
    for idx, name in enumerate(PCB_LABELS):
        if name in seg_bank and seg_bank[name].size:
            Z[idx, :] = seg_bank[name].mean(axis=0)
    if use_soft_dtw_phase_average:
        soft_payload = _extract_soft_dtw_phase_average(mat_struct)
        if soft_payload is not None:
            phase_soft, bary_soft, labels_soft = soft_payload
            label_to_row = {lbl: i for i, lbl in enumerate(labels_soft)}
            Z_soft = np.full((13, len(phase_soft)), np.nan, dtype=float)
            for idx, name in enumerate(PCB_LABELS):
                row_idx = label_to_row.get(name)
                if row_idx is None:
                    continue
                if 0 <= row_idx < bary_soft.shape[0]:
                    Z_soft[idx, :] = bary_soft[row_idx, :]
            if np.isfinite(Z_soft).any():
                X_rev = np.asarray(phase_soft, dtype=float).reshape(-1)
                Z = Z_soft
    Y_ax = np.array(pcb_pos, dtype=float)
    phase_limit_total = float(X_rev[-1])
    xlim = (0.0, min(2.0 * np.pi * revs, phase_limit_total))
    theta_marks = np.arange(0.0, xlim[1] + 1e-6, 2.0 * np.pi)
    if theta_marks.size == 0:
        theta_marks = np.array([0.0, 2.0 * np.pi])
    combustor_length_mm = parse_length_mm_from_metadata(mat_struct, mat_path)
    polyfit_last_idx = _resolve_polyfit_limit_idx(combustor_length_mm, Y_ax, polyfit_last_pcb)
    peak_summary = summarize_peaks(
        X_rev,
        Z,
        Y_ax,
        combustor_length_mm,
        secondary_seed_offset=secondary_seed_offset,
        secondary_seed_span=secondary_seed_span,
        secondary_seed_bounds=secondary_seed_bounds,
        secondary_window_gain=secondary_window_gain,
        secondary_allow_wrap_once=bool(secondary_allow_wrap_once),
    )
    data = {
        "ref_key": ref_key_norm,
        "seg_bank": seg_bank,
        "X_rev": X_rev,
        "avg_matrix": Z,
        "Y_ax": Y_ax,
        "theta_marks": theta_marks,
        "xlim": xlim,
        "peak_summary": peak_summary,
        "combustor_length_mm": combustor_length_mm,
        "polyfit_last_idx": polyfit_last_idx,
        "phase_max_discrete": min(PHASE_LIMIT_DISCRETE, phase_limit_total),
        "phase_max_map": min(PHASE_LIMIT_MAP, phase_limit_total),
        "f_det": f_det,
        "waves_per_rev": waves_per_rev,
        "obl_tol_deg": float(obl_tol_deg),
        "dpdy_smooth_frac": float(dpdy_smooth_frac),
        "dpdy_max_rank": int(dpdy_max_rank),
        "use_hermite_fit": bool(use_hermite_fit),
        "use_second_deriv_max": bool(use_second_deriv_max),
    }
    return build_figures(data, show_peak_gradient=show_peak_gradient, show_ridge_detect=show_ridge_detect)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------


class PressureMapAppV02:
    def __init__(self, root, default_args):
        self.root = root
        self.root.title("Pressure Map GUI V02")
        self.args = default_args
        initial_mat = getattr(default_args, "mat", None)
        target_path = None
        if initial_mat:
            target_path = Path(initial_mat)
        elif DEFAULT_DATASET_DIR.exists():
            target_path = DEFAULT_DATASET_DIR / DEFAULT_DATASET_FILE
        if target_path and not target_path.exists():
            target_path = None
        start_folder = target_path.parent if target_path else (DEFAULT_DATASET_DIR if DEFAULT_DATASET_DIR.exists() else None)
        self.current_folder = tk.StringVar(value=str(start_folder) if start_folder else "")
        self.ref_var = tk.StringVar(value=getattr(default_args, "ref", "pcb01"))
        self.revs_var = tk.IntVar(value=max(1, int(getattr(default_args, "revs", 3) or 3)))
        self.fdet_var = tk.DoubleVar(value=float(getattr(default_args, "fdet", 0.0) or 0.0))
        self.filter_var = tk.BooleanVar(value=bool(getattr(default_args, "use_filter", False)))
        self.filter_low_hz_var = tk.StringVar(value=str(getattr(default_args, "filter_low_hz", 100.0) or 100.0))
        self.filter_high_hz_var = tk.StringVar(value=str(getattr(default_args, "filter_high_hz", 100000.0) or 100000.0))
        self.obl_tol_deg_var = tk.DoubleVar(value=float(getattr(default_args, "obl_tol_deg", DEFAULT_OBLIQUE_TOL_DEG) or DEFAULT_OBLIQUE_TOL_DEG))
        default_multiplier = DEFAULT_SECONDARY_SEED_SPAN / np.pi
        self.secondary_span_multiplier_var = tk.DoubleVar(value=default_multiplier)
        self.use_second_seed_var = tk.BooleanVar(value=False)
        self.show_peak_gradient_var = tk.BooleanVar(value=False)
        self.show_ridge_detect_var = tk.BooleanVar(value=False)
        self.use_next_dpdy_max_var = tk.BooleanVar(value=False)
        self.dpdy_smooth_var = tk.DoubleVar(value=DEFAULT_DPDY_SMOOTH_FRAC)
        self.secondary_window_gain_var = tk.DoubleVar(value=DEFAULT_SECONDARY_WINDOW_GAIN)
        self.secondary_seed_min_var = tk.StringVar(value="")
        self.secondary_seed_max_var = tk.StringVar(value="")
        self.use_hermite_var = tk.BooleanVar(value=False)
        self.use_second_deriv_var = tk.BooleanVar(value=False)
        self.polyfit_last_var = tk.StringVar(value="")
        self.time_start_var = tk.StringVar(value="270")
        self.time_end_var = tk.StringVar(value="320")
        self.figures = []
        self.canvases = []
        self.auto_compute_pending = False
        self._build_ui()
        if target_path:
            self.select_initial_file(str(target_path))
            self.auto_compute_pending = True
        elif start_folder:
            self.populate_file_list(str(start_folder))
            if getattr(self, "mat_files", None):
                self.file_list.selection_clear(0, tk.END)
                self.file_list.selection_set(0)
                self.file_list.see(0)
        if self.auto_compute_pending:
            self.root.after(200, self._auto_compute_if_ready)

    def _build_ui(self):
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Folder...", command=self.choose_folder)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.root.destroy)
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)
        left_container = ttk.Frame(main)
        left_container.pack(side="left", fill="y")
        self.left_canvas = tk.Canvas(left_container, borderwidth=0, highlightthickness=0, height=600)
        left_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=self.left_canvas.yview)
        self.left_canvas.configure(yscrollcommand=left_scrollbar.set)
        self.left_canvas.pack(side="left", fill="y", expand=False)
        left_scrollbar.pack(side="right", fill="y")
        left = ttk.Frame(self.left_canvas)
        self.left_canvas_window = self.left_canvas.create_window((0, 0), window=left, anchor="nw")

        def _on_left_config(_event):
            self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))

        def _on_canvas_resize(event):
            self.left_canvas.itemconfigure(self.left_canvas_window, width=event.width)

        left.bind("<Configure>", _on_left_config)
        self.left_canvas.bind("<Configure>", _on_canvas_resize)

        def _on_mousewheel(event):
            delta = -1 if event.delta > 0 else 1
            if os.name == "nt":
                delta = -int(event.delta / 120)
            elif event.num == 5:
                delta = 1
            elif event.num == 4:
                delta = -1
            self.left_canvas.yview_scroll(delta, "units")

        self.left_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.left_canvas.bind_all("<Button-4>", _on_mousewheel)
        self.left_canvas.bind_all("<Button-5>", _on_mousewheel)

        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True)
        # Folder chooser
        folder_frame = ttk.Frame(left)
        folder_frame.pack(fill="x")
        ttk.Label(folder_frame, text="Folder").pack(side="left")
        self.folder_entry = ttk.Entry(folder_frame, textvariable=self.current_folder, width=36)
        self.folder_entry.pack(side="left", padx=6)
        ttk.Button(folder_frame, text="Browse...", command=self.choose_folder).pack(side="left")
        # File list
        ttk.Label(left, text=".mat files").pack(anchor="w", pady=(10, 2))
        self.file_list = tk.Listbox(left, height=12, width=42, exportselection=False)
        self.file_list.pack(fill="x")
        self.file_list.bind("<Double-1>", lambda _e: self.compute())
        # Options
        opt = ttk.LabelFrame(left, text="Options")
        opt.pack(fill="x", pady=(10, 0))
        ttk.Label(opt, text="Reference PCB").grid(row=0, column=0, sticky="w")
        ttk.Entry(opt, textvariable=self.ref_var, width=12).grid(row=0, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(opt, text="Revolutions").grid(row=1, column=0, sticky="w")
        tk.Spinbox(opt, from_=1, to=4, textvariable=self.revs_var, width=6).grid(row=1, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(opt, text="f_det (Hz, 0=auto)").grid(row=2, column=0, sticky="w")
        ttk.Entry(opt, textvariable=self.fdet_var, width=12).grid(row=2, column=1, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(opt, text="Bandpass filter", variable=self.filter_var).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(opt, text="Low pass [Hz]").grid(row=4, column=0, sticky="w")
        ttk.Entry(opt, textvariable=self.filter_low_hz_var, width=12).grid(row=4, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(opt, text="High pass [Hz]").grid(row=5, column=0, sticky="w")
        ttk.Entry(opt, textvariable=self.filter_high_hz_var, width=12).grid(row=5, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(opt, text="Obl tol (deg)").grid(row=6, column=0, sticky="w")
        ttk.Entry(opt, textvariable=self.obl_tol_deg_var, width=12).grid(row=6, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(opt, text="Seed span (×π)").grid(row=7, column=0, sticky="w")
        tk.Spinbox(
            opt,
            from_=0.5,
            to=6.0,
            increment=0.1,
            textvariable=self.secondary_span_multiplier_var,
            width=8,
            format="%.1f",
        ).grid(row=7, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(opt, text="Window gain").grid(row=8, column=0, sticky="w")
        tk.Spinbox(
            opt,
            from_=0.5,
            to=5.0,
            increment=0.1,
            textvariable=self.secondary_window_gain_var,
            width=8,
            format="%.1f",
        ).grid(row=8, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(opt, text="dP/dy smoothing").grid(row=9, column=0, sticky="w")
        tk.Spinbox(
            opt,
            from_=0.02,
            to=0.5,
            increment=0.01,
            textvariable=self.dpdy_smooth_var,
            width=8,
            format="%.2f",
        ).grid(row=9, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(opt, text="Time start [ms]").grid(row=10, column=0, sticky="w")
        ttk.Entry(opt, textvariable=self.time_start_var, width=12).grid(row=10, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(opt, text="Time end [ms]").grid(row=11, column=0, sticky="w")
        ttk.Entry(opt, textvariable=self.time_end_var, width=12).grid(row=11, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(opt, text="Fig 3 seed min [x\u03c0]").grid(row=12, column=0, sticky="w")
        ttk.Entry(opt, textvariable=self.secondary_seed_min_var, width=12).grid(row=12, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(opt, text="Fig 3 seed max [x\u03c0]").grid(row=13, column=0, sticky="w")
        ttk.Entry(opt, textvariable=self.secondary_seed_max_var, width=12).grid(row=13, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(opt, text="Fig 6 last PCB").grid(row=14, column=0, sticky="w")
        ttk.Entry(opt, textvariable=self.polyfit_last_var, width=12).grid(row=14, column=1, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(
            left,
            text="Fig 3a - peak gradient finder",
            variable=self.show_peak_gradient_var,
        ).pack(fill="x", pady=(6, 0), anchor="w")
        ttk.Checkbutton(
            left,
            text="Fig 5a - ride detect",
            variable=self.show_ridge_detect_var,
        ).pack(fill="x", pady=(0, 0), anchor="w")
        ttk.Checkbutton(
            left,
            text="Fig 3 - secondary seed - use 2nd maxima",
            variable=self.use_second_seed_var,
        ).pack(fill="x", pady=(0, 0), anchor="w")
        ttk.Checkbutton(
            left,
            text="Fig 6 - dP/dy max - use next maxima",
            variable=self.use_next_dpdy_max_var,
        ).pack(fill="x", pady=(0, 6), anchor="w")
        ttk.Checkbutton(
            left,
            text="Fig 6 - show Hermite spline",
            variable=self.use_hermite_var,
        ).pack(fill="x", pady=(0, 6), anchor="w")
        ttk.Checkbutton(
            left,
            text="Fig 6 - show d²P/dy² max",
            variable=self.use_second_deriv_var,
        ).pack(fill="x", pady=(0, 6), anchor="w")
        ttk.Button(left, text="Compute figures", command=self.compute).pack(fill="x", pady=(0, 0))
        # Notebook for figures
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)

    def select_initial_file(self, mat_path):
        path = Path(mat_path)
        if not path.exists():
            return
        folder = str(path.parent)
        self.current_folder.set(folder)
        self.populate_file_list(folder)
        try:
            idx = self.mat_files.index(path.name)
            self.file_list.selection_clear(0, tk.END)
            self.file_list.selection_set(idx)
            self.file_list.see(idx)
        except ValueError:
            pass

    def _auto_compute_if_ready(self):
        if self.get_selected_mat_path():
            self.compute()

    def choose_folder(self):
        folder = filedialog.askdirectory(title="Select folder", initialdir=self.current_folder.get() or os.getcwd())
        if folder:
            self.current_folder.set(folder)
            self.populate_file_list(folder)

    def populate_file_list(self, folder):
        self.file_list.delete(0, tk.END)
        folder_path = Path(folder)
        if not folder_path.exists():
            self.mat_files = []
            return
        files = sorted(p.name for p in folder_path.glob("*.mat"))
        self.mat_files = files
        for name in files:
            self.file_list.insert(tk.END, name)

    def get_selected_mat_path(self):
        if not getattr(self, "mat_files", None):
            return None
        selection = self.file_list.curselection()
        idx = selection[0] if selection else 0
        if idx >= len(self.mat_files):
            return None
        folder = self.current_folder.get()
        if not folder:
            return None
        return str(Path(folder) / self.mat_files[idx])

    def _parse_time_window(self):
        start_txt = self.time_start_var.get().strip()
        end_txt = self.time_end_var.get().strip()
        start_val = float(start_txt) / 1000.0 if start_txt else None
        end_val = float(end_txt) / 1000.0 if end_txt else None
        if start_val is not None and end_val is not None and start_val >= end_val:
            raise ValueError("Time start must be less than time end.")
        return (start_val, end_val)

    def _parse_filter_band(self):
        low_txt = self.filter_low_hz_var.get().strip()
        high_txt = self.filter_high_hz_var.get().strip()
        low_hz = float(low_txt) if low_txt else 100.0
        high_hz = float(high_txt) if high_txt else 100000.0
        if low_hz <= 0 or high_hz <= 0:
            raise ValueError("Filter cutoffs must be positive.")
        if low_hz >= high_hz:
            raise ValueError("Low pass must be less than high pass.")
        return (low_hz, high_hz)

    def compute(self):
        mat_path = self.get_selected_mat_path()
        if not mat_path:
            messagebox.showwarning("No file selected", "Please choose a .mat file.")
            return
        ref = self.ref_var.get().strip() or "pcb01"
        revs = max(1, self.revs_var.get())
        f_det = float(self.fdet_var.get())
        use_filter = bool(self.filter_var.get())
        filter_band = (100.0, 100000.0)
        if use_filter:
            try:
                filter_band = self._parse_filter_band()
            except ValueError as exc:
                messagebox.showerror("Filter settings error", str(exc))
                return
        obl_tol_deg = float(self.obl_tol_deg_var.get() or DEFAULT_OBLIQUE_TOL_DEG)
        span_multiplier = float(self.secondary_span_multiplier_var.get() or (DEFAULT_SECONDARY_SEED_SPAN / np.pi))
        span_val = max(0.1, span_multiplier) * np.pi
        secondary_rank = 1 if self.use_second_seed_var.get() else 0
        dpdy_max_rank = 1 if self.use_next_dpdy_max_var.get() else 0
        window_gain = float(self.secondary_window_gain_var.get() or DEFAULT_SECONDARY_WINDOW_GAIN)
        window_gain = max(0.5, min(5.0, window_gain))
        seed_lo = self.secondary_seed_min_var.get().strip()
        seed_hi = self.secondary_seed_max_var.get().strip()
        seed_bounds = None
        if seed_lo or seed_hi:
            try:
                lo_val = (float(seed_lo) * np.pi) if seed_lo else 0.0
                hi_val = (float(seed_hi) * np.pi) if seed_hi else float(2.0 * np.pi)
                if lo_val >= hi_val:
                    raise ValueError("Secondary seed min must be < max.")
                seed_bounds = (lo_val, hi_val)
            except ValueError as exc:
                messagebox.showerror("Secondary seed bounds error", str(exc))
                return
        try:
            time_window = self._parse_time_window()
        except ValueError as exc:
            messagebox.showerror("Time window error", str(exc))
            return
        dpdy_smooth = float(self.dpdy_smooth_var.get() or DEFAULT_DPDY_SMOOTH_FRAC)
        dpdy_smooth = min(0.5, max(0.01, dpdy_smooth))
        polyfit_last = self.polyfit_last_var.get().strip() or None
        try:
            figures = compute_figures(
                mat_path,
                ref,
                revs,
                f_det,
                use_filter=use_filter,
                filter_low_hz=filter_band[0],
                filter_high_hz=filter_band[1],
                obl_tol_deg=obl_tol_deg,
                secondary_seed_offset=secondary_rank,
                secondary_seed_span=span_val,
                secondary_seed_bounds=seed_bounds,
                dpdy_smooth_frac=dpdy_smooth,
                dpdy_max_rank=dpdy_max_rank,
                secondary_window_gain=window_gain,
                polyfit_last_pcb=polyfit_last,
                time_start=time_window[0],
                time_end=time_window[1],
                use_hermite_fit=self.use_hermite_var.get(),
                use_second_deriv_max=self.use_second_deriv_var.get(),
                show_peak_gradient=self.show_peak_gradient_var.get(),
                show_ridge_detect=self.show_ridge_detect_var.get(),
            )
        except Exception as exc:
            messagebox.showerror("Computation failed", f"Unable to compute figures:\n{exc}")
            return
        self.render_figures(figures)

    def clear_figures(self):
        for canvas in self.canvases:
            canvas.get_tk_widget().destroy()
        self.canvases = []
        self.figures = []
        for tab_id in self.notebook.tabs():
            self.notebook.forget(tab_id)

    def render_figures(self, figures):
        self.clear_figures()
        for label, fig in figures:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=label)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.pack(fill="both", expand=True)
            self.canvases.append(canvas)
            self.figures.append(fig)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Simplified pressure map GUI (V02)")
    default_mat_path = DEFAULT_DATASET_DIR / DEFAULT_DATASET_FILE
    ap.add_argument("--mat", default=str(default_mat_path), help="Optional .mat file to preload (default: CO1083 file)")
    ap.add_argument("--ref", default="pcb01", help="Reference PCB (default: pcb01)")
    ap.add_argument("--revs", type=int, default=2, help="Number of revolutions to show (default: 3)")
    ap.add_argument("--fdet", type=float, default=0.0, help="Detection frequency in Hz (0=auto)")
    ap.add_argument("--use-filter", action="store_true", help="Apply bandpass filter before averaging")
    ap.add_argument("--obl-tol-deg", type=float, default=DEFAULT_OBLIQUE_TOL_DEG, help="Oblique fit tolerance in degrees (default: 6)")
    args = ap.parse_args()
    root = tk.Tk()
    app = PressureMapAppV02(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
