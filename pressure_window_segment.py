"""
Pressure Window Segment Processor
---------------------------------
Creates a new MAT file with:
  D.window.data      -> time + pcb01..pcb15 windowed data
  D.window.segment   -> 6pi (3 rev) gradient-aligned segments (pcb01..pcb13)
  D.window.phase_ave -> phase-averaged traces from segmented data
  D.window.soft_dtw  -> optional Soft-DTW summary (when enabled)

Preserves all existing D fields and appends:
  D.window
"""
from __future__ import annotations

import argparse
import os
import glob
import warnings
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat
from scipy.signal import find_peaks

try:
    from tslearn.metrics import soft_dtw, soft_dtw_alignment
    from tslearn.barycenters import softdtw_barycenter
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance
    TSLEARN_AVAILABLE = True
except Exception:
    TSLEARN_AVAILABLE = False

PCB_WINDOW_LABELS = [f"pcb{ii:02d}" for ii in range(1, 16)]
PCB_AXIAL_LABELS = [f"pcb{ii:02d}" for ii in range(1, 14)]
DEFAULT_T_START = 0.27
DEFAULT_T_END = 0.32
DEFAULT_FDET_MIN_HZ = 3000.0
DEFAULT_FDET_MAX_HZ = 20000.0
DEFAULT_PEAK_LAST_WINDOW_S = 0.050
DEFAULT_PEAK_HEIGHT_FRAC = 0.2
DEFAULT_PEAK_MIN_SEP_FRAC = 0.6
DEFAULT_ALIGN_GRAD_LEFT_FRAC = 0.20
DEFAULT_ALIGN_REQUIRE_POSITIVE_GRAD = True
DEFAULT_SOFT_DTW_GAMMA = 0.01
DEFAULT_SOFT_DTW_SUBDIR = "soft_dtw"
DEFAULT_SOFT_DTW_PROGRESS_EVERY = 10
DEFAULT_COH_FFT_LOCAL_HALF_BW_HZ = 200.0
DEFAULT_COH_FFT_HARMONIC_FMAX_HZ = 15000.0


def _unwrap_cell(x):
    if isinstance(x, np.ndarray) and x.size == 1:
        return x.item()
    return x


def _unwrap_struct(x):
    if isinstance(x, np.ndarray):
        if x.dtype.names:
            if x.size == 1:
                return x[0, 0] if x.ndim == 2 else x.flat[0]
            return x
        if x.size == 1:
            return x.item()
    return x


def _get_field_raw(D, name):
    if name not in D.dtype.names:
        return None
    fld = D[name]
    return _unwrap_struct(fld)


def _get_field_scalar(D, name):
    val = _get_field_raw(D, name)
    if val is None:
        return None
    try:
        return float(np.array(val).astype(float).squeeze())
    except Exception:
        return val


def _get_nested_field(D, outer, inner):
    if outer not in D.dtype.names:
        return None
    obj = _unwrap_struct(D[outer])
    if not (hasattr(obj, "dtype") and obj.dtype.names and inner in obj.dtype.names):
        return None
    val = obj[inner]
    return _unwrap_struct(val)


def extract_time(D):
    if "time" not in D.dtype.names:
        raise ValueError("D.time not found.")
    t_field = _unwrap_struct(D["time"])

    if hasattr(t_field, "dtype") and t_field.dtype.names and "t" in t_field.dtype.names:
        t = _unwrap_struct(t_field["t"])
    else:
        if isinstance(t_field, tuple) and len(t_field) > 0:
            t = t_field[0]
        else:
            t = t_field

    t = np.array(t).astype(float).squeeze()
    return t.flatten()


def extract_pcb(D, key):
    if key not in D.dtype.names:
        return None
    fld = _unwrap_struct(D[key])
    if hasattr(fld, "dtype") and fld.dtype.names and "data" in fld.dtype.names:
        data = _unwrap_struct(fld["data"])
    else:
        data = fld
    data = np.array(data).astype(float).squeeze()
    return data.flatten()


def estimate_det_freq(time, signals, fmin=3000.0, fmax=20000.0):
    dt = float(np.mean(np.diff(time)))
    fs = 1.0 / dt
    n = len(time)
    win = np.hanning(n)
    f_est = []
    for sig in signals:
        x = sig - np.mean(sig)
        X = np.fft.rfft(win * x)
        freqs = np.fft.rfftfreq(n, d=dt)
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(mask):
            continue
        k = np.argmax(np.abs(X[mask]))
        f_est.append(freqs[mask][k])
    return float(np.mean(f_est)) if f_est else None


def _compute_fft_coherence_struct(
    time_s,
    signal,
    fmin_hz=DEFAULT_FDET_MIN_HZ,
    fmax_hz=DEFAULT_FDET_MAX_HZ,
    local_half_bw_hz=DEFAULT_COH_FFT_LOCAL_HALF_BW_HZ,
    harmonic_fmax_hz=DEFAULT_COH_FFT_HARMONIC_FMAX_HZ,
):
    """Minimal FFT coherence metrics aligned with coherence_paper_figures PDR logic."""
    t = np.asarray(time_s, dtype=float).reshape(-1)
    x = np.asarray(signal, dtype=float).reshape(-1)
    if t.size < 8 or x.size != t.size:
        raise ValueError("Invalid time/signal inputs for coherence FFT.")
    dt = float(np.mean(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid time spacing for coherence FFT.")
    fs = 1.0 / dt

    x = x - float(np.nanmean(x))
    x = np.nan_to_num(x, nan=0.0)
    n = int(x.size)
    if n < 4:
        raise ValueError("Too few samples for coherence FFT.")

    win = np.hanning(n)
    X = np.fft.rfft(win * x)
    f_hz = np.fft.rfftfreq(n, d=dt)
    mag = np.abs(X) / float(n)
    df_hz = float(fs / n)

    fmax_eff = float(min(float(fmax_hz), fs / 2.0))
    band_mask = (f_hz >= float(max(0.0, fmin_hz))) & (f_hz <= fmax_eff)
    if not np.any(band_mask):
        raise ValueError("FFT band contains no samples.")

    band_idxs = np.where(band_mask)[0]
    idx_peak = int(band_idxs[int(np.argmax(mag[band_mask]))])
    fp_hz = float(f_hz[idx_peak])
    fp_amp = float(mag[idx_peak])
    if not np.isfinite(fp_amp) or fp_amp <= 0:
        raise ValueError("FFT dominant peak amplitude is zero/invalid.")

    # Match coherence_paper_figures dominance ratio logic (peak divided by total energy in region).
    band_sum = float(np.sum(mag[band_mask]))
    pdr_g = (fp_amp / band_sum) if band_sum > 0 else float("inf")

    local_half_bw_hz = float(local_half_bw_hz)
    local_mask = band_mask & (f_hz >= (fp_hz - local_half_bw_hz)) & (f_hz <= (fp_hz + local_half_bw_hz))
    if np.any(local_mask):
        local_sum = float(np.sum(mag[local_mask]))
        pdr_l = (fp_amp / local_sum) if local_sum > 0 else float("inf")
    else:
        pdr_l = float("nan")

    # Harmonic ratio mirrors coherence_paper_figures "harmonic_ratio" style.
    f_cap = float(min(float(harmonic_fmax_hz), fs / 2.0))
    harmonic_union = np.zeros_like(mag, dtype=bool)
    harmonic_peak_points = []
    n_h = 2
    while (n_h * fp_hz - local_half_bw_hz) <= f_cap:
        center = n_h * fp_hz
        lo = max(0.0, center - local_half_bw_hz)
        hi = min(f_cap, center + local_half_bw_hz)
        if hi >= lo:
            mask_h = (f_hz >= lo) & (f_hz <= hi)
            if np.any(mask_h):
                harmonic_union |= mask_h
                idxs = np.where(mask_h)[0]
                idx_pk_h = int(idxs[int(np.argmax(mag[idxs]))])
                harmonic_peak_points.append((float(f_hz[idx_pk_h]), float(mag[idx_pk_h])))
        n_h += 1

    band_total_sum = float(np.sum(mag[harmonic_union])) if np.any(harmonic_union) else 0.0
    band_peak_sum = float(np.sum([pt[1] for pt in harmonic_peak_points])) if harmonic_peak_points else 0.0
    band_remaining_sum = float(max(0.0, band_total_sum - band_peak_sum))
    if band_remaining_sum <= 0.0:
        pdr_h = float("inf") if band_peak_sum > 0 else float("nan")
    else:
        pdr_h = float(band_peak_sum / band_remaining_sum)

    fh_hz = float("nan")
    fh_amp = float("nan")
    if harmonic_peak_points:
        fh_hz = float(harmonic_peak_points[0][0])
        fh_amp = float(harmonic_peak_points[0][1])

    return {
        "fp_hz": float(fp_hz),
        "fp_amp": float(fp_amp),
        "fh_hz": float(fh_hz) if np.isfinite(fh_hz) else float("nan"),
        "fh_amp": float(fh_amp) if np.isfinite(fh_amp) else float("nan"),
        "PDR_G": float(pdr_g) if np.isfinite(pdr_g) else pdr_g,
        "PDR_L": float(pdr_l) if np.isfinite(pdr_l) else pdr_l,
        "PDR_H": float(pdr_h) if np.isfinite(pdr_h) else pdr_h,
        "df_hz": float(df_hz),
        "local_half_bw_hz": float(local_half_bw_hz),
        "fft_band_hz": np.array([float(max(0.0, fmin_hz)), float(fmax_eff)], dtype=float),
        "harmonic_fmax_hz": float(f_cap),
    }


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


def build_grad_aligned_segments(time, signals_dict, ref_key, pk_idx, fs, f_det, waves_per_rev, revs=3,
                                grad_left_frac=0.2, require_positive_grad=True):
    f_rev = f_det / float(waves_per_rev)
    samples_per_rev = max(1, int(round(fs / f_rev)))
    seg_len = revs * samples_per_rev
    X_rev = np.linspace(0.0, 2 * np.pi * revs, seg_len)

    ref = signals_dict[ref_key]
    dref = np.gradient(ref)
    w_left = max(1, int(round(grad_left_frac * samples_per_rev)))

    g_idx = []
    for p in pk_idx:
        a = max(0, p - w_left)
        b = p + 1
        if b <= a + 1:
            continue
        if require_positive_grad:
            loc = np.argmax(dref[a:b])
        else:
            loc = np.argmax(np.abs(dref[a:b]))
        g = a + loc
        start = g - samples_per_rev
        end = g + (revs - 1) * samples_per_rev
        if start >= 0 and end <= len(ref):
            g_idx.append((g, start, end))

    seg_bank = {}
    for k, sig in signals_dict.items():
        segs = [sig[s:e] for (_, s, e) in g_idx]
        seg_bank[k] = np.stack(segs, axis=0) if segs else np.empty((0, seg_len))

    return seg_bank, X_rev, samples_per_rev


def _log(msg: str, verbose: bool = True):
    if not verbose:
        return
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _select_reference_cycle(cycles_scaled: np.ndarray) -> int:
    norms = np.linalg.norm(cycles_scaled, axis=1)
    return int(np.argsort(norms)[len(norms) // 2]) if norms.size else 0


def _soft_dtw_distance(a: np.ndarray, b: np.ndarray, gamma: float) -> float:
    try:
        return float(soft_dtw(a, b, gamma=gamma))
    except Exception:
        try:
            return float(soft_dtw(a.reshape(-1, 1), b.reshape(-1, 1), gamma=gamma))
        except Exception:
            return float("nan")


def _soft_dtw_alignment_matrix(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray | None:
    try:
        align = soft_dtw_alignment(a, b, gamma=gamma)
    except Exception:
        try:
            align = soft_dtw_alignment(a.reshape(-1, 1), b.reshape(-1, 1), gamma=gamma)
        except Exception:
            return None
    if isinstance(align, (tuple, list)):
        align = align[0]
    try:
        align = np.asarray(align, dtype=np.float64)
    except Exception:
        return None
    if align.shape[0] == len(a) + 1 and align.shape[1] == len(b) + 1:
        align = align[1:, 1:]
    return align


def _warp_cycle_using_alignment(cycle: np.ndarray, ref_len: int, align: np.ndarray) -> np.ndarray:
    if align.shape[0] == len(cycle) and align.shape[1] == ref_len:
        weights = align
    elif align.shape[1] == len(cycle) and align.shape[0] == ref_len:
        weights = align.T
    else:
        if len(cycle) >= ref_len:
            return cycle[:ref_len].copy()
        pad = np.full(ref_len, cycle[-1] if len(cycle) else 0.0, dtype=np.float64)
        pad[: len(cycle)] = cycle
        return pad
    warped = np.empty(ref_len, dtype=np.float64)
    for j in range(ref_len):
        w = weights[:, j]
        s = float(np.sum(w))
        if s > 0:
            warped[j] = float(np.dot(w, cycle) / s)
        else:
            warped[j] = float(cycle[min(j, len(cycle) - 1)])
    return warped


def compute_soft_dtw_alignment(cycles: np.ndarray, gamma: float, progress_cb=None):
    if not TSLEARN_AVAILABLE:
        return None
    cycles_array = np.asarray(cycles, dtype=np.float64)
    if cycles_array.ndim != 2 or cycles_array.shape[0] == 0:
        return None

    cycle_means = np.mean(cycles_array, axis=1)
    cycle_stds = np.std(cycles_array, axis=1)
    cycle_stds_safe = np.where(cycle_stds > 0, cycle_stds, 1.0)

    scaler = TimeSeriesScalerMeanVariance()
    cycles_scaled = scaler.fit_transform(cycles_array[:, :, None])
    cycles_scaled_2d = cycles_scaled[:, :, 0]

    ref_index = _select_reference_cycle(cycles_scaled_2d)
    ref_cycle_scaled = cycles_scaled_2d[ref_index]
    ref_len = len(ref_cycle_scaled)

    distances = np.full(cycles_scaled_2d.shape[0], np.nan, dtype=float)
    aligned_cycles = np.zeros((cycles_scaled_2d.shape[0], ref_len), dtype=float)
    for idx, cycle_scaled in enumerate(cycles_scaled_2d):
        distances[idx] = _soft_dtw_distance(cycle_scaled, ref_cycle_scaled, gamma)
        align = _soft_dtw_alignment_matrix(cycle_scaled, ref_cycle_scaled, gamma)
        warped_scaled = cycle_scaled.copy() if align is None else _warp_cycle_using_alignment(cycle_scaled, ref_len, align)
        aligned_cycles[idx, :] = warped_scaled * cycle_stds_safe[idx] + cycle_means[idx]
        if progress_cb is not None:
            progress_cb(idx + 1, cycles_scaled_2d.shape[0])

    try:
        barycenter_scaled = softdtw_barycenter(cycles_scaled, gamma=gamma)
    except Exception:
        return None
    if barycenter_scaled.ndim == 2:
        barycenter_scaled = barycenter_scaled[:, 0]

    mean_mu = float(np.mean(cycle_means))
    mean_std = float(np.mean(cycle_stds_safe))
    barycenter_raw = barycenter_scaled * mean_std + mean_mu

    return {
        "ref_index": int(ref_index),
        "barycenter": np.asarray(barycenter_raw, dtype=float),
        "distances": distances,
        "warped_cycles": aligned_cycles,
    }


def _compute_soft_dtw_for_bank(seg_bank, phase, gamma, save_distances, save_warped, progress_every, verbose):
    n_pcb = len(PCB_AXIAL_LABELS)
    seg_len = int(len(phase))
    barycenter = np.full((n_pcb, seg_len), np.nan, dtype=float)
    n_cycles = np.zeros(n_pcb, dtype=int)
    dist_mean = np.full(n_pcb, np.nan, dtype=float)
    dist_std = np.full(n_pcb, np.nan, dtype=float)

    distances_by_pcb = {}
    warped_by_pcb = {}

    for i, pcb in enumerate(PCB_AXIAL_LABELS):
        segs = seg_bank.get(pcb, np.empty((0, seg_len)))
        n_cycles[i] = int(segs.shape[0]) if segs.ndim == 2 else 0
        if n_cycles[i] == 0:
            continue
        result = compute_soft_dtw_alignment(segs, gamma=gamma, progress_cb=None)
        if result is None:
            continue
        barycenter[i, :] = result["barycenter"]
        distances = np.asarray(result["distances"], dtype=float)
        if distances.size:
            dist_mean[i] = float(np.nanmean(distances))
            dist_std[i] = float(np.nanstd(distances))
        if save_distances:
            distances_by_pcb[pcb] = distances
        if save_warped:
            warped_by_pcb[pcb] = np.asarray(result["warped_cycles"], dtype=float)

    return {
        "phase": np.asarray(phase, dtype=float),
        "pcb_labels": np.array(PCB_AXIAL_LABELS, dtype=object),
        "barycenter": barycenter,
        "n_cycles": n_cycles,
        "distance_mean": dist_mean,
        "distance_std": dist_std,
        "distances_by_pcb": distances_by_pcb,
        "warped_by_pcb": warped_by_pcb,
    }


def _write_soft_dtw_artifact(out_path: str, soft_dtw_payload: dict, subdir: str):
    out_file = Path(out_path)
    target_dir = out_file.parent / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = target_dir / f"{out_file.stem}_soft_dtw.npz"

    npz_payload = {
        "phase": np.asarray(soft_dtw_payload["phase"], dtype=float),
        "pcb_labels": np.asarray(soft_dtw_payload["pcb_labels"], dtype=object),
        "barycenter": np.asarray(soft_dtw_payload["barycenter"], dtype=float),
        "n_cycles": np.asarray(soft_dtw_payload["n_cycles"], dtype=int),
        "distance_mean": np.asarray(soft_dtw_payload["distance_mean"], dtype=float),
        "distance_std": np.asarray(soft_dtw_payload["distance_std"], dtype=float),
    }
    for pcb, arr in soft_dtw_payload.get("distances_by_pcb", {}).items():
        npz_payload[f"distances_{pcb}"] = np.asarray(arr, dtype=float)
    for pcb, arr in soft_dtw_payload.get("warped_by_pcb", {}).items():
        npz_payload[f"warped_{pcb}"] = np.asarray(arr, dtype=float)
    np.savez_compressed(artifact_path, **npz_payload)
    return str(artifact_path)


def process_mat(mat_path, t_start, t_end, ref_key=None, f_det_in=None, revs=3,
                use_soft_dtw=False, soft_dtw_gamma=DEFAULT_SOFT_DTW_GAMMA,
                soft_dtw_save_distances=False, soft_dtw_save_warped=False,
                soft_dtw_subdir=DEFAULT_SOFT_DTW_SUBDIR, soft_dtw_progress_every=DEFAULT_SOFT_DTW_PROGRESS_EVERY,
                out_path=None, verbose=True):
    _log(f"Loading MAT: {os.path.basename(str(mat_path))}", verbose)
    mat = loadmat(mat_path)
    if "D" not in mat:
        raise ValueError("MAT file has no variable D.")
    D = mat["D"][0, 0]

    time = extract_time(D)
    if t_end <= t_start:
        raise ValueError(f"Invalid time window: start={t_start} end={t_end}")

    mask = (time >= t_start) & (time <= t_end)
    if not np.any(mask):
        raise ValueError("Selected time window has no samples.")

    _log(f"Windowing data: {t_start:.3f} to {t_end:.3f} s", verbose)
    window_data = {"time": time[mask]}
    pcb_data = {}
    for k in PCB_WINDOW_LABELS:
        data = extract_pcb(D, k)
        if data is None:
            continue
        if data.shape[0] != time.shape[0]:
            warnings.warn(f"{k} length mismatch (len={data.shape[0]} vs {time.shape[0]}). Skipping.")
            continue
        window_data[k] = data[mask]
        pcb_data[k] = data[mask]

    axial_keys = [k for k in PCB_AXIAL_LABELS if k in pcb_data]
    if not axial_keys:
        raise ValueError("No axial PCB data (pcb01..pcb13) found in the file.")

    if ref_key is None:
        # Prefer PCB02 as default timing/alignment reference when available.
        ref_key = "pcb02" if "pcb02" in axial_keys else axial_keys[0]
    elif ref_key not in axial_keys:
        warnings.warn(f"Requested ref '{ref_key}' not found in axial keys; using default.")
        ref_key = "pcb02" if "pcb02" in axial_keys else axial_keys[0]

    dt = float(np.mean(np.diff(window_data["time"])))
    fs = 1.0 / dt

    radial_keys = [k for k in ("pcb14", "pcb15") if k in pcb_data]
    f_det_source = "user_override" if (f_det_in is not None and f_det_in > 0) else "fft_estimate"
    if f_det_in is not None and f_det_in > 0:
        f_det = float(f_det_in)
    else:
        sources = [pcb_data[k] for k in radial_keys] if radial_keys else [pcb_data[ref_key]]
        f_det = estimate_det_freq(
            window_data["time"],
            sources,
            fmin=DEFAULT_FDET_MIN_HZ,
            fmax=DEFAULT_FDET_MAX_HZ,
        )
        if f_det is None:
            f_det = 6200.0
            f_det_source = "fallback_default"
    _log(f"Detected f_det={f_det:.1f} Hz ({f_det_source})", verbose)

    if 4000.0 <= f_det <= 7000.0:
        waves_per_rev = 1
    elif f_det > 7000.0:
        waves_per_rev = 2
    else:
        waves_per_rev = 1

    coh_fft_key = "pcb01" if "pcb01" in pcb_data else ref_key
    coherence_fft_struct = {}

    try:
        coherence_fft_struct = _compute_fft_coherence_struct(
            window_data["time"],
            pcb_data[coh_fft_key],
            fmin_hz=DEFAULT_FDET_MIN_HZ,
            fmax_hz=DEFAULT_FDET_MAX_HZ,
            local_half_bw_hz=DEFAULT_COH_FFT_LOCAL_HALF_BW_HZ,
            harmonic_fmax_hz=DEFAULT_COH_FFT_HARMONIC_FMAX_HZ,
        )
        coherence_fft_struct["signal_key"] = str(coh_fft_key)
    except Exception as exc:
        warnings.warn(f"Coherence FFT metrics failed: {type(exc).__name__}: {exc}")
        coherence_fft_struct = {
            "signal_key": str(coh_fft_key),
            "fp_hz": float("nan"),
            "fp_amp": float("nan"),
            "fh_hz": float("nan"),
            "fh_amp": float("nan"),
            "PDR_G": float("nan"),
            "PDR_L": float("nan"),
            "PDR_H": float("nan"),
            "df_hz": float("nan"),
            "local_half_bw_hz": float(DEFAULT_COH_FFT_LOCAL_HALF_BW_HZ),
            "fft_band_hz": np.array([float(DEFAULT_FDET_MIN_HZ), float(DEFAULT_FDET_MAX_HZ)], dtype=float),
            "harmonic_fmax_hz": float(DEFAULT_COH_FFT_HARMONIC_FMAX_HZ),
        }

    pk_idx = pick_phase_locked_peaks(
        window_data["time"],
        pcb_data[ref_key],
        f_det,
        last_win_s=DEFAULT_PEAK_LAST_WINDOW_S,
        height_frac=DEFAULT_PEAK_HEIGHT_FRAC,
        min_sep_frac=DEFAULT_PEAK_MIN_SEP_FRAC,
    )

    seg_bank, X_rev, samples_per_rev = build_grad_aligned_segments(
        window_data["time"],
        {k: pcb_data[k] for k in axial_keys},
        ref_key,
        pk_idx,
        fs,
        f_det,
        waves_per_rev,
        revs=revs,
        grad_left_frac=DEFAULT_ALIGN_GRAD_LEFT_FRAC,
        require_positive_grad=DEFAULT_ALIGN_REQUIRE_POSITIVE_GRAD,
    )
    _log("Built phase-aligned segments", verbose)

    n_segments = 0
    for k in seg_bank:
        n_segments = max(n_segments, seg_bank[k].shape[0])

    seg_len = X_rev.shape[0]
    seg_stack = np.full((n_segments, len(PCB_AXIAL_LABELS), seg_len), np.nan, dtype=float)
    for i, k in enumerate(PCB_AXIAL_LABELS):
        if k not in seg_bank:
            continue
        segs = seg_bank[k]
        if segs.shape[0] == 0:
            continue
        seg_stack[:segs.shape[0], i, :] = segs

    phase_avg = np.nanmean(seg_stack, axis=0) if seg_stack.size else np.empty((len(PCB_AXIAL_LABELS), seg_len))
    _log(f"Computed phase average ({n_segments} segments)", verbose)

    segment_struct = {
        "phase": X_rev,
        "data": seg_stack,
        "pcb_labels": np.array(PCB_AXIAL_LABELS, dtype=object),
    }

    phase_ave_struct = {
        "phase": X_rev,
        "data": phase_avg,
        "pcb_labels": np.array(PCB_AXIAL_LABELS, dtype=object),
    }
    for i, k in enumerate(PCB_AXIAL_LABELS):
        phase_ave_struct[k] = phase_avg[i, :]

    soft_dtw_struct = {}
    soft_dtw_status = "disabled"
    soft_dtw_artifact = ""
    if use_soft_dtw:
        if not TSLEARN_AVAILABLE:
            soft_dtw_status = "skipped_no_tslearn"
            _log("Soft-DTW requested but tslearn is unavailable. Skipping.", verbose)
        elif n_segments == 0:
            soft_dtw_status = "skipped_no_segments"
            _log("Soft-DTW skipped because no segments were extracted.", verbose)
        else:
            _log(f"Running Soft-DTW (gamma={soft_dtw_gamma:g})", verbose)
            soft_payload = _compute_soft_dtw_for_bank(
                seg_bank=seg_bank,
                phase=X_rev,
                gamma=float(soft_dtw_gamma),
                save_distances=bool(soft_dtw_save_distances),
                save_warped=bool(soft_dtw_save_warped),
                progress_every=int(max(1, soft_dtw_progress_every)),
                verbose=bool(verbose),
            )
            soft_dtw_status = "ok"
            soft_dtw_struct = {
                "phase": soft_payload["phase"],
                "pcb_labels": soft_payload["pcb_labels"],
                "barycenter": soft_payload["barycenter"],
                "n_cycles": soft_payload["n_cycles"],
                "distance_mean": soft_payload["distance_mean"],
                "distance_std": soft_payload["distance_std"],
            }
            n_soft_pcb = int(np.count_nonzero(np.asarray(soft_payload["n_cycles"]) > 0))
            if soft_dtw_save_distances:
                soft_dtw_struct["distances"] = {
                    pcb: soft_payload["distances_by_pcb"].get(pcb, np.array([], dtype=float))
                    for pcb in PCB_AXIAL_LABELS
                }
            if out_path is not None:
                soft_dtw_artifact = _write_soft_dtw_artifact(
                    out_path=str(out_path),
                    soft_dtw_payload=soft_payload,
                    subdir=str(soft_dtw_subdir),
                )
                _log(f"Soft-DTW artifact: {soft_dtw_artifact}", verbose)
            _log(f"Soft-DTW complete ({n_soft_pcb} PCB channels processed)", verbose)

    window_config = {
        "version": "1.1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_file": str(mat_path),
        "time_crop": {
            "start_s": float(t_start),
            "end_s": float(t_end),
            "n_samples": int(window_data["time"].shape[0]),
            "fs_hz": float(fs),
        },
        "channels": {
            "data": np.array([k for k in PCB_WINDOW_LABELS if k in pcb_data], dtype=object),
            "segment": np.array(PCB_AXIAL_LABELS, dtype=object),
        },
        "fdet": {
            "value_hz": float(f_det),
            "source": f_det_source,
            "fft_band_hz": np.array([DEFAULT_FDET_MIN_HZ, DEFAULT_FDET_MAX_HZ], dtype=float),
            "waves_per_rev": int(waves_per_rev),
        },
        "alignment": {
            "reference_sensor": str(ref_key),
            "method": "max_positive_gradient",
            "anchor_phase_rad": float(2.0 * np.pi),
            "grad_left_frac": float(DEFAULT_ALIGN_GRAD_LEFT_FRAC),
            "require_positive_grad": bool(DEFAULT_ALIGN_REQUIRE_POSITIVE_GRAD),
        },
        "peak_pick": {
            "last_window_s": float(DEFAULT_PEAK_LAST_WINDOW_S),
            "height_frac": float(DEFAULT_PEAK_HEIGHT_FRAC),
            "min_sep_frac": float(DEFAULT_PEAK_MIN_SEP_FRAC),
            "n_peaks": int(len(pk_idx)),
        },
        "segment": {
            "revs": int(revs),
            "phase_span_rad": float(2.0 * np.pi * revs),
            "samples_per_rev": int(samples_per_rev),
            "segment_length_samples": int(seg_len),
            "n_segments": int(n_segments),
            "data_order": "segment,pcb,sample",
        },
        "soft_dtw": {
            "enabled": bool(use_soft_dtw),
            "available": bool(TSLEARN_AVAILABLE),
            "status": soft_dtw_status,
            "gamma": float(soft_dtw_gamma),
            "saved_distances": bool(soft_dtw_status == "ok" and soft_dtw_save_distances),
            "saved_warped_cycles": bool(soft_dtw_status == "ok" and soft_dtw_save_warped),
            "progress_every": int(max(1, soft_dtw_progress_every)),
            "subfolder": str(soft_dtw_subdir),
            "artifact_file": str(soft_dtw_artifact),
        },
    }

    window_struct = {
        "config": window_config,
        "data": window_data,
        "segment": segment_struct,
        "phase_ave": phase_ave_struct,
        "coherence": {
            "fft": coherence_fft_struct,
        },
    }
    if soft_dtw_struct:
        window_struct["soft_dtw"] = soft_dtw_struct

    # Preserve all original fields from D, then append/overwrite D.window.
    D_new = loadmat(mat_path, simplify_cells=True)["D"]
    if not isinstance(D_new, dict):
        raise ValueError("Expected D to be a struct/dict when simplify_cells=True.")
    D_new["window"] = window_struct

    _log("Updated D.window payload", verbose)
    return D_new


def main():
    parser = argparse.ArgumentParser(description="Create windowed and segmented pressure MAT file.")
    parser.add_argument("--mat", nargs="*", help="Input MAT file path(s).")
    parser.add_argument("--dir", default=None, help="Directory to batch process.")
    parser.add_argument("--pattern", default="*.mat", help="Glob pattern for --dir (default: *.mat).")
    parser.add_argument("--t-start", type=float, default=DEFAULT_T_START, help="Window start time (s).")
    parser.add_argument("--t-end", type=float, default=DEFAULT_T_END, help="Window end time (s).")
    parser.add_argument(
        "--ref",
        default="pcb01",
        help="Reference PCB for alignment (e.g., pcb01). Default: auto (prefers pcb01).",
    )
    parser.add_argument("--fdet", type=float, default=None, help="Override detonation frequency (Hz).")
    parser.add_argument("--revs", type=int, default=2, help="Number of revolutions in segment (default 3 -> 6pi).")
    parser.add_argument("--use-soft-dtw", action="store_true", help="Enable optional Soft-DTW processing.")
    parser.add_argument("--soft-dtw-gamma", type=float, default=DEFAULT_SOFT_DTW_GAMMA, help="Soft-DTW gamma parameter.")
    parser.add_argument("--soft-dtw-save-distances", action="store_true", help="Store per-cycle Soft-DTW distances in MAT.")
    parser.add_argument("--soft-dtw-save-warped", action="store_true", help="Store warped cycles in artifact NPZ (can be large).")
    parser.add_argument("--soft-dtw-subdir", default=DEFAULT_SOFT_DTW_SUBDIR, help="Subfolder for Soft-DTW artifact output.")
    parser.add_argument("--soft-dtw-progress-every", type=int, default=DEFAULT_SOFT_DTW_PROGRESS_EVERY, help="Print warp progress every N cycles.")
    parser.add_argument("--quiet", action="store_true", help="Reduce processing updates.")
    parser.add_argument("--out", default=None, help="Output MAT file path (single file only). Default: <input>_processed.mat")
    args = parser.parse_args()

    mat_paths = []
    if args.mat:
        mat_paths.extend(args.mat)
    if args.dir:
        if not os.path.isdir(args.dir):
            raise FileNotFoundError(f"Directory not found: {args.dir}")
        mat_paths.extend(glob.glob(os.path.join(args.dir, args.pattern)))

    if not mat_paths:
        raise ValueError("No input files found. Use --mat or --dir/--pattern.")

    if len(mat_paths) > 1 and args.out is not None:
        raise ValueError("--out can only be used with a single input file.")

    total_files = len(mat_paths)
    for idx, mat_path in enumerate(mat_paths, start=1):
        if not os.path.isfile(mat_path):
            warnings.warn(f"Skipping missing file: {mat_path}")
            continue

        _log(f"[{idx}/{total_files}] Processing {os.path.basename(mat_path)}", verbose=not args.quiet)
        out_path = args.out
        if out_path is None:
            root, ext = os.path.splitext(mat_path)
            out_path = f"{root}_processed{ext or '.mat'}"

        D_new = process_mat(
            mat_path,
            t_start=float(args.t_start),
            t_end=float(args.t_end),
            ref_key=args.ref,
            f_det_in=args.fdet,
            revs=int(args.revs),
            use_soft_dtw=bool(args.use_soft_dtw),
            soft_dtw_gamma=float(args.soft_dtw_gamma),
            soft_dtw_save_distances=bool(args.soft_dtw_save_distances),
            soft_dtw_save_warped=bool(args.soft_dtw_save_warped),
            soft_dtw_subdir=str(args.soft_dtw_subdir),
            soft_dtw_progress_every=int(max(1, args.soft_dtw_progress_every)),
            out_path=out_path,
            verbose=not args.quiet,
        )

        savemat(out_path, {"D": D_new}, do_compression=True)
        _log(f"Wrote: {out_path}", verbose=not args.quiet)


if __name__ == "__main__":
    main()
