"""Heart-rate estimation from the cardiac pulsation in NIRS signals."""

import warnings
from typing import Optional, Tuple

import numpy as np

from .filters import frequency_filter


def _parabolic_peak_offset(power: np.ndarray, k: int) -> float:
    """
    Return the sub-bin offset of the spectral peak at ``k`` by parabolic interpolation.
    """
    if k <= 0 or k >= len(power) - 1:
        return 0.0
    y_m1, y_0, y_p1 = power[k - 1], power[k], power[k + 1]
    denom = y_m1 - 2 * y_0 + y_p1
    if denom == 0:
        return 0.0
    return float(np.clip(0.5 * (y_m1 - y_p1) / denom, -0.5, 0.5))


def estimate_hr_from_trace(trace: np.ndarray,
                            fs: float,
                            freq_range: Tuple[float, float] = (0.7, 2.2),
                            window_length: float = 10.0,
                            step_size: Optional[float] = None,
                            min_cycles: int = 3,
                            zero_pad: int = 4):
    """
    Estimate heart rate from one time series by sliding-window spectral peaks.

    The trace is band-pass filtered to ``freq_range``; in each window the
    power-spectrum peak in that band is located with parabolic sub-bin
    interpolation.

    Parameters
    ----------
    trace : np.ndarray, shape (time,)
        Time series carrying the cardiac pulsation, e.g. a short-channel
        average. Non-finite samples are interpolated before filtering.
    fs : float
        Sampling rate in Hz.
    freq_range : tuple of float
        Cardiac band in Hz. Default (0.7, 2.2), i.e. 42-132 bpm.
    window_length : float
        Window length in seconds. Default 10. Sets the frequency resolution,
        about ``1 / window_length`` Hz before interpolation.
    step_size : float, optional
        Step between windows in seconds. Default ``window_length / 2``.
        Estimates closer than ``window_length`` share samples and are
        correlated.
    min_cycles : int
        Cycles at the low band edge a window should hold; a shorter window
        raises a warning. Default 3.
    zero_pad : int
        FFT zero-padding factor. Default 4.

    Returns
    -------
    times : np.ndarray, shape (n_windows,)
        Window centres in seconds from the start of ``trace``.
    hr_bpm : np.ndarray, shape (n_windows,)
        Heart rate in beats per minute. NaN where fewer than 80% of a window's
        samples are finite or the band has no power.
    confidence : np.ndarray, shape (n_windows,)
        Fraction of the band power at the chosen peak, in [0, 1].

    Raises
    ------
    ValueError
        If the upper band edge is at or above the Nyquist frequency.
    """
    lowcut, highcut = freq_range
    nyq = 0.5 * fs
    if highcut >= nyq:
        raise ValueError("Highcut exceeds Nyquist frequency.")

    # Fill non-finite samples before zero-phase filtering; keep the original
    # mask for the per-window gap check.
    finite_mask = np.isfinite(trace)
    if np.all(finite_mask):
        trace_filled = trace
    elif np.sum(finite_mask) >= 2:
        idx = np.arange(len(trace))
        trace_filled = trace.copy()
        trace_filled[~finite_mask] = np.interp(idx[~finite_mask], idx[finite_mask], trace[finite_mask])
    else:
        trace_filled = np.nan_to_num(trace, nan=0.0, posinf=0.0, neginf=0.0)

    filtered = frequency_filter(trace_filled, lowcut=lowcut, highcut=highcut, fs=fs, order=3)

    min_window_length = min_cycles / lowcut
    if window_length < min_window_length:
        warnings.warn("Window may be too short for stable HR estimation.", RuntimeWarning)

    window_samples = int(window_length * fs)
    if step_size is None:
        step_size = window_length / 2
    step_samples = max(int(step_size * fs), 1)

    n = len(filtered)
    starts = list(range(0, n - window_samples + 1, step_samples))
    n_windows = len(starts)

    times = np.full(n_windows, np.nan)
    hr_bpm = np.full(n_windows, np.nan)
    confidence = np.full(n_windows, np.nan)

    if n_windows == 0:
        return times, hr_bpm, confidence

    nfft = int(window_samples * zero_pad)
    taper = np.hanning(window_samples)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    band_mask = (freqs >= lowcut) & (freqs <= highcut)
    band_idx = np.where(band_mask)[0]
    df = fs / nfft

    for w_idx, start in enumerate(starts):
        stop = start + window_samples
        segment = filtered[start:stop]
        times[w_idx] = (start + window_samples / 2) / fs

        valid = finite_mask[start:stop]
        if np.sum(valid) < 0.8 * window_samples:
            continue

        seg = np.where(valid, segment, np.mean(segment[valid]))
        seg = (seg - np.mean(seg)) * taper

        power = np.abs(np.fft.rfft(seg, n=nfft)) ** 2
        if band_idx.size == 0 or not np.any(power[band_idx] > 0):
            continue

        k = band_idx[np.argmax(power[band_idx])]
        delta = _parabolic_peak_offset(power, k)
        peak_freq = freqs[k] + delta * df
        hr_bpm[w_idx] = peak_freq * 60.0

        total_power = np.sum(power[band_mask])
        confidence[w_idx] = power[k] / total_power if total_power > 0 else np.nan

    return times, hr_bpm, confidence
