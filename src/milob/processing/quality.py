# processing/quality.py
import numpy as np
from typing import Optional, Tuple, Union
import xarray as xr
from ..processing.filters import frequency_filter
import warnings

def _snr_block(working_data: np.ndarray,
               freq_range: Optional[Tuple[float, float]],
               noise_range: Optional[Tuple[float, float]],
               fs: Optional[float],
               order: int,
               epsilon: float) -> np.ndarray:
    """Compute SNR for one (time, channel, wavelength) block."""
    if freq_range is not None:
        signal = frequency_filter(working_data, lowcut=freq_range[0], highcut=freq_range[1], fs=fs, order=order)

        if noise_range is not None:
            noise = frequency_filter(working_data, lowcut=noise_range[0], highcut=noise_range[1], fs=fs, order=order)
        else:
            # Everything left over once the signal band is removed
            noise = working_data - signal

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice')
            sig_std = np.nanstd(signal, axis=0)
            noise_std = np.nanstd(noise, axis=0)
            raw_std = np.nanstd(working_data, axis=0)

        snr = sig_std / (noise_std + epsilon)
        # Flat/dead channels (zero variance in the raw trace) are not simply
        # "low SNR" — flag them explicitly rather than reporting 0.
        snr[np.isinf(snr) | (raw_std == 0)] = np.nan
        return snr

    # Broadband mode
    with np.errstate(divide='ignore', invalid='ignore'), \
        warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice')

            mu = np.nanmean(np.abs(working_data), axis=0)
            sigma = np.nanstd(working_data, axis=0)
            snr = mu / (sigma + epsilon)

    snr[np.isinf(snr) | (sigma == 0)] = np.nan
    return snr


def compute_snr(data: Union[np.ndarray, xr.DataArray],
                datatype_idx: int = 0,
                freq_range: Optional[Tuple[float, float]] = None,
                noise_range: Optional[Tuple[float, float]] = None,
                fs: Optional[float] = None,
                order: int = 3,
                epsilon: float = 1e-12,
                window_length: Optional[float] = None,
                step_size: Optional[float] = None,
                return_windows: bool = False) -> np.ndarray:
    """
    Compute the signal-to-noise ratio of each channel.

    Without ``freq_range``, reports broadband SNR as mean(abs(x)) / std(x),
    which is meaningful only for data with a real baseline level such as raw
    intensity; a warning is raised for data that looks zero-mean. With
    ``freq_range``, reports the ratio of in-band to out-of-band standard
    deviation. Given ``window_length``, the estimate is made in sliding
    windows and the median across them returned, which separates stable
    coupling quality from transient motion.

    Parameters
    ----------
    data : np.ndarray or xr.DataArray
        Shape (time, channel, wavelength) with an optional trailing datatype
        axis.
    datatype_idx : int
        Index of the datatype to use. Default 0.
    freq_range : tuple of (low, high), optional
        Frequency band in Hz treated as signal.
    noise_range : tuple of (low, high), optional
        Frequency band in Hz treated as noise. Used only with ``freq_range``;
        defaults to the residual outside it.
    fs : float, optional
        Sampling frequency in Hz. Required for band-limited or windowed mode;
        read from the metadata for a DataArray.
    order : int
        Butterworth filter order for band-limited mode. Default 3.
    epsilon : float
        Constant added to the denominator to avoid division by zero on flat
        channels. Default 1e-12.
    window_length : float, optional
        Window length in seconds. Without it, one estimate is made over the
        whole trace.
    step_size : float, optional
        Window step in seconds. Default is 50% overlap.
    return_windows : bool
        Also return the per-window values. Default False.

    Returns
    -------
    snr : np.ndarray
        Shape (n_channels, n_wavelengths), the median across windows when
        windowed.
    windows : np.ndarray, optional
        Shape (n_windows, n_channels, n_wavelengths). Returned only when
        ``return_windows`` is True.
    """
    # Extract values
    values = data.values if isinstance(data, xr.DataArray) else data

    # 4D Shape Management (T, C, W, D)
    if values.ndim == 3:
        values = values[..., np.newaxis]

    # Select Datatype (T, C, W)
    working_data = values[..., datatype_idx]

    if freq_range is not None and noise_range is not None:
        # If the two bands overlap, "noise" partly or fully contains
        # "signal", so the ratio no longer compares signal to a disjoint
        # reference — it structurally undershoots 1 regardless of data
        # quality (e.g. noise_range=(0,3) with freq_range=(1,3) computes
        # var(1-3Hz) / var(0-3Hz), not signal vs. noise).
        if freq_range[0] < noise_range[1] and noise_range[0] < freq_range[1]:
            warnings.warn(
                f"freq_range={freq_range} overlaps noise_range={noise_range} — "
                "the noise band should be disjoint from the signal band, "
                "otherwise the ratio no longer reflects signal vs. noise. "
                "Leave noise_range=None to use the (disjoint by construction) "
                "residual instead.",
                RuntimeWarning
            )

    # Resolve sampling rate whenever we need seconds->samples (band-limited
    # or windowed mode)
    if (freq_range is not None or window_length is not None) and fs is None:
        if isinstance(data, xr.DataArray):
            fs = data.attrs.get('sampling_rate')
            if fs is None and 'time' in data.coords:
                fs = 1.0 / np.nanmean(np.diff(data.time.values))
        if fs is None:
            raise ValueError("Sampling rate (fs) must be provided or exist in DataArray attrs.")

    if window_length is None:
        snr = _snr_block(working_data, freq_range, noise_range, fs, order, epsilon)

        if freq_range is None:
            # Warn if the data looks zero-mean (OD/concentration-like) in
            # broadband, whole-trace mode — the one combination with no
            # rescue (no band, no windowing).
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Mean of empty slice')
                dc = np.nanmean(working_data, axis=0)
                sigma = np.nanstd(working_data, axis=0)
            if np.nanmedian(np.abs(dc)) < 0.1 * np.nanmedian(sigma):
                warnings.warn(
                    "Data appears zero-mean (e.g. OD or concentration) — broadband "
                    "SNR is not well-defined for zero-mean data. This metric is "
                    "intended for raw intensity; pass freq_range for a band-limited "
                    "ratio instead.",
                    RuntimeWarning
                )

        if return_windows:
            return snr, snr[np.newaxis, ...]
        return snr

    # Windowed mode
    T = working_data.shape[0]
    if step_size is None:
        step_size = window_length / 2

    window_samples = int(window_length * fs)
    step_samples = max(1, int(step_size * fs))

    window_values = []
    for start in range(0, T - window_samples + 1, step_samples):
        stop = start + window_samples
        segment = working_data[start:stop]
        window_values.append(_snr_block(segment, freq_range, noise_range, fs, order, epsilon))

    if not window_values:
        raise ValueError(
            f"window_length ({window_length}s) is longer than the data ({T / fs:.1f}s)."
        )

    windows = np.stack(window_values, axis=0)  # (n_windows, C, W)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered')
        snr = np.nanmedian(windows, axis=0)

    if return_windows:
        return snr, windows
    return snr


def compute_sci(data: Union[np.ndarray, xr.DataArray],
                freq_range: Tuple[float, float] = (0.5, 2.5),
                fs: Optional[float] = None,
                window_length = 5.0,
                step_size=None,
                min_cycles=3,
                return_windows: bool = False) -> np.ndarray:
    """
    Compute the Scalp Coupling Index of each channel.

    Correlates the two wavelengths of each source-detector pair in the cardiac
    band, in sliding windows. The median across windows measures stable
    coupling quality; the low tail of the window distribution reflects
    transient motion. Requires at least two wavelengths.

    Parameters
    ----------
    data : xr.DataArray
        Optical density with dims (time, channel, wavelength).
    freq_range : tuple of (low, high)
        Cardiac band in Hz. Default (0.5, 2.5).
    fs : float, optional
        Sampling frequency in Hz. Read from the metadata if omitted.
    window_length : float
        Window length in seconds.
    step_size : float
        Window step in seconds. Default is 50% overlap.
    min_cycles : int
        Minimum cardiac cycles required per window.
    return_windows : bool
        Also return the per-window correlations. Default False.

    Returns
    -------
    sci : np.ndarray
        Shape (n_channels,), the median across windows, or NaN where the index
        could not be computed.
    windows : np.ndarray, optional
        Shape (n_windows, n_channels), with NaN for skipped windows. Returned
        only when ``return_windows`` is True.

    Examples
    --------
    >>> sci = compute_sci(od_data)
    >>> bad_channels = od_data.channel.values[sci < 0.8]

    References
    ----------
    Pollonini, L. et al. (2014). Hearing Research, 309, 84-93.
    Pollonini, L., Bortfeld, H., & Oghalai, J. S. (2016). Biomedical Optics
    Express, 7(12), 5104-5119.
    """
    from scipy.signal import butter, filtfilt

    is_xarray = isinstance(data, xr.DataArray)

    # Metadata & Sampling Rate Extraction
    if is_xarray:
        if fs is None:
            fs = data.attrs.get('sampling_rate')
            if fs is None and 'time' in data.coords:
                fs = 1.0 / np.nanmean(np.diff(data.time.values))

        values = data.values
    else:
        values = data

    if fs is None:
        raise ValueError("Sampling rate (fs) must be provided or exist in DataArray attrs.")

    if values.ndim != 3:
        raise ValueError(f"Input must be 3D (time, channel, wavelength), got {values.ndim}D")
    working_data = values

    T, n_channels, n_wavelengths = working_data.shape

    if n_wavelengths < 2:
        raise ValueError("SCI requires at least 2 wavelengths")

    # Bandpass filter
    lowcut, highcut = freq_range
    #data_filtered = bandpass(working_data, lowcut, highcut, fs=fs, order=3)
    nyq = 0.5 * fs
    if highcut >= nyq:
        raise ValueError("Highcut exceeds Nyquist frequency.")
    b, a = butter(N=3, Wn=[lowcut / nyq, highcut / nyq], btype="band")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data_filtered = filtfilt(b, a, working_data, axis=0)

    # Sliding window
    min_window_length = min_cycles / lowcut
    if window_length < min_window_length:
        warnings.warn("Window may be too short for stable SCI estimation.",
        RuntimeWarning
    )
    window_samples = int(window_length * fs)

    if step_size is None:
        step_size = window_length / 2

    step_samples = int(step_size * fs)

    # Same start/stop grid for every channel, so the per-window array below
    # has a consistent shape (n_windows, n_channels) — skipped windows are
    # left as NaN rather than omitted, matching compute_snr's windowing.
    starts = list(range(0, T - window_samples + 1, step_samples))
    n_windows = len(starts)
    window_sci = np.full((max(n_windows, 1), n_channels), np.nan)

    for chn in range(n_channels):
        for w_idx, start in enumerate(starts):
            stop = start + window_samples
            segment = data_filtered[start:stop, chn, :]  # (window, W)

            # skip window if too many NaNs
            if np.sum(np.isfinite(segment)) < 0.8 * segment.size:
                continue

            # compute pairwise correlations across wavelengths
            pair_corrs = []
            for i in range(n_wavelengths):
                for j in range(i + 1, n_wavelengths):
                    sig_i = segment[:, i]
                    sig_j = segment[:, j]

                    valid = np.isfinite(sig_i) & np.isfinite(sig_j)

                    if np.sum(valid) < 0.8 * window_samples:
                        continue

                    if np.std(sig_i[valid]) == 0 or np.std(sig_j[valid]) == 0:
                        continue

                    r = np.corrcoef(sig_i[valid], sig_j[valid])[0, 1]
                    pair_corrs.append(r)

            if pair_corrs:
                window_sci[w_idx, chn] = np.nanmean(pair_corrs)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered')
        sci = np.nanmedian(window_sci, axis=0)

    if return_windows:
        return sci, window_sci
    return sci


def compute_psp(data: Union[np.ndarray, xr.DataArray],
                freq_range: Tuple[float, float] = (0.5, 2.5),
                fs: Optional[float] = None,
                window_length: float = 10.0,
                step_size=None,
                min_cycles: int = 3,
                return_windows: bool = False) -> np.ndarray:
    """
    Compute the Peak Spectral Power of each channel.

    Within each window, both wavelengths are band-passed to the cardiac band
    and normalised, their full cross-correlation function is computed, and the
    peak of that function's power spectrum is taken. Motion synchronised
    across wavelengths inflates the Scalp Coupling Index but spreads this
    spectral power, so the two metrics are intended to be applied together.

    Parameters
    ----------
    data : xr.DataArray
        Optical density or intensity with dims (time, channel, wavelength).
    freq_range : tuple of (low, high)
        Cardiac band in Hz. Default (0.5, 2.5).
    fs : float, optional
        Sampling frequency in Hz. Read from the metadata if omitted.
    window_length : float
        Window length in seconds. Default 10.0. Frequency resolution is
        approximately 1/window_length, so short windows blur the peak.
    step_size : float
        Window step in seconds. Default is 50% overlap.
    min_cycles : int
        Minimum cardiac cycles required per window.
    return_windows : bool
        Also return the per-window values. Default False.

    Returns
    -------
    psp : np.ndarray
        Shape (n_channels,), the median across windows.
    windows : np.ndarray, optional
        Shape (n_windows, n_channels). Returned only when ``return_windows``
        is True.

    References
    ----------
    Pollonini, L. et al. (2014). Hearing Research, 309, 84-93.
    Pollonini, L., Bortfeld, H., & Oghalai, J. S. (2016). Biomedical Optics
    Express, 7(12), 5104-5119.
    """
    from scipy.signal import butter, filtfilt, periodogram

    is_xarray = isinstance(data, xr.DataArray)
    if is_xarray:
        if fs is None:
            fs = data.attrs.get('sampling_rate')
            if fs is None and 'time' in data.coords:
                fs = 1.0 / np.nanmean(np.diff(data.time.values))
        values = data.values
    else:
        values = data

    if fs is None:
        raise ValueError("Sampling rate (fs) must be provided or exist in DataArray attrs.")

    if values.ndim != 3:
        raise ValueError(f"Input must be 3D (time, channel, wavelength), got {values.ndim}D")
    working_data = values

    T, n_channels, n_wavelengths = working_data.shape
    if n_wavelengths < 2:
        raise ValueError("PSP requires at least 2 wavelengths")

    lowcut, highcut = freq_range
    nyq = 0.5 * fs
    if highcut >= nyq:
        raise ValueError("Highcut exceeds Nyquist frequency.")
    b, a = butter(N=3, Wn=[lowcut / nyq, highcut / nyq], btype="band")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data_filtered = filtfilt(b, a, working_data, axis=0)

    min_window_length = min_cycles / lowcut
    if window_length < min_window_length:
        warnings.warn("Window may be too short for stable PSP estimation.", RuntimeWarning)
    window_samples = int(window_length * fs)
    if step_size is None:
        step_size = window_length / 2
    step_samples = int(step_size * fs)

    starts = list(range(0, T - window_samples + 1, step_samples))
    n_windows = len(starts)
    window_psp = np.full((max(n_windows, 1), n_channels), np.nan)

    for chn in range(n_channels):
        for w_idx, start in enumerate(starts):
            stop = start + window_samples
            segment = data_filtered[start:stop, chn, :]  # (window, W)

            if np.sum(np.isfinite(segment)) < 0.8 * segment.size:
                continue

            pair_peaks = []
            for i in range(n_wavelengths):
                for j in range(i + 1, n_wavelengths):
                    sig_i = segment[:, i]
                    sig_j = segment[:, j]

                    valid = np.isfinite(sig_i) & np.isfinite(sig_j)
                    if np.sum(valid) < 0.8 * window_samples:
                        continue

                    std_i, std_j = np.std(sig_i[valid]), np.std(sig_j[valid])
                    if std_i == 0 or std_j == 0:
                        continue

                    xi = sig_i[valid] / std_i
                    xj = sig_j[valid] / std_j
                    n = xi.size

                    # Full cross-correlation across all lags via FFT (circular,
                    # Wiener-Khinchin) rather than a linear/'full'-mode
                    # correlate, which tapers the correlation toward the edges
                    # and distorts the resulting spectrum's peak.
                    Xi, Xj = np.fft.fft(xi), np.fft.fft(xj)
                    xcorr = np.fft.ifft(Xi * np.conj(Xj)).real / n

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        freqs, pxx = periodogram(xcorr, fs=fs, scaling='spectrum', window='boxcar')

                    # Restrict the peak search to (a small margin around) the
                    # cardiac band, guarding against DC/edge leakage.
                    band_mask = (freqs >= max(lowcut - 0.1, 0)) & (freqs <= highcut + 0.1)
                    if not band_mask.any():
                        continue
                    pair_peaks.append(np.max(pxx[band_mask]))

            if pair_peaks:
                window_psp[w_idx, chn] = np.mean(pair_peaks)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered')
        psp = np.nanmedian(window_psp, axis=0)

    if return_windows:
        return psp, window_psp
    return psp


def combine_sci_psp(sci_windows: np.ndarray,
                     psp_windows: np.ndarray,
                     sci_threshold: float = 0.8,
                     psp_threshold: float = 0.1,
                     logic: str = 'or') -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine per-window SCI and PSP into a single pass or fail series.

    Both metrics must have been computed on the same window grid. The result
    cannot be reconstructed from each metric's failure rate separately, since
    that does not say how far the two failure sets overlap.

    Parameters
    ----------
    sci_windows, psp_windows : np.ndarray
        Per-window values of shape (n_windows, n_channels), computed with the
        same band, window length and step. NaN entries count as failing.
    sci_threshold : float
        SCI floor for a window to pass. Default 0.8.
    psp_threshold : float
        PSP floor for a window to pass. Default 0.1.
    logic : {'or', 'and'}
        'or' passes a window when either metric does; 'and' requires both.
        Default 'or'.

    Returns
    -------
    combined_pass : np.ndarray of bool
        Shape (n_windows, n_channels).
    frac_good : np.ndarray
        Fraction of windows passing, per channel.
    """
    if sci_windows.shape != psp_windows.shape:
        raise ValueError(
            f"sci_windows shape {sci_windows.shape} != psp_windows shape "
            f"{psp_windows.shape} — both must be computed with the same "
            "freq_range/window_length/step_size so their window grids align."
        )
    if logic not in ('or', 'and'):
        raise ValueError(f"logic must be 'or' or 'and', got {logic!r}")

    # NaN comparisons evaluate to False, so skipped windows naturally count
    # as failing that metric without special-casing.
    pass_sci = sci_windows > sci_threshold
    pass_psp = psp_windows > psp_threshold
    combined_pass = (pass_sci | pass_psp) if logic == 'or' else (pass_sci & pass_psp)

    frac_good = np.mean(combined_pass, axis=0)
    return combined_pass, frac_good


def quality_summary(values: np.ndarray,
                    channel_labels: np.ndarray,
                    threshold: Optional[float] = None,
                    metric_name: str = "Metric",
                    verbose: bool = False,
                    extra_columns: Optional[dict] = None,
                    is_short: Optional[np.ndarray] = None) -> dict:
    """
    Summarise a channel-wise quality metric.

    Parameters
    ----------
    values : np.ndarray
        Metric value per channel.
    channel_labels : np.ndarray
        Channel labels, aligned with ``values``.
    threshold : float, optional
        Channels below this are counted as bad. Without it, only descriptive
        statistics are reported.
    metric_name : str
        Name used in the report.
    verbose : bool
        Print the summary. Default True.
    extra_columns : dict of {str: np.ndarray}, optional
        Further per-channel covariates to report alongside ``values``. They
        never affect the good or bad classification. An all-NaN column is
        reported as such.
    is_short : np.ndarray of bool, optional
        Per-channel short or long classification. When given alongside
        ``threshold``, the good-channel count is split by distance class.

    Returns
    -------
    dict
        Descriptive statistics, the good and bad channel counts and labels,
        and any extra columns.
    """
    valid_mask = np.isfinite(values)
    n_total = len(values)
    n_valid = np.sum(valid_mask)
    vals_valid = values[valid_mask]
    extra_columns = extra_columns or {}

    if threshold is not None:
        good_mask = valid_mask & (values >= threshold)
        bad_mask = valid_mask & (values < threshold)
        n_good = np.sum(good_mask)
        n_bad = np.sum(bad_mask)
        pct_good = 100.0 * n_good / n_valid if n_valid > 0 else 0.0
        bad_channels = channel_labels[bad_mask].tolist()

        if is_short is not None:
            is_short = np.asarray(is_short, dtype=bool)
            n_good_short = int(np.sum(good_mask & is_short))
            n_good_long = int(np.sum(good_mask & ~is_short))
            n_short_total = int(np.sum(is_short))
            n_long_total = int(np.sum(~is_short))
        else:
            n_good_short = n_good_long = n_short_total = n_long_total = None
    else:
        n_good = n_bad = pct_good = None
        bad_channels = []
        n_good_short = n_good_long = n_short_total = n_long_total = None

    summary = {
        'metric': metric_name,
        'threshold': threshold,
        'n_total': n_total,
        'n_valid': n_valid,
        'n_good': n_good,
        'n_bad': n_bad,
        'pct_good': pct_good,
        'n_good_short': n_good_short,
        'n_good_long': n_good_long,
        'n_short_total': n_short_total,
        'n_long_total': n_long_total,
        'mean': np.mean(vals_valid) if n_valid > 0 else np.nan,
        'median': np.median(vals_valid) if n_valid > 0 else np.nan,
        'std': np.std(vals_valid) if n_valid > 0 else np.nan,
        'all_channel_labels': channel_labels,
        'all_values': values,
        'bad_channels': bad_channels,
        'extra_columns': extra_columns,
    }

    if verbose:
        header = f"{metric_name.upper()} QUALITY SUMMARY"
        print("=" * 70)
        print(f"{header:^70}")
        print("=" * 70)
        print(f"Threshold: {threshold if threshold is not None else 'none (informational only)'}")
        print(f"\nTotal channels:       {n_total}")
        print(f"Valid {metric_name} computed:   {n_valid}")
        if threshold is not None:
            print(f"Poor quality:         {n_bad} ({100.0 * n_bad / n_valid if n_valid > 0 else 0:.1f}%)")
            print(f"Good quality:         {n_good} ({pct_good:.1f}%)")
            if n_good_short is not None:
                # % here is of ALL short/long channels (good + bad), not of
                # n_good -- "how much of this distance class survived",
                # not "what fraction of the good set is short".
                pct_short = 100.0 * n_good_short / n_short_total if n_short_total > 0 else 0.0
                pct_long = 100.0 * n_good_long / n_long_total if n_long_total > 0 else 0.0
                print(f"  Good Short:          {n_good_short} ({pct_short:.1f}%)")
                print(f"  Good Long:           {n_good_long} ({pct_long:.1f}%)")
        print(f"\n{metric_name} statistics (across channels):")
        print(f"  Mean:   {summary['mean']:.3f}")
        print(f"  Median: {summary['median']:.3f}")
        print(f"  Std:    {summary['std']:.3f}")

        for name, vals in extra_columns.items():
            vals = np.asarray(vals, dtype=float)
            valid = np.isfinite(vals)
            print(f"\n{name} (across channels):")
            if valid.any():
                print(f"  Mean:   {np.mean(vals[valid]):.3f}")
                print(f"  Median: {np.median(vals[valid]):.3f}")
                print(f"  Max:    {np.max(vals[valid]):.3f}")
            else:
                print("  Not computed (no floor was given for this covariate)")

        if threshold is not None and n_bad > 0:
            print(f"\nBad channels ({metric_name} < {threshold}):")
            # Wrap labels for readability
            labels_str = ", ".join(summary['bad_channels'])
            print(f"  {labels_str}")

        print("=" * 70)

    return summary

        
def save_summary_to_file(summary: dict, filepath: str):
    """
    Write a quality summary to a CSV-style text file.

    Parameters
    ----------
    summary : dict
        Summary as returned by :func:`quality_summary`.
    filepath : str
        Destination path. Per-channel arrays in ``summary['extra_columns']``
        are written as additional columns, with their across-channel mean
        recorded in the header.
    """
    from datetime import datetime
    import os

    extra_columns = summary.get('extra_columns', {}) or {}

    # 1. Prepare Metadata Lines (starting with # for easy skipping in Python/Excel)
    metadata = [
        f"# Metric: {summary['metric']}",
        f"# Threshold: {summary['threshold']}",
        f"# Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Mean_{summary['metric']}: {summary['mean']:.4f}",
    ]
    if summary['threshold'] is not None:
        metadata.append(f"# Pct_Good: {summary['pct_good']:.1f}")
        if summary.get('n_good_short') is not None:
            metadata.append(f"# Good_Short: {summary['n_good_short']} / {summary['n_short_total']}")
            metadata.append(f"# Good_Long: {summary['n_good_long']} / {summary['n_long_total']}")
    for name, vals in extra_columns.items():
        vals = np.asarray(vals, dtype=float)
        valid = np.isfinite(vals)
        mean_val = np.mean(vals[valid]) if valid.any() else np.nan
        metadata.append(f"# Mean_{name}: {mean_val:.4f}")

    header = "channel_label,status,value" + "".join(f",{name}" for name in extra_columns)
    metadata.append(header)

    # 2. Prepare Channel Data
    # We reconstruct the status per channel for the table
    all_channels = summary.get('all_channel_labels', [])
    all_values = summary.get('all_values', [])

    rows = []
    for i, (label, val) in enumerate(zip(all_channels, all_values)):
        if np.isnan(val):
            status = "INVALID"
        elif summary['threshold'] is None:
            status = "N/A"
        else:
            status = "GOOD" if val >= summary['threshold'] else "BAD"
        row = f"{label},{status},{val:.4f}"
        for name in extra_columns:
            extra_val = np.asarray(extra_columns[name], dtype=float)[i]
            row += f",{extra_val:.4f}" if np.isfinite(extra_val) else ",NaN"
        rows.append(row)

    # 3. Write to file
    with open(filepath, 'w') as f:
        f.write("\n".join(metadata) + "\n")
        f.write("\n".join(rows))
        
