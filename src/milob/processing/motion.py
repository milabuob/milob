# processing/motion.py
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm
from typing import Optional, Union
import xarray as xr
import pywt


def tddr(data: Union[np.ndarray, xr.DataArray], 
         fs: Optional[float] = None,
         filter_cutoff: float = 0.5,
         filter_order: int = 3,
         tune: float = 4.685,
         max_iter: int = 50,
         add_high_freq: bool = True) -> Union[np.ndarray, xr.DataArray]:
    """
    Apply Temporal Derivative Distribution Repair (TDDR) motion correction.

    Separates the signal into low- and high-frequency components, applies
    robust regression to the temporal derivative of the low-frequency part to
    shrink outlying fluctuations, integrates the result, and optionally adds
    the high-frequency component back.

    Parameters
    ----------
    data : np.ndarray or xr.DataArray
        Time series of shape (time,) or (time, channels), or a DataArray
        carrying a 'time' dimension.
    fs : float, optional
        Sampling frequency in Hz. Required for a plain array; read from the
        metadata for a DataArray.
    filter_cutoff : float
        Frequency in Hz separating the low and high components. Default 0.5.
    filter_order : int
        Butterworth filter order. Default 3.
    tune : float
        Tukey biweight tuning constant, 4.685 giving 95% efficiency for
        Gaussian data. Default 4.685.
    max_iter : int
        Maximum iterations of the robust estimator. Default 50.
    add_high_freq : bool
        Add the uncorrected high-frequency component back. Default True.
        Setting it False corrects more aggressively and may remove neural
        signal.

    Returns
    -------
    np.ndarray or xr.DataArray
        Motion-corrected data, of the same shape and type as the input.
        Coordinates and attributes are preserved for a DataArray.

    Examples
    --------
    >>> corrected = tddr(data, fs=10.0)
    >>> corrected = tddr(data_xr)

    References
    ----------
    Fishburn, F. A., Ludlum, R. S., Vaidya, C. J., & Medvedev, A. V. (2019).
    Temporal derivative distribution repair (TDDR): A motion correction method
    for fNIRS. NeuroImage, 184, 171-179.
    """
    # Determine input type and extract data
    is_xarray = isinstance(data, xr.DataArray)
    
    if is_xarray:
        # Extract sampling rate from xarray
        if fs is None:
            fs = data.attrs.get('sampling_rate')
        if fs is None and 'time' in data.coords:
            time = data.coords['time'].values
            if len(time) > 1:
                fs = 1.0 / np.nanmean(np.diff(time))
        
        if fs is None:
            raise ValueError("Sampling rate (fs) could not be determined from DataArray.")

        values = data.values
        coords = data.coords
        dims = data.dims
        attrs = data.attrs.copy()
    else:
        # NumPy array
        if fs is None:
            raise ValueError("fs (sampling frequency) is required for numpy array input")      
        values = data
        
    # =============================================
    # Universal Shape Management
    # =============================================
    original_shape = values.shape
    n_time = original_shape[0]

    # Use a flexible reshape: keep time (axis 0), flatten everything else
    # This works perfectly for (T,), (T, C), (T, C, W), or (T, C, W, D)
    flat_values = values.reshape(n_time, -1)
    n_streams = flat_values.shape[1]
    
    values_corrected = np.zeros_like(flat_values)
    
    # =============================================
    # Core TDDR Algorithm
    # =============================================
    # Process each flattened stream independently
    for i in range(n_streams):
        signal = flat_values[:, i]
                
        # Step 0: Separate high and low frequencies
        Fc = filter_cutoff * 2.0 / fs  # Normalized frequency
        
        if Fc < 1.0:
            # Design and apply Butterworth lowpass filter
            b, a = butter(filter_order, Fc, btype='low')
            signal_low = filtfilt(b, a, signal)
        else:
            # If cutoff is above Nyquist, use original signal
            signal_low = signal.copy()
        
        signal_high = signal - signal_low
        
        # Step 1: Compute temporal derivative
        deriv = np.diff(signal_low)
        
        # Step 2: Initialize observation weights
        w = np.ones_like(deriv)
        
        # Step 3: Iterative robust weight estimation
        mu = np.inf
        D = np.sqrt(np.finfo(values.dtype).eps)
        
        for iteration in range(max_iter):
            mu_old = mu
            
            # Step 3a: Estimate weighted mean
            mu = np.sum(w * deriv) / np.sum(w)
            
            # Step 3b: Calculate absolute residuals
            dev = np.abs(deriv - mu)
            
            # Step 3c: Robust estimate of standard deviation (MAD-based)
            sigma = 1.4826 * np.median(dev)
            
            # Prevent division by zero
            if sigma < D:
                sigma = D
            
            # Step 3d: Scale deviations by sigma and tuning parameter
            r = dev / (sigma * tune)
            
            # Step 3e: Calculate weights using Tukey's biweight function
            # w(r) = (1 - r^2)^2 for |r| < 1, else 0
            w = np.where(r < 1, (1 - r**2)**2, 0)
            
            # Step 3f: Check convergence
            if np.abs(mu - mu_old) < D * max(np.abs(mu), np.abs(mu_old)):
                break
        
        # Step 4: Apply robust weights to centered derivative
        deriv_corrected = w * (deriv - mu)
        
        # Step 5: Integrate corrected derivative
        # Prepend zero to match original signal length
        signal_low_corrected = np.cumsum(np.concatenate([[0], deriv_corrected]))
        
        # Step 6: Center the corrected signal
        signal_low_corrected = signal_low_corrected - np.mean(signal_low_corrected)
        
        # After processing, store back in the flattened array
        # Optionally merge with high-frequency component
        values_corrected[:, i] = signal_low_corrected + (signal_high if add_high_freq else 0)
    
    # =============================================
    # Restore Original Shape and Type
    # =============================================
    final_values = values_corrected.reshape(original_shape)

    if is_xarray:
        corrected = xr.DataArray(
            final_values,
            coords=coords,
            dims=dims,
            attrs=attrs
        )
        corrected.attrs['transformation'] = 'tddr_motion_correction'
        return corrected

    return final_values


def _find_runs(mask: np.ndarray):
    """Return the contiguous True runs of a boolean array as (start, stop) pairs."""
    if not mask.any():
        return []
    diff = np.diff(mask.astype(np.int8))
    starts = list(np.where(diff == 1)[0] + 1)
    stops = list(np.where(diff == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        stops = stops + [len(mask)]
    return list(zip(starts, stops))


def spline_correction(data: Union[np.ndarray, xr.DataArray],
                       fs: Optional[float] = None,
                       k: Optional[int] = None,
                       gamma: float = 0.9,
                       smoothing: Optional[float] = None,
                       recenter: bool = True) -> Union[np.ndarray, xr.DataArray]:
    """
    Apply spline-interpolation motion correction (MARA).

    Flags motion-contaminated samples where the moving standard deviation
    exceeds a threshold, subtracts a cubic spline fitted to each flagged
    segment, and restores baseline continuity using the levels immediately
    before and after it.

    Parameters
    ----------
    data : np.ndarray or xr.DataArray
        Time series, with the same shape conventions as :func:`tddr`.
    fs : float, optional
        Sampling frequency in Hz. Read from the metadata for a DataArray.
    k : int, optional
        Half-width in samples of the moving-standard-deviation window, which
        spans 2k+1 samples. Default round(2.5 * fs), chosen so that
        heartbeat-scale fluctuations are not read as motion.
    gamma : float
        Threshold multiplier: samples are flagged where the moving standard
        deviation exceeds mean + gamma * std. Default 0.9.
    smoothing : float, optional
        Smoothing factor for the per-segment spline fit. Defaults to an
        automatic heuristic based on segment length and variance.
    recenter : bool
        Subtract each channel's mean over the whole recording after
        correction, giving an exactly zero-mean result. Default True. Set it
        False only when something downstream removes a constant level before
        the Beer-Lambert conversion, which has no centring step of its own.

    Returns
    -------
    np.ndarray or xr.DataArray
        Motion-corrected data, of the same type as the input.

    References
    ----------
    Scholkmann, F. et al. (2010). Physiological Measurement, 31(5), 649-662.
    Novi, S. L. et al. (2020). Neurophotonics, 7(1), 015001.
    """
    is_xarray = isinstance(data, xr.DataArray)

    if is_xarray:
        if fs is None:
            fs = data.attrs.get('sampling_rate')
        if fs is None and 'time' in data.coords:
            time = data.coords['time'].values
            if len(time) > 1:
                fs = 1.0 / np.nanmean(np.diff(time))
        if fs is None:
            raise ValueError("Sampling rate (fs) could not be determined from DataArray.")

        values = data.values
        coords = data.coords
        dims = data.dims
        attrs = data.attrs.copy()
    else:
        if fs is None:
            raise ValueError("fs (sampling frequency) is required for numpy array input")
        values = data

    if k is None:
        k = max(1, round(2.5 * fs))
    window = 2 * k + 1

    original_shape = values.shape
    n_time = original_shape[0]
    flat_values = values.reshape(n_time, -1)
    n_streams = flat_values.shape[1]

    values_corrected = flat_values.copy().astype(float)

    for i in range(n_streams):
        signal = values_corrected[:, i]

        # Moving standard deviation S(t) via a sliding-window E[x^2] - E[x]^2.
        # 'nearest' padding approximates the paper's t=k+1..N-k restriction
        # (undefined at the very edges) without discarding boundary samples.
        mean_x = uniform_filter1d(signal, size=window, mode='nearest')
        mean_x2 = uniform_filter1d(signal ** 2, size=window, mode='nearest')
        S = np.sqrt(np.maximum(mean_x2 - mean_x ** 2, 0.0))

        threshold = np.mean(S) + gamma * np.std(S)
        contaminated = S > threshold

        for start, stop in _find_runs(contaminated):
            seg = signal[start:stop].copy()
            n_seg = stop - start

            if n_seg <= 3:
                # Too short to fit a cubic spline meaningfully; leave as-is.
                continue

            t_seg = np.arange(n_seg)
            seg_smoothing = smoothing if smoothing is not None else n_seg * np.var(seg)
            try:
                spl = UnivariateSpline(t_seg, seg, k=3, s=seg_smoothing)
                trend = spl(t_seg)
            except Exception:
                # Degenerate segment (e.g. constant data); nothing to remove.
                trend = np.full(n_seg, np.mean(seg))

            residual = seg - trend

            # Flatten the whole segment to the level immediately preceding
            # it (a short reference window, or the segment's own first
            # sample if it starts at t=0) -- this removes the transition's
            # own shape entirely, not just realigns its starting point.
            ref_window = min(k, start) if start > 0 else 0
            if ref_window > 0:
                pre_level = np.mean(signal[start - ref_window:start])
            else:
                pre_level = residual[0]

            values_corrected[start:stop, i] = residual + pre_level

            # A moving-std detector only flags the transition itself, not a
            # sustained plateau that follows it (locally flat, so it never
            # crosses the threshold) -- so a baseline SHIFT (not just a
            # brief spike) leaves that whole plateau uncorrected unless we
            # explicitly compare the level just after the segment to the
            # level just before it (the paper's "before and after" wording)
            # and propagate the difference forward.
            post_window = min(k, n_time - stop)
            if post_window > 0:
                post_level = np.mean(signal[stop:stop + post_window])
            else:
                post_level = pre_level

            if stop < n_time:
                values_corrected[stop:, i] += (pre_level - post_level)

    if recenter:
        values_corrected -= np.nanmean(values_corrected, axis=0, keepdims=True)

    final_values = values_corrected.reshape(original_shape)

    if is_xarray:
        corrected = xr.DataArray(
            final_values,
            coords=coords,
            dims=dims,
            attrs=attrs
        )
        corrected.attrs['transformation'] = 'spline_motion_correction'
        return corrected

    return final_values


def wavelet_correction(data: Union[np.ndarray, xr.DataArray],
                        fs: Optional[float] = None,
                        wavelet: str = 'db2',
                        level: Optional[int] = None,
                        alpha: Optional[float] = None,
                        iqr: Optional[float] = None,
                        recenter: bool = True) -> Union[np.ndarray, xr.DataArray]:
    """
    Apply wavelet-based motion artifact removal.

    Decomposes the signal by discrete wavelet transform and models the detail
    coefficients at each level as a physiological distribution plus large
    motion outliers. Coefficients far out in the tail of a robustly fitted
    distribution are zeroed and the signal reconstructed. Approximation
    coefficients are never thresholded, so the input's mean passes through
    unchanged.

    Not NaN-safe: because the transform is a whole-signal filter bank, one NaN
    sample can propagate across an entire channel's output.

    Parameters
    ----------
    data : np.ndarray or xr.DataArray
        Time series, with the same shape conventions as :func:`tddr`.
    fs : float, optional
        Sampling frequency in Hz. Used only for DataArray metadata; the
        transform itself does not need it.
    wavelet : str
        PyWavelets wavelet name. Default 'db2'.
    level : int, optional
        Decomposition depth. Defaults to the maximum level available for this
        wavelet and signal length.
    alpha : float, optional
        Two-sided probability threshold: a coefficient is zeroed when its
        probability under the fitted physiological distribution falls below
        it. The distribution's scale is estimated by median absolute
        deviation. Mutually exclusive with ``iqr``; a value of 0.1 is used if
        neither is given.
    iqr : float, optional
        Tukey-fence multiplier: a coefficient is zeroed when it falls outside
        [Q1 - iqr*IQR, Q3 + iqr*IQR] at its level. Smaller values are more
        aggressive. Mutually exclusive with ``alpha``. This is a different
        rule rather than a reparameterised ``alpha``, since the fence is built
        from quartiles rather than from the median absolute deviation; the two
        agree only in the Gaussian limit.
    recenter : bool
        Subtract each channel's mean over the whole recording after
        correction. Default True.

    Returns
    -------
    np.ndarray or xr.DataArray
        Motion-corrected data, of the same type as the input.

    Raises
    ------
    ValueError
        If both ``alpha`` and ``iqr`` are given.

    References
    ----------
    Molavi, B., & Dumont, G. A. (2012). Physiological Measurement, 33(2),
    259-270.
    Novi, S. L. et al. (2020). Neurophotonics, 7(1), 015001.
    """
    if alpha is not None and iqr is not None:
        raise ValueError(
            "Pass either alpha (probability threshold) or iqr (Tukey-fence "
            "multiplier), not both -- they are two ways of specifying the "
            "same threshold and there is no sensible way to honour both."
        )
    if alpha is None and iqr is None:
        alpha = 0.1

    is_xarray = isinstance(data, xr.DataArray)

    if is_xarray:
        if fs is None:
            fs = data.attrs.get('sampling_rate')
        if fs is None and 'time' in data.coords:
            time = data.coords['time'].values
            if len(time) > 1:
                fs = 1.0 / np.nanmean(np.diff(time))

        values = data.values
        coords = data.coords
        dims = data.dims
        attrs = data.attrs.copy()
    else:
        values = data

    original_shape = values.shape
    n_time = original_shape[0]
    flat_values = values.reshape(n_time, -1)
    n_streams = flat_values.shape[1]

    values_corrected = np.zeros_like(flat_values, dtype=float)

    wav = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(n_time, wav.dec_len)
    use_level = min(level, max_level) if level is not None else max_level
    if use_level < 1:
        raise ValueError(
            f"Signal too short ({n_time} samples) for wavelet '{wavelet}' to "
            "decompose at even one level."
        )

    for i in range(n_streams):
        signal = flat_values[:, i]

        coeffs = pywt.wavedec(signal, wavelet, level=use_level)
        # coeffs = [approximation, detail_coarsest, ..., detail_finest]
        new_coeffs = [coeffs[0]]
        for detail in coeffs[1:]:
            if iqr is not None:
                q1, q3 = np.percentile(detail, [25, 75])
                fence = iqr * (q3 - q1)
                keep = (detail >= q1 - fence) & (detail <= q3 + fence)
            else:
                center = np.median(detail)
                mad = np.median(np.abs(detail - center))
                sigma = 1.4826 * mad
                if sigma < np.finfo(float).eps:
                    sigma = np.finfo(float).eps

                p = 2 * norm.sf(np.abs(detail - center) / sigma)
                keep = p >= alpha

            corrected_detail = np.where(keep, detail, 0.0)
            new_coeffs.append(corrected_detail)

        reconstructed = pywt.waverec(new_coeffs, wavelet)
        # waverec can return a signal 1 sample longer than the input
        # depending on wavelet/level/parity; trim to match.
        values_corrected[:, i] = reconstructed[:n_time]

    if recenter:
        values_corrected -= np.nanmean(values_corrected, axis=0, keepdims=True)

    final_values = values_corrected.reshape(original_shape)

    if is_xarray:
        corrected = xr.DataArray(
            final_values,
            coords=coords,
            dims=dims,
            attrs=attrs
        )
        corrected.attrs['transformation'] = 'wavelet_motion_correction'
        return corrected

    return final_values
