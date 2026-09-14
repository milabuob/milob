import inspect
import warnings
import numpy as np
from functools import lru_cache
import xarray as xr
from .base import BaseAnalysis
from .fc_meta import FC_UNITS_META, CHANNEL_QUALITY_COORDS

from ..outputs.output_conn import FCOutput

# ---------------------------------------------------------------------------
# FC method registry
#
# Each registered function computes one similarity metric between all valid
# channel pairs and returns:
#   matrix     : ndarray (n_chans, n_chans, n_types, *extra_dims)
#   extra_dims : dict {dim_name: coord_values} -- {} for static methods
#                (e.g. {'freq': freqs} for coherence, {'freq': freqs,
#                'time': times} for wavelet coherence)
#   value_meta : dict with a 'units' key (see fc_meta.FC_UNITS_META for the
#                registered value spaces and what each implies)
#
# A method may return a FOURTH element:
#   sidecars   : dict {name: ndarray (n_chans, n_chans, n_types)} -- per-pair
#                support/diagnostic arrays stored ALONGSIDE 'value' in the
#                output Dataset rather than folded into it. Currently used by
#                'pearson' for its per-pair effective DOF ('n_eff'), which is
#                genuinely pair-specific (it depends on BOTH channels' own
#                autocorrelation) and so cannot be reduced to the single attrs
#                scalar that Welch coherence records. FCOutput.average() and
#                to_dataframe() already carry anything alongside 'value'
#                through automatically, so nothing downstream needs a matching
#                entry. Methods with nothing to attach return the 3-tuple.
#
# New methods (coherence, wavelet coherence, ...) are added by writing one
# function and registering it here -- FC.fit() itself does not change.
#
# FC.fit() only computes the raw similarity matrix. Post-processing that
# depends on interpretation choices (discarding/absoluting negative values,
# sparsity/absolute thresholding, binarizing) lives on FCOutput instead --
# see FCOutput.handle_negatives() / FCOutput.threshold() -- so a fit only
# has to be run once and can be reinterpreted or re-thresholded repeatedly
# without recomputation.
# ---------------------------------------------------------------------------

_FC_METHODS = {}

# Parameters FC.fit() used to accept directly but now delegates to FCOutput.
# Silently swallowing these via **method_kwargs would be a real footgun (old
# code keeps running, but the requested post-processing silently never
# happens), so fail fast with a pointer to the replacement instead.
_REMOVED_FIT_PARAMS = {
    'handle_negatives': "Call FCOutput.handle_negatives(mode=...) on the fitted result instead.",
    'threshold_type':   "Call FCOutput.threshold(...) on the fitted result instead.",
    'threshold_val':    "Call FCOutput.threshold(...) on the fitted result instead.",
    'binarize':         "Call FCOutput.threshold(...) on the fitted result instead.",
}

# Parameters that belong to the ROI-averaging step (FCOutput.roi_average),
# not to a method function -- the same footgun as _REMOVED_FIT_PARAMS, but
# from the other direction. Every fit method's signature ends in a catch-all
# **kwargs (they must, since fit() forwards one shared **method_kwargs to
# whichever method is registered), so e.g. weights='snr' passed to
# Session.run_fc() would travel method_kwargs -> FC.fit -> _fit_pearson and
# vanish there: the fit succeeds, the numbers look plausible, and the
# requested SNR weighting simply never happened. Caught here rather than at
# each entry point because every path -- Session/Study.run_fc, a
# cross-stream fit, a direct FC(stream).fit() -- funnels
# through fit(), so one check covers all of them.
#
# Names come from FCOutput.roi_average's live signature (plus its own ROI
# argument), so adding a parameter there extends this guard for free.
def _misdirected_fit_params():
    from ..outputs.output_conn import roi_average_params
    hint = ("It configures ROI averaging, not the FC fit itself. Pass "
            "roi_kwargs={{'{name}': ...}} alongside roi_map= to run_fc(), or "
            "call FCOutput.roi_average(rois, {name}=...) on the fitted result.")
    return {name: hint.format(name=name)
            for name in ('rois', 'roi_map', *roi_average_params())}


def _accepted_params(fn):
    """Return the named parameters a function declares, excluding its **kwargs."""
    return {p.name for p in inspect.signature(fn).parameters.values()
            if p.kind is not inspect.Parameter.VAR_KEYWORD}


def _check_method_kwargs(method, compute_fn, method_kwargs):
    """
    Reject or warn about keyword arguments the chosen FC method will not read.

    A name no registered method accepts raises; a name another method accepts
    warns and is ignored, so one shared settings dict can be reused across
    methods.

    Parameters
    ----------
    method : str
        Name of the FC method being run.
    compute_fn : callable
        The registered function for that method.
    method_kwargs : dict
        Keyword arguments passed to it.

    Raises
    ------
    ValueError
        If a name is accepted by no registered method.
    """
    accepted = _accepted_params(compute_fn)
    unused = [k for k in method_kwargs if k not in accepted]
    if not unused:
        return

    known_elsewhere = set().union(*(
        _accepted_params(fn)
        for fn in (*_FC_METHODS.values(), *_FC_CROSS_METHODS.values())
    ))
    unknown = [k for k in unused if k not in known_elsewhere]
    # Only the data arguments fit() supplies positionally are never
    # user-facing; drop them from the "you could have meant" list.
    offerable = sorted(accepted - {'values', 'valid_indices', 'n_chans', 'n_types',
                                   'group_a', 'group_b', 'n_chans_a', 'n_chans_b'})
    # Some methods (e.g. pearson) have no tunable parameters at all -- say
    # that, rather than printing an empty list as if it were a menu.
    accepts = (f"{method!r} accepts: {offerable}" if offerable
               else f"{method!r} takes no fit parameters")

    if unknown:
        raise TypeError(
            f"FC.fit(method={method!r}) got unexpected keyword argument(s) "
            f"{unknown} -- no registered FC method accepts them, so they would "
            f"be silently ignored and the result computed without them. "
            f"{accepts}. Post-processing options are not fit parameters: pass "
            f"roi_kwargs={{...}} (with roi_map=) or reduce={{...}} to run_fc() "
            f"instead."
        )

    warnings.warn(
        f"FC.fit(method={method!r}): {sorted(unused)} belong(s) to a different "
        f"FC method and is ignored here -- {accepts}. Harmless if you are "
        f"reusing one kwargs dict across methods; a real problem if you "
        f"expected it to affect this fit.",
        UserWarning, stacklevel=3,
    )


def register_fc_method(name):
    def decorator(fn):
        _FC_METHODS[name] = fn
        return fn
    return decorator


# Cross-only counterparts of a subset of _FC_METHODS -- computed via
# FC.fit(method=..., other=...), see that docstring. Each registered
# function computes ONLY the pairs between two disjoint channel groups
# (never within either group) and returns a compact, generally non-square
# matrix:
#   matrix     : ndarray (n_chans_a, n_chans_b, n_types, *extra_dims)
# rather than _FC_METHODS' padded square (n_chans, n_chans, ...). This is
# the piece that genuinely can't be obtained from either group's own
# single-stream FC.fit() alone -- within-group blocks are identical to
# calling FC.fit() on that group's stream by itself, so there is no
# corresponding "self-cross" registration needed. Not every _FC_METHODS
# entry has a cross counterpart (yet); FC.fit(other=...) raises a clear
# error naming what's available if the requested method has none.
_FC_CROSS_METHODS = {}


def register_fc_cross_method(name):
    def decorator(fn):
        _FC_CROSS_METHODS[name] = fn
        return fn
    return decorator


def pearson_n_eff(block, max_lag=None, taper=True):
    """
    Estimate the effective number of independent samples behind each Pearson r.

    Haemodynamic series are strongly autocorrelated, so successive samples are
    not independent and the variance of r is set by an effective sample size
    rather than by the record length. Uses Bartlett's formula, with each
    channel's own autocorrelation, so the result is pair-specific.

    This describes the variance of r, not its bias: a shared systemic component
    inflates r itself, which no effective sample size corrects. Regress
    nuisance signals out first and estimate this from the residuals.

    Parameters
    ----------
    block : np.ndarray
        Shape (n_times, n_channels), the series actually being correlated.
    max_lag : int, optional
        Number of lags summed. Default ``n_times // 5``. Sample autocorrelation
        estimates grow noisy at long lags, so the sum must be truncated.
    taper : bool
        Apply a Bartlett lag window, downweighting the least reliable long
        lags. Default True.

    Returns
    -------
    np.ndarray
        Shape (n_channels, n_channels), clipped to [2, n_times].

    References
    ----------
    Afyouni, S., Smith, S. M., & Nichols, T. E. (2019). NeuroImage, 199,
    609-625.
    """
    n_times, n_channels = block.shape
    if max_lag is None:
        max_lag = n_times // 5
    max_lag = int(min(max(max_lag, 0), n_times - 1))
    if max_lag == 0:
        return np.full((n_channels, n_channels), float(n_times))

    # Autocovariance of every channel in one FFT pass: with x demeaned,
    # irfft(|rfft(x)|^2) gives sum_t x_t x_{t+k} at lag k. Zero-padded past
    # 2N so the circular correlation the FFT computes equals the linear one.
    x = block - block.mean(axis=0, keepdims=True)
    nfft = int(2 ** np.ceil(np.log2(2 * n_times)))
    spec = np.fft.rfft(x, n=nfft, axis=0)
    acov = np.fft.irfft(spec * np.conj(spec), n=nfft, axis=0)[:max_lag + 1]

    var = acov[0]
    with np.errstate(invalid='ignore', divide='ignore'):
        rho = acov / var                                    # (max_lag+1, C)
    # A flat channel has zero variance and no meaningful ACF; leave its lags
    # at 0 so it contributes n_eff = N rather than NaN-poisoning its pairs.
    rho[:, ~np.isfinite(var) | (var <= 0)] = 0.0
    rho = rho[1:]                                           # drop lag 0

    if taper:
        w = 1.0 - np.arange(1, max_lag + 1) / (max_lag + 1.0)
        weighted = rho * w[:, np.newaxis]
    else:
        weighted = rho

    # One matmul gives sum_k w_k rho_x(k) rho_y(k) for EVERY pair at once.
    pair_sum = weighted.T @ rho
    denom = 1.0 + 2.0 * pair_sum
    with np.errstate(invalid='ignore', divide='ignore'):
        n_eff = n_times / denom
    return np.clip(np.where(denom > 0, n_eff, 2.0), 2.0, float(n_times))


@register_fc_method('pearson')
def _fit_pearson(values, valid_indices, n_chans, n_types, n_eff_max_lag=None,
                 n_eff_taper=True, **kwargs):
    """
    Compute Pearson correlation between all channel pairs.

    Static, with no frequency or time axis. Also records the effective degrees
    of freedom behind each correlation, per pair alongside the values and as a
    median summary in the attributes.

    Parameters
    ----------
    values : np.ndarray
        Stream data to correlate.
    valid_indices : np.ndarray
        Indices of the channels to include.
    n_chans, n_types : int
        Channel and chromophore counts.
    n_eff_max_lag : int, optional
        Forwarded to :func:`pearson_n_eff`.
    n_eff_taper : bool
        Forwarded to :func:`pearson_n_eff`.

    Returns
    -------
    tuple
        The connectivity matrix, its extra dimensions and its value metadata.
    """
    matrix = np.full((n_chans, n_chans, n_types), np.nan)
    n_eff = np.full((n_chans, n_chans, n_types), np.nan)

    for t in range(n_types):
        clean_data = values[:, valid_indices, t]

        if clean_data.shape[1] > 1:
            corr = np.corrcoef(clean_data, rowvar=False)
            matrix[np.ix_(valid_indices, valid_indices, [t])] = corr[:, :, np.newaxis]
            dof = pearson_n_eff(clean_data, max_lag=n_eff_max_lag, taper=n_eff_taper)
            n_eff[np.ix_(valid_indices, valid_indices, [t])] = dof[:, :, np.newaxis]

    # Summary for check_poolable: the median over distinct pairs (the
    # diagonal is a channel against itself, and is dropped by FC.fit anyway).
    off_diag = ~np.eye(n_chans, dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        n_eff_summary = float(np.nanmedian(n_eff[off_diag]))

    return matrix, {}, {
        'units': 'r',
        'n_eff': n_eff_summary,
        'n_eff_max_lag': n_eff_max_lag,
        'n_eff_taper': n_eff_taper,
    }, {'n_eff': n_eff}


def welch_n_eff(window, nperseg, noverlap, n_times):
    """
    Estimate the effective number of independent Welch segments.

    Overlapping segments share samples, so K overlapping segments reduce
    variance by less than K would. Each lag is weighted by the squared
    normalised overlap of the analysis window, giving exactly K for
    non-overlapping segments.

    This sets the positive bias floor of magnitude-squared coherence, which is
    approximately 1/n_eff for independent signals, and therefore governs
    whether two coherence estimates are comparable.

    Parameters
    ----------
    window : str or array-like
        Analysis window, as passed to ``scipy.signal.coherence``.
    nperseg : int
        Segment length in samples.
    noverlap : int
        Overlap in samples, already resolved to an integer.
    n_times : int
        Samples in the record.

    Returns
    -------
    float
        Effective segment count, or NaN when no segment fits.

    References
    ----------
    Welch, P. (1967). IEEE Transactions on Audio and Electroacoustics, 15(2),
    70-73.
    Harris, F. J. (1978). Proceedings of the IEEE, 66(1), 51-83.
    """
    from scipy.signal import get_window

    w = (get_window(window, nperseg, fftbins=True) if isinstance(window, str)
         else np.asarray(window, dtype=float))
    step = nperseg - noverlap
    if step <= 0 or n_times < nperseg:
        return float('nan')

    n_seg = 1 + (n_times - nperseg) // step
    den = (w ** 2).sum() ** 2
    correlated = 0.0
    for m in range(1, n_seg):
        off = m * step
        if off >= nperseg:
            break
        correlated += (n_seg - m) * ((w[:nperseg - off] * w[off:]).sum() ** 2) / den
    return float(n_seg ** 2 / (n_seg + 2 * correlated))


@lru_cache(maxsize=32)
def wavelet_coherence_n_eff(wavelet='cmor1.5-1.0', dj=0.25, scale_smooth_octaves=0.6,
                            n_times=2048, n_channels=8, random_state=0):
    """
    Calibrate the effective number of independent averages behind a wavelet
    coherence estimate.

    Measured on white noise rather than derived, since the effective degrees of
    freedom follow from the combined time and scale smoothing and have no
    usable closed form. The result depends only on the wavelet, the scale
    spacing and the scale-smoothing width, not on sampling rate, record length
    or scale, which is what makes it cacheable.

    Wavelet coherence has a much lower effective count, and therefore a much
    higher bias floor, than a typical Welch configuration, so raw values from
    the two methods must not be pooled or compared.

    Parameters
    ----------
    wavelet : str
        Wavelet identifier used in the fit.
    dj : float
        Scale spacing in fractions of an octave.
    scale_smooth_octaves : float
        Width in octaves of the scale smoothing.
    n_times, n_channels : int
        Size of the calibration simulation.
    random_state : int
        Seed, so results and the cache are reproducible.

    Returns
    -------
    float
        Effective independent averages. Its reciprocal is the expected
        coherence between two independent signals.
    """
    rng = np.random.default_rng(random_state)
    values = rng.standard_normal((n_times, n_channels, 1))
    matrix, _, meta = _fit_wavelet_coherence(
        values, np.arange(n_channels), n_channels, 1, fs=1.0, wavelet=wavelet,
        dj=dj, s0=2.0, scale_smooth_octaves=scale_smooth_octaves,
    )
    scales, _, _, _, _ = _wavelet_coherence_grid(
        n_times, 1.0, wavelet, dj, meta['s0'], meta['j1'])

    # Restrict to scales whose smoothing kernel comfortably fits the record --
    # the same interior regime any real fit's usable (non-COI) band sits in.
    usable = scales < n_times / 20.0
    if not usable.any():
        raise ValueError(
            f"wavelet_coherence_n_eff: n_times={n_times} is too short to "
            f"calibrate dj={dj}, wavelet={wavelet!r}; raise n_times.")

    iu = np.triu_indices(n_channels, 1)
    block = matrix[iu[0], iu[1], 0][:, usable, :]
    valid = block[np.isfinite(block)]
    return float(1.0 / valid.mean())


@register_fc_method('coherence')
def _fit_coherence(values, valid_indices, n_chans, n_types, fs=None, nperseg=None,
                    noverlap=None, window='hann', detrend='constant', **kwargs):
    """
    Compute magnitude-squared coherence between all channel pairs.

    Uses Welch's method, adding a 'freq' axis. Collapse it with
    ``FCOutput.reduce(freq=...)`` before thresholding or building graphs.

    Parameters
    ----------
    values : np.ndarray
        Stream data to correlate.
    valid_indices : np.ndarray
        Indices of the channels to include.
    n_chans, n_types : int
        Channel and chromophore counts.
    fs : float
        Sampling rate in Hz. Filled in from the source stream when not given.
    nperseg : int, optional
        Welch segment length in samples, fixing the frequency resolution.
        Defaults to ``n_times // 8``, floored so that short recordings keep a
        usable segment length. It cannot be changed afterwards by ``reduce``.
    noverlap : int, optional
        Segment overlap in samples.
    window : str or array-like
        Analysis window.
    detrend : str or bool
        Detrending applied to each segment.

    Returns
    -------
    tuple
        The connectivity matrix, its extra dimensions and its value metadata.
    """
    from scipy.signal import coherence as _coherence

    if fs is None:
        raise ValueError(
            "FC.fit(method='coherence') requires a sampling rate: pass fs=... "
            "explicitly, or ensure the source stream's data.attrs['sampling_rate'] is set."
        )

    n_times = values.shape[0]
    if nperseg is None:
        nperseg = max(n_times // 8, 8)
    nperseg = int(min(nperseg, n_times))
    # scipy resolves noverlap=None to nperseg // 2 internally; record the
    # actually-resolved value (not None) so it's reproducible from attrs
    # alone, e.g. by FC.correct_bias() re-deriving surrogates identically.
    resolved_noverlap = noverlap if noverlap is not None else nperseg // 2

    # The Welch frequency grid is fixed by fs/nperseg alone (independent of
    # the signal values), so compute it directly rather than wasting a
    # dummy coherence() call just to discover it.
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    matrix = np.full((n_chans, n_chans, n_types, len(freqs)), np.nan)

    for t in range(n_types):
        for a_idx, i in enumerate(valid_indices):
            for j in valid_indices[a_idx + 1:]:
                _, cxy = _coherence(
                    values[:, i, t], values[:, j, t], fs=fs, nperseg=nperseg,
                    noverlap=noverlap, window=window, detrend=detrend,
                )
                matrix[i, j, t, :] = cxy
                matrix[j, i, t, :] = cxy

    return matrix, {'freq': freqs}, {
        'units': 'coherence',
        'fs': fs, 'nperseg': nperseg, 'noverlap': resolved_noverlap,
        'window': window, 'detrend': detrend,
        # Effective independent segment count -- what sets this estimate's
        # bias floor (~1/n_eff) and makes it comparable to another coherence
        # estimate. Recorded here so FCOutput.average() can check it rather
        # than silently pooling estimates with different DOF. Wavelet
        # coherence deliberately does NOT record one -- see
        # wavelet_coherence_n_eff().
        'n_eff': welch_n_eff(window, nperseg, resolved_noverlap, n_times),
    }


def _build_time_smoothing_kernels(time_sigma_samples):
    """Precompute the per-scale Gaussian time-smoothing kernels for one fit."""
    kernels = []
    for sigma in time_sigma_samples:
        sigma = max(float(sigma), 1e-6)
        radius = int(4.0 * sigma + 0.5)
        t = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (t / sigma) ** 2)
        kernel /= kernel.sum()
        kernels.append(kernel)
    return kernels


def _smooth_wavelet(arr, dj, scale_smooth_octaves, time_kernels):
    """
    Smooth a wavelet quantity in time and scale.

    Both are required for a coherence estimate below one, for the same reason
    Welch coherence needs several segments averaged. Time smoothing uses a
    Gaussian whose width follows the scale; scale smoothing uses a fixed
    boxcar across neighbouring scales.

    Parameters
    ----------
    arr : np.ndarray
        Complex array of shape (n_scales, n_times).
    dj : float
        Scale spacing in fractions of an octave.
    scale_smooth_octaves : float
        Width in octaves of the scale smoothing.
    time_kernels : list of np.ndarray
        Per-scale time-smoothing kernels.

    Returns
    -------
    np.ndarray
        Smoothed array, same shape as the input.
    """
    from scipy.signal import fftconvolve
    from scipy.ndimage import uniform_filter1d

    time_smoothed = np.empty_like(arr)
    for row in range(arr.shape[0]):
        kernel = time_kernels[row]
        time_smoothed[row] = (
            fftconvolve(arr[row].real, kernel, mode='same')
            + 1j * fftconvolve(arr[row].imag, kernel, mode='same')
        )

    n_scale_smooth = max(int(round(scale_smooth_octaves / dj)), 1)
    return (
        uniform_filter1d(time_smoothed.real, size=n_scale_smooth, axis=0, mode='nearest')
        + 1j * uniform_filter1d(time_smoothed.imag, size=n_scale_smooth, axis=0, mode='nearest')
    )


def _wtc_pair_coherence(Wi, Wj, smoothed_i, smoothed_j, scales, dj,
                         scale_smooth_octaves, time_kernels, coi_mask, mask_coi, out_dtype):
    """
    Compute wavelet coherence for one channel pair.

    Parameters
    ----------
    Wi, Wj : np.ndarray
        Wavelet coefficients for the two channels.
    smoothed_i, smoothed_j : np.ndarray
        Their smoothed auto-power.
    scales : np.ndarray
        Wavelet scales.
    dj : float
        Scale spacing in fractions of an octave.
    scale_smooth_octaves : float
        Width in octaves of the scale smoothing.
    time_kernels : list of np.ndarray
        Per-scale time-smoothing kernels.
    coi_mask : np.ndarray
        Cone-of-influence mask.
    mask_coi : bool
        Apply the mask.
    out_dtype : dtype
        Storage type for the result.

    Returns
    -------
    np.ndarray
        Coherence of shape (n_scales, n_times).
    """
    cross = (Wi * np.conj(Wj)) / scales[:, None]
    smoothed_cross = _smooth_wavelet(cross, dj, scale_smooth_octaves, time_kernels)

    with np.errstate(invalid='ignore', divide='ignore'):
        coh = np.abs(smoothed_cross) ** 2 / (smoothed_i * smoothed_j)
    coh = np.clip(coh, 0.0, 1.0)

    if mask_coi:
        coh = np.where(coi_mask, coh, np.nan)

    return coh.astype(out_dtype)


def _wavelet_coherence_grid(n_times, fs, wavelet, dj, s0, j1):
    """
    Build the scale, frequency and time grid, with the cone-of-influence mask.

    Parameters
    ----------
    n_times : int
        Samples in the record.
    fs : float
        Sampling rate in Hz.
    wavelet : str
        Wavelet identifier.
    dj : float
        Scale spacing in fractions of an octave.
    s0 : float
        Smallest scale in samples.
    j1 : int
        Number of scale steps.

    Returns
    -------
    tuple
        Scales, frequencies, times and the cone-of-influence mask.
    """
    import pywt

    dt = 1.0 / fs

    if s0 is None:
        s0 = 2.0  # ~Nyquist, in samples (pywt scales are sample-based, not seconds)
    if j1 is None:
        j1 = int(np.floor((1.0 / dj) * np.log2(max(n_times / s0, 1.0))))
    # Reversed (largest scale first): frequency is inversely proportional to
    # scale, so this makes the resulting freq coordinate increasing -- same
    # convention as _fit_coherence's np.fft.rfftfreq, which xarray's
    # .sel(freq=slice(low, high)) (used by FCOutput.reduce()) requires; a
    # decreasing coordinate silently returns an empty selection instead of
    # erroring, which is a much worse failure mode than getting this order right.
    scales = s0 * 2.0 ** (np.arange(j1, -1, -1) * dj)

    freqs = pywt.scale2frequency(wavelet, scales) / dt
    times = np.arange(n_times) * dt

    # Cone of influence: e-folding time (seconds) beyond which edge effects
    # corrupt the estimate at a given scale -- sqrt(2)*scale is the standard
    # approximation for the Morlet wavelet (Torrence & Compo 1998).
    coi_half_width = scales * dt * np.sqrt(2)
    edge_distance = np.minimum(times, times[-1] - times)
    coi_mask = edge_distance[None, :] >= coi_half_width[:, None]  # (n_scales, n_times), True = trustworthy

    # The scale grid (and therefore each scale's time-smoothing sigma) is
    # fixed for the whole fit -- build the smoothing kernels once and reuse
    # them across every channel and pair, rather than rebuilding per call.
    time_kernels = _build_time_smoothing_kernels(scales)

    return scales, freqs, times, coi_mask, time_kernels


@register_fc_method('wavelet_coherence')
def _fit_wavelet_coherence(values, valid_indices, n_chans, n_types, fs=None,
                            wavelet='cmor1.5-1.0', dj=0.25, s0=None, j1=None,
                            scale_smooth_octaves=0.6, mask_coi=True,
                            cwt_method='fft', out_dtype=None, n_jobs=1, **kwargs):
    """
    Compute wavelet transform coherence between all channel pairs.

    Resolves coupling in both frequency and time, adding 'freq' and 'time'
    axes. The scale grid spans the full range the recording can resolve, from
    near Nyquist down to about half the record length. Collapse it with
    ``FCOutput.reduce(freq=..., time=...)`` before thresholding or building
    graphs.

    Parameters
    ----------
    values : np.ndarray
        Stream data to correlate.
    valid_indices : np.ndarray
        Indices of the channels to include.
    n_chans, n_types : int
        Channel and chromophore counts.
    fs : float
        Sampling rate in Hz. Filled in from the source stream when not given.
    wavelet : str
        Complex wavelet identifier. Default 'cmor1.5-1.0'. Coherence needs the
        transform's phase, so the wavelet must be complex.
    dj : float
        Scale spacing in fractions of an octave. Default 0.25. Smaller values
        give a finer frequency grid at greater cost.
    s0 : float, optional
        Smallest scale in samples. Defaults to about Nyquist.
    j1 : int, optional
        Number of scale steps. Defaults to spanning about half the record.
    scale_smooth_octaves : float
        Width in octaves of the boxcar smoothing across neighbouring scales,
        required for a valid estimate. Default 0.6. Time smoothing is tied to
        each scale rather than being a separate parameter.
    mask_coi : bool
        Set estimates inside the cone of influence to NaN, where the wavelet's
        support extends beyond the record. Default True.
    cwt_method : {'fft', 'conv'}
        Transform method. Default 'fft'.
    out_dtype : dtype
        Storage type for the result. A lower precision bounds the memory of
        the full time-frequency matrix, which grows quadratically with channel
        count. The computation itself stays at full precision.
    n_jobs : int
        Parallel jobs across the pair loop. Default 1.

    Returns
    -------
    tuple
        The connectivity matrix, its extra dimensions and its value metadata.

    References
    ----------
    Torrence, C., & Compo, G. P. (1998). Bulletin of the American
    Meteorological Society, 79(1), 61-78.
    Grinsted, A., Moore, J. C., & Jevrejeva, S. (2004). Nonlinear Processes in
    Geophysics, 11, 561-566.
    """
    import pywt

    from ..processing.fitting import run_parallel_fits

    if fs is None:
        raise ValueError(
            "FC.fit(method='wavelet_coherence') requires a sampling rate: pass fs=... "
            "explicitly, or ensure the source stream's data.attrs['sampling_rate'] is set."
        )

    scales, freqs, times, coi_mask, time_kernels = _wavelet_coherence_grid(
        values.shape[0], fs, wavelet, dj, s0, j1
    )
    n_scales = len(scales)
    # float32 by default: coherence is bounded [0, 1] and already a
    # smoothed estimate, so this loses nothing scientifically meaningful
    # while halving the size of the largest array either function
    # allocates -- callers who specifically want float64 can still pass
    # out_dtype=np.float64 explicitly.
    out_dtype = out_dtype or np.float32

    matrix = np.full((n_chans, n_chans, n_types, n_scales, len(times)), np.nan, dtype=out_dtype)

    for t_idx in range(n_types):
        # CWT and smoothed auto-power depend only on the individual channel,
        # not its pair partner -- compute each once per channel and reuse
        # across all of that channel's pairs, rather than recomputing per pair.
        W, smoothed_auto = {}, {}
        for i in valid_indices:
            coeffs, _ = pywt.cwt(values[:, i, t_idx], scales, wavelet, sampling_period=1.0 / fs, method=cwt_method)
            W[i] = coeffs
            auto_power = (np.abs(coeffs) ** 2 / scales[:, None]).astype(complex)
            smoothed_auto[i] = _smooth_wavelet(auto_power, dj, scale_smooth_octaves, time_kernels).real

        pairs = [(i, j) for a_idx, i in enumerate(valid_indices) for j in valid_indices[a_idx + 1:]]
        jobs = [
            (_wtc_pair_coherence, (W[i], W[j], smoothed_auto[i], smoothed_auto[j], scales, dj,
                                    scale_smooth_octaves, time_kernels, coi_mask, mask_coi, out_dtype))
            for i, j in pairs
        ]
        for (i, j), coh in zip(pairs, run_parallel_fits(jobs, n_jobs=n_jobs, backend='threading')):
            matrix[i, j, t_idx, :, :] = coh
            matrix[j, i, t_idx, :, :] = coh

    return matrix, {'freq': freqs, 'time': times}, {
        'units': 'coherence',
        'fs': fs, 'wavelet': wavelet, 'dj': dj, 's0': s0, 'j1': j1,
        'scale_smooth_octaves': scale_smooth_octaves, 'mask_coi': mask_coi,
        'cwt_method': cwt_method, 'out_dtype': np.dtype(out_dtype).name,
    }


@register_fc_cross_method('wavelet_coherence')
def _fit_wavelet_coherence_cross(values, group_a, group_b, n_chans_a, n_chans_b, n_types, fs=None,
                                  wavelet='cmor1.5-1.0', dj=0.25, s0=None, j1=None,
                                  scale_smooth_octaves=0.6, mask_coi=True,
                                  cwt_method='fft', out_dtype=None, n_jobs=1, **kwargs):
    """
    Compute wavelet coherence between two groups of channels only.

    Returns a compact (n_channels_a, n_channels_b) matrix rather than a padded
    square one. Within-group blocks are not recomputed, being identical to
    fitting each group on its own.

    Parameters
    ----------
    values : np.ndarray
        Stream data holding both groups.
    group_a, group_b : array-like
        Channel indices forming each group.
    n_chans_a, n_chans_b : int
        Channel count in each group.
    n_types : int
        Number of chromophores or wavelengths.
    fs : float
        Sampling rate in Hz.
    wavelet : str
        Complex wavelet identifier.
    dj : float
        Scale spacing in fractions of an octave.
    s0 : float, optional
        Smallest scale in samples.
    j1 : int, optional
        Number of scale steps.
    scale_smooth_octaves : float
        Width in octaves of the scale smoothing.
    mask_coi : bool
        Set estimates inside the cone of influence to NaN.
    cwt_method : {'fft', 'conv'}
        Transform method.
    out_dtype : dtype
        Storage type for the result.
    n_jobs : int
        Parallel jobs across the pair loop. Default 1.
    **kwargs
        Ignored; accepted so one settings dict can serve several methods.

    Returns
    -------
    tuple
        The cross matrix, its extra dimensions and its value metadata.
    """
    import pywt

    from ..processing.fitting import run_parallel_fits

    if fs is None:
        raise ValueError(
            "FC.fit(method='wavelet_coherence', other=...) requires a sampling rate: "
            "pass fs=... explicitly, or ensure the source stream's data.attrs['sampling_rate'] is set."
        )

    scales, freqs, times, coi_mask, time_kernels = _wavelet_coherence_grid(
        values.shape[0], fs, wavelet, dj, s0, j1
    )
    n_scales = len(scales)
    # float32 by default: coherence is bounded [0, 1] and already a
    # smoothed estimate, so this loses nothing scientifically meaningful
    # while halving the size of the largest array either function
    # allocates -- callers who specifically want float64 can still pass
    # out_dtype=np.float64 explicitly.
    out_dtype = out_dtype or np.float32

    matrix = np.full((n_chans_a, n_chans_b, n_types, n_scales, len(times)), np.nan, dtype=out_dtype)

    for t_idx in range(n_types):
        W_a, smoothed_a = {}, {}
        for i in group_a:
            coeffs, _ = pywt.cwt(values[:, i, t_idx], scales, wavelet, sampling_period=1.0 / fs, method=cwt_method)
            W_a[i] = coeffs
            auto_power = (np.abs(coeffs) ** 2 / scales[:, None]).astype(complex)
            smoothed_a[i] = _smooth_wavelet(auto_power, dj, scale_smooth_octaves, time_kernels).real

        W_b, smoothed_b = {}, {}
        for j in group_b:
            coeffs, _ = pywt.cwt(values[:, j, t_idx], scales, wavelet, sampling_period=1.0 / fs, method=cwt_method)
            W_b[j] = coeffs
            auto_power = (np.abs(coeffs) ** 2 / scales[:, None]).astype(complex)
            smoothed_b[j] = _smooth_wavelet(auto_power, dj, scale_smooth_octaves, time_kernels).real

        pairs = [(a_pos, b_pos, i, j) for a_pos, i in enumerate(group_a) for b_pos, j in enumerate(group_b)]
        jobs = [
            (_wtc_pair_coherence, (W_a[i], W_b[j], smoothed_a[i], smoothed_b[j], scales, dj,
                                    scale_smooth_octaves, time_kernels, coi_mask, mask_coi, out_dtype))
            for _, _, i, j in pairs
        ]
        for (a_pos, b_pos, i, j), coh in zip(pairs, run_parallel_fits(jobs, n_jobs=n_jobs, backend='threading')):
            # Index by the channel's own position within its side's FULL
            # channel range (i for side A; j - n_chans_a for side B, since
            # group_b is offset by n_chans_a to index into the *combined*
            # values array FC._fit_cross() builds -- see that function's
            # own comment), NOT by a_pos/b_pos (i/j's position WITHIN
            # group_a/group_b, i.e. among only the VALID channels). matrix
            # is allocated at the FULL (n_chans_a, n_chans_b) size, not
            # (len(group_a), len(group_b)), so writing at a_pos/b_pos
            # silently compacted every valid channel's result toward the
            # start of the array whenever any channel was excluded
            # (is_bad/is_short), leaving that many rows/columns at the END
            # permanently NaN regardless of which channels were actually
            # excluded -- and writing at the UNADJUSTED j (still offset by
            # n_chans_a) would instead index past the end of matrix's
            # second axis entirely. a_pos/b_pos were only ever needed to
            # look up W_a/W_b/smoothed_a/smoothed_b (dicts keyed by i/j,
            # not position), already done above when `pairs`/`jobs` were built.
            matrix[i, j - n_chans_a, t_idx, :, :] = coh

    return matrix, {'freq': freqs, 'time': times}, {
        'units': 'coherence',
        'fs': fs, 'wavelet': wavelet, 'dj': dj, 's0': s0, 'j1': j1,
        'scale_smooth_octaves': scale_smooth_octaves, 'mask_coi': mask_coi,
        'cwt_method': cwt_method, 'out_dtype': np.dtype(out_dtype).name,
    }


# ---------------------------------------------------------------------------
# Bias-correction registry
#
# Each registered function generates a null distribution for one FC method
# one way, and returns (null_mean, null_std) shaped like that method's
# observed matrix. FC.correct_bias() then reports the observed statistic in
# units of that null -- units='null_z'.
#
# What the null is FOR differs by method, and both cases are worth stating
# because the arithmetic is identical while the reasoning is not:
#
#   coherence -- corrects a known POSITIVE BIAS. Coherence estimated from
#     finitely many Welch segments is nonzero even between unrelated
#     signals, and the size of that floor depends on the segment count, so
#     raw values are not comparable across recordings.
#   pearson -- corrects the null's WIDTH, not its centre. An r between
#     unrelated channels averages to ~0 whatever the settings, but
#     haemodynamic series are heavily autocorrelated, so r fluctuates far
#     more than the usual N-2 DOF assumes and ordinary significance testing
#     is badly anticonservative (Santosa et al. 2017). Phase-randomized
#     surrogates preserve each channel's own spectrum -- hence its
#     autocorrelation -- so the null they generate is exactly as wide as
#     chance coupling between two series with THESE spectra.
#
# Neither corrects a shared systemic component. Surrogates preserve each
# channel's spectrum, contamination included, and the observed statistic
# keeps the real phase-locked systemic coupling the surrogates destroy --
# so a systemic confound is reported as significant, correctly, because it
# IS coupling. Only nuisance regression separates it from neural coupling;
# no null can, from two channels alone.
#
# Registered per (correction method, FC method) pair, since generating a
# surrogate null is method-specific (a Welch coherence null and a Pearson
# null share `_phase_randomize` but nothing else), and each entry declares
# which fit attrs it needs replayed and whether it is frequency-resolved --
# so FC.correct_bias() itself does not special-case any method. Structured
# as a registry, same as _FC_METHODS, in case a second surrogate scheme
# (e.g. circular time-shift) is added later -- phase randomization is the
# more rigorous default for fNIRS's low-frequency signals (a randomly
# time-shifted slow oscillation can spuriously realign with another signal
# at particular shifts in a way phase randomization, which randomizes every
# frequency's phase independently, does not).
# ---------------------------------------------------------------------------

_BIAS_CORRECTION_METHODS = {}


def register_bias_correction_method(name, fc_method, required_attrs=(), needs_freq=False):
    """
    Register a surrogate null generator for one FC method.

    Registering per method rather than per scheme prevents a null valid for
    one method being run against another whose values share the same units.

    Parameters
    ----------
    name : str
        Surrogate scheme, as passed to ``FC.correct_bias(method=...)``.
    fc_method : str
        The FC method this null is valid for.
    required_attrs : tuple of str
        Fit settings read back from the output's attributes and forwarded, so
        surrogates are evaluated exactly as the real fit was.
    needs_freq : bool
        Whether the null is frequency-resolved, requiring the observed output
        to retain its 'freq' axis.

    Returns
    -------
    callable
        Decorator registering the function.
    """
    def decorator(fn):
        _BIAS_CORRECTION_METHODS[(name, fc_method)] = {
            'fn': fn,
            'required_attrs': tuple(required_attrs),
            'needs_freq': needs_freq,
        }
        return fn
    return decorator


# ---------------------------------------------------------------------------
# FC distance registry
#
# Unlike _FC_METHODS (raw stream data in, a connectivity matrix out), these
# compare two ALREADY-COMPUTED connectivity matrices -- e.g. for subject
# identification / fingerprinting (Novi et al. 2023, Neurophotonics
# 10(1):013510): "how similar are these two FC matrices?" Kept as a separate
# registry rather than folded into _FC_METHODS since the input/output shape
# is fundamentally different (two matrices in, one scalar out, vs. one raw
# stream in, one matrix out) -- consumed by FCOutput.distance().
# ---------------------------------------------------------------------------

_FC_DISTANCE_METHODS = {}


def register_fc_distance_method(name):
    def decorator(fn):
        _FC_DISTANCE_METHODS[name] = fn
        return fn
    return decorator


@register_fc_distance_method('geodesic')
def _geodesic_distance(c1, c2, alpha=0.5):
    """
    Compute the affine-invariant Riemannian distance between two matrices.

    Evaluated through the generalised eigenvalues of the matrix pencil, which
    avoids forming a matrix square root or logarithm explicitly. Both inputs
    are shrunk toward the identity first, guaranteeing positive-definiteness:
    a finite recording can leave a correlation matrix with near-zero
    eigenvalues, and those directions would otherwise dominate the distance
    with sampling noise.

    Parameters
    ----------
    c1, c2 : np.ndarray
        Square, symmetric, positive-definite matrices of shape (n, n).
    alpha : float
        Shrinkage toward the identity, in [0, 1). Default 0.5. Lower values
        shrink less and suit well-conditioned matrices only.

    Returns
    -------
    float
        The distance.

    Raises
    ------
    ValueError
        If either input is non-square, non-finite or asymmetric, or if
        ``alpha`` is outside [0, 1).

    References
    ----------
    Novi, S. L. et al. (2023). Neurophotonics, 10(1), 013510.
    Pennec, X., Fillard, P., & Ayache, N. (2006). International Journal of
    Computer Vision, 66(1), 41-66.
    """
    from scipy.linalg import eigh

    if not 0.0 <= alpha < 1.0:
        raise ValueError(
            f"geodesic distance requires alpha in [0, 1) -- got {alpha}. "
            f"alpha=0.5 (default) reproduces Novi et al. 2023; alpha=1 would "
            f"map every matrix to the identity and make every distance 0."
        )

    for name, c in (('c1', c1), ('c2', c2)):
        if c.ndim != 2 or c.shape[0] != c.shape[1]:
            raise ValueError(
                f"geodesic distance is defined on square symmetric positive-definite "
                f"matrices -- {name} has shape {c.shape}. A rectangular matrix (e.g. an "
                f"cross-stream block from FC.fit(other=...)) "
                f"is not one; see FCOutput.distance() for what this metric does and does "
                f"not apply to."
            )
        if not np.isfinite(c).all():
            raise ValueError(
                f"geodesic distance requires finite entries -- {name} contains NaN or inf. "
                f"FCOutput.distance() restores the NaN self-loop diagonal and rejects "
                f"bad/excluded channels before calling this."
            )
        if not np.allclose(c, c.T):
            raise ValueError(
                f"geodesic distance is defined on SYMMETRIC positive-definite matrices -- "
                f"{name} is not symmetric. This is most often a cross-stream block "
                f"from FC.fit(other=...): corr(A_i, B_j) != corr(A_j, B_i), "
                f"so it is a cross-covariance operator, not a point on the SPD manifold the "
                f"metric is defined on. Compare full within-stream correlation matrices "
                f"instead. Note this check is load-bearing: "
                f"scipy's eigh() reads only one triangle, so without it a non-symmetric "
                f"input returns a confident number computed from half the data."
            )

    n = c1.shape[0]
    eye = np.eye(n)
    c1_reg = (1.0 - alpha) * c1 + alpha * eye
    c2_reg = (1.0 - alpha) * c2 + alpha * eye
    eigvals = eigh(c2_reg, c1_reg, eigvals_only=True)
    # Guard against a log(<=0) from residual floating-point noise around a
    # theoretically-positive eigenvalue (e.g. two near-identical matrices).
    eigvals = np.clip(eigvals, 1e-12, None)
    return float(np.sqrt(np.sum(np.log(eigvals) ** 2)))


def _phase_randomize(x, rng):
    """
    Generate a phase-randomised surrogate of a real time series.

    Preserves the series' power spectrum, and therefore its autocorrelation,
    while randomising the phase at every frequency, which destroys any timing
    relationship with another signal.

    Parameters
    ----------
    x : np.ndarray
        Real one-dimensional series.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    np.ndarray
        Real-valued surrogate, the same length as ``x``.
    """
    n = len(x)
    X = np.fft.rfft(x)
    random_phases = rng.uniform(0, 2 * np.pi, size=X.shape)
    random_phases[0] = 0.0
    if n % 2 == 0:
        random_phases[-1] = 0.0
    return np.fft.irfft(np.abs(X) * np.exp(1j * random_phases), n=n)


@register_bias_correction_method(
    'phase_randomization', 'coherence',
    required_attrs=('fs', 'nperseg', 'noverlap', 'window', 'detrend'),
    needs_freq=True)
def _correct_bias_phase_randomization(values, valid_indices, n_chans, n_types, freqs,
                                       fs, nperseg, noverlap, window, detrend,
                                       n_surrogates, rng):
    from scipy.signal import coherence as _coherence

    n_freq = len(freqs)
    null_stack = np.full((n_surrogates, n_chans, n_chans, n_types, n_freq), np.nan)

    for s in range(n_surrogates):
        # One independent phase-randomized surrogate per channel per
        # iteration, reused across all of that channel's pairs -- far
        # cheaper than regenerating a fresh surrogate per specific pair,
        # and statistically equivalent (each pair still compares two
        # independently randomized signals).
        surrogate_values = values.copy()
        for i in valid_indices:
            for t in range(n_types):
                surrogate_values[:, i, t] = _phase_randomize(values[:, i, t], rng)

        for t in range(n_types):
            for a_idx, i in enumerate(valid_indices):
                for j in valid_indices[a_idx + 1:]:
                    _, cxy = _coherence(
                        surrogate_values[:, i, t], surrogate_values[:, j, t], fs=fs,
                        nperseg=nperseg, noverlap=noverlap, window=window, detrend=detrend,
                    )
                    null_stack[s, i, j, t, :] = cxy
                    null_stack[s, j, i, t, :] = cxy

    # Self-loops and excluded-channel entries stay NaN across every surrogate
    # iteration (same as the real fit), so those slices are legitimately
    # all-NaN here -- suppress the resulting (harmless) "empty slice" warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(null_stack, axis=0), np.nanstd(null_stack, axis=0)


@register_bias_correction_method('phase_randomization', 'pearson')
def _correct_bias_phase_randomization_pearson(values, valid_indices, n_chans, n_types,
                                               freqs, n_surrogates, rng):
    """
    Build a phase-randomised null for Pearson correlation.

    Corrects the null's width rather than a positive floor, since correlations
    between unrelated channels already average to zero. Cheap relative to the
    coherence null, since every pair is computed in one call per surrogate.

    Parameters
    ----------
    values : np.ndarray
        Stream data.
    valid_indices : np.ndarray
        Indices of the channels included.
    n_chans, n_types : int
        Channel and chromophore counts.
    freqs : None
        Unused; Pearson has no frequency axis.
    n_surrogates : int
        Number of surrogate iterations.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Mean and standard deviation of the null, per pair.
    """
    null_stack = np.full((n_surrogates, n_chans, n_chans, n_types), np.nan)
    if len(valid_indices) < 2:
        return np.nanmean(null_stack, axis=0), np.nanstd(null_stack, axis=0)

    for s in range(n_surrogates):
        for t in range(n_types):
            # One independent surrogate per channel, reused across all of
            # that channel's pairs -- as in the coherence null, and for the
            # same reason: each pair still compares two independently
            # randomized signals.
            surrogate = np.column_stack([
                _phase_randomize(values[:, i, t], rng) for i in valid_indices
            ])
            corr = np.corrcoef(surrogate, rowvar=False)
            null_stack[s][np.ix_(valid_indices, valid_indices, [t])] = corr[:, :, np.newaxis]

    # Excluded-channel entries stay NaN across every surrogate iteration
    # (same as the real fit), so those slices are legitimately all-NaN --
    # suppress the resulting (harmless) "empty slice" warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(null_stack, axis=0), np.nanstd(null_stack, axis=0)


class FC(BaseAnalysis):
    def __init__(self, dataset):
        super().__init__(dataset)


    def fit(self, method='pearson', correct_bias=False, n_surrogates=200,
            correction_method='phase_randomization', random_state=None,
            *, other=None, **method_kwargs):
        """
        Compute the connectivity matrix for this stream.

        No post-processing is applied; threshold or handle negative values on the
        returned output.

        Parameters
        ----------
        method : str
            Similarity metric, looked up in the FC method registry. Default
            'pearson'.
        correct_bias : bool
            Also run surrogate bias correction and return that result instead of
            the raw fit, giving units of 'null_z'. The raw estimate is not
            retained. Default False.
        n_surrogates : int
            Surrogate count, forwarded to :meth:`correct_bias`.
        correction_method : str
            Surrogate scheme, forwarded to :meth:`correct_bias`.
        random_state : int, optional
            Seed, forwarded to :meth:`correct_bias`.
        other : Datastream, optional
            A second stream to compute cross-connectivity against, so that only
            pairs spanning the two streams are computed. The result is generally
            not square, with channel_i from this stream and channel_j from
            ``other``. Requires a method with a registered cross-mode
            implementation, and both streams must share their time and chromophore
            axes. Not supported together with ``correct_bias``.
        **method_kwargs
            Forwarded to the registered method, e.g. ``fs`` or ``nperseg``. A name
            no method accepts raises; a name belonging to another method warns and
            is ignored.

        Returns
        -------
        FCOutput
            The connectivity result.

        Raises
        ------
        ValueError
            If ``method`` is unregistered, if ``other`` is given for a method with
            no cross-mode implementation, or if ``other`` and ``correct_bias`` are
            combined.
        """
        for removed, hint in _REMOVED_FIT_PARAMS.items():
            if removed in method_kwargs:
                raise TypeError(f"FC.fit() no longer accepts '{removed}'. {hint}")

        for stray, hint in _misdirected_fit_params().items():
            if stray in method_kwargs:
                raise TypeError(f"FC.fit() does not accept '{stray}'. {hint}")

        if other is not None:
            if correct_bias:
                raise ValueError(
                    "FC.fit(other=..., correct_bias=True) is not supported -- "
                    "bias correction is not yet implemented for the cross-only path."
                )
            if method not in _FC_CROSS_METHODS:
                raise ValueError(
                    f"FC.fit(method={method!r}, other=...) has no cross-mode "
                    f"implementation. Available for cross computation: "
                    f"{sorted(_FC_CROSS_METHODS)}."
                )
            return self._fit_cross(other, method, method_kwargs)

        if method not in _FC_METHODS:
            raise ValueError(
                f"Unknown FC method '{method}'. Available: {sorted(_FC_METHODS)}"
            )

        # Extract data: (Time, Channel, Chromophore)
        data_obj = self.dataset.data
        values = data_obj.values

        # Handle 4D (TD-NIRS) by averaging bins or selecting one
        if len(data_obj.dims) == 4:
            values = np.mean(values, axis=3)
        n_times, n_chans, n_types = values.shape

        # Identify valid channels
        is_bad = data_obj.coords.get('is_bad', np.zeros(n_chans, dtype=bool))
        is_short = data_obj.coords.get('is_short', np.zeros(n_chans, dtype=bool))
        valid_mask = ~(is_bad | is_short)
        valid_indices = np.where(valid_mask)[0]

        compute_fn = _FC_METHODS[method]
        # Before the fs auto-fill below, so only what the CALLER actually
        # passed is judged (an auto-filled fs is never a user mistake).
        _check_method_kwargs(method, compute_fn, method_kwargs)

        # Auto-fill fs from the stream's own sampling rate for methods that
        # declare an `fs` parameter (e.g. coherence) and weren't given one
        # explicitly. Gated on the method actually having an `fs` parameter
        # so e.g. Pearson's history/config doesn't pick up an irrelevant fs.
        if 'fs' not in method_kwargs and 'fs' in inspect.signature(compute_fn).parameters:
            fs_attr = data_obj.attrs.get('sampling_rate')
            if fs_attr is not None:
                method_kwargs = {**method_kwargs, 'fs': fs_attr}

        result = compute_fn(
            values, valid_indices, n_chans, n_types, **method_kwargs
        )
        # Optional 4th element: per-pair sidecar arrays stored alongside
        # 'value' (see the FC method registry docstring). Methods with
        # nothing to attach return the 3-tuple.
        matrix, extra_dims, value_meta = result[:3]
        sidecars = dict(result[3]) if len(result) > 3 else {}

        # Remove self-loops (diagonal = NaN), regardless of any trailing
        # freq/time dims a given method may have added
        diag_idx = np.arange(n_chans)
        matrix[diag_idx, diag_idx, ...] = np.nan
        # Same for the sidecars: a channel's DOF against itself is not a
        # quantity anyone should read, and leaving it finite would make the
        # diagonal look like real support behind a NaN value.
        for arr in sidecars.values():
            arr[diag_idx, diag_idx, ...] = np.nan

        config = {
            'method': method, 'source_stream_name': self.dataset.name,
            **method_kwargs,
        }
        raw_output = self._package_results(matrix, extra_dims, value_meta, config,
                                           sidecars=sidecars)

        if not correct_bias:
            return raw_output
        return self.correct_bias(
            raw_output, method=correction_method,
            n_surrogates=n_surrogates, random_state=random_state,
        )


    def _fit_cross(self, other, method, method_kwargs):
        """
        Compute cross-connectivity against a second stream.

        Parameters
        ----------
        other : Datastream
            The second stream.
        method : str
            FC method, which must have a registered cross-mode implementation.
        method_kwargs : dict
            Forwarded to that implementation.

        Returns
        -------
        FCOutput
            Non-square result, with channel_i from this stream and channel_j from
            ``other``.
        """
        data_a = self.dataset.data
        data_b = other.data

        values_a = data_a.values
        values_b = data_b.values
        if len(data_a.dims) == 4:
            values_a = np.mean(values_a, axis=3)
        if len(data_b.dims) == 4:
            values_b = np.mean(values_b, axis=3)

        n_times_a, n_chans_a, n_types = values_a.shape
        n_times_b, n_chans_b, _ = values_b.shape
        if n_times_a != n_times_b:
            raise ValueError(
                f"FC.fit(other=...) requires both streams to share the same number "
                f"of time samples; got {n_times_a} vs {n_times_b}."
            )

        # Matching lengths (n_types == n_types_b) alone doesn't guarantee
        # the two streams mean the same thing by that axis -- one could be
        # 'chromophore' and the other 'wavelength', or both 'wavelength'
        # but at different actual wavelengths/order (e.g. two participants'
        # devices configured differently). Either would still pass a
        # length-only check and silently mislabel/misalign the result.
        type_dim_a = 'chromophore' if 'chromophore' in data_a.dims else 'wavelength'
        type_dim_b = 'chromophore' if 'chromophore' in data_b.dims else 'wavelength'
        if type_dim_a != type_dim_b:
            raise ValueError(
                f"FC.fit(other=...) requires both streams to use the same "
                f"chromophore/wavelength axis -- got '{type_dim_a}' vs '{type_dim_b}'."
            )
        if not np.array_equal(data_a.coords[type_dim_a].values, data_b.coords[type_dim_b].values):
            raise ValueError(
                f"FC.fit(other=...) requires both streams' '{type_dim_a}' coordinate "
                f"values (and order) to match -- got {list(data_a.coords[type_dim_a].values)} "
                f"vs {list(data_b.coords[type_dim_b].values)}."
            )

        is_bad_a = data_a.coords.get('is_bad', np.zeros(n_chans_a, dtype=bool))
        is_short_a = data_a.coords.get('is_short', np.zeros(n_chans_a, dtype=bool))
        valid_a = np.where(~(is_bad_a | is_short_a))[0]

        is_bad_b = data_b.coords.get('is_bad', np.zeros(n_chans_b, dtype=bool))
        is_short_b = data_b.coords.get('is_short', np.zeros(n_chans_b, dtype=bool))
        valid_b = np.where(~(is_bad_b | is_short_b))[0]

        # One combined (time, channel, type) array so the cross compute_fn
        # sees the same layout the intra-brain path does -- group_a/group_b
        # are positions within it, not each stream's own local indices.
        combined = np.concatenate([values_a, values_b], axis=1)
        group_a = valid_a
        group_b = valid_b + n_chans_a

        compute_fn = _FC_CROSS_METHODS[method]
        _check_method_kwargs(method, compute_fn, method_kwargs)

        if 'fs' not in method_kwargs and 'fs' in inspect.signature(compute_fn).parameters:
            fs_attr = data_a.attrs.get('sampling_rate')
            if fs_attr is not None:
                method_kwargs = {**method_kwargs, 'fs': fs_attr}

        matrix, extra_dims, value_meta = compute_fn(
            combined, group_a, group_b, n_chans_a, n_chans_b, n_types, **method_kwargs
        )

        config = {
            'method': method, 'source_stream_name': self.dataset.name,
            'other_stream_name': other.name, **method_kwargs,
        }
        return self._package_cross_results(matrix, data_a, data_b, other.history, extra_dims, value_meta, config)


    def correct_bias(self, fc_output, method='phase_randomization', n_surrogates=200, random_state=None):
        """
        Re-express a connectivity estimate against a surrogate null distribution.

        Returns (observed - null mean) / null standard deviation per pair, and per
        frequency where the method has one, with units of 'null_z'. For coherence
        this removes the positive bias floor left by a finite segment count; for
        Pearson it corrects the width of the null, which the conventional degrees
        of freedom understate for autocorrelated series.

        It does not correct for a shared systemic component: surrogates preserve
        each channel's spectrum, contamination included, so systemic coupling
        still emerges as significant. Regress nuisance signals out first.

        Requires the original stream, since a surrogate is built by
        phase-randomising the per-channel series and recomputing the statistic.
        The coherence null costs one coherence call per pair per surrogate, so it
        is substantially more expensive than the Pearson null.

        Parameters
        ----------
        fc_output : FCOutput
            The already-fitted output to correct. Its method selects the
            registered null, and the fit settings that null needs are read back
            from its attributes.
        method : str
            Surrogate scheme, looked up in the bias-correction registry. Default
            'phase_randomization'.
        n_surrogates : int
            Number of surrogate iterations averaged over. Default 200.
        random_state : int, optional
            Seed for reproducibility. None draws differently each call.

        Returns
        -------
        FCOutput
            Corrected output with ``units='null_z'``.

        Raises
        ------
        ValueError
            If no null is registered for the output's method, or if the null is
            frequency-resolved and the output has already been reduced.
        """
        attrs = fc_output.output.attrs
        fc_method = attrs.get('method')
        spec = _BIAS_CORRECTION_METHODS.get((method, fc_method))
        if spec is None:
            # Two distinguishable failures, worth separate messages: the FC
            # method has no null at all (e.g. wavelet coherence, whose
            # surrogates would have to be recomputed through the CWT, not
            # through Welch), or it has one under a different scheme name.
            for_this_method = sorted(
                n for n, m in _BIAS_CORRECTION_METHODS if m == fc_method)
            if for_this_method:
                raise ValueError(
                    f"Unknown bias-correction method '{method}' for "
                    f"FC.fit(method={fc_method!r}). Available: {for_this_method}."
                )
            raise ValueError(
                f"No bias correction is registered for FC.fit(method="
                f"{fc_method!r}). Available for: "
                f"{sorted({m for _, m in _BIAS_CORRECTION_METHODS})}."
            )

        da = fc_output.output['value']
        if spec['needs_freq'] and 'freq' not in da.dims:
            raise ValueError(
                "correct_bias() requires the full frequency-resolved output "
                "(before .reduce()) -- it needs to recompute coherence across "
                "surrogate data at every frequency to build the null distribution."
            )

        missing = [a for a in spec['required_attrs'] if a not in attrs]
        if missing:
            raise ValueError(
                f"correct_bias() needs {missing} from the original fit's attrs "
                f"to replay the surrogates identically, but they're missing "
                f"from this method={fc_method!r} output."
            )

        data_obj = self.dataset.data
        values = data_obj.values
        if len(data_obj.dims) == 4:
            values = np.mean(values, axis=3)
        n_times, n_chans, n_types = values.shape

        is_bad = data_obj.coords.get('is_bad', np.zeros(n_chans, dtype=bool))
        is_short = data_obj.coords.get('is_short', np.zeros(n_chans, dtype=bool))
        valid_indices = np.where(~(is_bad | is_short))[0]

        freqs = fc_output.output['freq'].values if spec['needs_freq'] else None
        rng = np.random.default_rng(random_state)

        null_mean, null_std = spec['fn'](
            values, valid_indices, n_chans, n_types, freqs,
            n_surrogates=n_surrogates, rng=rng,
            **{a: attrs[a] for a in spec['required_attrs']},
        )

        observed = da.values
        with np.errstate(invalid='ignore', divide='ignore'):
            null_z = (observed - null_mean) / null_std
        null_z = np.where(null_std == 0, np.nan, null_z)

        null_z_meta = FC_UNITS_META['null_z']
        new_attrs = dict(attrs)
        new_attrs['units'] = 'null_z'
        # The source space's own range no longer applies -- neither
        # coherence's (0, 1) nor Pearson's (-1, 1) bounds a null_z score.
        new_attrs.pop('range', None)
        if null_z_meta.get('cmap') is not None:
            new_attrs['cmap'] = null_z_meta['cmap']

        new_ds = fc_output.output.copy()
        new_ds['value'].values = null_z
        new_ds.attrs = new_attrs

        from .. import __version__
        history = list(fc_output.history) + [{
            'operation': 'FC.correct_bias',
            'params': {'method': method, 'n_surrogates': n_surrogates, 'random_state': random_state},
            'version': __version__,
        }]
        return FCOutput(data=new_ds, probe=fc_output.probe, analysis_type='Connectivity', history=history)


    def _package_results(self, matrix, extra_dims, value_meta, config, sidecars=None):
        data_obj = self.dataset.data
        type_dim = 'chromophore' if 'chromophore' in data_obj.dims else 'wavelength'

        coords = {
            'channel_i': data_obj.channel.values,
            'channel_j': data_obj.channel.values,
            type_dim: data_obj.coords[type_dim].values,
        }
        dims = ['channel_i', 'channel_j', type_dim]

        for dim_name, coord_vals in extra_dims.items():
            coords[dim_name] = coord_vals
            dims.append(dim_name)

        # Propagate per-channel quality metrics (if present upstream, e.g.
        # from sci_screen()/snr_screen()) so FCOutput.roi_average(weights=...)
        # can use them without needing the original Datastream around.
        for qname in CHANNEL_QUALITY_COORDS:
            if qname in data_obj.coords and data_obj.coords[qname].dims == ('channel',):
                coords[f'{qname}_i'] = ('channel_i', data_obj.coords[qname].values)
                coords[f'{qname}_j'] = ('channel_j', data_obj.coords[qname].values)

        units = value_meta.get('units')
        units_meta = FC_UNITS_META.get(units, {})
        # Carry the FULL value_meta into attrs (not just 'units') -- e.g.
        # coherence's resolved fs/nperseg/noverlap/window/detrend -- so the
        # exact Welch parameters used are reproducible from the output alone
        # (FC.correct_bias() reads these back out to regenerate surrogates
        # identically), not just for whichever ones happened to be passed
        # explicitly and therefore landed in history/config.
        attrs = {'method': config['method'], **value_meta}
        if units_meta.get('range') is not None:
            attrs['range'] = units_meta['range']
        if units_meta.get('cmap') is not None:
            attrs['cmap'] = units_meta['cmap']

        # Sidecars share the pair/type dims but never the trailing
        # freq/time/metric ones -- they describe the PAIR (how much
        # independent evidence backs it), not each cell of a spectrum.
        data_vars = {'value': (dims, matrix)}
        pair_dims = ['channel_i', 'channel_j', type_dim]
        for name, arr in (sidecars or {}).items():
            data_vars[name] = (pair_dims, arr)

        res_xr = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=attrs,
        )

        from .. import __version__
        history = list(self.dataset.history) + [{
            'operation': 'FC.fit', 'params': config, 'version': __version__
        }]
        return FCOutput(data=res_xr, probe=self.dataset.probe, analysis_type='Connectivity', history=history)


    def _package_cross_results(self, matrix, data_a, data_b, other_history, extra_dims, value_meta, config):
        """
        Package a cross-stream result into an FCOutput.

        Parameters
        ----------
        matrix : np.ndarray
            Cross-connectivity values.
        data_a, data_b : xr.DataArray
            The two source streams.
        other_history : list
            History of the second stream, recorded alongside this one's.
        extra_dims : dict
            Additional coordinate axes, such as frequency or time.
        value_meta : dict
            Units and related metadata for the values.
        config : dict
            Fit settings recorded on the output.

        Returns
        -------
        FCOutput
            Non-square result with no probe attached, since the two streams
            generally carry different geometry.
        """
        type_dim = 'chromophore' if 'chromophore' in data_a.dims else 'wavelength'

        coords = {
            'channel_i': data_a.channel.values,
            'channel_j': data_b.channel.values,
            type_dim: data_a.coords[type_dim].values,
        }
        dims = ['channel_i', 'channel_j', type_dim]

        for dim_name, coord_vals in extra_dims.items():
            coords[dim_name] = coord_vals
            dims.append(dim_name)

        for qname in CHANNEL_QUALITY_COORDS:
            if qname in data_a.coords and data_a.coords[qname].dims == ('channel',):
                coords[f'{qname}_i'] = ('channel_i', data_a.coords[qname].values)
            if qname in data_b.coords and data_b.coords[qname].dims == ('channel',):
                coords[f'{qname}_j'] = ('channel_j', data_b.coords[qname].values)

        units = value_meta.get('units')
        units_meta = FC_UNITS_META.get(units, {})
        attrs = {'method': config['method'], **value_meta}
        if units_meta.get('range') is not None:
            attrs['range'] = units_meta['range']
        if units_meta.get('cmap') is not None:
            attrs['cmap'] = units_meta['cmap']

        res_xr = xr.Dataset(
            data_vars={'value': (dims, matrix)},
            coords=coords,
            attrs=attrs,
        )

        from .. import __version__
        # Both operands' provenance, not just this stream's: a cross-only
        # fit is a genuine two-input operation, and (unlike the intra-brain
        # path) `other`'s history isn't guaranteed to surface anywhere else
        # if this FCOutput is used standalone, so record both here.
        history = list(self.dataset.history) + list(other_history) + [{
            'operation': 'FC.fit', 'params': config, 'version': __version__
        }]
        return FCOutput(data=res_xr, probe=None, analysis_type='Connectivity', history=history)
