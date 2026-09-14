# processing/filters.py
import numpy as np
from scipy.signal import butter, filtfilt, lfilter, sosfiltfilt, sosfreqz, detrend as sp_detrend
from typing import Union, Optional, Literal, Tuple
import xarray as xr

def _prepare_input(data: Union[np.ndarray, xr.DataArray], fs: Optional[float]) -> Tuple[np.ndarray, dict, float]:
    """Coerce input to an array and resolve the sampling rate."""
    is_xarray = isinstance(data, xr.DataArray)
    
    if is_xarray:
        # Detect Sampling Frequency
        if fs is None:
            if 'sampling_rate' in data.attrs:
                fs = data.attrs['sampling_rate']
            elif 'time' in data.coords:
                fs = 1.0 / np.nanmean(np.diff(data.time.values))
            else:
                raise ValueError("Sampling rate (fs) could not be detected from xarray attrs or time coord.")
        
        meta = {'coords': data.coords, 'dims': data.dims, 'attrs': data.attrs, 'is_xarray': True}
        return data.values, meta, fs
    else:
        if fs is None:
            raise ValueError("fs is required for numpy input")
        return data, {'is_xarray': False}, fs


def _iir_edge(values: np.ndarray, cutoff: float, nyq: float, btype: str,
              order: int, zero_phase: bool) -> np.ndarray:
    """Apply one high-pass or low-pass Butterworth edge along the time axis."""
    min_hz = 1e-3 * nyq
    max_hz = 0.99 * nyq
    cutoff = min(max_hz, max(min_hz, cutoff))

    sos = butter(order, cutoff / nyq, btype=btype, output='sos')
    if zero_phase:
        return sosfiltfilt(sos, values, axis=0)
    from scipy.signal import sosfilt
    return sosfilt(sos, values, axis=0)


def frequency_filter(data: Union[np.ndarray, xr.DataArray],
                      lowcut: Optional[float] = None,
                      highcut: Optional[float] = None,
                      fs: Optional[float] = None,
                      order: int = 3,
                      low_order: Optional[int] = None,
                      high_order: Optional[int] = None,
                      zero_phase: bool = True) -> Union[np.ndarray, xr.DataArray]:
    """
    Filter along the time axis, high-pass, low-pass or band-pass.

    The filter type follows from which cutoffs are given. A band-pass is
    applied as a high-pass stage followed by a low-pass stage, so each edge can
    carry its own order.

    Parameters
    ----------
    data : np.ndarray or xr.DataArray
        Time series, filtered along axis 0.
    lowcut : float, optional
        High-pass edge in Hz. At least one cutoff must be given.
    highcut : float, optional
        Low-pass edge in Hz.
    fs : float, optional
        Sampling frequency in Hz. Read from the array's metadata for a
        DataArray.
    order : int
        Butterworth order used for any edge without its own. Default 3.
    low_order, high_order : int, optional
        Per-edge order, falling back to ``order``.
    zero_phase : bool
        Filter forwards and backwards, giving zero phase distortion.
        Default True.

    Returns
    -------
    np.ndarray or xr.DataArray
        Filtered data, of the same type as the input.

    Examples
    --------
    >>> filtered = frequency_filter(data, lowcut=0.5, highcut=2.5, fs=10.0)
    >>> filtered = frequency_filter(data, lowcut=0.01, fs=10.0)
    """
    if lowcut is None and highcut is None:
        raise ValueError("At least one of lowcut/highcut must be provided.")

    values, meta, fs = _prepare_input(data, fs)
    nyq = 0.5 * fs

    filtered = values
    if lowcut is not None:
        filtered = _iir_edge(filtered, lowcut, nyq, 'high',
                              low_order if low_order is not None else order,
                              zero_phase)
    if highcut is not None:
        filtered = _iir_edge(filtered, highcut, nyq, 'low',
                              high_order if high_order is not None else order,
                              zero_phase)

    if meta['is_xarray']:
        return xr.DataArray(filtered, coords=meta['coords'], dims=meta['dims'], attrs=meta['attrs'])
    return filtered

def detrend(data: Union[np.ndarray, xr.DataArray], recenter: bool = True) -> Union[np.ndarray, xr.DataArray]:
    """
    Remove a linear trend from each channel along the time axis.

    Parameters
    ----------
    data : np.ndarray or xr.DataArray
        Time series, detrended along axis 0.
    recenter : bool
        Add each channel's original mean back after detrending. If False, the
        result is zero-mean. Default True.

    Returns
    -------
    np.ndarray or xr.DataArray
        Detrended data, of the same type as the input.

    Examples
    --------
    >>> detrended = detrend(data)
    """
    values, meta, _ = _prepare_input(data, fs=1.0)
    detrended = sp_detrend(values, axis=0, type='linear')
    if recenter:
        detrended = detrended + np.mean(values, axis=0, keepdims=True)

    if meta['is_xarray']:
        return xr.DataArray(detrended, coords=meta['coords'], dims=meta['dims'], attrs=meta['attrs'])
    return detrended