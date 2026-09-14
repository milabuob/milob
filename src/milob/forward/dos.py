"""
Continuous-wave and frequency-domain DOS forward models.

Thin assembly of the shared geometry kernels and dispersion relation.
Continuous wave is the zero-frequency case.
"""

import numpy as np
import xarray as xr

from . import kernels
from . import dispersion


def si_fd_fluence(rho, mua, musp, n, wavelength, freq=0.0, R_eff=None):
    """
    Semi-infinite CW or FD-DOS fluence.

    Parameters
    ----------
    rho : float or array-like
        Source-detector distance in cm.
    mua : float
        Absorption coefficient in cm^-1.
    musp : float
        Reduced scattering coefficient in cm^-1.
    n : float
        Refractive index of the medium.
    wavelength : float
        Wavelength in nm. Unused here, and present so that the signature
        matches the DCS counterpart.
    freq : float or array-like
        Modulation frequency in Hz. 0 for continuous wave.
    R_eff : float, optional
        Effective boundary reflectance. Computed from ``n`` against air if
        None.

    Returns
    -------
    np.ndarray of complex
        Fluence, broadcast over ``rho`` and ``freq``. The imaginary part is
        zero at zero frequency.
    """
    rho = np.asarray(rho, dtype=float)
    freq = np.asarray(freq, dtype=float)
    scalar_rho = rho.ndim == 0
    scalar_freq = freq.ndim == 0
    rho = np.atleast_1d(rho)
    freq = np.atleast_1d(freq)

    if R_eff is None:
        R_eff = kernels.Reff(n, 1.0)

    K2 = dispersion.k2(mua, musp, n, wavelength, freq=freq)  # shape (n_freq,)
    K = np.sqrt(K2.astype(complex))

    # Broadcasting: rho (n_ch, 1) x K (1, n_freq) -> phi (n_ch, n_freq)
    phi = kernels.si_kernel(rho[:, np.newaxis], K[np.newaxis, :], mua, musp, R_eff)

    if scalar_rho and scalar_freq:
        return phi[0, 0]
    if scalar_rho:
        return phi[0]
    if scalar_freq:
        return phi[:, 0]
    return phi


def _layer_R_eff(n):
    """Return the default boundary reflectance for each layer interface."""
    n = np.asarray(n, dtype=float)
    return [kernels.Reff(n[0], 1.0)] + [kernels.Reff(n[i - 1], n[i]) for i in range(1, len(n))]


def two_layer_fd_fluence(rho, z, mua, musp, n, wavelength, depth, freq=0.0,
                          R_eff=None, a=30.0, m=4000):
    """
    Two-layer CW or FD-DOS fluence, for a finite layer over a semi-infinite one.

    Parameters
    ----------
    rho : float
        Source-detector separation in cm.
    z : float
        Detector depth in cm, with 0 the boundary.
    mua, musp : sequence of float
        Absorption and reduced scattering per layer in cm^-1, top layer first.
    n : sequence of float
        Refractive index per layer.
    wavelength : float
        Wavelength in nm. Unused here, and present for signature parity with
        the DCS counterpart.
    depth : sequence of float
        Thickness in cm of each bounded layer; the last layer is
        semi-infinite.
    freq : float
        Modulation frequency in Hz. 0 for continuous wave.
    R_eff : sequence of float, optional
        Effective reflectance at each boundary, the top surface first.
        Computed from ``n`` if None.
    a : float
        Radius in cm truncating the Fourier-Bessel series.
    m : int
        Number of terms in the series.

    Returns
    -------
    complex
        Fluence for a unit source. The imaginary part is zero at zero
        frequency.
    """
    mua = np.asarray(mua, dtype=float)
    musp = np.asarray(musp, dtype=float)
    n = np.asarray(n, dtype=float)
    n_layers = len(mua)

    if R_eff is None:
        R_eff = _layer_R_eff(n)

    K_sq = [complex(dispersion.k2(mua[i], musp[i], n[i], wavelength, freq=freq))
            for i in range(n_layers)]

    f = kernels.two_layer_kernel(rho, z, K_sq, mua, musp, n, depth, R_eff, a=a, m=m)
    return complex(f)


def n_layer_fd_fluence(rho, z, mua, musp, n, wavelength, depth, freq=0.0,
                        R_eff_top=None, R_eff_bottom=None, s_max_factor=30.0, n_points=480):
    """
    General N-layer CW or FD-DOS fluence.

    Parameters
    ----------
    rho : float
        Source-detector separation in cm.
    z : float
        Detector depth in cm, with 0 the boundary.
    mua, musp : sequence of float
        Absorption and reduced scattering per layer in cm^-1, source layer
        first.
    n : sequence of float
        Refractive index per layer.
    wavelength : float
        Wavelength in nm. Unused here, and present for signature parity with
        the DCS counterpart.
    depth : sequence of float
        Thickness in cm of each bounded layer, top first. One shorter than the
        layer count for a semi-infinite base, otherwise the same length.
    freq : float
        Modulation frequency in Hz. 0 for continuous wave.
    R_eff_top : float, optional
        Effective reflectance at the top boundary. Computed from ``n[0]`` if
        None.
    R_eff_bottom : float, optional
        Effective reflectance at the base. None makes the base semi-infinite.
    s_max_factor : float
        Upper integration limit of the Hankel transform, as a multiple of the
        source layer's reduced scattering.
    n_points : int
        Number of quadrature points.

    Returns
    -------
    complex
        Fluence for a unit source.
    """
    mua = np.asarray(mua, dtype=float)
    musp = np.asarray(musp, dtype=float)
    n = np.asarray(n, dtype=float)
    depth = np.asarray(depth, dtype=float)
    n_layers = len(mua)

    finite_bottom = len(depth) == n_layers
    if not finite_bottom and len(depth) != n_layers - 1:
        raise ValueError(
            f"depth must have length {n_layers - 1} (semi-infinite bottom) "
            f"or {n_layers} (finite bottom), got {len(depth)}."
        )

    if R_eff_top is None:
        R_eff_top = kernels.Reff(n[0], 1.0)
    if finite_bottom and R_eff_bottom is None:
        R_eff_bottom = kernels.Reff(n[-1], 1.0)

    K_sq = [complex(dispersion.k2(mua[i], musp[i], n[i], wavelength, freq=freq))
            for i in range(n_layers)]

    f = kernels.n_layer_kernel(rho, z, K_sq, mua, musp, n, depth, R_eff_top,
                                R_eff_bottom=R_eff_bottom, s_max_factor=s_max_factor, n_points=n_points)
    return complex(f)


# ------------------------------------------------------------------
# TD-DOS forward models.
#
# Per skill debt item 7, TD-DOS needs no new per-geometry derivation: sweep
# the existing CW/FD-DOS fluence function over frequency and inverse-FFT to
# real time (Liemert & Kienle's own primary validated method for turning an
# FD solution into a TD one). ``_fd_sweep_to_td`` is the one shared
# mechanism; every geometry below is a 5-10 line configuration of it, dos
# methods regardless of whether they operate in the time or frequency
# domain. ``si_td_fluence_patterson`` is kept separately as an independent,
# hand-derived closed form for the semi-infinite case only, to
# cross-validate ``si_td_fluence`` -- see its own docstring.
# ------------------------------------------------------------------

def _fd_sweep_to_td(fd_fluence_func, t, freq_max=None, n_freq=None, **fd_kwargs):
    """
    Convert any frequency-domain fluence function into its time-domain form.

    Sweeps the model over frequency and inverse-transforms. Only the
    non-negative frequency half is evaluated, since the time-domain fluence is
    real and causal.

    Parameters
    ----------
    fd_fluence_func : callable
        A frequency-domain fluence function, called once per frequency.
    t : array-like
        Output time grid in seconds, uniformly spaced and starting at or after
        zero. The pulse is launched at zero.
    freq_max : float, optional
        Highest modulation frequency swept in Hz, setting the time resolution
        of the transform grid. Defaults to five times the Nyquist rate implied
        by the requested time spacing.
    n_freq : int, optional
        Number of frequency points, setting both the frequency resolution and
        the transform's time window, which must exceed the requested span or
        the diffuse tail wraps around. Defaults to an eightfold margin.
    **fd_kwargs
        Forwarded to ``fd_fluence_func`` on every call.

    Returns
    -------
    np.ndarray
        Real-valued fluence interpolated onto ``t``.
    """
    t = np.asarray(t, dtype=float)
    dt_req = (t[1] - t[0]) if len(t) > 1 else 1e-11
    t_max_req = max(float(t[-1]), dt_req)

    if freq_max is None:
        freq_max = 5.0 / (2.0 * dt_req)
    if n_freq is None:
        t_window = 8.0 * t_max_req
        n_freq = int(np.ceil(freq_max * t_window)) + 1
    n_freq = max(n_freq, 2)

    freq_grid = np.linspace(0.0, freq_max, n_freq)
    phi = np.array([fd_fluence_func(freq=float(f), **fd_kwargs) for f in freq_grid], dtype=complex)

    n_time = 2 * (n_freq - 1)
    df = freq_max / (n_freq - 1)
    psi_native = np.fft.irfft(phi, n=n_time)
    dt_native = 1.0 / (n_time * df)
    t_native = np.arange(n_time) * dt_native

    return np.interp(t, t_native, psi_native)


def si_td_fluence(t, rho, mua, musp, n, wavelength, R_eff=None, freq_max=None, n_freq=None):
    """
    Semi-infinite TD-DOS fluence, by frequency sweep and inverse transform.

    The general time-domain model, sharing its dispersion relation with the
    frequency-domain models. :func:`si_td_fluence_patterson` is an independent
    closed form for the same geometry, kept for cross-validation.

    Parameters
    ----------
    t : array-like
        Time since the pulse in seconds, starting at or after zero.
    rho : float or array-like
        Source-detector distance in cm.
    mua, musp : float
        Absorption and reduced scattering in cm^-1.
    n : float
        Refractive index of the medium.
    wavelength : float
        Wavelength in nm. Unused here.
    R_eff : float, optional
        Effective boundary reflectance. Computed from ``n`` if None.
    freq_max : float, optional
        Highest modulation frequency swept in Hz, setting the time resolution
        of the transform grid. Defaults to five times the Nyquist rate implied
        by the requested time spacing.
    n_freq : int, optional
        Number of frequency points, setting both the frequency resolution and
        the transform's time window, which must exceed the requested span or
        the diffuse tail wraps around. Defaults to an eightfold margin.

    Returns
    -------
    np.ndarray
        Real-valued fluence on the requested time grid.
    """
    if R_eff is None:
        R_eff = kernels.Reff(n, 1.0)
    return _fd_sweep_to_td(si_fd_fluence, t, freq_max=freq_max, n_freq=n_freq,
                            rho=rho, mua=mua, musp=musp, n=n, wavelength=wavelength, R_eff=R_eff)


def si_td_fluence_patterson(t, rho, mua, musp, n, A=1.0):
    """
    Semi-infinite TD-DOS fluence, in closed form.

    Uses the classical solution of the time-dependent diffusion equation under
    the usual approximation that scattering dominates absorption in the
    diffusion coefficient. Independent of :func:`si_td_fluence`, and kept as a
    cross-check on it.

    Parameters
    ----------
    t : array-like
        Time since the pulse in seconds. Non-positive values return zero.
    rho : float
        Source-detector distance in cm.
    mua, musp : float
        Absorption and reduced scattering in cm^-1.
    n : float
        Refractive index of the medium.
    A : float
        Amplitude scaling. Default 1.0.

    Returns
    -------
    np.ndarray
        Real-valued fluence on the requested time grid.

    References
    ----------
    Patterson, M. S., Chance, B., & Wilson, B. C. (1989). Applied Optics,
    28(12), 2331-2336.
    """
    t = np.asarray(t, dtype=float)
    v = dispersion.C_LIGHT / n  # cm/s

    D = 1.0 / (3.0 * musp)
    z0 = 1.0 / musp

    psi = np.zeros_like(t)
    mask = t > 0
    tt = t[mask]

    term1 = (4.0 * np.pi * D * v) ** (-1.5) * tt ** (-2.5)
    exponent = -mua * v * tt - (rho ** 2 + z0 ** 2) / (4.0 * D * v * tt)
    psi[mask] = A * term1 * np.exp(np.clip(exponent, -700, 700))

    return psi


def two_layer_td_fluence(t, rho, z, mua, musp, n, wavelength, depth,
                          R_eff=None, a=30.0, m=4000, freq_max=None, n_freq=None):
    """
    Two-layer TD-DOS fluence, by frequency sweep and inverse transform.

    Parameters are those of :func:`two_layer_fd_fluence`, with ``t`` in place
    of ``freq``, plus the sweep settings of :func:`_fd_sweep_to_td`.

    Returns
    -------
    np.ndarray
        Real-valued fluence on the requested time grid.
    """
    return _fd_sweep_to_td(two_layer_fd_fluence, t, freq_max=freq_max, n_freq=n_freq,
                            rho=rho, z=z, mua=mua, musp=musp, n=n, wavelength=wavelength,
                            depth=depth, R_eff=R_eff, a=a, m=m)


def n_layer_td_fluence(t, rho, z, mua, musp, n, wavelength, depth,
                        R_eff_top=None, R_eff_bottom=None, s_max_factor=30.0, n_points=480,
                        freq_max=None, n_freq=None):
    """
    General N-layer TD-DOS fluence, by frequency sweep and inverse transform.

    Parameters are those of :func:`n_layer_fd_fluence`, with ``t`` in place of
    ``freq``, plus the sweep settings of :func:`_fd_sweep_to_td`.

    Returns
    -------
    np.ndarray
        Real-valued fluence on the requested time grid.
    """
    return _fd_sweep_to_td(n_layer_fd_fluence, t, freq_max=freq_max, n_freq=n_freq,
                            rho=rho, z=z, mua=mua, musp=musp, n=n, wavelength=wavelength,
                            depth=depth, R_eff_top=R_eff_top, R_eff_bottom=R_eff_bottom,
                            s_max_factor=s_max_factor, n_points=n_points)


# ------------------------------------------------------------------
# `assemble` callbacks for processing.fitting.fd_model_opt +
# two_layer_fd_fluence -- see FD_Stream.fit_to_op(). Each maps the flat,
# merged parameter dict (forward_parameters + fitted + fixed) onto the
# array-shaped kwargs two_layer_fd_fluence expects. `n_1`/`n_2` (refractive
# indices) are always expected as fixed context, never fit.
# ------------------------------------------------------------------

def _two_layer_series_kwargs(flat):
    """Extract the optional Fourier-Bessel truncation settings from a parameter dict."""
    kwargs = {}
    if "a" in flat:
        kwargs["a"] = flat["a"]
    if "m" in flat:
        kwargs["m"] = flat["m"]
    return kwargs


def assemble_two_layer_fd(flat):
    """
    Map flat two-layer parameters onto the forward model's arguments.

    Fits every optical property and the layer thickness. Pin the thickness
    through ``fixed_params`` rather than writing another assembler.

    Parameters
    ----------
    flat : dict
        Merged fixed, fitted and context parameters.

    Returns
    -------
    dict
        Keyword arguments for :func:`two_layer_fd_fluence`.
    """
    return {
        "mua": [flat["mua_1"], flat["mua_2"]],
        "musp": [flat["musp_1"], flat["musp_2"]],
        "n": [flat["n_1"], flat["n_2"]],
        "depth": [flat["depth"]],
        "wavelength": flat["wavelength"],
        "freq": flat.get("freq", 0.0),
        "z": flat.get("z", 0.0),
        **_two_layer_series_kwargs(flat),
    }


def assemble_two_layer_fd_shared_musp(flat):
    """
    Map flat two-layer parameters onto the model, with scattering shared.

    Assumes one reduced scattering coefficient for both layers, a common
    constraint where absorption differs between them far more than scattering
    does.

    Parameters
    ----------
    flat : dict
        Merged fixed, fitted and context parameters.

    Returns
    -------
    dict
        Keyword arguments for :func:`two_layer_fd_fluence`.
    """
    return {
        "mua": [flat["mua_1"], flat["mua_2"]],
        "musp": [flat["musp_shared"], flat["musp_shared"]],
        "n": [flat["n_1"], flat["n_2"]],
        "depth": [flat["depth"]],
        "wavelength": flat["wavelength"],
        "freq": flat.get("freq", 0.0),
        "z": flat.get("z", 0.0),
        **_two_layer_series_kwargs(flat),
    }


TWO_LAYER_FD_PARAM_CONFIG = {
    "mua_1": {"bounds": (1e-3, 0.5), "log": True},
    "musp_1": {"bounds": (1.0, 30.0), "log": True},
    "mua_2": {"bounds": (1e-3, 0.5), "log": True},
    "musp_2": {"bounds": (1.0, 30.0), "log": True},
    "depth": {"bounds": (0.2, 3.0), "log": False},
}
"""Default param_config for assemble_two_layer_fd (fit every unknown,
including depth). Pass depth via fixed_params to pin it instead."""


def _n_layer_series_kwargs(flat):
    """Extract the optional Hankel-quadrature settings from a parameter dict."""
    kwargs = {}
    if "s_max_factor" in flat:
        kwargs["s_max_factor"] = flat["s_max_factor"]
    if "n_points" in flat:
        kwargs["n_points"] = flat["n_points"]
    return kwargs


def make_assemble_n_layer_fd(n_layers, finite_bottom=False):
    """
    Build an assembler for fitting an N-layer model with every property free.

    A factory rather than a function, since the flat parameter names depend on
    the layer count. Refractive indices are always fixed context.

    Parameters
    ----------
    n_layers : int
        Number of layers.
    finite_bottom : bool
        False (default) leaves the base semi-infinite, giving one fewer
        thickness parameter than layers. True makes it finite, giving one per
        layer.

    Returns
    -------
    callable
        Maps a flat parameter dict to keyword arguments for
        :func:`n_layer_fd_fluence`.
    """
    n_depths = n_layers if finite_bottom else n_layers - 1

    def assemble(flat):
        return {
            "mua": [flat[f"mua_{i}"] for i in range(1, n_layers + 1)],
            "musp": [flat[f"musp_{i}"] for i in range(1, n_layers + 1)],
            "n": [flat[f"n_{i}"] for i in range(1, n_layers + 1)],
            "depth": [flat[f"depth_{i}"] for i in range(1, n_depths + 1)],
            "wavelength": flat["wavelength"],
            "freq": flat.get("freq", 0.0),
            "z": flat.get("z", 0.0),
            **_n_layer_series_kwargs(flat),
        }

    return assemble


def make_n_layer_fd_param_config(n_layers, finite_bottom=False,
                                  mua_bounds=(1e-3, 0.5), musp_bounds=(1.0, 30.0),
                                  depth_bounds=(0.2, 3.0)):
    """
    Build a parameter configuration matching :func:`make_assemble_n_layer_fd`.

    Parameters
    ----------
    n_layers : int
        Number of layers.
    finite_bottom : bool
        Whether the base layer is finite. Default False.
    mua_bounds, musp_bounds, depth_bounds : tuple of (float, float)
        Bounds applied to every layer's absorption, scattering and thickness.

    Returns
    -------
    dict
        Free parameters and their bounds. Pin any thickness through
        ``fixed_params`` instead of removing it here.
    """
    n_depths = n_layers if finite_bottom else n_layers - 1
    config = {}
    for i in range(1, n_layers + 1):
        config[f"mua_{i}"] = {"bounds": mua_bounds, "log": True}
        config[f"musp_{i}"] = {"bounds": musp_bounds, "log": True}
    for i in range(1, n_depths + 1):
        config[f"depth_{i}"] = {"bounds": depth_bounds, "log": False}
    return config


def _fd_freq_axis(modulation_freq):
    """
    Build the frequency coordinate for a simulated FD stream.

    A stream carries the DC component alongside the complex measurement. At
    zero modulation the two coincide, so a single DC entry is emitted; a
    zero-modulation measurement is continuous wave.

    Parameters
    ----------
    modulation_freq : float
        Modulation frequency in Hz.

    Returns
    -------
    np.ndarray
        Frequency coordinate values.
    """
    f = float(modulation_freq)
    return np.array([0.0]) if f == 0.0 else np.array([0.0, f])


def simulate_fd_stream(probe, mua, musp, modulation_freq, n=1.33,
                        noise_level=None, rng=None):
    """
    Simulate an FD-DOS dataset for a semi-infinite homogeneous medium.

    Parameters
    ----------
    probe : Probe
        Must carry a channel configuration. Distances are read from the probe
        and assumed to be in cm.
    mua, musp : array-like
        Absorption and reduced scattering in cm^-1, one per probe wavelength.
    modulation_freq : float
        Modulation frequency in Hz.
    n : float
        Refractive index. Default 1.33.
    noise_level : float, optional
        Proportional Gaussian noise, as a fraction of the mean. None adds
        none.
    rng : int or numpy.random.Generator, optional
        Seed or generator for reproducible noise.

    Returns
    -------
    FD_Stream
        Shape (1, n_channels, n_wavelengths, 2), with the DC and modulation
        frequencies on the last axis.
    """
    from ..core.fd_nirs import FD_Stream

    mua = np.asarray(mua, dtype=float)
    musp = np.asarray(musp, dtype=float)

    n_wl = len(probe.wavelengths)
    if mua.shape != (n_wl,):
        raise ValueError(f"mua must have shape ({n_wl},) to match probe wavelengths, got {mua.shape}.")
    if musp.shape != (n_wl,):
        raise ValueError(f"musp must have shape ({n_wl},) to match probe wavelengths, got {musp.shape}.")

    distances = np.array(probe.distances, dtype=float)
    n_ch = len(distances)
    freqs = _fd_freq_axis(modulation_freq)

    R_eff = kernels.Reff(n, 1.0)
    data = np.zeros((1, n_ch, n_wl, len(freqs)), dtype=complex)

    for wi in range(n_wl):
        phi = si_fd_fluence(distances, mua[wi], musp[wi], n, probe.wavelengths[wi],
                             freq=freqs, R_eff=R_eff)
        # Conjugate to match the library phase convention: positive imaginary part
        # encodes positive phase delay (as stored by the OxiplexTS instrument).
        # si_fd_fluence returns physics-convention fluence (negative imaginary
        # part for a delayed signal).
        data[0, :, wi, :] = np.conj(phi)

    if noise_level is not None:
        rng = np.random.default_rng(rng)
        amp = np.abs(data)
        data = (data
                + rng.standard_normal(data.shape) * amp * noise_level
                + 1j * rng.standard_normal(data.shape) * amp * noise_level)

    data_xr = xr.DataArray(
        data,
        dims=['time', 'channel', 'wavelength', 'freq'],
        coords={
            'time': [0.0],
            'channel': probe.channel_labels,
            'wavelength': probe.wavelengths,
            'freq': freqs,
            'distance': ('channel', distances),
        },
        attrs={'status': 'raw', 'lengthUnit': 'cm'},
    )

    return FD_Stream(data=data_xr, probe=probe, name='simulated_fd', status='raw')


def simulate_two_layer_fd_stream(probe, mua, musp, n, depth, modulation_freq,
                                  noise_level=None, rng=None, a=30.0, m=4000):
    """
    Simulate an FD-DOS dataset for a two-layer medium.

    Parameters
    ----------
    probe : Probe
        Must carry a channel configuration, with distances in cm.
    mua, musp : array-like
        Shape (n_layers, n_wavelengths), in cm^-1, top layer first.
    n : array-like
        Refractive index per layer, shared across wavelengths.
    depth : array-like
        Thickness in cm of each bounded layer.
    modulation_freq : float
        Modulation frequency in Hz.
    noise_level : float, optional
        Proportional Gaussian noise, as a fraction of the mean. None adds
        none.
    rng : int or numpy.random.Generator, optional
        Seed or generator for reproducible noise.
    a : float
        Radius in cm truncating the Fourier-Bessel series.
    m : int
        Number of terms. The default is accurate but slow; reduce it for
        exploratory runs.

    Returns
    -------
    FD_Stream
        Shape (1, n_channels, n_wavelengths, 2).
    """
    from ..core.fd_nirs import FD_Stream

    mua = np.asarray(mua, dtype=float)
    musp = np.asarray(musp, dtype=float)
    n = np.asarray(n, dtype=float)
    depth = np.asarray(depth, dtype=float)
    n_layers, n_wl = mua.shape

    if musp.shape != (n_layers, n_wl):
        raise ValueError(f"musp must have shape {(n_layers, n_wl)} to match mua, got {musp.shape}.")
    if len(n) != n_layers:
        raise ValueError(f"n must have shape ({n_layers},), one refractive index per layer, got {n.shape}.")

    n_wl_probe = len(probe.wavelengths)
    if n_wl != n_wl_probe:
        raise ValueError(f"mua/musp must have shape ({n_layers}, {n_wl_probe}) to match probe wavelengths, got {mua.shape}.")

    distances = np.array(probe.distances, dtype=float)
    n_ch = len(distances)
    freqs = _fd_freq_axis(modulation_freq)

    data = np.zeros((1, n_ch, n_wl, len(freqs)), dtype=complex)
    for wi in range(n_wl):
        wl = float(probe.wavelengths[wi])
        for ci, rho in enumerate(distances):
            for fi, f in enumerate(freqs):
                phi = two_layer_fd_fluence(rho=rho, z=0.0, mua=mua[:, wi], musp=musp[:, wi],
                                            n=n, wavelength=wl, depth=depth, freq=f, a=a, m=m)
                # Same phase-convention conjugation as simulate_fd_stream.
                data[0, ci, wi, fi] = np.conj(phi)

    if noise_level is not None:
        rng = np.random.default_rng(rng)
        amp = np.abs(data)
        data = (data
                + rng.standard_normal(data.shape) * amp * noise_level
                + 1j * rng.standard_normal(data.shape) * amp * noise_level)

    data_xr = xr.DataArray(
        data,
        dims=['time', 'channel', 'wavelength', 'freq'],
        coords={
            'time': [0.0],
            'channel': probe.channel_labels,
            'wavelength': probe.wavelengths,
            'freq': freqs,
            'distance': ('channel', distances),
        },
        attrs={'status': 'raw', 'lengthUnit': 'cm'},
    )

    return FD_Stream(data=data_xr, probe=probe, name='simulated_two_layer_fd', status='raw')


def simulate_n_layer_fd_stream(probe, mua, musp, n, depth, modulation_freq,
                                noise_level=None, rng=None, s_max_factor=30.0, n_points=480):
    """
    Simulate an FD-DOS dataset for a general N-layer medium.

    Parameters
    ----------
    probe : Probe
        Must carry a channel configuration, with distances in cm.
    mua, musp : array-like
        Shape (n_layers, n_wavelengths), in cm^-1, top layer first.
    n : array-like
        Refractive index per layer.
    depth : array-like
        Thickness in cm of each bounded layer.
    modulation_freq : float
        Modulation frequency in Hz.
    noise_level : float, optional
        Proportional Gaussian noise, as a fraction of the mean. None adds
        none.
    rng : int or numpy.random.Generator, optional
        Seed or generator for reproducible noise.
    s_max_factor : float
        Upper integration limit of the Hankel transform.
    n_points : int
        Number of quadrature points.

    Returns
    -------
    FD_Stream
        Shape (1, n_channels, n_wavelengths, 2).
    """
    from ..core.fd_nirs import FD_Stream

    mua = np.asarray(mua, dtype=float)
    musp = np.asarray(musp, dtype=float)
    n = np.asarray(n, dtype=float)
    depth = np.asarray(depth, dtype=float)
    n_layers, n_wl = mua.shape

    if musp.shape != (n_layers, n_wl):
        raise ValueError(f"musp must have shape {(n_layers, n_wl)} to match mua, got {musp.shape}.")
    if len(n) != n_layers:
        raise ValueError(f"n must have shape ({n_layers},), one refractive index per layer, got {n.shape}.")

    n_wl_probe = len(probe.wavelengths)
    if n_wl != n_wl_probe:
        raise ValueError(f"mua/musp must have shape ({n_layers}, {n_wl_probe}) to match probe wavelengths, got {mua.shape}.")

    distances = np.array(probe.distances, dtype=float)
    n_ch = len(distances)
    freqs = _fd_freq_axis(modulation_freq)

    data = np.zeros((1, n_ch, n_wl, len(freqs)), dtype=complex)
    for wi in range(n_wl):
        wl = float(probe.wavelengths[wi])
        for ci, rho in enumerate(distances):
            for fi, f in enumerate(freqs):
                phi = n_layer_fd_fluence(rho=rho, z=0.0, mua=mua[:, wi], musp=musp[:, wi],
                                          n=n, wavelength=wl, depth=depth, freq=f,
                                          s_max_factor=s_max_factor, n_points=n_points)
                # Same phase-convention conjugation as simulate_fd_stream.
                data[0, ci, wi, fi] = np.conj(phi)

    if noise_level is not None:
        rng = np.random.default_rng(rng)
        amp = np.abs(data)
        data = (data
                + rng.standard_normal(data.shape) * amp * noise_level
                + 1j * rng.standard_normal(data.shape) * amp * noise_level)

    data_xr = xr.DataArray(
        data,
        dims=['time', 'channel', 'wavelength', 'freq'],
        coords={
            'time': [0.0],
            'channel': probe.channel_labels,
            'wavelength': probe.wavelengths,
            'freq': freqs,
            'distance': ('channel', distances),
        },
        attrs={'status': 'raw', 'lengthUnit': 'cm'},
    )

    return FD_Stream(data=data_xr, probe=probe, name='simulated_n_layer_fd', status='raw')


# ------------------------------------------------------------------
# TD-DOS simulation -- one per geometry, mirroring the FD ``simulate_*``
# functions above exactly (same probe/mua/musp/n/depth-style signature),
# swapping ``modulation_freq``/``freq`` for a gated time axis
# (``bin_width``/``n_bins``) built on the FD-sweep fluence functions above.
# ------------------------------------------------------------------

def _bin_td_fluence(fluence_func, bin_delays, bin_widths, **fluence_kwargs):
    """
    Approximate gated photon counts from a continuous TPSF.

    Samples the curve at each gate centre and scales by the gate width, which
    is valid while gates are narrow relative to the curve's own scale.

    Parameters
    ----------
    fluence_func : callable
        A time-domain fluence function.
    bin_delays, bin_widths : array-like
        Gate delay and width in seconds, of equal length.

    Returns
    -------
    np.ndarray
        Counts, the same shape as ``bin_delays``.
    """
    bin_delays = np.asarray(bin_delays, dtype=float)
    bin_widths = np.asarray(bin_widths, dtype=float)
    bin_centers = bin_delays + bin_widths / 2.0
    psi = fluence_func(t=bin_centers, **fluence_kwargs)
    return psi * bin_widths


def _default_td_gates(bin_width, n_bins):
    bin_delays = np.arange(n_bins) * bin_width
    bin_widths = np.full(n_bins, bin_width)
    return bin_delays, bin_widths


def _td_data_xr(data, probe, distances, bin_delays, bin_widths):
    return xr.DataArray(
        data,
        dims=['time', 'channel', 'wavelength', 'bin'],
        coords={
            'time': [0.0],
            'channel': probe.channel_labels,
            'wavelength': probe.wavelengths,
            'bin': np.arange(len(bin_delays)),
            'distance': ('channel', distances),
            'timeDelays': ('bin', bin_delays),
            'timeDelayWidths': ('bin', bin_widths),
        },
        attrs={'status': 'raw', 'lengthUnit': 'cm'},
    )


def _apply_td_noise(data, noise_level, rng):
    if noise_level is None:
        return data
    rng = np.random.default_rng(rng)
    data = data + rng.standard_normal(data.shape) * data * noise_level
    return np.clip(data, 0.0, None)


def simulate_si_td_stream(probe, mua, musp, n=1.33, bin_width=25e-12, n_bins=200,
                           noise_level=None, rng=None, freq_max=None, n_freq=None):
    """
    Simulate a gated TD-DOS dataset for a semi-infinite homogeneous medium.

    Parameters
    ----------
    probe : Probe
        Must carry a channel configuration, with distances in cm.
    mua, musp : array-like
        Absorption and reduced scattering in cm^-1, one per probe wavelength.
    n : float
        Refractive index. Default 1.33.
    bin_width : float
        Time-gate width in seconds. Default 25 ps.
    n_bins : int
        Number of gates. Default 200.
    noise_level : float, optional
        Proportional Gaussian noise, as a fraction of the mean. None adds
        none.
    rng : int or numpy.random.Generator, optional
        Seed or generator for reproducible noise.
    freq_max : float, optional
        Highest modulation frequency swept in Hz, setting the time resolution
        of the transform grid. Defaults to five times the Nyquist rate implied
        by the requested time spacing.
    n_freq : int, optional
        Number of frequency points, setting both the frequency resolution and
        the transform's time window, which must exceed the requested span or
        the diffuse tail wraps around. Defaults to an eightfold margin.

    Returns
    -------
    TD_Stream
        Shape (1, n_channels, n_wavelengths, n_bins) with ``status='raw'``,
        carrying gate delays and widths in seconds on the bin dimension.
    """
    from ..core.td_nirs import TD_Stream

    mua = np.asarray(mua, dtype=float)
    musp = np.asarray(musp, dtype=float)

    n_wl = len(probe.wavelengths)
    if mua.shape != (n_wl,):
        raise ValueError(f"mua must have shape ({n_wl},) to match probe wavelengths, got {mua.shape}.")
    if musp.shape != (n_wl,):
        raise ValueError(f"musp must have shape ({n_wl},) to match probe wavelengths, got {musp.shape}.")

    distances = np.array(probe.distances, dtype=float)
    n_ch = len(distances)
    bin_delays, bin_widths = _default_td_gates(bin_width, n_bins)

    R_eff = kernels.Reff(n, 1.0)
    data = np.zeros((1, n_ch, n_wl, n_bins), dtype=float)

    for wi in range(n_wl):
        wl = float(probe.wavelengths[wi])
        for ci, rho in enumerate(distances):
            data[0, ci, wi, :] = _bin_td_fluence(
                si_td_fluence, bin_delays, bin_widths,
                rho=rho, mua=mua[wi], musp=musp[wi], n=n, wavelength=wl, R_eff=R_eff,
                freq_max=freq_max, n_freq=n_freq,
            )

    data = _apply_td_noise(data, noise_level, rng)
    data_xr = _td_data_xr(data, probe, distances, bin_delays, bin_widths)
    return TD_Stream(data=data_xr, probe=probe, name='simulated_td', status='raw')


def simulate_two_layer_td_stream(probe, mua, musp, n, depth, bin_width=25e-12, n_bins=200,
                                  noise_level=None, rng=None, a=30.0, m=4000,
                                  freq_max=None, n_freq=None):
    """
    Simulate a gated TD-DOS dataset for a two-layer medium.

    Parameters are those of :func:`simulate_two_layer_fd_stream` for the
    medium, and :func:`simulate_si_td_stream` for the gating and sweep.

    Returns
    -------
    TD_Stream
        Shape (1, n_channels, n_wavelengths, n_bins) with ``status='raw'``.
    """
    from ..core.td_nirs import TD_Stream

    mua = np.asarray(mua, dtype=float)
    musp = np.asarray(musp, dtype=float)
    n = np.asarray(n, dtype=float)
    depth = np.asarray(depth, dtype=float)
    n_layers, n_wl = mua.shape

    if musp.shape != (n_layers, n_wl):
        raise ValueError(f"musp must have shape {(n_layers, n_wl)} to match mua, got {musp.shape}.")
    if len(n) != n_layers:
        raise ValueError(f"n must have shape ({n_layers},), one refractive index per layer, got {n.shape}.")

    n_wl_probe = len(probe.wavelengths)
    if n_wl != n_wl_probe:
        raise ValueError(f"mua/musp must have shape ({n_layers}, {n_wl_probe}) to match probe wavelengths, got {mua.shape}.")

    distances = np.array(probe.distances, dtype=float)
    n_ch = len(distances)
    bin_delays, bin_widths = _default_td_gates(bin_width, n_bins)

    data = np.zeros((1, n_ch, n_wl, n_bins), dtype=float)
    for wi in range(n_wl):
        wl = float(probe.wavelengths[wi])
        for ci, rho in enumerate(distances):
            data[0, ci, wi, :] = _bin_td_fluence(
                two_layer_td_fluence, bin_delays, bin_widths,
                rho=rho, z=0.0, mua=mua[:, wi], musp=musp[:, wi], n=n, wavelength=wl,
                depth=depth, a=a, m=m, freq_max=freq_max, n_freq=n_freq,
            )

    data = _apply_td_noise(data, noise_level, rng)
    data_xr = _td_data_xr(data, probe, distances, bin_delays, bin_widths)
    return TD_Stream(data=data_xr, probe=probe, name='simulated_two_layer_td', status='raw')


def simulate_n_layer_td_stream(probe, mua, musp, n, depth, bin_width=25e-12, n_bins=200,
                                noise_level=None, rng=None, s_max_factor=30.0, n_points=480,
                                freq_max=None, n_freq=None):
    """
    Simulate a gated TD-DOS dataset for a general N-layer medium.

    Parameters are those of :func:`simulate_n_layer_fd_stream` for the medium,
    and :func:`simulate_si_td_stream` for the gating and sweep.

    Returns
    -------
    TD_Stream
        Shape (1, n_channels, n_wavelengths, n_bins) with ``status='raw'``.
    """
    from ..core.td_nirs import TD_Stream

    mua = np.asarray(mua, dtype=float)
    musp = np.asarray(musp, dtype=float)
    n = np.asarray(n, dtype=float)
    depth = np.asarray(depth, dtype=float)
    n_layers, n_wl = mua.shape

    if musp.shape != (n_layers, n_wl):
        raise ValueError(f"musp must have shape {(n_layers, n_wl)} to match mua, got {musp.shape}.")
    if len(n) != n_layers:
        raise ValueError(f"n must have shape ({n_layers},), one refractive index per layer, got {n.shape}.")

    n_wl_probe = len(probe.wavelengths)
    if n_wl != n_wl_probe:
        raise ValueError(f"mua/musp must have shape ({n_layers}, {n_wl_probe}) to match probe wavelengths, got {mua.shape}.")

    distances = np.array(probe.distances, dtype=float)
    n_ch = len(distances)
    bin_delays, bin_widths = _default_td_gates(bin_width, n_bins)

    data = np.zeros((1, n_ch, n_wl, n_bins), dtype=float)
    for wi in range(n_wl):
        wl = float(probe.wavelengths[wi])
        for ci, rho in enumerate(distances):
            data[0, ci, wi, :] = _bin_td_fluence(
                n_layer_td_fluence, bin_delays, bin_widths,
                rho=rho, z=0.0, mua=mua[:, wi], musp=musp[:, wi], n=n, wavelength=wl,
                depth=depth, s_max_factor=s_max_factor, n_points=n_points,
                freq_max=freq_max, n_freq=n_freq,
            )

    data = _apply_td_noise(data, noise_level, rng)
    data_xr = _td_data_xr(data, probe, distances, bin_delays, bin_widths)
    return TD_Stream(data=data_xr, probe=probe, name='simulated_n_layer_td', status='raw')


