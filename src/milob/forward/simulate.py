"""
Forward simulation: one simulator per modality, with the geometry as an argument.

    simulate_fd_stream(probe, mua, musp, modulation_freq, geometry="two_layer", depth=[1.2])
    simulate_td_stream(probe, mua, musp, geometry="two_layer", depth=[1.2])
    simulate_dcs_stream(probe, mua, musp, taus, aDb=..., geometry="two_layer", depth=[1.2])

Each geometry is registered once in `GEOMETRIES`, with its kernel for each
modality and the shape arguments it needs; the three simulators share input
handling, time series, noise, stream assembly and history. ``geometry`` is a
registered name, a `Geometry`, or a kernel function (the same callable a fit
takes as ``forward_model``).

Homogeneous geometries take ``mua``/``musp`` per wavelength,
``(n_wavelengths,)``; layered geometries take ``(n_layers, n_wavelengths)``,
top layer first. Either may carry a leading time axis, sampled at ``fs``;
time points with identical parameters are computed once. DCS is
single-wavelength: a scalar, or ``(n_layers,)``, with an optional leading
time axis.
"""

from dataclasses import dataclass
import inspect

import numpy as np
import xarray as xr

from . import dynamics
from ._geometry import probe_distances_cm
from .dos import (si_fd_fluence, two_layer_fd_fluence, n_layer_fd_fluence,
                  si_td_fluence, two_layer_td_fluence, n_layer_td_fluence,
                  _fd_freq_axis, _bin_td_fluence, _default_td_gates, _apply_td_noise)
from .dcs import (si_dcs_g1, two_layer_dcs_g1, n_layer_dcs_g1,
                  _apply_tau_dependent_noise, _simulated_stream_coords)
from .noise_models import fd_noise_model


@dataclass(frozen=True)
class Geometry:
    """
    A medium geometry, as the simulators see it.

    Attributes
    ----------
    name : str
        Registry key, used in stream names and history.
    n_layers : int or None
        1 for a homogeneous medium (``mua``/``musp`` per wavelength), a fixed
        count for a layered model that supports only that many, None for any
        number of layers.
    shape : tuple of str
        Geometry arguments the kernels need (e.g. ``("depth",)``); each must
        be given to the simulator. ``depth`` has ``n_layers - 1`` entries
        (the bottom layer is semi-infinite), or ``n_layers`` when
        ``finite_bottom`` allows it.
    fd, td, dcs : callable or None
        Kernel per modality: the ``*_fd_fluence``, ``*_td_fluence`` and
        ``*_dcs_g1`` functions, called with keyword arguments. None if the
        geometry has no kernel for that modality.
    vectorised : tuple of str
        Modalities whose kernel takes every channel (and every frequency or
        delay) in one call, as the semi-infinite ones do.
    finite_bottom : bool
        Whether ``depth`` may also give the bottom layer a thickness (a
        finite slab), as ``n_layer`` allows.
    """
    name: str
    n_layers: int | None
    shape: tuple
    fd: object = None
    td: object = None
    dcs: object = None
    vectorised: tuple = ()
    finite_bottom: bool = False

    @property
    def layered(self):
        return self.n_layers != 1

    def kernel(self, modality):
        k = getattr(self, modality)
        if k is None:
            raise NotImplementedError(f"geometry {self.name!r} has no {modality.upper()} kernel.")
        return k


GEOMETRIES = {}
"""Registered geometries, by name. Add one with `register_geometry`."""

_ALIASES = {"si": "semi_infinite", "semi-infinite": "semi_infinite"}


def register_geometry(geometry):
    """Make `geometry` (a `Geometry`) available to the simulators by name and by kernel."""
    GEOMETRIES[geometry.name] = geometry
    return geometry


for _g in (
    Geometry("semi_infinite", 1, (), si_fd_fluence, si_td_fluence, si_dcs_g1, vectorised=("fd", "dcs")),
    Geometry("two_layer", 2, ("depth",), two_layer_fd_fluence, two_layer_td_fluence, two_layer_dcs_g1),
    Geometry("n_layer", None, ("depth",), n_layer_fd_fluence, n_layer_td_fluence, n_layer_dcs_g1,
             finite_bottom=True),
):
    register_geometry(_g)


def resolve_geometry(geometry):
    """
    The `Geometry` for a name, the alias 'si', a `Geometry`, or any
    registered kernel function (``two_layer_fd_fluence``,
    ``two_layer_dcs_g1``, ...).
    """
    if isinstance(geometry, Geometry):
        return geometry
    if callable(geometry):
        for g in GEOMETRIES.values():
            if geometry in (g.fd, g.td, g.dcs):
                return g
        raise ValueError(f"{getattr(geometry, '__name__', geometry)!r} is not the kernel of a registered "
                         f"geometry; register it with register_geometry(Geometry(...)).")
    name = _ALIASES.get(geometry, geometry)
    if name not in GEOMETRIES:
        raise ValueError(f"Unknown geometry {geometry!r}; choose from {sorted(GEOMETRIES)}.")
    return GEOMETRIES[name]


# ------------------------------------------------------------------
# Shared input handling
# ------------------------------------------------------------------

_SHAPE_ARGS = {"depth"}


def _split_params(geom, modality, params):
    """Split extra keyword arguments into the geometry's shape arguments and kernel options."""
    params = dict(params)
    missing = [a for a in geom.shape if a not in params]
    if missing:
        raise ValueError(f"geometry {geom.name!r} needs {', '.join(missing)}.")
    shape_args = _SHAPE_ARGS.union(*(g.shape for g in GEOMETRIES.values()))
    stray = sorted(shape_args & set(params) - set(geom.shape))
    if stray:
        raise ValueError(f"{', '.join(stray)} not used by geometry {geom.name!r}; "
                         f"did you mean another geometry? (it takes {list(geom.shape) or 'no shape arguments'})")
    shape = {a: params.pop(a) for a in geom.shape}
    sig = inspect.signature(geom.kernel(modality)).parameters
    takes_any = any(p.kind is p.VAR_KEYWORD for p in sig.values())
    unknown = [k for k in params if k not in sig]
    if unknown and not takes_any:
        options = [k for k, p in sig.items() if p.default is not p.empty
                   and k not in ("freq", "z", "freq_max", "n_freq")]
        raise TypeError(f"unexpected argument(s) {unknown} for geometry {geom.name!r} ({modality.upper()}); "
                        f"kernel options are {options}.")
    return shape, params


def _check_depth(geom, shape, n_layers):
    if "depth" not in shape:
        return
    allowed = (n_layers - 1, n_layers) if geom.finite_bottom else (n_layers - 1,)
    if np.size(shape["depth"]) not in allowed:
        bottom = f", or {n_layers} for a finite bottom layer" if geom.finite_bottom else ""
        raise ValueError(f"depth must have n_layers - 1 = {n_layers - 1} entries (one per bounded "
                         f"layer){bottom}; got {np.size(shape['depth'])}.")


def _optical_per_wavelength(values, geom, n_wl, name):
    """Coerce FD/TD mua or musp to shape (n_time, n_layers, n_wl)."""
    v = np.asarray(values, dtype=float)
    base = 2 if geom.layered else 1
    if not geom.layered and v.ndim == 0 and n_wl == 1:
        v = v.reshape(1)
    if v.ndim == base:
        v = v[None]
    expect = "(n_layers, n_wavelengths)" if geom.layered else "(n_wavelengths,)"
    if v.ndim != base + 1 or v.shape[-1] != n_wl:
        raise ValueError(f"{name} must have shape {expect} = (..., {n_wl}) for geometry {geom.name!r}, "
                         f"optionally with a leading time axis; got {np.shape(values)}.")
    return v if geom.layered else v[:, None, :]


def _optical_single(values, geom, name):
    """Coerce DCS mua, musp or flow to shape (n_time, n_layers)."""
    v = np.asarray(values, dtype=float)
    base = 1 if geom.layered else 0
    if v.ndim == base:
        v = v[None]
    if v.ndim != base + 1:
        expect = "(n_layers,)" if geom.layered else "a scalar"
        raise ValueError(f"{name} must be {expect} for geometry {geom.name!r}, optionally with a "
                         f"leading time axis; got shape {np.shape(values)}.")
    return v if geom.layered else v[:, None]


def _broadcast_time(arrays, names):
    try:
        return np.broadcast_arrays(*arrays)
    except ValueError:
        sizes = ", ".join(f"{n}: {a.shape[0]}" for n, a in zip(names, arrays))
        raise ValueError(f"{', '.join(names)} have different numbers of time points ({sizes}).") from None


def _check_layers(geom, n_layers):
    if geom.n_layers not in (None, n_layers):
        raise ValueError(f"geometry {geom.name!r} has {geom.n_layers} layers; mua/musp have {n_layers}.")


def _refractive(n, geom, n_layers):
    n = np.asarray(n, dtype=float)
    if not geom.layered:
        if n.ndim:
            raise ValueError(f"n must be a scalar for geometry {geom.name!r}.")
        return float(n)
    if n.ndim == 0:
        return np.full(n_layers, float(n))
    if n.shape != (n_layers,):
        raise ValueError(f"n must be a scalar or one refractive index per layer ({n_layers}), got {n.shape}.")
    return n


def _layer(v, geom):
    """Return per-layer values as the kernel expects them: an array, or a scalar if homogeneous."""
    return v if geom.layered else float(v[0])


def _memo(compute, *rows):
    """Evaluate `compute` per time point, once per distinct set of parameters."""
    cache, out = {}, []
    for args in zip(*rows):
        key = b"".join(np.ascontiguousarray(a).tobytes() for a in args)
        if key not in cache:
            cache[key] = compute(*args)
        out.append(cache[key])
    return np.stack(out)


def _plain(v):
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return v
    return float(a) if a.ndim == 0 else a.tolist()


def _stream_name(geom, modality):
    return f"simulated_{modality}" if geom.name == "semi_infinite" else f"simulated_{geom.name}_{modality}"


# ------------------------------------------------------------------
# Simulators
# ------------------------------------------------------------------

def simulate_fd_stream(probe, mua, musp, modulation_freq, n=1.33,
                       noise_level=None, rng=None, noise_type="proportional",
                       fs=1.0, *, geometry="semi_infinite", **geometry_params):
    """
    Simulate an FD-DOS dataset for any registered geometry.

    Parameters
    ----------
    probe : Probe
        Must have channel configuration. Distances are read from
        probe.distances and converted to cm from probe.lengthUnit ('cm', 'mm'
        or 'm'; a probe without a unit is read as cm).
    mua, musp : array-like
        Absorption / reduced scattering coefficients (cm^-1).
        Homogeneous geometries: ``(n_wavelengths,)``. Layered geometries:
        ``(n_layers, n_wavelengths)``, top layer first. Either
        may have a leading time axis for a time series; `mua` and `musp`
        broadcast against each other along it, so either may be constant.
    modulation_freq : float
        Modulation frequency in Hz (e.g. 110e6). 0 gives a single DC entry
        (CW data; see ``FD_Stream.to_cw``).
    n : float or array-like
        Refractive index; for a layered geometry a scalar (shared) or one
        per layer. Default 1.33.
    noise_level : float or array-like, optional
        Relative noise (std of ln A and of the phase in rad); a list gives
        one value per channel. No noise if None. See ``noise_type``.
    rng : int or numpy.random.Generator, optional
        Seed or Generator for reproducible noise.
    noise_type : {"proportional", "shot"}
        "proportional" (default): ``noise_level`` on every channel as given.
        "shot": ``noise_level`` at the shortest channel, each channel scaled
        by sqrt(AC_ref / AC). See ``forward.noise_models.fd_noise_model``.
    fs : float
        Sampling rate (Hz) of a time series; sets the ``time`` coordinate
        (``arange(n_times) / fs``). Default 1.0.
    geometry : str, Geometry or callable
        'semi_infinite' (default), 'two_layer' or 'n_layer'; or the kernel
        function itself (e.g. ``two_layer_fd_fluence``). See `GEOMETRIES`.
    **geometry_params
        The geometry's shape arguments (``depth`` for the layered
        geometries), plus any option of its FD kernel (e.g. ``m_tol=`` for
        'two_layer', ``n_points=`` for 'n_layer'), passed through unchanged.

    Returns
    -------
    FD_Stream
        Shape (n_times, n_channels, n_wavelengths, n_freq), with
        freq=[0, modulation_freq].
    """
    from ..core.fd_nirs import FD_Stream

    geom = resolve_geometry(geometry)
    kernel = geom.kernel("fd")
    shape, opts = _split_params(geom, "fd", geometry_params)

    n_wl = len(probe.wavelengths)
    mua_t, musp_t = _broadcast_time([_optical_per_wavelength(mua, geom, n_wl, "mua"),
                                     _optical_per_wavelength(musp, geom, n_wl, "musp")], ["mua", "musp"])
    n_time, n_layers = mua_t.shape[:2]
    _check_layers(geom, n_layers)
    _check_depth(geom, shape, n_layers)
    n_k = _refractive(n, geom, n_layers)
    if "z" in inspect.signature(kernel).parameters:
        opts = {"z": 0.0, **opts}

    distances = probe_distances_cm(probe, untagged='cm')
    freqs = _fd_freq_axis(modulation_freq)

    def one_time(mua_i, musp_i):
        phi = np.zeros((len(distances), n_wl, len(freqs)), dtype=complex)
        for wi, wl in enumerate(np.asarray(probe.wavelengths, dtype=float)):
            kw = dict(mua=_layer(mua_i[:, wi], geom), musp=_layer(musp_i[:, wi], geom), n=n_k,
                      wavelength=wl, **shape, **opts)
            if "fd" in geom.vectorised:
                phi[:, wi, :] = kernel(distances, freq=freqs, **kw)
            else:
                for ci, rho in enumerate(distances):
                    for fi, f in enumerate(freqs):
                        phi[ci, wi, fi] = kernel(rho=rho, freq=f, **kw)
        # Conjugate to the library phase convention: positive imaginary part is
        # a positive phase delay (as stored by the OxiplexTS); the kernels
        # return physics-convention fluence.
        return np.conj(phi)

    data = _memo(one_time, mua_t, musp_t)

    if noise_level is not None:
        data = fd_noise_model(data, noise_level, rng=rng, noise_type=noise_type, distances=distances)

    data_xr = xr.DataArray(
        data,
        dims=['time', 'channel', 'wavelength', 'freq'],
        coords={
            'time': np.arange(n_time) / float(fs),
            'channel': probe.channel_labels,
            'wavelength': probe.wavelengths,
            'freq': freqs,
            'distance': ('channel', distances),
        },
        attrs={'status': 'raw', 'lengthUnit': 'cm', 'sampling_rate': float(fs)},
    )
    stream = FD_Stream(data=data_xr, probe=probe, name=_stream_name(geom, "fd"), status='raw')
    stream.add_history('simulate_fd_stream', {
        'geometry': geom.name, **_plain(shape),
        'mua': _plain(mua_t if n_time > 1 else mua_t[0]) if geom.layered
        else _plain(mua_t[:, 0] if n_time > 1 else mua_t[0, 0]),
        'musp': _plain(musp_t if n_time > 1 else musp_t[0]) if geom.layered
        else _plain(musp_t[:, 0] if n_time > 1 else musp_t[0, 0]),
        'n': _plain(n_k), 'modulation_freq': modulation_freq, 'fs': float(fs),
        'noise_level': None if noise_level is None else _plain(noise_level),
        'noise_type': noise_type, **_plain({k: v for k, v in opts.items() if k != 'z'}),
    })
    return stream


def simulate_td_stream(probe, mua, musp, n=1.33, bin_width=25e-12, n_bins=200,
                       noise_level=None, rng=None, freq_max=None, n_freq=None,
                       fs=1.0, *, geometry="semi_infinite", **geometry_params):
    """
    Simulate a TD-DOS (gated TPSF) dataset for any registered geometry.

    The TPSF is the inverse Fourier transform of the FD solution over a
    frequency sweep (``forward.dos._fd_sweep_to_td``). Gates are
    point-sampled at their centres and scaled by their width.

    Parameters
    ----------
    probe, mua, musp, n, fs, geometry, **geometry_params
        As in `simulate_fd_stream` (``**geometry_params`` go to the TD
        kernel, e.g. ``geometry=two_layer_td_fluence``).
    bin_width : float
        Time-gate width (s), default 25 ps.
    n_bins : int
        Number of gates, default 200 (a 5 ns window at the default width).
    noise_level : float, optional
        Proportional Gaussian noise on the gated counts (std/mean), per bin
        (not Poisson-aware). No noise if None.
    rng : int or numpy.random.Generator, optional
        Seed or Generator for reproducible noise.
    freq_max, n_freq
        Frequency sweep behind the TPSF; the time window it resolves is
        (n_freq - 1) / freq_max, which must exceed the gated window. See
        ``forward.dos._fd_sweep_to_td``.

    Returns
    -------
    TD_Stream
        Shape (n_times, n_channels, n_wavelengths, n_bins), status='raw',
        with ``timeDelays``/``timeDelayWidths`` coords (s) on ``bin``.
    """
    from ..core.td_nirs import TD_Stream

    geom = resolve_geometry(geometry)
    kernel = geom.kernel("td")
    shape, opts = _split_params(geom, "td", geometry_params)

    n_wl = len(probe.wavelengths)
    mua_t, musp_t = _broadcast_time([_optical_per_wavelength(mua, geom, n_wl, "mua"),
                                     _optical_per_wavelength(musp, geom, n_wl, "musp")], ["mua", "musp"])
    n_time, n_layers = mua_t.shape[:2]
    _check_layers(geom, n_layers)
    _check_depth(geom, shape, n_layers)
    n_k = _refractive(n, geom, n_layers)
    if "z" in inspect.signature(kernel).parameters:
        opts = {"z": 0.0, **opts}

    distances = probe_distances_cm(probe, untagged='cm')
    bin_delays, bin_widths = _default_td_gates(bin_width, n_bins)

    def one_time(mua_i, musp_i):
        counts = np.zeros((len(distances), n_wl, n_bins))
        for wi, wl in enumerate(np.asarray(probe.wavelengths, dtype=float)):
            for ci, rho in enumerate(distances):
                counts[ci, wi, :] = _bin_td_fluence(
                    kernel, bin_delays, bin_widths, rho=rho,
                    mua=_layer(mua_i[:, wi], geom), musp=_layer(musp_i[:, wi], geom), n=n_k,
                    wavelength=wl, freq_max=freq_max, n_freq=n_freq, **shape, **opts)
        return counts

    data = _apply_td_noise(_memo(one_time, mua_t, musp_t), noise_level, rng)

    data_xr = xr.DataArray(
        data,
        dims=['time', 'channel', 'wavelength', 'bin'],
        coords={
            'time': np.arange(n_time) / float(fs),
            'channel': probe.channel_labels,
            'wavelength': probe.wavelengths,
            'bin': np.arange(n_bins),
            'distance': ('channel', distances),
            'timeDelays': ('bin', bin_delays),
            'timeDelayWidths': ('bin', bin_widths),
        },
        attrs={'status': 'raw', 'lengthUnit': 'cm', 'sampling_rate': float(fs)},
    )
    stream = TD_Stream(data=data_xr, probe=probe, name=_stream_name(geom, "td"), status='raw')
    stream.add_history('simulate_td_stream', {
        'geometry': geom.name, **_plain(shape),
        'mua': _plain(mua_t if n_time > 1 else mua_t[0]) if geom.layered
        else _plain(mua_t[:, 0] if n_time > 1 else mua_t[0, 0]),
        'musp': _plain(musp_t if n_time > 1 else musp_t[0]) if geom.layered
        else _plain(musp_t[:, 0] if n_time > 1 else musp_t[0, 0]),
        'n': _plain(n_k), 'bin_width': bin_width, 'n_bins': n_bins,
        'freq_max': freq_max, 'n_freq': n_freq, 'fs': float(fs),
        'noise_level': None if noise_level is None else _plain(noise_level),
        **_plain({k: v for k, v in opts.items() if k != 'z'}),
    })
    return stream


def simulate_dcs_stream(probe, mua, musp, taus, wavelength=None, n=1.33, z=0.0,
                        aDb=None, alpha=None, Db=None, beta=1.0, motion="brownian",
                        noise_level=None, rng=None, noise_type='gaussian', noise_params=None,
                        fs=1.0, *, geometry="semi_infinite", **params):
    """
    Simulate a CW-DCS g2(tau) dataset for any registered geometry.

    Parameters
    ----------
    probe : Probe
        Must have channel configuration; distances as in `simulate_fd_stream`.
    mua, musp : float or array-like
        Absorption / reduced scattering coefficients (cm^-1) at the DCS
        wavelength: a scalar for a homogeneous geometry, ``(n_layers,)`` for
        a layered one. A leading time axis gives a time series.
    taus : array-like
        Correlation delays (s).
    wavelength : float, optional
        Wavelength (nm). If None, read from probe.wavelengths[0].
    n : float or array-like
        Refractive index; for a layered geometry a scalar (shared) or one
        per layer. Default 1.33.
    z : float
        Detector depth (cm), for geometries whose kernel takes one; z=0 is
        the boundary.
    aDb : float or array-like, optional
        alpha*Db, the flow index, shaped like `mua` (per layer for a layered
        geometry, with an optional time axis). Mutually exclusive with
        ``alpha``/``Db``.
    alpha, Db : float or array-like, optional
        Decomposed ground truth: fraction of moving scatterers and the
        Brownian diffusion coefficient, shaped like `aDb`. Recorded in
        history.
    beta : float
        Siegert coherence factor.
    motion : str
        Motion submodel, shared across layers, see ``forward.dynamics.msd``.
    noise_level : float or array-like, optional
        Proportional Gaussian noise on g2 (std/mean), one value or one per
        channel; used when ``noise_type='gaussian'``. No noise if None.
    rng : int or numpy.random.Generator, optional
        Seed or Generator for reproducible noise, for either ``noise_type``.
    noise_type : {'gaussian', 'tau_dependent'}
        'gaussian' (default): proportional noise set by ``noise_level``.
        'tau_dependent': per-lag photon-counting noise
        (``noise_models.zhou_noise_model``) set by ``noise_params``.
    noise_params : dict, optional
        Required for ``noise_type='tau_dependent'``: ``t_int`` (s) and
        ``intensity`` (counts/s, scalar or one per channel); optional
        ``correlator_type`` ('linear' or 'multi_tau'). ``beta`` always comes
        from the argument above.
    fs : float
        Sampling rate (Hz) of a time series. Default 1.0.
    geometry : str, Geometry or callable
        As in `simulate_fd_stream`; the kernel form is e.g.
        ``two_layer_dcs_g1``.
    **params
        The geometry's shape arguments, any option of its DCS kernel (series
        settings), and motion-submodel parameters (e.g. ``tc=`` for
        "langevin").

    Returns
    -------
    DCS_Stream
        Shape (n_times, n_channels, 1, n_taus).
    """
    from ..core.dcs_stream import DCS_Stream

    geom = resolve_geometry(geometry)
    kernel = geom.kernel("dcs")
    shape, opts = _split_params(geom, "dcs", params)

    if (aDb is None) == (alpha is None and Db is None):
        raise ValueError("Provide exactly one of `aDb` or (`alpha` and `Db`).")
    decomposed = aDb is None
    if decomposed:
        if alpha is None or Db is None:
            raise ValueError("Both `alpha` and `Db` are required together.")
        aDb = dynamics.compose(np.asarray(alpha, dtype=float), np.asarray(Db, dtype=float))

    if wavelength is None:
        wavelength = float(probe.wavelengths[0])

    mua_t, musp_t, aDb_t = _broadcast_time([_optical_single(mua, geom, "mua"),
                                            _optical_single(musp, geom, "musp"),
                                            _optical_single(aDb, geom, "the flow")],
                                           ["mua", "musp", "the flow"])
    n_time, n_layers = mua_t.shape
    _check_layers(geom, n_layers)
    _check_depth(geom, shape, n_layers)
    n_k = _refractive(n, geom, n_layers)
    if "z" in inspect.signature(kernel).parameters:
        opts = {"z": z, **opts}
    elif z != 0.0:
        raise ValueError(f"z (detector depth) is only supported for geometries whose kernel takes it; "
                         f"{geom.name!r} places the detector on the surface.")

    distances = probe_distances_cm(probe, untagged='cm')
    taus = np.asarray(taus, dtype=float)
    n_ch, n_tau = len(distances), len(taus)

    def one_time(mua_i, musp_i, aDb_i):
        kw = dict(mua=_layer(mua_i, geom), musp=_layer(musp_i, geom), n=n_k, wavelength=wavelength,
                  aDb=_layer(aDb_i, geom), motion=motion, **shape, **opts)
        if "dcs" in geom.vectorised:
            g1 = np.reshape(kernel(distances, tau=taus, **kw), (n_ch, n_tau))
        else:
            g1 = np.array([[kernel(rho=rho, tau=tau, **kw) for tau in taus] for rho in distances])
        return (1.0 + beta * np.abs(g1) ** 2)[:, None, :]

    data = _memo(one_time, mua_t, musp_t, aDb_t)

    if noise_type == 'gaussian':
        if noise_level is not None:
            level = np.asarray(noise_level, dtype=float)
            if level.ndim == 1:
                if level.shape != (n_ch,):
                    raise ValueError(f"noise_level has {level.size} entries; expected a "
                                     f"scalar or one per channel ({n_ch}).")
                level = level.reshape(1, n_ch, 1, 1)
            rng = np.random.default_rng(rng)
            data = data + rng.standard_normal(data.shape) * data * level
    elif noise_type == 'tau_dependent':
        data = _apply_tau_dependent_noise(data, taus, beta, rng, noise_params)
    else:
        raise ValueError(f"noise_type must be 'gaussian' or 'tau_dependent', got {noise_type!r}")

    data_xr = xr.DataArray(
        data,
        dims=['time', 'channel', 'wavelength', 'tau'],
        coords=_simulated_stream_coords(probe, taus, wavelength, distances,
                                        time=np.arange(n_time) / float(fs)),
        attrs={'status': 'raw', 'lengthUnit': 'cm', 'observation': 'g2',
               'sampling_rate': float(fs)},
    )
    stream = DCS_Stream(data=data_xr, probe=probe, name=_stream_name(geom, "dcs"), status='raw')

    def as_given(v):
        # a scalar or per-layer value stays as passed; a time series is recorded per time point
        return _plain(v[:, 0] if not geom.layered else v) if n_time > 1 else _plain(_layer(v[0], geom))

    history = {'geometry': geom.name, **_plain(shape),
               'mua': as_given(mua_t), 'musp': as_given(musp_t), 'n': _plain(n_k), 'z': z,
               'beta': beta, 'motion': motion, 'aDb': as_given(aDb_t), 'fs': float(fs),
               'noise_level': None if noise_level is None else _plain(noise_level),
               'noise_type': noise_type, **_plain({k: v for k, v in opts.items() if k != 'z'})}
    if decomposed:
        history.update({'alpha': _plain(alpha), 'Db': _plain(Db)})
    stream.add_history('simulate_dcs_stream', history)
    return stream
