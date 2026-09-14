"""
DCS forward models.

Thin assembly of the shared geometry kernels and dispersion relation.
These return the field autocorrelation g1, never g2: the Siegert relation
is a separate observation step, so the same solver serves conventional and
interferometric DCS.
"""

import numpy as np
import xarray as xr

from . import kernels
from . import dispersion
from . import dynamics
from . import noise_models
from .dos import _layer_R_eff, _two_layer_series_kwargs, _n_layer_series_kwargs


def si_dcs_g1(rho, mua, musp, n, wavelength, tau, aDb, z=0.0, motion="brownian",
              R_eff=None, **motion_params):
    """
    Semi-infinite DCS field autocorrelation.

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
        Wavelength in nm.
    tau : float or array-like
        Correlation delay in seconds.
    aDb : float
        Flow index in cm^2/s. Ignored for random motion, which takes ``aV2``
        through ``motion_params`` instead.
    z : float
        Detector depth in cm, with 0 the boundary.
    motion : str
        Scatterer-motion submodel.
    R_eff : float, optional
        Effective boundary reflectance. Computed from ``n`` if None.
    **motion_params
        Forwarded to the motion submodel, e.g. ``tc`` or ``aV2``.

    Returns
    -------
    np.ndarray of complex
        Field autocorrelation normalised so that g1(0) = 1, broadcast over
        ``rho`` and ``tau``.
    """
    rho = np.asarray(rho, dtype=float)
    tau = np.asarray(tau, dtype=float)
    scalar_rho = rho.ndim == 0
    scalar_tau = tau.ndim == 0
    rho = np.atleast_1d(rho)
    tau = np.atleast_1d(tau)

    if R_eff is None:
        R_eff = kernels.Reff(n, 1.0)

    if motion == "brownian" or motion == "langevin":
        motion_params = {"aDb": aDb, **motion_params}

    K2_tau = dispersion.k2(mua, musp, n, wavelength, tau=tau, motion=motion, **motion_params)
    K_tau = np.sqrt(K2_tau.astype(complex))
    K_0 = np.sqrt(dispersion.k2(mua, musp, n, wavelength, tau=0.0).astype(complex))

    # Broadcasting: rho (n_ch, 1) x K (1, n_tau) -> (n_ch, n_tau)
    G1_tau = kernels.si_kernel(rho[:, np.newaxis], K_tau[np.newaxis, :], mua, musp, R_eff, z=z)
    G1_0 = kernels.si_kernel(rho[:, np.newaxis], K_0, mua, musp, R_eff, z=z)

    g1 = G1_tau / G1_0

    if scalar_rho and scalar_tau:
        return g1[0, 0]
    if scalar_rho:
        return g1[0]
    if scalar_tau:
        return g1[:, 0]
    return g1


def two_layer_dcs_g1(rho, z, mua, musp, n, wavelength, depth, tau, aDb,
                      motion="brownian", R_eff=None, a=30.0, m=4000, **motion_params):
    """
    Two-layer DCS field autocorrelation.

    Flow is specified per layer, which is the point of a layered model: a
    near-static superficial layer over a perfused one separates extracerebral
    from cerebral flow. Evaluates one delay per call, unlike the semi-infinite
    model.

    Parameters
    ----------
    rho : float
        Source-detector separation in cm.
    z : float
        Detector depth in cm, with 0 the boundary.
    mua, musp, n : sequence of float
        Absorption in cm^-1, reduced scattering in cm^-1, and refractive
        index, one per layer.
    wavelength : float
        Wavelength in nm.
    depth : sequence of float
        Thickness in cm of each bounded layer.
    tau : float
        Correlation delay in seconds.
    aDb : sequence of float
        Flow index per layer in cm^2/s.
    motion : str
        Scatterer-motion submodel, shared across layers.
    R_eff : sequence of float, optional
        Effective reflectance at each boundary. Computed from ``n`` if None.
    a : float
        Radius in cm truncating the Fourier-Bessel series.
    m : int
        Number of terms in the series.
    **motion_params
        Forwarded to the motion submodel, per layer where they differ.

    Returns
    -------
    complex
        Field autocorrelation normalised so that g1(0) = 1.
    """
    mua = np.asarray(mua, dtype=float)
    musp = np.asarray(musp, dtype=float)
    n = np.asarray(n, dtype=float)
    aDb = np.asarray(aDb, dtype=float)
    n_layers = len(mua)

    if R_eff is None:
        R_eff = _layer_R_eff(n)

    def k_sq_layers(tau_value):
        K_sq = []
        for i in range(n_layers):
            layer_params = dict(motion_params)
            if motion == "brownian" or motion == "langevin":
                layer_params["aDb"] = aDb[i]
            K_sq.append(complex(dispersion.k2(mua[i], musp[i], n[i], wavelength,
                                               tau=tau_value, motion=motion, **layer_params)))
        return K_sq

    K_sq_tau = k_sq_layers(tau)
    K_sq_0 = k_sq_layers(0.0)

    G1_tau = kernels.two_layer_kernel(rho, z, K_sq_tau, mua, musp, n, depth, R_eff, a=a, m=m)
    G1_0 = kernels.two_layer_kernel(rho, z, K_sq_0, mua, musp, n, depth, R_eff, a=a, m=m)

    return complex(G1_tau) / complex(G1_0)


def n_layer_dcs_g1(rho, z, mua, musp, n, wavelength, depth, tau, aDb,
                    motion="brownian", R_eff_top=None, R_eff_bottom=None,
                    s_max_factor=30.0, n_points=480, **motion_params):
    """
    General N-layer DCS field autocorrelation.

    Flow is specified per layer. Evaluates one delay per call.

    Parameters
    ----------
    rho : float
        Source-detector separation in cm.
    z : float
        Detector depth in cm, with 0 the boundary.
    mua, musp, n : sequence of float
        Absorption in cm^-1, reduced scattering in cm^-1, and refractive
        index, one per layer.
    wavelength : float
        Wavelength in nm.
    depth : sequence of float
        Thickness in cm of each bounded layer.
    tau : float
        Correlation delay in seconds.
    aDb : sequence of float
        Flow index per layer in cm^2/s.
    motion : str
        Scatterer-motion submodel, shared across layers.
    R_eff_top : float, optional
        Effective reflectance at the top boundary.
    R_eff_bottom : float, optional
        Effective reflectance at the base. None makes the base semi-infinite.
    s_max_factor : float
        Upper integration limit of the Hankel transform.
    n_points : int
        Number of quadrature points.
    **motion_params
        Forwarded to the motion submodel.

    Returns
    -------
    complex
        Field autocorrelation normalised so that g1(0) = 1.
    """
    mua = np.asarray(mua, dtype=float)
    musp = np.asarray(musp, dtype=float)
    n = np.asarray(n, dtype=float)
    aDb = np.asarray(aDb, dtype=float)
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

    def k_sq_layers(tau_value):
        K_sq = []
        for i in range(n_layers):
            layer_params = dict(motion_params)
            if motion == "brownian" or motion == "langevin":
                layer_params["aDb"] = aDb[i]
            K_sq.append(complex(dispersion.k2(mua[i], musp[i], n[i], wavelength,
                                               tau=tau_value, motion=motion, **layer_params)))
        return K_sq

    K_sq_tau = k_sq_layers(tau)
    K_sq_0 = k_sq_layers(0.0)

    G1_tau = kernels.n_layer_kernel(rho, z, K_sq_tau, mua, musp, n, depth, R_eff_top,
                                     R_eff_bottom=R_eff_bottom, s_max_factor=s_max_factor, n_points=n_points)
    G1_0 = kernels.n_layer_kernel(rho, z, K_sq_0, mua, musp, n, depth, R_eff_top,
                                   R_eff_bottom=R_eff_bottom, s_max_factor=s_max_factor, n_points=n_points)

    return complex(G1_tau) / complex(G1_0)


# ------------------------------------------------------------------
# `assemble` callbacks for processing.fitting.dcs_g1_model_opt + the
# layered/curved *_dcs_g1 models above. Each maps the flat, merged
# parameter dict (forward_parameters + fitted + fixed -- including `beta`)
# onto the array-shaped kwargs the corresponding *_dcs_g1 function expects.
# `n_i` (refractive indices) are always fixed context, never fit -- same
# convention as the FD `assemble_*` callbacks in `forward.dos`. `beta`
# (needed only for the default `observation="siegert"`) is deliberately
# never included in what these return: it belongs to the observation step,
# not the forward model -- `dcs_g1_model_opt` reads it from the flat dict
# directly, same as it does for `assemble=None` (si_dcs_g1).
# ------------------------------------------------------------------

def _dcs_motion_kwargs(flat):
    """Extract the optional motion-submodel settings from a parameter dict."""
    kwargs = {}
    for key in ("tc", "aV2"):
        if key in flat:
            kwargs[key] = flat[key]
    return kwargs


def assemble_two_layer_dcs(flat):
    """
    Map flat two-layer parameters onto the DCS model's arguments.

    Fits every optical property, each layer's flow, and the layer thickness.

    Parameters
    ----------
    flat : dict
        Merged fixed, fitted and context parameters.

    Returns
    -------
    dict
        Keyword arguments for :func:`two_layer_dcs_g1`.
    """
    return {
        "rho": flat["rho"],
        "mua": [flat["mua_1"], flat["mua_2"]],
        "musp": [flat["musp_1"], flat["musp_2"]],
        "n": [flat["n_1"], flat["n_2"]],
        "depth": [flat["depth"]],
        "wavelength": flat["wavelength"],
        "z": flat.get("z", 0.0),
        "aDb": [flat["aDb_1"], flat["aDb_2"]],
        "motion": flat.get("motion", "brownian"),
        **_dcs_motion_kwargs(flat),
        **_two_layer_series_kwargs(flat),
    }


TWO_LAYER_DCS_PARAM_CONFIG = {
    "mua_1": {"bounds": (1e-3, 0.5), "log": True},
    "musp_1": {"bounds": (1.0, 30.0), "log": True},
    "mua_2": {"bounds": (1e-3, 0.5), "log": True},
    "musp_2": {"bounds": (1.0, 30.0), "log": True},
    "depth": {"bounds": (0.2, 3.0), "log": False},
    "aDb_1": {"bounds": (1e-10, 1e-6), "log": True},
    "aDb_2": {"bounds": (1e-10, 1e-6), "log": True},
    "beta": {"bounds": (0.0, 1.0), "log": False},
}
"""Default param_config for assemble_two_layer_dcs (fit every unknown,
including depth and beta). Pass any of these via fixed_params to pin it
instead -- e.g. a known beta from a calibration measurement."""


def make_assemble_n_layer_dcs(n_layers, finite_bottom=False):
    """
    Build an assembler for fitting an N-layer DCS model with everything free.

    A factory rather than a function, since the flat parameter names depend on
    the layer count.

    Parameters
    ----------
    n_layers : int
        Number of layers.
    finite_bottom : bool
        False (default) leaves the base semi-infinite. True makes it finite,
        adding a thickness parameter.

    Returns
    -------
    callable
        Maps a flat parameter dict to keyword arguments for
        :func:`n_layer_dcs_g1`.
    """
    n_depths = n_layers if finite_bottom else n_layers - 1

    def assemble(flat):
        return {
            "rho": flat["rho"],
            "mua": [flat[f"mua_{i}"] for i in range(1, n_layers + 1)],
            "musp": [flat[f"musp_{i}"] for i in range(1, n_layers + 1)],
            "n": [flat[f"n_{i}"] for i in range(1, n_layers + 1)],
            "depth": [flat[f"depth_{i}"] for i in range(1, n_depths + 1)],
            "wavelength": flat["wavelength"],
            "z": flat.get("z", 0.0),
            "aDb": [flat[f"aDb_{i}"] for i in range(1, n_layers + 1)],
            "motion": flat.get("motion", "brownian"),
            **_dcs_motion_kwargs(flat),
            **_n_layer_series_kwargs(flat),
        }

    return assemble


def make_n_layer_dcs_param_config(n_layers, finite_bottom=False,
                                   mua_bounds=(1e-3, 0.5), musp_bounds=(1.0, 30.0),
                                   depth_bounds=(0.2, 3.0), aDb_bounds=(1e-10, 1e-6)):
    """
    Build a parameter configuration matching :func:`make_assemble_n_layer_dcs`.

    Parameters
    ----------
    n_layers : int
        Number of layers.
    finite_bottom : bool
        Whether the base layer is finite. Default False.
    mua_bounds, musp_bounds, depth_bounds, aDb_bounds : tuple of (float, float)
        Bounds applied to every layer's absorption, scattering, thickness and
        flow.

    Returns
    -------
    dict
        Free parameters and their bounds, including the coherence factor.
    """
    n_depths = n_layers if finite_bottom else n_layers - 1
    config = {"beta": {"bounds": (0.0, 1.0), "log": False}}
    for i in range(1, n_layers + 1):
        config[f"mua_{i}"] = {"bounds": mua_bounds, "log": True}
        config[f"musp_{i}"] = {"bounds": musp_bounds, "log": True}
        config[f"aDb_{i}"] = {"bounds": aDb_bounds, "log": True}
    for i in range(1, n_depths + 1):
        config[f"depth_{i}"] = {"bounds": depth_bounds, "log": False}
    return config


def _apply_tau_dependent_noise(data, taus, beta, rng, noise_params):
    """
    Apply the photon-counting noise model to a simulated correlation array.

    The coherence factor used to recover the field autocorrelation is always
    the one that generated the clean curve. A single generator is shared
    across channels, so their draws are independent but reproducible.

    Parameters
    ----------
    data : np.ndarray
        Shape (1, n_channels, 1, n_taus), the clean g2 curves.
    taus : np.ndarray
        Correlation delays in seconds.
    beta : float
        Coherence factor.
    rng : int or numpy.random.Generator, optional
        Seed or generator.
    noise_params : dict
        Integration time and photon count rate. The rate may be a scalar or
        one value per channel, since a more distant detector collects fewer
        photons.

    Returns
    -------
    np.ndarray
        Noisy curves, the same shape as ``data``.
    """
    if noise_params is None:
        raise ValueError(
            "noise_params is required when noise_type='tau_dependent' -- "
            "needs 't_int' and 'intensity'; 'correlator_type' is optional."
        )
    if 'intensity' not in noise_params:
        raise ValueError(
            "noise_params must include 'intensity' (photon count rate, "
            "counts/s) -- a scalar shared across channels, or an array-like "
            "with one entry per channel."
        )

    rng = np.random.default_rng(rng)
    noisy = data.copy()
    n_ch = data.shape[1]

    intensity = np.atleast_1d(np.asarray(noise_params['intensity'], dtype=float))
    if intensity.size == 1:
        intensity = np.full(n_ch, intensity[0])
    elif intensity.size != n_ch:
        raise ValueError(
            f"noise_params['intensity'] has {intensity.size} entries but "
            f"there are {n_ch} channels -- pass a scalar (shared across "
            f"channels) or an array with exactly one entry per channel."
        )
    other_params = {k: v for k, v in noise_params.items() if k != 'intensity'}

    for ci in range(n_ch):
        noisy[0, ci, 0, :] = noise_models.zhou_noise_model(
            data[0, ci, 0, :], taus, rng=rng, beta=beta,
            intensity=intensity[ci], **other_params
        )
    return noisy


def _simulated_stream_coords(probe, taus, wavelength, distances):
    """
    Build the coordinate dict shared by the DCS simulators.

    Includes the optode IDs whenever the probe carries a channel
    configuration, matching what the file readers attach, so that a simulated
    stream can be passed to channel aggregation.

    Parameters
    ----------
    probe : Probe
        Probe the stream is built on.
    taus : array-like
        Correlation delays in seconds.
    wavelength : float
        Wavelength in nm.
    distances : array-like
        Source-detector distances.

    Returns
    -------
    dict
        Coordinates for the simulated stream.
    """
    coords = {
        'time': [0.0],
        'channel': probe.channel_labels,
        'wavelength': [wavelength],
        'tau': taus,
        'distance': ('channel', distances),
    }
    if probe.has_channels:
        channel_table = probe.list_channels()
        coords['source'] = ('channel', np.asarray(channel_table['source']))
        coords['detector'] = ('channel', np.asarray(channel_table['detector']))
    return coords


def simulate_dcs_stream(probe, mua, musp, taus, wavelength=None, n=1.33, z=0.0,
                         aDb=None, alpha=None, Db=None, beta=1.0, motion="brownian",
                         noise_level=None, rng=None, noise_type='gaussian', noise_params=None,
                         **motion_params):
    """
    Simulate a CW-DCS dataset for a semi-infinite homogeneous medium.

    Parameters
    ----------
    probe : Probe
        Must carry a channel configuration, with distances in cm.
    mua, musp : float
        Absorption and reduced scattering in cm^-1, shared across the probe's
        wavelengths. Pass arrays if they vary by wavelength.
    taus : array-like
        Correlation delays in seconds.
    wavelength : float, optional
        Wavelength in nm. Read from the probe if omitted.
    n : float
        Refractive index. Default 1.33.
    z : float
        Detector depth in cm, with 0 the boundary.
    aDb : float, optional
        The bundled flow index in cm^2/s. Mutually exclusive with ``alpha``
        and ``Db``.
    alpha, Db : float, optional
        Moving-scatterer fraction and diffusion coefficient, recorded
        separately in the history even though only their product is
        physically active. Mutually exclusive with ``aDb``.
    beta : float
        Coherence factor of the Siegert relation.
    motion : str
        Scatterer-motion submodel.
    noise_level : float, optional
        Proportional Gaussian noise on g2, as a fraction of the mean. Used
        only when ``noise_type`` is 'gaussian'. None adds none.
    rng : int or numpy.random.Generator, optional
        Seed or generator for reproducible noise.
    noise_type : {'gaussian', 'tau_dependent'}
        'gaussian' applies proportional noise; 'tau_dependent' applies the
        photon-counting model, which needs ``noise_params``.
    noise_params : dict, optional
        Settings for the photon-counting model: integration time, and a
        photon count rate that may be scalar or one value per channel.

    Returns
    -------
    DCS_Stream
        Shape (1, n_channels, n_wavelengths, n_taus), holding g2.
    """
    from ..core.dcs_stream import DCS_Stream

    if (aDb is None) == (alpha is None and Db is None):
        raise ValueError("Provide exactly one of `aDb` or (`alpha` and `Db`).")

    decomposed = aDb is None
    if decomposed:
        if alpha is None or Db is None:
            raise ValueError("Both `alpha` and `Db` are required together.")
        aDb = dynamics.compose(alpha, Db)

    if wavelength is None:
        wavelength = float(probe.wavelengths[0])

    distances = np.array(probe.distances, dtype=float)
    n_ch = len(distances)
    taus = np.asarray(taus, dtype=float)
    n_tau = len(taus)

    g1 = si_dcs_g1(distances, mua, musp, n, wavelength, taus, aDb, z=z,
                    motion=motion, **motion_params)
    g2 = 1.0 + beta * np.abs(g1) ** 2

    data = g2.reshape(1, n_ch, 1, n_tau)

    if noise_type == 'gaussian':
        if noise_level is not None:
            rng = np.random.default_rng(rng)
            data = data + rng.standard_normal(data.shape) * data * noise_level
    elif noise_type == 'tau_dependent':
        data = _apply_tau_dependent_noise(data, taus, beta, rng, noise_params)
    else:
        raise ValueError(f"noise_type must be 'gaussian' or 'tau_dependent', got {noise_type!r}")

    data_xr = xr.DataArray(
        data,
        dims=['time', 'channel', 'wavelength', 'tau'],
        coords=_simulated_stream_coords(probe, taus, wavelength, distances),
        attrs={'status': 'raw', 'lengthUnit': 'cm', 'observation': 'g2'},
    )

    stream = DCS_Stream(data=data_xr, probe=probe, name='simulated_dcs', status='raw')

    history_params = {'mua': mua, 'musp': musp, 'n': n, 'z': z, 'beta': beta,
                       'motion': motion, 'aDb': aDb, **motion_params}
    if decomposed:
        history_params.update({'alpha': alpha, 'Db': Db})
    stream.add_history('simulate_dcs_stream', history_params)

    return stream


def _dcs_alpha_Db_to_aDb(aDb, alpha, Db):
    """Resolve a flow index from either the bundled value or its two factors."""
    if (aDb is None) == (alpha is None and Db is None):
        raise ValueError("Provide exactly one of `aDb` or (`alpha` and `Db`).")
    decomposed = aDb is None
    if decomposed:
        if alpha is None or Db is None:
            raise ValueError("Both `alpha` and `Db` are required together.")
        aDb = dynamics.compose(np.asarray(alpha, dtype=float), np.asarray(Db, dtype=float))
    return np.asarray(aDb, dtype=float), decomposed


def simulate_two_layer_dcs_stream(probe, mua, musp, n, depth, taus, wavelength=None,
                                   aDb=None, alpha=None, Db=None, beta=1.0, motion="brownian",
                                   noise_level=None, rng=None, noise_type='gaussian', noise_params=None,
                                   a=30.0, m=4000, **motion_params):
    """
    Simulate a CW-DCS dataset for a two-layer medium.

    Parameters
    ----------
    probe : Probe
        Must carry a channel configuration, with distances in cm.
    mua, musp : array-like
        Absorption and reduced scattering per layer in cm^-1.
    n : array-like
        Refractive index per layer.
    depth : array-like
        Thickness in cm of each bounded layer.
    taus : array-like
        Correlation delays in seconds.
    wavelength : float, optional
        Wavelength in nm. Read from the probe if omitted.
    aDb : float, optional
        The bundled flow index in cm^2/s. Mutually exclusive with ``alpha``
        and ``Db``.
    alpha, Db : float, optional
        Moving-scatterer fraction and diffusion coefficient, recorded
        separately in the history even though only their product is
        physically active. Mutually exclusive with ``aDb``.
    beta : float
        Coherence factor of the Siegert relation.
    motion : str
        Scatterer-motion submodel.
    noise_level : float, optional
        Proportional Gaussian noise on g2, as a fraction of the mean. Used
        only when ``noise_type`` is 'gaussian'. None adds none.
    rng : int or numpy.random.Generator, optional
        Seed or generator for reproducible noise.
    noise_type : {'gaussian', 'tau_dependent'}
        'gaussian' applies proportional noise; 'tau_dependent' applies the
        photon-counting model, which needs ``noise_params``.
    noise_params : dict, optional
        Settings for the photon-counting model: integration time, and a
        photon count rate that may be scalar or one value per channel.
    a : float
        Radius in cm truncating the Fourier-Bessel series.
    m : int
        Number of terms in the series.

    Returns
    -------
    DCS_Stream
        Shape (1, n_channels, n_wavelengths, n_taus), holding g2.
    """
    from ..core.dcs_stream import DCS_Stream

    aDb, decomposed = _dcs_alpha_Db_to_aDb(aDb, alpha, Db)
    mua = np.asarray(mua, dtype=float)
    musp = np.asarray(musp, dtype=float)
    n = np.asarray(n, dtype=float)

    if wavelength is None:
        wavelength = float(probe.wavelengths[0])

    distances = np.array(probe.distances, dtype=float)
    n_ch = len(distances)
    taus = np.asarray(taus, dtype=float)
    n_tau = len(taus)

    g1 = np.zeros((n_ch, n_tau), dtype=complex)
    for ci, rho in enumerate(distances):
        for ti, tau in enumerate(taus):
            g1[ci, ti] = two_layer_dcs_g1(rho, 0.0, mua, musp, n, wavelength, depth, tau, aDb,
                                           motion=motion, a=a, m=m, **motion_params)
    g2 = 1.0 + beta * np.abs(g1) ** 2

    data = g2.reshape(1, n_ch, 1, n_tau)

    if noise_type == 'gaussian':
        if noise_level is not None:
            rng = np.random.default_rng(rng)
            data = data + rng.standard_normal(data.shape) * data * noise_level
    elif noise_type == 'tau_dependent':
        data = _apply_tau_dependent_noise(data, taus, beta, rng, noise_params)
    else:
        raise ValueError(f"noise_type must be 'gaussian' or 'tau_dependent', got {noise_type!r}")

    data_xr = xr.DataArray(
        data,
        dims=['time', 'channel', 'wavelength', 'tau'],
        coords=_simulated_stream_coords(probe, taus, wavelength, distances),
        attrs={'status': 'raw', 'lengthUnit': 'cm', 'observation': 'g2'},
    )

    stream = DCS_Stream(data=data_xr, probe=probe, name='simulated_two_layer_dcs', status='raw')

    history_params = {'mua': mua, 'musp': musp, 'n': n, 'depth': depth, 'beta': beta,
                       'motion': motion, 'aDb': aDb, **motion_params}
    if decomposed:
        history_params.update({'alpha': alpha, 'Db': Db})
    stream.add_history('simulate_two_layer_dcs_stream', history_params)

    return stream


def simulate_n_layer_dcs_stream(probe, mua, musp, n, depth, taus, wavelength=None,
                                 aDb=None, alpha=None, Db=None, beta=1.0, motion="brownian",
                                 noise_level=None, rng=None, noise_type='gaussian', noise_params=None,
                                 s_max_factor=30.0, n_points=480, **motion_params):
    """
    Simulate a CW-DCS dataset for a general N-layer medium.

    Parameters
    ----------
    probe : Probe
        Must carry a channel configuration, with distances in cm.
    mua, musp : array-like
        Absorption and reduced scattering per layer in cm^-1.
    n : array-like
        Refractive index per layer.
    depth : array-like
        Thickness in cm of each bounded layer.
    taus : array-like
        Correlation delays in seconds.
    wavelength : float, optional
        Wavelength in nm. Read from the probe if omitted.
    aDb : float, optional
        The bundled flow index in cm^2/s. Mutually exclusive with ``alpha``
        and ``Db``.
    alpha, Db : float, optional
        Moving-scatterer fraction and diffusion coefficient, recorded
        separately in the history even though only their product is
        physically active. Mutually exclusive with ``aDb``.
    beta : float
        Coherence factor of the Siegert relation.
    motion : str
        Scatterer-motion submodel.
    noise_level : float, optional
        Proportional Gaussian noise on g2, as a fraction of the mean. Used
        only when ``noise_type`` is 'gaussian'. None adds none.
    rng : int or numpy.random.Generator, optional
        Seed or generator for reproducible noise.
    noise_type : {'gaussian', 'tau_dependent'}
        'gaussian' applies proportional noise; 'tau_dependent' applies the
        photon-counting model, which needs ``noise_params``.
    noise_params : dict, optional
        Settings for the photon-counting model: integration time, and a
        photon count rate that may be scalar or one value per channel.
    s_max_factor : float
        Upper integration limit of the Hankel transform.
    n_points : int
        Number of quadrature points.

    Returns
    -------
    DCS_Stream
        Shape (1, n_channels, n_wavelengths, n_taus), holding g2.
    """
    from ..core.dcs_stream import DCS_Stream

    aDb, decomposed = _dcs_alpha_Db_to_aDb(aDb, alpha, Db)
    mua = np.asarray(mua, dtype=float)
    musp = np.asarray(musp, dtype=float)
    n = np.asarray(n, dtype=float)

    if wavelength is None:
        wavelength = float(probe.wavelengths[0])

    distances = np.array(probe.distances, dtype=float)
    n_ch = len(distances)
    taus = np.asarray(taus, dtype=float)
    n_tau = len(taus)

    g1 = np.zeros((n_ch, n_tau), dtype=complex)
    for ci, rho in enumerate(distances):
        for ti, tau in enumerate(taus):
            g1[ci, ti] = n_layer_dcs_g1(rho, 0.0, mua, musp, n, wavelength, depth, tau, aDb,
                                         motion=motion, s_max_factor=s_max_factor, n_points=n_points,
                                         **motion_params)
    g2 = 1.0 + beta * np.abs(g1) ** 2

    data = g2.reshape(1, n_ch, 1, n_tau)

    if noise_type == 'gaussian':
        if noise_level is not None:
            rng = np.random.default_rng(rng)
            data = data + rng.standard_normal(data.shape) * data * noise_level
    elif noise_type == 'tau_dependent':
        data = _apply_tau_dependent_noise(data, taus, beta, rng, noise_params)
    else:
        raise ValueError(f"noise_type must be 'gaussian' or 'tau_dependent', got {noise_type!r}")

    data_xr = xr.DataArray(
        data,
        dims=['time', 'channel', 'wavelength', 'tau'],
        coords=_simulated_stream_coords(probe, taus, wavelength, distances),
        attrs={'status': 'raw', 'lengthUnit': 'cm', 'observation': 'g2'},
    )

    stream = DCS_Stream(data=data_xr, probe=probe, name='simulated_n_layer_dcs', status='raw')

    history_params = {'mua': mua, 'musp': musp, 'n': n, 'depth': depth, 'beta': beta,
                       'motion': motion, 'aDb': aDb, **motion_params}
    if decomposed:
        history_params.update({'alpha': alpha, 'Db': Db})
    stream.add_history('simulate_n_layer_dcs_stream', history_params)

    return stream


