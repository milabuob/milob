"""
DCS forward models.

Thin assembly of the shared geometry kernels and dispersion relation.
These return the field autocorrelation g1, never g2: the Siegert relation
is a separate observation step, so the same solver serves conventional and
interferometric DCS.
"""

import numpy as np

from . import kernels
from . import dispersion
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
                      motion="brownian", R_eff=None, a=30.0, m=8000, m_tol=1e-8, **motion_params):
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
        Maximum number of terms in the series.
    m_tol : float
        Relative tolerance at which the series stops; see
        :func:`~milob.forward.kernels.two_layer_kernel`.
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

    G1_tau = kernels.two_layer_kernel(rho, z, K_sq_tau, mua, musp, n, depth, R_eff, a=a, m=m, m_tol=m_tol)
    G1_0 = kernels.two_layer_kernel(rho, z, K_sq_0, mua, musp, n, depth, R_eff, a=a, m=m, m_tol=m_tol)

    return complex(G1_tau) / complex(G1_0)


def n_layer_dcs_g1(rho, z, mua, musp, n, wavelength, depth, tau, aDb,
                    motion="brownian", R_eff_top=None, R_eff_bottom=None,
                    s_max_factor=30.0, n_points=None, **motion_params):
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
    n_points : int, optional
        Number of quadrature points. None (default) chooses it from the
        separation; see :func:`~milob.forward.kernels.n_layer_kernel`.
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
        Shape (n_times, n_channels, 1, n_taus), the clean g2 curves.
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

    for ti in range(data.shape[0]):
        for ci in range(n_ch):
            noisy[ti, ci, 0, :] = noise_models.zhou_noise_model(
                data[ti, ci, 0, :], taus, rng=rng, beta=beta,
                intensity=intensity[ci], **other_params
            )
    return noisy


def _simulated_stream_coords(probe, taus, wavelength, distances, time=None):
    """
    Build the coordinate dict of a simulated DCS stream.

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
    time : array-like, optional
        Time coordinate. Defaults to a single sample at 0.

    Returns
    -------
    dict
        Coordinates for the simulated stream.
    """
    coords = {
        'time': [0.0] if time is None else time,
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


