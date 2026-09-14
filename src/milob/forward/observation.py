"""
Observation operators mapping a forward model's field quantity to what an
instrument records.

Kept separate from the forward models so that one solver can serve both
conventional and interferometric DCS.
"""

import numpy as np
from scipy.special import erfc


#: Fitted parameters that belong to B, not to the medium. When a fit is
#: unpacked into a ``core.opt_prop_stream.OptPropStream`` these are routed
#: to ``obs_params`` rather than onto the ``op`` axis, so ``op`` holds only
#: recovered properties of the tissue (see that class, and the
#: milob-package-design skill's invariant 4 -- keeping B out of the forward
#: model is only half the separation if B's parameters then get stored as
#: though they were optical properties).
#:
#: ``beta`` is the Siegert coherence factor. Future additions land here as
#: they become fittable: IRF shape parameters (see ``emg_irf``), detector
#: dark counts, FD amplitude/phase calibration, interferometric reference
#: terms.
OBSERVATION_PARAMS = frozenset({"beta"})


def is_observation_param(name):
    """
    Return True if a fitted parameter belongs to the observation operator.

    Per-layer suffixes are respected, so 'beta_2' is recognised as 'beta' is.

    Parameters
    ----------
    name : str
        Fitted parameter name.

    Returns
    -------
    bool
    """
    return name in OBSERVATION_PARAMS or name.rsplit("_", 1)[0] in OBSERVATION_PARAMS


def split_fit_params(param_names):
    """
    Partition fitted parameter names into medium and observation groups.

    Parameters
    ----------
    param_names : list of str
        Names in the order the fit reported them.

    Returns
    -------
    tuple of (list of str, list of str)
        Names destined for the op axis, and for the observation axis, each in
        the original order.

    Examples
    --------
    >>> split_fit_params(['mua', 'musp', 'aDb', 'beta'])
    (['mua', 'musp', 'aDb'], ['beta'])
    """
    medium, observation = [], []
    for name in param_names:
        (observation if is_observation_param(name) else medium).append(name)
    return medium, observation


def identity(x):
    """
    Return the field quantity unchanged.

    The observation operator for DOS amplitude and phase, and for
    interferometric DCS.

    Parameters
    ----------
    x : array-like
        Field quantity.

    Returns
    -------
    array-like
        The input, unchanged.
    """
    return x


def siegert(g1, beta):
    """
    Apply the Siegert relation, g2(tau) = 1 + beta * |g1(tau)|^2.

    Parameters
    ----------
    g1 : array-like of complex
        Normalised field autocorrelation from a DCS forward model.
    beta : float
        Coherence factor, in [0, 1].

    Returns
    -------
    np.ndarray
        Intensity autocorrelation g2.
    """
    return 1.0 + beta * np.abs(g1) ** 2


def convolve_irf(psi_t, irf_t):
    """
    Convolve a model TPSF with an instrument response function.

    Lets a forward model be compared against raw gated data, rather than
    deconvolving noisy data by the response. Use :func:`identity` for data
    that is already deconvolved.

    Parameters
    ----------
    psi_t : array-like
        Ideal model TPSF, uniformly sampled.
    irf_t : array-like
        Measured or parametric response on the same time spacing, normalised
        to unit area before convolving so its amplitude does not rescale the
        result.

    Returns
    -------
    np.ndarray
        Same length as ``psi_t``. Assumes the response peaks near its own
        first sample, as both a measured response and :func:`emg_irf` do.
    """
    irf_t = np.asarray(irf_t, dtype=float)
    irf_sum = irf_t.sum()
    irf_t = irf_t / irf_sum if irf_sum > 0 else irf_t
    return np.convolve(np.asarray(psi_t, dtype=float), irf_t, mode='full')[:len(psi_t)]


def emg_irf(t, t0, sigma, tau):
    """
    Evaluate an exponentially modified Gaussian instrument response.

    The fallback response for :func:`convolve_irf` when none was measured. Its
    shape parameters can be fitted jointly with the optical properties.

    Parameters
    ----------
    t : array-like
        Time grid in seconds, on the same spacing as the model TPSF.
    t0 : float
        Centre of the Gaussian component.
    sigma : float
        Width of the Gaussian component.
    tau : float
        Decay constant of the exponential tail.

    Returns
    -------
    np.ndarray
        Response evaluated at ``t``.
    """
    # Floor to avoid a division-by-zero blow-up, not to constrain the
    # physical scale -- must sit far below any realistic sigma/tau for a
    # picosecond-to-nanosecond TD-DOS TPSF (this used to be 1e-5 assuming
    # the caller's own pre-unification ``t`` was in nanoseconds; ``t`` here
    # is seconds throughout ``forward.dos``, so that floor silently
    # clamped any physically realistic value up to 10 microseconds --
    # verified directly: it flattened the whole pulse to a near-constant,
    # not the expected sharply peaked EMG shape).
    tau = np.maximum(tau, 1e-15)
    sigma = np.maximum(sigma, 1e-15)

    term1e = (sigma ** 2 / (2 * tau ** 2)) - ((t - t0) / tau)
    term1 = (1 / (2 * tau)) * np.exp(np.clip(term1e, -700, 700))

    term2_arg = (sigma / (np.sqrt(2) * tau)) - ((t - t0) / (np.sqrt(2) * sigma))
    term2 = erfc(np.clip(term2_arg, -20, 20))

    return term1 * term2
