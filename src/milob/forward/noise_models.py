import numpy as np
from scipy.optimize import curve_fit


def zhou_noise_model(
    clean_g2s,
    taus,
    rng=None,
    beta=None,
    t_int=None,
    intensity=None,
    correlator_type = 'linear'
):
    """
    Add photon noise to a simulated DCS correlation curve.

    Recovers the field autocorrelation from the input through the Siegert
    relation, estimates its decay rate, applies the analytic per-delay noise
    standard deviation as additive Gaussian noise, and converts back.

    Parameters
    ----------
    clean_g2s : np.ndarray
        Shape (n_taus,), the noise-free g2 curve.
    taus : np.ndarray
        Shape (n_taus,), correlation delays in seconds.
    rng : int or numpy.random.Generator, optional
        Seed or generator for reproducible draws.
    beta : float
        Coherence factor. Required, and used both to recover the field
        autocorrelation and in the noise variance.
    t_int : float
        Integration time in seconds. Required.
    intensity : float
        Photon count rate in counts per second. Required.
    correlator_type : {'linear', 'multi_tau'}
        Correlator architecture the delays came from. 'linear' uses the
        channel index for the lag term; 'multi_tau' uses the delay divided by
        the bin width, since bin width is not constant across octaves.

    Returns
    -------
    np.ndarray
        Shape (n_taus,), the noisy g2 curve.

    References
    ----------
    Zhou, C. et al. (2006). Optics Express, 14(3), 1125-1144.
    """
    rng = np.random.default_rng(rng)

    def find_gamma(taus, g1s):
            def f(tau, gamma):
                return np.exp(-gamma*tau)
        
            gamma = curve_fit(f, taus, g1s)
            # print(gamma)
            return gamma[0]

    g2_reduced = clean_g2s-1
    g1 = np.sqrt(g2_reduced/beta)

    ###Estimate gamma for noise model
    gamma = find_gamma(taus, g1)    

    # Bin widths: T[0] = first spacing, T[i] = taus[i] - taus[i-1]
    T = np.empty_like(taus)
    T[1:] = taus[1:] - taus[:-1]
    T[0] = T[1]
    T = np.clip(T, 1e-30, None)

    L = np.shape(taus)[0]
    # Lag index m: (1, L)

    if correlator_type == 'linear':
        m = np.arange(1, L + 1)
    elif correlator_type == 'multi_tau':
        m = taus / T
    else:
        raise ValueError(f"correlator_type must be 'linear' or 'multi_tau', got {correlator_type!r}")


    # Intermediate exponentials
    exp_2gT = np.exp(-2.0 * gamma * T)     # (B, L)
    exp_2gt = np.exp(-2.0 * gamma * taus)  # (B, L)
    exp_gt = np.exp(-1.0 * gamma * taus)   # (B, L)

    # Mean photon count per bin; clamp to avoid divide-by-zero in padded region
    n_hat = np.clip(intensity * T, 1e-10, None)
    denom = np.clip(1.0 - exp_2gT, 1e-10, None)

    numerator = (
        (1.0 + exp_2gT) * (1.0 + exp_2gt)
        + 2.0 * m * (1.0 - exp_2gT) * exp_2gt
    )

    variance = (T / t_int) * (
        beta**2 * (numerator / denom)
        + 2.0 / n_hat * beta * (1.0 + exp_2gt)
        + 1.0 / n_hat**2 * (1.0 + beta * exp_gt)
    )

    noise_std = np.sqrt(variance)
    
    z = rng.standard_normal((L))

    noisy_g2_reduced = g2_reduced + z * noise_std
    noisy_g2 = 1.0 + noisy_g2_reduced

    return noisy_g2


FD_NOISE_TYPES = ("proportional", "shot")


def fd_noise_model(data, noise_level, rng=None, noise_type="proportional", distances=None):
    """
    Add complex Gaussian noise to a simulated FD-DOS array.

    The real and imaginary parts receive independent noise with standard
    deviation ``sigma * |data|``. For small ``sigma`` this is a relative
    amplitude noise (std of ln A) and a phase noise in radians, both equal
    to ``sigma``, which is the per-channel noise that ``channel_sigma``
    expects in the FD fits.

    Parameters
    ----------
    data : ndarray, complex
        Noise-free fluence, shape (time, channel, wavelength, freq).
    noise_level : float or array-like
        For ``"proportional"``, the relative noise ``sigma``: one value, or
        one per channel. For ``"shot"``, a single value giving the relative
        noise at the shortest channel.
    rng : int or numpy.random.Generator, optional
        Seed or generator for reproducible noise.
    noise_type : {"proportional", "shot"}
        ``"proportional"`` applies ``noise_level`` directly. ``"shot"`` scales
        each channel as ``sigma_i = noise_level * sqrt(|data_ref| / |data_i|)``,
        with ``ref`` the shortest channel, evaluated separately for every
        time point, wavelength and frequency.
    distances : array-like, optional
        Source-detector separation per channel. Required for ``"shot"``.

    Returns
    -------
    ndarray, complex
        Noisy copy of ``data``.

    Raises
    ------
    ValueError
        If ``noise_type`` is unknown, ``noise_level`` has the wrong shape, or
        ``distances`` is missing for ``"shot"``.
    """
    if noise_type not in FD_NOISE_TYPES:
        raise ValueError(f"noise_type must be one of {FD_NOISE_TYPES}, got {noise_type!r}.")
    data = np.asarray(data, dtype=complex)
    n_ch = data.shape[1]
    amp = np.abs(data)
    level = np.asarray(noise_level, dtype=float)

    if noise_type == "proportional":
        if level.ndim == 0:
            sigma = np.full((1, n_ch, 1, 1), float(level))
        elif level.shape == (n_ch,):
            sigma = level.reshape(1, n_ch, 1, 1)
        else:
            raise ValueError(f"noise_level has shape {level.shape}; expected a scalar "
                             f"or one value per channel ({n_ch},).")
    else:
        if level.ndim != 0:
            raise ValueError("noise_type='shot' takes a scalar noise_level: the relative "
                             "noise at the shortest channel.")
        if distances is None:
            raise ValueError("noise_type='shot' needs the channel distances to find "
                             "the shortest (reference) channel.")
        ref = int(np.argmin(np.asarray(distances, dtype=float)))
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = float(level) * np.sqrt(amp[:, ref:ref + 1] / amp)

    rng = np.random.default_rng(rng)
    return (data
            + rng.standard_normal(data.shape) * amp * sigma
            + 1j * rng.standard_normal(data.shape) * amp * sigma)


def fd_noise_sigma(data, noise_level, noise_type="proportional", distances=None):
    """
    Per-channel relative noise that :func:`fd_noise_model` applies.

    Pass the result as ``channel_sigma`` to an FD fit so the fit uses the
    noise that generated the data.

    Parameters
    ----------
    data : array-like, complex
        Noise-free fluence at one time point, wavelength and frequency,
        shape (channel,).
    noise_level, noise_type, distances
        As in :func:`fd_noise_model`.

    Returns
    -------
    ndarray
        Relative noise per channel, shape (channel,).
    """
    data = np.asarray(data, dtype=complex).reshape(1, -1, 1, 1)
    n_ch = data.shape[1]
    level = np.asarray(noise_level, dtype=float)
    if noise_type == "proportional":
        return np.broadcast_to(level, (n_ch,)).astype(float).copy()
    ref = int(np.argmin(np.asarray(distances, dtype=float)))
    amp = np.abs(data[0, :, 0, 0])
    return float(level) * np.sqrt(amp[ref] / amp)
