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

