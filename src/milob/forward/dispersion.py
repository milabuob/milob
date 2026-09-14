"""
The dispersion relation shared by CW, frequency-domain and correlation
diffusion.

Setting the correlation delay to zero recovers DOS; setting the modulation
frequency to zero recovers conventional DCS; leaving both active gives
frequency-domain DCS. Geometry lives in :mod:`milob.forward.kernels` and
the mapping to a recorded signal in :mod:`milob.forward.observation`.

Time-domain techniques are not covered here, since their field equation is
solved directly in real time rather than through a Helmholtz form.
"""

import numpy as np

from . import dynamics

C_LIGHT = 2.99792458e10  # cm/s


def k2(mua, musp, n, wavelength, freq=0.0, tau=0.0, motion="brownian", **motion_params):
    """
    Evaluate the dispersion relation K^2.

    K^2 combines a static absorption and scattering term, a source-modulation
    term that vanishes for continuous wave, and a correlation sink that
    vanishes without motion. All terms use the full transport coefficient
    mua + musp rather than the common approximation.

    Parameters
    ----------
    mua : float
        Absorption coefficient in cm^-1.
    musp : float
        Reduced scattering coefficient in cm^-1.
    n : float
        Refractive index of the medium.
    wavelength : float
        Wavelength in nm.
    freq : float or array-like
        Source modulation frequency in Hz. 0 for continuous wave.
    tau : float or array-like
        Correlation delay in seconds. 0 for DOS.
    motion : {'brownian', 'langevin', 'random'}
        Scatterer-motion submodel, used only for a non-zero delay.
    **motion_params
        Forwarded to the motion submodel, e.g. ``aDb`` or ``tc``.

    Returns
    -------
    np.ndarray of complex
        Broadcast over ``mua``, ``musp``, ``freq`` and ``tau``.
    """
    mut = mua + musp
    v = C_LIGHT / n
    wavelength_cm = wavelength / 1e7
    k0 = 2.0 * np.pi * n / wavelength_cm

    static = 3.0 * mua * mut
    modulation = 1j * 3.0 * mut * (2.0 * np.pi * freq) / v

    sink_magnitude = dynamics.msd(tau, model=motion, **motion_params)
    sink = mut * musp * k0**2 * sink_magnitude

    # Always return a numpy (array-like) complex value, even for all-scalar
    # inputs, so callers can uniformly call .astype()/np.sqrt() on the result.
    return np.asarray(static + modulation + sink, dtype=complex)
