"""
The spectral operator mapping tissue composition to optical properties.

Absorption is the sum over chromophores of extinction times concentration,
and reduced scattering follows a power law in wavelength. This is what lets
a fit recover HbO, HbR and the scattering parameters directly from
multi-wavelength data, rather than fitting absorption and scattering per
wavelength and unmixing afterwards.

Only haemoglobin is modelled here, with extinction coefficients taken from
the bundled Prahl table.
"""

import numpy as np

from ..processing.mbll import get_extinction_coefficients_Prahl

WAVELENGTH_PIVOT_NM = 800.0


#: Fitted parameter names that belong to composition space `c` -- tissue-level
#: *causes*, whose spectral operator S produces the optical properties.
#:
#: `A`/`b` are here deliberately. They are not chromophore concentrations, but
#: they are the *generator* of musp(lambda) rather than its value at any one
#: wavelength, which puts them on the cause side of S. `musp` itself is the
#: consequence and belongs in theta -- that line is what stops a partially
#: spectral fit (`HbO`/`HbR` + a free `musp`) from filing an optical property
#: as though it were composition. See `TissueStream`.
COMPOSITION_PARAMS = frozenset({"HbO", "HbR", "HbT", "H2O", "Lipid", "A", "b"})


def is_composition_param(name):
    """
    Return True if a fitted parameter belongs to composition space.

    Parameters
    ----------
    name : str
        Fitted parameter name.

    Returns
    -------
    bool
    """
    return name in COMPOSITION_PARAMS or name.rsplit("_", 1)[0] in COMPOSITION_PARAMS


def split_params_by_space(param_names):
    """
    Partition fitted parameter names into composition, medium and observation.

    A fit spanning more than one space recovers quantities that no single
    stream can hold, which is why the split exists.

    Parameters
    ----------
    param_names : list of str
        Names in the order the fit reported them.

    Returns
    -------
    tuple of (list of str, list of str, list of str)
        Composition, medium and observation names, each in the original order.

    Examples
    --------
    >>> split_params_by_space(['HbO', 'HbR', 'A', 'b'])
    (['HbO', 'HbR', 'A', 'b'], [], [])
    >>> split_params_by_space(['HbO', 'HbR', 'A', 'b', 'aDb', 'beta'])
    (['HbO', 'HbR', 'A', 'b'], ['aDb'], ['beta'])
    """
    from .observation import is_observation_param

    composition, medium, observation = [], [], []
    for name in param_names:
        if is_observation_param(name):
            observation.append(name)
        elif is_composition_param(name):
            composition.append(name)
        else:
            medium.append(name)
    return composition, medium, observation


def extinction_at(wavelength):
    """
    Return the haemoglobin extinction coefficients at one wavelength.

    Reads and interpolates the full table on every call, so call it once per
    wavelength outside any optimisation loop.

    Parameters
    ----------
    wavelength : float
        Wavelength in nm.

    Returns
    -------
    tuple of (float, float)
        Oxy- and deoxyhaemoglobin extinction in cm^-1 per mole per litre,
        base e.
    """
    E = get_extinction_coefficients_Prahl([wavelength])["E_matrix"]
    return float(E[0, 0]), float(E[0, 1])


def mua_from_composition(hbo, hbr, eps_hbo, eps_hbr):
    """
    Compute absorption from haemoglobin concentrations.

    Parameters
    ----------
    hbo, hbr : float
        Oxy- and deoxyhaemoglobin concentrations in micromolar.
    eps_hbo, eps_hbr : float
        Extinction coefficients at this wavelength, in cm^-1 per mole per
        litre, as returned by :func:`extinction_at`.

    Returns
    -------
    float
        Absorption coefficient in cm^-1.
    """
    return (eps_hbo * hbo + eps_hbr * hbr) * 1e-6


def musp_powerlaw(A, b, wavelength, wavelength_0=WAVELENGTH_PIVOT_NM):
    """
    Evaluate the scattering power law.

    Returns ``A * (wavelength / wavelength_0) ** -b``, pivoted inside the
    measured band so that ``A`` is the reduced scattering at the pivot. This
    decorrelates the two parameters, unlike the form written against raw
    wavelength, where ``A`` is an extrapolation to 1 nm.

    The exponent is weakly constrained when only two wavelengths are measured,
    since its leverage is set by their separation in log wavelength. Widely
    separated wavelengths help; check its propagated uncertainty before
    relying on a point estimate.

    Parameters
    ----------
    A : float
        Reduced scattering in cm^-1 at the pivot wavelength.
    b : float
        Scattering exponent, dimensionless.
    wavelength : float or array-like
        Wavelength in nm.
    wavelength_0 : float
        Pivot wavelength in nm. Default 800.

    Returns
    -------
    float or np.ndarray
        Reduced scattering in cm^-1.

    References
    ----------
    Bevilacqua, F. et al. (1999). Applied Optics, 38(22), 4939-4950.
    """
    return A * (wavelength / wavelength_0) ** (-b)


def assemble_spectral_fd(flat):
    """
    Map composition parameters onto the semi-infinite FD-DOS model's arguments.

    Used as the ``assemble`` callback when fitting HbO, HbR and the scattering
    parameters directly against multi-wavelength data.

    Parameters
    ----------
    flat : dict
        Merged fixed, fitted and context parameters for one wavelength block.
        Must carry that block's ``wavelength``, ``n``, ``freq`` and the
        extinction coefficients, precomputed rather than looked up here.

    Returns
    -------
    dict
        Keyword arguments for the forward model.
    """
    mua = mua_from_composition(flat["HbO"], flat["HbR"], flat["eps_hbo"], flat["eps_hbr"])
    musp = musp_powerlaw(flat["A"], flat["b"], flat["wavelength"])
    return {
        "mua": mua,
        "musp": musp,
        "n": flat["n"],
        "wavelength": flat["wavelength"],
        "freq": flat.get("freq", 0.0),
        "R_eff": flat.get("R_eff"),
    }


def assemble_spectral_dcs(flat):
    """
    Map composition parameters onto the semi-infinite DCS model's arguments.

    The DCS counterpart of :func:`assemble_spectral_fd`, letting a DCS block
    share one composition with FD-DOS blocks in a joint fit. Absorption and
    scattering at the laser line are evaluated from that shared composition,
    so the laser wavelength need not be one of the measured FD wavelengths.

    Parameters
    ----------
    flat : dict
        Merged fixed, fitted and context parameters for the block. Must carry
        the block's ``wavelength``, ``n`` and extinction coefficients.

    Returns
    -------
    dict
        Keyword arguments for the forward model.
    """
    from .dynamics import bfi_param_name

    motion = flat.get("motion", "brownian")
    motion_param = bfi_param_name(motion)

    out = {
        "rho": flat["rho"],
        "mua": mua_from_composition(flat["HbO"], flat["HbR"],
                                    flat["eps_hbo"], flat["eps_hbr"]),
        "musp": musp_powerlaw(flat["A"], flat["b"], flat["wavelength"]),
        "n": flat["n"],
        "wavelength": flat["wavelength"],
        "z": flat.get("z", 0.0),
        "motion": motion,
        "R_eff": flat.get("R_eff"),
        motion_param: flat[motion_param],
    }
    if motion == "langevin" and "tc" in flat:
        out["tc"] = flat["tc"]
    return out


TISSUE_FD_PARAM_CONFIG = {
    "HbO": {"bounds": (0.1, 150.0), "log": True},
    "HbR": {"bounds": (0.1, 150.0), "log": True},
    "A": {"bounds": (1.0, 30.0), "log": True},
    "b": {"bounds": (0.1, 4.0), "log": True},
}
"""Default param_config for assemble_spectral_fd (HbO/HbR in uM, A = musp
(cm^-1) at WAVELENGTH_PIVOT_NM, b dimensionless). Broad tissue-plausible
bounds, same spirit as `forward.dos`'s per-geometry defaults -- narrow them
if you have independent prior knowledge, since `b` in particular is weakly
constrained with only 2 wavelengths (see `musp_powerlaw`'s Notes)."""


TISSUE_DCS_PARAM_CONFIG = {
    **TISSUE_FD_PARAM_CONFIG,
    "aDb": {"bounds": (1e-10, 1e-6), "log": True},
    "beta": {"bounds": (0.0, 1.0), "log": False},
}
"""Default param_config for a joint FD-DOS + CW-DCS spectral fit: the four
composition parameters shared by every block, plus the DCS block's own
motion (`aDb`) and observation (`beta`) parameters.

Note what is *not* here: `mua` and `musp`. That is the point -- they are no
longer free parameters at all, but consequences of the composition via `S`,
which is what ties the wavelengths (and the two modalities) together."""
