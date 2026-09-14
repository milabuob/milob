"""
Mean-square-displacement submodels for the DCS correlation sink.

Every function takes the bundled, identifiable parameter, such as the
product of the moving-scatterer fraction and the diffusion coefficient,
rather than the two factors separately. Only that product is recoverable
by inversion. :func:`compose` is the one place the factors appear apart,
for simulation.
"""

import numpy as np


def msd_brownian(tau, aDb=0.0):
    """
    Mean square displacement for Brownian scatterer motion.

    Parameters
    ----------
    tau : array-like
        Correlation delay in seconds.
    aDb : float
        Blood flow index in cm^2/s, the product of the moving-scatterer
        fraction and the diffusion coefficient. Default 0, so that DOS-only
        callers need not supply it.

    Returns
    -------
    np.ndarray
        Mean square displacement in cm^2.
    """
    return 6.0 * aDb * tau


def msd_langevin(tau, aDb=0.0, tc=0.0):
    """
    Mean square displacement for Langevin scatterer motion.

    Interpolates between ballistic motion at delays short relative to the
    velocity correlation time and Brownian motion at long delays.

    Parameters
    ----------
    tau : array-like
        Correlation delay in seconds.
    aDb : float
        Blood flow index in cm^2/s.
    tc : float
        Velocity correlation time in seconds. Zero reduces this to Brownian
        motion.

    Returns
    -------
    np.ndarray
        Mean square displacement in cm^2.
    """
    if tc == 0:
        return msd_brownian(tau, aDb)
    return 6.0 * aDb * (tau - tc * (1.0 - np.exp(-tau / tc)))


def msd_random(tau, aV2=0.0):
    """
    Mean square displacement for random ballistic flow.

    Parameters
    ----------
    tau : array-like
        Correlation delay in seconds.
    aV2 : float
        Ballistic flow index in cm^2/s^2, the product of the moving-scatterer
        fraction and the squared velocity. Default 0.

    Returns
    -------
    np.ndarray
        Mean square displacement in cm^2.
    """
    return aV2 * tau**2


_MODELS = {
    "brownian": msd_brownian,
    "langevin": msd_langevin,
    "random": msd_random,
}


def msd(tau, model="brownian", **motion_params):
    """
    Evaluate the named mean-square-displacement submodel.

    Parameters
    ----------
    tau : array-like
        Correlation delay in seconds.
    model : {'brownian', 'langevin', 'random'}
        Submodel to use.
    **motion_params
        Forwarded to the submodel, e.g. ``aDb``, ``tc`` or ``aV2``.

    Returns
    -------
    np.ndarray
        Mean square displacement in cm^2.
    """
    try:
        fn = _MODELS[model]
    except KeyError:
        raise ValueError(
            f"Unknown motion model '{model}'; choose one of {sorted(_MODELS)}."
        )
    return fn(tau, **motion_params)


def compose(alpha, rate):
    """
    Combine a moving-scatterer fraction with a motion parameter.

    For simulation only, where the two factors may be specified separately.
    Inversion recovers only their product.

    Parameters
    ----------
    alpha : float
        Fraction of moving scatterers.
    rate : float
        Motion parameter, such as a diffusion coefficient or squared velocity.

    Returns
    -------
    float
        The bundled parameter.
    """
    return alpha * rate


# ----------------------------------------------------------------------
# Fitting name <-> storage label
# ----------------------------------------------------------------------
#
# The fitting layer must call the bundled motion parameter whatever the
# forward model's kwarg is named -- ``aDb`` for Brownian/Langevin,
# ``aV2`` for random ballistic flow. The *storage* layer
# (``core.opt_prop_stream.OptPropStream``'s ``op`` axis) deliberately does
# not inherit that commitment: it labels the recovered quantity ``bfi``,
# so a stored stream doesn't assert a scatterer-motion model in the name
# of its own coordinate. The pair of functions below is the (invertible)
# bridge, and ``motion_model`` is what determines the inverse -- which is
# why every producer stores it in ``data.attrs`` alongside the values.

#: Fitting-facing name of the bundled, identifiable motion parameter for
#: each MSD submodel. Note these carry *different units* -- see flow_unit().
BFI_PARAM = {
    "brownian": "aDb",
    "langevin": "aDb",
    "random": "aV2",
}

#: The motion-model-agnostic label the same quantity is stored under.
BFI_LABEL = "bfi"


def bfi_param_name(motion_model="brownian"):
    """
    Return the fitting parameter name for a motion model.

    Parameters
    ----------
    motion_model : {'brownian', 'langevin', 'random'}
        Motion model.

    Returns
    -------
    str
        'aDb' or 'aV2'.
    """
    try:
        return BFI_PARAM[motion_model]
    except KeyError:
        raise ValueError(
            f"Unknown motion model '{motion_model}'; choose one of "
            f"{sorted(BFI_PARAM)}."
        )


def to_storage_label(param_name, motion_model="brownian"):
    """
    Map a fitting parameter name to its storage label.

    Flow parameters become 'bfi', preserving any layer suffix. Every other
    name passes through unchanged.

    Parameters
    ----------
    param_name : str
        Fitting-facing name.
    motion_model : str, optional
        Motion model behind the parameter.

    Returns
    -------
    str
        Storage label.

    Examples
    --------
    >>> to_storage_label('aDb_2')
    'bfi_2'
    >>> to_storage_label('aV2', motion_model='random')
    'bfi'
    """
    fit_name = bfi_param_name(motion_model)
    if param_name == fit_name:
        return BFI_LABEL
    if param_name.startswith(fit_name + "_"):
        return f"{BFI_LABEL}_{param_name[len(fit_name) + 1:]}"
    return param_name


def from_storage_label(label, motion_model="brownian"):
    """
    Recover the fitting parameter name from a storage label.

    The inverse of :func:`to_storage_label`, needed to refit or re-simulate
    from a stored stream, since forward models take 'aDb' or 'aV2' rather than
    'bfi'.

    Parameters
    ----------
    label : str
        Storage label.
    motion_model : str, optional
        Motion model behind the parameter.

    Returns
    -------
    str
        Fitting-facing name.

    Examples
    --------
    >>> from_storage_label('bfi_2')
    'aDb_2'
    """
    fit_name = bfi_param_name(motion_model)
    if label == BFI_LABEL:
        return fit_name
    if label.startswith(BFI_LABEL + "_"):
        return f"{fit_name}_{label[len(BFI_LABEL) + 1:]}"
    return label


def flow_unit(motion_model="brownian", length_unit="cm"):
    """
    Return the units of a stored flow value.

    Brownian and Langevin motion give a diffusion coefficient; random
    ballistic flow gives a squared velocity. The two are different physical
    quantities under the same label, which is why the motion model must travel
    with the data.

    Parameters
    ----------
    motion_model : {'brownian', 'langevin', 'random'}
        Motion model.
    length_unit : str
        Length unit the fit worked in. Default 'cm'.

    Returns
    -------
    str
        Units of the stored value.

    Examples
    --------
    >>> flow_unit('brownian')
    'cm^2/s'
    >>> flow_unit('random')
    'cm^2/s^2'
    """
    bfi_param_name(motion_model)  # validate
    per_time = "s^2" if motion_model == "random" else "s"
    return f"{length_unit}^2/{per_time}"
