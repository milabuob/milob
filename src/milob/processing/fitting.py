import inspect
from typing import NamedTuple

import numpy as np
import scipy.optimize
from scipy.stats import qmc
import xarray as xr

from ..forward import observation as _observation


# =======================================================================
# Parallel dispatch for independent per-slice fits
# =======================================================================

def run_parallel_fits(jobs, n_jobs=1, backend=None):
    """
    Run independent fits serially or in parallel.

    Parameters
    ----------
    jobs : list of tuple
        Each entry is a (callable, args) pair, where ``callable(*args)`` runs
        one fit and returns its result. Keep whatever the callable closes over
        lightweight, since it is pickled for process-based backends.
    n_jobs : int
        1 (default) runs serially without importing joblib. -1 uses every
        core, and any other integer that many workers.
    backend : str, optional
        Forwarded to ``joblib.Parallel``. None leaves joblib's process-based
        default, which suits optimiser-bound work. Pass 'threading' for jobs
        dominated by GIL-releasing C code, where threads avoid copying large
        read-only inputs into each worker.

    Returns
    -------
    list
        One result per job, in the order given.
    """
    if n_jobs == 1:
        return [func(*args) for func, args in jobs]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs, backend=backend)(delayed(func)(*args) for func, args in jobs)


def default_n_starts(n_params, floor=8, per_param=4):
    """
    Return the default number of multi-start attempts for a fit.

    Scales linearly with the number of free parameters, since a space-filling
    design needs more starts as the search space grows.

    Parameters
    ----------
    n_params : int
        Number of free parameters.
    floor : int
        Minimum number of starts regardless of dimension. Default 8.
    per_param : int
        Starts added per free parameter. Default 4.

    Returns
    -------
    int
        Number of starts to run.
    """
    return max(floor, per_param * n_params)


def _resolve_sigma(sigma, n):
    """Normalise sigma to None or an array matching the transformed residual."""
    if sigma is None:
        return None
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim == 0:
        return np.full(n, float(sigma))
    if sigma.shape != (n,):
        raise ValueError(
            f"sigma has shape {sigma.shape}, but the (transformed) residual "
            f"has length {n} -- pass a scalar, or an array matching the "
            f"transformed residual, not the raw measurement."
        )
    return sigma


def _multistart_optimize(fit_func, xdata, ydata, starts, bounds, n_jobs=1, context="",
                          sigma=None, absolute_sigma=False):
    """Run the optimiser from every start and return the lowest-cost result."""
    def _try(p0):
        try:
            popt, pcov = scipy.optimize.curve_fit(
                fit_func, xdata, ydata, p0=p0, bounds=bounds, method="trf",
                sigma=sigma, absolute_sigma=absolute_sigma,
            )
        except RuntimeError:
            return None
        resid = fit_func(xdata, *popt) - ydata
        if sigma is not None:
            resid = resid / sigma
        cost = np.sum(resid ** 2)
        return (cost, popt, pcov)

    jobs = [(_try, (p0,)) for p0 in starts]
    raw = run_parallel_fits(jobs, n_jobs=n_jobs)
    results = [r for r in raw if r is not None]

    if not results:
        return None

    results.sort(key=lambda r: r[0])
    best_cost, best_popt, best_pcov = results[0]

    # Signal 1: a distinct, comparably-good alternative among this call's
    # own starts.
    distance_threshold = 0.1  # Euclidean, in the normalised [0, 1]^n_params start space
    cost_tolerance = max(0.1 * best_cost, 0.05)
    alternate = None
    for cost, popt, _ in results[1:]:
        if np.linalg.norm(popt - best_popt) > distance_threshold and cost <= best_cost + cost_tolerance:
            alternate = (cost, popt)
            break

    # Signal 2: the best point itself sits at/against a bound -- same
    # [0.02, 0.98] margin `starts` itself is kept off of, since a converged
    # value inside that margin is indistinguishable from "still at the edge".
    bound_margin = 0.02
    at_bound = np.any(best_popt < bound_margin) or np.any(best_popt > 1 - bound_margin)

    # Free diagnostic: fraction of this call's own successful starts that
    # ended up near the best point (same distance_threshold as signal 1).
    n_near_best = sum(1 for _, popt, _ in results if np.linalg.norm(popt - best_popt) <= distance_threshold)
    convergence_fraction = n_near_best / len(results)

    if alternate is not None or at_bound:
        label = f" ({context})" if context else ""
        reasons = []
        if alternate is not None:
            reasons.append(
                f"a distinct start converged to a comparably good cost "
                f"({alternate[0]:.3g} vs. best {best_cost:.3g})"
            )
        if at_bound:
            reasons.append("the best point sits at/against a parameter bound")
        print(
            f"Warning{label}: {' and '.join(reasons)} -- this fit may not be "
            f"well-identified at the current n_starts/param_config "
            f"({n_near_best}/{len(results)} successful starts converged near "
            f"the returned point). The reported uncertainty reflects only the "
            f"local curvature around that point, not this risk -- see "
            f"_multistart_optimize's docstring before trusting it as-is."
        )

    return best_cost, best_popt, best_pcov


# =======================================================================
# Generalised TD-DOS Fitting Procedure
# =======================================================================
#
# The TD-DOS analogue of `fd_model_opt`/`dcs_g1_model_opt`: any
# `forward.dos.*_td_fluence`/`*_td_fluence_patterson` forward model, fit
# against either a raw (`observation="convolve_irf"`) or already-deconvolved
# (`observation="identity"`) measured TPSF, via the same param_config/
# assemble/multistart engine used by FD-DOS/DCS. `td_moments_model_opt`
# below is the moments-based sibling: it fits the same forward models'
# numeric moments instead of the full curve, generalising the closed-form
# semi-infinite-only Liebert et al. (2003) relation
# (`processing.time_domain.calculate_optical_properties_moments`, kept
# as-is for its cheap exact special case) to any geometry.

def _resolve_td_irf(irf, t, flat):
    """Resolve the instrument response to an array on the fitting time grid."""
    if callable(irf):
        return np.asarray(irf(t, flat), dtype=float)
    return np.asarray(irf, dtype=float)


def td_model_opt(
        t,
        rho,
        data,
        forward_model,
        forward_parameters,
        param_config,
        fixed_params=None,
        assemble=None,
        observation="identity",
        irf=None,
        n_starts=None,
        random_state=0,
        n_jobs=1,
        sigma=None,
        absolute_sigma=False,
    return_covariance=False,
        log_fit=True,
):
    """
    Fit a TD-DOS forward model to a measured TPSF by nonlinear least squares.

    Parameters
    ----------
    t : array-like
        Time-of-flight grid in seconds that ``data`` is sampled on.
    rho : float
        Source-detector distance in cm.
    data : array-like
        Measured TPSF, the same length as ``t``. Raw gated counts when
        ``observation='convolve_irf'``, or a deconvolved curve otherwise.
    forward_model : callable
        Model called once as ``forward_model(t=t, rho=rho, **model_kwargs)``.
    forward_parameters : dict
        Fixed context passed to ``forward_model``, such as refractive index or
        wavelength.
    param_config : dict
        Free parameters and their bounds, e.g.
        ``{"mua": {"bounds": (1e-3, 0.5), "log": True}}``.
    fixed_params : dict, optional
        Parameters held at a known value rather than fitted.
    assemble : callable, optional
        Maps the merged flat parameter dict onto the model's arguments. None
        passes the flat dict through unchanged.
    observation : {'identity', 'convolve_irf'}
        Whether the model output is compared directly with the data or first
        convolved with the instrument response.
    irf : array-like or callable, optional
        Instrument response, required when convolving.
    n_starts : int, optional
        Number of optimiser runs from different starting points, keeping the
        best. The first start is the bounds' midpoint and the rest form a
        Latin-hypercube design. Scales with the number of free parameters when
        None; 1 recovers plain single-start behaviour.
    random_state : int or numpy.random.Generator
        Seed for the Latin-hypercube starting points.
    n_jobs : int
        Parallel jobs across the starts. Default 1, meaning serial.
    sigma : float or array-like, optional
        Known per-point noise on the transformed residual. If None, the
        reported covariance is self-calibrated from the fit's own residual,
        which makes a noiseless synthetic fit report a near-zero uncertainty.
    absolute_sigma : bool
        Treat ``sigma`` as a known absolute noise level, anchoring the
        covariance to it. Default False, matching ``curve_fit``.
    return_covariance : bool
        Also return the full parameter covariance matrix. Default False.
    log_fit : bool
        Fit in log10 counts, appropriate for photon-counting data spanning
        several decades. Default True.

    Returns
    -------
    results, errors : dict, dict
        Fitted values and one-sigma uncertainties, keyed by every name in
        ``param_config`` and ``fixed_params``, the latter with zero error.
    covariance : np.ndarray, optional
        Full covariance over the free parameters. Returned only when
        ``return_covariance`` is True.
    """
    if fixed_params is None:
        fixed_params = {}
    if observation == "convolve_irf" and irf is None:
        raise ValueError(
            "observation='convolve_irf' requires irf= (a measured IRF array "
            "or a callable (t, flat) -> irf_t)."
        )

    t = np.asarray(t, dtype=float)
    data = np.asarray(data, dtype=float)

    param_names = [p for p in param_config if p not in fixed_params]
    n_params = len(param_names)

    if assemble is None:
        accepted = {
            name for name, p in inspect.signature(forward_model).parameters.items()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        } - {"A"}

    def evaluate(flat):
        model_kwargs = assemble(flat) if assemble is not None else {
            k: v for k, v in flat.items() if k in accepted
        }
        psi_t = forward_model(t=t, rho=rho, **model_kwargs)
        psi_t = flat.get("A", 1.0) * psi_t

        if callable(observation):
            return observation(psi_t, flat)
        if observation == "identity":
            return _observation.identity(psi_t)
        if observation == "convolve_irf":
            irf_t = _resolve_td_irf(irf, t, flat)
            return _observation.convolve_irf(psi_t, irf_t)
        raise ValueError(f"Unknown observation '{observation}'; choose 'identity', 'convolve_irf', or a callable.")

    def _transform(signal):
        if log_fit:
            return np.log10(np.clip(signal, 0.0, None) + 1e-12)
        return signal

    def fit_func(_, *x):
        fit_params = {
            name: _unit_interval_from(x[i], param_config[name])
            for i, name in enumerate(param_names)
        }
        flat = {**forward_parameters, **fit_params, **fixed_params}
        return _transform(evaluate(flat))

    ydata = _transform(data)
    xdata = np.arange(len(ydata))

    resolved_n_starts = default_n_starts(n_params) if n_starts is None else n_starts

    bounds = ([0] * n_params, [1] * n_params)
    starts = [np.full(n_params, 0.5)]
    n_extra = max(resolved_n_starts, 1) - 1
    if n_extra > 0:
        sampler = qmc.LatinHypercube(d=n_params, seed=random_state)
        starts += list(0.02 + 0.96 * sampler.random(n=n_extra))

    best = _multistart_optimize(fit_func, xdata, ydata, starts, bounds,
                                 n_jobs=n_jobs, context="td_model_opt",
                                 sigma=_resolve_sigma(sigma, len(ydata)),
                                 absolute_sigma=absolute_sigma)
    if best is None:
        print("Fit failed: every start raised RuntimeError")
        failed = (
            {name: np.nan for name in param_config},
            {name: np.nan for name in param_config},
        )
        return (*failed, _failed_covariance(param_names)) if return_covariance else failed
    _, popt, pcov = best

    param_std = np.sqrt(np.diag(pcov))

    results, errors = {}, {}
    for i, name in enumerate(param_names):
        cfg = param_config[name]
        results[name] = _unit_interval_from(popt[i], cfg)
        errors[name] = _unit_interval_uncertainty(popt[i], param_std[i], cfg)

    results.update(fixed_params)
    for name in fixed_params:
        errors[name] = 0.0

    if return_covariance:
        # Full covariance over the *fitted* parameters only -- `fixed_params`
        # were never estimated, so they have no variance to report (their
        # `errors` entry is 0.0 for the same reason).
        return results, errors, _covariance_in_param_space(
            popt, pcov, param_names, param_config)
    return results, errors


# ------------------------------------------------------------------
# General (any-geometry) moments-based fit
# ------------------------------------------------------------------

def _model_moments(t, psi_t):
    """Integrate a model TPSF to give its moments m0, m1 and m2."""
    psi_t = np.clip(np.asarray(psi_t, dtype=float), 0.0, None)
    m0 = np.trapezoid(psi_t, t)
    if m0 <= 0:
        return np.nan, np.nan, np.nan
    m1 = np.trapezoid(t * psi_t, t) / m0
    m2 = np.trapezoid((t - m1) ** 2 * psi_t, t) / m0
    return m0, m1, m2


def td_moments_model_opt(
        t,
        rho,
        moments,
        forward_model,
        forward_parameters,
        param_config,
        fixed_params=None,
        assemble=None,
        n_starts=None,
        random_state=0,
        n_jobs=1,
        sigma=None,
        absolute_sigma=False,
    return_covariance=False,
        moments_used=("m1", "m2"),
):
    """
    Fit a TD-DOS forward model to measured TPSF moments.

    Integrates the model TPSF numerically, so any geometry can be fitted on
    moments rather than on the full curve.

    Parameters
    ----------
    t : array-like
        Dense time grid in seconds on which the model moments are integrated.
        It should cover the TPSF's support, not just the measured gates.
    rho : float
        Source-detector distance in cm.
    moments : dict
        Measured, IRF-corrected moments, e.g. ``{"m0": ..., "m1": ...}``.
    forward_model : callable
        Model whose moments are fitted.
    forward_parameters : dict
        Fixed context passed to ``forward_model``.
    param_config : dict
        Free parameters and their bounds.
    fixed_params : dict, optional
        Parameters held at a known value rather than fitted.
    assemble : callable, optional
        Maps the merged flat parameter dict onto the model's arguments.
    n_starts : int, optional
        Number of optimiser runs from different starting points, keeping the
        best. The first start is the bounds' midpoint and the rest form a
        Latin-hypercube design. Scales with the number of free parameters when
        None; 1 recovers plain single-start behaviour.
    random_state : int or numpy.random.Generator
        Seed for the Latin-hypercube starting points.
    n_jobs : int
        Parallel jobs across the starts. Default 1, meaning serial.
    sigma : float or array-like, optional
        Known per-point noise on the transformed residual. If None, the
        reported covariance is self-calibrated from the fit's own residual,
        which makes a noiseless synthetic fit report a near-zero uncertainty.
    absolute_sigma : bool
        Treat ``sigma`` as a known absolute noise level, anchoring the
        covariance to it. Default False, matching ``curve_fit``.
    return_covariance : bool
        Also return the full parameter covariance matrix. Default False.
    moments_used : tuple of str
        Which moments enter the residual. Default ('m1', 'm2'), since m0
        depends on injected power and coupling efficiency.

    Returns
    -------
    results, errors : dict, dict
        Fitted values and one-sigma uncertainties, keyed by every name in
        ``param_config`` and ``fixed_params``, the latter with zero error.
    covariance : np.ndarray, optional
        Full covariance over the free parameters. Returned only when
        ``return_covariance`` is True.
    """
    if fixed_params is None:
        fixed_params = {}

    t = np.asarray(t, dtype=float)
    param_names = [p for p in param_config if p not in fixed_params]
    n_params = len(param_names)

    if assemble is None:
        accepted = {
            name for name, p in inspect.signature(forward_model).parameters.items()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        } - {"A"}

    def evaluate(flat):
        model_kwargs = assemble(flat) if assemble is not None else {
            k: v for k, v in flat.items() if k in accepted
        }
        psi_t = forward_model(t=t, rho=rho, **model_kwargs)
        m0, m1, m2 = _model_moments(t, psi_t)
        model_moments = {"m0": m0, "m1": m1, "m2": m2}
        return np.array([model_moments[m] for m in moments_used])

    # m0 (~1), m1 (~1e-9 s) and m2 (~1e-19 s^2) differ by many orders of
    # magnitude -- comparing them as raw absolute residuals would let
    # whichever moment happens to have the largest absolute value swamp
    # curve_fit's cost and leave the others effectively unconstrained.
    # Fitting the *fractional* deviation from each measured moment instead
    # (target 0 for every channel) keeps every moment's contribution O(1)
    # regardless of its physical units, the same reasoning behind
    # `fd_model_opt`'s log-amplitude-ratio residual transform.
    measured = np.array([moments[m] for m in moments_used], dtype=float)

    # A zero or non-finite measured moment (bad/missing channel, masked-out
    # time bin) makes the fractional residual below (division by `measured`)
    # inf/NaN for every evaluation -- curve_fit can't recover from that, and
    # the failure mode without this guard is either a confusing optimizer
    # exception or a silently garbage fit rather than a clean "this slice
    # can't be fit" result. Same NaN-filled return shape as the "every start
    # raised RuntimeError" case below, so callers (fit_to_op_moments's
    # per-slice loop) don't need to special-case this failure differently.
    if not np.all(np.isfinite(measured)) or np.any(measured == 0):
        unfittable = (
            {name: np.nan for name in param_config},
            {name: np.nan for name in param_config},
        )
        return ((*unfittable, _failed_covariance(param_names))
                if return_covariance else unfittable)

    def fit_func(_, *x):
        fit_params = {
            name: _unit_interval_from(x[i], param_config[name])
            for i, name in enumerate(param_names)
        }
        flat = {**forward_parameters, **fit_params, **fixed_params}
        model_vals = evaluate(flat)
        return (model_vals - measured) / measured

    ydata = np.zeros(len(measured))
    xdata = np.arange(len(ydata))

    resolved_n_starts = default_n_starts(n_params) if n_starts is None else n_starts
    bounds = ([0] * n_params, [1] * n_params)
    starts = [np.full(n_params, 0.5)]
    n_extra = max(resolved_n_starts, 1) - 1
    if n_extra > 0:
        sampler = qmc.LatinHypercube(d=n_params, seed=random_state)
        starts += list(0.02 + 0.96 * sampler.random(n=n_extra))

    best = _multistart_optimize(fit_func, xdata, ydata, starts, bounds,
                                 n_jobs=n_jobs, context="td_moments_model_opt",
                                 sigma=_resolve_sigma(sigma, len(ydata)),
                                 absolute_sigma=absolute_sigma)
    if best is None:
        print("Fit failed: every start raised RuntimeError")
        failed = (
            {name: np.nan for name in param_config},
            {name: np.nan for name in param_config},
        )
        return (*failed, _failed_covariance(param_names)) if return_covariance else failed
    _, popt, pcov = best
    param_std = np.sqrt(np.diag(pcov))

    results, errors = {}, {}
    for i, name in enumerate(param_names):
        cfg = param_config[name]
        results[name] = _unit_interval_from(popt[i], cfg)
        errors[name] = _unit_interval_uncertainty(popt[i], param_std[i], cfg)

    results.update(fixed_params)
    for name in fixed_params:
        errors[name] = 0.0

    if return_covariance:
        # Full covariance over the *fitted* parameters only -- `fixed_params`
        # were never estimated, so they have no variance to report (their
        # `errors` entry is 0.0 for the same reason).
        return results, errors, _covariance_in_param_space(
            popt, pcov, param_names, param_config)
    return results, errors


# ------------------------------------------------------------------
# Generalised FD-DOS Fitting Procedure
# ------------------------------------------------------------------
#
# Nonlinear least-squares recovery of optical properties from multi-distance
# FD-DOS amplitude/phase data, minimising a residual constructed to cancel
# the unknown source-detector coupling coefficient (amplitude) and cable
# phase offset (phase) without extra free "calibration" parameters -- the
# default `residual_transform` implements eq. 2 of Martins, Forti & Mesquita
# (2025), Spectrosc. J. 3, 14, self-referencing every channel's log-amplitude
# and phase against one reference channel.
#
# Both what gets compared (`residual_transform`) and how the flat fitted
# parameters map onto the forward model's actual call signature (`assemble`)
# are pluggable, so the same engine covers the semi-infinite model (flat
# param names already match `si_fd_fluence`'s kwargs, `assemble=None`) and,
# later, layered models (where `assemble` packs per-layer scalars into the
# arrays `two_layer_fd_fluence` expects) without changing the optimiser.

def _fd_residual_reference_channel(amp, phase, rho, ref_idx=0):
    """Normalise amplitude and phase against one reference channel."""
    idx = [i for i in range(len(rho)) if i != ref_idx]
    log_amp_ratio = np.log(amp[idx] / amp[ref_idx])
    phase_diff = phase[idx] - phase[ref_idx]
    return np.concatenate([log_amp_ratio, phase_diff])


def _fd_residual_consecutive_channels(amp, phase, rho):
    """Normalise amplitude and phase between consecutive channels."""
    log_amp_ratio = np.log(amp[1:] / amp[:-1])
    phase_diff = phase[1:] - phase[:-1]
    return np.concatenate([log_amp_ratio, phase_diff])


_FD_RESIDUAL_TRANSFORMS = {
    "reference_channel": _fd_residual_reference_channel,
    "consecutive_channels": _fd_residual_consecutive_channels,
}


def _resolve_fd_residual_transform(transform):
    """Resolve the named residual transform to a callable."""
    if callable(transform):
        return transform
    try:
        return _FD_RESIDUAL_TRANSFORMS[transform]
    except KeyError:
        raise ValueError(
            f"Unknown residual_transform '{transform}'; choose one of "
            f"{sorted(_FD_RESIDUAL_TRANSFORMS)} or pass a callable "
            f"(amp, phase, rho, **kwargs) -> ndarray."
        )


def _unit_interval_from(x, cfg):
    """Map a parameter onto the unit interval given its bounds."""
    lo, hi = cfg["bounds"]
    if cfg["log"]:
        lo, hi = np.log10(lo), np.log10(hi)
        return 10 ** (lo + x * (hi - lo))
    return lo + x * (hi - lo)


def _unit_interval_uncertainty(x, sigma_x, cfg):
    lo, hi = cfg["bounds"]
    if cfg["log"]:
        lo_log, hi_log = np.log10(lo), np.log10(hi)
        log_val = lo_log + x * (hi_log - lo_log)
        p = 10 ** log_val
        dpdx = p * np.log(10) * (hi_log - lo_log)
        return sigma_x * dpdx
    return sigma_x * (hi - lo)


def _covariance_in_param_space(popt, pcov, param_names, param_config):
    """Transform a covariance from the optimiser's space back to parameter space."""
    scale = np.array([
        _unit_interval_uncertainty(popt[i], 1.0, param_config[name])
        for i, name in enumerate(param_names)
    ])
    return np.asarray(pcov) * np.outer(scale, scale)


def _failed_covariance(param_names):
    n = len(param_names)
    return np.full((n, n), np.nan)


def fd_model_opt(
    rho,
    data,
    forward_model,
    forward_parameters,
    param_config,
    fixed_params=None,
    assemble=None,
    residual_transform="reference_channel",
    n_starts=None,
    random_state=0,
    n_jobs=1,
    sigma=None,
    absolute_sigma=False,
    return_covariance=False,
    **residual_kwargs,
):
    """
    Fit an FD-DOS forward model to complex fluence by nonlinear least squares.

    Passing lists for ``rho``, ``data`` and ``forward_parameters`` fits several
    blocks jointly against one shared parameter vector, which is what makes a
    spectral parameter such as HbO identifiable across wavelengths.

    Parameters
    ----------
    rho : array-like or list of array-like
        Source-detector distances in cm, one per channel, or one array per
        block.
    data : array-like or list of array-like
        Measured complex fluence, matching ``rho`` in shape and order.
    forward_model : callable
        Model called once per distance as
        ``forward_model(rho=rho_i, **model_kwargs)``, returning complex
        fluence.
    forward_parameters : dict or list of dict
        Fixed context passed to ``forward_model``, such as refractive index,
        wavelength or modulation frequency. A list switches on multi-block
        mode, with each block transformed independently before the residuals
        are concatenated.
    param_config : dict
        Free parameters and their bounds, shared across every block.
    fixed_params : dict, optional
        Parameters held at a known value rather than fitted.
    assemble : callable, optional
        Maps the merged flat parameter dict onto the model's arguments. None
        passes the flat dict through unchanged, which suits a model whose
        argument names already match.
    residual_transform : str or callable
        What is compared between model and data: 'reference_channel'
        (default), 'consecutive_channels', or a callable
        ``(amp, phase, rho, **residual_kwargs) -> ndarray`` applied identically
        to both.
    n_starts : int, optional
        Number of optimiser runs from different starting points, keeping the
        best. The first start is the bounds' midpoint and the rest form a
        Latin-hypercube design. Scales with the number of free parameters when
        None; 1 recovers plain single-start behaviour.
    random_state : int or numpy.random.Generator
        Seed for the Latin-hypercube starting points.
    n_jobs : int
        Parallel jobs across the starts. Default 1, meaning serial.
    sigma : float or array-like, optional
        Known per-point noise on the transformed residual. If None, the
        reported covariance is self-calibrated from the fit's own residual,
        which makes a noiseless synthetic fit report a near-zero uncertainty.
    absolute_sigma : bool
        Treat ``sigma`` as a known absolute noise level, anchoring the
        covariance to it. Default False, matching ``curve_fit``.
    return_covariance : bool
        Also return the full parameter covariance matrix. Default False.
    **residual_kwargs
        Forwarded to the residual transform, e.g. ``ref_idx``.

    Returns
    -------
    results, errors : dict, dict
        Fitted values and one-sigma uncertainties, keyed by every name in
        ``param_config`` and ``fixed_params``, the latter with zero error.
    covariance : np.ndarray, optional
        Full covariance over the free parameters. Returned only when
        ``return_covariance`` is True.

    References
    ----------
    Martins, G. G., Forti, R. M., & Mesquita, R. C. (2025). Spectroscopy
    Journal, 3, 14.
    """
    if fixed_params is None:
        fixed_params = {}

    multi_block = isinstance(forward_parameters, (list, tuple))
    if multi_block:
        rho_is_list = isinstance(rho, (list, tuple))
        data_is_list = isinstance(data, (list, tuple))
        if not (rho_is_list and data_is_list and len(rho) == len(data) == len(forward_parameters)):
            raise ValueError(
                "forward_parameters is a list (multi-block fit), so rho and data "
                "must be lists of the same length, one array per block -- got "
                f"{len(forward_parameters)} forward_parameters, "
                f"{len(rho) if rho_is_list else 'a single'} rho, "
                f"{len(data) if data_is_list else 'a single'} data."
            )
        blocks_raw = list(zip(rho, data, forward_parameters))
    else:
        blocks_raw = [(rho, data, forward_parameters)]

    blocks = [
        (np.asarray(r, dtype=float), np.asarray(d, dtype=complex), fp)
        for r, d, fp in blocks_raw
    ]

    transform = _resolve_fd_residual_transform(residual_transform)

    param_names = [p for p in param_config if p not in fixed_params]
    n_params = len(param_names)

    def evaluate(rho_i, model_kwargs):
        phi = np.array([forward_model(rho=r, **model_kwargs) for r in rho_i])
        return np.abs(phi), np.angle(phi)

    def residual(fit_params):
        pieces = []
        for rho_i, _, fp_i in blocks:
            flat = {**fp_i, **fit_params, **fixed_params}
            model_kwargs = flat if assemble is None else assemble(flat)
            amp_model, phase_model = evaluate(rho_i, model_kwargs)
            pieces.append(transform(amp_model, phase_model, rho_i, **residual_kwargs))
        return np.concatenate(pieces)

    def fit_func(_, *x):
        fit_params = {
            name: _unit_interval_from(x[i], param_config[name])
            for i, name in enumerate(param_names)
        }
        return residual(fit_params)

    ydata = np.concatenate([
        transform(np.abs(d), np.angle(d), rho_i, **residual_kwargs)
        for rho_i, d, _ in blocks
    ])
    # curve_fit requires an xdata argument matching ydata's length, but the
    # actual per-block rho arrays are closed over in `blocks` -- fit_func
    # never reads this positional argument, same as the single-block version
    # only used its `rho` parameter for the transform's channel count, not
    # curve_fit's own bookkeeping.
    xdata = np.arange(len(ydata))

    resolved_n_starts = default_n_starts(n_params) if n_starts is None else n_starts

    bounds = ([0] * n_params, [1] * n_params)
    starts = [np.full(n_params, 0.5)]
    n_extra = max(resolved_n_starts, 1) - 1
    if n_extra > 0:
        sampler = qmc.LatinHypercube(d=n_params, seed=random_state)
        # Scale from [0, 1] to [0.02, 0.98] -- stay off the exact bounds,
        # where curve_fit's trust-region step can behave poorly.
        starts += list(0.02 + 0.96 * sampler.random(n=n_extra))

    best = _multistart_optimize(fit_func, xdata, ydata, starts, bounds,
                                 n_jobs=n_jobs, context="fd_model_opt",
                                 sigma=_resolve_sigma(sigma, len(ydata)),
                                 absolute_sigma=absolute_sigma)
    if best is None:
        print("Fit failed: every start raised RuntimeError")
        failed = (
            {name: np.nan for name in param_config},
            {name: np.nan for name in param_config},
        )
        return (*failed, _failed_covariance(param_names)) if return_covariance else failed
    _, popt, pcov = best

    param_std = np.sqrt(np.diag(pcov))

    results, errors = {}, {}
    for i, name in enumerate(param_names):
        cfg = param_config[name]
        results[name] = _unit_interval_from(popt[i], cfg)
        errors[name] = _unit_interval_uncertainty(popt[i], param_std[i], cfg)

    results.update(fixed_params)
    for name in fixed_params:
        errors[name] = 0.0

    if return_covariance:
        # Full covariance over the *fitted* parameters only -- `fixed_params`
        # were never estimated, so they have no variance to report (their
        # `errors` entry is 0.0 for the same reason).
        return results, errors, _covariance_in_param_space(
            popt, pcov, param_names, param_config)
    return results, errors


# ------------------------------------------------------------------
# Generalised DCS Fitting Procedure (g1-only forward model)
# ------------------------------------------------------------------
#
# The DCS analogue of `fd_model_opt`, and the engine behind
# `DCS_Stream.fit_to_bfi()`: every DCS forward model in `forward.dcs`
# (`si_dcs_g1`, `two_layer_dcs_g1`, `n_layer_dcs_g1`) returns g1 only,
# never g2 -- per invariant 4 of the milob-package-design skill, the Siegert
# relation (or identity, for interferometric DCS) is applied here as a
# separate, explicit, `forward.observation`-backed step, not baked into the
# forward model. `beta` is therefore an observation parameter (via
# `param_config`/`fixed_params`, same as any other flat name), not a
# forward-model kwarg.

def _dcs_g1_siegert_observation(g1, flat):
    return _observation.siegert(g1, flat["beta"])


def _dcs_g1_identity_observation(g1, flat):
    return _observation.identity(g1)


_DCS_G1_OBSERVATIONS = {
    "siegert": _dcs_g1_siegert_observation,
    "identity": _dcs_g1_identity_observation,
}


def _resolve_dcs_g1_observation(obs):
    """Resolve the named observation operator to a callable."""
    if callable(obs):
        return obs
    try:
        return _DCS_G1_OBSERVATIONS[obs]
    except KeyError:
        raise ValueError(
            f"Unknown observation '{obs}'; choose one of "
            f"{sorted(_DCS_G1_OBSERVATIONS)} or pass a callable (g1, flat) -> recorded_signal."
        )


def _dcs_g1_residual(y, observation_name):
    """Build the residual comparing a model g1 with the measured curve."""
    if observation_name == "identity" or not isinstance(observation_name, str):
        y = np.asarray(y, dtype=complex)
        return np.concatenate([y.real, y.imag])
    return np.real(np.asarray(y, dtype=float))


def dcs_g1_model_opt(
    tau,
    data,
    forward_model,
    forward_parameters,
    param_config,
    fixed_params=None,
    assemble=None,
    observation="siegert",
    n_starts=None,
    random_state=0,
    n_jobs=1,
    sigma=None,
    absolute_sigma=False,
    return_covariance=False,
):
    """
    Fit a DCS forward model to one measured correlation curve.

    Fits a single channel and wavelength at a time.

    Parameters
    ----------
    tau : array-like
        Correlation delays in seconds, one per sample in ``data``.
    data : array-like
        Measured curve: real g2(tau) when ``observation='siegert'``, or
        complex g1(tau) when ``observation='identity'``.
    forward_model : callable
        Model called once per delay as
        ``forward_model(tau=tau_i, **model_kwargs)``, returning complex g1
        normalised so that g1(0) = 1.
    forward_parameters : dict
        Fixed context passed to ``forward_model``, such as absorption,
        scattering, refractive index and the motion model.
    param_config : dict
        Free parameters and their bounds.
    fixed_params : dict, optional
        Parameters held at a known value rather than fitted.
    assemble : callable, optional
        Maps the merged flat parameter dict onto the model's arguments.
    observation : {'siegert', 'identity'}
        Maps the model's g1 onto the measured quantity.
    n_starts : int, optional
        Number of optimiser runs from different starting points, keeping the
        best. The first start is the bounds' midpoint and the rest form a
        Latin-hypercube design. Scales with the number of free parameters when
        None; 1 recovers plain single-start behaviour.
    random_state : int or numpy.random.Generator
        Seed for the Latin-hypercube starting points.
    n_jobs : int
        Parallel jobs across the starts. Default 1, meaning serial.
    sigma : float or array-like, optional
        Known per-point noise on the transformed residual. If None, the
        reported covariance is self-calibrated from the fit's own residual,
        which makes a noiseless synthetic fit report a near-zero uncertainty.
    absolute_sigma : bool
        Treat ``sigma`` as a known absolute noise level, anchoring the
        covariance to it. Default False, matching ``curve_fit``.
    return_covariance : bool
        Also return the full parameter covariance matrix. Default False.

    Returns
    -------
    results, errors : dict, dict
        Fitted values and one-sigma uncertainties, keyed by every name in
        ``param_config`` and ``fixed_params``, the latter with zero error.
    covariance : np.ndarray, optional
        Full covariance over the free parameters. Returned only when
        ``return_covariance`` is True.
    """
    if fixed_params is None:
        fixed_params = {}

    tau = np.asarray(tau, dtype=float)
    obs_fn = _resolve_dcs_g1_observation(observation)

    param_names = [p for p in param_config if p not in fixed_params]
    n_params = len(param_names)

    if assemble is None:
        # No packing needed (e.g. si_dcs_g1's flat kwargs already match) --
        # but the flat dict still carries observation-only keys (`beta`),
        # which forward_model does not accept. Filter to the forward
        # model's actual explicit keyword names (not **kwargs catch-alls,
        # which would defeat the filter by accepting anything) rather than
        # passing the flat dict straight through, so `observation="siegert"`
        # keeps working out of the box for the common single-medium case.
        accepted = {
            name for name, p in inspect.signature(forward_model).parameters.items()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }

    def evaluate(flat):
        model_kwargs = assemble(flat) if assemble is not None else {
            k: v for k, v in flat.items() if k in accepted
        }
        g1 = np.array([forward_model(tau=t, **model_kwargs) for t in tau])
        return obs_fn(g1, flat)

    def fit_func(tau, *x):
        fit_params = {
            name: _unit_interval_from(x[i], param_config[name])
            for i, name in enumerate(param_names)
        }
        flat = {**forward_parameters, **fit_params, **fixed_params}
        return _dcs_g1_residual(evaluate(flat), observation)

    ydata = _dcs_g1_residual(data, observation)

    resolved_n_starts = default_n_starts(n_params) if n_starts is None else n_starts

    bounds = ([0] * n_params, [1] * n_params)
    starts = [np.full(n_params, 0.5)]
    n_extra = max(resolved_n_starts, 1) - 1
    if n_extra > 0:
        sampler = qmc.LatinHypercube(d=n_params, seed=random_state)
        starts += list(0.02 + 0.96 * sampler.random(n=n_extra))

    best = _multistart_optimize(fit_func, tau, ydata, starts, bounds,
                                 n_jobs=n_jobs, context="dcs_g1_model_opt",
                                 sigma=_resolve_sigma(sigma, len(ydata)),
                                 absolute_sigma=absolute_sigma)
    if best is None:
        print("Fit failed: every start raised RuntimeError")
        failed = (
            {name: np.nan for name in param_config},
            {name: np.nan for name in param_config},
        )
        return (*failed, _failed_covariance(param_names)) if return_covariance else failed
    _, popt, pcov = best

    param_std = np.sqrt(np.diag(pcov))

    results, errors = {}, {}
    for i, name in enumerate(param_names):
        cfg = param_config[name]
        results[name] = _unit_interval_from(popt[i], cfg)
        errors[name] = _unit_interval_uncertainty(popt[i], param_std[i], cfg)

    results.update(fixed_params)
    for name in fixed_params:
        errors[name] = 0.0

    if return_covariance:
        # Full covariance over the *fitted* parameters only -- `fixed_params`
        # were never estimated, so they have no variance to report (their
        # `errors` entry is 0.0 for the same reason).
        return results, errors, _covariance_in_param_space(
            popt, pcov, param_names, param_config)
    return results, errors


# =======================================================================
# Joint multi-modality fitting
# =======================================================================
#
# One optimisation over a parameter vector shared by measurements of
# different kinds -- the FD-DOS + CW-DCS fusion the framework's Sec 8
# describes, where the joint null space is the intersection of the
# individual ones.
#
# This is deliberately *additive*: `fd_model_opt` and `dcs_g1_model_opt`
# are untouched and remain the way to fit either modality on its own. What
# is new is a way to hand the same shared parameters to several
# measurements at once.
#
# The abstraction is small because both existing fitters already contain
# it. Strip the multistart bookkeeping from either and what remains is a
# pair: a function from parameters to a model vector, and the data vector
# to compare it against. `ResidualBlock` is just that pair made explicit
# so more than one can be summed.
#
# What a joint fit is actually *for*
# ----------------------------------
# Not the point estimates. With FD-DOS pinning `mua`/`musp` and CW-DCS
# pinning `aDb`, the two blocks constrain nearly disjoint parameters, so
# fitting them together barely moves the recovered values relative to the
# usual two-step workflow (fit FD, feed its mua/musp into the DCS fit).
#
# The payoff is the **uncertainty**. The two-step workflow treats the
# FD-derived `mua`/`musp` as exact when it fits `aDb`, so the BFi error bar
# it reports omits their contribution entirely. A joint fit propagates it,
# through the `mua`-`aDb` off-diagonal of the covariance this function
# returns -- which is why `return_covariance` defaults to True here and not
# in the single-modality fitters.


class ResidualBlock:
    """
    One measurement's contribution to a joint cost.

    Holds a model-data pair with its own noise level and fixed context, so
    blocks measured at different wavelengths, distances or modulation
    frequencies can share a parameter vector without sharing anything else.

    Parameters
    ----------
    model_fn : callable
        Maps the merged parameter dict to a vector the same length as
        ``data``, already through whatever transform the data went through.
    data : array-like
        The measured vector, transformed the same way.
    sigma : float or array-like
        Measurement noise on ``data``, in the same units. Required, since
        least squares otherwise weights blocks by point count and units.
    forward_parameters : dict, optional
        Fixed context merged in before ``model_fn`` is called.
    name : str
        Label used in error messages.
    """

    def __init__(self, model_fn, data, sigma, forward_parameters=None, name="block"):
        self.name = name
        self.data = np.asarray(data, dtype=float)
        self.model_fn = model_fn
        self.forward_parameters = dict(forward_parameters or {})
        if sigma is None:
            raise ValueError(
                f"Block '{name}' has no sigma. A joint fit needs a noise "
                f"level per block: with two measurements of different kinds "
                f"there is no way to infer their *relative* weight from the "
                f"data alone, and getting it wrong misstates the reported "
                f"uncertainty by roughly the ratio of the two noise levels. "
                f"See joint_model_opt's Notes."
            )
        self.sigma = _resolve_sigma(sigma, len(self.data))

    @property
    def n_points(self):
        return len(self.data)

    def residual(self, flat):
        """
        Evaluate this block's weighted residual.

        Parameters
        ----------
        flat : dict
            Merged parameter values.

        Returns
        -------
        np.ndarray
            Residual divided by ``sigma``.
        """
        merged = {**self.forward_parameters, **flat}
        return (np.asarray(self.model_fn(merged), dtype=float) - self.data) / self.sigma


def _accepted_kwargs(forward_model):
    """Return the keyword arguments a forward model accepts."""
    return {
        name for name, p in inspect.signature(forward_model).parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.KEYWORD_ONLY)
    }


def fd_block(rho, data, forward_model, forward_parameters, *, sigma,
             assemble=None, residual_transform="reference_channel",
             name="fd", **residual_kwargs):
    """
    Build an FD-DOS block for :func:`joint_model_opt`.

    Parameters
    ----------
    rho : array-like
        Source-detector distances in cm.
    data : array-like
        Measured complex fluence, one value per distance.
    forward_model : callable
        Model evaluated per distance.
    forward_parameters : dict, optional
        Fixed context passed to ``forward_model``.
    sigma : float or array-like
        Noise on the transformed residual, not on the raw complex fluence.
    assemble : callable, optional
        Maps the merged flat parameter dict onto the model's arguments.
    residual_transform : str or callable
        Applied identically to model and data.
    name : str, optional
        Label used in error messages.

    Returns
    -------
    ResidualBlock
    """
    rho = np.asarray(rho, dtype=float)
    transform = _resolve_fd_residual_transform(residual_transform)
    data = np.asarray(data)
    ydata = transform(np.abs(data), np.angle(data), rho, **residual_kwargs)

    accepted = None if assemble is not None else _accepted_kwargs(forward_model)

    def model_fn(flat):
        model_kwargs = (assemble(flat) if assemble is not None
                        else {k: v for k, v in flat.items() if k in accepted})
        phi = np.array([forward_model(rho=r, **model_kwargs) for r in rho])
        return transform(np.abs(phi), np.angle(phi), rho, **residual_kwargs)

    return ResidualBlock(model_fn, ydata, sigma, forward_parameters, name)


def dcs_block(tau, data, forward_model, forward_parameters, *, sigma,
              assemble=None, observation="siegert", name="dcs"):
    """
    Build a DCS block for :func:`joint_model_opt`.

    The forward model returns g1 and the observation step is applied
    separately, so ``beta`` remains an observation parameter.

    Parameters
    ----------
    tau : array-like
        Correlation delays in seconds.
    data : array-like
        Measured g2(tau), or complex g1 for interferometric DCS.
    forward_model : callable
        Model evaluated per delay.
    forward_parameters : dict, optional
        Fixed context passed to ``forward_model``.
    sigma : float or array-like
        Noise on the measured correlation function.
    assemble : callable, optional
        Maps the merged flat parameter dict onto the model's arguments.
    observation : {'siegert', 'identity'}
        Maps the model's g1 onto the measured quantity.
    name : str, optional
        Label used in error messages.

    Returns
    -------
    ResidualBlock
    """
    tau = np.asarray(tau, dtype=float)
    obs_fn = _resolve_dcs_g1_observation(observation)
    ydata = _dcs_g1_residual(data, observation)

    accepted = None if assemble is not None else _accepted_kwargs(forward_model)

    def model_fn(flat):
        model_kwargs = (assemble(flat) if assemble is not None
                        else {k: v for k, v in flat.items() if k in accepted})
        # Per-tau loop, not a vectorised call: only si_dcs_g1 accepts an
        # array tau -- the layered and curved geometries raise on one.
        g1 = np.array([forward_model(tau=t, **model_kwargs) for t in tau])
        return _dcs_g1_residual(obs_fn(g1, flat), observation)

    return ResidualBlock(model_fn, ydata, sigma, forward_parameters, name)


def joint_model_opt(blocks, param_config, fixed_params=None, n_starts=None,
                    random_state=0, n_jobs=1, return_covariance=True,
                    absolute_sigma=True):
    """
    Fit one shared parameter vector against several measurement blocks.

    Each block's weighted residual is concatenated into a single cost, so a
    parameter appearing in more than one block is constrained by all of them.
    A parameter a block's model does not accept is filtered out for that
    block.

    Each block must carry a ``sigma``. Without per-block weighting, a block's
    influence would scale with its point count and units rather than its
    precision, which leaves the recovered values roughly unchanged but
    misstates the uncertainty by about the ratio of the blocks' noise levels.

    Parameters
    ----------
    blocks : list of ResidualBlock
        Built with :func:`fd_block` or :func:`dcs_block`.
    param_config : dict
        Free parameters and their bounds, shared across every block.
    fixed_params : dict, optional
        Parameters held at a known value rather than fitted.
    n_starts : int, optional
        Number of multi-start attempts.
    random_state : int or numpy.random.Generator
        Seed for the starting points.
    n_jobs : int
        Parallel jobs across the starts. Default 1.
    return_covariance : bool
        Also return the full parameter covariance. Default True here, since
        the cross-covariance is the reason to fit jointly.
    absolute_sigma : bool
        Treat each block's ``sigma`` as a known absolute noise level. Default
        True. Set False when the sigmas are only relative weights, in which
        case the covariance is rescaled by the fit's residual variance.

    Returns
    -------
    results, errors : dict, dict
        Fitted values and one-sigma uncertainties.
    covariance : np.ndarray, optional
        Full covariance over the free parameters, ordered as ``param_config``
        minus ``fixed_params``. Returned when ``return_covariance`` is True.
    """
    if fixed_params is None:
        fixed_params = {}
    if not blocks:
        raise ValueError("joint_model_opt needs at least one block.")

    param_names = [p for p in param_config if p not in fixed_params]
    n_params = len(param_names)
    if n_params == 0:
        raise ValueError("Every parameter in param_config is also in "
                         "fixed_params -- nothing left to fit.")

    def fit_func(_, *x):
        fit_params = {
            name: _unit_interval_from(x[i], param_config[name])
            for i, name in enumerate(param_names)
        }
        flat = {**fit_params, **fixed_params}
        return np.concatenate([b.residual(flat) for b in blocks])

    # Each block's residual() already subtracts its data and divides by its
    # sigma, so the target is zero and the weighting is done. Passing
    # sigma=ones with absolute_sigma=True then stops curve_fit re-scaling
    # the covariance by the residual variance, which would undo exactly
    # what the per-block sigmas established. With absolute_sigma=False it
    # does rescale, which is what you want when the sigmas were only
    # relative weights.
    n_total = sum(b.n_points for b in blocks)
    ydata = np.zeros(n_total)
    xdata = np.arange(n_total)

    resolved_n_starts = default_n_starts(n_params) if n_starts is None else n_starts
    bounds = ([0] * n_params, [1] * n_params)
    starts = [np.full(n_params, 0.5)]
    n_extra = max(resolved_n_starts, 1) - 1
    if n_extra > 0:
        sampler = qmc.LatinHypercube(d=n_params, seed=random_state)
        starts += list(0.02 + 0.96 * sampler.random(n=n_extra))

    best = _multistart_optimize(fit_func, xdata, ydata, starts, bounds,
                                 n_jobs=n_jobs, context="joint_model_opt",
                                 sigma=np.ones(n_total),
                                 absolute_sigma=absolute_sigma)
    if best is None:
        print("Fit failed: every start raised RuntimeError")
        failed = (
            {name: np.nan for name in param_config},
            {name: np.nan for name in param_config},
        )
        return (*failed, _failed_covariance(param_names)) if return_covariance else failed

    _, popt, pcov = best
    param_std = np.sqrt(np.diag(pcov))

    results, errors = {}, {}
    for i, name in enumerate(param_names):
        cfg = param_config[name]
        results[name] = _unit_interval_from(popt[i], cfg)
        errors[name] = _unit_interval_uncertainty(popt[i], param_std[i], cfg)

    results.update(fixed_params)
    for name in fixed_params:
        errors[name] = 0.0

    if return_covariance:
        return results, errors, _covariance_in_param_space(
            popt, pcov, param_names, param_config)
    return results, errors


class JointFitResult(NamedTuple):
    """
    Result of a fit spanning more than one parameter space.

    A named 2-tuple, so it unpacks as ``tissue, optical = result``. Both
    streams carry the same ``covariance`` sidecar, whose param axis spans
    every fitted parameter across both spaces, so a cross-space term is
    readable from either.

    Attributes
    ----------
    tissue : TissueStream or None
        Composition space. None when the fit recovered no composition
        parameters.
    optical : OptPropStream or None
        Optical properties at the requested wavelengths, with any observation
        parameters in ``obs_params``. None when the fit recovered neither.
    """
    tissue: object
    optical: object


def fit_joint(blocks, param_config, *, wavelengths, fixed_params=None,
              motion_model="brownian", name=None, events=None,
              history=None, n_starts=None, n_jobs=1, random_state=0,
              absolute_sigma=True, length_unit="cm"):
    """
    Run a joint fit and pack the result into one stream per parameter space.

    Parameters
    ----------
    blocks : list of ResidualBlock
        Built with :func:`fd_block` or :func:`dcs_block`, all sharing
        ``param_config``.
    param_config : dict
        Free parameters and their bounds.
    wavelengths : float or sequence of float
        Wavelengths at which to evaluate the returned optical properties.
        Where composition was recovered these are evaluated through the
        spectral model, so they need not be measured wavelengths.
    fixed_params : dict, optional
        Parameters held at a known value rather than fitted.
    motion_model : str
        Scatterer-motion model, selecting the flow storage label and unit.
    name : str, optional
        Name carried onto the returned streams.
    events : Events, optional
        Events carried onto the returned streams.
    history : list, optional
        History carried onto the returned streams.
    n_starts : int, optional
        Number of multi-start attempts.
    n_jobs : int
        Parallel jobs across the starts. Default 1.
    random_state : int or numpy.random.Generator
        Seed for the starting points.
    absolute_sigma : bool
        Treat each block's ``sigma`` as a known absolute noise level.
    length_unit : str
        Unit the blocks' forward models worked in, recorded on the output.

    Returns
    -------
    JointFitResult
        A (tissue, optical) pair, either of which may be None. Both carry the
        same full-parameter covariance. No probe is attached, since the result
        collapses every block onto a single fitted location.
    """
    import xarray as xr

    from ..core.opt_prop_stream import OptPropStream, fit_attrs
    from ..core.tissue_stream import TissueStream
    from ..forward.dynamics import to_storage_label
    from ..forward.spectral import split_params_by_space

    fixed_params = fixed_params or {}
    param_names = [p for p in param_config if p not in fixed_params]
    composition, medium, observation = split_params_by_space(param_names)

    results, errors, cov = joint_model_opt(
        blocks, param_config, fixed_params=fixed_params, n_starts=n_starts,
        n_jobs=n_jobs, random_state=random_state,
        return_covariance=True, absolute_sigma=absolute_sigma)

    wavelengths = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    channel = ['fitted_location']
    time = [0.0]
    storage = [to_storage_label(n, motion_model) for n in param_names]

    # One covariance for the whole fit, shared by both streams. Its param
    # axis is a superset of either stream's payload axis -- that is the
    # point: it is the only object that sees across the spaces.
    cov_da = xr.DataArray(
        cov[np.newaxis, np.newaxis, :, :],
        coords={'time': time, 'channel': channel,
                'param': storage, 'param_2': storage},
        dims=['time', 'channel', 'param', 'param_2'],
        attrs={'description': 'parameter covariance from the joint fit'},
    )

    def _pack(names, axis, extra_attrs=None):
        if not names:
            return None, None
        values = np.array([[[results[n] for n in names]]])
        sigmas = np.array([[[errors[n] for n in names]]])
        coords = {'time': time, 'channel': channel, axis: names}
        dims = ['time', 'channel', axis]
        return (xr.DataArray(values, coords=coords, dims=dims,
                             attrs=dict(extra_attrs or {})),
                xr.DataArray(sigmas, coords=coords, dims=dims,
                             attrs={'description': '1-sigma from the joint fit'}))

    stream_kwargs = dict(name=name, events=events)

    tissue = None
    if composition:
        data, unc = _pack(composition, 'component',
                          {'transformation': 'fit_joint'})
        tissue = TissueStream(data=data, uncertainty=unc, covariance=cov_da,
                              status='absolute_conc',
                              history=list(history or []), **stream_kwargs)

    optical = None
    if medium or observation:
        # theta at the requested wavelengths. Composition-derived mua/musp
        # come through S (so any wavelength is available, measured or not);
        # directly fitted theta parameters (a free musp, bfi) are repeated
        # across the wavelength axis, being achromatic or already
        # wavelength-specific by name.
        op_slices = []
        for wl in wavelengths:
            entries = {}
            if composition and {'HbO', 'HbR'} <= set(composition):
                from ..forward.spectral import extinction_at, mua_from_composition
                eps_hbo, eps_hbr = extinction_at(float(wl))
                entries['mua'] = mua_from_composition(
                    results['HbO'], results['HbR'], eps_hbo, eps_hbr)
            if composition and {'A', 'b'} <= set(composition):
                from ..forward.spectral import musp_powerlaw
                entries['musp'] = musp_powerlaw(results['A'], results['b'],
                                                float(wl))
            for n in medium:
                entries[to_storage_label(n, motion_model)] = results[n]
            op_slices.append(entries)

        labels = list(op_slices[0])
        op_values = np.array([[[[s[l] for l in labels] for s in op_slices]]])
        op_coords = {'time': time, 'channel': channel,
                     'wavelength': wavelengths, 'op': labels}
        op_dims = ['time', 'channel', 'wavelength', 'op']
        op_da = xr.DataArray(op_values, coords=op_coords, dims=op_dims)
        op_da.attrs.update(fit_attrs(
            fitting_model='joint', transformation='fit_joint',
            length_unit=length_unit,
            motion_model=motion_model if any(
                l.startswith('bfi') for l in labels) else None))

        # 1-sigma on the directly fitted entries; derived mua/musp are left
        # NaN here rather than guessed -- use TissueStream.to_op(), which
        # propagates them properly from the covariance.
        err_values = np.full_like(op_values, np.nan)
        for li, label in enumerate(labels):
            if label in storage:
                fit_name = param_names[storage.index(label)]
                err_values[..., li] = errors[fit_name]
        err_da = xr.DataArray(err_values, coords=op_coords, dims=op_dims)

        obs_da, obs_err = _pack(observation, 'obs_param')
        optical = OptPropStream(
            data=op_da, uncertainty=err_da, covariance=cov_da,
            obs_params=obs_da, obs_uncertainty=obs_err,
            status='op', history=list(history or []), **stream_kwargs)

    return JointFitResult(tissue=tissue, optical=optical)
