import copy

import numpy as np
import xarray as xr
from typing import Optional
from .datastream import Datastream
from .nirs import NirsStream


class DCS_Stream(NirsStream):
    """
    Continuous-wave diffuse correlation spectroscopy data.

    Data has dims (time, channel, wavelength, tau), where 'tau' holds the
    correlation delay times in seconds. Blood flow is recovered by
    :meth:`fit_to_bfi`, which returns an
    :class:`~milob.core.opt_prop_stream.OptPropStream`.

    Parameters
    ----------
    data : xr.DataArray
        Shape (time, channel, wavelength, tau).
    probe : Probe
        Optode geometry.

    Attributes
    ----------
    data.attrs['observation'] : {'g2', 'g1'}
        Which quantity the data holds: 'g2' for conventional DCS, real-valued,
        or 'g1' for interferometric DCS, complex-valued.
    status : {'raw', 'processed'}
        'raw' as loaded or simulated, 'processed' after a transform such as
        :meth:`aggregate_channels`.
    """

    def __init__(self, data, probe, **kwargs):
        super().__init__(data, probe, **kwargs)

        if 'tau' not in self.data.dims:
            raise ValueError(
                "DCS_Stream requires a 'tau' dimension (correlation delays in seconds)."
            )

        # 'g2' (conventional DCS) is the only kind any current loader/
        # simulator produces -- setdefault() so it's always present without
        # requiring every existing construction path (from_snirf,
        # simulate_*_dcs_stream, ...) to set it explicitly, while still
        # letting an interferometric-DCS loader mark 'g1' data when one
        # exists.
        self.data.attrs.setdefault('observation', 'g2')

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def taus(self) -> np.ndarray:
        """Correlation delay times, in seconds."""
        return self.data.tau.values

    @property
    def observation(self) -> str:
        """
        Quantity held in ``data``.

        Returns
        -------
        {'g2', 'g1'}
            'g2' for conventional DCS, real-valued; 'g1' for interferometric DCS,
            complex-valued and normalised so that g1(0) = 1.
        """
        return self.data.attrs.get('observation', 'g2')

    # ------------------------------------------------------------------
    # Transformations  (stubs — implementations in processing/dcs.py)
    # ------------------------------------------------------------------

    

    def _resolve_fit_groups(self, channel_grouping):
        """Normalise ``channel_grouping`` to a {name: [channel index]} mapping."""
        labels = [str(c) for c in self.data.channel.values]

        if channel_grouping is None or channel_grouping == 'per_channel':
            return {lbl: [i] for i, lbl in enumerate(labels)}

        if channel_grouping in ('shared', 'all'):
            # 'fitted_location' matches FD_Stream.fit_to_op's label for the
            # same idea: one parameter vector for a spatial location rather
            # than for an individual channel.
            return {'fitted_location': list(range(len(labels)))}

        if isinstance(channel_grouping, dict):
            from ..processing.dcs import _resolve_manual_groups
            # position_tol=np.inf: aggregate_channels warns when grouped
            # channels don't share an SDS because it *averages* them, which
            # only makes sense for co-located channels. Here differing SDS
            # is the entire point -- the rho-spread is what constrains a
            # layered model -- so that warning would be backwards.
            return _resolve_manual_groups(self.data, labels, channel_grouping,
                                          position_tol=np.inf)

        raise ValueError(
            f"channel_grouping={channel_grouping!r} not understood. Use None/"
            f"'per_channel' (one fit per channel), 'shared' (one fit using "
            f"every channel), or a dict {{name: [channels]}}."
        )

    def fit_to_bfi(
        self,
        forward_model=None,
        assemble=None,
        param_config: Optional[dict] = None,
        fixed_params: Optional[dict] = None,
        forward_parameters: Optional[dict] = None,
        observation: Optional[str] = None,
        channel_grouping=None,
        n_starts: Optional[int] = None,
        n_jobs: int = 1,
        sigma: Optional[float] = None,
        absolute_sigma: bool = False,
        inplace: bool = False,
    ) -> "OptPropStream":
        """
        Fit a DCS forward model to the correlation curves.

        Every (time, channel, wavelength) slice is fitted independently unless
        ``channel_grouping`` combines channels into a shared fit. DCS cannot
        identify ``mua`` and ``musp`` from its own data, so supply them through
        ``forward_parameters``, typically from a co-registered FD-DOS measurement.

        Parameters
        ----------
        forward_model : callable, optional
            Model to fit. Defaults to :func:`~milob.forward.dcs.si_dcs_g1`.
        assemble : callable, optional
            Maps flat fitted parameter names onto the model's arguments. Not
            needed for the semi-infinite model.
        param_config : dict, optional
            Parameters to fit and their bounds. The default fits ``aDb`` and
            ``beta`` and applies only to the semi-infinite model; other geometries
            must supply their own.
        fixed_params : dict, optional
            Parameters held at a known value rather than fitted.
        forward_parameters : dict, optional
            Fixed context passed to ``forward_model`` alongside the resolved
            ``rho`` and ``wavelength``, such as ``mua``, ``musp`` and ``n``.
        observation : {'siegert', 'identity'}, optional
            Maps the model's g1 to the measured quantity. Defaults to 'siegert'
            for g2 data and 'identity' for g1 data.
        channel_grouping : {None, 'per_channel', 'shared'} or dict, optional
            Which channels share one parameter vector. None or 'per_channel'
            fits each channel separately. 'shared' fits all channels jointly and
            returns a single 'fitted_location' channel. A dict of
            {name: [channels]} defines explicit groups; unlisted channels are
            dropped.
        n_starts : int, optional
            Number of multi-start attempts. Scales with the number of free
            parameters if None.
        n_jobs : int
            Parallel jobs across independent slices. Default 1, meaning serial.
        sigma : float, optional
            Known per-point noise level on the correlation curve. If None, the
            reported uncertainty is self-calibrated from each slice's residual.
        absolute_sigma : bool
            Anchor the uncertainty to ``sigma`` rather than self-calibrating.
            Default False.
        inplace : bool
            Ignored; a new object is always returned.

        Returns
        -------
        OptPropStream
            Shape (time, channel, wavelength, op). For the semi-infinite default,
            op is ['bfi'], holding the fitted flow parameter under a
            motion-model-agnostic label. The fitted ``beta`` belongs to the
            observation operator and is returned in ``obs_params`` instead.
            ``motion_model`` and ``flowUnit`` are recorded in ``data.attrs``,
            since bfi is cm^2/s under Brownian motion and cm^2/s^2 under random
            ballistic flow.
        """
        from ..processing.fitting import (dcs_g1_model_opt, dcs_block,
                                          joint_model_opt, run_parallel_fits)
        from ..forward.dcs import si_dcs_g1
        from .opt_prop_stream import (OptPropStream, pack_fit_arrays,
                                      pack_covariance, fit_attrs)

        if forward_model is None:
            forward_model = si_dcs_g1

        if param_config is None:
            param_config = {
                "aDb": {"bounds": (1e-10, 1e-6), "log": True},
                "beta": {"bounds": (0.0, 1.0), "log": False},
            }

        if observation is None:
            observation = "siegert" if self.observation == "g2" else "identity"

        distances = self.data.coords['distance'].values
        if self.data.attrs.get('lengthUnit') == 'mm':
            distances = distances / 10.0
        distances = np.asarray(distances, dtype=float)

        taus = self.taus
        time_vals = self.data.time.values
        channel_vals = self.data.channel.values
        wavelength_vals = self.data.wavelength.values
        n_time = len(time_vals)
        n_ch = len(channel_vals)
        n_wl = len(wavelength_vals)

        # Plain array, not the xarray object -- lightweight to close over
        # when n_jobs != 1 dispatches each slice to a separate process.
        data_arr = self.data.values  # (time, channel, wavelength, tau)

        # ------------------------------------------------------------------
        # Per-wavelength / per-slice optical properties in `forward_parameters`.
        #
        # `mua`/`musp` are the only fixed context that genuinely varies with
        # wavelength (and, via a co-registered DOS time series, with time).
        # Everything else in `forward_parameters` -- `n`, `z`, `depth`,
        # `radius_outer`, per-layer motion params like `tc`/`aV2`, ... -- is
        # geometry/dynamics context that stays a scalar (or a per-*layer*
        # array whose length has nothing to do with `n_wl`), so it is left
        # completely untouched here.
        #
        # An optical-property entry may be:
        #   * a scalar                       -> used for every slice (unchanged)
        #   * a length-`n_wl` vector         -> one value per wavelength
        #   * anything broadcastable to
        #     (time, channel, wavelength)    -> fully time/channel-resolved
        # Matched entries are expanded to the full (n_time, n_ch, n_wl) grid
        # now so `_fit_slice` can pull a plain scalar per slice; the forward
        # model therefore still only ever sees scalar `mua`/`musp`. A shape
        # that will not broadcast (e.g. a bare length-`n_time` vector, which
        # is ambiguous) raises here rather than being silently collapsed to
        # its first element inside `si_dcs_g1`'s scalar-return path.
        #
        # Layered/curved geometries name their per-layer properties `mua_1`,
        # `musp_2`, ...  -- those match too, so e.g. a two-layer fit of
        # multi-wavelength data can pass `mua_1`/`mua_2` as per-wavelength
        # vectors and each layer still gets the right value per wavelength.
        def _is_optical_property(name):
            for stem in ("mua", "musp"):
                if name == stem:
                    return True
                if name.startswith(stem + "_") and name[len(stem) + 1:].isdigit():
                    return True
            return False

        forward_parameters = dict(forward_parameters) if forward_parameters else {}
        fp_target = (n_time, n_ch, n_wl)
        for _name, _val in forward_parameters.items():
            if not _is_optical_property(_name):
                continue
            _arr = np.asarray(_val)
            if _arr.ndim == 0:
                continue  # scalar -- passed straight through
            try:
                forward_parameters[_name] = np.broadcast_to(_arr, fp_target)
            except ValueError:
                raise ValueError(
                    f"forward_parameters[{_name!r}] has shape {_arr.shape}; an "
                    f"optical property must be a scalar, a length-{n_wl} array "
                    f"(one value per wavelength), or an array broadcastable to "
                    f"(time, channel, wavelength) = {fp_target}."
                ) from None

        groups = self._resolve_fit_groups(channel_grouping)
        group_names = list(groups)
        n_grp = len(group_names)

        param_names = list(param_config.keys())
        n_p = len(param_names)
        op_array = np.full((n_time, n_grp, n_wl, n_p), np.nan)
        err_array = np.full_like(op_array, np.nan)
        cov_array = np.full((n_time, n_grp, n_wl, n_p, n_p), np.nan)

        slices = [(ti, gi, wi, wl)
                  for ti in range(n_time)
                  for gi in range(n_grp)
                  for wi, wl in enumerate(wavelength_vals)]
        # See FD_Stream.fit_to_op's identical comment: route n_jobs into
        # whichever loop actually has more than one thing to do -- across
        # slices when there's more than one, or into dcs_g1_model_opt's own
        # multi-start loop when there's only a single slice to fit.
        single_slice = len(slices) == 1
        outer_n_jobs = 1 if single_slice else n_jobs
        inner_n_jobs = n_jobs if single_slice else 1

        def _slice_forward_parameters(ti, ci, wi, wl):
            out = {"rho": distances[ci], "wavelength": float(wl)}
            for _name, _val in forward_parameters.items():
                # Array entries were expanded to (n_time, n_ch, n_wl) above
                # (semi-infinite path only) -- pull this slice's scalar.
                # Everything else (genuine scalars, per-layer arrays on the
                # layered path) passes straight through unchanged.
                if isinstance(_val, np.ndarray) and _val.ndim == 3:
                    out[_name] = _val[ti, ci, wi]
                else:
                    out[_name] = _val
            return out

        def _fit_slice(ti, gi, wi, wl):
            members = groups[group_names[gi]]

            if len(members) == 1:
                # Unchanged single-channel path.
                ci = members[0]
                return dcs_g1_model_opt(
                    tau=taus,
                    data=data_arr[ti, ci, wi, :],
                    forward_model=forward_model,
                    forward_parameters=_slice_forward_parameters(ti, ci, wi, wl),
                    param_config=param_config,
                    fixed_params=fixed_params,
                    assemble=assemble,
                    observation=observation,
                    n_starts=n_starts,
                    n_jobs=inner_n_jobs,
                    sigma=sigma,
                    absolute_sigma=absolute_sigma,
                    return_covariance=True,
                )

            # Grouped: one block per member channel, all sharing one
            # parameter vector. The channels differ only in `rho`, and it's
            # that rho-dependence that carries the depth information a
            # layered model needs -- see this method's `channel_grouping`
            # docstring.
            blocks = [
                dcs_block(taus, data_arr[ti, ci, wi, :], forward_model,
                          _slice_forward_parameters(ti, ci, wi, wl),
                          sigma=1.0 if sigma is None else sigma,
                          assemble=assemble, observation=observation,
                          name=str(channel_vals[ci]))
                for ci in members
            ]
            return joint_model_opt(
                blocks, param_config,
                fixed_params=fixed_params,
                n_starts=n_starts,
                n_jobs=inner_n_jobs,
                return_covariance=True,
                # sigma=None means "no noise model supplied", so the unit
                # sigmas above are relative weights only -- let the fit
                # rescale the covariance from its own residual, matching
                # what dcs_g1_model_opt(sigma=None) already does.
                absolute_sigma=absolute_sigma if sigma is not None else False,
            )

        jobs = [(_fit_slice, (ti, gi, wi, wl)) for ti, gi, wi, wl in slices]
        results = run_parallel_fits(jobs, n_jobs=outer_n_jobs)

        for (ti, gi, wi, wl), (result, error, cov) in zip(slices, results):
            for pi, name in enumerate(param_names):
                op_array[ti, gi, wi, pi] = result.get(name, np.nan)
                err_array[ti, gi, wi, pi] = error.get(name, np.nan)
            # `cov` spans only the *free* parameters, in param_config order
            # minus anything pinned via fixed_params; place it into the
            # full-width matrix so every slice has the same labelled axes.
            free = [n for n in param_names if n not in (fixed_params or {})]
            idx = [param_names.index(n) for n in free]
            cov_array[np.ix_([ti], [gi], [wi], idx, idx)] = cov

        # Grouping collapses channels, so the source probe no longer
        # describes the output's channel axis. Drop it in that case rather
        # than carrying a probe whose channel count disagrees with the data
        # -- the same reason FD_Stream.fit_to_op() returns a probe-less
        # stream for its single 'fitted_location' channel.
        regrouped = group_names != [str(c) for c in channel_vals]
        out_probe = None if regrouped else self.probe
        channel_vals = np.array(group_names)

        # Route each fitted parameter to its proper home: properties of the
        # medium onto the 'op' axis (with aDb/aV2 relabelled 'bfi'),
        # parameters of the observation operator B (beta) into obs_params.
        # Keeping beta out of 'op' is the storage half of invariant 4 -- the
        # forward model already never returns g2, and now the stored
        # parameter vector never claims a coherence factor is an optical
        # property either.
        motion_model = (forward_parameters or {}).get('motion', 'brownian')

        op_da, err_da, obs_da, obs_err_da = pack_fit_arrays(
            op_array, err_array, param_names,
            coords={'time': time_vals,
                    'channel': channel_vals,
                    'wavelength': wavelength_vals},
            dims=['time', 'channel', 'wavelength'],
            motion_model=motion_model,
        )

        cov_da = pack_covariance(
            cov_array, param_names,
            coords={'time': time_vals,
                    'channel': channel_vals,
                    'wavelength': wavelength_vals},
            dims=['time', 'channel', 'wavelength'],
            motion_model=motion_model,
        )

        op_da.attrs.update(fit_attrs(
            fitting_model=getattr(forward_model, '__name__', 'custom'),
            transformation='fit_to_bfi',
            length_unit='cm',            # distances were converted above
            motion_model=motion_model,
        ))

        # An array-valued forward parameter (per-wavelength / per-slice
        # mua/musp etc.) is recorded by shape, not by dumping the whole
        # grid, so history stays light and JSON-serialisable.
        history_forward_parameters = {
            k: (f"<ndarray shape={tuple(v.shape)}>"
                if isinstance(v, np.ndarray) and v.ndim else v)
            for k, v in forward_parameters.items()
        }

        history = self.history.copy()
        history.append(self._history_entry("fit_to_bfi", {
            "forward_model": getattr(forward_model, '__name__', 'custom'),
            "param_config": dict(param_config),
            "fixed_params": dict(fixed_params) if fixed_params else {},
            "forward_parameters": history_forward_parameters,
            "observation": observation,
            "motion_model": motion_model,
            "n_starts": n_starts,
        }))

        return OptPropStream(
            data=op_da,
            uncertainty=err_da,
            obs_params=obs_da,
            obs_uncertainty=obs_err_da,
            covariance=cov_da,
            forward_parameters=dict(forward_parameters) if forward_parameters else {},
            probe=out_probe,
            name=self.name,
            events=self.events,
            status="op",
            history=history,
        )

    def aggregate_channels(self, groups: Optional[dict] = None, *,
                            positions: Optional[dict] = None,
                            position_tol: float = 5.0,
                            method: str = 'mean',
                            inplace: bool = False):
        """
        Combine co-located channels sharing a source-detector separation.

        Only the channel dimension is affected; tau is untouched. With ``groups``
        given, members are combined as specified. Without it, channels are
        partitioned by source and each source's detectors clustered by position.

        Parameters
        ----------
        groups : dict, optional
            {new_channel_name: [member channel names or 0-based indices]}. If
            omitted, groups are found automatically and a probe is required.
        positions : dict, optional
            {new_channel_name: (x, y, z)} overriding a group's centroid position,
            in the same units as the probe coordinates.
        position_tol : float
            Distance threshold in mm: the clustering radius in automatic mode, and
            the separation-mismatch warning tolerance in manual mode. Default 5.0.
        method : {'mean', 'median'}
            How each group's channels are combined. Default 'mean'. 'median' is
            not defined for complex g1 data and raises ValueError there.
        inplace : bool
            Modify this stream in place. Default False.

        Returns
        -------
        DCS_Stream
            Stream of the same type with aggregated channels, carrying a new Probe
            holding one synthetic source-detector pair per aggregated group,
            positioned at the group centroid and preserving its mean separation.
            Channels not in any group pass through unchanged. The membership of
            each group is recorded in ``history[-1]['params']['groups']``.

        Raises
        ------
        ValueError
            If ``method='median'`` is used on complex-valued data.

        Examples
        --------
        >>> agg = dcs.aggregate_channels({'frontal_L': ['S1D1', 'S1D2']})
        >>> agg = dcs.aggregate_channels(position_tol=8.0)
        """
        from ..processing.dcs import _resolve_manual_groups, cluster_channels_by_source

        if method not in ('mean', 'median'):
            raise ValueError(f"method={method!r} not understood. Use 'mean' or 'median'.")
        if method == 'median' and np.iscomplexobj(self.data.values):
            raise ValueError(
                "method='median' is not defined for complex-valued data "
                "(observation='g1'). Use method='mean' instead."
            )

        channel_labels = list(self.data.channel.values)

        if groups is not None:
            resolved_groups = _resolve_manual_groups(
                self.data, channel_labels, groups, position_tol
            )
        else:
            if self.probe is None:
                raise ValueError(
                    "Automatic channel aggregation requires a probe (for "
                    "source/detector positions). Pass an explicit `groups` "
                    "dict instead, or attach a probe to this stream."
                )
            if 'source' not in self.data.coords or 'detector' not in self.data.coords:
                raise ValueError(
                    "Automatic aggregation requires 'source' and 'detector' "
                    "coordinates on the 'channel' dimension."
                )

            source_ids = self.data['source'].values
            detector_ids = self.data['detector'].values
            unit_to_mm = 10.0 if self.probe.lengthUnit == 'cm' else 1.0
            detector_positions_mm = self.probe.d_pos[detector_ids - 1] * unit_to_mm  # 1-based -> 0-based

            resolved_groups = cluster_channels_by_source(
                source_ids, detector_positions_mm, position_tol
            )

        new_data, new_probe = _aggregate_build(
            self.data, self.probe, resolved_groups, positions or {}, method
        )

        history_params = {
            'mode': 'manual' if groups is not None else 'automatic',
            'position_tol': position_tol,
            'method': method,
            'n_groups': len(resolved_groups),
            'groups': {
                name: [channel_labels[i] for i in idxs]
                for name, idxs in resolved_groups.items()
            },
        }

        if inplace:
            self.data = new_data
            self.probe = new_probe
            # Same reasoning as the non-inplace branch below: averaging
            # channels would require combining sidecar variances in
            # quadrature, which isn't defined here, so they're dropped
            # loudly rather than left attached at the wrong shape.
            self._carry_sidecars(self, operation='aggregate_channels()',
                                 propagate=False)
            self.add_history('aggregate_channels', history_params)
            self._mark_processed()
            return self

        # propagate_sidecars=False: this is an *average*, not a selection.
        # Carrying a fit uncertainty across would need the member channels'
        # variances combined in quadrature -- a real propagation rule that
        # doesn't exist here -- so the sidecars are dropped with a warning
        # rather than silently reported as though they still applied.
        new_stream = self._rebuild(
            new_data,
            operation='aggregate_channels()',
            propagate_sidecars=False,
            probe=new_probe,
        )
        new_stream.add_history('aggregate_channels', history_params)
        new_stream._mark_processed()
        return new_stream

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self):
        n_tau = len(self.data.tau)
        wls = list(self.data.wavelength.values) if 'wavelength' in self.data.coords else '?'
        return (
            f"<DCS_Stream | {self.name} | Status: {self.status} | "
            f"Channels: {len(self.data.channel)} | "
            f"Wavelengths: {wls} | "
            f"tau bins: {n_tau} | "
            f"Duration: {len(self.data.time)} frames>"
        )


# ---------------------------------------------------------------------------
# aggregate_channels() Probe/Datastream rebuild — shared by DCS_Stream and
# its subclasses. The group-decision logic (manual validation, automatic
# clustering) is pure array math with no Probe dependency, so it lives in
# processing/dcs.py instead (see _resolve_manual_groups() and
# cluster_channels_by_source() there). This function is different: it
# synthesises Probe geometry and rebuilds probe._channels/caches directly,
# which is core object-model bookkeeping, not a swappable algorithm — same
# split as Datastream.drop_channels(), which has no processing/ counterpart
# either.
# ---------------------------------------------------------------------------

def _aggregate_build(data, probe, groups, positions_override, method='mean'):
    """Average grouped channels and rebuild the channel dimension and probe."""
    channel_labels = list(data.channel.values)
    n_channels = len(channel_labels)

    grouped_idx = {i for idxs in groups.values() for i in idxs}
    unused = set(positions_override) - set(groups)
    if unused:
        raise ValueError(
            f"`positions` override given for group(s) not present in "
            f"`groups`: {sorted(unused)}."
        )

    # Output plan, in original channel order: (new_name, member_idxs).
    first_of_group = {min(idxs): name for name, idxs in groups.items()}
    plan = []
    consumed = set()
    for i in range(n_channels):
        if i in consumed:
            continue
        if i in first_of_group:
            name = first_of_group[i]
            idxs = groups[name]
            plan.append((name, idxs))
            consumed.update(idxs)
        elif i not in grouped_idx:
            plan.append((channel_labels[i], [i]))
            consumed.add(i)

    # 'is_bad'/'is_short'/'channel_idx' don't survive a channel-count
    # change; Datastream._initialize_masks() re-derives is_bad/is_short for
    # the new channel set once the rebuilt stream is constructed. 'source'/
    # 'detector'/'distance' are recomputed explicitly below (a plain mean
    # doesn't survive a merge either, and would leave concat with a
    # coordinate present on some slices but not others) -- drop all of them
    # up front so every slice has an identical coord set going into concat.
    drop_coords = [c for c in ('is_bad', 'is_short', 'channel_idx',
                                'source', 'detector', 'distance')
                   if c in data.coords]
    base = data.drop_vars(drop_coords) if drop_coords else data

    slices = []
    for _, idxs in plan:
        if len(idxs) == 1:
            slices.append(base.isel(channel=idxs))
        else:
            reduce = getattr(base.isel(channel=idxs), method)
            slices.append(
                reduce(dim='channel', skipna=False, keep_attrs=True)
                    .expand_dims(channel=[0])
            )

    new_data = xr.concat(slices, dim='channel')
    # A merged slice's .mean()/.median()(dim='channel') drops 'channel' entirely, and
    # .expand_dims(channel=[0]) above puts it back at the front rather than
    # its original position -- transpose back to the source dim order (not
    # just relying on concat's ordering) so a run with no passthrough
    # channels doesn't silently come out as (channel, time, wavelength, ...).
    new_data = new_data.transpose(*data.dims)
    new_data = new_data.assign_coords(channel=[name for name, _ in plan])
    new_data.attrs = dict(data.attrs)

    has_source   = 'source'   in data.coords
    has_detector = 'detector' in data.coords
    has_distance = 'distance' in data.coords

    if probe is None:
        for c in ('source', 'detector', 'distance'):
            if c in new_data.coords:
                new_data = new_data.drop_vars(c)
        return new_data, None

    if grouped_idx and not (has_source and has_detector):
        raise ValueError(
            "Cannot synthesise probe geometry for aggregated channels: "
            "stream has a probe but is missing 'source'/'detector' "
            "coordinates."
        )

    orig_source   = data['source'].values   if has_source   else None
    orig_detector = data['detector'].values if has_detector else None
    orig_distance = data['distance'].values if has_distance else None

    n_orig_sources   = probe.n_sources
    n_orig_detectors = probe.n_detectors

    extra_s_pos, extra_d_pos = [], []
    extra_s_labels, extra_d_labels = [], []
    new_source, new_detector, new_distance = [], [], []

    for name, idxs in plan:
        if len(idxs) == 1:
            i = idxs[0]
            new_source.append(int(orig_source[i]) if has_source else None)
            new_detector.append(int(orig_detector[i]) if has_detector else None)
            new_distance.append(float(orig_distance[i]) if has_distance else None)
            continue

        # Merged group -> synthetic optode pair: midpoint = centroid (or the
        # `positions` override), separation preserved so the pair's own
        # geometric distance matches the group's averaged SDS.
        s_ids = orig_source[idxs]
        d_ids = orig_detector[idxs]
        avg_src = probe.s_pos[s_ids - 1].mean(axis=0)
        avg_det = probe.d_pos[d_ids - 1].mean(axis=0)
        default_mid = (avg_src + avg_det) / 2.0
        mid = np.asarray(positions_override[name], dtype=float) if name in positions_override else default_mid
        half = (avg_det - avg_src) / 2.0

        new_s_idx = n_orig_sources + len(extra_s_pos) + 1
        new_d_idx = n_orig_detectors + len(extra_d_pos) + 1
        extra_s_pos.append(mid - half)
        extra_d_pos.append(mid + half)
        extra_s_labels.append(f"v{name}_S")
        extra_d_labels.append(f"v{name}_D")

        new_source.append(new_s_idx)
        new_detector.append(new_d_idx)
        new_distance.append(
            float(np.mean(orig_distance[idxs])) if has_distance
            else float(np.linalg.norm(avg_det - avg_src))
        )

    if has_source:
        new_data = new_data.assign_coords(source=('channel', new_source))
    if has_detector:
        new_data = new_data.assign_coords(detector=('channel', new_detector))
    if has_distance:
        new_data = new_data.assign_coords(distance=('channel', new_distance))

    new_probe = copy.copy(probe)
    if extra_s_pos:
        new_probe.s_pos = np.vstack([probe.s_pos, np.array(extra_s_pos)])
        new_probe.d_pos = np.vstack([probe.d_pos, np.array(extra_d_pos)])
        new_probe.source_labels   = list(probe.source_labels)   + extra_s_labels
        new_probe.detector_labels = list(probe.detector_labels) + extra_d_labels
    new_probe._channels = {'sources': new_source, 'detectors': new_detector}
    new_probe._channel_labels = [name for name, _ in plan]
    new_probe._distances = new_distance if has_distance else None
    new_probe.rois = {}  # old ROI channel lists no longer valid post-aggregation

    return new_data, new_probe


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

