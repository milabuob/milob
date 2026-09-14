
import logging
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from .output import BaseOutput
from ..analysis.fc_meta import (
    FC_UNITS_META, CHANNEL_QUALITY_COORDS, resolve_z_transform,
)


def _apply_threshold_logic(matrix, threshold_type, threshold_val, binarize):
    """Apply a sparsity or absolute threshold to one connectivity matrix."""
    processed_matrix = matrix.copy()
    n_types = matrix.shape[2]

    for t in range(n_types):
        m = processed_matrix[:, :, t]

        # Identify where we actually have data (non-NaN)
        mask_has_data = ~np.isnan(m)
        valid_values = m[mask_has_data]

        if len(valid_values) == 0:
            continue

        # Determine the cutoff
        if threshold_type == 'sparsity':
            # Calculate percentile based only on valid data
            cutoff = np.percentile(valid_values, 100 * (1 - threshold_val))
        elif threshold_type == 'absolute':
            cutoff = threshold_val
        else:
            continue

        # Apply Threshold -- only touch the parts that HAVE data; parts
        # that were NaN remain NaN.
        below_cutoff = (m < cutoff) & mask_has_data
        above_cutoff = (m >= cutoff) & mask_has_data

        if binarize:
            m[above_cutoff] = 1
            m[below_cutoff] = 0
        else:
            m[below_cutoff] = 0

        processed_matrix[:, :, t] = m

    return processed_matrix


def _self_referential_quality_weight(raw_values, reference_values, steepness=1.0):
    """Convert a raw per-channel quality metric into a weight in [0, 1], normalised against the distribution over the whole probe."""
    reference_values = np.asarray(reference_values, dtype=float)
    center = np.median(reference_values)
    mad = np.median(np.abs(reference_values - center))
    spread = 1.4826 * mad  # scales MAD to be comparable to SD under normality
    z = steepness * (np.asarray(raw_values, dtype=float) - center) / max(spread, 1e-8)
    return 1.0 / (1.0 + np.exp(-z))


# 'metric' is static like 'chromophore', not to-be-collapsed like
# 'freq'/'time': a method emitting several named metrics gives a complete
# point estimate per pair for each, just one more categorical label -- NOT
# a spectral/temporal axis reduce() should average over (that would mix
# unlike metrics). Keeping it out of reduce()'s own extra_dims sweep is the
# same reason it needs to be in this set, not a separate concern.
_STATIC_FC_DIMS = {'channel_i', 'channel_j', 'chromophore', 'wavelength', 'metric'}


def _require_reduced(value_da, method_name):
    """Raise if a frequency- or time-resolved output has not been collapsed by reduce()."""
    extra = [d for d in value_da.dims if d not in _STATIC_FC_DIMS]
    if extra:
        raise ValueError(
            f"{method_name}() requires a reduced (channel x channel x chromophore) "
            f"output, but this one still has {extra} dimension(s) (from a "
            f"frequency-/time-resolved method, e.g. coherence). Call .reduce() first."
        )


def _weighted_nanmean(block, pair_weights):
    """NaN-safe weighted mean of a channel-pair block."""
    w = pair_weights.reshape(pair_weights.shape + (1,) * (block.ndim - 2))
    mask = ~np.isnan(block)
    w_masked = np.where(mask, w, 0.0)
    numer = np.nansum(block * w_masked, axis=(0, 1))
    denom = np.sum(w_masked, axis=(0, 1))
    with np.errstate(invalid='ignore', divide='ignore'):
        result = numer / denom
    return np.where(denom == 0, np.nan, result)


def _weighted_nanstd(block, pair_weights, weighted_mean):
    """NaN-safe weighted standard deviation of a channel-pair block."""
    w = pair_weights.reshape(pair_weights.shape + (1,) * (block.ndim - 2))
    mask = ~np.isnan(block)
    w_masked = np.where(mask, w, 0.0)
    dev_sq = (block - weighted_mean) ** 2
    numer = np.nansum(dev_sq * w_masked, axis=(0, 1))
    denom = np.sum(w_masked, axis=(0, 1))
    with np.errstate(invalid='ignore', divide='ignore'):
        var = numer / denom
    return np.where(denom == 0, np.nan, np.sqrt(var))


# attrs that describe the OUTPUT rather than the estimator that produced it --
# excluded when checking whether two FCOutputs are poolable, since differing
# here is meaningless (they are display metadata or bookkeeping, not settings).
_NON_ESTIMATOR_ATTRS = frozenset({
    'units', 'method', 'range', 'cmap', 'sd_space', 'n_eff',
})


def check_poolable(outputs, tol=0.1, check='warn'):
    """
    Check that a set of connectivity outputs may be averaged together.

    Outputs computed in different value spaces or by different methods are
    never poolable and raise. Outputs from the same method with different
    settings usually are, so they warn instead; the warning is suppressed
    when every output records an effective degrees-of-freedom ``n_eff``
    and those agree to within ``tol``.

    Parameters
    ----------
    outputs : list of FCOutput
        Outputs to check.
    tol : float, optional
        Relative tolerance on the spread of ``n_eff``. Default is 0.1.
    check : {'warn', 'raise', 'ignore'}, optional
        What to do about a settings mismatch. A mismatch of units or
        method always raises unless ``'ignore'``. Default is ``'warn'``.

    Raises
    ------
    ValueError
        If the outputs differ in value space or method.
    """
    if check == 'ignore' or len(outputs) < 2:
        return

    attrs = [o.output.attrs for o in outputs]
    for key in ('units', 'method'):
        seen = {a.get(key) for a in attrs}
        if len(seen) > 1:
            raise ValueError(
                f"FCOutput.average: cannot pool outputs with different "
                f"{key!r} -- got {sorted(str(v) for v in seen)}. These are "
                f"different quantities on different scales; averaging them "
                f"produces a number that means nothing. Average within each "
                f"{key} and compare the results instead."
            )

    n_effs = [a.get('n_eff') for a in attrs]
    have_n_eff = all(v is not None and np.isfinite(v) for v in n_effs)
    if have_n_eff:
        lo, hi = float(np.min(n_effs)), float(np.max(n_effs))
        spread = (hi - lo) / hi if hi > 0 else 0.0
        if spread <= tol:
            return                      # provably comparable; settings may differ
        # units is identical across outputs here -- a mismatch already raised.
        if attrs[0].get('units') == 'r':
            why = ("these r values are backed by different amounts of "
                   "independent evidence, so an unweighted mean over-weights "
                   "the least reliable of them")
        else:
            why = ("these estimates sit on different bias floors (~1/n_eff), "
                   "so their mean is not a clean estimate of a common coupling")
        msg = (f"effective DOF spans {lo:.1f}-{hi:.1f} ({spread:.0%} spread, "
               f"tol={tol:.0%}) -- {why}")
    else:
        differing = sorted({
            k for k in set().union(*(a.keys() for a in attrs))
            if k not in _NON_ESTIMATOR_ATTRS
            and len({repr(a.get(k)) for a in attrs}) > 1
        })
        if not differing:
            return
        msg = (f"outputs disagree on estimator settings {differing}, and none "
               f"records 'n_eff', so comparability cannot be verified")

    if check == 'raise':
        raise ValueError(f"FCOutput.average: {msg}.")
    logging.getLogger('milob').warning(f"FCOutput.average: {msg}.")


class FCOutput(BaseOutput):
    """
    Functional connectivity between channels or regions.

    Holds a square matrix of connectivity values in ``value``, indexed by
    ``channel_i`` and ``channel_j``, for each chromophore. Methods that
    resolve connectivity in frequency or time add further dimensions --
    ``freq`` for coherence, ``freq`` and ``time`` for wavelet coherence --
    which :meth:`reduce` collapses to the static shape the thresholding,
    graph and plotting methods expect. After :meth:`roi_average` the two
    channel axes carry ROI names instead.

    The attributes record which value space the numbers live in: ``units``
    names it, and ``method``, ``range`` and ``cmap`` follow from it. These
    govern how values are pooled and displayed, so that averaging happens in
    a variance-stabilising transform where one applies, and outputs from
    incompatible methods are refused rather than silently combined.

    Parameters
    ----------
    data : xarray.Dataset
        Connectivity values in ``value``, with any per-pair diagnostics such
        as ``n_eff`` stored alongside.
    probe : Probe, optional
        Probe geometry, required for the spatial plotting methods. Left as
        None for a cross-stream result, which spans two probes.
    analysis_type : str, optional
        Label stored with the output. Default is ``'Connectivity'``.
    history : list, optional
        Processing history inherited from the source stream.
    """
    def __init__(self, data, probe=None, analysis_type="Connectivity", history=None):
        super().__init__(data=data, probe=probe, analysis_type=analysis_type, history=history)

    @classmethod
    def average(cls, outputs, agg='mean', check='warn', tol=0.1):
        """
        Average several connectivity outputs into one.

        Channel pairs that are NaN in an output do not contribute to it; pairs
        NaN in every output stay NaN.

        Parameters
        ----------
        outputs : list of FCOutput
            Outputs to combine.
        agg : {'mean', 'median'}, optional
            ``'mean'`` averages in the value space's variance-stabilising
            transform where one is defined -- arctanh(r) for Pearson,
            arctanh(sqrt(C)) for coherence -- and back-transforms afterwards.
            ``'median'`` is taken on the raw values. Default is ``'mean'``.
        check : {'warn', 'raise', 'ignore'}, optional
            How strictly to enforce poolability; see :func:`check_poolable`.
            Default is ``'warn'``.
        tol : float, optional
            Relative tolerance on the spread of ``n_eff``. Default is 0.1.

        Returns
        -------
        FCOutput
            The averaged result.
        """
        if not outputs:
            raise ValueError("outputs list is empty.")
        if agg not in ('mean', 'median'):
            raise ValueError(f"Unknown agg '{agg}'. Available: 'mean', 'median'.")
        if check not in ('warn', 'raise', 'ignore'):
            raise ValueError(
                f"Unknown check '{check}'. Available: 'warn', 'raise', 'ignore'.")

        # Before any arithmetic: pooling outputs from different methods (or
        # from the same method at materially different DOF) produces a number
        # with no interpretation, and nothing downstream can detect it.
        check_poolable(outputs, tol=tol, check=check)

        units = outputs[0].output.attrs.get('units')
        # Averaged in whichever variance-stabilising space this value space
        # declares -- arctanh(r) for Pearson, arctanh(sqrt(C)) for coherence
        # (Welch and wavelet alike), nothing for an already-transformed or
        # non-correlation space. agg='median' skips it: a median commutes
        # with any monotonic transform, so transforming would only cost
        # precision.
        ztf = resolve_z_transform(units) if agg == 'mean' else None

        stack = np.array([o.output['value'].values for o in outputs])
        if ztf:
            stack = ztf['forward'](stack)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            combined = np.nanmedian(stack, axis=0) if agg == 'median' else np.nanmean(stack, axis=0)

        if ztf:
            combined = ztf['inverse'](combined)

        ref = outputs[0].output
        data_vars = {'value': (list(ref['value'].dims), combined)}

        # Everything ALONGSIDE 'value' -- roi_average()'s n_channels, sd and
        # mean_<metric> diagnostics -- used to be dropped here, so averaging
        # occurrences (which run_fc(segment_events=True) always does) silently
        # returned an output that had lost exactly the information needed to
        # judge it: how many channel pairs backed each ROI value, and how
        # variable they were. Averaged with a plain nanmean, since they are
        # already summary statistics rather than r-values -- no Fisher-Z step
        # applies to a count or a spread.
        for name in ref.data_vars:
            if name == 'value':
                continue
            if not all(name in o.output.data_vars for o in outputs):
                continue
            side = np.array([o.output[name].values for o in outputs], dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                data_vars[name] = (list(ref[name].dims), np.nanmean(side, axis=0))

        ds = xr.Dataset(data_vars, coords=ref.coords, attrs=ref.attrs)
        # Carried over, not hardcoded: an average of ROI-level outputs is
        # still ROI-level, and to_dataframe() keys its column naming off this.
        return cls(data=ds, probe=outputs[0].probe,
                   analysis_type=outputs[0].analysis_type,
                   history=cls._consolidate_histories(outputs, 'FCOutput.average'))


    def distance(self, other, method='geodesic', **method_kwargs):
        """
        Measure the distance between this connectivity matrix and another.

        Used to compare two matrices, for instance in subject identification.
        The geodesic metric is defined on symmetric positive-definite
        matrices, so it accepts a full correlation matrix over a session's
        channels but not an inter-brain cross-block, which is not symmetric.

        Parameters
        ----------
        other : FCOutput
            Output to compare against. Must carry the same channel labels in
            the same order and the same chromophore axis as this one, and
            neither may retain an extra axis such as ``metric``.
        method : str, optional
            Name of a distance in the connectivity distance registry. Default
            is ``'geodesic'``, the affine-invariant Riemannian distance,
            validated for Pearson correlation matrices.
        **method_kwargs
            Passed to the registered distance function; ``'geodesic'`` accepts
            ``alpha``, the shrinkage toward the identity.

        Returns
        -------
        dict
            Distance per chromophore or wavelength.

        Raises
        ------
        ValueError
            If the two outputs are not comparable, or the matrix is not
            symmetric.

        References
        ----------
        .. [1] Novi, S.L. et al. (2023). Fast acquisition of resting-state
               functional connectivity with functional near-infrared
               spectroscopy. Neurophotonics, 10(1), 013510.

        Examples
        --------
        Leave-one-out subject identification::

            candidates = {'sub-01': fc_train_01, 'sub-02': fc_train_02}
            distances = {sid: fc_test.distance(fc)['HbT']
                         for sid, fc in candidates.items()}
            identified = min(distances, key=distances.get)
        """
        from ..analysis.connectivity import _FC_DISTANCE_METHODS

        if method not in _FC_DISTANCE_METHODS:
            raise ValueError(f"Unknown FC distance method '{method}'. Available: {sorted(_FC_DISTANCE_METHODS)}")

        _require_reduced(self.output['value'], 'distance')
        _require_reduced(other.output['value'], 'distance')

        # _require_reduced only guarantees dims are drawn from
        # {channel_i, channel_j, chromophore, wavelength, metric} -- it
        # does not guarantee a chromophore axis specifically (a wavelength-
        # space FCOutput, from a stream not yet run through mbll(), has
        # 'wavelength' instead -- see connectivity.py's own type_dim
        # convention) or that 'metric' is absent. Resolve which
        # axis EACH output actually uses -- neither may be present at all
        # (e.g. an output already collapsed across chromophore/wavelength
        # some other way), which must fail clearly here rather than
        # silently falling back to 'wavelength' and hitting a confusing
        # KeyError at .coords[type_dim] below.
        def _resolve_type_dim(fc, arg_name):
            if 'chromophore' in fc.output.dims:
                return 'chromophore'
            if 'wavelength' in fc.output.dims:
                return 'wavelength'
            raise ValueError(
                f"distance() requires {arg_name} to have a 'chromophore' or 'wavelength' "
                f"dimension -- got dims {list(fc.output['value'].dims)}."
            )

        type_dim = _resolve_type_dim(self, 'this output')
        other_type_dim = _resolve_type_dim(other, '`other`')
        if type_dim != other_type_dim:
            raise ValueError(
                f"distance() requires both FCOutputs to use the same chromophore/"
                f"wavelength axis -- got '{type_dim}' vs '{other_type_dim}'."
            )

        # A matrix-level comparison needs exactly one scalar per channel
        # pair per type_dim value -- any dim beyond that (e.g. a
        # static 'metric' axis, which _require_reduced deliberately does
        # NOT collapse, see its own docstring) makes .sel(chromophore=...)
        # return a non-2D slice, which would otherwise fail deep inside
        # np.fill_diagonal with a confusing "dimensions must be of equal
        # length" error instead of this explicit one. Checked on BOTH
        # operands -- an extra dim on `other` alone would otherwise pass
        # this check and only surface later as a shape mismatch.
        extra = [d for d in self.output['value'].dims if d not in ('channel_i', 'channel_j', type_dim)]
        other_extra = [d for d in other.output['value'].dims if d not in ('channel_i', 'channel_j', type_dim)]
        if extra or other_extra:
            bad = extra or other_extra
            which = 'this output' if extra else '`other`'
            raise ValueError(
                f"distance() requires a single value per channel pair per {type_dim} -- "
                f"{which} still has {bad} dimension(s) (e.g. a 'metric' axis). "
                f"Select one first, e.g. .sel({bad[0]}=<value>), before calling distance()."
            )

        if list(self.output.channel_i.values) != list(other.output.channel_i.values) or \
           list(self.output.channel_j.values) != list(other.output.channel_j.values):
            raise ValueError(
                "distance() requires both FCOutputs to have identical channel_i/channel_j "
                "labels in the same order -- got different channel sets/orderings. Align "
                "them first (e.g. select/reindex to a common channel set) before comparing."
            )

        compute_fn = _FC_DISTANCE_METHODS[method]
        result = {}
        for val in self.output.coords[type_dim].values:
            m1 = self.output['value'].sel({type_dim: val}).values.copy()
            m2 = other.output['value'].sel({type_dim: val}).values.copy()

            # Self-loops (diagonal) are NaN by convention (FC.fit()) -- but
            # a correlation matrix's true diagonal is always 1, needed for
            # the regularization inside compute_fn, so restore it here
            # rather than leaving NaN.
            #
            # Only NaN diagonal entries are filled, never finite ones. On a
            # matrix whose diagonal carries real data -- an inter-brain
            # cross-block, where entry (i, i) is a genuine A_i-B_i
            # correlation and not a self-loop -- an unconditional
            # fill_diagonal(1.0) would silently overwrite it, and since the
            # result has no NaNs left it would sail past the check below and
            # reach the metric as corrupted data. Left intact, such a matrix
            # is caught by the symmetry guard inside the metric instead.
            for m in (m1, m2):
                nan_diag = np.isnan(np.diagonal(m))
                if nan_diag.any():
                    idx = np.flatnonzero(nan_diag)
                    m[idx, idx] = 1.0

            if np.isnan(m1).any() or np.isnan(m2).any():
                raise ValueError(
                    f"distance() requires a complete matrix with no bad/excluded channels "
                    f"({type_dim} '{val}' has NaN off-diagonal entries) -- unlike "
                    f"averaging methods, a matrix-level comparison can't skip individual "
                    f"missing pairs."
                )

            result[str(val)] = compute_fn(m1, m2, **method_kwargs)

        return result


    def to_matrix(self, chromophore: str = 'HbT', head: bool = False, n: int = 5, exclude_bad: bool = True):
        """
        Return the connectivity values as a wide channel-by-channel table.

        Parameters
        ----------
        chromophore : str, optional
            Chromophore to extract. Default is ``'HbT'``.
        head : bool, optional
            If True, return only the first ``n`` rows and columns. Default is
            False.
        n : int, optional
            Number of rows and columns kept when ``head`` is True. Default is 5.
        exclude_bad : bool, optional
            Drop channels that are entirely NaN. Default is True.

        Returns
        -------
        pandas.DataFrame
            Square matrix indexed by channel label.
        """
        import pandas as pd

        _require_reduced(self.output['value'], 'to_matrix')
        matrix_slice = self.output.value.sel(chromophore=chromophore)

        if exclude_bad:
            matrix_slice = matrix_slice.dropna(dim='channel_i', how='all')
            matrix_slice = matrix_slice.dropna(dim='channel_j', how='all')

        channels = matrix_slice.channel_i.values

        if head:
            matrix_slice = matrix_slice.isel(
                channel_i=slice(0, n),
                channel_j=slice(0, n)
            )
            channels = channels[:n]

        return pd.DataFrame(data=matrix_slice.values, index=channels, columns=channels)


    def to_dataframe(self, upper_triangle='auto', value_name=None) -> "pd.DataFrame":
        """
        Return the connectivity values as a long-form table.

        One row per channel pair and chromophore, or per ROI pair for a
        ROI-averaged output, with any ``metric`` axis carried through. The
        value column is named after the output's units rather than a generic
        name, and a Fisher-z column is added where that transform is valid.
        Support columns produced by :meth:`roi_average`, such as
        ``n_channels`` and the ``mean_<metric>`` quality diagnostics, are
        included when present. Subject-level metadata is added by
        :meth:`milob.Study.to_dataframe`, not here.

        Parameters
        ----------
        upper_triangle : bool or 'auto', optional
            Whether to keep only the upper triangle. Every method in this
            release is symmetric, so ``'auto'`` keeps it. Default is ``'auto'``.
        value_name : str, optional
            Override the name of the value column, useful when concatenating
            tables produced by different methods.

        Returns
        -------
        pandas.DataFrame
            The long-form table. The diagonal is dropped for channel-level
            output, where it is a self-loop, and kept for ROI-level output,
            where it is each ROI's mean internal connectivity.
        """
        import pandas as pd
        import numpy as np

        _require_reduced(self.output['value'], 'to_dataframe')
        units = self.output.attrs.get('units')
        ztf = resolve_z_transform(units)
        include_z = ztf is not None

        # ROI-level outputs put ROI names on channel_i/channel_j (roi_average
        # collapses the channel axes in place rather than renaming the dims).
        # The column names follow what the labels ACTUALLY are, so a table of
        # ROI-averaged FC never claims to be channel-level.
        is_roi = self.analysis_type == 'Connectivity_ROI'
        label_i, label_j = ('roi_i', 'roi_j') if is_roi else ('channel_i', 'channel_j')

        # Name the value column after the value space ('r', 'coherence',
        # 'null_z', ...) instead of a generic 'value', so a merged/exported
        # table stays self-describing. Pass value_name= to override.
        value_col = value_name or units or 'value'

        # Carry through whatever roi_average() attached alongside the value:
        # n_channels (how many channel pairs backed this cell -- varies per
        # row once bad channels differ per subject/ROI, which is exactly the
        # bias worth modelling on), sd, and any mean_<metric> diagnostics.
        # Everything stored alongside 'value' becomes a column, keyed by its
        # own name -- so a new support/diagnostic variable added to
        # roi_average() appears here automatically instead of needing a
        # matching entry in this method.
        sidecar_cols = {}
        for name in self.output.data_vars:
            if name == 'value':
                continue
            if name == 'sd':
                # sd lives in the space the mean was taken in -- 'sd_z' when
                # that was Fisher-z, so the column name can't be mistaken for
                # an r-scale spread.
                #
                # attrs['sd_space'] is authoritative (it records what
                # roi_average ACTUALLY did, including agg='median', which
                # skips the transform even for r-valued units). It is only
                # absent on outputs built before that attr existed, so fall
                # back to the same units metadata that decides the transform
                # in the first place -- rather than silently mislabelling a
                # z-space spread as 'sd'.
                space = self.output.attrs.get('sd_space')
                if space is None:
                    space = 'z' if include_z else 'raw'
                sidecar_cols['sd'] = 'sd_z' if space == 'z' else 'sd'
                continue
            sidecar_cols[name] = str(name)

        rows = []
        channels = self.output.channel_i.values
        has_metric = 'metric' in self.output.value.dims
        metric_values = self.output.metric.values if has_metric else [None]

        for metric in metric_values:
            if upper_triangle == 'auto':
                triangle_only = True
            else:
                triangle_only = upper_triangle

            for chrom in self.output.chromophore.values:
                sel = {'chromophore': chrom}
                if has_metric:
                    sel['metric'] = metric
                matrix = self.output.value.sel(sel)
                for i, ch_i in enumerate(channels):
                    # Diagonal INCLUDED (j starts at i, not i+1). For a
                    # channel-level output it is a self-loop, already NaN,
                    # and dropped by the isnan check below -- so this only
                    # adds rows where the diagonal means something: at ROI
                    # level it is that ROI's mean INTERNAL connectivity
                    # (between its own distinct channels), a real quantity
                    # that was previously computed and then discarded here.
                    j_start = i if triangle_only else 0
                    for j in range(j_start, len(channels)):
                        ch_j = channels[j]
                        val = float(matrix.sel(channel_i=ch_i, channel_j=ch_j).values)
                        if np.isnan(val):
                            continue
                        row = {
                            label_i:      str(ch_i),
                            label_j:      str(ch_j),
                            'chromophore': str(chrom),
                            value_col:    val,
                        }
                        if has_metric:
                            row['metric'] = str(metric)
                        if ztf:
                            row[ztf['column']] = float(ztf['forward'](val))
                        for src, dest in sidecar_cols.items():
                            side = self.output[src]
                            s_sel = {k: v for k, v in sel.items() if k in side.dims}
                            row[dest] = float(
                                side.sel(channel_i=ch_i, channel_j=ch_j, **s_sel).values
                            )
                        rows.append(row)

        df = pd.DataFrame(rows)

        # Channel-level output: annotate each channel with the ROI it belongs
        # to, if the probe defines any. Skipped for ROI-level output, where
        # the label columns ARE the ROIs -- mapping ROI names through a
        # channel->ROI lookup is what produced the all-NaN roi_i/roi_j
        # columns this used to emit.
        if not is_roi and self.probe is not None and self.probe.rois:
            roi_map = self.probe.channel_roi_map
            df.insert(df.columns.get_loc(label_i) + 1, 'roi_i', df[label_i].map(roi_map))
            df.insert(df.columns.get_loc(label_j) + 1, 'roi_j', df[label_j].map(roi_map))

        return df


    def reduce(self, freq=None, time=None, agg='mean') -> "FCOutput":
        """
        Collapse frequency- and time-resolved dimensions to a static matrix.

        Methods such as coherence add a ``freq`` dimension, and wavelet
        coherence adds ``freq`` and ``time``. Most downstream methods --
        :meth:`threshold`, :meth:`to_graphs`, :meth:`to_matrix` and the
        plotting methods -- require the static shape this produces. The step
        is deliberate rather than automatic, so the band and window can be
        chosen, and revisited, without refitting.

        Parameters
        ----------
        freq : tuple of float, optional
            Frequency band ``(low, high)`` in Hz to select before aggregating.
            The full spectrum is used if omitted.
        time : tuple of float, optional
            Time window ``(t0, t1)`` in seconds to select before aggregating,
            for methods with a ``time`` dimension. The full recording is used
            if omitted.
        agg : {'mean', 'median', 'max'}, optional
            How to aggregate within the selection. Default is ``'mean'``.

        Returns
        -------
        FCOutput
            New output with the extra dimensions collapsed.
        """
        import numpy as np
        import xarray as xr

        da = self.output['value']
        extra_dims = [d for d in da.dims if d not in _STATIC_FC_DIMS]

        if not extra_dims:
            raise ValueError(
                "This FCOutput has no frequency/time dimension to reduce -- "
                "reduce() is only meaningful for frequency-/time-resolved "
                "methods (e.g. coherence, wavelet coherence)."
            )
        if agg not in ('mean', 'median', 'max'):
            raise ValueError(f"Unknown agg '{agg}'. Available: 'mean', 'median', 'max'.")

        if freq is not None:
            if 'freq' not in da.dims:
                raise ValueError("freq= was given but this output has no 'freq' dimension.")
            da = da.sel(freq=slice(freq[0], freq[1]))
        if time is not None:
            if 'time' not in da.dims:
                raise ValueError("time= was given but this output has no 'time' dimension.")
            da = da.sel(time=slice(time[0], time[1]))

        agg_fn = {'mean': np.nanmean, 'median': np.nanmedian, 'max': np.nanmax}[agg]
        reduced = da.reduce(agg_fn, dim=extra_dims)

        new_ds = xr.Dataset({'value': reduced}, coords=reduced.coords, attrs=self.output.attrs)
        history = self.history + [self._history_entry('reduce', {
            'freq': freq, 'time': time, 'agg': agg, 'reduced_dims': extra_dims,
        })]
        return FCOutput(new_ds, probe=self.probe, analysis_type=self.analysis_type, history=history)


    def roi_average(self, rois, agg='mean', weights=None, weight_steepness=1.0) -> "FCOutput":
        """
        Average connectivity within and between regions of interest.

        Each ROI-to-ROI block combines every channel pair spanning the two
        regions, skipping NaN pairs. Dimensions beyond the two channel axes,
        such as ``freq``, ``time`` or ``metric``, are preserved, so this can
        be applied before or after :meth:`reduce` with the same result.

        Parameters
        ----------
        rois : Probe or dict
            A probe carrying ROIs defined with ``add_roi()``, or a mapping of
            ROI name to a list of channel labels.
        agg : {'mean', 'median'}, optional
            ``'mean'`` averages in the Fisher-z transform and back-transforms
            where the units support it, and plainly otherwise; it is the only
            option that accepts ``weights``. ``'median'`` is taken on the raw
            values. Default is ``'mean'``.
        weights : str or dict, optional
            Per-channel weights for the mean. A string names a quality
            coordinate already attached to the output, ``'sci'`` or ``'snr'``,
            which is converted to a weight against the quality distribution of
            the whole probe. A dict of ``{channel: weight}`` is used as given.
            Each pair's weight is the product of its two channel weights.
            Unweighted if omitted.
        weight_steepness : float, optional
            Sigmoid steepness of the quality-to-weight transform, used only
            when ``weights`` names a coordinate. Higher values sharpen the
            distinction between good and poor channels. Default is 1.0.

        Returns
        -------
        FCOutput
            New output indexed by ROI, whose diagonal is each region's mean
            internal connectivity. Adds ``n_channels``, the number of channel
            pairs contributing to each block, and ``mean_<metric>``, the
            unweighted mean of each available quality coordinate over the
            contributing channels.
        """
        import numpy as np
        import xarray as xr

        if agg not in ('mean', 'median'):
            raise ValueError(f"Unknown agg '{agg}'. Available: 'mean', 'median'.")
        if agg == 'median' and weights is not None:
            raise ValueError("weights is not supported with agg='median'.")

        if isinstance(rois, dict):
            roi_map   = rois
            out_probe = self.probe
        else:
            roi_map   = rois.rois
            out_probe = rois

        if not roi_map:
            raise ValueError(
                "No ROIs provided. Pass a dict {name: [channels]} or a Probe "
                "with ROIs defined via probe.add_roi()."
            )

        units = self.output.attrs.get('units')
        # See FCOutput.average() for why this is keyed to the value space
        # rather than to the FC method, and why agg='median' opts out.
        ztf = resolve_z_transform(units) if agg == 'mean' else None

        roi_names = list(roi_map.keys())
        out_ch_i  = set(self.output.channel_i.values)
        out_ch_j  = set(self.output.channel_j.values)
        n_roi     = len(roi_names)

        # Every dim beyond channel_i/channel_j (chromophore alone, or
        # chromophore+freq[+time], or chromophore+metric) is
        # carried through untouched -- only the channel axes are ever
        # collapsed here. Replaces the old hardcoded 'chromophore'-only
        # handling; see class-level history entry for the roi_average
        # generalization.
        extra_dims   = list(self.output.value.dims[2:])
        extra_coords = {d: self.output.coords[d].values for d in extra_dims}

        raw = self.output.value.values
        # NaN-safe: every registered forward transform maps NaN -> NaN
        data_for_agg = ztf['forward'](raw) if ztf else raw

        ci_idx = {ch: i for i, ch in enumerate(self.output.channel_i.values)}
        cj_idx = {ch: j for j, ch in enumerate(self.output.channel_j.values)}

        # A channel is "valid" (contributes to the probe-wide reference
        # distribution below, and to the mean_<metric> diagnostics) if it
        # has at least one non-NaN entry anywhere across chromophore/freq/
        # time/metric -- same idiom as to_matrix()/to_graphs()' bad-channel
        # exclusion, generalized from the old hardcoded axis=(1,2)/(0,2)
        # (chromophore-only) to however many trailing dims this output has.
        valid_i_mask = ~np.all(np.isnan(raw), axis=tuple(range(1, raw.ndim)))
        valid_j_mask = ~np.all(np.isnan(raw), axis=tuple([0] + list(range(2, raw.ndim))))

        w_i_lookup = w_j_lookup = None
        if weights is not None:
            if isinstance(weights, str):
                if f'{weights}_i' not in self.output.coords or f'{weights}_j' not in self.output.coords:
                    raise ValueError(
                        f"No '{weights}' quality coordinate found on this FCOutput. "
                        f"Run the corresponding quality screen (e.g. sci_screen()/snr_screen()) "
                        f"before FC.fit() so it gets propagated, or pass an explicit weights dict."
                    )
                raw_i = self.output.coords[f'{weights}_i'].values
                raw_j = self.output.coords[f'{weights}_j'].values
                # Reference populations are computed and normalized SEPARATELY
                # per side: in cross-stream FC, channel_i/channel_j can be
                # different participants/devices, and pooling their quality
                # distributions together would reintroduce the cross-
                # population comparison this scheme is designed to avoid.
                w_i_vals = _self_referential_quality_weight(raw_i, raw_i[valid_i_mask], weight_steepness)
                w_j_vals = _self_referential_quality_weight(raw_j, raw_j[valid_j_mask], weight_steepness)
                w_i_lookup = dict(zip(self.output.channel_i.values, w_i_vals))
                w_j_lookup = dict(zip(self.output.channel_j.values, w_j_vals))
            elif isinstance(weights, dict):
                w_i_lookup = weights
                w_j_lookup = weights
            else:
                raise ValueError("weights must be None, a coordinate name string, or a dict {channel: weight}.")

        # mean_<metric> diagnostics: computed for every propagated quality
        # coordinate regardless of whether it's the one used for `weights`
        # (or whether weights was used at all) -- see docstring.
        quality_metrics = {}
        for qname in CHANNEL_QUALITY_COORDS:
            if f'{qname}_i' in self.output.coords and f'{qname}_j' in self.output.coords:
                quality_metrics[qname] = (
                    dict(zip(self.output.channel_i.values, self.output.coords[f'{qname}_i'].values)),
                    dict(zip(self.output.channel_j.values, self.output.coords[f'{qname}_j'].values)),
                )

        avg      = np.full((n_roi, n_roi) + raw.shape[2:], np.nan)
        sd       = np.full((n_roi, n_roi) + raw.shape[2:], np.nan)
        n_pairs_mat = np.zeros((n_roi, n_roi), dtype=int)
        n_ch_i_mat  = np.zeros((n_roi, n_roi), dtype=int)
        n_ch_j_mat  = np.zeros((n_roi, n_roi), dtype=int)
        quality_i = {qname: np.full((n_roi, n_roi), np.nan) for qname in quality_metrics}
        quality_j = {qname: np.full((n_roi, n_roi), np.nan) for qname in quality_metrics}

        # (roi_a, roi_b) blocks that ended up with no valid pair at all --
        # collected here and reported once after the loop rather than
        # per-block, so a probe with several dead ROIs produces one legible
        # message instead of a wall of numpy RuntimeWarnings.
        empty_blocks = []

        for i, roi_a in enumerate(roi_names):
            chs_a = [ch for ch in roi_map[roi_a] if ch in out_ch_i]

            for j, roi_b in enumerate(roi_names):
                chs_b = [ch for ch in roi_map[roi_b] if ch in out_ch_j]

                if not chs_a or not chs_b:
                    continue

                rows  = [ci_idx[ch] for ch in chs_a]
                cols  = [cj_idx[ch] for ch in chs_b]
                # np.ix_ on just the 2 channel axes -- numpy's indexing
                # rules leave any further (chromophore/freq/time/metric)
                # axes as implicit full slices, so this works unchanged
                # whether data_for_agg is 3D or 5D.
                block = data_for_agg[np.ix_(rows, cols)]

                # A block with no surviving pair at all: every reduction
                # below (nanmean/nanmedian/nanstd, weighted or not) would
                # emit numpy's "Mean of empty slice"/"Degrees of freedom
                # <= 0" RuntimeWarning and return exactly the NaN that avg
                # and sd already hold from initialization -- so skip it.
                # n_pairs/n_channels stay 0 and the quality means stay NaN,
                # which is the same state the full path produces.
                #
                # Two very different situations land here, so they are
                # reported differently below: a one-channel ROI's own
                # diagonal block is empty *by construction* (its only cell
                # is the self-pair, already NaN) and is not worth a word,
                # whereas any other empty block means every channel pair
                # between those ROIs was screened out -- a real coverage
                # loss the caller should see.
                if np.all(np.isnan(block)):
                    if not (i == j and len(chs_a) == 1):
                        empty_blocks.append((roi_a, roi_b))
                    continue

                # `spread` is deliberately computed in the SAME space the
                # average was taken in (Fisher-z for r), and is NOT
                # back-transformed: tanh is nonlinear, so tanh(sd) is not
                # the sd of the back-transformed values and would not
                # support the interval arithmetic an sd is used for.
                # attrs['sd_space'] records which space that was.
                if agg == 'median':
                    reduced = np.nanmedian(block, axis=(0, 1))
                    spread = np.nanstd(block, axis=(0, 1))
                elif weights is None:
                    reduced = np.nanmean(block, axis=(0, 1))
                    spread = np.nanstd(block, axis=(0, 1))
                else:
                    w_a = np.array([w_i_lookup[ch] for ch in chs_a], dtype=float)
                    w_b = np.array([w_j_lookup[ch] for ch in chs_b], dtype=float)
                    pair_w = np.outer(w_a, w_b)
                    reduced = _weighted_nanmean(block, pair_w)
                    spread = _weighted_nanstd(block, pair_w, reduced)

                avg[i, j, ...] = ztf['inverse'](reduced) if ztf else reduced
                sd[i, j, ...] = spread
                # "Contributed at all, anywhere" (any non-NaN across every
                # trailing dim) rather than the old single-chromophore-slice
                # proxy -- same result in practice wherever bad-channel NaNs
                # are chromophore-independent (the normal case), but no
                # longer relies on that assumption, and stays meaningful
                # when freq/time/metric slices differ (e.g. COI).
                valid_cells = np.any(~np.isnan(block), axis=tuple(range(2, block.ndim)))

                if i == j:
                    # Within-ROI block: roi_a and roi_b are the SAME channel
                    # set, so (a,b) and (b,a) are one channel pair counted
                    # twice (self-loops are already NaN). Count the upper
                    # triangle only, so a diagonal cell's support is directly
                    # comparable to an off-diagonal one.
                    n_pairs_mat[i, j] = int(np.sum(np.triu(valid_cells, k=1)))
                else:
                    # Between-ROI block: the two channel sets are disjoint, so
                    # every cell is a distinct pair, counted once. The mirrored
                    # roi_b x roi_a block holds the same pairs transposed --
                    # that duplication is ACROSS blocks, not within one, so
                    # this is not halved (and to_dataframe emits one triangle).
                    n_pairs_mat[i, j] = int(np.sum(valid_cells))

                # How many channels each SIDE actually contributed. More
                # diagnostic than the pair count alone: it says which region
                # lost coverage, which the product cannot -- 1640 pairs could
                # be 41x40 or 82x20.
                contrib_a = [ch for k, ch in enumerate(chs_a) if valid_cells[k].any()]
                contrib_b = [ch for k, ch in enumerate(chs_b) if valid_cells[:, k].any()]
                n_ch_i_mat[i, j] = len(contrib_a)
                n_ch_j_mat[i, j] = len(contrib_b)

                # Reported per side rather than pooled: channel_i and
                # channel_j can be different streams (cross-stream FC) or
                # simply different regions, and one pooled mean hides a
                # region that is individually poor -- the same reasoning that
                # makes `weights` normalize each side separately.
                for qname, (q_i_lookup, q_j_lookup) in quality_metrics.items():
                    if contrib_a:
                        quality_i[qname][i, j] = np.nanmean([q_i_lookup[ch] for ch in contrib_a])
                    if contrib_b:
                        quality_j[qname][i, j] = np.nanmean([q_j_lookup[ch] for ch in contrib_b])

        if empty_blocks:
            shown = ", ".join(f"{a}-{b}" for a, b in empty_blocks[:6])
            more = f" (+{len(empty_blocks) - 6} more)" if len(empty_blocks) > 6 else ""
            logging.getLogger('milob').warning(
                f"roi_average: {len(empty_blocks)} ROI block(s) have no valid channel "
                f"pair and are NaN in the result: {shown}{more}. Every channel pair "
                f"between those ROIs was excluded (bad channels / quality screening) "
                f"-- check n_pairs on the returned output."
            )

        # Sidecars from the channel-level fit are deliberately NOT carried
        # through: an ROI value averages many channel pairs that are not
        # independent of each other, so a mean of their per-pair 'n_eff'
        # would read as ROI-level degrees of freedom while overstating them
        # badly. n_pairs/n_channels_* below say what actually backs the cell;
        # the attrs 'n_eff' summary survives for check_poolable.
        data_vars = {
            'value':        (['channel_i', 'channel_j'] + extra_dims, avg),
            'sd':           (['channel_i', 'channel_j'] + extra_dims, sd),
            'n_pairs':      (['channel_i', 'channel_j'], n_pairs_mat),
            'n_channels_i': (['channel_i', 'channel_j'], n_ch_i_mat),
            'n_channels_j': (['channel_i', 'channel_j'], n_ch_j_mat),
        }
        for qname in quality_metrics:
            data_vars[f'mean_{qname}_i'] = (['channel_i', 'channel_j'], quality_i[qname])
            data_vars[f'mean_{qname}_j'] = (['channel_i', 'channel_j'], quality_j[qname])

        ds = xr.Dataset(
            data_vars,
            coords={
                'channel_i': roi_names,
                'channel_j': roi_names,
                **extra_coords,
            },
            attrs={**self.output.attrs,
                   'sd_space': 'z' if ztf else 'raw'},
        )
        history = self.history + [self._history_entry('roi_average', {
            'rois': list(roi_map.keys()), 'agg': agg,
            'weights': weights if not isinstance(weights, dict) else 'custom_dict',
            'weight_steepness': weight_steepness,
            # Recorded so a stored result says which space it was averaged
            # in -- raw and z-space means differ, and nothing else on the
            # output distinguishes them.
            'z_transform': FC_UNITS_META.get(units, {}).get('z_transform') if ztf else None,
        })]
        return FCOutput(ds, probe=out_probe, analysis_type='Connectivity_ROI', history=history)

    def handle_negatives(self, mode='discard'):
        """
        Re-interpret negative connectivity values.

        Meaningful only for signed metrics such as Pearson correlation.

        Parameters
        ----------
        mode : {'discard', 'absolute', 'keep'}, optional
            ``'discard'`` sets negative values to zero, ``'absolute'`` takes
            their magnitude, and ``'keep'`` leaves them unchanged while
            recording the decision in the history. Default is ``'discard'``.

        Returns
        -------
        FCOutput
            New output with the chosen treatment applied.

        Raises
        ------
        ValueError
            If the metric has no negative values to re-interpret.
        """
        units = self.output.attrs.get('units')
        meta = FC_UNITS_META.get(units)
        if not meta or not meta.get('signed', False):
            raise ValueError(
                f"handle_negatives is not meaningful for units='{units}' "
                f"-- there are no negative values to reinterpret."
            )
        if mode not in ('discard', 'absolute', 'keep'):
            raise ValueError(f"Unknown mode '{mode}'. Available: 'discard', 'absolute', 'keep'.")

        matrix = self.output['value'].values.copy()
        if mode == 'discard':
            matrix[matrix < 0] = 0
        elif mode == 'absolute':
            matrix = np.abs(matrix)

        new_attrs = dict(self.output.attrs)
        base_range = meta.get('range')
        if mode in ('discard', 'absolute') and base_range is not None:
            new_attrs['range'] = (0.0, base_range[1])
            new_attrs['cmap'] = 'viridis'

        new_ds = self.output.copy()
        new_ds['value'].values = matrix
        new_ds.attrs = new_attrs

        history = self.history + [self._history_entry('FCOutput.handle_negatives', {'mode': mode})]
        return FCOutput(new_ds, probe=self.probe, analysis_type=self.analysis_type, history=history)


    def threshold(self, threshold_type='sparsity', threshold_val=0.15, binarize=True):
        """
        Threshold the connectivity matrix, typically before building a graph.

        Parameters
        ----------
        threshold_type : {'sparsity', 'absolute'}, optional
            ``'sparsity'`` keeps the strongest ``threshold_val`` fraction of
            values; ``'absolute'`` keeps values at or above ``threshold_val``.
            Default is ``'sparsity'``.
        threshold_val : float, optional
            Fraction or value to threshold at. Default is 0.15.
        binarize : bool, optional
            If True, surviving values become 1, giving an unweighted graph; if
            False they keep their magnitude. Values below the threshold are set
            to zero either way. Default is True.

        Returns
        -------
        FCOutput
            New thresholded output.
        """
        _require_reduced(self.output['value'], 'threshold')
        matrix = self.output['value'].values.copy()
        processed = _apply_threshold_logic(matrix, threshold_type, threshold_val, binarize)

        new_ds = self.output.copy()
        new_ds['value'].values = processed

        history = self.history + [self._history_entry('FCOutput.threshold', {
            'threshold_type': threshold_type, 'threshold_val': threshold_val, 'binarize': binarize,
        })]
        return FCOutput(new_ds, probe=self.probe, analysis_type="Thresholded_Connectivity", history=history)


    def to_graphs(self, remove_bad_channels=True):
        """
        Build a graph representation of the connectivity matrix.

        Thresholding is not applied here; call :meth:`threshold` first if it
        is wanted.

        Parameters
        ----------
        remove_bad_channels : bool, optional
            Drop channels that are entirely NaN before building the graphs.
            Default is True.

        Returns
        -------
        GraphOutput
            One graph per chromophore, ready for network analysis.
        """
        import numpy as np
        import networkx as nx

        _require_reduced(self.output['value'], 'to_graphs')
        graphs = {}
        data = self.output.value
        
        for chrom in data.chromophore.values:
            # Extract matrix for each chromophore
            matrix = data.sel(chromophore=chrom).values.copy()
            
            # Identify bad channels (entire row is NaN)
            is_valid = ~np.all(np.isnan(matrix), axis=1)
            valid_indices = np.where(is_valid)[0]
            
            # Extract only the valid submatrix
            clean_matrix = matrix[np.ix_(valid_indices, valid_indices)]
            
            # Replace NaNs with 0 so they become isolated nodes
            adj = np.nan_to_num(clean_matrix, nan=0.0)        
            np.fill_diagonal(adj, 0)    # Remove self-loops (diagonal must be 0 for NetworkX)
            
            # Build Graph 
            # If the matrix was binarized in FC.fit, this creates an unweighted graph
            # If not, it creates a weighted graph with a 'weight' attribute
            G = nx.from_numpy_array(adj)
            
            # Map node integers back to channel labels
            all_channel_names = data.channel_i.values
            mapping = {i: all_channel_names[idx] for i, idx in enumerate(valid_indices)}
            nx.relabel_nodes(G, mapping, copy=False)

            # Store Full Probe in the graph metadata
            G.graph['all_channels'] = list(all_channel_names)
            graphs[str(chrom)] = G

        # Return the Output object WITHOUT metrics computed yet
        from .output_graph import GraphOutput
        return GraphOutput(graphs=graphs, probe=self.probe, source_conn=self, analysis_type='Network_Metrics')


    
    # ****************************
    #  VISUALIZATION TOOLS
    # ****************************

    def view_spectrum(self, chromophore=None):
        """
        Open an interactive viewer for frequency-resolved connectivity.

        Steps through channel pairs, shows the spectrum averaged across pairs,
        and zooms into a frequency range, which is useful before choosing a
        band for :meth:`reduce`. Requires a Jupyter frontend.

        Parameters
        ----------
        chromophore : str, optional
            Chromophore shown initially. Defaults to the first available.
        """
        from ipywidgets import widgets
        import matplotlib.pyplot as plt
        import numpy as np
        from ..viz.interactive import build_widget_browser

        if 'freq' not in self.output['value'].dims:
            raise ValueError(
                "view_spectrum() is only meaningful for frequency-resolved "
                "methods (e.g. coherence) -- this output has no 'freq' dimension."
            )
        if 'time' in self.output['value'].dims:
            raise ValueError(
                "view_spectrum() plots value-vs-frequency and can't represent a "
                "second (time) dimension -- this output is time-AND-frequency "
                "resolved (e.g. wavelet coherence). Use view_scalogram() instead "
                "for a 2-D time-frequency heatmap, or .reduce(time=(t0, t1))/"
                ".reduce(freq=(low, high)) to collapse to a single window/band "
                "first if you want the 1-D view here."
            )

        freqs = self.output['freq'].values
        channels = list(self.output.channel_i.values)
        chrom_options = list(self.output.chromophore.values)
        default_chrom = chromophore or chrom_options[0]

        def _on_mode_change(change, widgets_map):
            is_single = change['new'] == 'Single pair'
            widgets_map['channel_i'].disabled = not is_single
            widgets_map['channel_j'].disabled = not is_single
            widgets_map['agg'].disabled = is_single

        def update_plot(mode, channel_i, channel_j, chromophore, agg, freq_range):
            da = self.output['value']
            if mode == 'Average across pairs':
                agg_fn = np.nanmean if agg == 'mean' else np.nanmedian
                slice_ = da.sel(chromophore=chromophore).values  # (chan_i, chan_j, freq); self-loops already NaN
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # some freq bins may be all-NaN across pairs
                    spectrum = agg_fn(slice_, axis=(0, 1))
                label = f"{agg} across all valid pairs"
            else:
                spectrum = da.sel(channel_i=channel_i, channel_j=channel_j, chromophore=chromophore).values
                label = f"{channel_i} – {channel_j}"

            from ..viz import theme
            plt.close('all')
            fig, ax = plt.subplots(figsize=theme.FIGSIZE['wide'])
            ax.plot(freqs, spectrum)
            ax.set_xlim(freq_range[0], freq_range[1])

            # Rescale y to only the visible frequency range, not the full
            # spectrum -- otherwise zooming in on the x-axis leaves a flat,
            # unreadable line at whatever scale the full spectrum has.
            visible = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            y_visible = spectrum[visible]
            y_visible = y_visible[~np.isnan(y_visible)]
            if len(y_visible):
                pad = 0.05 * (y_visible.max() - y_visible.min() or 1.0)
                ax.set_ylim(y_visible.min() - pad, y_visible.max() + pad)

            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel(self.output.attrs.get('units', 'value'))
            method_label = self.output.attrs.get('method', '').capitalize()
            ax.set_title(f"{method_label} spectrum: {label} ({chromophore})")
            theme.style_quantitative_axes(ax)
            plt.show()

        built = build_widget_browser(
            widget_specs=[
                {'name': 'mode', 'type': 'toggle_buttons',
                 'options': ['Average across pairs', 'Single pair'], 'description': 'Mode'},
                {'name': 'channel_i', 'type': 'dropdown', 'options': channels,
                 'value': channels[0], 'description': 'Channel i'},
                {'name': 'channel_j', 'type': 'dropdown', 'options': channels,
                 'value': channels[min(1, len(channels) - 1)], 'description': 'Channel j'},
                {'name': 'chromophore', 'type': 'dropdown', 'options': chrom_options,
                 'value': default_chrom, 'description': 'Chromophore'},
                {'name': 'agg', 'type': 'dropdown', 'options': ['mean', 'median'],
                 'value': 'mean', 'description': 'Aggregate'},
                {'name': 'freq_range', 'type': 'range_slider',
                 'min': float(freqs.min()), 'max': float(freqs.max()),
                 'value': [float(freqs.min()), float(freqs.max())],
                 'step': float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.01,
                 'description': 'Freq range (Hz)', 'continuous_update': False,
                 'layout': widgets.Layout(width='400px'), 'style': {'description_width': 'initial'}},
            ],
            render_fn=update_plot,
            columns=[['mode'], ['channel_i', 'channel_j'], ['chromophore', 'agg'], ['freq_range']],
            observers=[('mode', _on_mode_change)],
        )
        # Sync the initial enabled/disabled state to match the mode toggle's
        # starting value (observers only fire on subsequent changes).
        _on_mode_change({'new': built['mode'].value}, built)


    def view_scalogram(self, chromophore=None, cmap=None):
        """
        Open an interactive time-frequency viewer for connectivity.

        Displays a scalogram of a time- and frequency-resolved output, with
        controls to zoom in time and frequency, which is useful before
        choosing a band and window for :meth:`reduce`. Where a cone of
        influence was applied at fit time, the masked cells are NaN and appear
        blank. Requires a Jupyter frontend.

        Parameters
        ----------
        chromophore : str, optional
            Chromophore shown initially. Defaults to the first available.
        cmap : str, optional
            Colormap. Defaults to the one recorded on the output.
        """
        from ipywidgets import widgets
        import matplotlib.pyplot as plt
        import numpy as np
        from ..viz import theme
        from ..viz.interactive import build_widget_browser

        da = self.output['value']
        if 'freq' not in da.dims or 'time' not in da.dims:
            raise ValueError(
                "view_scalogram() is only meaningful for time-AND-frequency-resolved "
                "methods (e.g. wavelet coherence) -- this output is missing 'freq' "
                "and/or 'time'. Use view_spectrum() for frequency-only methods (e.g. coherence)."
            )

        freqs = self.output['freq'].values
        times = self.output['time'].values
        channels = list(self.output.channel_i.values)
        chrom_options = list(self.output.chromophore.values)
        default_chrom = chromophore or chrom_options[0]
        default_cmap = cmap or self.output.attrs.get('cmap', theme.SEQUENTIAL_CMAP)
        vmin, vmax = self.output.attrs.get('range', (0.0, 1.0))

        def _on_mode_change(change, widgets_map):
            is_single = change['new'] == 'Single pair'
            widgets_map['channel_i'].disabled = not is_single
            widgets_map['channel_j'].disabled = not is_single
            widgets_map['agg'].disabled = is_single

        def update_plot(mode, channel_i, channel_j, chromophore, agg,
                         time_min, time_max, freq_min, freq_max):
            if mode == 'Average across pairs':
                agg_fn = np.nanmean if agg == 'mean' else np.nanmedian
                slice_ = da.sel(chromophore=chromophore).values  # (chan_i, chan_j, freq, time)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # some cells may be all-NaN across pairs (e.g. outside COI)
                    plane = agg_fn(slice_, axis=(0, 1))
                label = f"{agg} across all valid pairs"
            else:
                plane = da.sel(channel_i=channel_i, channel_j=channel_j, chromophore=chromophore).values
                label = f"{channel_i} – {channel_j}"

            plt.close('all')
            fig, ax = plt.subplots(figsize=theme.FIGSIZE['wide'])
            mesh = ax.pcolormesh(times, freqs, plane, cmap=default_cmap, vmin=vmin, vmax=vmax, shading='auto')
            ax.set_yscale('log')

            # Apply the requested zoom, ignoring an invalid (min >= max)
            # entry rather than erroring on it -- leaves the axis at its
            # previous (valid) extent until the user corrects the box.
            if time_min < time_max:
                ax.set_xlim(time_min, time_max)
            if freq_min < freq_max:
                ax.set_ylim(freq_min, freq_max)

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            method_label = self.output.attrs.get('method', '').capitalize()
            ax.set_title(f"{method_label} scalogram: {label} ({chromophore})")
            fig.colorbar(mesh, ax=ax, label=self.output.attrs.get('units', 'value'))
            plt.show()

        text_box_layout = widgets.Layout(width='160px')
        text_box_style = {'description_width': 'initial'}

        built = build_widget_browser(
            widget_specs=[
                {'name': 'mode', 'type': 'toggle_buttons',
                 'options': ['Average across pairs', 'Single pair'], 'description': 'Mode'},
                {'name': 'channel_i', 'type': 'dropdown', 'options': channels,
                 'value': channels[0], 'description': 'Channel i'},
                {'name': 'channel_j', 'type': 'dropdown', 'options': channels,
                 'value': channels[min(1, len(channels) - 1)], 'description': 'Channel j'},
                {'name': 'chromophore', 'type': 'dropdown', 'options': chrom_options,
                 'value': default_chrom, 'description': 'Chromophore'},
                {'name': 'agg', 'type': 'dropdown', 'options': ['mean', 'median'],
                 'value': 'mean', 'description': 'Aggregate'},
                {'name': 'time_min', 'type': 'bounded_float_text',
                 'value': float(times.min()), 'min': float(times.min()), 'max': float(times.max()),
                 'description': 'Time min (s)', 'style': text_box_style, 'layout': text_box_layout},
                {'name': 'time_max', 'type': 'bounded_float_text',
                 'value': float(times.max()), 'min': float(times.min()), 'max': float(times.max()),
                 'description': 'Time max (s)', 'style': text_box_style, 'layout': text_box_layout},
                {'name': 'freq_min', 'type': 'bounded_float_text',
                 'value': float(freqs.min()), 'min': float(freqs.min()), 'max': float(freqs.max()),
                 'description': 'Freq min (Hz)', 'style': text_box_style, 'layout': text_box_layout},
                {'name': 'freq_max', 'type': 'bounded_float_text',
                 'value': float(freqs.max()), 'min': float(freqs.min()), 'max': float(freqs.max()),
                 'description': 'Freq max (Hz)', 'style': text_box_style, 'layout': text_box_layout},
            ],
            render_fn=update_plot,
            columns=[['mode'], ['channel_i', 'channel_j'], ['chromophore', 'agg'],
                     ['time_min', 'time_max'], ['freq_min', 'freq_max']],
            observers=[('mode', _on_mode_change)],
        )
        # Sync the initial enabled/disabled state to match the mode toggle's
        # starting value (observers only fire on subsequent changes).
        _on_mode_change({'new': built['mode'].value}, built)


    def plot_matrix(self, chromophore: str = 'HbO', cmap=None, vmin=None, vmax=None, annot=False, **kwargs):
        """
        Plot the connectivity matrix as a labelled heatmap.

        Parameters
        ----------
        chromophore : str, optional
            Chromophore to display. Default is ``'HbO'``.
        cmap : str, optional
            Colormap. Defaults to the one recorded on the output.
        vmin, vmax : float, optional
            Colour scale limits. Taken from the data if omitted.
        annot : bool, optional
            If True, print each cell's value on the heatmap. Default is False.
        **kwargs
            Passed to :func:`milob.viz.matrix.plot_matrix`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the heatmap.
        """
        from ..viz.matrix import plot_matrix
        from ..viz import theme

        df = self.to_matrix(chromophore, exclude_bad=True)

        # Defaults follow this output's units/handle_negatives state (Pearson
        # kept: -1..1 diverging; folded to non-negative: 0..1 sequential)
        # rather than assuming correlation.
        range_lo, range_hi = self.output.attrs.get('range', (-1, 1))
        vmin = range_lo if vmin is None else vmin
        vmax = range_hi if vmax is None else vmax
        cmap = cmap or self.output.attrs.get('cmap', theme.DIVERGING_CMAP)
        method_label = self.output.attrs.get('method', 'correlation')

        return plot_matrix(
            df, cmap=cmap, vmin=vmin, vmax=vmax, annot=annot,
            colorbar_label=f"{method_label} value",
            title=f"Connectivity Matrix: {chromophore} (Valid Channels Only)",
            **kwargs
        )
        
        
        
    def plot_connectivity_map(self, chromophore='HbO', threshold=0.5,
                              show_labels=True, **kwargs):
        """
        Plot the connectivity network on a 2-D probe layout.

        Parameters
        ----------
        chromophore : str, optional
            Chromophore to display. Default is ``'HbO'``.
        threshold : float, optional
            Minimum absolute value for a connection to be drawn. Default is 0.5.
        show_labels : bool, optional
            If True, annotate each node with its channel label. Default is True.
        **kwargs
            Passed to the underlying plotting function, which accepts
            ``label_fontsize`` among others.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the map.
        """
        from ..viz.topo import plot_connectivity_map

        matrix = self.to_matrix(chromophore, exclude_bad=True)
        return plot_connectivity_map(
            self.probe, matrix, chromophore=chromophore, threshold=threshold,
            show_labels=show_labels, label_fontsize=kwargs.get('label_fontsize', 8),
            ax=kwargs.get('ax'),
        )
    
    
    def plot_seed_connectivity(self, seed_channel: str, chromophore='HbO', **kwargs):
        """
        Plot a topographic map of connectivity to one seed channel.

        Parameters
        ----------
        seed_channel : str
            Label of the channel used as the seed.
        chromophore : str, optional
            Chromophore to display. Default is ``'HbO'``.
        **kwargs
            Passed to :func:`milob.viz.topo.plot_topo_map`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the map.
        """
        _require_reduced(self.output['value'], 'plot_seed_connectivity')
        available = list(self.output.channel_i.values)
        if seed_channel not in available:
            raise ValueError(
                f"Channel '{seed_channel}' not found in results. "
                f"Available channels (first 10): {available[:10]}"
            )

        # Extract the specific row (or column) for the seed
        # Result is a 1D vector of correlations [n_channels]
        seed_vector = self.output.value.sel(
            channel_i=seed_channel, 
            chromophore=chromophore
        ).values

        # Use the existing topo engine to interpolate and plot
        from ..viz.topo import plot_topo_map
        
        title = kwargs.pop('title', f"Seed Connectivity: {seed_channel} ({chromophore})")

        # Defaults follow this output's units (diverging RdBu_r for signed
        # metrics like Pearson r, sequential for e.g. coherence).
        range_lo, range_hi = self.output.attrs.get('range', (-1, 1))
        kwargs.setdefault('vmin', range_lo)
        kwargs.setdefault('vmax', range_hi)
        kwargs.setdefault('cmap', self.output.attrs.get('cmap', 'RdBu_r'))
        return plot_topo_map(
            self.probe,
            values=seed_vector,
            title=title,
            **kwargs
        )

    def plot_3d(
        self,
        chromophore='HbO',
        mode='connectome',
        seed_channel=None,
        threshold=0.3,
        reference_landmarks=None,
        backend='matplotlib',
        **kwargs,
    ):
        """
        Plot connectivity on a 3-D head model in MNI space.

        Probe positions are mapped into MNI152 space by landmark-based
        rigid-body coregistration, then rendered on the scalp surface.

        Parameters
        ----------
        chromophore : str, optional
            Chromophore to display. Default is ``'HbO'``.
        mode : {'connectome', 'nodes'}, optional
            ``'connectome'`` draws a line between each channel pair above
            ``threshold``, coloured by sign and scaled by magnitude, with nodes
            coloured by mean degree. ``'nodes'`` draws spheres only. Default is
            ``'connectome'``.
        seed_channel : str, optional
            In ``'nodes'`` mode, colour each node by its connectivity to this
            channel rather than by mean degree.
        threshold : float, optional
            Minimum absolute value for an edge to be drawn in connectome mode.
            Default is 0.3.
        reference_landmarks : dict, optional
            MNI reference landmarks as ``{label: (x, y, z)}`` in mm. Defaults
            to the MNI152 fiducials.
        backend : {'matplotlib', 'pyvista'}, optional
            ``'matplotlib'`` renders a static figure; ``'pyvista'`` opens an
            interactive window and requires the ``threed`` extra. Default is
            ``'matplotlib'``.
        **kwargs
            Passed to :func:`milob.imaging.surface.plot_probe_3d`, which
            accepts ``surface``, ``view``, ``show_labels``, ``cmap`` and others.

        Returns
        -------
        tuple or pyvista.Plotter
            ``(fig, ax)`` for the matplotlib backend, a plotter for PyVista.

        Examples
        --------
        >>> fc.plot_3d(chromophore='HbT', threshold=0.4)
        >>> fc.plot_3d(chromophore='HbT', mode='nodes', seed_channel='S1D4')
        """
        from ..imaging.surface import plot_probe_3d

        _require_reduced(self.output['value'], 'plot_3d')
        if self.probe is None:
            raise ValueError("This FCOutput has no probe attached.")

        coreg    = self.probe.coreg(reference_landmarks=reference_landmarks)
        corr     = self.output['value'].sel(chromophore=chromophore).values
        channels = list(self.output.channel_i.values)

        if mode == 'connectome':
            kwargs.setdefault('title', f"Connectivity ({chromophore}) – MNI space")
            return plot_probe_3d(
                coreg,
                connectivity_matrix=corr,
                channel_labels=channels,
                mode='connectome',
                threshold=threshold,
                backend=backend,
                **kwargs,
            )

        # ── nodes mode ────────────────────────────────────────────────────
        if seed_channel is not None:
            if seed_channel not in channels:
                raise ValueError(
                    f"'{seed_channel}' not found. "
                    f"Available (first 10): {channels[:10]}"
                )
            idx    = channels.index(seed_channel)
            values = corr[idx, :]
            range_lo, range_hi = self.output.attrs.get('range', (-1, 1))
            kwargs.setdefault('title',          f"Seed: {seed_channel} ({chromophore}) – MNI space")
            kwargs.setdefault('colorbar_label', f"value with {seed_channel}")
            kwargs.setdefault('cmap',           self.output.attrs.get('cmap', 'RdBu_r'))
            kwargs.setdefault('vmin', range_lo)
            kwargs.setdefault('vmax', range_hi)
        else:
            mat = corr.copy()
            np.fill_diagonal(mat, np.nan)
            values = np.nanmean(np.abs(mat), axis=1)
            kwargs.setdefault('title',          f"Mean connectivity ({chromophore}) – MNI space")
            kwargs.setdefault('colorbar_label', 'Mean |r|')
            kwargs.setdefault('cmap',           'RdYlBu_r')

        return plot_probe_3d(
            coreg,
            values=values,
            channel_labels=channels,
            mode='nodes',
            backend=backend,
            **kwargs,
        )

# ---------------------------------------------------------------------------
# Shared FC post-processing (reduce -> roi_average)
#
# Every batch-level FC entry point (Session.run_fc, Study.run_fc)
# applies the same two optional post-processing steps to
# each occurrence/run/surrogate as soon as it is computed, rather than to
# the accumulated result -- that's what bounds memory across a batch (see
# any of those methods' `reduce` docstring). They all route through
# apply_fc_postprocess() so the step stays defined once: adding a
# post-processing option here reaches every entry point, instead of five
# copies drifting apart.
# ---------------------------------------------------------------------------

def roi_average_params():
    """
    Return the option names :meth:`FCOutput.roi_average` accepts.

    These are the legal keys of a ``roi_kwargs`` dict, read from the live
    signature rather than hardcoded.

    Returns
    -------
    tuple of str
        Parameter names other than the ROI map itself.
    """
    import inspect
    return tuple(
        name for name in inspect.signature(FCOutput.roi_average).parameters
        if name not in ('self', 'rois')
    )


def check_fc_postprocess(roi_map=None, roi_kwargs=None, caller='run_fc'):
    """
    Validate ROI post-processing options before any connectivity is computed.

    Called at each entry point so an unusable option fails immediately
    rather than after a batch has run, and so ``roi_kwargs`` given without
    ``roi_map`` raises instead of being ignored.

    Parameters
    ----------
    roi_map : dict, Probe or callable, optional
        ROI definition to validate.
    roi_kwargs : dict, optional
        Extra options for :meth:`FCOutput.roi_average`.
    caller : str, optional
        Name used in error messages. Default is ``'run_fc'``.

    Raises
    ------
    ValueError
        If the options are unusable, or ``roi_kwargs`` is given without
        ``roi_map``.
    """
    if not roi_kwargs:
        return
    if roi_map is None:
        raise ValueError(
            f"{caller}: roi_kwargs={roi_kwargs!r} was given without roi_map -- "
            f"there is no ROI averaging step for it to configure. Pass "
            f"roi_map= as well, or drop roi_kwargs."
        )
    allowed = roi_average_params()
    unknown = [k for k in roi_kwargs if k not in allowed]
    if unknown:
        raise TypeError(
            f"{caller}: unknown roi_kwargs {unknown} -- FCOutput.roi_average() "
            f"accepts {list(allowed)}."
        )


def resolve_roi_map(roi_map, stream, caller='run_fc'):
    """
    Resolve an ROI map against one subject's stream and check it applies.

    A callable is called with that subject's probe after preprocessing, so
    the assignment sees the channels that survived channel dropping; this
    covers montages whose channel-to-region mapping differs between
    subjects. Whatever it returns is checked against the stream's channel
    labels, since :meth:`FCOutput.roi_average` intersects silently and an
    ROI matching nothing would otherwise produce a NaN block with no error.

    Parameters
    ----------
    roi_map : dict, Probe or callable
        A mapping of ROI name to channel labels, a probe carrying ROIs, or
        a callable taking a probe and returning either of those. A callable
        returning None is taken to have added the ROIs to the probe itself.
    stream : Datastream
        Stream whose probe and channel labels the map is resolved against.
    caller : str, optional
        Name used in messages. Default is ``'run_fc'``.

    Returns
    -------
    dict
        ROI name to channel labels, restricted to channels present.

    Raises
    ------
    ValueError
        If the resolved map matches none of the stream's channels.
    """
    if roi_map is None:
        return None

    if callable(roi_map):
        probe = getattr(stream, 'probe', None)
        if probe is None:
            raise ValueError(
                f"{caller}: a callable roi_map needs a Probe to assign from, but "
                f"stream '{getattr(stream, 'name', '?')}' has none. Pass a plain "
                f"dict {{roi: [channels]}} instead."
            )
        produced = roi_map(probe)
        if produced is None:
            produced = probe.rois
            if not produced:
                raise ValueError(
                    f"{caller}: callable roi_map returned None and left no ROIs on "
                    f"the probe -- it should either return a dict/Probe, or define "
                    f"ROIs via probe.add_roi() so they can be read back."
                )
        elif not (isinstance(produced, dict) or hasattr(produced, 'rois')):
            raise TypeError(
                f"{caller}: callable roi_map returned {type(produced).__name__}; "
                f"expected a dict {{roi: [channels]}}, a Probe, or None (having "
                f"defined the ROIs on the probe via add_roi())."
            )
        roi_map = produced

    mapping = roi_map if isinstance(roi_map, dict) else roi_map.rois
    if not mapping:
        raise ValueError(f"{caller}: roi_map is empty -- no ROIs to average over.")

    available = set(np.asarray(stream.data.channel.values).tolist())
    empty = [name for name, chs in mapping.items()
             if not any(ch in available for ch in chs)]

    if len(empty) == len(mapping):
        example = next(iter(mapping.values()), [])
        raise ValueError(
            f"{caller}: no channel in roi_map matches stream "
            f"'{getattr(stream, 'name', '?')}' -- every ROI would be NaN. "
            f"roi_map labels look like {list(example)[:3]}; the stream's look "
            f"like {sorted(available)[:3]}. Check for a labelling mismatch "
            f"(e.g. role-prefixed 'infant:S1D1' labels on a single-subject "
            f"stream)."
        )
    if empty:
        warnings.warn(
            f"{caller}: ROI(s) {empty} match no channel in stream "
            f"'{getattr(stream, 'name', '?')}' and will be NaN -- expected if "
            f"channel dropping removed the whole region for this subject, a "
            f"labelling mistake otherwise.",
            UserWarning, stacklevel=3,
        )
    return roi_map


def apply_fc_postprocess(result, reduce=None, roi_map=None, roi_kwargs=None):
    """
    Apply the optional reduce and ROI-averaging steps to one output.

    Parameters
    ----------
    result : FCOutput
        Output to post-process.
    reduce : dict, optional
        Keyword arguments for :meth:`FCOutput.reduce`, for example
        ``{'freq': (0.01, 0.1)}``.
    roi_map : dict or Probe, optional
        ROI definition passed to :meth:`FCOutput.roi_average`.
    roi_kwargs : dict, optional
        Extra options for that call; see :func:`roi_average_params`. Used
        only when ``roi_map`` is given.

    Returns
    -------
    FCOutput
        The post-processed output, or the input unchanged if neither step
        was requested.
    """
    if reduce is not None:
        result = result.reduce(**reduce)
    if roi_map is not None:
        result = result.roi_average(roi_map, **(roi_kwargs or {}))
    return result
