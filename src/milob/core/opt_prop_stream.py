import os
import re

import numpy as np
import xarray as xr

from .datastream import ParameterStream

#: Fitted parameters that describe the *shape of the domain* rather than a
#: point property of the medium. They still live on the ``op`` axis when
#: they were recovered by the fit (they carry a 1-sigma like any other
#: fitted quantity, which an attribute could not hold), but they're listed
#: separately so ``geometry`` can report them and so ``optical_labels`` can
#: exclude them. When passed in fixed instead of fitted they never reach the
#: ``op`` axis at all -- they stay in ``forward_parameters``.
GEOMETRY_LABELS = frozenset({'depth', 'radius', 'radius_outer', 'half_height'})

_INDEXED = re.compile(r'^(?P<base>.+)_(?P<index>\d+)$')


def _base_label(label):
    """Strip a layer suffix from an op label, e.g. 'mua_2' -> 'mua'."""
    match = _INDEXED.match(label)
    return match.group('base') if match else label


def _label_index(label):
    """Return the layer index encoded in an op label, or None if unlayered."""
    match = _INDEXED.match(label)
    return int(match.group('index')) if match else None


def pack_fit_arrays(values, errors, param_names, coords, dims,
                    motion_model=None):
    """
    Split a packed fit result into the arrays an OptPropStream holds.

    Medium parameters go to the ``op`` axis and observation-operator
    parameters to ``obs_param``. A fitted ``aDb`` or ``aV2`` is relabelled
    ``bfi``, with the motion model recorded separately.

    Parameters
    ----------
    values, errors : np.ndarray
        Shape (*leading, n_params): fitted values and their one-sigma
        uncertainties, with parameters on the last axis in ``param_names``
        order.
    param_names : list of str
        Fitting-facing parameter names, i.e. the keys of ``param_config``.
    coords : dict
        Coordinates for the leading dimensions.
    dims : list of str
        Names of the leading dimensions, in order.
    motion_model : str, optional
        Selects the aDb or aV2 to bfi mapping. None for a DOS-only fit.

    Returns
    -------
    tuple of (xr.DataArray, xr.DataArray, xr.DataArray or None, xr.DataArray or None)
        Values and uncertainties for the op axis, then for the observation
        parameters. The observation pair is (None, None) when the fit
        recovered none.
    """
    from ..forward.dynamics import to_storage_label
    from ..forward.observation import split_fit_params

    medium_names, obs_names = split_fit_params(param_names)
    column = {name: i for i, name in enumerate(param_names)}

    def _pack(names, axis_name, labels, extra_attrs=None):
        if not names:
            return None, None
        idx = [column[n] for n in names]
        axis_coords = {**coords, axis_name: labels}
        axis_dims = [*dims, axis_name]
        return (
            xr.DataArray(values[..., idx], coords=axis_coords, dims=axis_dims,
                         attrs=dict(extra_attrs or {})),
            xr.DataArray(errors[..., idx], coords=axis_coords, dims=axis_dims,
                         attrs={'description': '1-sigma propagated uncertainty'}),
        )

    op_labels = [to_storage_label(n, motion_model) if motion_model else n
                 for n in medium_names]

    op_da, op_err = _pack(medium_names, 'op', op_labels)
    obs_da, obs_err = _pack(obs_names, 'obs_param', obs_names)
    return op_da, op_err, obs_da, obs_err


def pack_covariance(values, param_names, coords, dims, motion_model=None):
    """
    Wrap a per-slice parameter covariance in a labelled DataArray.

    Medium and observation parameters share one ``param``/``param_2`` axis
    pair, so cross-terms between them are retained.

    Parameters
    ----------
    values : np.ndarray
        Shape (*leading, n_params, n_params).
    param_names : list of str
        Fitting-facing names, matching the matrix row and column order.
    coords : dict
        Coordinates for the leading dimensions.
    dims : list of str
        Names of the leading dimensions, in order.
    motion_model : str, optional
        Applies the same aDb or aV2 to bfi relabelling as the op axis.

    Returns
    -------
    xr.DataArray or None
        None when ``values`` is None.
    """
    if values is None:
        return None

    from ..forward.dynamics import to_storage_label

    labels = [to_storage_label(n, motion_model) if motion_model else n
              for n in param_names]
    return xr.DataArray(
        values,
        coords={**coords, 'param': labels, 'param_2': labels},
        dims=[*dims, 'param', 'param_2'],
        attrs={'description': 'parameter covariance from the fit'},
    )


def fit_attrs(fitting_model=None, transformation=None, length_unit='cm',
              motion_model=None, **extra):
    """
    Build the metadata stamped onto a fitted ``op`` array.

    Parameters
    ----------
    transformation : str
        Name of the fit that produced the values.
    length_unit : str
        Unit the fitter worked in. Absorption and scattering are per this unit.
    motion_model : str, optional
        Scatterer-motion model behind a recovered ``bfi``.

    Returns
    -------
    dict
        Attributes including ``lengthUnit``, ``motion_model`` and a
        ``flowUnit`` derived from the two.
    """
    from ..forward.dynamics import flow_unit

    attrs = {'lengthUnit': length_unit, 'units': f'{length_unit}^-1'}
    if fitting_model is not None:
        attrs['fitting_model'] = fitting_model
    if transformation is not None:
        attrs['transformation'] = transformation
    if motion_model is not None:
        attrs['motion_model'] = motion_model
        attrs['flowUnit'] = flow_unit(motion_model, length_unit)
    attrs.update(extra)
    return attrs


class OptPropStream(ParameterStream):
    """
    Medium properties recovered by inverting a forward model.

    Data has dims (time, <spatial>, wavelength, op), where <spatial> is
    'channel' for a channel-space fit or 'voxel' for a reconstruction, and the
    op axis carries whatever the fit recovered::

        op = ['mua', 'musp']                 semi-infinite DOS
        op = ['mua', 'musp', 'bfi']          joint DOS and DCS
        op = ['bfi']                         DCS with known optical properties
        op = ['mua_1', 'musp_1', 'mua_2',
              'musp_2', 'depth']             two-layer

    A quantity belongs on ``op`` when it is a property of the medium and was
    recovered by the fit. Fixed geometry supplied to the fit lives in
    ``forward_parameters`` and is reported by :attr:`geometry`; parameters of
    the observation operator, such as ``beta``, live in ``obs_params``.

    Blood flow is stored under the label ``bfi`` whatever the scatterer-motion
    model, with :attr:`motion_model` recording which physics produced it, since
    ``aDb`` and ``aV2`` carry different units. :meth:`fit_param_names` maps the
    labels back for refitting.

    Absolute values and perturbations both use the same labels and are
    distinguished by ``status``: 'delta_mua' for a reconstructed perturbation,
    otherwise a fitted absolute.

    Parameters
    ----------
    data : xr.DataArray
        Shape (time, <spatial>, wavelength, op).
    probe : Probe, optional
        Optode geometry. For a voxel-indexed stream it records the montage the
        values came from.
    uncertainty : xr.DataArray, optional
        One-sigma fit uncertainty, same shape as ``data``.
    obs_params : xr.DataArray, optional
        Shape (time, channel, wavelength, obs_param). Fitted parameters of the
        observation operator.
    obs_uncertainty : xr.DataArray, optional
        One-sigma uncertainty on ``obs_params``, same shape.
    covariance : xr.DataArray, optional
        Per-slice parameter covariance, spanning medium and observation
        parameters on a shared axis pair.
    forward_parameters : dict, optional
        Fixed context the fit ran under, such as refractive index or a pinned
        layer depth.
    voxel_grid : imaging.voxel_grid.VoxelGrid, optional
        Required for a voxel-indexed stream: the domain the 'voxel' dim indexes
        into.
    resolution_diag : xr.DataArray, optional
        Shape (<spatial>, wavelength). A resolution-matrix diagonal, where 1.0
        is fully resolved.

    Notes
    -----
    The sidecars are kept aligned with ``data`` by inherited within-space
    operations. An operation with no propagation rule drops them with a
    warning.
    """

    #: Framework-space tag: optical properties theta (see ParameterStream
    #: and the milob-package-design skill's invariant 2). Blood flow is part
    #: of theta, so a DCS fit is an M -> theta crossing exactly as an FD-DOS
    #: fit is -- there is no separate 'bfi' space any more.
    SPACE = 'theta'

    _SIDECARS = ('uncertainty', 'obs_params', 'obs_uncertainty', 'covariance',
                 'resolution_diag')

    def __init__(self, data, probe=None, uncertainty=None, obs_params=None,
                 obs_uncertainty=None, covariance=None,
                 forward_parameters=None, voxel_grid=None,
                 resolution_diag=None, **kwargs):
        super().__init__(data, probe, **kwargs)
        self.uncertainty = uncertainty
        self.obs_params = obs_params
        self.obs_uncertainty = obs_uncertainty
        self.covariance = covariance
        self.forward_parameters = dict(forward_parameters or {})
        self.voxel_grid = voxel_grid
        self.resolution_diag = resolution_diag

        if 'op' not in self.data.dims:
            raise ValueError(
                "OptPropStream requires an 'op' dimension. Expected e.g. "
                "op=['mua', 'musp'], op=['bfi'], op=['mua'] for a CW-DOT "
                "reconstruction, or a layered vocabulary like "
                "op=['mua_1', 'musp_1', 'mua_2', 'musp_2', 'depth']."
            )
        present = [d for d in self._SPATIAL_DIMS if d in self.data.dims]
        if len(present) != 1:
            raise ValueError(
                f"OptPropStream needs exactly one spatial dimension out of "
                f"{self._SPATIAL_DIMS}; got {present or 'none'} in "
                f"{tuple(self.data.dims)}."
            )
        if present[0] == 'voxel' and self.voxel_grid is None:
            raise ValueError(
                "A voxel-indexed OptPropStream needs a voxel_grid. Unlike "
                "'channel', which recovers its geometry from `probe`, "
                "'voxel' is a bare integer index -- without the grid the "
                "stream cannot say where any of its values are."
            )

    # ------------------------------------------------------------------
    # Device readers
    # ------------------------------------------------------------------

    #: Extensions routed to each underlying parser by `from_openmotion`.
    _OPENMOTION_DB_EXTS = ('.db', '.sqlite', '.sqlite3')
    _OPENMOTION_CSV_EXTS = ('.csv',)

    @classmethod
    def from_openmotion(cls, path, sds_mm=None, *, session_id=None,
                        session_label=None, wavelength=795.0,
                        sc_threshold=None, name=None):
        """
        Load blood flow from an OpenMotion SCOS device.

        Reads either the scan database or a corrected-CSV export, chosen from the
        file extension. The device measures the flow index in hardware rather than
        by fitting, so there is no motion model, no flow unit and no fit
        uncertainty.

        Parameters
        ----------
        path : str
            Path to the ``scans.db`` SQLite file or a corrected-CSV export.
        sds_mm : array-like of 8 floats, optional
            Source-detector separations in mm for cameras 1 to 8, shared by both
            sensor modules. Defaults to the standard hardware spacing. Unused for a
            reduced-mode recording holding only per-side averages, where the
            returned probe is None.
        session_id : int, optional
            Session to load. Database only; required when the database holds more
            than one session and ``session_label`` is not given.
        session_label : str, optional
            Session label to load, taking the most recent match. Database only.
        wavelength : float
            Laser wavelength in nm. Not recorded by the device. Default 795.0.
        sc_threshold : float or None
            Source-detector distance in mm below which a channel is short.
            Default None.
        name : str, optional
            Stream name. Defaults to the session label, or the CSV's filename stem.

        Returns
        -------
        OptPropStream
            Shape (time, channel, wavelength=1, op=['bfi']).

        Raises
        ------
        ValueError
            If ``session_id`` or ``session_label`` is given for a CSV path.
        """
        from ..io.openmotion import read_openmotion, read_openmotion_csv, DEFAULT_SDS_MM

        sds_mm = DEFAULT_SDS_MM if sds_mm is None else sds_mm
        ext = os.path.splitext(str(path))[1].lower()

        if ext in cls._OPENMOTION_CSV_EXTS:
            if session_id is not None or session_label is not None:
                raise ValueError(
                    "session_id/session_label select a session within a "
                    "scans.db and don't apply to a CSV export, which holds "
                    "exactly one scan's frames."
                )
            data_xr, probe, session = read_openmotion_csv(
                path, sds_mm, wavelength=wavelength, sc_threshold=sc_threshold,
            )
            history_op = 'from_openmotion_csv'
            history_params = {'csv_path': str(path)}
        elif ext in cls._OPENMOTION_DB_EXTS:
            data_xr, probe, session = read_openmotion(
                path, sds_mm, session_id=session_id, session_label=session_label,
                wavelength=wavelength, sc_threshold=sc_threshold,
            )
            meta = session['session_meta']
            history_op = 'from_openmotion'
            history_params = {
                'db_path': str(path),
                'session_id': session['id'],
                'subject_id': meta.get('subject_id'),
                'operator': meta.get('operator'),
                'scan_id': meta.get('scan_id'),
                'started_at_iso': meta.get('started_at_iso'),
            }
        else:
            raise ValueError(
                f"Can't tell whether {path!r} is an OpenMotion scans.db or "
                f"a corrected-CSV export from its extension {ext!r} -- "
                f"expected one of {cls._OPENMOTION_DB_EXTS + cls._OPENMOTION_CSV_EXTS}."
            )

        obj = cls(
            data=data_xr,
            probe=probe,
            name=name or session['session_label'],
            status='bfi',
        )
        obj.add_history(history_op, {
            **history_params,
            'device': 'OpenMotion (Open Water)',
            'session_label': session['session_label'],
            'sds_mm': list(sds_mm) if sds_mm is not None else None,
            'wavelength': wavelength,
        })
        return obj

    # ------------------------------------------------------------------
    # The op vocabulary
    # ------------------------------------------------------------------

    @property
    def op_labels(self):
        """Labels present on the ``op`` axis."""
        return [str(v) for v in self.data.op.values]

    @property
    def optical_labels(self):
        """Labels on the ``op`` axis that are optical properties."""
        return [l for l in self.op_labels if _base_label(l) not in GEOMETRY_LABELS]

    @property
    def obs_labels(self):
        """Labels of the fitted observation-operator parameters."""
        if self.obs_params is None:
            return []
        return [str(v) for v in self.obs_params.obs_param.values]

    @property
    def is_layered(self):
        """True if the op labels carry layer suffixes."""
        return any(_label_index(l) is not None for l in self.op_labels)

    @property
    def n_layers(self):
        """Number of layers encoded in the op labels, or 1 if unlayered."""
        indices = [_label_index(l) for l in self.op_labels]
        indices = [i for i in indices if i is not None]
        return max(indices) if indices else 1

    def fit_param_names(self):
        """
        Map each ``op`` label to the parameter name the forward model expects.

        Applies this stream's motion model, so ``bfi`` becomes ``aDb`` or ``aV2``.

        Returns
        -------
        dict of {str: str}
            Storage label to forward-model keyword.
        """
        from ..forward.dynamics import from_storage_label
        model = self.motion_model or 'brownian'
        return {label: from_storage_label(label, model) for label in self.op_labels}

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def length_unit(self):
        """Length unit the fit worked in. Absorption and scattering are per this unit."""
        return self.data.attrs.get('lengthUnit', 'cm')

    @property
    def motion_model(self):
        """Scatterer-motion model behind a recovered ``bfi``, or None."""
        return self.data.attrs.get('motion_model')

    @property
    def flow_unit(self):
        """
        Units of ``bfi``.

        Returns
        -------
        str or None
            Derived from the motion model and length unit. None when no flow was
            recovered.
        """
        if self.motion_model is None or 'bfi' not in {_base_label(l) for l in self.op_labels}:
            return None
        from ..forward.dynamics import flow_unit
        return flow_unit(self.motion_model, self.length_unit)

    @property
    def fitting_model(self):
        """Name of the fit that produced this stream."""
        return self.data.attrs.get('fitting_model', 'unknown')

    @property
    def geometry(self):
        """
        Domain geometry this fit assumed.

        Returns
        -------
        dict
            Entries supplied fixed, as scalars, together with entries recovered by
            the fit, as DataArrays.
        """
        out = {k: v for k, v in self.forward_parameters.items()
               if _base_label(k) in GEOMETRY_LABELS}
        for label in self.op_labels:
            if _base_label(label) in GEOMETRY_LABELS:
                out[label] = self.data.sel(op=label)
        return out

    @property
    def is_multiwavelength(self):
        """True if more than one wavelength is present."""
        return 'wavelength' in self.data.dims

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def select(self, label):
        """
        Return one stored parameter by its exact ``op`` label.

        Parameters
        ----------
        label : str
            Label to select, e.g. 'mua', 'bfi', 'musp_2' or 'depth'.

        Returns
        -------
        xr.DataArray
            Shape (time, <spatial>, wavelength).
        """
        if label not in self.op_labels:
            raise ValueError(
                f"'{label}' is not on this stream's op axis. "
                f"Available: {self.op_labels}"
            )
        return self.data.sel(op=label)

    def _select_base(self, base):
        """Resolve a base name against the op vocabulary, stacking layers if present."""
        labels = self.op_labels
        if base in labels:
            return self.data.sel(op=base)

        indexed = [l for l in labels if _base_label(l) == base]
        if not indexed:
            raise ValueError(
                f"No '{base}' in this stream. Available op labels: {labels}"
            )

        indexed.sort(key=_label_index)
        stacked = xr.concat([self.data.sel(op=l) for l in indexed], dim='layer')
        return stacked.assign_coords(layer=[_label_index(l) for l in indexed])

    @property
    def mua(self):
        """Absorption coefficient, per :attr:`length_unit`."""
        return self._select_base('mua')

    @property
    def musp(self):
        """Reduced scattering coefficient, per :attr:`length_unit`."""
        return self._select_base('musp')

    @property
    def bfi(self):
        """
        Blood flow index.

        Returns
        -------
        xr.DataArray
            Values in :attr:`flow_unit`, which depends on the motion model.
        """
        return self._select_base('bfi')

    @property
    def beta(self):
        """
        Coherence factor of the observation operator.

        Returns
        -------
        xr.DataArray or None
            None when ``beta`` was not fitted.
        """
        if self.obs_params is None or 'beta' not in self.obs_labels:
            return None
        return self.obs_params.sel(obs_param='beta')

    def covariance_of(self, label_a, label_b=None):
        """
        Covariance between two fitted parameters.

        Parameters
        ----------
        label_a, label_b : str
            Parameter labels, which may name medium or observation parameters.

        Returns
        -------
        xr.DataArray or None
            None when no covariance sidecar is present.
        """
        if self.covariance is None:
            return None
        if label_b is None:
            label_b = label_a
        return self.covariance.sel(param=label_a, param_2=label_b)

    def correlation(self):
        """
        Correlation matrix of the fitted parameters.

        Returns
        -------
        xr.DataArray or None
            Covariance normalised by the parameter standard deviations, or None
            when no covariance sidecar is present.
        """
        if self.covariance is None:
            return None
        cov = self.covariance
        n = cov.sizes['param']
        # Operate on the array directly: 'param'/'param_2' share their
        # labels, so an xarray-level divide would try to align them.
        diag = np.arange(n)
        sd = np.sqrt(np.abs(cov.values[..., diag, diag]))
        with np.errstate(invalid='ignore', divide='ignore'):
            corr = cov.values / (sd[..., :, None] * sd[..., None, :])
        return cov.copy(data=corr)

    def uncertainty_of(self, label):
        """
        One-sigma uncertainty on a single parameter.

        Parameters
        ----------
        label : str
            Parameter label, on either the op or observation axis.

        Returns
        -------
        xr.DataArray or None
            None when the relevant sidecar was never set or has been dropped.
        """
        if label in self.op_labels:
            if self.uncertainty is None:
                return None
            return self.uncertainty.sel(op=label)
        if label in self.obs_labels:
            if self.obs_uncertainty is None:
                return None
            return self.obs_uncertainty.sel(obs_param=label)
        raise ValueError(
            f"'{label}' is neither an op label {self.op_labels} nor an "
            f"observation parameter {self.obs_labels}."
        )

    def at_wavelength(self, wavelength):
        """
        Select a single wavelength.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm.

        Returns
        -------
        OptPropStream
            Stream holding that wavelength alone.
        """
        if not self.is_multiwavelength:
            raise ValueError("This stream has no 'wavelength' dimension.")
        return self.data.sel(wavelength=wavelength, method='nearest')

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    def to_concentration(self,
                         water_fraction: float = None,
                         layer: int = None,
                         inplace: bool = False):
        """
        Unmix absorption into chromophore concentrations.

        Inverts the extinction-coefficient matrix across wavelengths. For a
        voxel-indexed stream this is applied after reconstruction, voxel by voxel.

        Parameters
        ----------
        water_fraction : float, optional
            Water volume fraction subtracted from absorption before unmixing.
            Defaults to 0.75 for a fitted absolute stream and 0.0 for a
            ``'delta_mua'`` perturbation, where water cancels in the difference.
        layer : int, optional
            Which layer's absorption to unmix. Required for a layered stream.
        inplace : bool
            Ignored; a new object is always returned, since the component
            dimension replaces wavelength.

        Returns
        -------
        TissueStream
            Shape (time, <spatial>, component) with component = ['HbO', 'HbR'],
            carrying ``status='absolute_conc'`` or ``'conc'`` to match the input.
            Uncertainty is propagated through the unmixing where the source stream
            carries one. ``resolution_diag`` is not carried, being indexed by the
            consumed wavelength dimension.

        Raises
        ------
        ValueError
            If a non-zero ``water_fraction`` is given for a perturbation, or if
            ``layer`` is omitted for a layered stream.
        """
        from .tissue_stream import TissueStream
        from ..processing.time_domain import opt_params_to_conc

        if self.is_layered and layer is None:
            raise ValueError(
                f"This is a layered stream (op={self.op_labels}); 'the' mua "
                f"is ambiguous. Pass layer=1..{self.n_layers} to choose which "
                f"layer's mua to unmix."
            )

        is_delta = self.status == 'delta_mua'
        if water_fraction is None:
            water_fraction = 0.0 if is_delta else 0.75
        if is_delta and water_fraction:
            raise ValueError(
                "water_fraction must be 0.0 for a status='delta_mua' stream. "
                "This is a perturbation around a baseline, and water's "
                "contribution to mua is part of that baseline, so it cancels "
                "in the difference -- subtracting it again would remove a "
                "constant that is not there."
            )

        n_wl = self.data.sizes.get('wavelength', 0)
        if n_wl < 2:
            raise ValueError(
                f"Spectral unmixing of two chromophores needs at least 2 "
                f"wavelengths, got {n_wl}."
            )

        if layer is not None and self.is_layered:
            mua = self.select(f'mua_{layer}')
            unc = None
        else:
            mua = self._select_base('mua')
            unc = (self.uncertainty.sel(op='mua')
                   if self.uncertainty is not None
                   and 'op' in getattr(self.uncertainty, 'dims', ())
                   else self.uncertainty)

        result = opt_params_to_conc(
            mua=mua,
            water_corr=(water_fraction > 0),
            water_frac=water_fraction,
            attrs=self.data.attrs,
            component_dim='component',
            status='conc' if is_delta else 'absolute_conc',
            uncertainty=unc,
        )
        conc_data, unc_data = result if unc is not None else (result, None)

        if self.spatial_dim == 'voxel':
            from ..imaging import IMAGING_DTYPE
            conc_data = conc_data.astype(IMAGING_DTYPE)
            if unc_data is not None:
                unc_data = unc_data.astype(IMAGING_DTYPE)

        history = self.history.copy()
        history.append(self._history_entry('to_concentration', {
            'water_fraction': water_fraction,
            'water_correction': water_fraction > 0,
            'layer': layer,
            'source': 'OptPropStream',
            'spatial_dim': self.spatial_dim,
            'delta': is_delta,
            'wavelengths': [float(w) for w in np.asarray(self.data.wavelength.values)],
            'uncertainty_propagated': unc_data is not None,
            'resolution_diag_dropped': self.resolution_diag is not None,
        }))

        return TissueStream(
            data=conc_data,
            probe=self.probe,
            voxel_grid=self.voxel_grid,
            uncertainty=unc_data,
            name=self.name,
            events=self.events,
            status='conc' if is_delta else 'absolute_conc',
            history=history,
        )

    def decompose_flow(self, alpha, alpha_uncertainty=None, layer=None):
        """
        Split the flow index into its composition factors, bfi = alpha * Db.

        Only the product is identifiable from optical data, so ``alpha`` is an
        explicit assumption rather than a fitted quantity and has no default.

        Parameters
        ----------
        alpha : float
            Assumed fraction of moving scatterers, in (0, 1].
        alpha_uncertainty : float, optional
            One-sigma on ``alpha``, propagated into the uncertainty on Db when
            given.
        layer : int, optional
            Which layer's flow to decompose, for a layered stream.

        Returns
        -------
        TissueStream
            Shape (time, channel, component) with component = ['Db'], or ['V2']
            under random ballistic motion. ``alpha`` is recorded in ``data.attrs``
            and in the history rather than stored as a component.
        """
        import numpy as np

        from ..forward.dynamics import bfi_param_name
        from .tissue_stream import TissueStream

        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1]; got {alpha}.")

        label = 'bfi' if layer is None else f'bfi_{layer}'
        if label not in self.op_labels:
            raise ValueError(
                f"No '{label}' on this stream (op={self.op_labels}). "
                f"decompose_flow() needs a recovered blood flow index."
            )

        motion = self.motion_model or 'brownian'
        # 'aDb' -> 'Db', 'aV2' -> 'V2': the same quantity without the alpha.
        component = bfi_param_name(motion).lstrip('a')

        bfi = self.select(label)
        db = bfi / alpha

        uncertainty = None
        bfi_sigma = self.uncertainty_of(label)
        if bfi_sigma is not None:
            var = (bfi_sigma / alpha) ** 2
            if alpha_uncertainty is not None:
                var = var + (bfi / alpha ** 2) ** 2 * alpha_uncertainty ** 2
            uncertainty = np.sqrt(var).expand_dims(component=[component], axis=-1)

        data = db.expand_dims(component=[component], axis=-1)
        data.attrs.update({
            'assumed_alpha': float(alpha),
            'assumed_alpha_uncertainty': (None if alpha_uncertainty is None
                                          else float(alpha_uncertainty)),
            'motion_model': motion,
            'units': self.flow_unit,
            'transformation': 'decompose_flow',
        })

        history = self.history.copy()
        history.append(self._history_entry('decompose_flow', {
            'alpha': float(alpha),
            'alpha_uncertainty': alpha_uncertainty,
            'motion_model': motion,
            'source_label': label,
            'note': 'alpha assumed, not fitted -- only the product alpha*Db '
                    'is identifiable from optical data',
        }))

        return TissueStream(
            data=data, uncertainty=uncertainty, probe=self.probe,
            name=self.name, events=self.events, status='flow_decomposed',
            history=history,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self):
        wls = (list(self.data.wavelength.values)
               if 'wavelength' in self.data.coords else '?')
        obs = f"Obs: {self.obs_labels} | " if self.obs_labels else ""
        motion = f"Motion: {self.motion_model} | " if self.motion_model else ""
        return (
            f"<OptPropStream | {self.name} | "
            f"op: {self.op_labels} | "
            f"{obs}{motion}"
            f"Wavelengths: {wls} | "
            f"{self.spatial_dim.capitalize()}s: {self.n_spatial:,} | "
            f"Duration: {len(self.data.time)} frames>"
        )
