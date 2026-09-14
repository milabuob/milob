import numpy as np
import xarray as xr
from .datastream import ParameterStream


def _pivot():
    """Pivot wavelength of the scattering power law, in nm."""
    from ..forward.spectral import WAVELENGTH_PIVOT_NM
    return WAVELENGTH_PIVOT_NM


def _quadratic_form(covariance, jacobian):
    """Delta-method variance of a scalar function of several fitted parameters."""
    names = list(jacobian)
    total = None
    for a in names:
        for b in names:
            term = jacobian[a] * jacobian[b] * covariance.sel(param=a, param_2=b)
            total = term if total is None else total + term
    return total


class TissueStream(ParameterStream):
    """
    Tissue composition and scattering parameters recovered from optical
    properties.

    Data has dims (time, <spatial>, component), where <spatial> is 'channel' or
    'voxel'. The component axis holds chromophore concentrations in micromolar
    and, where a spectral fit supplied them, the scattering power-law amplitude
    ``A`` and exponent ``b`` of musp(lambda) = A * (lambda / lambda_0) ** -b.

    Produced by :meth:`~milob.core.opt_prop_stream.OptPropStream.to_concentration`
    and :meth:`~milob.core.fd_nirs.FD_Stream.fit_to_conc`. Relative changes from
    CW processing stay in ``CW_Stream`` with ``status='conc'`` instead.

    Parameters
    ----------
    data : xr.DataArray
        Shape (time, <spatial>, component).
    probe : Probe
        Optode geometry. For a voxel-indexed stream it records the montage the
        values came from.
    voxel_grid : imaging.voxel_grid.VoxelGrid, optional
        Required for a voxel-indexed stream: the domain the 'voxel' dim indexes
        into.
    uncertainty : xr.DataArray, optional
        One-sigma fit uncertainty, same shape as ``data``. Set by
        ``fit_to_conc()``; None for streams from ``to_concentration()``.
    status : {'absolute_conc', 'conc'}
        'absolute_conc' for a fitted absolute composition, 'conc' for a
        perturbation.
    """
    #: Framework-space tag: composition/tissue c (see ParameterStream and
    #: the milob-package-design skill's invariant 2). Distinct from
    #: OptPropStream's 'theta' -- to_concentration() (theta -> c) is a real
    #: across-space crossing, not a within-space copy.
    SPACE = 'c'

    #: Kept aligned with `data` by every inherited within-space operation --
    #: see Datastream._rebuild().
    _SIDECARS = ('uncertainty', 'covariance')

    def __init__(self, data, probe=None, uncertainty=None, covariance=None,
                 voxel_grid=None, **kwargs):
        super().__init__(data, probe, **kwargs)
        self.uncertainty = uncertainty
        self.voxel_grid = voxel_grid
        #: Full parameter covariance from the fit that produced this stream,
        #: dims (time, channel, param, param_2) -- `None` for streams built
        #: by `OptPropStream.to_concentration()`'s linear unmixing, which
        #: doesn't currently propagate one. Set by `FD_Stream.fit_to_conc()`,
        #: where it matters: HbO and HbR are strongly anti-correlated, so
        #: anything derived from both (HbT, StO2) needs the off-diagonal to
        #: get its own uncertainty right.
        self.covariance = covariance

        if 'component' not in self.data.dims:
            raise ValueError(
                "TissueStream requires a 'component' dimension. "
                "Expected e.g. component=['HbO', 'HbR'] or "
                "component=['HbO', 'HbR', 'A', 'b']."
            )
        present = [d for d in self._SPATIAL_DIMS if d in self.data.dims]
        if len(present) != 1:
            raise ValueError(
                f"TissueStream needs exactly one spatial dimension out of "
                f"{self._SPATIAL_DIMS}; got {present or 'none'} in "
                f"{tuple(self.data.dims)}."
            )
        if present[0] == 'voxel' and self.voxel_grid is None:
            raise ValueError(
                "A voxel-indexed TissueStream needs a voxel_grid: 'voxel' is "
                "a bare integer index, so without the grid the stream cannot "
                "say where any of its values are."
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def components(self) -> list:
        """Component labels present in this stream."""
        return list(self.data.component.values)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def select(self, component: str) -> xr.DataArray:
        """
        Return the time series for a single component.

        Parameters
        ----------
        component : str
            Component label, e.g. 'HbO', 'HbR', 'H2O', 'A' or 'b'.

        Returns
        -------
        xr.DataArray
            Shape (time, <spatial>).
        """
        if component not in self.components:
            raise ValueError(
                f"'{component}' not found. Available: {self.components}"
            )
        return self.data.sel(component=component)

    def hbo(self) -> xr.DataArray:
        """Oxyhaemoglobin concentration, equivalent to ``select('HbO')``."""
        return self.select('HbO')

    def hbr(self) -> xr.DataArray:
        """Deoxyhaemoglobin concentration, equivalent to ``select('HbR')``."""
        return self.select('HbR')

    def scattering_amplitude(self) -> xr.DataArray:
        """
        Scattering power-law amplitude ``A``.

        Returns
        -------
        xr.DataArray
            musp in cm^-1 at the pivot wavelength.
        """
        return self.select('A')

    def scattering_power(self) -> xr.DataArray:
        """
        Scattering power-law exponent ``b``.

        Returns
        -------
        xr.DataArray
            Dimensionless exponent. Weakly constrained when only two wavelengths
            were fitted.
        """
        return self.select('b')

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    def to_op(self, wavelengths, inplace: bool = False):
        """
        Evaluate optical properties from this composition, at any wavelength.

        The inverse of :meth:`~milob.core.opt_prop_stream.OptPropStream.to_concentration`.
        Because the spectral model is physical rather than interpolated, it is
        valid outside the fitted wavelengths, which makes it useful for evaluating
        properties at a DCS laser line. Requires 'HbO', 'HbR', 'A' and 'b' on the
        component axis.

        Parameters
        ----------
        wavelengths : float or sequence of float
            Wavelengths in nm to evaluate at.
        inplace : bool
            Ignored; a new object is always returned, since the component axis is
            replaced by wavelength and op.

        Returns
        -------
        OptPropStream
            Shape (time, <spatial>, wavelength, op), with op = ['mua', 'musp'].
            Uncertainty is propagated by the delta method when this stream carries
            a covariance sidecar, and is None otherwise.
        """
        import numpy as np

        from ..forward.spectral import (WAVELENGTH_PIVOT_NM, extinction_at,
                                        mua_from_composition, musp_powerlaw)
        from .opt_prop_stream import OptPropStream, fit_attrs

        required = ['HbO', 'HbR', 'A', 'b']
        missing = [c for c in required if c not in self.components]
        if missing:
            raise ValueError(
                f"to_op() needs {required} on the component "
                f"axis; missing {missing}. This stream has {self.components}. "
                f"A stream from OptPropStream.to_concentration() carries "
                f"concentrations only -- no scattering shape -- so musp "
                f"cannot be reconstructed from it; use FD_Stream.fit_to_conc()."
            )

        wavelengths = np.atleast_1d(np.asarray(wavelengths, dtype=float))

        hbo, hbr = self.select('HbO'), self.select('HbR')
        amp, power = self.select('A'), self.select('b')

        op_slices, err_slices = [], []
        for wl in wavelengths:
            eps_hbo, eps_hbr = extinction_at(float(wl))
            mua = mua_from_composition(hbo, hbr, eps_hbo, eps_hbr)
            musp = musp_powerlaw(amp, power, float(wl))
            op_slices.append(xr.concat([mua, musp], dim='op')
                             .assign_coords(op=['mua', 'musp']))
            err_slices.append(self._propagate_to_op(wl, eps_hbo, eps_hbr,
                                                    amp, power, musp))

        data = xr.concat(op_slices, dim='wavelength').assign_coords(
            wavelength=wavelengths).transpose('time', self.spatial_dim, 'wavelength', 'op')
        data.attrs.update(fit_attrs(
            fitting_model='spectral_operator_S',
            transformation='to_op',
            length_unit=self.data.attrs.get('lengthUnit', 'cm'),
        ))

        uncertainty = None
        if all(e is not None for e in err_slices):
            uncertainty = xr.concat(err_slices, dim='wavelength').assign_coords(
                wavelength=wavelengths).transpose('time', self.spatial_dim, 'wavelength', 'op')

        history = self.history.copy()
        history.append(self._history_entry('to_op', {
            'wavelengths': [float(w) for w in wavelengths],
            'pivot_nm': WAVELENGTH_PIVOT_NM,
            'uncertainty_propagated': uncertainty is not None,
        }))

        return OptPropStream(
            data=data,
            uncertainty=uncertainty,
            probe=self.probe,
            voxel_grid=self.voxel_grid,
            name=self.name,
            events=self.events,
            status='op',
            history=history,
        )

    def _propagate_to_op(self, wl, eps_hbo, eps_hbr, amp, power, musp):
        """Propagate composition uncertainty into optical properties by the delta method."""
        import numpy as np

        if self.covariance is None:
            return None

        ratio = float(wl) / _pivot()
        # d(mua)/d(HbO), d(mua)/d(HbR)  -- the 1e-6 is the uM -> mol/L factor
        # mua_from_composition applies.
        jac_mua = {'HbO': eps_hbo * 1e-6, 'HbR': eps_hbr * 1e-6}
        # d(musp)/dA = musp/A ; d(musp)/db = -musp * ln(lambda/lambda_0)
        jac_musp = {'A': musp / amp, 'b': -musp * np.log(ratio)}

        var_mua = _quadratic_form(self.covariance, jac_mua)
        var_musp = _quadratic_form(self.covariance, jac_musp)

        return xr.concat([np.sqrt(np.abs(var_mua)), np.sqrt(np.abs(var_musp))],
                         dim='op').assign_coords(op=['mua', 'musp'])

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def hbt(self) -> xr.DataArray:
        """
        Total haemoglobin, HbT = HbO + HbR.

        Returns
        -------
        xr.DataArray
            Shape (time, <spatial>).
        """
        return self.select('HbO') + self.select('HbR')

    def sto2(self) -> xr.DataArray:
        """
        Oxygen saturation, StO2 = HbO / HbT.

        Returns
        -------
        xr.DataArray
            Saturation in the range 0 to 1.
        """
        hbt = self.hbt()
        return self.select('HbO') / hbt

    def oef(self) -> xr.DataArray:
        """
        Oxygen extraction fraction, OEF = HbR / HbT.

        Returns
        -------
        xr.DataArray
            Fraction in the range 0 to 1.
        """
        hbt = self.hbt()
        return self.select('HbR') / hbt

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            # 'absolute' was hardcoded here when this class only ever held
            # a fitted absolute composition. It now also holds reconstructed
            # perturbations, and a repr that calls a Delta "absolute" is
            # exactly the kind of thing that gets copied into a figure
            # caption.
            f"<TissueStream | {self.name} | "
            f"{'delta' if self.status == 'conc' else 'absolute'} | "
            f"Components: {self.components} | "
            f"{self.spatial_dim.capitalize()}s: {self.n_spatial:,} | "
            f"Duration: {len(self.data.time)} frames>"
        )
