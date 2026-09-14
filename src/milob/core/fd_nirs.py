import numpy as np
import xarray as xr
from typing import Optional
from .datastream import Datastream
from .nirs import NirsStream


class FD_Stream(NirsStream):
    """
    Frequency-domain diffuse optical spectroscopy data.

    Data has dims (time, channel, wavelength, freq) and complex dtype. The
    'freq' dimension holds modulation frequencies in Hz: entry 0 is the DC
    component, with zero imaginary part, and each non-zero frequency holds
    ``AC_amplitude * exp(i * phase)``. A single-frequency instrument therefore
    gives a freq axis of length two.

    Parameters
    ----------
    data : xr.DataArray
        Shape (time, channel, wavelength, freq), complex.
    probe : Probe
        Optode geometry.
    """

    def __init__(self, data, probe, **kwargs):
        super().__init__(data, probe, **kwargs)

        if 'freq' not in self.data.dims:
            raise ValueError(
                "FD_Stream requires a 'freq' dimension (modulation frequencies in Hz). "
                "Use freq=0 for DC and freq=f_mod for complex AC."
            )

        if not np.iscomplexobj(self.data.values):
            raise ValueError(
                "FD_Stream data must be complex dtype. "
                "DC at freq=0 should have imaginary=0; "
                "AC at freq=f_mod should be AC_amplitude * exp(i * phase_rad)."
            )

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_oxiplex(cls, filepath, detector, wavelength_mask, *, sc_threshold, **kwargs):
        """
        Load an ISS OxiplexTS text file.

        Parameters
        ----------
        filepath : str
            Path to the text file.
        detector : int, optional
            Detector to read.
        wavelength_mask : array-like, optional
            Wavelengths to keep.
        sc_threshold : float or None
            Source-detector distance in mm below which a channel is short.

        Returns
        -------
        FD_Stream
        """
        from ..io.oxiplex import read_oxiplex
        return read_oxiplex(filepath, detector, wavelength_mask, sc_threshold=sc_threshold, **kwargs)

    @classmethod
    def read_iss(cls, datafile, layout_file, *, modulation_frequency, sc_threshold=None, **kwargs):
        """
        Load an ISS Imagent BOXY record with its probe layout file.

        Parameters
        ----------
        datafile : str
            Path to the BOXY ascii record file.
        layout_file : str
            Path to the probe .layout file, holding the SD struct and channel map.
        modulation_frequency : float
            Instrument RF modulation frequency in Hz. Not stored in the file, so
            it must be supplied.
        sc_threshold : float or None
            Source-detector distance in mm below which a channel is short.
        apply_calibration : bool
            Apply the file's own AC/DC/phase calibration values. Default False,
            as they are normally already applied to the stored data.

        Returns
        -------
        FD_Stream
        """
        from ..io.dos_readers import read_ISS_BOXY
        return read_ISS_BOXY(datafile, layout_file, modulation_frequency=modulation_frequency,
                              sc_threshold=sc_threshold, **kwargs)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def modulation_frequencies(self) -> np.ndarray:
        """Modulation frequencies present in this stream, in Hz."""
        freqs = self.data.freq.values
        return freqs[freqs != 0]

    def dc(self) -> xr.DataArray:
        """
        DC intensity component.

        Returns
        -------
        xr.DataArray
            Shape (time, channel, wavelength), real-valued.
        """
        return self.data.sel(freq=0).real

    def ac(self, freq: float = None) -> xr.DataArray:
        """
        AC amplitude at a modulation frequency.

        Parameters
        ----------
        freq : float, optional
            Modulation frequency in Hz. Selected automatically if only one
            non-zero frequency is present.

        Returns
        -------
        xr.DataArray
            Shape (time, channel, wavelength), real-valued.
        """
        freq = self._resolve_freq(freq)
        return np.abs(self.data.sel(freq=freq))

    def phase(self, freq: float = None) -> xr.DataArray:
        """
        Phase shift at a modulation frequency.

        Parameters
        ----------
        freq : float, optional
            Modulation frequency in Hz. Selected automatically if only one
            non-zero frequency is present.

        Returns
        -------
        xr.DataArray
            Shape (time, channel, wavelength), in radians.
        """
        freq = self._resolve_freq(freq)
        return np.angle(self.data.sel(freq=freq))

    def _resolve_freq(self, freq: float) -> float:
        """Resolve a modulation frequency, defaulting to the only non-zero one."""
        if freq is not None:
            return freq
        mod_freqs = self.modulation_frequencies
        if len(mod_freqs) == 1:
            return float(mod_freqs[0])
        raise ValueError(
            f"Multiple modulation frequencies found: {mod_freqs}. "
            "Specify which one with the 'freq' argument."
        )

    # ------------------------------------------------------------------
    # Modality conversion
    # ------------------------------------------------------------------

    def to_cw(self, inplace: bool = False) -> 'CW_Stream':
        """
        Collapse to the DC component, giving CW-equivalent data.

        CW is the zero-frequency slice of the frequency domain, so this is a
        modality conversion rather than an approximation. The AC measurement is
        discarded, and with it the phase information that separates absorption
        from scattering at a single distance.

        Parameters
        ----------
        inplace : bool
            Ignored; a new object is always returned, since the freq dimension is
            consumed and the dtype becomes real.

        Returns
        -------
        CW_Stream
            Shape (time, channel, wavelength), real-valued.
        """
        from .cw_nirs import CW_Stream

        cw_data = self.dc()
        cw_data.attrs = {**self.data.attrs, 'transformation': 'to_cw'}
        cw_data.attrs.pop('modulation_frequency_Hz', None)

        history = self.history.copy()
        history.append(self._history_entry('to_cw', {
            'source_modality': 'FD',
            'discarded_frequencies': [float(f) for f in self.modulation_frequencies],
        }))

        return CW_Stream(
            data=cw_data,
            probe=self.probe,
            name=self.name,
            events=self.events,
            status='raw',
            history=history,
        )

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    def slopefit_to_op(self,
                       distances: Optional[np.ndarray] = None,
                       n: float = 1.33,
                       freq: float = None,
                       inplace: bool = False) -> 'OptPropStream':
        """
        Derive optical properties by multi-distance slope fitting.

        Fits across source-detector distances at each wavelength and time point,
        using the large-distance asymptotic approximation. The channel dimension
        collapses to a single fitted location at the centroid of the sources.

        Parameters
        ----------
        distances : np.ndarray, optional
            Source-detector distances in cm. Read from ``data.coords['distance']``
            if omitted.
        n : float
            Tissue refractive index. Default 1.33. Semi-infinite geometry only;
            a two-layer model takes ``n_1`` and ``n_2`` through
            ``forward_parameters``.
        freq : float, optional
            Modulation frequency in Hz. Selected automatically if only one is
            present.
        inplace : bool
            Ignored; a new OptPropStream is always returned.

        Returns
        -------
        OptPropStream
            Shape (time, 1, wavelength, op) with op = ['mua', 'musp']. The single
            channel is the fitted spatial location.
        """
        from ..processing.frequency_domain import fd_slopefitting
        from .opt_prop_stream import OptPropStream

        freq = self._resolve_freq(freq)

        op_data, uncertainty_data = fd_slopefitting(
            self.data,
            distances=distances,
            n=n,
            modulation_frequency=freq
        )

        history = self.history.copy()
        history.append(self._history_entry('slopefit_to_op', {'n': n, 'modulation_frequency': freq}))

        return OptPropStream(
            data=op_data,
            uncertainty=uncertainty_data,
            name=self.name,
            events=self.events,
            status='op',
            history=history
        )

    def fit_to_op(self,
                  distances: Optional[np.ndarray] = None,
                  n: float = 1.33,
                  freq: float = None,
                  param_config: Optional[dict] = None,
                  fixed_params: Optional[dict] = None,
                  forward_model=None,
                  forward_parameters: Optional[dict] = None,
                  assemble=None,
                  residual_transform: str = "reference_channel",
                  n_starts: Optional[int] = None,
                  random_state: int = 0,
                  n_jobs: int = 1,
                  sigma: Optional[float] = None,
                  absolute_sigma: bool = False,
                  **residual_kwargs) -> 'OptPropStream':
        """
        Derive optical properties by nonlinear least-squares fitting.

        Fits the exact forward model at every distance, at the cost of an
        iterative optimisation per time point and wavelength.

        Parameters
        ----------
        distances : np.ndarray, optional
            Source-detector distances in cm. Read from ``data.coords['distance']``
            if omitted.
        n : float
            Tissue refractive index. Default 1.33. Semi-infinite geometry only;
            a two-layer model takes ``n_1`` and ``n_2`` through
            ``forward_parameters``.
        freq : float, optional
            Modulation frequency in Hz. Selected automatically if only one is
            present.
        param_config : dict, optional
            Parameters to fit and their bounds. The default fits ``mua`` and
            ``musp`` and applies only to the semi-infinite model; other geometries
            must supply their own.
        fixed_params : dict, optional
            Parameters held at a known value rather than fitted. Leave them in
            ``param_config`` as well; this only removes them from the free set.
        forward_model : callable, optional
            Model to fit. Defaults to :func:`~milob.forward.dos.si_fd_fluence`.
        forward_parameters : dict, optional
            Fixed context passed to ``forward_model`` alongside the resolved
            ``n``, ``wavelength`` and ``freq``.
        assemble : callable, optional
            Maps flat fitted parameter names onto the model's arguments. Not
            needed for the semi-infinite model.
        residual_transform : str or callable
            Transform applied to the residual before fitting. See
            :func:`~milob.processing.fitting.fd_model_opt`.
        n_starts : int, optional
            Number of multi-start attempts. Scales with the number of free
            parameters if None.
        random_state : int
            Seed for the multi-start sampler.
        n_jobs : int
            Parallel jobs across independent slices. Default 1, meaning serial.
        sigma : float, optional
            Known per-point noise level on the transformed residual. If None, the
            reported uncertainty is self-calibrated from each slice's residual.
        absolute_sigma : bool
            Anchor the uncertainty to ``sigma`` rather than self-calibrating.
            Default False.
        **residual_kwargs
            Forwarded to the residual transform.

        Returns
        -------
        OptPropStream
            Shape (time, 1, wavelength, op), with op taken from ``param_config``
            and defaulting to ['mua', 'musp']. The single channel is the fitted
            spatial location.

        References
        ----------
        Martins, G. G., Forti, R. M., & Mesquita, R. C. (2025). Spectroscopy
        Journal, 3, 14.
        """
        from ..processing.fitting import fd_model_opt, run_parallel_fits
        from ..forward.dos import si_fd_fluence
        from .opt_prop_stream import (OptPropStream, pack_fit_arrays,
                                      pack_covariance, fit_attrs)

        if forward_model is None:
            forward_model = si_fd_fluence

        if param_config is None:
            param_config = {
                "mua": {"bounds": (1e-3, 0.5), "log": True},
                "musp": {"bounds": (1.0, 30.0), "log": True},
            }

        freq = self._resolve_freq(freq)

        if distances is None:
            distances = self.data.coords['distance'].values
            if self.data.attrs.get('lengthUnit') == 'mm':
                distances = distances / 10.0
        distances = np.asarray(distances, dtype=float)

        # Plain array, not the xarray object -- lightweight to close over
        # when n_jobs != 1 dispatches each slice to a separate process.
        complex_arr = self.data.sel(freq=freq).values  # (time, channel, wavelength)

        time_vals = self.data.time.values
        wavelength_vals = self.data.wavelength.values
        n_time = len(time_vals)
        n_wl = len(wavelength_vals)

        param_names = list(param_config.keys())
        n_p = len(param_names)
        op_array = np.full((n_time, 1, n_wl, n_p), np.nan)
        err_array = np.full_like(op_array, np.nan)
        cov_array = np.full((n_time, 1, n_wl, n_p, n_p), np.nan)

        slices = [(ti, wi, wl) for ti in range(n_time) for wi, wl in enumerate(wavelength_vals)]
        # n_jobs parallelises whichever loop actually has more than one
        # thing to do: across slices when there's more than one (the usual
        # case), or -- since that would otherwise leave n_jobs with nothing
        # to parallelise -- into fd_model_opt's own multi-start loop when
        # there's only a single slice (e.g. one time point, one wavelength).
        # Never both at once: nesting two independent n_jobs=-1 dispatches
        # would try to oversubscribe to (cores)^2 worker processes.
        single_slice = len(slices) == 1
        outer_n_jobs = 1 if single_slice else n_jobs
        inner_n_jobs = n_jobs if single_slice else 1

        def _fit_slice(ti, wi, wl):
            # Conjugate: stored data uses the library/instrument phase
            # convention (positive imaginary part = positive phase delay,
            # see simulate_fd_stream), while si_fd_fluence/forward.dos
            # return the physics-convention fluence (negative imaginary
            # part for a delayed signal) -- the same conjugation
            # simulate_fd_stream applies when generating this data.
            y = np.conj(complex_arr[ti, :, wi])

            slice_forward_parameters = {"n": n, "wavelength": float(wl), "freq": freq}
            if forward_parameters:
                slice_forward_parameters.update(forward_parameters)

            return fd_model_opt(
                rho=distances,
                data=y,
                forward_model=forward_model,
                forward_parameters=slice_forward_parameters,
                param_config=param_config,
                fixed_params=fixed_params,
                assemble=assemble,
                residual_transform=residual_transform,
                n_starts=n_starts,
                random_state=random_state,
                n_jobs=inner_n_jobs,
                sigma=sigma,
                absolute_sigma=absolute_sigma,
                return_covariance=True,
                **residual_kwargs,
            )

        jobs = [(_fit_slice, (ti, wi, wl)) for ti, wi, wl in slices]
        results = run_parallel_fits(jobs, n_jobs=outer_n_jobs)

        for (ti, wi, wl), (result, error, cov) in zip(slices, results):
            for pi, name in enumerate(param_names):
                op_array[ti, 0, wi, pi] = result.get(name, np.nan)
                err_array[ti, 0, wi, pi] = error.get(name, np.nan)
            free = [n for n in param_names if n not in (fixed_params or {})]
            idx = [param_names.index(n) for n in free]
            cov_array[np.ix_([ti], [0], [wi], idx, idx)] = cov

        op_da, err_da, obs_da, obs_err_da = pack_fit_arrays(
            op_array, err_array, param_names,
            coords={'time': time_vals,
                    'channel': ['fitted_location'],
                    'wavelength': wavelength_vals},
            dims=['time', 'channel', 'wavelength'],
        )

        cov_da = pack_covariance(
            cov_array, param_names,
            coords={'time': time_vals,
                    'channel': ['fitted_location'],
                    'wavelength': wavelength_vals},
            dims=['time', 'channel', 'wavelength'],
        )

        op_da.attrs.update(fit_attrs(
            fitting_model=getattr(forward_model, '__name__', 'custom'),
            transformation='fit_to_op',
            length_unit='cm',            # distances were converted above
            modulation_frequency_Hz=freq,
            refractive_index=n,
        ))

        history = self.history.copy()
        history.append(self._history_entry('fit_to_op', {
            'n': n,
            'modulation_frequency': freq,
            'param_config': dict(param_config),
            'fixed_params': dict(fixed_params) if fixed_params else {},
            'forward_parameters': dict(forward_parameters) if forward_parameters else {},
            'forward_model': getattr(forward_model, '__name__', 'custom'),
            'residual_transform': residual_transform if isinstance(residual_transform, str) else 'custom',
            'n_starts': n_starts,
        }))

        return OptPropStream(
            data=op_da,
            uncertainty=err_da,
            obs_params=obs_da,
            obs_uncertainty=obs_err_da,
            covariance=cov_da,
            forward_parameters=dict(forward_parameters) if forward_parameters else {},
            name=self.name,
            events=self.events,
            status='op',
            history=history,
        )

    def fit_to_conc(self,
                     distances: Optional[np.ndarray] = None,
                     n: float = 1.33,
                     freq: float = None,
                     param_config: Optional[dict] = None,
                     fixed_params: Optional[dict] = None,
                     forward_model=None,
                     forward_parameters: Optional[dict] = None,
                     assemble=None,
                     residual_transform: str = "reference_channel",
                     n_starts: Optional[int] = None,
                     random_state: int = 0,
                     n_jobs: int = 1,
                     sigma: Optional[float] = None,
                     absolute_sigma: bool = False,
                     **residual_kwargs) -> 'TissueStream':
        """
        Fit tissue composition and scattering shape directly from the raw data.

        All wavelengths are fitted simultaneously in one nonlinear least-squares
        problem, with HbO, HbR and the scattering power-law parameters A and b as
        shared free parameters. This constrains composition more tightly than
        fitting each wavelength's optical properties and unmixing afterwards.

        Parameters
        ----------
        distances : np.ndarray, optional
            Source-detector distances in cm. Read from ``data.coords['distance']``
            if omitted.
        n : float
            Tissue refractive index. Default 1.33. Semi-infinite geometry only;
            a two-layer model takes ``n_1`` and ``n_2`` through
            ``forward_parameters``.
        freq : float, optional
            Modulation frequency in Hz. Selected automatically if only one is
            present.
        param_config : dict, optional
            Parameters to fit and their bounds. Defaults to
            :data:`~milob.forward.spectral.TISSUE_FD_PARAM_CONFIG`, fitting HbO,
            HbR, A and b.
        fixed_params : dict, optional
            Parameters held at a known value rather than fitted. Leave them in
            ``param_config`` as well; this only removes them from the free set.
        forward_model : callable, optional
            Model to fit. Defaults to :func:`~milob.forward.dos.si_fd_fluence`.
        forward_parameters : dict, optional
            Fixed context passed to ``forward_model`` alongside the resolved
            ``n``, ``wavelength`` and ``freq``.
        assemble : callable, optional
            Maps flat fitted parameter names onto the model's arguments. Not
            needed for the semi-infinite model.
        residual_transform : str or callable
            Transform applied to the residual before fitting. See
            :func:`~milob.processing.fitting.fd_model_opt`.
        n_starts : int, optional
            Number of multi-start attempts. Scales with the number of free
            parameters if None.
        random_state : int
            Seed for the multi-start sampler.
        n_jobs : int
            Parallel jobs across independent slices. Default 1, meaning serial.
        sigma : float, optional
            Known per-point noise level on the transformed residual. If None, the
            reported uncertainty is self-calibrated from each slice's residual.
        absolute_sigma : bool
            Anchor the uncertainty to ``sigma`` rather than self-calibrating.
            Default False.
        **residual_kwargs
            Forwarded to the residual transform.

        Returns
        -------
        TissueStream
            Shape (time, 1, component), with component taken from ``param_config``
            and defaulting to ['HbO', 'HbR', 'A', 'b']. The single channel is the
            fitted spatial location, and ``uncertainty`` carries the propagated
            one-sigma fit uncertainty.

        Raises
        ------
        ValueError
            If fewer than two wavelengths are present, since composition is
            unidentifiable from one wavelength however many distances are
            measured.

        Notes
        -----
        The scattering exponent ``b`` is weakly constrained with only two
        wavelengths; its leverage grows with their separation in log-wavelength
        space. Check its propagated uncertainty before relying on it.
        """
        from ..processing.fitting import fd_model_opt, run_parallel_fits
        from ..forward.dos import si_fd_fluence
        from ..forward.spectral import assemble_spectral_fd, extinction_at, TISSUE_FD_PARAM_CONFIG
        from .tissue_stream import TissueStream

        if forward_model is None:
            forward_model = si_fd_fluence
        if assemble is None:
            assemble = assemble_spectral_fd
        if param_config is None:
            param_config = TISSUE_FD_PARAM_CONFIG

        freq = self._resolve_freq(freq)

        if distances is None:
            distances = self.data.coords['distance'].values
            if self.data.attrs.get('lengthUnit') == 'mm':
                distances = distances / 10.0
        distances = np.asarray(distances, dtype=float)

        # Plain array, not the xarray object -- lightweight to close over
        # when n_jobs != 1 dispatches each time point to a separate process.
        complex_arr = self.data.sel(freq=freq).values  # (time, channel, wavelength)

        time_vals = self.data.time.values
        wavelength_vals = self.data.wavelength.values
        n_time = len(time_vals)
        n_wl = len(wavelength_vals)

        if n_wl < 2:
            raise ValueError(
                f"fit_to_conc requires at least 2 wavelengths to identify "
                f"composition and scattering shape -- got {n_wl} "
                f"({list(wavelength_vals)}). A single wavelength gives 1 "
                f"equation (mua(lambda)) for 2 unknowns (HbO, HbR) no matter "
                f"how many channels/distances are measured; same for "
                f"musp(lambda) = A*(lambda/lambda_0)^-b's A/b. Use "
                f"fit_to_op() + OptPropStream.to_concentration() instead for "
                f"single-wavelength data."
            )

        param_names = list(param_config.keys())

        # fit_to_conc's contract is composition in, composition out -- one
        # space, one stream. A param_config that also carries theta or
        # observation parameters (a partially spectral fit: HbO/HbR with a
        # free musp) recovers quantities in more than one space, and there is
        # no single stream that can hold them without filing an optical
        # property as though it were a tissue-level cause. Refuse rather than
        # mis-file: silently putting 'musp' on the component axis of a
        # SPACE='c' stream is what this check exists to prevent.
        from ..forward.spectral import split_params_by_space
        _, medium, obs = split_params_by_space(param_names)
        if medium or obs:
            raise ValueError(
                f"fit_to_conc() recovers composition only, but param_config "
                f"also contains {medium + obs}, which are not tissue-level "
                f"causes: {medium} are optical properties (theta) and {obs} "
                f"are observation parameters. Storing them on the "
                f"'component' axis would file them as composition. Use "
                f"fit_joint(), which returns (tissue, optical) so each "
                f"parameter lands in its own space."
            )

        tissue_array = np.full((n_time, 1, len(param_names)), np.nan)
        err_array = np.full_like(tissue_array, np.nan)

        # See fit_to_op's identical comment: route n_jobs into whichever
        # loop actually has more than one thing to do.
        single_slice = n_time == 1
        outer_n_jobs = 1 if single_slice else n_jobs
        inner_n_jobs = n_jobs if single_slice else 1

        def _fit_slice(ti):
            rho_blocks, data_blocks, forward_parameters_blocks = [], [], []
            for wi, wl in enumerate(wavelength_vals):
                # Same conjugation as fit_to_op -- see its comment for the
                # phase-convention rationale.
                y = np.conj(complex_arr[ti, :, wi])
                eps_hbo, eps_hbr = extinction_at(float(wl))

                slice_forward_parameters = {
                    "n": n, "wavelength": float(wl), "freq": freq,
                    "eps_hbo": eps_hbo, "eps_hbr": eps_hbr,
                }
                if forward_parameters:
                    slice_forward_parameters.update(forward_parameters)

                rho_blocks.append(distances)
                data_blocks.append(y)
                forward_parameters_blocks.append(slice_forward_parameters)

            return fd_model_opt(
                rho=rho_blocks,
                data=data_blocks,
                forward_model=forward_model,
                forward_parameters=forward_parameters_blocks,
                param_config=param_config,
                fixed_params=fixed_params,
                assemble=assemble,
                residual_transform=residual_transform,
                n_starts=n_starts,
                random_state=random_state,
                n_jobs=inner_n_jobs,
                sigma=sigma,
                absolute_sigma=absolute_sigma,
                return_covariance=True,
                **residual_kwargs,
            )

        jobs = [(_fit_slice, (ti,)) for ti in range(n_time)]
        results = run_parallel_fits(jobs, n_jobs=outer_n_jobs)

        n_p = len(param_names)
        cov_array = np.full((n_time, 1, n_p, n_p), np.nan)
        for ti, (result, error, cov) in enumerate(results):
            for pi, name in enumerate(param_names):
                tissue_array[ti, 0, pi] = result.get(name, np.nan)
                err_array[ti, 0, pi] = error.get(name, np.nan)
            free = [nm for nm in param_names if nm not in (fixed_params or {})]
            idx = [param_names.index(nm) for nm in free]
            cov_array[np.ix_([ti], [0], idx, idx)] = cov

        tissue_coords = {
            'time': time_vals,
            'channel': ['fitted_location'],
            'component': param_names,
        }

        tissue_da = xr.DataArray(
            tissue_array, coords=tissue_coords, dims=['time', 'channel', 'component'],
            attrs={'units': 'uM (HbO/HbR), cm^-1 (A), dimensionless (b)',
                   'transformation': 'fit_to_conc',
                   'modulation_frequency_Hz': freq, 'refractive_index': n},
        )
        err_da = xr.DataArray(
            err_array, coords=tissue_coords, dims=['time', 'channel', 'component'],
            attrs={'description': '1-sigma propagated uncertainty'},
        )
        # HbO and HbR come out strongly anti-correlated from a joint
        # spectral fit, so the off-diagonal matters for anything derived
        # from both (HbT, StO2) -- keep it rather than only the diagonal.
        cov_da = xr.DataArray(
            cov_array,
            coords={'time': time_vals, 'channel': ['fitted_location'],
                    'param': param_names, 'param_2': param_names},
            dims=['time', 'channel', 'param', 'param_2'],
            attrs={'description': 'parameter covariance from the fit'},
        )

        history = self.history.copy()
        history.append(self._history_entry('fit_to_conc', {
            'n': n,
            'modulation_frequency': freq,
            'wavelengths': [float(w) for w in wavelength_vals],
            'param_config': dict(param_config),
            'fixed_params': dict(fixed_params) if fixed_params else {},
            'forward_parameters': dict(forward_parameters) if forward_parameters else {},
            'forward_model': getattr(forward_model, '__name__', 'custom'),
            'assemble': getattr(assemble, '__name__', 'custom'),
            'residual_transform': residual_transform if isinstance(residual_transform, str) else 'custom',
            'n_starts': n_starts,
        }))

        return TissueStream(
            data=tissue_da,
            uncertainty=err_da,
            covariance=cov_da,
            name=self.name,
            events=self.events,
            status='absolute_conc',
            history=history,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def _info_lines(self):
        lines = super()._info_lines()
        lines += ["", "--- FD Configuration ---"]
        freqs = self.data.freq.values
        dc_present = 0 in freqs
        mod_freqs = freqs[freqs != 0]
        mod_str = ", ".join(
            f"{f / 1e6:.0f} MHz" if f >= 1e6 else f"{f:.0f} Hz"
            for f in mod_freqs
        ) if len(mod_freqs) > 0 else "none"
        lines += [f"DC:  {'yes' if dc_present else 'no'}", f"AC:  {mod_str}"]
        return lines

    def __repr__(self):
        freqs = list(self.data.freq.values)
        return (
            f"<FD_Stream | {self.name} | Status: {self.status} | "
            f"Channels: {len(self.data.channel)} | "
            f"Frequencies (Hz): {freqs} | "
            f"Duration: {len(self.data.time)} frames>"
        )
