from milob.core.cw_nirs import CW_Stream
from milob.core.opt_prop_stream import OptPropStream
from milob.core.tissue_stream import TissueStream
from .datastream import Datastream
from .nirs import NirsStream
from ..processing import time_domain
from ..viz import timedomain as viz_timedomain
import copy
import numpy as np
import xarray as xr

class TD_Stream(NirsStream):
    """
    Time-domain diffuse optical spectroscopy data.

    Data has dims (time, channel, wavelength, bin) when ``status='raw'``,
    holding gated photon counts with arrival times in ``coords['timeDelays']``,
    or (time, channel, wavelength, moment) when ``status='moment'``, holding
    [m0, m1, m2]: total intensity, mean arrival time and variance.

    Parameters
    ----------
    data : xr.DataArray
        Shape (time, channel, wavelength, bin) or (..., moment).
    probe : Probe
        Optode geometry.
    """

    def __init__(self, data, probe, **kwargs):
        super().__init__(data, probe, **kwargs)

    # ------------------------------------------------------------------
    # TD to CW modality conversion
    # ------------------------------------------------------------------

    def to_cw(self) -> 'CW_Stream':
        """
        Collapse the time-domain axis to give CW-equivalent data.

        Raw gated data is summed over bins; moment data selects m0.

        Returns
        -------
        CW_Stream
            Shape (time, channel, wavelength).
        """
        from .cw_nirs import CW_Stream

        cw_data, meta = time_domain.convert_to_cw(datastream=self)

        return CW_Stream(
            data=cw_data,
            probe=self.probe,
            name=meta['name'],
            events=meta['events'],
            status=meta['status'],
            history=meta['history']
        )

    # ------------------------------------------------------------------
    # Moment computation
    # ------------------------------------------------------------------

    def to_moments(self, sheppards_correction=True, inplace: bool = False) -> 'TD_Stream':
        """
        Compute the moments [m0, m1, m2] from gated TPSF data.

        Replaces the 'bin' dimension with a 'moment' dimension.

        Parameters
        ----------
        sheppards_correction : bool
            Apply Sheppard's correction for the finite gate width. Default True.
        inplace : bool
            Modify this stream in place. Default False.

        Returns
        -------
        TD_Stream
            Stream with ``status='moment'``.
        """
        moments_data, meta = time_domain.calculate_moments(self, sheppards_corr=sheppards_correction)

        if inplace:
            self.data = moments_data
            self.name = meta['name']
            self.history = meta['history']
            self.status = 'moment'
            return self

        return TD_Stream(
            data=moments_data,
            probe=self.probe,
            name=meta['name'],
            events=meta['events'],
            status='moment',
            history=meta['history']
        )

    def select_moment(self, mom: str) -> 'TD_Stream':
        """
        Select a single moment.

        Parameters
        ----------
        mom : str
            Moment label, one of 'm0', 'm1' or 'm2'.

        Returns
        -------
        TD_Stream
            Shape (time, channel, wavelength, 1).
        """
        if self.status != 'moment':
            raise ValueError(
                f"select_moment() requires status='moment', got '{self.status}'. "
                "Call to_moments() first."
            )
        if mom not in self.data.moment.values:
            raise ValueError(
                f"Moment '{mom}' not found. Available: {list(self.data.moment.values)}"
            )

        history = self.history.copy()
        history.append(self._history_entry('select_moment', {'moment': mom}))

        return TD_Stream(
            data=self.data.sel(moment=[mom]),
            probe=self.probe,
            events=self.events,
            name=self.name,
            status=self.status,
            history=history
        )
    

    def correct_moments_for_irf(self, irf_stream, inplace: bool = False) -> 'TD_Stream':
        """
        Correct measured moments for the instrument response function.

        Parameters
        ----------
        irf_stream : TD_Stream
            Stream holding the measured IRF, in moment form.
        inplace : bool
            Modify this stream in place. Default False.

        Returns
        -------
        TD_Stream
            Stream with ``status='moment'`` and IRF-corrected moments.
        """

        moments_corr, meta = time_domain.apply_irf_correction(self, irf_stream)

        if inplace:
            self.data = moments_corr
            self.name = meta['name']
            self.history = meta['history']
            self.status = 'moment'
            return self

        return TD_Stream(
            data=moments_corr,
            probe=self.probe,
            name=meta['name'],
            events=meta['events'],
            status='moment',
            history=meta['history']
        )

    # ------------------------------------------------------------------
    # Optical property inversion
    # ------------------------------------------------------------------

    def moments_to_optical_params(self, n: float = 1.37) -> 'OptPropStream':
        """
        Derive optical properties from TD moments in closed form.

        Uses the semi-infinite homogeneous diffusion relations, consuming the
        'moment' dimension and producing an 'op' dimension. Requires
        ``status='moment'``.

        For general-geometry form, see 'fit_to_op_moments'.

        Parameters
        ----------
        n : float
            Refractive index of the medium. Default 1.37.

        Returns
        -------
        OptPropStream
            Shape (time, channel, wavelength, op) with op = ['mua', 'musp'], in
            cm^-1.

        References
        ----------
        Liebert, A. et al. (2003). Applied Optics, 42(28), 5785-5792.
        """
        from .opt_prop_stream import OptPropStream

        optical_data, meta = time_domain.calculate_optical_properties_moments(self, n=n)

        return OptPropStream(
            data=optical_data,
            probe=self.probe,
            name=meta["name"],
            events=meta["events"],
            status="optical",
            history=meta["history"]
        )
    
    def fit_to_op(self,
                  distances=None,
                  n: float = 1.37,
                  param_config: dict = None,
                  fixed_params: dict = None,
                  forward_model=None,
                  forward_parameters: dict = None,
                  assemble=None,
                  observation: str = "identity",
                  irf=None,
                  n_starts: int = None,
                  random_state: int = 0,
                  n_jobs: int = 1,
                  sigma: float = None,
                  absolute_sigma: bool = False,
                  log_fit: bool = True) -> 'OptPropStream':
        """
        Derive optical properties by nonlinear least-squares fitting of the TPSF.

        Requires ``status='raw'``. A single TPSF carries enough information to fit
        absorption, scattering and an amplitude scale from one channel, so every
        channel is fitted independently and the channel dimension is preserved.

        Parameters
        ----------
        distances : np.ndarray, optional
            Source-detector distances in cm. Read from ``data.coords['distance']``
            if omitted.
        n : float
            Refractive index. Default 1.37.
        param_config : dict, optional
            Parameters to fit and their bounds. Defaults to fitting ``mua``,
            ``musp`` and an amplitude scale ``A``.
        fixed_params : dict, optional
            Parameters held at a known value rather than fitted.
        forward_model : callable, optional
            Model to fit. Defaults to
            :func:`~milob.forward.dos.si_td_fluence_patterson`, the closed form.
            :func:`~milob.forward.dos.si_td_fluence` is the general FD-sweep
            equivalent and is far slower per evaluation.
        forward_parameters : dict, optional
            Fixed context passed to ``forward_model`` alongside the resolved ``n``
            and ``wavelength``.
        assemble : callable, optional
            Maps flat fitted parameter names onto the model's arguments. Not
            needed for the semi-infinite model.
        observation : {'identity', 'convolve_irf'} or callable
            'identity' (default) for an already-deconvolved TPSF, or
            'convolve_irf' for a raw one, which requires ``irf``.
        irf : array-like or callable, optional
            Instrument response, required when ``observation='convolve_irf'``.
        n_starts : int, optional
            Number of multi-start attempts.
        random_state : int
            Seed for the multi-start sampler.
        n_jobs : int
            Parallel jobs across independent slices. Default 1.
        sigma : float, optional
            Known per-point noise level. If None, uncertainty is self-calibrated
            from each slice's residual.
        absolute_sigma : bool
            Anchor the uncertainty to ``sigma``. Default False.
        log_fit : bool
            Fit the logarithm of the TPSF rather than its amplitude.

        Returns
        -------
        OptPropStream
            Shape (time, channel, wavelength, op), with op taken from
            ``param_config`` and defaulting to ['mua', 'musp', 'A'].
        """
        from ..processing.fitting import td_model_opt, run_parallel_fits
        from ..forward.dos import si_td_fluence_patterson
        from .opt_prop_stream import (OptPropStream, pack_fit_arrays,
                                      pack_covariance, fit_attrs)

        if self.status != 'raw' or 'bin' not in self.data.dims:
            raise ValueError(
                f"fit_to_op requires status='raw' gated TPSF data (a 'bin' "
                f"dimension) -- got status='{self.status}'."
            )

        if forward_model is None:
            forward_model = si_td_fluence_patterson
        if param_config is None:
            # A's bounds are deliberately enormous (rather than a
            # "reasonable-looking" tighter range): the forward models in
            # forward.dos return an unnormalised "unit source" fluence
            # whose absolute scale bears no fixed relationship to a real
            # instrument's raw count scale (or to the arbitrary units this
            # package's own simulate_*_td_stream happens to produce), so A
            # has to be free to land anywhere from a small fraction to many
            # orders of magnitude above 1.
            param_config = {
                "mua": {"bounds": (1e-3, 0.5), "log": True},
                "musp": {"bounds": (1.0, 30.0), "log": True},
                "A": {"bounds": (1e-30, 1e20), "log": True},
            }

        bin_delays = self.data.coords['timeDelays'].values
        bin_widths = self.data.coords['timeDelayWidths'].values
        t = np.asarray(bin_delays, dtype=float) + np.asarray(bin_widths, dtype=float) / 2.0

        if distances is None:
            distances = self.data.coords['distance'].values
            if self.data.attrs.get('lengthUnit') == 'mm':
                distances = distances / 10.0
        distances = np.asarray(distances, dtype=float)

        data_arr = self.data.values  # (time, channel, wavelength, bin)
        time_vals = self.data.time.values
        wavelength_vals = self.data.wavelength.values
        channel_vals = self.data.channel.values
        n_time, n_ch, n_wl = len(time_vals), len(channel_vals), len(wavelength_vals)

        param_names = list(param_config.keys())
        n_p = len(param_names)
        op_array = np.full((n_time, n_ch, n_wl, n_p), np.nan)
        err_array = np.full_like(op_array, np.nan)
        cov_array = np.full((n_time, n_ch, n_wl, n_p, n_p), np.nan)

        slices = [(ti, ci, wi) for ti in range(n_time) for ci in range(n_ch) for wi in range(n_wl)]
        # Same n_jobs routing convention as FD_Stream.fit_to_op(): parallelise
        # whichever loop has more than one thing to do.
        single_slice = len(slices) == 1
        outer_n_jobs = 1 if single_slice else n_jobs
        inner_n_jobs = n_jobs if single_slice else 1

        def _fit_slice(ti, ci, wi):
            y = data_arr[ti, ci, wi, :]
            rho = float(distances[ci])
            wl = float(wavelength_vals[wi])

            slice_forward_parameters = {"n": n, "wavelength": wl}
            if forward_parameters:
                slice_forward_parameters.update(forward_parameters)

            return td_model_opt(
                t=t, rho=rho, data=y,
                forward_model=forward_model,
                forward_parameters=slice_forward_parameters,
                param_config=param_config,
                fixed_params=fixed_params,
                assemble=assemble,
                observation=observation,
                irf=irf,
                n_starts=n_starts,
                random_state=random_state,
                n_jobs=inner_n_jobs,
                sigma=sigma,
                absolute_sigma=absolute_sigma,
                return_covariance=True,
                log_fit=log_fit,
            )

        jobs = [(_fit_slice, s) for s in slices]
        results = run_parallel_fits(jobs, n_jobs=outer_n_jobs)

        for (ti, ci, wi), (result, error, cov) in zip(slices, results):
            for pi, pname in enumerate(param_names):
                op_array[ti, ci, wi, pi] = result.get(pname, np.nan)
                err_array[ti, ci, wi, pi] = error.get(pname, np.nan)
            free = [nm for nm in param_names if nm not in (fixed_params or {})]
            idx = [param_names.index(nm) for nm in free]
            cov_array[np.ix_([ti], [ci], [wi], idx, idx)] = cov

        op_da, err_da, obs_da, obs_err_da = pack_fit_arrays(
            op_array, err_array, param_names,
            coords={'time': time_vals,
                    'channel': channel_vals,
                    'wavelength': wavelength_vals},
            dims=['time', 'channel', 'wavelength'],
        )
        cov_da = pack_covariance(
            cov_array, param_names,
            coords={'time': time_vals,
                    'channel': channel_vals,
                    'wavelength': wavelength_vals},
            dims=['time', 'channel', 'wavelength'],
        )

        op_da.attrs.update(fit_attrs(
            fitting_model=getattr(forward_model, '__name__', 'custom'),
            transformation='fit_to_op',
            length_unit='cm',            # distances were converted above
            refractive_index=n,
        ))

        history = self.history.copy()
        history.append(self._history_entry('fit_to_op', {
            'n': n,
            'param_config': dict(param_config),
            'fixed_params': dict(fixed_params) if fixed_params else {},
            'forward_model': getattr(forward_model, '__name__', 'custom'),
            'observation': observation if isinstance(observation, str) else 'custom',
            'n_starts': n_starts,
        }))

        return OptPropStream(
            data=op_da,
            uncertainty=err_da,
            obs_params=obs_da,
            obs_uncertainty=obs_err_da,
            covariance=cov_da,
            probe=self.probe,
            name=self.name,
            events=self.events,
            status='op',
            history=history,
        )

    def fit_to_op_moments(self,
                           distances=None,
                           n: float = 1.37,
                           t=None,
                           param_config: dict = None,
                           fixed_params: dict = None,
                           forward_model=None,
                           forward_parameters: dict = None,
                           assemble=None,
                           n_starts: int = None,
                           random_state: int = 0,
                           n_jobs: int = 1,
                           sigma: float = None,
                           absolute_sigma: bool = False,
                           moments_used=("m1", "m2")) -> 'OptPropStream':
        """
        Derive optical properties by fitting model moments to measured moments.

        The general-geometry counterpart of :meth:`moments_to_optical_params`,
        which solves the semi-infinite case in closed form. Requires
        ``status='moment'``; call :meth:`to_moments` and, where an IRF was
        measured, :meth:`correct_moments_for_irf` first.

        Parameters
        ----------
        distances : np.ndarray, optional
            Source-detector distances in cm. Read from ``data.coords['distance']``
            if omitted.
        n : float
            Refractive index. Default 1.37.
        t : array-like, optional
            Time grid in seconds on which the model TPSF's moments are integrated.
            Defaults to 3000 points spanning 0 to 6 ns.
        param_config : dict, optional
            Parameters to fit and their bounds. Defaults to ``mua`` and ``musp``.
        fixed_params : dict, optional
            Parameters held at a known value rather than fitted.
        forward_model : callable, optional
            Model whose moments are fitted.
        forward_parameters : dict, optional
            Fixed context passed to ``forward_model``.
        assemble : callable, optional
            Maps flat fitted parameter names onto the model's arguments.
        n_starts : int, optional
            Number of multi-start attempts.
        random_state : int
            Seed for the multi-start sampler.
        n_jobs : int
            Parallel jobs across independent slices. Default 1.
        sigma : float, optional
            Known per-point noise level on the moments.
        absolute_sigma : bool
            Anchor the uncertainty to ``sigma``. Default False.
        moments_used : tuple of str
            Which moments to fit. Default ('m1', 'm2').

        Returns
        -------
        OptPropStream
            Shape (time, channel, wavelength, op), with op taken from
            ``param_config`` and defaulting to ['mua', 'musp'].
        """
        from ..processing.fitting import td_moments_model_opt, run_parallel_fits
        from ..forward.dos import si_td_fluence_patterson
        from .opt_prop_stream import (OptPropStream, pack_fit_arrays,
                                      pack_covariance, fit_attrs)

        if self.status != 'moment':
            raise ValueError(
                f"fit_to_op_moments requires status='moment' data -- got "
                f"status='{self.status}'. Call to_moments() (and, with a "
                f"measured IRF, correct_moments_for_irf()) first."
            )

        if forward_model is None:
            forward_model = si_td_fluence_patterson
        if param_config is None:
            param_config = {
                "mua": {"bounds": (1e-3, 0.5), "log": True},
                "musp": {"bounds": (1.0, 30.0), "log": True},
            }
        if t is None:
            t = np.linspace(1e-12, 6e-9, 3000)
        t = np.asarray(t, dtype=float)

        if distances is None:
            distances = self.data.coords['distance'].values
            if self.data.attrs.get('lengthUnit') == 'mm':
                distances = distances / 10.0
        distances = np.asarray(distances, dtype=float)

        m0_arr = self.data.sel(moment='m0').values
        m1_arr = self.data.sel(moment='m1').values
        m2_arr = self.data.sel(moment='m2').values

        time_vals = self.data.time.values
        wavelength_vals = self.data.wavelength.values
        channel_vals = self.data.channel.values
        n_time, n_ch, n_wl = len(time_vals), len(channel_vals), len(wavelength_vals)

        param_names = list(param_config.keys())
        n_p = len(param_names)
        op_array = np.full((n_time, n_ch, n_wl, n_p), np.nan)
        err_array = np.full_like(op_array, np.nan)
        cov_array = np.full((n_time, n_ch, n_wl, n_p, n_p), np.nan)

        slices = [(ti, ci, wi) for ti in range(n_time) for ci in range(n_ch) for wi in range(n_wl)]
        single_slice = len(slices) == 1
        outer_n_jobs = 1 if single_slice else n_jobs
        inner_n_jobs = n_jobs if single_slice else 1

        def _fit_slice(ti, ci, wi):
            moments = {
                "m0": float(m0_arr[ti, ci, wi]),
                "m1": float(m1_arr[ti, ci, wi]),
                "m2": float(m2_arr[ti, ci, wi]),
            }
            rho = float(distances[ci])
            wl = float(wavelength_vals[wi])

            slice_forward_parameters = {"n": n, "wavelength": wl}
            if forward_parameters:
                slice_forward_parameters.update(forward_parameters)

            return td_moments_model_opt(
                t=t, rho=rho, moments=moments,
                forward_model=forward_model,
                forward_parameters=slice_forward_parameters,
                param_config=param_config,
                fixed_params=fixed_params,
                assemble=assemble,
                n_starts=n_starts,
                random_state=random_state,
                n_jobs=inner_n_jobs,
                sigma=sigma,
                absolute_sigma=absolute_sigma,
                return_covariance=True,
                moments_used=moments_used,
            )

        jobs = [(_fit_slice, s) for s in slices]
        results = run_parallel_fits(jobs, n_jobs=outer_n_jobs)

        for (ti, ci, wi), (result, error, cov) in zip(slices, results):
            for pi, pname in enumerate(param_names):
                op_array[ti, ci, wi, pi] = result.get(pname, np.nan)
                err_array[ti, ci, wi, pi] = error.get(pname, np.nan)
            free = [nm for nm in param_names if nm not in (fixed_params or {})]
            idx = [param_names.index(nm) for nm in free]
            cov_array[np.ix_([ti], [ci], [wi], idx, idx)] = cov

        op_da, err_da, obs_da, obs_err_da = pack_fit_arrays(
            op_array, err_array, param_names,
            coords={'time': time_vals,
                    'channel': channel_vals,
                    'wavelength': wavelength_vals},
            dims=['time', 'channel', 'wavelength'],
        )
        cov_da = pack_covariance(
            cov_array, param_names,
            coords={'time': time_vals,
                    'channel': channel_vals,
                    'wavelength': wavelength_vals},
            dims=['time', 'channel', 'wavelength'],
        )

        op_da.attrs.update(fit_attrs(
            fitting_model=getattr(forward_model, '__name__', 'custom'),
            transformation='fit_to_op_moments',
            length_unit='cm',            # distances were converted above
            refractive_index=n,
        ))

        history = self.history.copy()
        history.append(self._history_entry('fit_to_op_moments', {
            'n': n,
            'param_config': dict(param_config),
            'moments_used': list(moments_used),
            'forward_model': getattr(forward_model, '__name__', 'custom'),
            'n_starts': n_starts,
        }))

        return OptPropStream(
            data=op_da,
            uncertainty=err_da,
            obs_params=obs_da,
            obs_uncertainty=obs_err_da,
            covariance=cov_da,
            probe=self.probe,
            name=self.name,
            events=self.events,
            status='op',
            history=history,
        )

    def fit_to_conc(self, water_fraction: float = 0.75, **fit_to_op_kwargs) -> 'TissueStream':
        """
        Derive chromophore concentrations from TD-DOS data.

        Fits optical properties per wavelength with :meth:`fit_to_op`, then unmixes
        them across wavelengths.

        Parameters
        ----------
        water_fraction : float
            Water volume fraction subtracted from absorption before unmixing.
        **fit_to_op_kwargs
            Forwarded to :meth:`fit_to_op`.

        Returns
        -------
        TissueStream
            Shape (time, channel, component) with component = ['HbO', 'HbR'].
        """
        op_stream = self.fit_to_op(**fit_to_op_kwargs)
        return op_stream.to_concentration(water_fraction=water_fraction)


    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_td_moments(self, channel_name=None, wl_indices=None):
        """
        Plot the moments m0, m1 and m2 over time for one channel.

        Parameters
        ----------
        channel_name : str, optional
            Channel label. Defaults to the first channel.
        wl_indices : list of int, optional
            Wavelength indices to plot. Defaults to [0, 1].

        Returns
        -------
        tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        if wl_indices is None:
            wl_indices = [0, 1]
        return viz_timedomain.plot_td_moments(
            self.data, channel_name=channel_name, wl_indices=wl_indices
        )

    def plot_tpsf(self, channel: str, wl: int,
                  time_pt: int, normalise: bool=False, y_max=None, ax=None):
        """
        Plot the TPSF for one channel, wavelength and time point.

        Parameters
        ----------
        channel : str or int, optional
            Channel to plot. Defaults to the first.
        wl : float or int, optional
            Wavelength to plot. Defaults to the first.
        time_pt : int, optional
            Time index to plot. Defaults to the first.
        normalise : bool
            Scale the curve to a peak of one. Default False.
        y_max : float, optional
            Upper limit of the y-axis.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Created if omitted.

        Returns
        -------
        tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        if wl is None:
            wl = self.data.wavelength.isel(wavelength=0)

        return viz_timedomain.plot_tpsf(
            self, channel=channel,
            wl=wl, time_pt=time_pt, normalise=normalise, y_max=y_max, ax=ax
        )

    def _info_lines(self):
        lines = super()._info_lines()
        lines += ["", "--- TD Configuration ---"]
        if self.status == 'raw' and 'bin' in self.data.dims:
            n_bins = len(self.data.bin)
            if 'timeDelays' in self.data.coords:
                td = self.data.coords['timeDelays'].values
                lines.append(f"Format:  raw ({n_bins} bins, {td[0]:.1f}–{td[-1]:.1f} ns)")
            else:
                lines.append(f"Format:  raw ({n_bins} bins)")
        elif self.status == 'moment' and 'moment' in self.data.dims:
            moments = list(self.data.moment.values)
            lines.append(f"Format:  moments {moments}")
        return lines
