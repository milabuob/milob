from .datastream import Datastream
from .nirs import NirsStream
from ..processing import mbll
from typing import Optional, Tuple, Union
import numpy as np


class CW_Stream(NirsStream):
    def __init__(self, data, probe, **kwargs):
        super().__init__(data, probe, **kwargs)
        
    
    @classmethod
    def from_nirs(cls, filepath, *, sc_threshold, coord_file=None, length_unit="cm", name=None, **kwargs):
        """
        Load the optical CW stream from a .nirs file.

        Parameters
        ----------
        filepath : str
            Path to the .nirs file.
        sc_threshold : float or None
            Source-detector distance in mm below which a channel is classified as
            short. Fixed at construction. Pass None if the probe has no
            short-separation channels.
        coord_file : str, optional
            AtlasViewer .txt file supplying 3-D probe geometry, overriding the
            geometry in the .nirs file.
        length_unit : {'cm', 'mm'}
            Unit of the probe coordinates. Default 'cm'.
        name : str, optional
            Name for the stream. Default 'optical'.

        Returns
        -------
        CW_Stream
            The optical stream. Auxiliary channels and the stimulus matrix are
            not returned; use ``Session.from_nirs()`` to retain them.
        """
        from ..core.session import Session  # lazy import
        optical_name = name or "nirs"
        session = Session.from_nirs(
            filepath,
            coord_file=coord_file,
            length_unit=length_unit,
            optical_name=optical_name,
            sc_threshold=sc_threshold,
            **kwargs,
        )
        return session.streams[optical_name]
    
    
    def sci_screen(self,
                    freq_range: Tuple[float, float] = (0.5, 2.5),
                    threshold: Optional[float] = None,
                    motion_floor: Optional[float] = 0.8,
                    window_length: float = 10.0,
                    step_size=None,
                    min_cycles: int = 3,
                    show_summary: bool = True,
                    export_results: bool = True,
                    plot: bool = False,
                    inplace: bool = False):
        """
        Compute the Scalp Coupling Index (SCI) and optionally exclude channels.

        SCI correlates the cardiac pulsation between wavelengths in sliding
        windows. The per-channel median is stored as ``coords['sci']`` and the
        fraction of motion-affected windows as ``coords['sci_motion_burden']``.
        Defined for CW intensity or OD data only.

        Parameters
        ----------
        freq_range : tuple of (low, high)
            Cardiac frequency band in Hz. Default (0.5, 2.5).
        threshold : float, optional
            Median SCI below which a channel is marked bad. Typical value 0.8.
            If None (default), no channel is excluded.
        motion_floor : float, optional
            Per-window SCI below which a window counts as motion-affected.
            Default 0.8. If None, ``sci_motion_burden`` is reported as NaN.
        window_length : float
            Window length in seconds. Default 10.0.
        step_size : float, optional
            Window step in seconds. Default is 50% overlap.
        min_cycles : int
            Minimum cardiac cycles required per window. Default 3.
        show_summary : bool
            Print a summary table. Default True.
        export_results : bool
            Write a quality log to ``./data_quality/{name}_sci_quality.txt``.
        plot : bool
            Show a histogram of SCI values. Default False.
        inplace : bool
            Modify this stream in place. Default False.

        Returns
        -------
        CW_Stream
            Stream carrying ``coords['sci']`` and ``coords['sci_motion_burden']``.

        References
        ----------
        Pollonini, L. et al. (2014). Hearing Research, 309, 84-93.
        """
        from ..processing.quality import (
            compute_sci, quality_summary, save_summary_to_file
        )
        from ..viz.quality import plot_quality_histogram
        import warnings

        if 'wavelength' not in self.data.dims:
            raise ValueError(
                "SCI requires a 'wavelength' dimension. "
                "Data must be in raw intensity or OD format with multiple wavelengths."
            )

        sci, windows = compute_sci(self.data, freq_range=freq_range,
                          window_length=window_length,
                          step_size=step_size, min_cycles=min_cycles,
                          return_windows=True)

        target = self if inplace else self.copy()

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            if motion_floor is not None:
                motion_burden = np.nanmean(windows < motion_floor, axis=0)
            else:
                motion_burden = np.full(sci.shape, np.nan)

        target.data.coords['sci'] = ('channel', sci)
        target.data.coords['sci_motion_burden'] = ('channel', motion_burden)

        summary = quality_summary(
            sci,
            channel_labels=self.data.channel.values,
            threshold=threshold,
            metric_name="SCI",
            verbose=show_summary,
            is_short=self.data.coords['is_short'].values,
            extra_columns={'motion_burden': motion_burden},
        )

        if export_results:
            import os
            export_dir = "./data_quality"
            os.makedirs(export_dir, exist_ok=True)
            fname = os.path.join(export_dir, f"{self.name}_sci_quality.txt")
            save_summary_to_file(summary, fname)
            if show_summary:
                print(f"Quality log exported to {fname}")

        target.add_history('sci_screen', {
            'freq_range': freq_range,
            'threshold': threshold,
            'motion_floor': motion_floor,
            'n_bad_marked': len(summary['bad_channels'])
        })

        if threshold is not None and len(summary['bad_channels']) > 0:
            target.mark_bad_channels(summary['bad_channels'], inplace=True)

        if plot:
            import matplotlib.pyplot as plt
            plot_quality_histogram(
                sci,
                threshold=threshold if threshold is not None else motion_floor,
                metric_name="SCI",
            )
            plt.show()

        return target


    def psp_screen(self,
                    freq_range: Tuple[float, float] = (0.5, 2.5),
                    threshold: Optional[float] = None,
                    motion_floor: Optional[float] = 0.1,
                    window_length: float = 10.0,
                    step_size=None,
                    min_cycles: int = 3,
                    show_summary: bool = True,
                    export_results: bool = True,
                    plot: bool = False,
                    inplace: bool = False):
        """
        Compute the Peak Spectral Power (PSP) and optionally exclude channels.

        PSP is the peak of the power spectrum of the cross-wavelength
        correlation, evaluated in sliding windows. It detects motion synchronised
        across wavelengths, which inflates SCI but spreads spectral power away
        from the cardiac frequency. The per-channel median is stored as
        ``coords['psp']`` and the fraction of motion-affected windows as
        ``coords['psp_motion_burden']``.

        Parameters
        ----------
        freq_range : tuple of (low, high)
            Cardiac frequency band in Hz. Default (0.5, 2.5).
        threshold : float, optional
            Median PSP below which a channel is marked bad. Typical value 0.1.
            If None (default), no channel is excluded.
        motion_floor : float, optional
            Per-window PSP below which a window counts as motion-affected.
            Default 0.1. If None, ``psp_motion_burden`` is reported as NaN.
        window_length : float
            Window length in seconds. Default 10.0. Frequency resolution is
            approximately 1/window_length, so short windows blur the peak.
        step_size : float, optional
            Window step in seconds. Default is 50% overlap.
        min_cycles : int
            Minimum cardiac cycles required per window. Default 3.
        show_summary : bool
            Print a summary table. Default True.
        export_results : bool
            Write a quality log to ``./data_quality/{name}_psp_quality.txt``.
        plot : bool
            Show a histogram of PSP values. Default False.
        inplace : bool
            Modify this stream in place. Default False.

        Returns
        -------
        CW_Stream
            Stream carrying ``coords['psp']`` and ``coords['psp_motion_burden']``.

        References
        ----------
        Pollonini, L. et al. (2014). Hearing Research, 309, 84-93.
        Pollonini, L., Bortfeld, H., & Oghalai, J. S. (2016). Biomedical Optics
        Express, 7(12), 5104-5119.
        """
        from ..processing.quality import (
            compute_psp, quality_summary, save_summary_to_file
        )
        from ..viz.quality import plot_quality_histogram
        import warnings

        if 'wavelength' not in self.data.dims:
            raise ValueError(
                "PSP requires a 'wavelength' dimension. "
                "Data must be in raw intensity or OD format with multiple wavelengths."
            )

        psp, windows = compute_psp(self.data, freq_range=freq_range,
                          window_length=window_length,
                          step_size=step_size, min_cycles=min_cycles,
                          return_windows=True)

        target = self if inplace else self.copy()

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            if motion_floor is not None:
                motion_burden = np.nanmean(windows < motion_floor, axis=0)
            else:
                motion_burden = np.full(psp.shape, np.nan)

        target.data.coords['psp'] = ('channel', psp)
        target.data.coords['psp_motion_burden'] = ('channel', motion_burden)

        summary = quality_summary(
            psp,
            channel_labels=self.data.channel.values,
            threshold=threshold,
            metric_name="PSP",
            verbose=show_summary,
            is_short=self.data.coords['is_short'].values,
            extra_columns={'motion_burden': motion_burden},
        )

        if export_results:
            import os
            export_dir = "./data_quality"
            os.makedirs(export_dir, exist_ok=True)
            fname = os.path.join(export_dir, f"{self.name}_psp_quality.txt")
            save_summary_to_file(summary, fname)
            if show_summary:
                print(f"Quality log exported to {fname}")

        target.add_history('psp_screen', {
            'freq_range': freq_range,
            'threshold': threshold,
            'motion_floor': motion_floor,
            'n_bad_marked': len(summary['bad_channels'])
        })

        if threshold is not None and len(summary['bad_channels']) > 0:
            target.mark_bad_channels(summary['bad_channels'], inplace=True)

        if plot:
            import matplotlib.pyplot as plt
            plot_quality_histogram(
                psp,
                threshold=threshold if threshold is not None else motion_floor,
                metric_name="PSP",
            )
            plt.show()

        return target


    def sci_psp_screen(self,
                    freq_range: Tuple[float, float] = (0.5, 2.5),
                    logic: str = 'or',
                    sci_threshold: float = 0.8,
                    psp_threshold: float = 0.1,
                    threshold: Optional[float] = 0.0,
                    window_length: float = 10.0,
                    step_size=None,
                    min_cycles: int = 3,
                    show_summary: bool = True,
                    export_results: bool = True,
                    inplace: bool = False):
        """
        Combine per-window SCI and PSP into a single quality covariate.

        Both metrics are computed on the same window grid and combined per window
        by ``logic``. The fraction of windows passing is stored as
        ``coords['sci_psp_frac_good']``, alongside ``coords['sci']`` and
        ``coords['psp']``.

        Parameters
        ----------
        freq_range : tuple of (low, high)
            Cardiac frequency band in Hz. Default (0.5, 2.5).
        logic : {'or', 'and'}
            'or' marks a window good if either metric passes; 'and' requires
            both. Default 'or'.
        sci_threshold : float
            Per-window SCI floor for the combination. Default 0.8.
        psp_threshold : float
            Per-window PSP floor for the combination. Default 0.1.
        threshold : float, optional
            Value of ``frac_good`` at or below which a channel is marked bad.
            Default 0.0, excluding only channels where no window passed. Set to
            None to skip exclusion.
        window_length : float
            Window length in seconds, shared by both metrics. Default 10.0.
        step_size : float, optional
            Window step in seconds. Default is 50% overlap.
        min_cycles : int
            Minimum cardiac cycles required per window. Default 3.
        show_summary : bool
            Print a summary table. Default True.
        export_results : bool
            Write a quality log to ``./data_quality/{name}_sci_psp_quality.txt``.
        inplace : bool
            Modify this stream in place. Default False.

        Returns
        -------
        CW_Stream
            Stream carrying ``coords['sci']``, ``coords['psp']`` and
            ``coords['sci_psp_frac_good']``.

        References
        ----------
        Pollonini, L. et al. (2014). Hearing Research, 309, 84-93.
        Pollonini, L., Bortfeld, H., & Oghalai, J. S. (2016). Biomedical Optics
        Express, 7(12), 5104-5119.
        """
        from ..processing.quality import (
            compute_sci, compute_psp, combine_sci_psp,
            quality_summary, save_summary_to_file
        )

        if 'wavelength' not in self.data.dims:
            raise ValueError(
                "SCI/PSP require a 'wavelength' dimension. "
                "Data must be in raw intensity or OD format with multiple wavelengths."
            )

        sci, sci_windows = compute_sci(self.data, freq_range=freq_range,
                          window_length=window_length,
                          step_size=step_size, min_cycles=min_cycles,
                          return_windows=True)
        psp, psp_windows = compute_psp(self.data, freq_range=freq_range,
                          window_length=window_length,
                          step_size=step_size, min_cycles=min_cycles,
                          return_windows=True)

        _, frac_good = combine_sci_psp(sci_windows, psp_windows,
                                        sci_threshold=sci_threshold,
                                        psp_threshold=psp_threshold,
                                        logic=logic)

        target = self if inplace else self.copy()
        target.data.coords['sci'] = ('channel', sci)
        target.data.coords['psp'] = ('channel', psp)
        target.data.coords['sci_psp_frac_good'] = ('channel', frac_good)

        # quality_summary classifies "bad" as strictly < threshold. frac_good
        # legitimately lands exactly on 0.0 for a truly dead channel (unlike
        # SNR/SCI/PSP, which essentially never land exactly on a threshold),
        # so nudge the comparison value to make that boundary inclusive
        # without changing quality_summary's shared, strict-< behaviour.
        summary_threshold = None if threshold is None else threshold + 1e-9
        summary = quality_summary(
            frac_good,
            channel_labels=self.data.channel.values,
            threshold=summary_threshold,
            metric_name=f"SCI_{logic.upper()}_PSP_frac_good",
            verbose=show_summary,
            is_short=self.data.coords['is_short'].values,
            extra_columns={'sci': sci, 'psp': psp},
        )

        if export_results:
            import os
            export_dir = "./data_quality"
            os.makedirs(export_dir, exist_ok=True)
            fname = os.path.join(export_dir, f"{self.name}_sci_psp_quality.txt")
            save_summary_to_file(summary, fname)
            if show_summary:
                print(f"Quality log exported to {fname}")

        target.add_history('sci_psp_screen', {
            'freq_range': freq_range,
            'logic': logic,
            'sci_threshold': sci_threshold,
            'psp_threshold': psp_threshold,
            'threshold': threshold,
            'n_bad_marked': len(summary['bad_channels'])
        })

        if threshold is not None and len(summary['bad_channels']) > 0:
            target.mark_bad_channels(summary['bad_channels'], inplace=True)

        return target


    def estimate_hr(self,
                     freq_range: Tuple[float, float] = (0.7, 2.2),
                     window_length: float = 10.0,
                     step_size=None,
                     min_cycles: int = 3,
                     channel_selection: str = 'short',
                     weight_by_quality: bool = True,
                     name: Optional[str] = None):
        """
        Estimate heart rate from the cardiac pulsation in intensity or OD data.

        The selected channels are averaged over wavelength, z-scored, combined
        (optionally weighted by PSP or SCI) and passed to
        :func:`~milob.processing.cardiac.estimate_hr_from_trace`. Bad channels
        are excluded.

        Parameters
        ----------
        freq_range : tuple of float
            Cardiac band in Hz. Default (0.7, 2.2).
        window_length : float
            Window length in seconds. Default 10.
        step_size : float, optional
            Step between windows in seconds. Default ``window_length / 2``.
        min_cycles : int
            Minimum cardiac cycles per window. Default 3.
        channel_selection : {'short', 'long', 'all'}
            Channels to combine. 'short' (default) falls back to the long
            channels, with a warning, when there are no short channels.
        weight_by_quality : bool
            Weight channels by the 'psp' or 'sci' coordinate when present
            (from :meth:`psp_screen` or :meth:`sci_screen`). Default True.
        name : str, optional
            Name of the returned stream. Default ``'{name}_hr'``.

        Returns
        -------
        AuxStream
            Signals 'hr' (beats per minute) and 'hr_confidence', one sample per
            window.

        Raises
        ------
        ValueError
            If the data has no wavelength dimension, the status is not 'raw',
            'processed' or 'od', ``channel_selection`` is unknown, or no usable
            channel remains.
        """
        import warnings as _warnings
        import xarray as xr
        from ..processing.cardiac import estimate_hr_from_trace
        from .auxiliary import AuxStream

        if 'wavelength' not in self.data.dims:
            raise ValueError(
                "estimate_hr requires OD/intensity data with a 'wavelength' dimension."
            )
        if self.status not in ('raw', 'processed', 'od'):
            raise ValueError(
                f"estimate_hr expects raw intensity or OD data (status 'raw', 'processed', "
                f"or 'od'), got status='{self.status}'. Cardiac pulsation extraction assumes "
                f"an intensity-like signal, not a post-MBLL derived quantity. 'processed' "
                f"means filtered/motion-corrected/quality-screened raw intensity -- still "
                f"intensity-like, so it's allowed here same as 'raw'."
            )

        fs = self.data.attrs.get('sampling_rate')
        if fs is None:
            fs = 1.0 / np.nanmean(np.diff(self.data.time.values))

        is_short = self.data.coords.get('is_short')
        is_bad = self.data.coords.get('is_bad')
        n_channels = self.data.sizes['channel']

        if channel_selection == 'short':
            sel = is_short.values.copy() if is_short is not None else np.ones(n_channels, dtype=bool)
            if is_short is None or not sel.any():
                _warnings.warn(
                    "No short-separation channels found on this probe; "
                    "falling back to long channels for HR estimation.", RuntimeWarning
                )
                sel = (~is_short.values) if is_short is not None else np.ones(n_channels, dtype=bool)
        elif channel_selection == 'long':
            sel = (~is_short.values) if is_short is not None else np.ones(n_channels, dtype=bool)
        elif channel_selection == 'all':
            sel = np.ones(n_channels, dtype=bool)
        else:
            raise ValueError("channel_selection must be one of 'short', 'long', 'all'.")

        if is_bad is not None:
            sel = sel & ~is_bad.values
        if not sel.any():
            raise ValueError("No usable (non-bad) channels available for HR estimation.")

        # Cardiac pulsation is present at every wavelength; average across
        # wavelength first (this is not a chromophore-unmixing step), then
        # across the selected channels.
        chan_data = self.data.isel(channel=sel).mean('wavelength')  # (time, channel)

        weights = None
        if weight_by_quality:
            for qkey in ('psp', 'sci'):
                if qkey in chan_data.coords:
                    w = np.clip(chan_data.coords[qkey].values, 0, None)
                    if np.isfinite(w).any() and np.nansum(w) > 0:
                        weights = w
                        break

        values = chan_data.values  # (time, n_sel_channels)
        # Mask +/-inf (possible in OD) as well as NaN.
        values = np.where(np.isfinite(values), values, np.nan)
        # z-score each channel so channels of unequal amplitude don't dominate
        mean = np.nanmean(values, axis=0, keepdims=True)
        std = np.nanstd(values, axis=0, keepdims=True)
        std[std == 0] = np.nan
        z = (values - mean) / std

        if weights is not None:
            trace = np.nansum(z * weights, axis=1) / np.nansum(weights)
        else:
            trace = np.nanmean(z, axis=1)

        times, hr_bpm, confidence = estimate_hr_from_trace(
            trace, fs=fs, freq_range=freq_range, window_length=window_length,
            step_size=step_size, min_cycles=min_cycles,
        )

        abs_times = self.data.time.values[0] + times
        out = xr.DataArray(
            np.stack([hr_bpm, confidence], axis=-1),
            dims=('time', 'signal'),
            coords={'time': abs_times, 'signal': ['hr', 'hr_confidence']},
        )

        hr_stream = AuxStream(out, signal_type='hr', name=name or f"{self.name}_hr")
        if len(abs_times) > 1:
            hr_stream.data.attrs['sampling_rate'] = 1.0 / np.median(np.diff(abs_times))
        hr_stream.data.attrs['units'] = 'bpm'
        hr_stream.add_history('estimate_hr', {
            'source_stream': self.name,
            'input_status': self.status,
            'freq_range': freq_range,
            'window_length': window_length,
            'step_size': step_size if step_size is not None else window_length / 2,
            'min_cycles': min_cycles,
            'channel_selection': channel_selection,
            'n_channels_used': int(sel.sum()),
            'weighted': weights is not None,
        })
        return hr_stream

    def to_od(self, baseline=None, baseline_window=None, inplace=False):
        """
        Convert intensity to optical density.

        Parameters
        ----------
        baseline : np.ndarray, optional
            Pre-computed baseline intensity. Computed from the data if None.
        baseline_window : tuple of (start, end), optional
            Time indices over which to compute the baseline.
        inplace : bool
            Modify this stream in place. Default False.

        Returns
        -------
        CW_Stream
            Stream holding optical density, with ``status='od'``.

        Examples
        --------
        >>> od = raw.to_od()
        >>> od = raw.to_od(baseline_window=(0, 100))
        """
        if self.status == 'od':
            print("Warning: Data is already in OD. Skipping conversion.")
            return self if inplace else self.copy()
        
        # Call function
        od_data = mbll.intensity_to_od(
            self.data,
            baseline=baseline,
            baseline_window=baseline_window
        )
        
        # Setup the target Dataset object
        target = self if inplace else self.copy()
        target.data = od_data
        target.status = 'od'
    
        # Check for any NaN channels
        other_dims = [d for d in od_data.dims if d != 'channel']
        is_nan_channel = np.isnan(od_data).all(dim=other_dims)
        nan_labels = od_data.channel.values[is_nan_channel.values]
        
        target.add_history('to_od', {
            'baseline_window': baseline_window,
            'baseline_provided': baseline is not None,
        })

        if len(nan_labels) > 0:
            target.mark_bad_channels(nan_labels, inplace=True)
            target.add_history('auto_mark_nan_channels', {'channels': list(nan_labels)})

        return target
        
    
    def mbll(self, dpf: Union[float, list, np.ndarray] = [6.0, 6.0], inplace: bool=False):
        """
        Convert optical density to haemoglobin concentration changes.

        Applies the modified Beer-Lambert law, giving relative changes
        (delta HbO, delta HbR) rather than absolute concentrations.

        Parameters
        ----------
        dpf : float, list or ndarray
            Differential pathlength factor. A scalar applies to every wavelength;
            a sequence gives one value per wavelength. Default [6.0, 6.0].
        inplace : bool
            Modify this stream in place. Default False.

        Returns
        -------
        CW_Stream
            Stream holding concentration changes, with ``status='conc'``.
        """
        if self.status != 'od':
            raise ValueError("Data status is not 'od'. Ensure input is optical density.")

        # self.probe.distances returns a list, physics expects an array
        dist_array = np.array(self.probe.distances)
        wav_array = np.array(self.probe.wavelengths)

        # Call type-aware function
        conc_data = mbll.od_to_concentration(
            self.data,
            wavelengths=wav_array,
            distances = dist_array,
            dpf = np.array(dpf)
        )
        target = self if inplace else self.copy()
        target.data = conc_data
        target.status = 'conc'
        target.add_history('mbll', {'dpf': np.atleast_1d(dpf).tolist()})
        return target

    def reconstruct(self, operator, mua0=None, musp0=None, n: float = 1.4, *, R_eff=None,
                     greens_fn=None, phi0_source: str = "model", phi0_measured=None,
                     jacobian_fn=None, jacobian_kwargs: Optional[dict] = None,
                     alpha: Optional[float] = None, lambda1: Optional[float] = None,
                     lambda2: Optional[float] = None,
                     method: str = "svd", channel_noise=None,
                     channel_mask=None, inplace: bool = False) -> "OptPropStream":
        """
        Reconstruct an absorption change on a voxel grid from OD.

        Builds one sensitivity matrix per wavelength and applies a Tikhonov
        inverse (:func:`~milob.processing.fitting.tikhonov_solve`) to every
        frame. The stream must have ``status='od'`` and dims
        (time, channel, wavelength).

        Parameters
        ----------
        operator : SensitivityOperator or VoxelGrid
            A :class:`~milob.forward.sensitivity.SensitivityOperator` whose
            channel labels match this stream (use ``operator.for_probe(stream)``
            to align them), or a bare ``VoxelGrid``, in which case ``mua0`` and
            ``musp0`` are required and the matrices are built here.
        mua0, musp0 : float, optional
            Baseline absorption and reduced scattering in cm^-1. Taken from
            the operator when one is given.
        n : float
            Refractive index. Default 1.4.
        R_eff, greens_fn, phi0_source, phi0_measured
            Passed to :func:`~milob.forward.jacobian.build_jacobian` when it is
            the ``jacobian_fn``; ignored otherwise.
        jacobian_fn : callable, optional
            Builds the matrix for one wavelength, with the signature and return
            dict of :func:`~milob.forward.jacobian.build_jacobian` (the
            default). Not allowed together with a ``SensitivityOperator``.
        jacobian_kwargs : dict, optional
            Extra keyword arguments for ``jacobian_fn``.
        alpha, lambda1, lambda2, method, channel_noise
            Passed to :func:`~milob.processing.fitting.tikhonov_solve`.
            ``lambda1=0.01, lambda2=0.1`` is a common choice.
        channel_mask : array-like of bool, optional
            Channels to use. Defaults to the operator's mask, or all channels.
        inplace : bool
            Ignored; a new stream is always returned.

        Returns
        -------
        OptPropStream
            Voxel-indexed, dims (time, voxel, wavelength, op) with
            ``op=['mua']`` and ``status='delta_mua'``. ``uncertainty`` holds the
            per-voxel variance and ``resolution_diag`` the resolution-matrix
            diagonal per wavelength.

        Raises
        ------
        ValueError
            If the status is not 'od', the stream has extra dims, the
            operator's channels differ from the stream's, both an operator and
            a ``jacobian_fn`` are given, a bare grid lacks ``mua0``/``musp0``,
            or an included channel contains NaN or inf.
        """
        if self.status != 'od':
            raise ValueError("Data status is not 'od'. Ensure input is optical density.")

        extra_dims = set(self.data.dims) - {'time', 'channel', 'wavelength'}
        if extra_dims:
            raise ValueError(
                "CW_Stream.reconstruct() requires OD data with only "
                f"(time, channel, wavelength) dims, got {self.data.dims} "
                f"(unexpected: {extra_dims})."
            )

        from ..forward.jacobian import build_jacobian, si_greens_adapter
        from ..forward.sensitivity import SensitivityOperator
        from ..imaging import IMAGING_DTYPE
        from ..processing.fitting import tikhonov_solve
        from .opt_prop_stream import OptPropStream
        import xarray as xr

        op_obj = None
        if isinstance(operator, SensitivityOperator):
            op_obj = operator
            voxel_grid = op_obj.voxel_grid
            stream_labels = [str(c) for c in self.data.channel.values]
            if stream_labels != op_obj.channel_labels:
                raise ValueError(
                    f"This operator's rows are {op_obj.n_channels} channels of a "
                    f"different (or differently ordered) montage than the stream's "
                    f"{len(stream_labels)}. Reconstructing would pair each "
                    "measurement with the wrong voxel sensitivities. Realign "
                    "first: `operator.for_probe(stream)`."
                )
            if jacobian_fn is not None:
                raise ValueError(
                    "Pass either a SensitivityOperator or a jacobian_fn, not "
                    "both -- the operator already carries its matrices."
                )
            jacobian_fn = op_obj.as_jacobian_fn()
            if mua0 is None:
                mua0 = op_obj.baseline.get('mua0')
            if musp0 is None:
                musp0 = op_obj.baseline.get('musp0')
            if channel_mask is None:
                channel_mask = op_obj.channel_mask
        else:
            voxel_grid = operator
            if mua0 is None or musp0 is None:
                raise ValueError(
                    "reconstruct() with a bare VoxelGrid needs an explicit "
                    "homogeneous baseline (mua0, musp0 in cm^-1). Building a "
                    "forward.sensitivity.SensitivityOperator instead carries "
                    "the baseline with the Jacobians."
                )

        if jacobian_fn is None:
            jacobian_fn = build_jacobian
        extra = dict(jacobian_kwargs or {})
        if jacobian_fn is build_jacobian:
            extra.setdefault('greens_fn', greens_fn or si_greens_adapter)
            extra.setdefault('R_eff', R_eff)
            extra.setdefault('phi0_source', phi0_source)
            extra.setdefault('phi0_measured', phi0_measured)

        wavelength_vals = np.asarray(self.data.wavelength.values, dtype=float)
        time_vals = self.data.time.values
        n_time = len(time_vals)
        n_wl = len(wavelength_vals)
        n_voxels = voxel_grid.n_voxels

        # Stored as IMAGING_DTYPE; tikhonov_solve works in float64.
        delta_mua = np.full((n_time, n_voxels, n_wl), np.nan, dtype=IMAGING_DTYPE)
        variance = np.full((n_time, n_voxels, n_wl), np.nan, dtype=IMAGING_DTYPE)
        resolution = np.full((n_voxels, n_wl), np.nan, dtype=IMAGING_DTYPE)
        alphas_used = {}

        for wi, wl in enumerate(wavelength_vals):
            jac = jacobian_fn(self.probe, voxel_grid, mua0=mua0, musp0=musp0, n=n,
                               wavelength=float(wl), channel_mask=channel_mask, **extra)
            J = jac['matrix']

            y = self.data.sel(wavelength=wl, method='nearest') \
                         .transpose('channel', 'time').values

            # Excluded channels are zeroed (their rows of J are zero, so they
            # cannot affect x); an included channel with NaN is an error.
            row_mask = jac['channel_mask']
            if row_mask is None:
                row_mask = np.ones(y.shape[0], dtype=bool)
            row_mask = np.asarray(row_mask, dtype=bool)
            bad_rows = row_mask & ~np.isfinite(y).all(axis=1)
            if bad_rows.any():
                labels = np.asarray(jac['channel_labels'])[bad_rows]
                raise ValueError(
                    f"reconstruct(): {bad_rows.sum()} channel(s) included by "
                    f"channel_mask contain NaN/inf at wavelength {wl:g}, which "
                    f"would make the entire reconstructed image NaN. Offending "
                    f"channels: {list(labels[:10])}"
                    f"{' ...' if bad_rows.sum() > 10 else ''}. Either exclude them "
                    f"via channel_mask, or drop them from the stream first "
                    f"(e.g. drop_bad_channels()). Note regress_nuisance() returns "
                    f"NaN for every channel it did not fit, short channels included."
                )
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

            x, unc, info = tikhonov_solve(J, y, alpha=alpha, lambda1=lambda1,
                                           lambda2=lambda2, method=method,
                                           channel_noise=channel_noise,
                                           return_uncertainty=True)
            delta_mua[:, :, wi] = x.T
            variance[:, :, wi] = unc[np.newaxis, :]
            resolution[:, wi] = info['resolution_diag']
            alphas_used[float(wl)] = float(info['alpha'])

        coords = {'time': time_vals, 'voxel': np.arange(n_voxels),
                  'wavelength': wavelength_vals, 'op': ['mua']}
        # Keep the source attrs (e.g. sampling_rate) and add the units.
        recon_attrs = dict(self.data.attrs)
        recon_attrs.update({'units': 'cm^-1', 'mua0': mua0, 'musp0': musp0})
        data_da = xr.DataArray(delta_mua[..., np.newaxis], coords=coords,
                                dims=['time', 'voxel', 'wavelength', 'op'],
                                attrs=recon_attrs)
        unc_da = xr.DataArray(
            variance[..., np.newaxis], coords=coords,
            dims=['time', 'voxel', 'wavelength', 'op'],
            attrs={'description': 'per-voxel variance (diagonal approximation)'},
        )
        res_da = xr.DataArray(
            resolution, coords={'voxel': np.arange(n_voxels), 'wavelength': wavelength_vals},
            dims=['voxel', 'wavelength'],
        )

        record = {
            'mua0': mua0, 'musp0': musp0, 'n': n, 'alpha': alpha,
            'method': method, 'n_voxels': n_voxels,
            'alpha_used_per_wavelength': alphas_used,
            'lambda1': lambda1, 'lambda2': lambda2,
        }
        if op_obj is not None:
            record.update(op_obj.history_entry())
        else:
            record['jacobian_fn'] = getattr(jacobian_fn, '__name__', 'custom')
        history = self.history.copy()
        history.append(self._history_entry('reconstruct', record))

        return OptPropStream(
            data=data_da, voxel_grid=voxel_grid, probe=self.probe,
            uncertainty=unc_da, resolution_diag=res_da, name=self.name,
            events=self.events, status='delta_mua', history=history,
        )

    def regress_nuisance(self, nuisance_method='sc_pca',
                          add_drift='intercept', drift_cutoff=0.01,
                          n_components=0.8, method='robust', n_jobs=1):
        """
        Regress systemic noise and drift out of the data.

        Fits a GLM of nuisance regressors to each channel and returns the
        residual.

        Parameters
        ----------
        nuisance_method : {'none', 'sc_average', 'sc_pca', 'sc_nearest', 'global_pca', 'global_average'}
            Source of the nuisance regressors. The 'sc_*' options build them from
            the short-separation channels, the 'global_*' options from all
            channels. Default 'sc_pca'.
        add_drift : {'none', 'intercept', 'intercept+trend', 'dct'}
            Drift terms included alongside the nuisance regressors. Default
            'intercept'. 'dct' adds a discrete-cosine basis below
            ``drift_cutoff``; the drift is removed from the output (only the
            intercept is added back), so it also acts as a high-pass filter.
        drift_cutoff : float
            High-pass cutoff in Hz for ``add_drift='dct'``. Default 0.01.
        n_components : int or float
            Components kept for 'sc_pca' and 'global_pca'. An integer keeps that
            many; a float in (0, 1) keeps enough to explain that fraction of the
            variance. Default 0.8.
        method : {'ols', 'robust', 'ar-irls'}
            Estimator for the nuisance model. Default 'robust'.
        n_jobs : int
            Parallel jobs for the per-channel fits. Default 1.

        Returns
        -------
        CW_Stream
            Stream holding the residual after regression. Short channels are
            excluded from the fit and returned as NaN.

        References
        ----------
        Gregg, N. M. et al. (2010). Brain specificity of diffuse optical imaging.
        Frontiers in Neuroenergetics, 2, 14.
        """
        from ..analysis.glm import GLM

        # Setup a "Nuisance-Only" GLM
        model = GLM(self)

        # Create only nuisance regressors (No tasks)
        model.create_nuisance_regressors(
            add_drift=add_drift,
            drift_cutoff=drift_cutoff,
            nuisance_method=nuisance_method,
            n_components=n_components
        )

        # Explicitly build a nuisance-only design matrix
        model.create_design_matrix(include_tasks=False, include_nuisance=True)
        
        # Fit and extract residuals
        cleaned_values = model.fit(method=method, return_residuals=True, n_jobs=n_jobs)
        
        # Return a new stream object with the residuals as the data
        new_stream = self.copy()
        new_stream.data.values = cleaned_values
        history_params = {
            'nuisance_method': nuisance_method,
            'add_drift': add_drift,
            'drift_cutoff': drift_cutoff if add_drift == 'dct' else None,
            'n_components': n_components,
            'method': method,
        }
        
        if method == 'ar-irls':
            history_params['steps'] = ['pre_whitening', 'regression']
            history_params['note'] = (
                "ar-irls runs AR pre-whitening then robust regression as one "
                "call. Nuisance regressors are typically slow/tonic, and the "
                "pre-whitening filter can suppress exactly that kind of "
                "regressor's fitted coefficient toward zero, undoing the "
                "intended correction -- verify the effect size against "
                "method='ols' or 'robust' before trusting these results."
            )
        new_stream.add_history('regress_nuisance', history_params)
        return new_stream