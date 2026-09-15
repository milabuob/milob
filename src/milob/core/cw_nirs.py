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

    def regress_nuisance(self, nuisance_method='sc_pca',
                          add_drift='intercept', n_components=0.8,
                          method='robust', n_jobs=1):
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
        add_drift : {'none', 'intercept', 'intercept+trend'}
            Drift terms included alongside the nuisance regressors. Default
            'intercept'.
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