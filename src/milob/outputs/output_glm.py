from abc import ABC
import warnings
import xarray as xr
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from typing import Optional
from .output import BaseOutput


def _fdr_bh(p_values, axis):
    """Benjamini-Hochberg adjusted p-values along `axis`, ignoring NaN entries."""
    from statsmodels.stats.multitest import multipletests

    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    src = np.moveaxis(p_values, axis, -1)
    dst = np.moveaxis(adjusted, axis, -1)

    for idx in np.ndindex(src.shape[:-1]):
        row = src[idx]
        valid = np.isfinite(row)
        if not valid.any():
            continue
        dst[idx][valid] = multipletests(row[valid], method='fdr_bh')[1]

    return adjusted


#: Expected sign of a task-evoked response, used by tail='directional'.
_EXPECTED_SIGN = {'HbO': +1, 'HbT': +1, 'HbR': -1}


def _resolve_tail(tail, outputs):
    """
    Return ``tail`` if given, otherwise the tail shared by ``outputs``.

    Raises
    ------
    ValueError
        If ``tail`` is None and the outputs were built with different tails.
    """
    if tail is not None:
        return tail
    tails = {o.output.attrs.get('tail', 'two') for o in outputs}
    if len(tails) > 1:
        raise ValueError(
            f"Inputs were built with different tails {sorted(tails)}; pass "
            f"tail= explicitly.")
    return tails.pop()


def _p_and_significance(t_values, df, alpha, tail, payload_levels, payload_axis):
    """
    Return p-values and the significance mask for a t-map.

    Parameters
    ----------
    t_values : np.ndarray
        t-statistics.
    df : float or np.ndarray
        Degrees of freedom, scalar or shaped like ``t_values``.
    alpha : float
        Significance threshold.
    tail : {'two', 'directional'}
        'two' is two-tailed. 'directional' is one-tailed in the expected
        direction of each payload level (positive for HbO/HbT, negative for
        HbR).
    payload_levels : sequence of str
        Payload level along ``payload_axis``.
    payload_axis : int
        Axis of ``t_values`` holding the payload levels.

    Returns
    -------
    p : np.ndarray
        p-values of the test.
    is_significant : np.ndarray of bool
        ``p < alpha``.

    Raises
    ------
    ValueError
        If ``tail`` is unknown, or 'directional' is used with a payload level
        that has no expected direction.
    """
    df = np.broadcast_to(np.asarray(df, dtype=float), np.shape(t_values))
    if tail == 'two':
        p = np.clip(2 * (1 - stats.t.cdf(np.abs(t_values), df)), 0.0, 1.0)
        return p, p < alpha

    if tail != 'directional':
        raise ValueError(f"tail must be 'two' or 'directional', got {tail!r}.")

    unknown = [lv for lv in payload_levels if lv not in _EXPECTED_SIGN]
    if unknown:
        raise ValueError(
            f"tail='directional' has no expected response direction for "
            f"{unknown}. It is defined only for {sorted(_EXPECTED_SIGN)}; "
            f"pass tail='two' for this output."
        )

    p = np.empty(np.shape(t_values), dtype=float)
    index = [slice(None)] * np.ndim(t_values)
    for i, level in enumerate(payload_levels):
        index[payload_axis] = i
        sl = tuple(index)
        # sf for an expected increase, cdf for an expected decrease.
        p[sl] = (stats.t.sf(t_values[sl], df[sl]) if _EXPECTED_SIGN[level] > 0
                 else stats.t.cdf(t_values[sl], df[sl]))

    p = np.clip(p, 0.0, 1.0)
    return p, p < alpha


def contrast_vector(c_logic, reg_names):
    """
    Return contrast weights over the regressors.

    Parameters
    ----------
    c_logic : dict or array-like
        ``{regressor_name: weight}``, or weights in regressor order. Names not in
        ``reg_names`` are skipped with a warning.
    reg_names : list of str
        Regressor names in design-matrix order.

    Returns
    -------
    np.ndarray, shape (n_regressors,)
    """
    if isinstance(c_logic, dict):
        c_vector = np.zeros(len(reg_names))
        for label, weight in c_logic.items():
            if label in reg_names:
                c_vector[reg_names.index(label)] = weight
            else:
                print(f"Warning: Regressor '{label}' not found in this session.")
        return c_vector
    return np.asarray(c_logic, dtype=float)


class GLMOutput(BaseOutput):
    """
    Results of a general linear model fit.

    Holds regression estimates over channels or voxels: betas and their
    standard errors from the estimator, and, after
    :meth:`compute_contrasts`, t-statistics, p-values and significance
    flags.

    Parameters
    ----------
    data : xarray.Dataset
        Fitted quantities, indexed by ``channel`` or ``voxel``.
    probe : Probe, optional
        Probe geometry, required for the plotting methods.
    analysis_type : str, optional
        Label stored with the output. Default is ``'GLM'``.
    history : list, optional
        Processing history inherited from the source stream.
    voxel_grid : VoxelGrid, optional
        Geometry for a voxel-indexed output, the counterpart of ``probe``
        for a channel-indexed one.
    """
    def __init__(self, data, probe=None, analysis_type="GLM", history=None,
                 voxel_grid=None):
        # Pass 'data' up to 'BaseOutput' which will assign it to 'self.output'
        super().__init__(data=data, probe=probe, analysis_type=analysis_type, history=history)
        #: Geometry of a voxel-indexed output; None for channel space.
        self.voxel_grid = voxel_grid

    @property
    def spatial_dim(self):
        """Name of the dimension indexing space, ``'channel'`` or ``'voxel'``."""
        for dim in ('channel', 'voxel'):
            if dim in self.output.dims:
                return dim
        raise ValueError(f"No spatial dimension in {tuple(self.output.dims)}.")

    @property
    def payload_dim(self):
        """
        Name of the non-spatial dimension carrying the fitted quantity.

        Either ``'chromophore'``, ``'component'`` or ``'wavelength'``,
        depending on the stream the model was fitted to. Selection and
        plotting keywords are spelled ``chromophore=`` in every case.
        """
        for dim in ('chromophore', 'component', 'wavelength'):
            if dim in self.output.dims:
                return dim
        raise ValueError(f"No payload dimension in {tuple(self.output.dims)}.")

    @classmethod
    def average(cls, outputs, alpha=0.05, tail=None):
        """
        Average several GLM outputs into one.

        Betas are averaged with nanmean and standard errors pooled as the
        standard error of the mean. t-statistics, p-values and significance
        flags are recomputed from the pooled estimates using the summed
        degrees of freedom.

        Parameters
        ----------
        outputs : list of GLMOutput
            Outputs to average, each already passed through
            :meth:`compute_contrasts`.
        alpha : float, optional
            Significance threshold. Default is 0.05.
        tail : {'two', 'directional'}, optional
            Test direction, as in :meth:`compute_contrasts`. None (default)
            uses the tail the inputs were built with.

        Returns
        -------
        GLMOutput
            The averaged result.

        Raises
        ------
        ValueError
            If any input has not been through :meth:`compute_contrasts`.

        Notes
        -----
        Unlike :meth:`compute_contrasts`, ``is_significant`` is set by a
        one-tailed test in the expected direction: an increase for HbO and
        HbT, a decrease for HbR.
        """
        import warnings

        if not outputs:
            raise ValueError("outputs list is empty.")

        ref = outputs[0].output
        if 'se' not in ref.data_vars:
            raise ValueError(
                "GLMOutput.average() requires inference outputs (after compute_contrasts). "
                "Call compute_contrasts() on each output before averaging."
            )

        beta_stack = np.array([o.output['beta'].values for o in outputs])
        se_stack   = np.array([o.output['se'].values   for o in outputs])

        N_valid = np.sum(~np.isnan(beta_stack), axis=0).clip(min=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            beta_avg = np.nanmean(beta_stack, axis=0)
            se_avg   = np.sqrt(np.nanmean(se_stack ** 2, axis=0) / N_valid)

        tail      = _resolve_tail(tail, outputs)
        df_total  = sum(o.output.attrs.get('df', 100) for o in outputs)
        t_values  = np.where(se_avg > 0, beta_avg / se_avg, np.nan)
        pdim      = next(d for d in ('chromophore', 'component', 'wavelength')
                         if d in ref.dims)
        dims   = list(ref['beta'].dims)

        p_values, is_sig = _p_and_significance(
            t_values, df_total, alpha, tail,
            list(ref.coords[pdim].values), dims.index(pdim))

        coords = {d: ref.coords[d] for d in dims}
        ds = xr.Dataset(
            data_vars={
                'beta':           (dims, beta_avg),
                'se':             (dims, se_avg),
                't_stat':         (dims, t_values),
                'p_val':          (dims, p_values),
                'is_significant': (dims, is_sig),
            },
            coords=coords,
            attrs={'df': df_total, 'tail': tail},
        )
        return cls(data=ds, probe=outputs[0].probe,
                   analysis_type=outputs[0].analysis_type,
                   voxel_grid=getattr(outputs[0], 'voxel_grid', None),
                   history=cls._consolidate_histories(outputs, 'GLMOutput.average'))

    
    def compute_contrasts(self, contrast_dict, alpha=0.05, fdr_correction=False,
                          tail='two'):
        """
        Evaluate contrasts of the fitted regressors.

        Parameters
        ----------
        contrast_dict : dict
            Mapping of contrast name to its weights, given either as a mapping
            of regressor name to weight or as a vector whose order matches the
            ``regressor`` dimension::

                {'Task_vs_Rest': {'Task': 1, 'Rest': -1}}
                {'Task_vs_Rest': [1, 0, -1, 0]}

            Named regressors absent from the fit are ignored with a warning.
        alpha : float, optional
            Significance threshold. Default is 0.05.
        fdr_correction : bool, optional
            Apply a Benjamini-Hochberg correction across the spatial dimension,
            independently for each contrast and each chromophore. Each contrast
            and chromophore map is treated as its own family, since HbO and HbR
            are dependent readings of one response and pooling them would
            enlarge the family without adding an independent test. Default is
            False.
        tail : {'two', 'directional'}
            'two' (default) tests ``c'beta != 0``. 'directional' tests
            ``c'beta > 0`` for HbO/HbT and ``c'beta < 0`` for HbR, i.e. that the
            first-named condition evokes the larger response. ``p_val`` always
            matches the test behind ``is_significant``.

        Returns
        -------
        GLMOutput
            New output holding ``beta``, ``se``, ``t_stat``, ``p_val`` and
            ``is_significant`` over the requested contrasts. Under
            ``fdr_correction``, ``p_val`` holds the adjusted values, so every
            downstream table and plot reflects the correction, and the raw
            values are kept as ``p_val_uncorrected``.

        Notes
        -----
        p-values are two-tailed, and ``is_significant`` is not
        direction-specific.

        References
        ----------
        .. [1] Benjamini, Y. and Hochberg, Y. (1995). Controlling the false
               discovery rate: a practical and powerful approach to multiple
               testing. Journal of the Royal Statistical Society B, 57(1), 289-300.
        """
        results_list = []
        
        # Ensure degrees of freedom is a valid number
        df = self.output.attrs.get('df', 100)
        
        # Determine dimension - 3D (channel, chromophore) or 4D (channel, chromophore, bin)
        # Exclude the 'regressor' dims which are collapsed during contrast
        core_dims = [d for d in self.output.beta.dims if d != 'regressor']
        
        # Get regressor names from the GLM fit
        reg_names = list(self.output.regressor.values)
        
        for c_name, c_logic in contrast_dict.items():
            c_vector = contrast_vector(c_logic, reg_names)
            
            # Convert contrast to a named DataArray for automatic alignment
            c = xr.DataArray(c_vector, coords={'regressor': reg_names}, dims=['regressor'])
            
            # Calculate Contrast Beta: c' * Beta
            # Sums across the 'regressor' dimension
            con_beta = (self.output.beta * c).sum(dim='regressor', skipna=False)
            
            # Calculate Contrast Variance: c' * Cov * c
            cj = c.rename({'regressor': 'regressor_j'})
            ci = c.rename({'regressor': 'regressor_i'})
            tmp = (self.output.covariance * cj).sum(dim='regressor_j', skipna=False)
            con_var = (tmp * ci).sum(dim='regressor_i', skipna=False)
            
            # Calculate Stats
            beta_vals = con_beta.values
            var_vals  = con_var.values.clip(min=1e-15)
            se_values = np.sqrt(var_vals)

            t_values = beta_vals / se_values

            pdim = next(d for d in ('chromophore', 'component', 'wavelength')
                        if d in con_beta.dims)

            p_values, is_sig = _p_and_significance(
                t_values, df, alpha, tail,
                list(con_beta.coords[pdim].values), core_dims.index(pdim))

            p_uncorrected = p_values
            if fdr_correction:
                p_values = _fdr_bh(p_values, axis=core_dims.index(self.spatial_dim))
                # The mask has to follow the p-values it is derived from.
                is_sig = p_values < alpha

            # Store in Dataset
            data_vars = {
                'beta':           (core_dims, beta_vals),
                'se':             (core_dims, se_values),
                't_stat':         (core_dims, t_values),
                'p_val':          (core_dims, p_values),
                'is_significant': (core_dims, is_sig),
            }
            if fdr_correction:
                data_vars['p_val_uncorrected'] = (core_dims, p_uncorrected)

            ds_con = xr.Dataset(
                data_vars=data_vars,
                coords = {d: self.output.coords[d] for d in core_dims}
            )
            
            results_list.append(ds_con.expand_dims(contrast=[c_name]))
            
        # Combine all contrasts
        contrast_results = xr.concat(results_list, dim='contrast')
        # roi_average() and average() re-derive p from t, so they need the
        # df these t-values have and the tail they were tested with.
        contrast_results.attrs.update({'df': df, 'tail': tail})

        # Carry the input's provenance (stream history + GLM.fit) forward and
        # record this step, so an "Inference" output stays self-documenting.
        history = list(self.history) + [self._history_entry(
            'GLMOutput.compute_contrasts',
            {'contrasts': list(contrast_dict.keys()),
             'alpha': alpha,
             'tail': tail,
             'fdr_correction': fdr_correction},
        )]

        return GLMOutput(data=contrast_results, probe=self.probe,
                         voxel_grid=self.voxel_grid,
                         analysis_type="Inference", history=history)
    
    
    
    def combine_hb(self, alpha=0.05):
        """
        Combine HbO and HbR evidence into a single per-channel test.

        p-values are pooled with Fisher's method, and a channel is flagged
        only where HbO increases and HbR decreases.

        Parameters
        ----------
        alpha : float, optional
            Significance threshold. Default is 0.05.

        Returns
        -------
        GLMOutput
            New output with ``'combined'`` added along the chromophore
            dimension.
        """
        # Extract HbO and HbR data for the specific contrast
        pdim = self.payload_dim
        hbo = self.output.sel({pdim: 'HbO'})
        hbr = self.output.sel({pdim: 'HbR'})

        # Check directionality mask (physiological validation)
        # B_k(hb1) > 0 and B_k(hb2) < 0
        valid_mask = (hbo.beta.values > 0) & (hbr.beta.values < 0)

        # Apply Fisher's Formula: X2 = -2 * (ln(p_hbo) + ln(p_hbr))
        # (used np.log(p) to clip p slightly to avoid log(0) errors)
        p_hbo = hbo.p_val.values.clip(min=1e-10)
        p_hbr = hbr.p_val.values.clip(min=1e-10)
        
        chi2_stat = -2 * (np.log(p_hbo) + np.log(p_hbr))

        # Calculate combined p-value from Chi2 distribution
        combined_p = 1 - stats.chi2.cdf(chi2_stat, df=4)

        # Build results dataset
        current_dims = hbo.t_stat.dims # e.g., ('contrast', 'channel')
        
        combined_ds = xr.Dataset(
            data_vars={
                't_stat': (current_dims, np.sqrt(chi2_stat)),   # brings magnitude of chi-squared down to be comparable to t-values
                'p_val': (current_dims, combined_p),
                'beta': (current_dims, hbo.beta.values - hbr.beta.values),
                'is_significant': (current_dims, (combined_p < alpha) & valid_mask)
                # 'is_significant': (current_dims, (combined_p < alpha) & valid_mask & hbo.is_significant.values & hbr.is_significant.values)
            },
            # Use the coords from the original hbo slice
            coords=hbo.coords
        )
        
        # Add the chromophore dimension back explicitly
        combined_ds = combined_ds.expand_dims({pdim: ['combined']})
        
        # Append back to the original object
        full_data = xr.concat([self.output, combined_ds], dim=pdim)
        
        # Ensure dimensions are consistent
        full_data = full_data.transpose(*self.output.dims)
        
        # Update metadata
        full_data.attrs = self.output.attrs
        full_data.attrs['has_combined_metrics'] = True
        
        return GLMOutput(data=full_data, probe=self.probe,
                         analysis_type=self.analysis_type,
                         voxel_grid=self.voxel_grid)
    
    
    @property
    def p_var(self):
        """Variance of each contrast estimate, the squared standard error."""
        # If this is an Inference object, the variance is (beta / t_stat)^2
        return (self.output.beta / self.output.t_stat)**2
    
    
        
    def plot_topographic_map(self, contrast: str, val_type: str = 'beta', chromophore: str = 'HbO', bin_idx: int = 0, **kwargs):
        """
        Plot a 2-D topographic map of one contrast.

        Parameters
        ----------
        contrast : str
            Name of the contrast or regressor to display.
        val_type : str, optional
            Variable to map, such as ``'beta'`` or ``'t_stat'``. Default is
            ``'beta'``.
        chromophore : str, optional
            Chromophore to display. Default is ``'HbO'``.
        bin_idx : int, optional
            Index along the ``bin`` dimension when present. Default is 0.
        **kwargs
            Passed to :func:`milob.viz.topo.plot_topo_map`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the map.

        Raises
        ------
        ValueError
            If the output holds no channel-wise data, or if the selected
            values are all NaN.
        """
        if 'channel' not in self.output.dims:
            raise ValueError("This Output object does not contain channel-wise data.")

        from ..viz.topo import plot_topo_map
        
        # Handle Coordinate Names (Inference vs Estimators)
        # If the object came from 'compute_contrasts', use 'contrast'. Otherwise use 'regressor'.
        selector = 'contrast' if 'contrast' in self.output.dims else 'regressor'
        
        # Extract Data 
        data_array = self.output[val_type].sel({selector: contrast, self.payload_dim: chromophore})

        # Handle extra dimensions 
        if 'bin' in data_array.dims:
            values = data_array.isel(bin=bin_idx).values
        else:
            values = data_array.values 

        values = np.squeeze(values)

        # Filter NaNs for the interpolator
        mask = ~np.isnan(values)
        if not np.any(mask):
            raise ValueError(f"No valid data to plot for {contrast} {chromophore}")
        
        title = f"{contrast} ({chromophore}, {val_type})"

        return plot_topo_map(self.probe, values=values, title=title, **kwargs)




    def plot_FCmatrix(self, **kwargs):
        """
        Plot the channel-by-channel connectivity matrix.

        Parameters
        ----------
        **kwargs
            Passed to :func:`milob.viz.matrix.plot_matrix`.

        Returns
        -------
        matplotlib.axes.Axes or None
            The axes containing the matrix, or None if the output is not a
            connectivity result.
        """
        if self.analysis_type != 'Connectivity':
            print("Matrix plot not applicable for this analysis type.")
            return None

        from ..viz.matrix import plot_matrix
        kwargs.setdefault('colorbar_label', 'Correlation')
        return plot_matrix(self.output['correlation'].values, **kwargs)
    
    
    def plot_topographic_map_with_significance(self, contrast: str, val_type: str = 'beta',
                                            chromophore: str = 'HbO', alpha: float = 0.05,
                                            bin_idx: int = 0, title=None, show_labels=False, **kwargs):
        """
        Plot a topographic map with significant channels marked.

        Parameters
        ----------
        contrast : str
            Name of the contrast to display.
        val_type : str, optional
            Variable to map. Default is ``'beta'``.
        chromophore : str, optional
            Chromophore to display. Default is ``'HbO'``.
        alpha : float, optional
            Significance threshold for the markers. Default is 0.05.
        bin_idx : int, optional
            Index along the ``bin`` dimension when present. Default is 0.
        title : str, optional
            Title for the plot.
        show_labels : bool, optional
            If True, annotate each channel with its label. Default is False.
        **kwargs
            Passed to :func:`milob.viz.topo.plot_topo_with_significance`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the map.

        Raises
        ------
        ValueError
            If the output holds no channel-wise data.
        """
        if 'channel' not in self.output.dims:
            raise ValueError("This Output object does not contain channel-wise data.")

        # Extract values
        selector = 'contrast' if 'contrast' in self.output.dims else 'regressor'
        data_array = self.output[val_type].sel({selector: contrast, self.payload_dim: chromophore})
        if 'bin' in data_array.dims:
            values = np.squeeze(data_array.isel(bin=bin_idx).values)
        else:
            values = np.squeeze(data_array.values)

        # Extract significance mask
        sig_array = self.output['is_significant'].sel({selector: contrast, self.payload_dim: chromophore})
        if 'bin' in sig_array.dims:
            sig_mask = np.squeeze(sig_array.isel(bin=bin_idx).values)
        else:
            sig_mask = np.squeeze(sig_array.values)

        # Call the probe-aware plotting function
        from ..viz.topo import plot_topo_with_significance
        return plot_topo_with_significance(
            probe=self.probe,
            values=values,
            sig_mask=sig_mask,
            title=title,
            show_labels=show_labels,
            **kwargs
        )
    

    def plot_group_topographic_map(self, sessions, contrast: str, chromophore: str = 'HbO',
                                val_type: str = 't_stat', alpha: float = 0.05, marker: str = 'o',
                                show_labels=False, title_prefix: str = '', ncols: int = 3,
                                figsize=None):
        """
        Plot one topographic map per session on a shared layout.

        Parameters
        ----------
        sessions : list of GLMOutput
            One output per session or participant.
        contrast : str
            Name of the contrast to display.
        chromophore : str, optional
            Chromophore to display, or ``'combined'``. Default is ``'HbO'``.
        val_type : str, optional
            Variable to map. Default is ``'t_stat'``.
        alpha : float, optional
            Significance threshold for the markers. Default is 0.05.
        marker : str, optional
            Marker style for significant channels. Default is ``'o'``.
        show_labels : bool, optional
            If True, annotate each channel with its label. Default is False.
        title_prefix : str, optional
            Text prepended to each subplot title.
        ncols : int, optional
            Number of subplot columns. Default is 3.
        figsize : tuple, optional
            Figure size in inches. Chosen from the grid shape if omitted.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the grid of maps.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from ..viz.topo import plot_topo_with_significance
        from ..viz import theme

        n_sessions = len(sessions)
        nrows = int(np.ceil(n_sessions / ncols))
        if figsize is None:
            figsize = theme.spatial_panel_figsize(nrows, ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.array(axes).flatten()  # flatten in case of 2D array

        # Determine fixed vmin/vmax
        all_vals = []
        for out in sessions:
            arr = out.output[val_type].sel({'contrast': contrast, out.payload_dim: chromophore}).values
            all_vals.append(arr)
        all_vals = np.concatenate([v.flatten() for v in all_vals if v is not None])
        vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)

        for i, out in enumerate(sessions):
            ax = axes[i]

            vals = out.output[val_type].sel({'contrast': contrast, out.payload_dim: chromophore}).values
            sig_mask = out.output['is_significant'].sel({'contrast': contrast, out.payload_dim: chromophore}).values

            plot_topo_with_significance(
                probe=out.probe,
                values=vals,
                sig_mask=sig_mask,
                ax=ax,
                cmap=theme.DIVERGING_CMAP,
                vmin=vmin,
                vmax=vmax,
                show_labels=show_labels,
                show_cbar=False,
                title=f"{title_prefix} {i+1}",
                marker=marker
            )

        # Hide any empty axes
        for j in range(n_sessions, len(axes)):
            axes[j].axis('off')

        # Shared colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        im = axes[0].images[0]
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label(val_type)

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.show()


    # ------------------------------------------------------------------
    # Tabular helpers
    # ------------------------------------------------------------------

    def to_table(self, contrast: str, chromophore: str = "HbO") -> pd.DataFrame:
        """
        Extract one contrast and chromophore as a flat table.

        Works on channel-level and ROI-level outputs alike; the index is
        named after whichever dimension is present.

        Parameters
        ----------
        contrast : str
            Name of the contrast or regressor to extract.
        chromophore : str, optional
            Chromophore to extract, or ``'combined'``. Default is ``'HbO'``.

        Returns
        -------
        pandas.DataFrame
            Indexed by channel or ROI, with columns ``beta``, ``t_stat``,
            ``p_val`` and ``is_significant``, plus ``n_subjects`` or
            ``n_channels`` where present.
        """
        selector = "contrast" if "contrast" in self.output.dims else "regressor"
        sl = self.output.sel({selector: contrast, self.payload_dim: chromophore})

        # Support both channel-level and ROI-level outputs
        if "roi" in self.output.dims:
            index_vals, index_name = sl.roi.values, "roi"
        else:
            index_vals, index_name = sl.channel.values, "channel"

        cols: dict = {
            "beta": sl.beta.values,
            "se":   sl.se.values if "se" in sl else np.full(len(index_vals), np.nan),
            "t_stat": sl.t_stat.values,
            "p_val": sl.p_val.values,
            "is_significant": sl.is_significant.values,
        }
        if "p_val_uncorrected" in sl:
            cols["p_val_uncorrected"] = sl.p_val_uncorrected.values
        if "n_subjects" in sl:
            cols["n_subjects"] = sl.n_subjects.values.astype(int)
        if "n_channels" in sl:
            cols["n_channels"] = sl.n_channels.values.astype(int)

        df = pd.DataFrame(cols, index=index_vals)
        df.index.name = index_name
        return df


    def to_dataframe(self, variables=None) -> pd.DataFrame:
        """
        Return all results as a long-form table.

        One row per contrast, spatial unit and chromophore. Subject-level
        metadata is added by :meth:`milob.Study.to_dataframe`, not here.

        Parameters
        ----------
        variables : list of str, optional
            Data columns to include, for example ``['beta', 'p_val']``. All
            available columns are returned if omitted. The index columns are
            always included.

        Returns
        -------
        pandas.DataFrame
            The long-form table.
        """
        selector = "contrast" if "contrast" in self.output.dims else "regressor"
        tables = []
        for con in self.output[selector].values:
            for chrom in self.output[self.payload_dim].values:
                t = self.to_table(str(con), str(chrom))
                t = t.reset_index().assign(**{selector: str(con), 'chromophore': str(chrom)})
                tables.append(t)
        df = pd.concat(tables, ignore_index=True)

        # Add ROI column if the probe has ROIs defined
        if self.probe is not None and self.probe.rois and 'channel' in df.columns:
            roi_map = self.probe.channel_roi_map
            df.insert(df.columns.get_loc('channel') + 1, 'roi', df['channel'].map(roi_map))

        if variables is not None:
            index_cols = {selector, 'channel', 'voxel', 'roi',
                          'chromophore', 'component'}
            data_cols  = [c for c in df.columns if c not in index_cols]
            unknown    = [v for v in variables if v not in data_cols]
            if unknown:
                raise ValueError(
                    f"Unknown variable(s): {unknown}. "
                    f"Available: {data_cols}"
                )
            keep = [c for c in df.columns if c in index_cols or c in variables]
            df   = df[keep]

        return df


    def roi_average(self, rois, alpha: float = 0.05, tail=None) -> "GLMOutput":
        """
        Average results within each region of interest.

        Replaces the spatial dimension with ``roi``. Call it on the output of
        :meth:`compute_contrasts`. Channels that are NaN are skipped.

        Parameters
        ----------
        rois : Probe or dict
            A probe carrying ROIs defined with ``add_roi()``, or a mapping of
            ROI name to a list of channel labels::

                roi_map = {'prefrontal': ['S1D1', 'S2D1'], 'motor': ['S3D3']}
                stats_roi = stats.roi_average(roi_map)

        alpha : float, optional
            Significance threshold used to recompute ``is_significant``.
            Default is 0.05.
        tail : {'two', 'directional'}, optional
            Test direction, as in :meth:`compute_contrasts`. None (default)
            uses the tail this output was built with.

        Returns
        -------
        GLMOutput
            New output indexed by ``roi``, with ``n_channels`` giving the
            number of valid channels behind each cell.

        Notes
        -----
        ``beta`` is averaged with nanmean and ``se`` propagated as the
        standard error of the mean, which assumes channels are independent
        and so underestimates the true error. For second-level inference,
        run group statistics on the ROI-averaged betas rather than relying on
        the recomputed t-statistics.
        """
        tail = _resolve_tail(tail, [self])

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

        roi_names    = list(roi_map.keys())
        out_channels = set(self.output.channel.values)
        df_val       = self.output.attrs.get('df', 100)
        selector     = "contrast" if "contrast" in self.output.dims else "regressor"

        has_se = 'se' in self.output

        roi_slices = []
        for roi_name in roi_names:
            roi_chs = [ch for ch in roi_map[roi_name] if ch in out_channels]

            if not roi_chs:
                import logging
                logging.getLogger('milob').warning(
                    f"roi_average: ROI '{roi_name}' has no channels in this output — filled with NaN."
                )
                ref = self.output.isel(channel=0)
                nan_ds = xr.Dataset(
                    {var: xr.full_like(ref[var], fill_value=np.nan) for var in ref.data_vars},
                    coords=ref.coords,
                    attrs=ref.attrs,
                )
                roi_slices.append(nan_ds)
                continue

            block   = self.output.sel(channel=roi_chs)
            n_valid = (~np.isnan(block.beta)).sum(dim='channel')

            # Average betas
            beta_roi = block.beta.mean(dim='channel', skipna=True)

            ch_axis = block.beta.dims.index('channel')
            n_valid_f = n_valid.values.astype(float)

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if has_se:
                    # Propagate SE for the mean of n_valid independent estimates:
                    # SE_ROI = sqrt(nanmean(SE_k^2) / n_valid)
                    se_roi_vals = np.sqrt(
                        np.nanmean(block.se.values ** 2, axis=ch_axis)
                        / np.where(n_valid_f > 0, n_valid_f, np.nan)
                    )
                else:
                    # Fallback: recover SE from stored t_stat and beta
                    se_k = np.abs(block.beta.values / np.where(block.t_stat.values != 0, block.t_stat.values, np.nan))
                    se_roi_vals = np.sqrt(
                        np.nanmean(se_k ** 2, axis=ch_axis)
                        / np.where(n_valid_f > 0, n_valid_f, np.nan)
                    )

            t_roi_vals = beta_roi.values / np.where(se_roi_vals > 0, se_roi_vals, np.nan)

            non_ch_dims   = [d for d in block.beta.dims if d != 'channel']
            pdim = next(d for d in ('chromophore', 'component', 'wavelength')
                        if d in non_ch_dims)
            new_p, new_sig = _p_and_significance(
                t_roi_vals, df_val, alpha, tail,
                list(block.coords[pdim].values), non_ch_dims.index(pdim))

            non_ch_coords = {d: block.beta.coords[d] for d in non_ch_dims}

            roi_ds = xr.Dataset(
                {
                    'beta':           (non_ch_dims, beta_roi.values),
                    'se':             (non_ch_dims, se_roi_vals),
                    't_stat':         (non_ch_dims, t_roi_vals),
                    'p_val':          (non_ch_dims, new_p),
                    'is_significant': (non_ch_dims, new_sig),
                    'n_channels':     (non_ch_dims, n_valid.values.astype(int)),
                },
                coords=non_ch_coords,
            )
            roi_slices.append(roi_ds)

        combined = xr.concat(roi_slices, dim=xr.DataArray(roi_names, dims='roi'))

        combined.attrs.update(self.output.attrs)
        combined.attrs['tail'] = tail
        return GLMOutput(combined, probe=out_probe, analysis_type=self.analysis_type + '_ROI')

    # ------------------------------------------------------------------
    # Cluster-level inference
    # ------------------------------------------------------------------

    @staticmethod
    def _group_residuals(Y, V, beta, tau_sq, group_method):
        """
        Return the residuals of a group model, shape (subject, space).

        Weighted models return residuals scaled by the square root of their weights.
        Missing cells are zero.
        """
        if group_method == 'weighted':
            ok = np.isfinite(Y) & np.isfinite(V) & (V > 0)
            w = np.where(ok, 1.0 / (np.where(ok, V, 1.0) + tau_sq[np.newaxis]), 0.0)
            r = np.sqrt(w) * (np.where(ok, Y, 0.0) - beta[np.newaxis])
        else:
            ok = np.isfinite(Y)
            r = np.where(ok, Y, 0.0) - np.where(np.isfinite(beta), beta, 0.0)[np.newaxis]
        return np.where(ok, r, 0.0)

    def cluster_inference(self, method='grf', cluster_p=0.001, alpha=0.05,
                          tail=None, connectivity=18, smooth_fwhm='auto',
                          statistic='mass', n_perm=None, seed=0, null=None,
                          adjacency=None, tfce_params=None):
        """
        Apply family-wise error correction at the cluster level.

        Clusters are formed at the uncorrected threshold ``cluster_p`` and each
        receives a family-wise error (FWE) corrected p-value. Each contrast and
        chromophore map is its own family.

        ``method='grf'`` uses non-stationary Gaussian random field theory on a group
        voxel map, with smoothness estimated from the group model's residuals.
        ``method='permutation'`` uses max-statistic permutation inference on voxel
        or channel maps: subject-wise sign flips for a group output (exhaustive when
        ``2**n_subjects <= n_perm``), or a circular-shift null for a single subject.

        Parameters
        ----------
        method : {'grf', 'permutation'}
            Inference method. Default 'grf'.
        cluster_p : float
            Uncorrected voxelwise p-value that forms clusters. Not used by TFCE.
            Default 0.001.
        alpha : float
            FWE level for ``is_significant``. Default 0.05.
        tail : {'two', 'directional'}, optional
            'two' tests positive and negative clusters; 'directional' tests positive
            HbO/HbT and negative HbR only. None (default) uses the output's tail.
        connectivity : {6, 18, 26}
            Voxel neighbourhood on the lattice. Default 18.
        smooth_fwhm : 'auto', float or None
            GRF only. Smoothing of the local smoothness estimate; see
            :func:`~milob.stats.cluster.estimate_rpv`.
        statistic : {'mass', 'extent', 'tfce'}
            Permutation only. Cluster statistic. Default 'mass'.
        n_perm : int, optional
            Permutation only, group outputs: number of sign flips. Default 5000.
        seed : int
            Seed for the sign flips.
        null : PermutationNull, optional
            Required for a single-subject permutation test, from
            :meth:`Session.permutation_null`.
        adjacency : scipy.sparse matrix, optional
            Neighbourhood graph. Required for channel outputs, e.g.
            :func:`~milob.stats.permutation.channel_adjacency`. Built from the
            voxel grid for voxel outputs.
        tfce_params : dict, optional
            ``E``, ``H`` and ``dh`` for TFCE.

        Returns
        -------
        GLMOutput
            Copy in which ``is_significant`` marks voxels in clusters with
            p_FWE < ``alpha`` (the previous flags are kept as
            ``is_significant_uncorrected``), with ``cluster_id``, ``cluster_p_fwe``,
            ``cluster_statistic`` and ``cluster_threshold``. GRF adds ``rpv``,
            ``fwhm_mm`` and ``search_resels``; permutation adds ``p_fwe_voxel`` and,
            for TFCE, ``tfce`` and ``p_fwe_tfce``.

        Raises
        ------
        ValueError
            If ``method`` is unknown, GRF is used on a channel or single-subject
            output, a single-subject permutation test has no ``null``, a channel
            output has no ``adjacency``, a voxel output has no ``voxel_grid``, or
            'directional' is used on a payload without an expected direction.

        Notes
        -----
        GRF is an approximation and can be liberal at low degrees of freedom; a
        warning is issued when df < 20.

        References
        ----------
        Hayasaka, S., Phan, K. L., Liberzon, I., Worsley, K. J., & Nichols, T. E.
        (2004). Nonstationary cluster-size inference with random field and
        permutation methods. NeuroImage, 22(2), 676-687.

        Nichols, T. E., & Holmes, A. P. (2002). Nonparametric permutation tests for
        functional neuroimaging: a primer with examples. Human Brain Mapping,
        15(1), 1-25.

        Smith, S. M., & Nichols, T. E. (2009). Threshold-free cluster enhancement.
        NeuroImage, 44(1), 83-98.
        """
        from ..stats import cluster as _cl
        from ..stats import permutation as _perm

        if method not in ('grf', 'permutation'):
            raise ValueError(f"method must be 'grf' or 'permutation', got {method!r}.")
        out = self.output
        space, pdim = self.spatial_dim, self.payload_dim
        is_group = 'subject_beta' in out.data_vars

        if method == 'grf':
            if space != 'voxel':
                raise ValueError(
                    "Random field theory needs a lattice; this output is indexed by "
                    f"{space!r}. Use method='permutation' with an adjacency.")
            if not is_group:
                raise ValueError(
                    "method='grf' needs the subject maps behind a group model, to "
                    "estimate smoothness from its residuals. Use "
                    "Study.get_group_stats(keep_subject_maps=True).")
        elif not is_group and null is None:
            raise ValueError(
                "A single-subject permutation test needs its null: "
                "null=session.permutation_null(stream, pipeline, hypotheses).")

        if space == 'voxel':
            if self.voxel_grid is None:
                raise ValueError("No voxel_grid attached: output.voxel_grid = grid.")
            if adjacency is None:
                adjacency = _cl.lattice_adjacency(self.voxel_grid, connectivity)
        elif adjacency is None:
            raise ValueError(
                "A channel-space output needs adjacency=, e.g. "
                "stats.permutation.channel_adjacency(midpoints_mm, max_mm=30).")

        tail = _resolve_tail(tail, [self])
        group_method = out.attrs.get('group_method', 'simple')
        contrasts, levels = list(out.contrast.values), list(out[pdim].values)
        if tail == 'directional':
            unknown = [lv for lv in levels if lv not in _EXPECTED_SIGN]
            if unknown:
                raise ValueError(f"tail='directional' has no expected direction for {unknown}.")

        if null is not None:
            self._check_null(null, contrasts, levels)
        flips = exhaustive = None
        if method == 'permutation' and is_group:
            n_subj = out.sizes['subject']
            flips, exhaustive = _perm.sign_flips(n_subj, n_perm or 5000, seed)

        n_sp = out.sizes[space]
        nc, npl = len(contrasts), len(levels)
        shape = (nc, n_sp, npl)
        cid = np.zeros(shape, dtype=np.int64)
        cpf = np.full(shape, np.nan)
        cstat = np.full(shape, np.nan)
        thr = np.full((nc, npl), np.nan)
        extra = {}
        if method == 'grf':
            extra = {'rpv': np.full(shape, np.nan), 'fwhm_mm': np.full((nc, npl, 3), np.nan),
                     'search_resels': np.full((nc, npl, 4), np.nan)}
        else:
            extra = {'p_fwe_voxel': np.full(shape, np.nan)}
            if statistic == 'tfce':
                extra.update({'tfce': np.full(shape, np.nan),
                              'p_fwe_tfce': np.full(shape, np.nan)})
        min_df = np.inf

        for ci, con in enumerate(contrasts):
            for pi, lev in enumerate(levels):
                sel = {'contrast': con, pdim: lev}
                t = out.t_stat.sel(sel).transpose(space).values.astype(float)
                if not np.isfinite(t).any():
                    continue
                sign = 'both' if tail == 'two' else _EXPECTED_SIGN[lev]

                if is_group:
                    Y = out.subject_beta.sel(sel).transpose('subject', space).values
                    V = out.subject_var.sel(sel).transpose('subject', space).values
                    n_sub = out.n_subjects.sel(sel).transpose(space).values
                    n_max = int(np.nanmax(n_sub))
                    df = n_max - 1

                if method == 'grf':
                    partial = (n_sub < n_max) & np.isfinite(t)
                    if partial.any():
                        warnings.warn(
                            f"{con} / {lev}: {int(partial.sum())} voxels have fewer "
                            f"than {n_max} subjects and are left out of the search "
                            f"region -- RFT needs one df across the map.", stacklevel=2)
                        t = np.where(partial, np.nan, t)
                    tau = (out.tau_sq.sel(sel).transpose(space).values
                           if 'tau_sq' in out.data_vars else np.zeros(n_sp))
                    beta = out.beta.sel(sel).transpose(space).values
                    resid = self._group_residuals(Y, V, beta, tau, group_method)
                    res = _cl.grf_cluster_inference(
                        t, resid, df, self.voxel_grid, cluster_p=cluster_p,
                        sign=sign, connectivity=connectivity,
                        smooth_fwhm=smooth_fwhm, adjacency=adjacency)
                    extra['rpv'][ci, :, pi] = res['rpv']['rpv']
                    extra['fwhm_mm'][ci, pi] = res['rpv']['global_fwhm_mm']
                    extra['search_resels'][ci, pi] = res['resels']
                    cstat_key = 'resels'
                else:
                    if is_group:
                        t_null = _perm.sign_flip_t(Y, V, flips, method=group_method)
                        if not exhaustive:
                            t_null = t_null[1:]       # row 0 is the identity
                    else:
                        t_null = null.t[:, null.contrasts.index(con), :,
                                        null.payload_levels.index(lev)]
                        df = null.df
                    res = _perm.permutation_inference(
                        t, t_null, adjacency, df=df, cluster_p=cluster_p, sign=sign,
                        statistic=statistic, exhaustive=bool(exhaustive),
                        tfce_params=tfce_params, alpha=alpha)
                    extra['p_fwe_voxel'][ci, :, pi] = res['p_fwe_voxel']
                    if statistic == 'tfce':
                        extra['tfce'][ci, :, pi] = res['tfce']
                        extra['p_fwe_tfce'][ci, :, pi] = res['p_fwe_tfce']
                    cstat_key = 'statistic'

                cid[ci, :, pi] = res['cluster_id']
                cpf[ci, :, pi] = res['cluster_p_fwe']
                for c in res['clusters']:
                    cstat[ci, cid[ci, :, pi] == c['id'], pi] = c[cstat_key]
                thr[ci, pi] = res['threshold']
                min_df = min(min_df, df)

        if method == 'grf' and min_df < 20:
            warnings.warn(
                f"Cluster RFT at df = {min_df:g} can be liberal, most at "
                f"cluster_p=0.001 and in rough or non-stationary images; "
                f"consider method='permutation'.", stacklevel=2)

        dims = ('contrast', space, pdim)
        ds = out.copy()
        ds['is_significant_uncorrected'] = out['is_significant']
        ds['cluster_id'] = (dims, cid)
        ds['cluster_p_fwe'] = (dims, cpf)
        ds['cluster_statistic'] = (dims, cstat)
        ds['is_significant'] = (dims, np.nan_to_num(cpf, nan=1.0) < alpha)
        ds['cluster_threshold'] = (('contrast', pdim), thr)
        if method == 'grf':
            ds['rpv'] = (dims, extra['rpv'])
            ds['fwhm_mm'] = (('contrast', pdim, 'axis'), extra['fwhm_mm'])
            ds['search_resels'] = (('contrast', pdim, 'resel_dim'), extra['search_resels'])
            ds = ds.assign_coords(axis=['x', 'y', 'z'], resel_dim=[0, 1, 2, 3])
        else:
            for k in ('p_fwe_voxel', 'tfce', 'p_fwe_tfce'):
                if k in extra:
                    ds[k] = (dims, extra[k])

        params = {'method': method, 'cluster_p': cluster_p, 'alpha': alpha,
                  'tail': tail, 'connectivity': connectivity,
                  'group_method': group_method if is_group else None}
        if method == 'grf':
            params['smooth_fwhm'] = smooth_fwhm
        else:
            params.update({'statistic': statistic, 'seed': seed,
                           'n_perm': int(len(flips) if flips is not None else null.n_perm),
                           'exhaustive': bool(exhaustive),
                           'null_kind': 'sign_flip' if is_group else null.kind,
                           'null_params': None if is_group else null.params,
                           'tfce_params': tfce_params})
        ds.attrs.update({'tail': tail, 'cluster_method': method,
                         'cluster_forming_p': cluster_p, 'cluster_alpha': alpha,
                         'connectivity': connectivity})
        if method == 'permutation':
            ds.attrs.update({'cluster_statistic_kind': statistic,
                             'n_perm': params['n_perm']})
        history = list(self.history) + [self._history_entry(
            'GLMOutput.cluster_inference', params)]
        return GLMOutput(data=ds, probe=self.probe, voxel_grid=self.voxel_grid,
                         analysis_type=self.analysis_type, history=history)

    def _check_null(self, null, contrasts, levels):
        """Raise if ``null`` was not built for this output."""
        missing = [c for c in contrasts if c not in null.contrasts]
        if missing:
            raise ValueError(f"The null has no maps for contrasts {missing}.")
        if list(null.payload_levels) != list(levels):
            raise ValueError(f"Null payload {null.payload_levels} != output {levels}.")
        here = self.output[self.spatial_dim].values
        if len(null.space_coords) != len(here) or not np.array_equal(
                np.asarray(null.space_coords).astype(str), np.asarray(here).astype(str)):
            raise ValueError("The null's spatial units do not match this output's.")
        if null.observed_t is not None:
            obs = null.observed_t[[null.contrasts.index(c) for c in contrasts]]
            mine = self.output.t_stat.transpose('contrast', self.spatial_dim,
                                                self.payload_dim).values
            ok = np.isfinite(obs) & np.isfinite(mine)
            scale = max(1.0, float(np.nanmax(np.abs(mine))))
            if ok.any() and np.max(np.abs(obs[ok] - mine[ok])) > 1e-3 * scale:
                raise ValueError(
                    "The null's unpermuted t-map does not reproduce this output's "
                    "t_stat -- it was built from a different stream, pipeline or "
                    "contrast set.")

    def cluster_table(self, contrast, chromophore='HbO', voxel_mask=None):
        """
        List the clusters of one map after :meth:`cluster_inference`.

        Parameters
        ----------
        contrast : str
            Contrast name.
        chromophore : str
            Payload level. Default 'HbO'.
        voxel_mask : array-like of bool, optional
            Voxel mask, e.g. grey and white matter. Adds the fraction of each
            cluster inside it.

        Returns
        -------
        pandas.DataFrame
            One row per cluster with ``cluster``, ``sign``, ``n_voxels``, ``resels``
            (GRF only), ``statistic``, ``peak_t``, ``peak_x``/``peak_y``/``peak_z``
            (voxel outputs), ``p_fwe``, ``significant`` and, with ``voxel_mask``,
            ``frac_in_mask``. Sorted by ``p_fwe``.

        Raises
        ------
        ValueError
            If :meth:`cluster_inference` has not been run.
        """
        if 'cluster_id' not in self.output.data_vars:
            raise ValueError("Run cluster_inference() first.")
        sel = {'contrast': contrast, self.payload_dim: chromophore}
        ids = self.output.cluster_id.sel(sel).values
        t = self.output.t_stat.sel(sel).values.astype(float)
        rpv = self.output.rpv.sel(sel).values if 'rpv' in self.output.data_vars else None
        cstat = (self.output.cluster_statistic.sel(sel).values
                 if 'cluster_statistic' in self.output.data_vars else None)
        p = self.output.cluster_p_fwe.sel(sel).values
        pos = self.voxel_grid.positions if self.voxel_grid is not None else None
        alpha = self.output.attrs.get('cluster_alpha', 0.05)
        rows = []
        for k in np.unique(ids[ids > 0]):
            m = np.flatnonzero(ids == k)
            peak = m[np.argmax(np.abs(t[m]))]
            x, y, z = pos[peak] if pos is not None else (np.nan,) * 3
            row = {'cluster': int(k), 'sign': '+' if t[peak] > 0 else '-',
                   'n_voxels': len(m),
                   'resels': float(np.nansum(rpv[m])) if rpv is not None else np.nan,
                   'statistic': float(cstat[m[0]]) if cstat is not None else np.nan,
                   'peak_t': float(t[peak]), 'peak_x': x, 'peak_y': y, 'peak_z': z,
                   'p_fwe': float(p[m[0]]), 'significant': bool(p[m[0]] < alpha)}
            if voxel_mask is not None:
                row['frac_in_mask'] = float(np.asarray(voxel_mask, dtype=bool)[m].mean())
            rows.append(row)
        cols = ['cluster', 'sign', 'n_voxels', 'resels', 'statistic', 'peak_t', 'peak_x',
                'peak_y', 'peak_z', 'p_fwe', 'significant'] + (
                    ['frac_in_mask'] if voxel_mask is not None else [])
        return (pd.DataFrame(rows, columns=cols)
                  .sort_values(['p_fwe', 'n_voxels'], ascending=[True, False])
                  .reset_index(drop=True))

    def add_metadata(self, meta_df: pd.DataFrame, on: str = "channel") -> pd.DataFrame:
        """
        Merge channel-level metadata into the long-form results table.

        Parameters
        ----------
        meta_df : pandas.DataFrame
            Metadata to merge, carrying a column or index named after ``on``.
        on : str, optional
            Join key. Default is ``'channel'``.

        Returns
        -------
        pandas.DataFrame
            The results table with the metadata columns joined on.
        """
        results = self.to_dataframe()
        if on not in results.columns:
            raise ValueError(
                f"Join key '{on}' not found in the results table. "
                f"Available columns: {list(results.columns)}."
            )
        return results.merge(meta_df, on=on, how="left")


    #: Where a projected flag map is cut into "yes" and "no". Both surface
    #: methods below smooth their values onto a mesh, and `plot_topo_3d`
    #: additionally blends overlapping channels, so a boolean does not stay a
    #: boolean by the time it reaches a vertex: it arrives as the fraction of
    #: local support that was flagged. Half is the natural cut -- the flagged
    #: channels have to carry more of the sensitivity here than the unflagged
    #: ones before the voxel is called significant.
    FLAG_CUT = 0.5

    def _style_flag_map(self, kwargs):
        """Set plotting defaults for a boolean map: one flat colour, no colorbar."""
        import matplotlib.colors as mcolors
        from ..viz import theme

        # Limits are deliberately left automatic. A one-colour colormap paints
        # every value the same, so [vmin, vmax] changes no pixel here -- while
        # pinning them to [0, 1] made the smoothing's last-bit rounding
        # (max 1 + 4e-16) trip plot_surface_map's out-of-range warning on a
        # figure that had nothing wrong with it.
        kwargs.setdefault('cmap', mcolors.ListedColormap([theme.SIGNIFICANT_COLOR]))
        kwargs.setdefault('show_colorbar', False)
        return self.FLAG_CUT

    def _is_flag(self, val_type):
        """Whether `val_type` names a boolean flag rather than a magnitude."""
        return (val_type == 'is_significant'
                or (val_type in self.output.data_vars
                    and self.output[val_type].dtype == bool))

    def plot_voxel_slices(
        self,
        contrast=None,
        val_type='t_stat',
        chromophore='HbO',
        background=None,
        sig_mask=False,
        **kwargs,
    ):
        """
        Plot a voxel-space result over anatomical slices.

        Parameters
        ----------
        contrast : str
            Contrast (or regressor) to show.
        val_type : str
            Variable to plot, e.g. 'beta', 't_stat' or 'p_val'. Default 't_stat'.
        chromophore : str
            Payload level. Default 'HbO'.
        background : str, nibabel image or (volume, affine), optional
            Structural volume in the frame of the voxel grid.
        sig_mask : bool
            Show only voxels flagged in ``is_significant``. Default False.
        **kwargs
            Passed to :func:`~milob.viz.volume.plot_voxel_slices`, e.g. ``axis``,
            ``slices``, ``threshold`` or ``cmap``.

        Returns
        -------
        tuple
            ``(fig, axes)``.

        Raises
        ------
        ValueError
            If the output is not voxel-indexed, has no ``voxel_grid``, no contrast
            is given, or ``sig_mask`` is set without ``is_significant``.
        """
        from ..viz.volume import plot_voxel_slices as _plot

        if self.spatial_dim != 'voxel':
            raise ValueError(
                f"plot_voxel_slices needs a voxel-space output; this one is "
                f"indexed by {self.spatial_dim!r}.")
        if self.voxel_grid is None:
            raise ValueError(
                "This GLMOutput has no voxel_grid attached, so its voxels have "
                "no positions. Assign one: output.voxel_grid = grid.")
        if contrast is None:
            raise ValueError("Please input a contrast.")

        selector = 'contrast' if 'contrast' in self.output.dims else 'regressor'

        sel = {selector: contrast, self.payload_dim: chromophore}
        values = self.output[val_type].sel(**sel).values.astype(float)
        if sig_mask:
            if 'is_significant' not in self.output.data_vars:
                raise ValueError(
                    "sig_mask=True needs 'is_significant' -- run "
                    "compute_contrasts() (or a group model) first."
                )
            values = np.where(self.output.is_significant.sel(**sel).values,
                              values, np.nan)

        kwargs.setdefault('colorbar_label', f"{chromophore} {val_type}")
        kwargs.setdefault('title', f"{contrast} ({chromophore} {val_type}) - MNI space")
        return _plot(values, self.voxel_grid, background=background, **kwargs)

    def plot_voxel_3d(
        self,
        contrast=None,
        val_type='t_stat',
        chromophore='HbO',
        sig_mask=False,
        surface=None,
        views=('left', 'right'),
        threshold=None,
        tail='both',
        **kwargs,
    ):
        """
        Plot a voxel-space result on a cortical surface.

        Values are smoothed onto the mesh; for a boolean ``val_type`` each vertex
        holds the local flagged fraction and ``FLAG_CUT`` sets the cut.

        Parameters
        ----------
        contrast : str
            Contrast (or regressor) to show.
        val_type : str
            Variable to plot, e.g. 'beta', 't_stat' or 'is_significant'.
            Default 't_stat'.
        chromophore : str
            Payload level. Default 'HbO'.
        sig_mask : bool
            Show only voxels flagged in ``is_significant``. Default False.
        surface : Surface, optional
            Cortical surface in the frame of the voxel grid. Defaults to the
            fsaverage pial surface.
        views : sequence
            One panel per view. Default ('left', 'right').
        threshold : float, optional
            Hide values that do not pass this, in the direction of ``tail``.
        tail : {'both', 'positive', 'negative'}
            Side of ``threshold`` to keep. Default 'both'.
        **kwargs
            Passed to :func:`~milob.imaging.surface.plot_surface_map`.

        Returns
        -------
        tuple or pyvista.Plotter
            ``(fig, axes)`` for matplotlib, or the plotter for pyvista.

        Raises
        ------
        ValueError
            If the output is not voxel-indexed, has no ``voxel_grid``, no contrast
            is given, or ``sig_mask`` is set without ``is_significant``.
        """
        from ..imaging.surface import plot_surface_map

        if self.spatial_dim != 'voxel':
            raise ValueError(
                "plot_voxel_3d needs a voxel-space output; this one is indexed "
                f"by {self.spatial_dim!r}. Use plot_glm_3d() for channel-space "
                "results."
            )
        if self.voxel_grid is None:
            raise ValueError(
                "This GLMOutput has no voxel_grid attached, so its voxels have "
                "no positions to place on a surface. It is set automatically "
                "when the GLM is fit on a voxel-indexed TissueStream; if this "
                "output was rebuilt or loaded from disk, assign it directly: "
                "output.voxel_grid = grid."
            )
        if contrast is None:
            raise ValueError("Please input a contrast.")

        selector = 'contrast' if 'contrast' in self.output.dims else 'regressor'

        sel = {selector: contrast, self.payload_dim: chromophore}
        values = self.output[val_type].sel(**sel).values.astype(float)

        if sig_mask:
            if 'is_significant' not in self.output.data_vars:
                raise ValueError(
                    "sig_mask=True needs 'is_significant' -- run "
                    "compute_contrasts() (or a group model) first."
                )
            keep = self.output.is_significant.sel(**sel).values
            values = np.where(keep, values, np.nan)

        if self._is_flag(val_type):
            cut = self._style_flag_map(kwargs)
            if threshold is None:
                threshold, tail = cut, 'positive'
            kwargs.setdefault(
                'title', f"{contrast} - {chromophore} significant (MNI space)")

        kwargs.setdefault('colorbar_label', f"{chromophore} {val_type}")
        kwargs.setdefault('title', f"{contrast} ({chromophore} {val_type}) - MNI space")
        return plot_surface_map(values, self.voxel_grid, surface=surface,
                                views=views, threshold=threshold, tail=tail,
                                **kwargs)

    def plot_topo_3d(
        self,
        sensitivity,
        contrast=None,
        val_type='t_stat',
        chromophore='HbO',
        sig_mask=False,
        mode='weighted',
        sensitivity_threshold=1e-2,
        show_channels=True,
        coreg=None,
        surface=None,
        views=('left', 'right'),
        wavelength=None,
        **kwargs,
    ):
        """
        Plot a channel-space result on the cortex through the sensitivity.

        Channel values are spread over the voxels each channel is sensitive to with
        :meth:`SensitivityOperator.backproject`. Nothing is inverted, so the map is
        topographic, not tomographic.

        Parameters
        ----------
        sensitivity : SensitivityOperator
            Operator for this probe, with its voxel grid in the frame of
            ``surface``. Channels are matched by label.
        contrast : str
            Contrast (or regressor) to show.
        val_type : str
            Variable to plot, e.g. 'beta', 't_stat' or 'is_significant'.
            Default 't_stat'.
        chromophore : str
            Payload level. Default 'HbO'.
        sig_mask : bool
            Leave out channels not flagged in ``is_significant``, using the
            'combined' flags when the output has them. Default False.
        mode : {'weighted', 'winner'}
            How overlapping channels combine; see
            :meth:`SensitivityOperator.backproject`.
        sensitivity_threshold : float
            Normalised sensitivity a channel must exceed to contribute to a voxel.
            Default 1e-2.
        show_channels : bool
            Draw the channel midpoints. Default True.
        coreg : Coregistration, optional
            Registration for placing the midpoints; must be the one the grid was
            built with. Defaults to ``probe.coreg()``.
        surface : Surface, optional
            Cortical surface. Defaults to the fsaverage pial surface.
        views : sequence
            One panel per view. Default ('left', 'right').
        wavelength : float, optional
            Use one wavelength's sensitivity instead of the average.
        **kwargs
            Passed to :func:`~milob.imaging.surface.plot_surface_map`.

        Returns
        -------
        tuple or pyvista.Plotter
            ``(fig, axes)`` for matplotlib, or the plotter for pyvista.

        Raises
        ------
        ValueError
            If the output is not channel-indexed, no contrast is given, ``sig_mask``
            is set without ``is_significant``, or ``show_channels`` is set without a
            probe.
        """
        from ..imaging.surface import plot_surface_map

        if self.spatial_dim != 'channel':
            raise ValueError(
                f"plot_topo_3d projects channel-space results; this one is "
                f"indexed by {self.spatial_dim!r}. A reconstructed output is "
                "already in voxel space -- use plot_voxel_3d()."
            )
        if contrast is None:
            raise ValueError("Please input a contrast.")

        selector = 'contrast' if 'contrast' in self.output.dims else 'regressor'
        sel = {selector: contrast, self.payload_dim: chromophore}
        values = self.output[val_type].sel(**sel).values.astype(float)
        channels = [str(c) for c in self.output.channel.values]

        if sig_mask:
            if 'is_significant' not in self.output.data_vars:
                raise ValueError(
                    "sig_mask=True needs 'is_significant' -- run "
                    "compute_contrasts() (or a group model) first."
                )
            flags = self.output.is_significant
            # combine_hb() stores its flags under 'combined'.
            payload = list(flags.coords[self.payload_dim].values) \
                if self.payload_dim in flags.coords else []
            if chromophore in payload:
                keep = flags.sel(**sel).values
            elif 'combined' in payload:
                keep = flags.sel({selector: contrast,
                                  self.payload_dim: 'combined'}).values
            else:
                keep = flags.sel({selector: contrast}).values
            values = np.where(keep, values, np.nan)

        voxel_values = sensitivity.backproject(
            values, channel_labels=channels, mode=mode,
            threshold=sensitivity_threshold, wavelength=wavelength,
        )

        if show_channels and 'channels' not in kwargs:
            if self.probe is None:
                raise ValueError(
                    "show_channels=True needs a probe on this output to place "
                    "the dots; pass show_channels=False, or coreg=... with "
                    "channels=... yourself."
                )
            coreg = coreg if coreg is not None else self.probe.coreg()
            kwargs['channels'] = coreg.mni_channel_midpoints

        if self._is_flag(val_type):
            cut = self._style_flag_map(kwargs)
            kwargs.setdefault('threshold', cut)
            kwargs.setdefault('tail', 'positive')
            kwargs.setdefault(
                'title',
                f"{contrast} - {chromophore} significant, topographic "
                f"(sensitivity >= {sensitivity_threshold:g} of channel peak)")

        kwargs.setdefault('colorbar_label', f"{chromophore} {val_type}")
        kwargs.setdefault(
            'title',
            f"{contrast} ({chromophore} {val_type}) - topographic, "
            f"sensitivity >= {sensitivity_threshold:g} of channel peak")
        return plot_surface_map(voxel_values, sensitivity.voxel_grid,
                                surface=surface, views=views, **kwargs)

    def plot_glm_3d(
        self,
        contrast=None,
        val_type='t_stat',
        chromophore='HbO',
        sig_mask=False,
        seed_channel=None,
        reference_landmarks=None,
        backend='matplotlib',
        **kwargs,
    ):
        """
        Plot GLM results on a 3-D head model in MNI space.

        Probe positions are mapped into MNI152 space by landmark-based
        rigid-body coregistration, then rendered on the scalp surface.

        Parameters
        ----------
        contrast : str
            Name of the contrast to display. Required.
        val_type : str, optional
            Variable to display, such as ``'beta'`` or ``'t_stat'``. Default
            is ``'t_stat'``.
        chromophore : str, optional
            Chromophore to display. Default is ``'HbO'``.
        sig_mask : bool, optional
            If True, show only channels flagged significant. Default is False.
        seed_channel : str, optional
            Channel used to colour the nodes when displaying a seed-based map.
        reference_landmarks : dict, optional
            MNI reference landmarks as ``{label: (x, y, z)}`` in mm. Defaults
            to the MNI152 fiducials.
        backend : {'matplotlib', 'pyvista'}, optional
            ``'matplotlib'`` renders a static figure; ``'pyvista'`` opens an
            interactive window and requires the ``threed`` extra. Default is
            ``'matplotlib'``.
        **kwargs
            Passed to :func:`milob.imaging.surface.plot_probe_3d`, which
            accepts ``surface``, ``view``, ``show_labels``, ``cmap`` and
            others. ``surface`` defaults to the MNI152 scalp, since a channel
            value is a boundary measurement with no depth resolution; pass
            ``surface='brain'`` or ``'both'`` for the cortical surface.

        Returns
        -------
        tuple or pyvista.Plotter
            ``(fig, ax)`` for the matplotlib backend, a plotter for PyVista.

        Raises
        ------
        ValueError
            If no contrast is given, or the output has no probe attached.
        """

        from ..imaging.surface import plot_probe_3d

        if contrast is None:
            raise ValueError("Please input a contrast.")
        
        # Add WARNING TO ENSURE 'COMBINED HB' HAS BEEN RAN FOR SIG MASK
        # if sig_mask and if self.output
        #     raise ValueError("Significance mask requires 'combine_hb()' to have been ran.")

        if self.probe is None:
            raise ValueError("This GLMOutput has no probe attached.")

        coreg = self.probe.coreg(reference_landmarks=reference_landmarks)  # coregister to MNI

        # Select values for plotting
        values = self.output[val_type].sel({'contrast': contrast, self.payload_dim: chromophore}).values
        channels = list(self.output.channel.values)
        colourbar_label = f"{chromophore} {val_type}"

        # Apply significance mask
        if sig_mask:
            mask = self.output.is_significant.sel(
                contrast=contrast,
                chromophore="combined"
            ).values

            values = values.copy()
            values[~mask] = np.nan

        vlim = np.nanmax(np.abs(values))    # symmetric colour scaling

        # Defaults chosen for a signed statistical map; every one of them is
        # overridable through **kwargs, as are the plot_probe_3d arguments
        # this method never sets itself (surface, show_labels, ...).
        kwargs.setdefault('cmap', 'RdBu_r')
        kwargs.setdefault('vmin', -vlim)
        kwargs.setdefault('vmax', vlim)
        kwargs.setdefault('title', f"{contrast} ({chromophore} {val_type}) - MNI Space")
        kwargs.setdefault('colorbar_label', colourbar_label)

        return plot_probe_3d(
            coreg,
            values=values,
            channel_labels=channels,
            mode='nodes',
            backend=backend,
            **kwargs,
        )