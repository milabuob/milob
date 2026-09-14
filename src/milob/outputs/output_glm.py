from abc import ABC
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
        #: Geometry for a 'voxel'-indexed output -- what `probe` is to a
        #: 'channel'-indexed one. Carried so a reconstructed result can be
        #: plotted on anatomy later (`plot_voxel_3d`); None for channel-space
        #: outputs. See `analysis.glm.GLM._package_results`.
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
    def average(cls, outputs, alpha=0.05):
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

        df_total  = sum(o.output.attrs.get('df', 100) for o in outputs)
        t_values  = np.where(se_avg > 0, beta_avg / se_avg, np.nan)
        p_values  = 2 * (1 - stats.t.cdf(np.abs(t_values), df_total))
        p_upper   = stats.t.sf(t_values, df_total)
        p_lower   = stats.t.cdf(t_values, df_total)

        pdim      = next(d for d in ('chromophore', 'component', 'wavelength')
                         if d in ref.dims)
        chrom     = list(ref.coords[pdim].values)
        is_sig    = np.zeros_like(beta_avg, dtype=bool)
        for i, c in enumerate(chrom):
            if c in ('HbO', 'HbT'):
                is_sig[..., i] = p_upper[..., i] < alpha
            elif c == 'HbR':
                is_sig[..., i] = p_lower[..., i] < alpha

        dims   = list(ref['beta'].dims)
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
            attrs={'df': df_total},
        )
        return cls(data=ds, probe=outputs[0].probe,
                   analysis_type=outputs[0].analysis_type,
                   voxel_grid=getattr(outputs[0], 'voxel_grid', None),
                   history=cls._consolidate_histories(outputs, 'GLMOutput.average'))

    
    def compute_contrasts(self, contrast_dict, alpha=0.05, fdr_correction=False):
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
            if isinstance(c_logic, dict):
                c_vector = np.zeros(len(reg_names))
                for label, weight in c_logic.items():
                    if label in reg_names:
                        idx = reg_names.index(label)
                        c_vector[idx] = weight
                    else:
                        print(f"Warning: Regressor '{label}' not found in this session.")
            else:
                # Fallback for fixed lists (c vector directly)
                c_vector = np.array(c_logic)
            
            # Convert contrast to a named DataArray for automatic alignment
            c = xr.DataArray(c_vector, coords={'regressor': reg_names}, dims=['regressor'])
            
            # Calculate Contrast Beta: c' * Beta
            # Sums across the 'regressor' dimension
            con_beta = (self.output.beta * c).sum(dim='regressor')
            
            # Calculate Contrast Variance: c' * Cov * c
            cj = c.rename({'regressor': 'regressor_j'})
            ci = c.rename({'regressor': 'regressor_i'})
            tmp = (self.output.covariance * cj).sum(dim='regressor_j')
            con_var = (tmp * ci).sum(dim='regressor_i')
            
            # Calculate Stats
            beta_vals = con_beta.values
            var_vals  = con_var.values.clip(min=1e-15)
            se_values = np.sqrt(var_vals)

            t_values = beta_vals / se_values
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df)) # two-tailed

            p_uncorrected = p_values
            if fdr_correction:
                p_values = _fdr_bh(p_values, axis=core_dims.index(self.spatial_dim))

            # Calculating one-tailed p-value for is_significant mask to account for expected response
            p_upper = stats.t.sf(t_values, df)
            p_lower = stats.t.cdf(t_values, df)

            pdim = next(d for d in ('chromophore', 'component', 'wavelength')
                        if d in con_beta.dims)
            chrom = list(con_beta.coords[pdim].values)
            is_sig = np.zeros_like(beta_vals, dtype=bool)

            # for i, chrom in enumerate(chrom):
            #     if chrom == "HbO":
            #         is_sig[:, i] = (p_upper[:, i] < alpha)
            #     elif chrom == "HbR":
            #         is_sig[:, i] = (p_lower[:, i] < alpha)
            #     elif chrom == "HbT":
            #         is_sig[:, i] = p_upper[:, i] < alpha

            is_sig = p_values < alpha # not direction-specific

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
        
        # Carry the input's provenance (stream history + GLM.fit) forward and
        # record this step, so an "Inference" output stays self-documenting.
        history = list(self.history) + [self._history_entry(
            'GLMOutput.compute_contrasts',
            {'contrasts': list(contrast_dict.keys()),
             'alpha': alpha,
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


    def roi_average(self, rois, alpha: float = 0.05) -> "GLMOutput":
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
        from scipy import stats as scipy_stats

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

            new_p   = 2 * (1 - scipy_stats.t.cdf(np.abs(t_roi_vals), df_val))
            new_p   = np.clip(new_p, 0.0, 1.0)
            new_sig = new_p < alpha

            non_ch_dims   = [d for d in block.beta.dims if d != 'channel']
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
        return GLMOutput(combined, probe=out_probe, analysis_type=self.analysis_type + '_ROI')


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