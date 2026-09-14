from abc import ABC
import xarray as xr
import numpy as np

class BaseOutput(ABC):
    def __init__(self, data, probe=None, analysis_type=None, history=None):
        self.output = data      # generic xarray Dataset containing results
        self.probe = probe      # Needed for topographic plotting
        self.analysis_type = analysis_type
        self.history = history or []   # provenance: input stream(s) + this analysis step

    @staticmethod
    def _history_entry(operation, params=None):
        """Build a history entry with the same schema used by Datastream."""
        from .. import __version__
        return {'operation': operation, 'params': params or {}, 'version': __version__}

    @staticmethod
    def _strip_identity(entry):
        """Drop per-source identity fields so histories of the same recipe compare equal."""
        if not isinstance(entry, dict):
            return entry
        params = {k: v for k, v in entry.get('params', {}).items()
                  if k not in ('name', 'source_stream_name')}
        return {**entry, 'params': params}

    @staticmethod
    def _consolidate_histories(outputs, operation_name):
        """Merge the histories of several outputs into one shared recipe plus source names."""
        from .. import __version__

        histories = [getattr(o, 'history', None) or [] for o in outputs]
        recipes = [[BaseOutput._strip_identity(e) for e in h] for h in histories]
        if recipes and any(r != recipes[0] for r in recipes):
            print(f"Warning: {operation_name} is combining outputs whose "
                  f"processing/analysis recipe differs (beyond source identity) "
                  f"-- the combined result's recorded history reflects only the "
                  f"first output's recipe and may not represent every input "
                  f"faithfully.")

        def _source_name(h):
            if h and isinstance(h[-1], dict):
                return h[-1].get('params', {}).get('source_stream_name')
            return None

        base = histories[0] if histories else []
        return base + [{
            'operation': operation_name,
            'params': {'n_sources': len(outputs), 'source_stream_names': [_source_name(h) for h in histories]},
            'version': __version__,
        }]

    def __repr__(self):
        # Check if output exists before accessing .sizes
        if self.output is not None:
            dims = dict(self.output.sizes)
            dim_str = f"Dims: {dims}"
        else:
            dim_str = "Data: Not yet computed"
        return f"<milob Output | Type: {self.analysis_type} | {dim_str}>"
    

    def save(self, filepath: str):
        """
        Write the output to a NetCDF file.

        Parameters
        ----------
        filepath : str
            Destination path. A ``.nc`` extension is added if absent.

        Notes
        -----
        The analysis type and processing history are stored as file
        attributes, so a saved output can be reloaded with :meth:`load`.
        """
        import json

        attrs = dict(self.output.attrs)
        if self.analysis_type:
            attrs['analysis_type'] = self.analysis_type
        # History is provenance (invariant: every operation is traceable),
        # so it has to survive a save/load round-trip too, not just the
        # data -- NetCDF attrs are flat (no nested dicts/lists), so this
        # is stored as one JSON string and parsed back out in load().
        # default=str rather than a strict encoder: history entries are
        # provenance metadata to read, not data fed back into computation,
        # so a non-JSON-native value (e.g. a stray numpy scalar in some
        # 'params') degrading to its string form on save is an acceptable
        # trade for never crashing a save over it.
        attrs['history_json'] = json.dumps(self.history, default=str)

        # NetCDF attrs accept only str/Number/ndarray/list/tuple/bytes --
        # None is a real value several methods leave in attrs (e.g.
        # wavelet_coherence's s0/j1 when left at their auto-computed
        # defaults, see _fit_wavelet_coherence's value_meta), and isn't one
        # of those types. Round-tripped through a private sentinel string
        # here and converted back in load() -- not dropped outright, since
        # "this setting was left at its default" is itself meaningful
        # provenance, not something to silently lose on save. Bools need
        # the same treatment for the same reason (e.g. has_combined_metrics
        # =True).
        attrs = {
            k: ('__none__' if v is None else int(v) if isinstance(v, (bool, np.bool_)) else v)
            for k, v in attrs.items()
        }

        ds = self.output.assign_attrs(attrs)

        # NetCDF4 does not support boolean dtypes — cast to int8 at the I/O
        # boundary. assign() also returns a new Dataset, so this doesn't
        # touch self.output either.
        bool_vars = [v for v in ds.data_vars if ds[v].dtype == np.dtype(bool)]
        if bool_vars:
            ds = ds.assign({v: ds[v].astype(np.int8) for v in bool_vars})

        ds.to_netcdf(filepath)

    @classmethod
    def load(cls, filepath: str):
        """
        Load an output previously written by :meth:`save`.

        Called on a concrete subclass, it returns an instance of that
        subclass. The probe is not stored by :meth:`save` and is therefore
        not restored; assign one afterwards if plotting requires it.

        Parameters
        ----------
        filepath : str
            Path to a NetCDF file written by :meth:`save`.

        Returns
        -------
        BaseOutput
            An instance of the subclass on which the method was called.
        """
        import json

        ds = xr.open_dataset(filepath)
        analysis_type = ds.attrs.pop('analysis_type', None)
        history = json.loads(ds.attrs.pop('history_json', '[]'))
        # Reverse save()'s '__none__' sentinel round-trip. isinstance-guarded:
        # some attrs are array-valued (e.g. a saved 'range' tuple comes back
        # as an ndarray), and `array == '__none__'` is an elementwise
        # comparison, not the scalar bool this check needs.
        ds.attrs = {
            k: (None if isinstance(v, str) and v == '__none__' else v)
            for k, v in ds.attrs.items()
        }
        return cls(data=ds, probe=None, analysis_type=analysis_type, history=history)

    

    def plot_topographic_map(self, contrast: str, val_type: str = 'beta', chromophore: str = 'HbO', bin_idx: int = 0, **kwargs):
        """
        Plot a 2-D topographic map of one contrast.

        Parameters
        ----------
        contrast : str
            Name of the contrast or regressor to display.
        val_type : str, optional
            Variable to map, such as ``'beta'`` or ``'tstat'``. Default is
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
        data_array = self.output[val_type].sel({selector: contrast, 'chromophore': chromophore})

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

        return plot_topo_map(self.probe, values=values, **kwargs)




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
        kwargs.setdefault('colorbar_label', 'Value')
        return plot_matrix(self.output['value'].values, **kwargs)


    
    def plot_tstat_map_multi(self, regressor, p_threshold: float=0.05, condition: str='hbo_and_hbr', **kwargs):
        from ..viz.topo import plot_tstat_conjunction_multi
        from ..viz import theme
        import matplotlib.pyplot as plt
        import numpy as np

        if isinstance(regressor, str):
            regressors = [regressor]
        else:
            regressors = regressor

        num_regs = len(regressors)
        fig, axes = plt.subplots(1, num_regs, figsize=theme.spatial_panel_figsize(1, num_regs), squeeze=False)

        # Calculate global vmax for consistent colorbar across plots
        all_tstats = []
        for reg in regressors:
            # We only care about t-stats that will be plotted (HbO betas)
            val = self.output['t_stat'].sel(contrast=reg, chromophore='HbO').values
            all_tstats.append(val)
        
        # Determine the highest t-stat across all channels and all regressors
        global_vmax = np.nanmax(all_tstats) if len(all_tstats) > 0 else 1.0
        if global_vmax <= 0: global_vmax = 1.0 # Avoid scale errors

        # Plotting
        for i, reg in enumerate(regressors):
            ax = axes[0, i]
            
            hbo_tstats = self.output['t_stat'].sel(contrast=reg, chromophore='HbO').values
            hbo_p = self.output['p_val'].sel(contrast=reg, chromophore='HbO').values
            hbr_p = self.output['p_val'].sel(contrast=reg, chromophore='HbR').values
            hbo_betas = self.output['beta'].sel(contrast=reg, chromophore='HbO').values
            hbr_betas = self.output['beta'].sel(contrast=reg, chromophore='HbR').values

            plot_tstat_conjunction_multi(
                self.probe, hbo_tstats, hbo_p, hbr_p, hbo_betas, hbr_betas, 
                p_threshold=p_threshold,
                condition=condition,
                ax=ax, 
                title=f"Regressor: {reg}",
                vmax=global_vmax # Pass the consistent max value
            )

        # Overall Figure Title
        if condition == 'hbo_and_hbr':
            cond_text = r"HbO$\uparrow$ AND HbR$\downarrow$"
        elif condition == 'hbo':
            cond_text = r"HbO$\uparrow$"
        elif condition == 'hbr':
            cond_text = r"HbR$\downarrow$"


        fig.suptitle(fr"Conjunction Activation Maps ({cond_text}, $p < {p_threshold}$)" + "\n" + 
                     "Heatmap: Beta $t$-statistics for HbO", 
                     fontsize=16, fontweight='bold', y=0.95)

        plt.tight_layout(rect=[0, 0.03, 1, 0.9]) # Adjust layout to make room for suptitle
        return fig, axes