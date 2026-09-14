from typing import Optional, Union, Tuple
import numpy as np
import xarray as xr
from .datastream import MeasurementStream


class NirsStream(MeasurementStream):

    @classmethod
    def from_snirf(cls, filepath, *, sc_threshold, name=None, external_events_path=None,
                    calculate_moments=False, **kwargs):
        """
        Load a SNIRF file into the matching stream subclass.

        Auxiliary channels are ignored; use ``Session.from_snirf()`` to load them
        alongside the optical data.

        Parameters
        ----------
        filepath : str
            Path to the .snirf file.
        sc_threshold : float or None
            Source-detector distance in mm below which a channel is classified as
            short. Fixed on the Probe built for this stream. Pass None if the
            probe has no short-separation channels.
        name : str, optional
            Stream name. Derived from the modality if omitted.
        external_events_path : str, optional
            BIDS .tsv events file to use instead of the embedded stimulus data.
        calculate_moments : bool
            Convert gated TD data to moments after loading. Default False.

        Returns
        -------
        NirsStream
            A CW_Stream, TD_Stream, FD_Stream or DCS_Stream, according to the
            file's modality.
        """
        from ..io.snirf import read_snirf, get_snirf_modality

        modality = get_snirf_modality(filepath)

        if 'TD' in modality:
            from .td_nirs import TD_Stream
            target_class = TD_Stream
            default_name = "td_stream"
        elif modality == 'CW':
            from .cw_nirs import CW_Stream
            target_class = CW_Stream
            default_name = "cw_stream"
        elif modality == 'FD':
            from .fd_nirs import FD_Stream
            target_class = FD_Stream
            default_name = "fd_stream"
        elif modality in ('DCS_G2', 'DCS_BFI', 'DCS'):
            from .dcs_stream import DCS_Stream
            target_class = DCS_Stream
            default_name = "dcs_stream"
        else:
            from .auxiliary import AuxStream
            target_class = AuxStream
            default_name = "aux_stream"

        data_xr, probe, snirf_events = read_snirf(filepath, sc_threshold=sc_threshold, **kwargs)

        if modality == 'FD':
            from ..io.snirf import convert_snirf_fd_to_complex
            data_xr = convert_snirf_fd_to_complex(data_xr)

        detected_status = data_xr.attrs.get('status', 'raw')

        final_events = snirf_events
        if external_events_path:
            from .events import Events
            final_events = Events.from_bids(external_events_path)

        obj = target_class(
            data=data_xr,
            probe=probe,
            events=final_events,
            name=name or default_name,
            status=detected_status,
        )

        if external_events_path:
            import os
            obj.add_history('overwrite_events_from_bids', {
                'events_file': os.path.basename(external_events_path)
            })

        if 'TD' in modality and calculate_moments:
            if obj.status == 'raw':
                obj = obj.to_moments()

        return obj

    def to_snirf(self, filepath, ml_format='indexed'):
        """
        Write this stream to a SNIRF file.

        For writing a stream together with auxiliary channels, use
        ``Session.to_snirf()``.

        Parameters
        ----------
        filepath : str
            Destination .snirf path, created or overwritten.
        ml_format : {'indexed', 'array'}
            Encoding of the measurement list. 'indexed' (default) writes one
            ``measurementList{k}`` subgroup per column; 'array' writes a compact
            ``measurementLists`` group, which is more efficient for large channel
            counts. ``/nirs/formatVersion`` is '1.0' either way.
        """
        from ..io.snirf import write_snirf
        write_snirf(self, filepath, ml_format=ml_format)
    """
    Intermediate class for all multi-wavelength NIRS data (CW, TD, FD).
    Inherits from Datastream and provides wavelength-specific quality metrics.
    """
    
    def __init__(self, *args, **kwargs):
        # Pass everything (data, probe, name, etc.) to Datastream
        super().__init__(*args, **kwargs)

        # Validation: Ensure either physical wavelengths OR processed chromophores
        valid_dims = ['wavelength', 'chromophore']
        if not any(dim in self.data.dims for dim in valid_dims):
            raise ValueError(f"{self.__class__.__name__} must have either a 'wavelength' or 'chromophore' dimension.")


    def get_wavelengths(self) -> np.ndarray:
        """Wavelengths present in this stream, in nm."""
        return self.data.wavelength.values

    def _info_lines(self):
        lines = super()._info_lines()
        lines.append("")

        if 'wavelength' in self.data.dims:
            wls = self.data.wavelength.values.tolist()
            lines += ["--- NIRS ---", f"Wavelengths: {wls} nm"]
        elif 'chromophore' in self.data.dims:
            chroms = list(self.data.chromophore.values)
            lines += ["--- NIRS ---", f"Chromophores: {chroms}"]

        if 'is_short' in self.data.coords and 'is_bad' in self.data.coords:
            is_bad = self.data.coords['is_bad'].values
            is_short = self.data.coords['is_short'].values
            good = ~is_bad
            unit = getattr(self.probe, 'lengthUnit', None) or 'mm'
            threshold = self.data.attrs.get('sc_threshold', '?')
            lines.append(f"Short-channel threshold: {threshold} {unit}")
            lines.append(
                f"Good long: {np.sum(good & ~is_short)} | "
                f"Good short: {np.sum(good & is_short)} | "
                f"Bad: {int(np.sum(is_bad))}"
            )

        quality_coords = [c.upper() for c in ('sci', 'snr') if c in self.data.coords]
        if quality_coords:
            lines.append(f"Quality metrics: {', '.join(quality_coords)}")

        return lines




    def keep_channels(self, min_distance=None, max_distance=None, update_bad=True, show_stats=False, view=False, inplace=False):
        """
        Mark channels outside a source-detector distance range as bad.

        Short-separation channels are always kept, whatever their distance.

        Parameters
        ----------
        min_distance : float, optional
            Minimum allowed distance in mm. Shorter channels are marked bad.
        max_distance : float, optional
            Maximum allowed distance in mm. Longer channels are marked bad.
        update_bad : bool
            Write the result to the 'is_bad' coordinate. Default True.
        show_stats : bool
            Print the number of good short and long channels remaining.
        view : bool
            Open the interactive channel viewer afterwards. Default False.
        inplace : bool
            Modify this stream in place. Default False.

        Returns
        -------
        NirsStream
            Stream with the out-of-range channels marked bad.
        """
        target = self if inplace else self.copy()
        
        # Get distances in mm
        distances = np.array(self.probe.distances)
        if getattr(self.probe, "lengthUnit", "mm") == "cm":
            distances *= 10

        channels = self.data.channel.values
        is_bad = self.data.coords["is_bad"].values.copy()
        is_short = self.data.coords["is_short"].values

        # Build filtering mask
        to_mark_bad = np.zeros_like(distances, dtype=bool)
        if min_distance is not None:
            to_mark_bad |= distances < min_distance
        if max_distance is not None:
            to_mark_bad |= distances > max_distance

        # Update is_bad if requested
        if update_bad:
            channels_to_discard = channels[to_mark_bad & ~is_bad & ~is_short]

            if len(channels_to_discard) > 0:
                target.data.coords["is_bad"].loc[dict(channel=channels_to_discard)] = True
                target.add_history('keep_channels', {
                    'min_distance': min_distance, 'max_distance': max_distance,
                    'channels_marked_bad': list(channels_to_discard),
                })
                if show_stats:
                    print(f"Action: Marked {len(channels_to_discard)} channels as bad "
                          f"(dist: min={min_distance}, max={max_distance})")

        # Optional statistics
        if show_stats:
            current_bad = target.data.coords["is_bad"].values
            print("----- Channel Statistics -----")
            print(f"Total: {len(channels)} | Remaining Good: {np.sum(~current_bad)}")
            print(f"Good Short: {np.sum((~current_bad) & is_short)}")
            print(f"Good Long: {np.sum((~current_bad) & (~is_short))}")
            
        # Optional viewer
        if view:
            target.view_channel()

        return target


    def plot_probe_2d(self, **kwargs):
        return self.probe.plot_2d(**kwargs)

    def plot_probe_3d(self, **kwargs):
        return self.probe.plot_3d(**kwargs)


    def view_channel(self):
        """
        Open an interactive viewer for inspecting and toggling channel quality.

        Requires ipywidgets and a notebook front-end. Changes to channel status
        are written back to this stream's coordinates and history.
        """
        import matplotlib.pyplot as plt
        from ipywidgets import widgets
        from ..viz.spectra import plot_timeseries_psd
        from ..viz.interactive import build_widget_browser

        # 1. Identity dimensions and metadata
        type_dim = 'chromophore' if 'chromophore' in self.data.dims else 'wavelength'
        has_datatype = 'datatype' in self.data.dims
        fs = self.data.attrs.get('sampling_rate', 10.0)

        # Channel -> SD distance (mm) lookup, for display in the plot title
        distances_mm = np.array(self.probe.distances, dtype=float)
        if getattr(self.probe, "lengthUnit", "mm") == "cm":
            distances_mm = distances_mm * 10
        distance_lookup = dict(zip(self.probe.channel_labels, distances_mm))

        all_channels = list(self.data.channel.values)
        dist_min = float(np.floor(distances_mm.min()))
        dist_max = float(np.ceil(distances_mm.max()))

        def filter_channels_by_distance(change, widgets_map):
            """Return the channel labels falling within the selected distance range."""
            channel_sel = widgets_map['channels']
            lo, hi = change['new']
            in_range = [ch for ch in all_channels if lo <= distance_lookup.get(ch, np.nan) <= hi]
            if not in_range:
                return
            still_selected = [ch for ch in channel_sel.value if ch in in_range]
            channel_sel.options = in_range
            channel_sel.value = still_selected if still_selected else [in_range[0]]

        # 2. Plotting Logic
        def update_plot(channels, types, fmax, datatype):
            if not channels or not types:
                return

            # Prepare the selection dictionary for xarray
            sel_dict = { 'channel': list(channels), type_dim: list(types) }
            if has_datatype:
                sel_dict['datatype'] = datatype
            
            # Sliced data for the plot
            plot_data = self.data.sel(**sel_dict)
            
            # Check current bad status for the title
            is_bad_vals = self.data.coords['is_bad'].sel(channel=list(channels)).values
            all_bad = all(is_bad_vals)
            some_bad = any(is_bad_vals) and not all_bad
            
            title_color = 'red' if some_bad or all_bad else 'black'
            status_text = " [BAD]" if all_bad else (" [MIXED]" if some_bad else " [GOOD]")

            plt.close('all') 
            
            # Passing channels and types explicitly to match function signature
            # Overlay the stream's own markers when it has any, so the
            # signal can be read against the task (Events is falsy/empty
            # when the stream carries no triggers).
            events = self.events if (self.events and self.events.n_events > 0) else None

            fig = plot_timeseries_psd(
                plot_data, 
                channels=list(channels), 
                types=list(types), 
                fmin=0, 
                fmax=fmax,
                events=events
            )
            
            chan_labels = [
                f"{ch} ({distance_lookup[ch]:.1f}mm)" if ch in distance_lookup else ch
                for ch in channels
            ]
            fig.suptitle(f"Channels: {', '.join(chan_labels)}{status_text}",
                        color=title_color, fontsize=14, fontweight='bold')
            plt.show()

        def assemble_ui(w):
            col1 = widgets.VBox([w['dist_range'], w['channels']])
            col2 = widgets.VBox([w['types'], w['datatype'], w['fmax']])
            col3 = widgets.VBox([widgets.Label("Channel Quality:"), w['mark_bad'], w['mark_good'], w['status']])
            return widgets.HBox([col1, col2, col3], layout=widgets.Layout(padding='10px', border='1px solid #ddd'))

        built = build_widget_browser(
            widget_specs=[
                {'name': 'dist_range', 'type': 'range_slider', 'watch': False,
                 'min': dist_min, 'max': dist_max, 'step': 0.5, 'value': (dist_min, dist_max),
                 'description': 'SD Dist (mm)', 'continuous_update': False,
                 'layout': widgets.Layout(width='300px'), 'style': {'description_width': 'initial'}},
                {'name': 'channels', 'type': 'select_multiple',
                 'options': all_channels, 'value': [all_channels[0]], 'description': 'Channels',
                 'layout': widgets.Layout(width='250px', height='150px')},
                {'name': 'types', 'type': 'select_multiple',
                 'options': list(self.data.coords[type_dim].values),
                 'value': list(self.data.coords[type_dim].values),
                 'description': type_dim.capitalize(),
                 'layout': widgets.Layout(width='200px', height='150px')},
                {'name': 'fmax', 'type': 'slider',
                 'min': 0.1, 'max': fs / 2, 'step': 0.1, 'value': fs / 4,
                 'description': 'Freq Max', 'continuous_update': False},
                {'name': 'datatype', 'type': 'dropdown',
                 'options': list(self.data.datatype.values) if has_datatype else ['Default'],
                 'value': self.data.datatype.values[0] if has_datatype else 'Default',
                 'description': 'Datatype', 'disabled': not has_datatype},
                {'name': 'mark_bad', 'type': 'button', 'description': 'Mark as Bad',
                 'button_style': 'danger', 'icon': 'trash', 'layout': widgets.Layout(width='150px')},
                {'name': 'mark_good', 'type': 'button', 'description': 'Mark as Good',
                 'button_style': 'success', 'icon': 'check', 'layout': widgets.Layout(width='150px')},
                {'name': 'status', 'type': 'html', 'value': '<b>Status:</b> Ready'},
            ],
            render_fn=update_plot,
            observers=[('dist_range', filter_channels_by_distance)],
            assemble_ui=assemble_ui,
        )

        def update_channel_status(channels, state):
            """Toggle the selected channel between good and bad."""
            if not channels:
                return
            self.data.coords['is_bad'].loc[dict(channel=list(channels))] = state
            verb = "bad" if state else "good"
            built['status'].value = f"<b>Status:</b> Marked {list(channels)} as {verb}."
            # Force a plot refresh by bumping the slider slightly
            built['fmax'].value = built['fmax'].value + 0.00001
            built['fmax'].value = built['fmax'].value - 0.00001

        built['mark_bad'].on_click(lambda b: update_channel_status(built['channels'].value, True))
        built['mark_good'].on_click(lambda b: update_channel_status(built['channels'].value, False))