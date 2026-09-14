# import xarray as xr
import numpy as np
import warnings
from typing import Optional
from .events import Events

class Datastream:
    #: Explicit framework-space tag (see MeasurementStream/ParameterStream
    #: below, and the milob-package-design skill's invariant 2) -- None here
    #: is a safe default: it never equals a real tag, so an operation
    #: touching an untagged stream is conservatively classified as
    #: across-space rather than silently treated as within-space.
    SPACE = None

    #: Dimension names that can carry the *spatial* index of a stream, in
    #: priority order. A stream is indexed either by measurement position
    #: (``channel``) or, after tomographic reconstruction, by image
    #: position (``voxel``) -- never both.
    #:
    #: This is read off ``data.dims`` rather than stored as an attribute on
    #: purpose. The dims *are* the record; a second copy could disagree
    #: with them, which is the same desynchronisation ``_SIDECARS`` and
    #: ``_rebuild`` exist to prevent one level down. What genuinely does
    #: need storing is the *geometry*: ``channel`` recovers its positions
    #: from ``probe``, but ``voxel`` is a bare integer index, so a
    #: voxel-indexed stream carries a ``voxel_grid``.
    _SPATIAL_DIMS = ('channel', 'voxel')

    #: Names of DataArray attributes that travel *alongside* ``self.data``
    #: and must stay aligned with it -- fit uncertainties, observation
    #: parameters, resolution diagnostics. Declared per subclass; see
    #: _rebuild()/_carry_sidecars() for why this has to be declared rather
    #: than discovered, and invariant 7 (uncertainty is first-class and
    #: propagates) for why silently losing one is a real bug and not a
    #: cosmetic one.
    _SIDECARS = ()


    @property
    def spatial_dim(self):
        """Name of the dimension this stream is indexed by: 'channel' or 'voxel'."""
        for dim in self._SPATIAL_DIMS:
            if dim in self.data.dims:
                return dim
        raise ValueError(
            f"This stream has none of {self._SPATIAL_DIMS} as a dimension; "
            f"got {tuple(self.data.dims)}."
        )

    @property
    def n_spatial(self):
        """Size of the spatial dimension."""
        return self.data.sizes[self.spatial_dim]
    def __init__(self, data, probe=None, name=None, status="raw", events=None, history=None):
        """
        Hold a labelled data array together with its acquisition context.

        Parameters
        ----------
        data : xr.DataArray
            Values, with dims (time, channel, wavelength, ...).
        probe : Probe
            Optode geometry. Its ``sc_threshold`` drives short/long channel
            classification.
        events : Events, optional
            Markers recorded during acquisition, such as task onsets.
        name : str, optional
            Label for the stream.
        status : str
            Processing stage: 'raw', 'processed', 'od', 'conc', 'bfi', and so on.
        """
        self.data = data
        self.probe = probe
        self.name = name or "default"
        self.status = status    # could be inferred from the class itself; consider removing later
        self.events = events if events is not None else Events()

        modality = self.__class__.__name__
        if history is not None:
            self.history = history
        else:
            self.history = []
            self.add_history('initialize', {'modality': modality, 'name': self.name})

        if self.probe is not None:
            self._initialize_masks()


    @property
    def modality(self):
        """Modality of this stream, e.g. 'CW', 'TD', 'FD', 'DCS'."""
        # return self.__class__.__name__.replace('Datastream', '') # Should this be returning for example 'TD_Stream' instead of 'TD'?
        return self.__class__.__name__.replace('_Stream', '')

    @staticmethod
    def _history_entry(operation, params=None):
        """Build a provenance record with keys 'operation', 'params' and 'version'."""
        from .. import __version__
        return {'operation': operation, 'params': params or {}, 'version': __version__}

    def add_history(self, operation, params=None):
        """Append a provenance entry to this stream's history."""
        self.history.append(self._history_entry(operation, params))

    def _mark_processed(self):
        """Advance status from 'raw' to 'processed', leaving any other status unchanged."""
        if self.status == 'raw':
            self.status = 'processed'


    def _initialize_masks(self):
        """Add the 'is_bad' and 'is_short' channel coordinates if they are absent."""
        if 'channel' not in self.data.dims:
            return

        if 'is_short' not in self.data.coords:
            sc_threshold = self.probe.sc_threshold
            short_channels = self.probe.get_short_channels()
            self.data.coords['is_short'] = ('channel', short_channels)
            if sc_threshold is None:
                self.data.attrs['sc_threshold'] = None
            else:
                length_unit = getattr(self.probe, 'lengthUnit', None) or 'mm'
                self.data.attrs['sc_threshold'] = sc_threshold / 10 if length_unit == 'cm' else sc_threshold
            self.add_history('initialize_masks', {'sc_threshold': sc_threshold})

        if 'is_bad' not in self.data.coords:
            other_dims = [d for d in self.data.dims if d != 'channel']
            initially_bad = np.isnan(self.data).all(dim=other_dims).values
            self.data.coords['is_bad'] = ('channel', initially_bad)


    def roi_channel_summary(self, rois=None):
        """
        Count good, bad, short and long channels per ROI.

        Uses this stream's own 'is_bad' and 'is_short' coordinates, so results
        reflect the quality screens applied to this stream specifically.

        Parameters
        ----------
        rois : Probe or dict, optional
            A Probe carrying ROIs defined through ``add_roi()``, or a
            ``{name: [channel_labels]}`` mapping. Defaults to this stream's probe.

        Returns
        -------
        pandas.DataFrame
            One row per ROI, with columns n_channels, n_good, n_bad, n_short,
            n_long, n_good_short, n_good_long, pct_good_short and pct_good_long.
            Percentages are of each ROI's own short or long total. The
            short/long threshold in mm is attached as ``df.attrs['sc_threshold']``.

        Examples
        --------
        >>> stream.snr_screen(threshold=4.0)
        >>> stream.roi_channel_summary()
        """
        import numpy as np
        import pandas as pd

        if rois is None:
            rois = self.probe
        roi_map = rois.rois if hasattr(rois, 'rois') else rois
        if not roi_map:
            raise ValueError(
                "No ROIs given/defined. Pass rois=... or define them on the "
                "probe first via probe.add_roi()/add_roi_by_*()."
            )

        channels = self.data.channel.values
        is_bad = self.data.coords['is_bad'].values
        is_short = self.data.coords['is_short'].values
        chan_idx = {ch: i for i, ch in enumerate(channels)}

        rows = []
        for roi_name, roi_channels in roi_map.items():
            idx = np.array([chan_idx[ch] for ch in roi_channels if ch in chan_idx], dtype=int)

            roi_bad = is_bad[idx]
            roi_short = is_short[idx]
            roi_good = ~roi_bad

            n_short = int(np.sum(roi_short))
            n_long = int(np.sum(~roi_short))
            n_good_short = int(np.sum(roi_good & roi_short))
            n_good_long = int(np.sum(roi_good & ~roi_short))

            rows.append({
                'roi': roi_name,
                'n_channels': len(idx),
                'n_good': int(np.sum(roi_good)),
                'n_bad': int(np.sum(roi_bad)),
                'n_short': n_short,
                'n_long': n_long,
                'n_good_short': n_good_short,
                'n_good_long': n_good_long,
                'pct_good_short': 100.0 * n_good_short / n_short if n_short > 0 else np.nan,
                'pct_good_long': 100.0 * n_good_long / n_long if n_long > 0 else np.nan,
            })

        df = pd.DataFrame(rows).set_index('roi')

        threshold = self.data.attrs.get('sc_threshold')
        length_unit = getattr(self.probe, 'lengthUnit', None) or 'mm'
        print(f"(n_short/n_long computed at sc_threshold={threshold}{length_unit}, "
              f"this stream's own classification)")
        df.attrs['sc_threshold'] = threshold  # programmatic access, e.g. df.attrs['sc_threshold']

        return df


    # ------------------------------------------------------------------
    # Rebuilding a stream (and keeping its sidecars attached)
    # ------------------------------------------------------------------

    def _rebuild(self, new_data, *, sidecar_transform=None, operation=None,
                 propagate_sidecars=True, **overrides):
        """Build a new stream of this type around ``new_data``, carrying sidecars across."""
        import copy as _copy

        new = _copy.copy(self)
        new.data = new_data
        # copy.copy is shallow -- without this the new stream would share
        # the same list object, and appending to its history would also
        # mutate the source stream's.
        new.history = self.history.copy()
        for key, value in overrides.items():
            setattr(new, key, value)

        self._carry_sidecars(new, sidecar_transform,
                             operation=operation or 'this operation',
                             propagate=propagate_sidecars)
        return new

    def _carry_sidecars(self, target, transform=None, *, operation,
                        propagate=True):
        """Copy this stream's sidecars onto ``target``, transformed to match its data."""
        for name in self._SIDECARS:
            value = getattr(self, name, None)

            if value is None:
                setattr(target, name, None)
                continue

            if not propagate:
                setattr(target, name, None)
                warnings.warn(
                    f"{operation} has no propagation rule for '{name}', so it "
                    f"was dropped from the returned {type(self).__name__}. "
                    f"Recompute it by refitting the transformed stream.",
                    UserWarning, stacklevel=3,
                )
                continue

            setattr(target, name,
                    value.copy(deep=True) if transform is None else transform(value))

        return target

    def copy(self):
        """
        Return a deep copy of the data and every declared sidecar.

        The probe and event table are shared with the original rather than
        duplicated.

        Returns
        -------
        Datastream
            Copy of this stream.
        """
        return self._rebuild(self.data.copy(deep=True), operation='copy()')

    
    def _info_lines(self):
        lines = [f"=== {self.__class__.__name__}: {self.name} ===", ""]

        fs = self.data.attrs.get('sampling_rate')
        n_frames = len(self.data.time)
        duration_str = f"{n_frames / fs:.1f}s" if fs else f"{n_frames} frames"
        fs_str = f" @ {fs} Hz" if fs else ""
        lines += [
            "--- Data ---",
            f"Status:    {self.status}",
            f"Duration:  {duration_str}{fs_str}",
        ]

        if 'is_bad' in self.data.coords:
            is_bad = self.data.coords['is_bad'].values
            n_total = len(is_bad)
            n_bad = int(np.sum(is_bad))
            lines.append(f"Channels:  {n_total} total | {n_total - n_bad} good | {n_bad} bad")
        else:
            lines.append(f"Channels:  {len(self.data.channel)}")

        lines += ["", "--- Events ---"]
        if self.events and self.events.n_events > 0:
            by_cond = self.events.n_events_by_condition()
            cond_str = ", ".join(f"{k} ({v})" for k, v in by_cond.items())
            lines += [f"Total:      {self.events.n_events}", f"Conditions: {cond_str}"]
        else:
            lines.append("No events")

        if self.history:
            last = self.history[-1]
            last_str = last.get('operation', str(last)) if isinstance(last, dict) else str(last)
            lines += ["", "--- Processing ---", f"Last step: {last_str}"]

        return lines

    def info(self):
        print("\n".join(self._info_lines()))

    def plot_events(self, **kwargs):
        """
        Plot this stream's events against time, coloured by condition.

        The x-axis spans the recording's full time range.

        Returns
        -------
        tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        from ..viz.events import plot_events as _plot_events

        kwargs.setdefault("title", f"{self.name} - events")
        fig, ax = _plot_events(self.events, **kwargs)

        if kwargs.get("tmin") is None and kwargs.get("tmax") is None:
            t = getattr(self.data, "time", None)
            if t is not None and t.size:
                x0, x1 = ax.get_xlim()
                ax.set_xlim(min(x0, float(t.values[0])), max(x1, float(t.values[-1])))
        return fig, ax

    def __repr__(self):
        return f"<Data Stream | {self.name} | Modality: {self.modality} | Status: {self.status} | Duration: {len(self.data.time)} frames | Channels: {len(self.data.channel)}>"
    
    
    
    
    @classmethod
    def concatenate(cls, streams, name=None, preserve_gap=False):
        """
        Concatenate several optical-density streams into one.

        Parameters
        ----------
        streams : list of Datastream
            Ordered streams to concatenate, all of the same concrete type.
        name : str, optional
            Name for the output stream.
        preserve_gap : bool
            Keep any inter-run gap encoded in the streams' time coordinates.

        Returns
        -------
        Datastream
            Concatenated stream, of the same type as the inputs.

        Examples
        --------
        >>> merged = CW_Stream.concatenate([run1_od, run2_od], name="task_merged")
        """
        from .utils import concatenate_streams
        return concatenate_streams(streams, name=name, preserve_gap=preserve_gap)

    def resample(self, new_fs, inplace=False):
        """
        Resample the time axis to a new sampling rate.

        Uses polyphase resampling with an anti-aliasing filter, so downsampling is
        safe for data that is not already band-limited. Event onsets are stored in
        seconds and are carried across unchanged; sidecars carrying a 'time'
        dimension are resampled alongside the data.

        Parameters
        ----------
        new_fs : float
            Target sampling rate in Hz.
        inplace : bool
            Modify this stream in place. Default False.

        Returns
        -------
        Datastream
            Resampled stream.
        """
        import xarray as xr
        from fractions import Fraction
        from scipy.signal import resample_poly

        fs = self.data.attrs.get('sampling_rate')
        if fs is None:
            raise ValueError(
                "resample() needs the current rate in "
                "data.attrs['sampling_rate']; this stream has none.")
        fs, new_fs = float(fs), float(new_fs)
        if new_fs <= 0:
            raise ValueError(f"new_fs must be positive, got {new_fs}.")

        if 'time' not in self.data.dims:
            raise ValueError(
                f"resample() needs a 'time' dimension, got {self.data.dims}.")

        ratio = Fraction(new_fs / fs).limit_denominator(1000)
        up, down = ratio.numerator, ratio.denominator
        if up == down:
            return self if inplace else self.copy()

        time_axis = self.data.dims.index('time')
        t0 = float(self.data.time.values[0])

        def _do(da):
            if 'time' not in da.dims:
                return da
            axis = da.dims.index('time')
            vals = resample_poly(np.asarray(da.values, dtype=float), up, down,
                                 axis=axis)
            n_new = vals.shape[axis]
            new_time = t0 + np.arange(n_new) / new_fs
            coords = {k: v for k, v in da.coords.items() if k != 'time'}
            coords['time'] = new_time
            out = xr.DataArray(vals, dims=da.dims, coords=coords,
                               attrs=dict(da.attrs))
            out.attrs['sampling_rate'] = new_fs
            return out

        new_data = _do(self.data)
        if inplace:
            self.data = new_data
            self._carry_sidecars(self, _do, operation='resample')
            self.add_history('resample', {'from_fs': fs, 'new_fs': new_fs,
                                          'up': up, 'down': down,
                                          'n_frames': int(new_data.sizes['time'])})
            return self

        new = self._rebuild(new_data, sidecar_transform=_do, operation='resample')
        new.add_history('resample', {'from_fs': fs, 'new_fs': new_fs,
                                     'up': up, 'down': down,
                                     'n_frames': int(new_data.sizes['time'])})
        return new


    def time_average(self, n=None, window_s=None, *, method='mean',
                      skipna=True, partial='keep', time_label='center',
                      inplace=False):
        """
        Average consecutive, non-overlapping blocks of frames along time.

        Only the time dimension is collapsed. Event onsets are preserved and
        ``data.attrs['sampling_rate']`` is updated to the new frame rate.

        Parameters
        ----------
        n : int, optional
            Frames per block. Give exactly one of ``n`` or ``window_s``.
        window_s : float, optional
            Block width in seconds, measured on this stream's time coordinate.
        method : {'mean', 'median'}
            Reduction applied within each block. Default 'mean'.
        skipna : bool
            Ignore NaNs within a block. Default True.
        partial : {'keep', 'drop'}
            What to do with a final block shorter than the others.
        time_label : {'start', 'center', 'end'}
            Where in the block the output time coordinate is placed.
        inplace : bool
            Modify this stream in place. Default False.

        Returns
        -------
        Datastream
            Averaged stream.

        Examples
        --------
        >>> ds10 = stream.time_average(n=10)
        >>> ds1hz = stream.time_average(window_s=1.0)
        """
        from .utils import average_stream
        new = average_stream(
            self, n=n, window_s=window_s, method=method, skipna=skipna,
            partial=partial, time_label=time_label,
        )
        if inplace:
            # block averaging changes the length of the time axis, so there
            # is nothing to mutate in place -- swap this stream's state for
            # the rebuilt one and return self for pipeline chaining.
            self.__dict__.update(new.__dict__)
            return self
        return new

    def stream_slicer(self, start_idx, end_idx):
        """
        Slice the stream between two sample indices.

        Parameters
        ----------
        start_idx : int
            First sample to keep.
        end_idx : int
            One past the last sample to keep.

        Returns
        -------
        Datastream
            Stream covering the requested span.
        """
        import copy

        t_start = self.events.table.iloc[start_idx]['onset']
        t_end = self.events.table.iloc[end_idx]['onset']

        def _slice_time(da):
            # Sidecars that aren't time-indexed pass through untouched --
            # see _rebuild()'s sidecar_transform contract.
            return da.sel(time=slice(t_start, t_end)) if 'time' in da.dims else da

        new_stream = self._rebuild(
            self.data.sel(time=slice(t_start, t_end)),
            sidecar_transform=_slice_time,
            operation='stream_slicer()',
        )

        # Correct Event Table Slice
        new_stream.events = copy.deepcopy(self.events)

        table_slice = self.events.table.iloc[start_idx:end_idx+1].copy()
        table_slice.loc[:, 'onset'] = table_slice['onset'] - t_start

        new_stream.events.table = table_slice

        new_stream.add_history('stream_slicer', {'start_idx': start_idx, 'end_idx': end_idx})

        return new_stream


    def segment_by_events(self, labels=None, pad=(0.0, 0.0), min_duration=None):
        """
        Split this stream into per-condition segments using its Events table.

        Each event row becomes a window ``[onset - pad[0], onset + duration +
        pad[1]]``. Rows sharing a label stay separate rather than being merged.
        Each segment's time axis and event onsets are re-based to start at zero.

        Parameters
        ----------
        labels : str or list of str, optional
            Condition labels to extract. Defaults to every label in the events
            table.
        pad : tuple of (float, float)
            Seconds added before onset and after offset, clipped to the stream's
            time range.
        min_duration : float, optional
            Drop segments shorter than this many seconds.

        Returns
        -------
        dict of {str: list of Datastream}
            Segments per label, ordered by onset. Labels with no surviving segment
            are omitted.

        Examples
        --------
        >>> segments = od_stream.segment_by_events(labels=['rare', 'prevalent'])
        >>> rare_fc = [FC(seg).fit() for seg in segments['rare']]
        """
        import copy
        import pandas as pd

        if self.events is None or self.events.n_events == 0:
            raise ValueError(f"Stream '{self.name}' has no events to segment by.")

        table = self.events.table
        if labels is not None:
            if isinstance(labels, str):
                labels = [labels]
            table = table[table['label'].isin(labels)]

        if table.empty:
            raise ValueError(f"No events found for labels={labels} in stream '{self.name}'.")

        pre, post = pad
        t_min = float(self.data.time.values[0])
        t_max = float(self.data.time.values[-1])
        full_events = self.events.table

        segments = {}
        for label, group in table.groupby('label', sort=False):
            seg_list = []
            for _, row in group.sort_values('onset').iterrows():
                duration = 0.0 if pd.isna(row['duration']) else float(row['duration'])
                t_start = max(row['onset'] - pre, t_min)
                t_end = min(row['onset'] + duration + post, t_max)

                if t_end <= t_start:
                    continue
                if min_duration is not None and (t_end - t_start) < min_duration:
                    continue

                def _window_time(da, _s=t_start, _e=t_end):
                    # Slice *and* re-origin, exactly as the payload below --
                    # slicing alone would leave a sidecar on the old time
                    # origin. Non-time-indexed sidecars pass through.
                    if 'time' not in da.dims:
                        return da
                    w = da.sel(time=slice(_s, _e))
                    return w.assign_coords(time=w.time.values - _s)

                windowed = self.data.sel(time=slice(t_start, t_end))
                new_stream = self._rebuild(
                    windowed.assign_coords(time=windowed.time.values - t_start),
                    sidecar_transform=_window_time,
                    operation='segment_by_events()',
                )

                inside = full_events[
                    (full_events['onset'] >= t_start) & (full_events['onset'] <= t_end)
                ].copy()
                inside['onset'] = inside['onset'] - t_start

                new_stream.events = copy.deepcopy(self.events)
                new_stream.events.table = inside

                new_stream.add_history('segment_by_events', {
                    'label': label, 'onset': float(row['onset']), 'duration': duration,
                    'pad': list(pad), 't_start': t_start, 't_end': t_end,
                })

                seg_list.append(new_stream)

            # '_cond-<label>' is already unique on its own for a single
            # block; only append '_seg-<i>' when a label actually produced
            # more than one window (e.g. many short 'physical' periods
            # scattered through an uncontrolled-timing recording).
            if len(seg_list) == 1:
                seg_list[0].name = f"{self.name}_cond-{label}"
            else:
                for i, seg in enumerate(seg_list):
                    seg.name = f"{self.name}_cond-{label}_seg-{i}"

            if seg_list:
                segments[label] = seg_list

        return segments


    def frequency_filter(self, lowcut: float = None, highcut: float = None,
                order: int = 3, low_order: Optional[int] = None,
                high_order: Optional[int] = None, inplace: bool = False):
        """
        Apply a band-pass, low-pass or high-pass filter to the time axis.

        The filter type follows from which cutoffs are given. For a band-pass, the
        two edges are applied as independent stages and may use different orders.

        Parameters
        ----------
        lowcut : float, optional
            High-pass edge in Hz.
        highcut : float, optional
            Low-pass edge in Hz.
        order : int
            Butterworth order used for any edge without its own order. Default 3.
        low_order : int, optional
            Order for the low-pass edge.
        high_order : int, optional
            Order for the high-pass edge.
        inplace : bool
            Modify this stream in place. Default False.

        Returns
        -------
        Datastream
            Filtered stream.
        """
        from ..processing.filters import frequency_filter as _frequency_filter

        # Get sampling rate from the object metadata
        fs = self.data.attrs.get('sampling_rate')
        if fs is None:
            raise ValueError(f"Sampling rate not found in {self.name} metadata. Cannot filter.")

        if lowcut is None and highcut is None:
            return self if inplace else self.copy()   # No filter parameters provided, do nothing

        target = self if inplace else self.copy()
        target.data = _frequency_filter(
            target.data, lowcut=lowcut, highcut=highcut, fs=fs,
            order=order, low_order=low_order, high_order=high_order
        )

        target.add_history('frequency_filter', {
            'lowcut': lowcut, 'highcut': highcut, 'order': order,
            'low_order': low_order, 'high_order': high_order,
        })
        target._mark_processed()
        return target

    def filter(self, lowcut: float = None, highcut: float = None, order: int = 3, inplace: bool = False):
        """Alias for :meth:`frequency_filter`."""
        return self.frequency_filter(lowcut=lowcut, highcut=highcut, order=order, inplace=inplace)

    def detrend(self, recenter: bool = True, inplace: bool = False):
        """
        Remove a linear trend from each channel by least-squares line fit.

        Applies to real-valued time series. Raw FD data is complex, and TD and DCS
        data are histogram or autocorrelation curves rather than time series.

        Parameters
        ----------
        recenter : bool
            Add each channel's original mean back after detrending. If False, the
            result is zero-mean. Default True.
        inplace : bool
            Modify this stream in place. Default False.

        Returns
        -------
        Datastream
            Detrended stream.
        """
        from ..processing.filters import detrend as _detrend

        target = self if inplace else self.copy()
        target.data = _detrend(target.data, recenter=recenter)

        target.add_history('detrend', {'recenter': recenter})
        target._mark_processed()
        return target


    def tddr(self, 
             filter_cutoff: float = 0.5,
             filter_order: int = 3,
             tune: float = 4.685,
             max_iter: int = 50,
             add_high_freq: bool = True,
             channels: Optional[list] = None,
             inplace: bool = False):
        """
        Apply Temporal Derivative Distribution Repair (TDDR) motion correction.

        Parameters
        ----------
        filter_cutoff : float
            Frequency in Hz separating the low and high components. Default 0.5.
        filter_order : int
            Butterworth filter order. Default 3.
        tune : float
            Tukey biweight tuning constant. Default 4.685.
        max_iter : int
            Maximum iterations of the robust estimator. Default 50.
        add_high_freq : bool
            Add the high-frequency component back after correction. Default True.
        channels : list, optional
            Channel labels to correct. Defaults to all good channels.
        inplace : bool
            Modify this stream in place. Default False.

        Returns
        -------
        Datastream
            Motion-corrected stream.

        Examples
        --------
        >>> corrected = od.tddr()
        >>> result = raw.to_od().tddr().mbll()

        References
        ----------
        Fishburn, F. A., Ludlum, R. S., Vaidya, C. J., & Medvedev, A. V. (2019).
        Temporal derivative distribution repair (TDDR): A motion correction method
        for fNIRS. NeuroImage, 184, 171-179.
        """
        from ..processing.motion import tddr
        
        # Determine which channels to process
        # If no channels specified, use all
        target_data = self.data if channels is None else self.data.sel(channel=channels)
                    
        # Apply TDDR 
        corrected_data = tddr(
            target_data,
            filter_cutoff=filter_cutoff,
            filter_order=filter_order,
            tune=tune,
            max_iter=max_iter,
            add_high_freq=add_high_freq
        )
        
        target = self if inplace else self.copy()
        if channels is None:
            target.data = corrected_data
        else:
            # Update only specific channels safely
            target.data.loc[dict(channel=channels)] = corrected_data
        
        target.add_history('tddr', {
            'filter_cutoff': filter_cutoff,
            'filter_order': filter_order,
            'tune': tune,
            'max_iter': max_iter,
            'add_high_freq': add_high_freq,
            'channels': channels,
        })
        target._mark_processed()

        return target


    def spline_correct(self,
             k: Optional[int] = None,
             gamma: float = 0.9,
             smoothing: Optional[float] = None,
             recenter: bool = True,
             channels: Optional[list] = None,
             inplace: bool = False):
        """
        Apply spline-interpolation motion correction (MARA).

        Flags motion-contaminated segments from a moving standard deviation, then
        subtracts a cubic spline fitted to each flagged segment and restores
        baseline continuity.

        Parameters
        ----------
        k : int, optional
            Half-width in samples of the moving-standard-deviation window, which
            spans 2k+1 samples. Default round(2.5 * fs).
        gamma : float
            Threshold multiplier: samples whose moving standard deviation exceeds
            mean + gamma * std are flagged. Default 0.9.
        smoothing : float, optional
            Smoothing factor for the per-segment spline fit. Chosen automatically
            if omitted.
        recenter : bool
            Subtract each channel's mean after correction, giving a zero-mean
            result. Default True.
        channels : list, optional
            Channel labels to correct. Defaults to all channels.
        inplace : bool
            Modify this stream in place. Default False.

        Returns
        -------
        Datastream
            Motion-corrected stream.

        References
        ----------
        Scholkmann, F. et al. (2010). Physiological Measurement, 31(5), 649-662.
        Novi, S. L. et al. (2020). Neurophotonics, 7(1), 015001.
        """
        from ..processing.motion import spline_correction

        target_data = self.data if channels is None else self.data.sel(channel=channels)

        corrected_data = spline_correction(
            target_data,
            k=k,
            gamma=gamma,
            smoothing=smoothing,
            recenter=recenter,
        )

        target = self if inplace else self.copy()
        if channels is None:
            target.data = corrected_data
        else:
            target.data.loc[dict(channel=channels)] = corrected_data

        target.add_history('spline_correct', {
            'k': k,
            'gamma': gamma,
            'smoothing': smoothing,
            'recenter': recenter,
            'channels': channels,
        })
        target._mark_processed()

        return target


    def wavelet_correct(self,
             wavelet: str = 'db2',
             level: Optional[int] = None,
             alpha: Optional[float] = None,
             iqr: Optional[float] = None,
             recenter: bool = True,
             channels: Optional[list] = None,
             inplace: bool = False):
        """
        Apply wavelet-based motion artifact removal.

        Decomposes each channel by discrete wavelet transform, zeroes detail
        coefficients lying far out in the tail of a robustly fitted distribution,
        and reconstructs by inverse transform.

        Parameters
        ----------
        wavelet : str
            PyWavelets wavelet name. Default 'db2'.
        level : int, optional
            Decomposition depth. Defaults to the maximum level available for this
            wavelet and signal length.
        alpha : float, optional
            Probability threshold for zeroing a coefficient. Mutually exclusive
            with ``iqr``. Used with a value of 0.1 if neither is given.
        iqr : float, optional
            Tukey-fence multiplier: coefficients outside
            [Q1 - iqr*IQR, Q3 + iqr*IQR] are zeroed at each level. Smaller values
            are more aggressive. Mutually exclusive with ``alpha``.
        recenter : bool
            Subtract each channel's mean after correction, giving a zero-mean
            result. Default True.
        channels : list, optional
            Channel labels to correct. Defaults to all channels.
        inplace : bool
            Modify this stream in place. Default False.

        Returns
        -------
        Datastream
            Motion-corrected stream.

        References
        ----------
        Molavi, B., & Dumont, G. A. (2012). Physiological Measurement, 33(2),
        259-270.
        Novi, S. L. et al. (2020). Neurophotonics, 7(1), 015001.
        """
        from ..processing.motion import wavelet_correction

        target_data = self.data if channels is None else self.data.sel(channel=channels)

        corrected_data = wavelet_correction(
            target_data,
            wavelet=wavelet,
            level=level,
            alpha=alpha,
            iqr=iqr,
            recenter=recenter,
        )

        target = self if inplace else self.copy()
        if channels is None:
            target.data = corrected_data
        else:
            target.data.loc[dict(channel=channels)] = corrected_data

        target.add_history('wavelet_correct', {
            'wavelet': wavelet,
            'level': level,
            'alpha': (0.1 if (alpha is None and iqr is None) else alpha),
            'iqr': iqr,
            'recenter': recenter,
            'channels': channels,
        })
        target._mark_processed()

        return target


    def snr_screen(self,
                threshold: Optional[float] = 4.0,
                show_summary: bool = True,
                export_results = True,
                datatype_idx: int = 0,
                freq_range: Optional[tuple] = None,
                noise_range: Optional[tuple] = None,
                fs: Optional[float] = None,
                order: int = 3,
                window_length: Optional[float] = 5.0,
                step_size: Optional[float] = None,
                motion_floor: Optional[float] = 4.0,
                inplace: bool = False):
        """
        Compute per-channel signal-to-noise ratio and mark dead channels as bad.

        Intended for raw intensity. SNR is estimated in sliding windows; the
        median across windows is stored as ``coords['snr']`` and the fraction of
        low-SNR windows as ``coords['snr_motion_burden']``.

        Parameters
        ----------
        threshold : float, optional
            Median SNR below which a channel is marked bad. Default 4.0. Set to
            None to store the metrics without excluding any channel.
        show_summary : bool
            Print a summary table. Default True.
        export_results : bool
            Write a log to ``./data_quality/{name}_snr_quality.txt``.
        datatype_idx : int
            Index of the data type to analyse. Default 0.
        freq_range : tuple of (low, high), optional
            Frequency band in Hz treated as signal. If None, broadband SNR
            (mean/std) is used, which requires data with a real DC level.
        noise_range : tuple of (low, high), optional
            Frequency band in Hz treated as noise. Used only with ``freq_range``;
            if None, the residual outside ``freq_range`` is used.
        fs : float, optional
            Sampling rate in Hz. Read from stream metadata if omitted.
        order : int
            Butterworth filter order for band-limited mode. Default 3.
        window_length : float, optional
            Window length in seconds. Default 5.0. Set to None for a single
            whole-trace estimate, which makes motion_burden unavailable.
        step_size : float, optional
            Window step in seconds. Default is 50% overlap.
        motion_floor : float, optional
            Per-window SNR below which a window counts as motion-affected.
            Default 4.0. If None, motion_burden is reported as NaN.
        inplace : bool
            Modify this stream in place. Default False.

        Returns
        -------
        Datastream
            Stream carrying ``coords['snr']`` and ``coords['snr_motion_burden']``.
        """
        from ..processing.quality import compute_snr, quality_summary, save_summary_to_file
        import warnings

        # Compute SNR: median_snr has shape (n_channels, n_wavelengths);
        # windows has shape (n_windows, n_channels, n_wavelengths) — a single
        # "window" (the whole trace) when window_length is None.
        median_snr, windows = compute_snr(
            self.data,
            datatype_idx=datatype_idx,
            freq_range=freq_range,
            noise_range=noise_range,
            fs=fs,
            order=order,
            window_length=window_length,
            step_size=step_size,
            return_windows=True,
        )

        target = self if inplace else self.copy()

        # Reduce across wavelengths: worst-wavelength median SNR per channel
        with np.errstate(divide='ignore', invalid='ignore'), \
            warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='All-NaN slice encountered')

                if median_snr.ndim > 1:
                    min_snr = np.nanmin(median_snr, axis=1)
                    windows_min = np.nanmin(windows, axis=2)  # (n_windows, n_channels)
                else:
                    min_snr = median_snr
                    windows_min = windows

                # Fraction of windows below motion_floor, per channel.
                # motion_floor is independent of threshold — if it's None,
                # there's nothing to compare windows against, so report NaN
                # rather than guessing a floor.
                if motion_floor is not None:
                    motion_burden = np.nanmean(windows_min < motion_floor, axis=0)
                else:
                    motion_burden = np.full(min_snr.shape, np.nan)

        target.data.coords['snr'] = ('channel', min_snr)
        target.data.coords['snr_motion_burden'] = ('channel', motion_burden)

        # Generate summary. threshold=None here means "informational only" —
        # quality_summary skips the good/bad classification in that case.
        summary = quality_summary(
            min_snr,
            channel_labels=self.data.channel.values,
            threshold=threshold,
            metric_name="SNR",
            verbose=show_summary,
            is_short=self.data.coords['is_short'].values,
            extra_columns={'motion_burden': motion_burden},
        )

        # Determine dead channels: median SNR (worst wavelength) below floor.
        # threshold=None means "don't exclude any channel" — metrics are still
        # computed and stored above, just no marking happens.
        if threshold is not None:
            if median_snr.ndim > 1:
                is_bad_mask = np.any(median_snr < threshold, axis=1)
            else:
                is_bad_mask = median_snr < threshold

            # Standardise NaNs as bad
            is_bad_mask |= np.isnan(min_snr)
            bad_labels = self.data.channel.values[np.where(is_bad_mask)[0]]
        else:
            bad_labels = np.array([], dtype=self.data.channel.values.dtype)

        # Automated Export
        if export_results:
            import os

            export_dir = "./data_quality"
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)

            fname = os.path.join(export_dir, f"{self.name}_snr_quality.txt")
            save_summary_to_file(summary, fname)
            if show_summary:
                print(f"Quality log exported to {fname}")


        # History and feedback
        target.add_history('snr_screen', {
            'threshold': threshold,
            'freq_range': freq_range,
            'noise_range': noise_range,
            'window_length': window_length,
            'step_size': step_size,
            'motion_floor': motion_floor,
            'logic': 'any_wavelength_median_below_threshold',
            'n_bad_marked': len(bad_labels)
        })


        # Apply the marks
        if len(bad_labels) > 0:
            target.mark_bad_channels(bad_labels, inplace=True)

        if show_summary:
            if threshold is not None:
                print(f"SNR Screen: Marked {len(bad_labels)} channels as dead (Any Wavelength median SNR < {threshold})")
            else:
                print("SNR Screen: threshold=None, no channels excluded (metrics stored only)")

        target._mark_processed()
        return target
    
    
    def mark_bad_channels(self, channels, inplace=True):
        """
        Mark channels as bad.

        Parameters
        ----------
        channels : str or list of str
            Channel label(s) to mark.
        inplace : bool
            Modify this stream in place. Default False.

        Returns
        -------
        Datastream
            Stream with the channels marked in ``coords['is_bad']``.

        Examples
        --------
        >>> stream.mark_bad_channels('S1D2')
        >>> stream.mark_bad_channels(['S1D2', 'S3D4'])
        """
        target = self if inplace else self.copy()
        
        if isinstance(channels, str):
            channels = [channels]
            
        try:
            target.data.coords['is_bad'].loc[dict(channel=list(channels))] = True
        except KeyError:
            print(f"Warning: Channels {channels} were not found in the dataset.")
        
        # Ensure history is updated on the correct object
        target.add_history('mark_bad_channels', {'channels': channels})

        return target

    def drop_channels(self, channels):
        """
        Remove channels from the stream and the probe permanently.

        Source and detector positions are retained, so channels sharing a source
        or detector with a dropped channel are unaffected.

        Parameters
        ----------
        channels : str or list of str
            Channel label(s) to drop, e.g. ``'S1D4'``.

        Returns
        -------
        Datastream
            Stream with the channels removed.

        Examples
        --------
        >>> stream = stream.drop_channels('S1D4')
        >>> stream = stream.drop_channels(['S1D4', 'S2D6'])
        """
        import copy

        if isinstance(channels, str):
            channels = [channels]
        channels = list(channels)

        existing = list(self.data.channel.values)
        missing  = [ch for ch in channels if ch not in existing]
        if missing:
            raise ValueError(
                f"Channel(s) not found in stream: {missing}. "
                f"Available channels (first 10): {existing[:10]}"
            )

        # ── Drop from xarray ─────────────────────────────────────────────
        new_data = self.data.drop_sel(channel=channels)

        # ── Update probe ─────────────────────────────────────────────────
        new_probe = None
        if self.probe is not None:
            drop_set  = set(channels)
            all_labels = self.probe.channel_labels
            keep_mask  = np.array([lbl not in drop_set for lbl in all_labels])

            new_probe = copy.copy(self.probe)
            new_probe._channels = {
                k: np.asarray(v)[keep_mask]
                for k, v in self.probe._channels.items()
            }
            new_probe._channel_labels = None   # clear label cache
            new_probe._distances = None        # clear distance cache (indices shift when channels are dropped)

            # Remove dropped channels from any ROI definitions
            if new_probe.rois:
                new_probe.rois = {
                    roi: [ch for ch in chs if ch not in drop_set]
                    for roi, chs in new_probe.rois.items()
                }

        def _drop_channels(da):
            # Sidecars without a 'channel' dim pass through -- see
            # _rebuild()'s sidecar_transform contract.
            return da.drop_sel(channel=channels) if 'channel' in da.dims else da

        new_stream = self._rebuild(
            new_data,
            sidecar_transform=_drop_channels,
            operation='drop_channels()',
            probe=new_probe,
        )
        new_stream.add_history('drop_channels', {'channels': channels})
        return new_stream


    def drop_bad_channels(self, protect_short=True, show_stats=False):
        """
        Remove every channel currently marked bad.

        Which channels are bad is evaluated at the moment of the call, so the
        result depends on which quality screens have already run.

        Parameters
        ----------
        protect_short : bool
            Keep short-separation channels even when marked bad, so they remain
            available as regressors. Default True.
        show_stats : bool
            Print how many channels were dropped. Default True.

        Returns
        -------
        Datastream
            Stream with the bad channels removed.

        Examples
        --------
        >>> stream = stream.snr_screen().drop_bad_channels()
        """
        channels = np.asarray(self.data.channel.values)
        is_bad = np.asarray(self.data.coords['is_bad'].values)

        if protect_short and 'is_short' in self.data.coords:
            is_short = np.asarray(self.data.coords['is_short'].values)
            to_drop = channels[is_bad & ~is_short]
            n_protected = int(np.sum(is_bad & is_short))
        else:
            to_drop = channels[is_bad]
            n_protected = 0

        to_drop = to_drop.tolist()

        if show_stats:
            print("----- drop_bad_channels -----")
            print(f"Dropped: {len(to_drop)} | Remaining: {len(channels) - len(to_drop)}")
            if n_protected:
                print(f"Kept {n_protected} bad short channel(s) (protect_short=True)")

        new_stream = self.drop_channels(to_drop) if to_drop else self.copy()
        new_stream.add_history('drop_bad_channels', {
            'protect_short': protect_short,
            'channels_dropped': to_drop,
            'bad_short_channels_kept': n_protected,
        })
        return new_stream


class MeasurementStream(Datastream):
    """
    Base class for streams holding acquired measurements.

    Marks a stream as living in measurement space, as opposed to the parameter
    space of :class:`ParameterStream`.
    """
    SPACE = 'M'


class ParameterStream(Datastream):
    """
    Base class for streams holding quantities recovered by a fit.

    Marks a stream as living in parameter space, as opposed to the measurement
    space of :class:`MeasurementStream`. Subclasses include
    :class:`~milob.core.opt_prop_stream.OptPropStream` and
    :class:`~milob.core.tissue_stream.TissueStream`.
    """
    pass

