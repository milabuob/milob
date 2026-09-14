import numpy as np


def concatenate_streams(streams, name=None, preserve_gap=False):
    """
    Concatenate two or more optical-density streams from the same session.

    Parameters
    ----------
    streams : list of Datastream
        Ordered streams to concatenate. All must be the same concrete type,
        have status 'od', and share channel set, extra dimensions and
        sampling rate.
    name : str, optional
        Name for the output stream. Defaults to 'concat_<name1>_<name2>_...'.
    preserve_gap : bool
        If True, a non-zero starting value on a stream's time coordinate is
        treated as the intended gap before that run. If False (default), runs
        are made continuous, each starting one sample period after the last.

    Returns
    -------
    Datastream
        Concatenated stream of the same type as the inputs, with merged events
        (onsets shifted to the new time axis) and the union of bad-channel
        masks.

    Raises
    ------
    ValueError
        If fewer than two streams are given, a stream is not in OD state, or
        sampling rates, channels or dimensions do not match.
    TypeError
        If the streams are not all the same concrete type.
    """
    import xarray as xr
    import pandas as pd
    from .events import Events

    # ------------------------------------------------------------------ #
    # 1. Validation                                                        #
    # ------------------------------------------------------------------ #
    if len(streams) < 2:
        raise ValueError("At least 2 streams are required for concatenation.")

    stream_type = type(streams[0])
    if not all(type(s) is stream_type for s in streams):
        type_names = [type(s).__name__ for s in streams]
        raise TypeError(
            f"All streams must be the same type. Got: {type_names}."
        )

    for i, s in enumerate(streams):
        if s.status != 'od':
            raise ValueError(
                f"Stream {i} ('{s.name}') has status='{s.status}'. "
                "Convert all streams to OD with to_od() before concatenating."
            )

    fs_list = [s.data.attrs.get('sampling_rate') for s in streams]
    if len(set(fs_list)) > 1:
        raise ValueError(
            f"Streams have mismatched sampling rates: {fs_list}. "
            "Resample to a common rate before concatenating."
        )
    fs = fs_list[0]

    ref_channels = streams[0].data.coords['channel'].values
    for i, s in enumerate(streams[1:], 1):
        if not np.array_equal(ref_channels, s.data.coords['channel'].values):
            raise ValueError(
                f"Stream {i} ('{s.name}') has a different channel set from "
                f"stream 0 ('{streams[0].name}'). All streams must share "
                "identical channel configurations."
            )

    extra_dims = [d for d in streams[0].data.dims if d not in ('time', 'channel')]
    for i, s in enumerate(streams[1:], 1):
        for dim in extra_dims:
            if dim not in s.data.dims:
                raise ValueError(
                    f"Stream {i} ('{s.name}') is missing dimension '{dim}'."
                )
            if not np.array_equal(
                streams[0].data.coords[dim].values,
                s.data.coords[dim].values,
            ):
                raise ValueError(
                    f"Stream {i} ('{s.name}') has different '{dim}' values "
                    f"from stream 0 ('{streams[0].name}')."
                )

    # ------------------------------------------------------------------ #
    # 2. Compute per-stream time offsets                                  #
    # ------------------------------------------------------------------ #
    dt = 1.0 / fs if fs else float(
        np.diff(streams[0].data.time.values[:2])[0]
    )

    time_offsets = [0.0]
    prev_end = float(streams[0].data.time.values[-1])

    for stream in streams[1:]:
        t_start = float(stream.data.time.values[0])
        gap = t_start if (preserve_gap and t_start > 0) else dt
        offset = prev_end + gap - t_start
        time_offsets.append(offset)
        prev_end = float(stream.data.time.values[-1]) + offset

    # ------------------------------------------------------------------ #
    # 3. Collect bad-channel union before dropping per-channel coords     #
    # ------------------------------------------------------------------ #
    channel_scalar_coords = [
        c for c in streams[0].data.coords
        if streams[0].data.coords[c].dims == ('channel',) and c != 'channel'
    ]

    bad_union = None
    if 'is_bad' in streams[0].data.coords:
        bad_union = streams[0].data.coords['is_bad'].values.copy()
        for s in streams[1:]:
            if 'is_bad' in s.data.coords:
                bad_union = bad_union | s.data.coords['is_bad'].values

    # ------------------------------------------------------------------ #
    # 4. Build shifted arrays                                             #
    # ------------------------------------------------------------------ #
    shifted_arrays = []
    for stream, offset in zip(streams, time_offsets):
        da = stream.data
        # Drop per-channel scalar coords so xr.concat never sees conflicts
        # (e.g. is_bad differs between runs; _initialize_masks will restore them)
        drop = [c for c in channel_scalar_coords if c in da.coords]
        if drop:
            da = da.drop_vars(drop)
        if offset != 0.0:
            da = da.assign_coords(time=da.time.values + offset)
        shifted_arrays.append(da)

    concatenated_data = xr.concat(shifted_arrays, dim='time')
    concatenated_data.attrs = streams[0].data.attrs.copy()

    # ------------------------------------------------------------------ #
    # 5. Merge events (shift onsets to match the new time axis)           #
    # ------------------------------------------------------------------ #
    event_tables = []
    for stream, offset in zip(streams, time_offsets):
        if stream.events and stream.events.n_events > 0:
            tbl = stream.events.table.copy()
            tbl['onset'] = tbl['onset'] + offset
            event_tables.append(tbl)

    if event_tables:
        merged_df = (
            pd.concat(event_tables, ignore_index=True)
            .sort_values('onset')
            .reset_index(drop=True)
        )
        merged_events = Events.from_dataframe(merged_df)
    else:
        merged_events = Events()

    # ------------------------------------------------------------------ #
    # 6. Construct output stream                                          #
    # ------------------------------------------------------------------ #
    out_name = name or "concat_" + "_".join(s.name for s in streams)

    result = stream_type(
        data=concatenated_data,
        probe=streams[0].probe,
        name=out_name,
        status='od',
        events=merged_events,
    )

    # Apply the union of bad channels from all input runs
    if bad_union is not None:
        bad_labels = result.data.coords['channel'].values[bad_union]
        if len(bad_labels) > 0:
            result.mark_bad_channels(bad_labels, inplace=True)

    run_names = [s.name for s in streams]
    result.add_history('concatenate', {'run_names': run_names, 'preserve_gap': preserve_gap})

    return result


def average_stream(stream, *, n=None, window_s=None, method='mean',
                         skipna=True, partial='keep', time_label='center',
                         name=None):
    """
    Average consecutive, non-overlapping blocks of frames along time.

    Only the time dimension is collapsed, so this applies to any stream type
    and any extra dimensions. For complex FD data, 'mean' is a coherent
    average and 'median' is rejected.

    Parameters
    ----------
    stream : Datastream
        Any stream whose data has a time dimension.
    n : int, optional
        Frames per block. Give exactly one of ``n`` or ``window_s``.
    window_s : float, optional
        Block width in seconds, measured on the stream's time coordinate.
        Tolerates a non-uniform time axis.
    method : {'mean', 'median'}
        Reduction applied within each block. Default 'mean'.
    skipna : bool
        Ignore NaNs within a block. Default True.
    partial : {'keep', 'drop'}
        Whether to average or discard a final, incompletely filled block.
        Default 'keep'.
    time_label : {'center', 'start', 'end'}
        Timestamp given to each output frame. Default 'center'.
    name : str, optional
        Name for the returned stream. Defaults to the input stream's name.

    Returns
    -------
    Datastream
        New stream with a shortened time axis and an updated
        ``data.attrs['sampling_rate']``. Event onsets carry through unchanged.
        Sidecars such as fit uncertainties are dropped with a warning.

    Raises
    ------
    ValueError
        If neither or both of ``n`` and ``window_s`` are given, on a
        non-positive value, if the stream has no time dimension, if
        ``method='median'`` is used on complex data, or if ``partial='drop'``
        removes every block.

    Examples
    --------
    >>> ds10 = average_stream(stream, n=10)
    >>> ds1hz = average_stream(stream, window_s=1.0)
    """
    import xarray as xr

    if (n is None) == (window_s is None):
        raise ValueError("Provide exactly one of n= or window_s=.")
    if method not in ('mean', 'median'):
        raise ValueError(f"method must be 'mean' or 'median', got {method!r}.")
    if partial not in ('keep', 'drop'):
        raise ValueError(f"partial must be 'keep' or 'drop', got {partial!r}.")
    if time_label not in ('center', 'start', 'end'):
        raise ValueError(
            f"time_label must be 'center', 'start' or 'end', got {time_label!r}.")

    data = stream.data
    if 'time' not in data.dims:
        raise ValueError(
            f"Stream '{stream.name}' has no 'time' dimension to average over.")
    if method == 'median' and np.iscomplexobj(data.values):
        raise ValueError(
            "method='median' is undefined for complex (FD) data; use "
            "method='mean' for average.")

    t = np.asarray(data.time.values, dtype=float)
    N = t.size
    dt = float(np.median(np.diff(t))) if N > 1 else 0.0

    # --- form blocks as lists of frame indices --------------------------- #
    if n is not None:
        n = int(n)
        if n < 1:
            raise ValueError(f"n must be a positive integer, got {n}.")
        segments = [np.arange(s, min(s + n, N)) for s in range(0, N, n)]
        if partial == 'drop' and segments and segments[-1].size < n:
            segments = segments[:-1]
    else:
        w = float(window_s)
        if w <= 0:
            raise ValueError(f"window_s must be positive, got {w}.")
        span = t[-1] - t[0] if N else 0.0
        n_bins = max(1, int(np.ceil(span / w - 1e-9)))
        edges = t[0] + w * np.arange(n_bins + 1)
        edges[-1] = max(edges[-1], t[-1]) + 1e-9
        n_valid = n_bins
        if partial == 'drop' and not np.isclose(n_bins * w, span):
            n_valid = n_bins - 1                 # trailing bin not fully covered
        bin_of = np.digitize(t, edges) - 1
        segments = [np.where(bin_of == b)[0] for b in range(n_valid)]
        segments = [seg for seg in segments if seg.size]

    if not segments:
        raise ValueError(
            "partial='drop' removed every block -- the record is shorter than "
            "one full block.")

    # --- per-block timestamps + block-id map ---------------------------- #
    block_id = np.full(N, -1, dtype=int)
    new_times = np.empty(len(segments), dtype=float)
    for k, seg in enumerate(segments):
        block_id[seg] = k
        tk = t[seg]
        new_times[k] = (float(tk.mean()) if time_label == 'center'
                        else float(tk[0]) if time_label == 'start'
                        else float(tk[-1]))

    keep = block_id >= 0
    sub = data.isel(time=np.where(keep)[0])
    gid = xr.DataArray(block_id[keep], dims='time', coords={'time': sub.time})
    grouped = sub.groupby(gid.rename('__block__'))
    reduced = (grouped.mean('time', skipna=skipna) if method == 'mean'
               else grouped.median('time', skipna=skipna))
    out = (reduced.rename({'__block__': 'time'})
                  .assign_coords(time=new_times)
                  .transpose(*data.dims))
    out.attrs = dict(data.attrs)
    if out.attrs.get('sampling_rate') and new_times.size > 1:
        out.attrs['sampling_rate'] = float(1.0 / np.mean(np.diff(new_times)))

    new = stream._rebuild(out, operation='time_average',
                          propagate_sidecars=False)
    new.add_history('time_average', {
        'n': n, 'window_s': window_s, 'method': method, 'skipna': skipna,
        'partial': partial, 'time_label': time_label,
        'n_blocks': len(segments), 'n_frames_in': int(N),
    })
    new._mark_processed()
    if name is not None:
        new.name = name
    return new


def ensure_4d(values: np.ndarray):
    """
    Standardise an array to four dimensions (time, channel, type, datatype).

    Parameters
    ----------
    values : np.ndarray
        Array of three or four dimensions.

    Returns
    -------
    values_4d : np.ndarray
        The four-dimensional array.
    has_dummy : bool
        True if a fourth dimension was added, so it can be collapsed later.
    """
    has_dummy = False
    if values.ndim == 3:
        values = values[..., np.newaxis]
        has_dummy = True
    elif values.ndim == 2:
        # Handle (Time, Channel) if needed, though rare in your current setup
        values = values[..., np.newaxis, np.newaxis]
        has_dummy = True
        
    return values, has_dummy


def fisher_z(correlation_matrix):
    """Convert correlation values to Fisher z, z = arctanh(r)."""
    return np.arctanh(np.clip(correlation_matrix, -0.999, 0.999))


def inverse_fisher_z(z_matrix):
    """Convert Fisher z values back to correlations, r = tanh(z)."""
    return np.tanh(z_matrix)
