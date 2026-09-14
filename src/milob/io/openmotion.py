import json
import os
import sqlite3

import numpy as np
import pandas as pd
import xarray as xr

from ..core.probe import Probe

#: Cameras per sensor module and the left/right module offset used purely
#: to keep the two modules' detector positions from overlapping in the
#: Probe's coordinate frame. The device gives no shared physical frame for
#: "left module" vs "right module" (they're independent probe placements),
#: so this offset is bookkeeping, not a measurement -- it only has to be
#: bigger than any SDS so the two modules' points don't collide.
_CAMS_PER_MODULE = 8
_SIDE_ORIGIN_MM = {0: np.array([0.0, 0.0, 0.0]), 1: np.array([0.0, 100.0, 0.0])}
_SIDE_NAME = {0: 'left', 1: 'right'}

#: Default source-detector separations (mm) for cameras 1-8, confirmed by
#: hardware spec. The 4x2 grid (docs/CameraArrangement.md) pairs cameras by
#: row -- (1,8), (2,7), (3,6), (4,5) -- and each row sits at one distance:
#: 10mm, 17.5mm, 27.5mm, 35mm respectively.
DEFAULT_SDS_MM = (10.0, 17.5, 27.5, 35.0, 35.0, 27.5, 17.5, 10.0)


def _load_session_row(conn, session_id, session_label):
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if session_id is not None:
        cur.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"No session with id={session_id} in this database.")
        return dict(row)

    if session_label is not None:
        cur.execute(
            "SELECT * FROM sessions WHERE session_label = ? "
            "ORDER BY session_start DESC, id DESC LIMIT 1",
            (session_label,),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"No session labelled '{session_label}' in this database.")
        return dict(row)

    cur.execute("SELECT * FROM sessions ORDER BY session_start, id")
    rows = cur.fetchall()
    if len(rows) != 1:
        labels = [(r['id'], r['session_label']) for r in rows]
        raise ValueError(
            f"Database holds {len(rows)} sessions; pass session_id or "
            f"session_label to select one. Available (id, label): {labels}"
        )
    return dict(rows[0])


def _channel_label(side, cam_id):
    return f"{_SIDE_NAME[side]}_avg" if cam_id == -1 else f"S{side + 1}D{cam_id + 1}"


def _build_probe(df, sds_mm, sc_threshold, wavelength):
    sds_mm = np.asarray(sds_mm, dtype=float)
    if sds_mm.shape != (_CAMS_PER_MODULE,):
        raise ValueError(
            f"sds_mm must give one source-detector separation (mm) per "
            f"camera (8 values, cameras 1-8); got shape {sds_mm.shape}."
        )

    combos = (df.loc[df['cam_id'] >= 0, ['side', 'cam_id']]
              .drop_duplicates()
              .sort_values(['side', 'cam_id']))

    s_pos, d_pos, source_list, detector_list = [], [], [], []
    source_id_by_side = {}
    for side, cam_id in combos.itertuples(index=False):
        if side not in source_id_by_side:
            s_pos.append(_SIDE_ORIGIN_MM[side])
            source_id_by_side[side] = len(s_pos)  # 1-based
        d_pos.append(_SIDE_ORIGIN_MM[side] + [sds_mm[cam_id], 0.0, 0.0])
        source_list.append(source_id_by_side[side])
        detector_list.append(len(d_pos))  # 1-based

    channels_config = {
        'sources': source_list,
        'detectors': detector_list,
        'wavelengths': [wavelength] * len(source_list),
    }
    probe = Probe(np.array(s_pos), np.array(d_pos), wavelengths=[wavelength],
                  lengthUnit='mm', channels=channels_config,
                  sc_threshold=sc_threshold)

    # probe.channel_labels is the authority on channel naming (it numbers
    # detectors globally across both modules, e.g. right side = D9-D16, not
    # D1-D8 -- Probe.get_distance has no notion of per-source detector
    # numbering). label_by_combo lets the caller build a matching 'channel'
    # column on session_data rows without duplicating that numbering here.
    distance_mm = [sds_mm[cam_id] for _, cam_id in combos.itertuples(index=False)]
    side_coord = [int(side) for side, _ in combos.itertuples(index=False)]
    camera_coord = [int(cam_id) + 1 for _, cam_id in combos.itertuples(index=False)]
    channel_labels = probe.channel_labels
    label_by_combo = {
        (int(side), int(cam_id)): label
        for (side, cam_id), label in zip(combos.itertuples(index=False), channel_labels)
    }

    return probe, channel_labels, side_coord, camera_coord, distance_mm, label_by_combo


def _pivot_bfi(df, channel_labels, side_coord, camera_coord, distance_mm, label_by_combo,
              wavelength, reduced_mode):
    """
    Pivot long per-frame rows into the stream's data array.

    Parameters
    ----------
    df : pandas.DataFrame
        Long frame with side, camera, frame, timestamp, flow and quality
        columns.
    channel_labels : list of str
        Channel labels in order.
    side_coord, camera_coord : array-like
        Side and camera identifiers per channel.
    distance_mm : array-like
        Source-detector distance per channel.
    label_by_combo : dict
        Maps a side and camera pair to its channel index.
    wavelength : float
        Wavelength in nm.
    reduced_mode : bool
        Whether the recording holds only per-side averages.

    Returns
    -------
    xr.DataArray
        Shape (time, channel, wavelength, op) with op = ['bfi'].
    """
    df = df.assign(channel=[label_by_combo[(s, c)] for s, c in zip(df['side'], df['cam_id'])])

    bfi = df.pivot(index='frame_id', columns='channel', values='bfi').reindex(columns=channel_labels).sort_index()
    quality = df.pivot(index='frame_id', columns='channel', values='quality').reindex(columns=channel_labels).sort_index()
    timestamps = df.groupby('frame_id')['timestamp_s'].mean().reindex(bfi.index)

    n_time = bfi.shape[0]
    sampling_rate = (round(1.0 / np.median(np.diff(timestamps.values)), 1)
                     if n_time > 1 else None)

    coords = {
        'time': timestamps.values,
        'channel': channel_labels,
        'wavelength': [wavelength],
        'op': ['bfi'],
        'side': ('channel', side_coord),
        'camera': ('channel', camera_coord),
    }
    if distance_mm is not None:
        coords['distance'] = ('channel', distance_mm)

    data_xr = xr.DataArray(
        bfi.to_numpy()[:, :, np.newaxis, np.newaxis],
        coords=coords,
        dims=['time', 'channel', 'wavelength', 'op'],
        attrs={
            'modality': 'SCOS',
            'units': 'a.u.',
            'lengthUnit': 'mm',
            'timeUnit': 's',
            'sampling_rate': sampling_rate,
            'reduced_mode': reduced_mode,
        },
    )
    return data_xr.assign_coords(quality=(('time', 'channel'), quality.to_numpy()))


def _melt_corrected_csv(df_wide):
    """
    Reshape a wide corrected-CSV export into long per-frame rows.

    Two layouts are recognised from the columns present: one column per camera
    and side, or, in reduced mode, one per side holding that side's spatial
    average.

    Parameters
    ----------
    df_wide : pandas.DataFrame
        The exported table.

    Returns
    -------
    df_long : pandas.DataFrame
        Side, camera, frame, timestamp, flow and quality columns.
    reduced_mode : bool
        True when the export holds only per-side averages.
    """
    columns = set(df_wide.columns)
    is_per_cam = 'bfi_l1' in columns or 'bfi_r1' in columns
    is_reduced = not is_per_cam and ('bfi_left' in columns or 'bfi_right' in columns)

    if not (is_per_cam or is_reduced):
        raise ValueError(
            "No bfi_* columns found -- not a recognised OpenMotion "
            "corrected-CSV export (expected the 'History -> Export CSV' / "
            "SDK CsvSink output, with bfi_l1..bfi_r8 or bfi_left/bfi_right "
            "columns)."
        )

    if is_reduced:
        combos = [(0, -1, 'bfi_left', 'quality_left'), (1, -1, 'bfi_right', 'quality_right')]
    else:
        combos = [
            (side_idx, cam_idx, f"bfi_{prefix}{cam_idx + 1}", f"quality_{prefix}{cam_idx + 1}")
            for side_idx, prefix in ((0, 'l'), (1, 'r'))
            for cam_idx in range(8)
        ]

    parts = []
    for side, cam_id, bfi_col, quality_col in combos:
        if bfi_col not in columns:
            continue
        parts.append(pd.DataFrame({
            'side': side,
            'cam_id': cam_id,
            'frame_id': df_wide['frame_id'],
            'timestamp_s': df_wide['timestamp_s'],
            'bfi': df_wide[bfi_col],
            'quality': df_wide[quality_col] if quality_col in columns else np.nan,
        }))

    df_long = pd.concat(parts, ignore_index=True).dropna(subset=['bfi'])
    if df_long.empty:
        raise ValueError("CSV has bfi_* columns but no numeric BFI data rows.")
    return df_long, is_reduced


def read_openmotion(db_path, sds_mm=DEFAULT_SDS_MM, *, session_id=None, session_label=None,
                    wavelength=795.0, sc_threshold=None):
    """
    Load one scan session from an OpenMotion scan database.

    Reads the session tables directly, without the device's own SDK. The rows
    read are the pipeline's interval-corrected output, one per camera per
    frame. A reduced-mode session instead holds only each side's spatial
    average, for which no probe geometry is returned.

    Parameters
    ----------
    db_path : str
        Path to the SQLite scan database.
    sds_mm : array-like of 8 floats, optional
        Source-detector separations in mm for cameras 1 to 8, shared by both
        sensor modules. Defaults to the standard hardware spacing. Unused in
        reduced mode, where no per-camera geometry exists.
    wavelength : float
        Laser wavelength in nm. Not recorded by the device, which is
        single-wavelength. Default 795.0.
    sc_threshold : float or None
        Source-detector distance in mm below which a channel is classified as
        short. Default None, since every channel here shares one detector
        design.
    session_id : int, optional
        Session to load. Required when the database holds more than one
        session and ``session_label`` is not given.
    session_label : str, optional
        Session label to load, taking the most recent match.

    Returns
    -------
    data_xr : xr.DataArray
        Shape (time, channel, wavelength, op) with op = ['bfi'].
    probe : Probe or None
        None for a reduced-mode session.
    session : dict
        The matched session row, with its metadata parsed from JSON.
    """
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        session = _load_session_row(conn, session_id, session_label)
        session['session_meta'] = json.loads(session.get('session_meta') or '{}')

        df = pd.read_sql_query(
            "SELECT side, cam_id, frame_id, timestamp_s, bfi, quality "
            "FROM session_data WHERE session_id = ? ORDER BY side, cam_id, frame_id",
            conn, params=(session['id'],),
        )

    if df.empty:
        raise ValueError(f"Session {session['id']} ('{session['session_label']}') "
                         f"has no session_data rows.")

    meta = session['session_meta']
    reduced_mode = bool(meta.get('sdk_flags', {}).get('reduced_mode', False))
    df = df[df['cam_id'] == -1] if reduced_mode else df[df['cam_id'] >= 0]

    if reduced_mode:
        probe = None
        combos = df[['side']].drop_duplicates().sort_values('side')
        side_coord = [int(side) for side, in combos.itertuples(index=False)]
        camera_coord = [-1] * len(side_coord)
        distance_mm = None
        label_by_combo = {(side, -1): _channel_label(side, -1) for side in side_coord}
        channel_labels = [label_by_combo[(side, -1)] for side in side_coord]
    else:
        if sds_mm is None:
            raise ValueError(
                "sds_mm is required for a non-reduced-mode session (per-camera "
                "geometry) -- pass the 8 camera source-detector separations, in mm."
            )
        probe, channel_labels, side_coord, camera_coord, distance_mm, label_by_combo = _build_probe(
            df, sds_mm, sc_threshold, wavelength)

    data_xr = _pivot_bfi(df, channel_labels, side_coord, camera_coord, distance_mm,
                        label_by_combo, wavelength, reduced_mode)

    return data_xr, probe, session


def read_openmotion_csv(csv_path, sds_mm=DEFAULT_SDS_MM, *, wavelength=795.0, sc_threshold=None):
    """
    Load one scan's blood flow from an OpenMotion corrected-CSV export.

    Builds the same array as :func:`read_openmotion`. Only the flow and
    quality columns are read. This is the app's corrected per-frame export,
    not the raw per-side histogram CSV, which carries no flow values.

    A CSV carries no session metadata, so the returned session dict holds only
    a label derived from the filename together with the source and mode.

    Parameters
    ----------
    csv_path : str
        Path to the corrected-CSV export.
    sds_mm : array-like of 8 floats, optional
        Source-detector separations in mm for cameras 1 to 8, shared by both
        sensor modules. Defaults to the standard hardware spacing. Unused in
        reduced mode, where no per-camera geometry exists.
    wavelength : float
        Laser wavelength in nm. Not recorded by the device, which is
        single-wavelength. Default 795.0.
    sc_threshold : float or None
        Source-detector distance in mm below which a channel is classified as
        short. Default None, since every channel here shares one detector
        design.

    Returns
    -------
    data_xr : xr.DataArray
        Shape (time, channel, wavelength, op) with op = ['bfi'].
    probe : Probe or None
        None for a reduced-mode export.
    session : dict
        Label, source and mode.
    """
    df_wide = pd.read_csv(csv_path)
    df, reduced_mode = _melt_corrected_csv(df_wide)

    if reduced_mode:
        probe = None
        combos = df[['side']].drop_duplicates().sort_values('side')
        side_coord = [int(side) for side, in combos.itertuples(index=False)]
        camera_coord = [-1] * len(side_coord)
        distance_mm = None
        label_by_combo = {(side, -1): _channel_label(side, -1) for side in side_coord}
        channel_labels = [label_by_combo[(side, -1)] for side in side_coord]
    else:
        if sds_mm is None:
            raise ValueError(
                "sds_mm is required for a per-camera CSV export (per-camera "
                "geometry) -- pass the 8 camera source-detector separations, in mm."
            )
        probe, channel_labels, side_coord, camera_coord, distance_mm, label_by_combo = _build_probe(
            df, sds_mm, sc_threshold, wavelength)

    data_xr = _pivot_bfi(df, channel_labels, side_coord, camera_coord, distance_mm,
                        label_by_combo, wavelength, reduced_mode)

    session = {
        'id': None,
        'session_label': os.path.splitext(os.path.basename(csv_path))[0],
        'session_meta': {'reduced_mode': reduced_mode, 'source': 'corrected_csv'},
    }

    return data_xr, probe, session
