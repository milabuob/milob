import os
import re
import warnings
from datetime import datetime

import numpy as np
from numpy.typing import ArrayLike
import xarray as xr
from scipy.io import loadmat

from ..core.probe import Probe
from ..core.events import Events
from ..core.fd_nirs import FD_Stream

# ISS Imagent "BOXY" ascii record files declare more MUX channels than are
# physically wired to a source on this probe (16 here, only 8 wired -- the
# rest carry a sentinel 0 distance and a 1e30 calibration Factor). The probe
# .layout file's 'channel_map' array (NOT part of the SD struct itself, a
# sibling top-level var) gives the physical BOXY channel number for each row
# of SD.MeasList, in order. That correspondence was confirmed for this
# dataset by checking that channel_map's 8 entries are exactly the file's
# 8 nonzero-distance channels, and that SD-geometry distances line up with
# the file's own '#DISTANCE SETTINGS' values for those channels to within
# ~0.1 cm (see `_DIST_CHECK_TOL_CM` -- the residual is consistent with the
# probe's two wavelengths riding on separate, slightly offset fiber tips at
# each nominal source location, not a mismatched pairing).
_DIST_CHECK_TOL_CM = 0.15

_CHANNEL_COL_RE = re.compile(r'^([A-Za-z])-(AC|DC|Ph)(\d+)$')


def _read_lines(filepath):
    with open(filepath, 'r') as f:
        return [line.rstrip('\n') for line in f]


def _section_start(lines, header, from_idx=0):
    for i in range(from_idx, len(lines)):
        if lines[i].strip() == header:
            return i
    raise ValueError(f"Section {header!r} not found in file")


def _next_nonblank(lines, from_idx):
    for i in range(from_idx, len(lines)):
        if lines[i].strip():
            return i
    raise ValueError("Ran out of lines looking for the next non-blank line")


def _parse_wf_calibration(lines):
    """
    Parse the calibration block into per-column term and factor values.

    Returns
    -------
    dict of {str: tuple of (float, float)}
        Column name to its additive term and multiplicative factor. The factor
        is NaN for phase columns, which carry a term only.
    """
    start = _section_start(lines, '#WF CALIBRATION VALUES')
    header_idx = None
    for i in range(start, len(lines)):
        if lines[i].startswith('Val Type') and '-AC1' in lines[i]:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find the WF calibration 'Val Type' header row")

    col_names = lines[header_idx].split('\t')[1:]
    col_names = [c for c in col_names if c]

    term_idx = _next_nonblank(lines, header_idx + 1)
    if not lines[term_idx].startswith('Term'):
        raise ValueError(f"Expected a 'Term' row after the WF calibration header, got: {lines[term_idx]!r}")
    factor_idx = _next_nonblank(lines, term_idx + 1)
    if not lines[factor_idx].startswith('Factor'):
        raise ValueError(f"Expected a 'Factor' row after 'Term', got: {lines[factor_idx]!r}")

    terms = lines[term_idx].split('\t')[1:1 + len(col_names)]
    factors = lines[factor_idx].split('\t')[1:1 + len(col_names)]

    cal = {}
    for name, t, f in zip(col_names, terms, factors):
        term = float(t)
        factor = np.nan if f.strip() == 'X' else float(f)
        cal[name] = (term, factor)
    return cal


def _parse_distance_settings(lines):
    """
    Parse the distance-settings block.

    Returns
    -------
    dict of {int: float}
        Channel number to source-detector distance in cm.
    """
    start = _section_start(lines, '#DISTANCE SETTINGS')
    header_idx = None
    for i in range(start, len(lines)):
        if lines[i].startswith('A-1'):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find the '#DISTANCE SETTINGS' channel header row")

    labels = [c for c in lines[header_idx].split('\t') if c]
    values_idx = _next_nonblank(lines, header_idx + 1)
    values = [float(v) for v in lines[values_idx].split('\t') if v]

    dist = {}
    for label, v in zip(labels, values):
        n = int(label.split('-')[1])
        dist[n] = v
    return dist


def _parse_data_block(lines):
    """
    Parse the data block.

    Returns
    -------
    col_idx : dict of {str: int}
        Column name to its index.
    data : np.ndarray
        Shape (n_times, n_columns).
    """
    start = _section_start(lines, '#DATA BEGINS')
    header_idx = _next_nonblank(lines, start + 1)
    col_names = [c for c in lines[header_idx].split('\t') if c]
    col_idx = {name: i for i, name in enumerate(col_names)}

    data_lines = [l for l in lines[header_idx + 1:] if l.strip() and not l.startswith('#')]
    if not data_lines:
        raise ValueError("No data rows found after '#DATA BEGINS'")
    data = np.array([l.split() for l in data_lines], dtype=float)
    if data.shape[1] != len(col_names):
        raise ValueError(
            f"Data row has {data.shape[1]} fields but the header lists "
            f"{len(col_names)} columns."
        )
    return col_idx, data


def read_ISS_BOXY(datafile: str, layout_file: str, *, modulation_frequency: float,
                   sc_threshold: float = None, name: str = None,
                   apply_calibration: bool = False) -> FD_Stream:
    """
    Load an ISS Imagent BOXY record into a frequency-domain stream.

    The record carries no wavelength or modulation-frequency information and
    declares more multiplexer channels than are physically connected. The
    paired layout file resolves which channel corresponds to each source,
    detector and wavelength.

    The instrument's manual marker column becomes rows in the stream's event
    table, with zero duration and the label 'button_press'. An older export
    without that column yields an empty table.

    Parameters
    ----------
    datafile : str
        Path to the BOXY ascii record.
    layout_file : str
        Path to the .layout file, holding the SD struct and a channel map
        giving the physical channel number for each measurement-list row.
    modulation_frequency : float
        Instrument RF modulation frequency in Hz. Not recorded in the file, so
        it must be supplied; the header's waveform frequency is the
        cross-correlation frequency, not this.
    sc_threshold : float or None
        Source-detector distance in mm below which a channel is classified as
        short. Pass None if the probe design has no short-separation
        channels.
    name : str, optional
        Stream name. Defaults to 'fd-nirs'.
    apply_calibration : bool
        Apply the file's own calibration terms and factors to the raw columns.
        Default False, since the stored values normally already include them
        and re-applying double-calibrates. Phase is converted from degrees to
        radians either way.

    Returns
    -------
    FD_Stream
        Shape (time, channel, wavelength, freq), complex, with the DC and
        modulation frequencies on the last axis.
    """
    layout = loadmat(layout_file, struct_as_record=False, squeeze_me=True)
    SD = layout['SD']
    if 'channel_map' not in layout:
        raise ValueError(
            f"{layout_file} has no top-level 'channel_map' array -- it "
            f"doesn't look like an ISS BOXY probe layout (see module "
            f"docstring)."
        )
    channel_map = np.atleast_1d(layout['channel_map']).astype(int)
    ml = np.atleast_2d(SD.MeasList)
    if channel_map.shape[0] != ml.shape[0]:
        raise ValueError(
            f"channel_map has {channel_map.shape[0]} entries but "
            f"SD.MeasList has {ml.shape[0]} rows -- they should be 1:1."
        )

    src_pos_all = np.atleast_2d(SD.SrcPos)
    det_pos_all = np.atleast_2d(SD.DetPos)
    lam = np.atleast_1d(SD.Lambda)

    src_ids = ml[:, 0].astype(int)
    det_ids = ml[:, 1].astype(int)
    wl_idx = ml[:, 3].astype(int)
    computed_dist = np.array([
        np.linalg.norm(src_pos_all[s - 1] - det_pos_all[d - 1])
        for s, d in zip(src_ids, det_ids)
    ])

    lines = _read_lines(datafile)
    cal = _parse_wf_calibration(lines)
    file_dist = _parse_distance_settings(lines)
    col_idx, data = _parse_data_block(lines)

    detector_letters = {m.group(1) for name in col_idx if (m := _CHANNEL_COL_RE.match(name))}
    if len(detector_letters) != 1:
        raise NotImplementedError(
            f"Expected exactly one detector letter in the data columns, "
            f"found {sorted(detector_letters)}. Multi-detector BOXY files "
            f"aren't handled yet."
        )
    det_letter = detector_letters.pop()

    # Sanity-check the layout/file pairing: geometry computed from SD should
    # roughly match the file's own recorded distance for the mapped channel.
    for ch, d_sd, d_file_key in zip(channel_map, computed_dist, channel_map):
        d_file = file_dist.get(int(d_file_key))
        if d_file is None or abs(d_file) < 1e-9:
            raise ValueError(
                f"{layout_file}'s channel_map points at BOXY channel "
                f"{ch}, but {datafile} records a 0/missing distance for "
                f"it -- these files don't look like a matching pair."
            )
        if abs(d_file - d_sd) > _DIST_CHECK_TOL_CM:
            warnings.warn(
                f"Channel {ch}: layout geometry gives {d_sd:.3f} cm but "
                f"the BOXY file's own Distance Settings say {d_file:.3f} "
                f"cm (tolerance {_DIST_CHECK_TOL_CM} cm) -- double-check "
                f"this layout file is paired with this data file."
            )

    n_time = data.shape[0]
    n_ch = channel_map.shape[0]
    ac_raw = np.empty((n_time, n_ch))
    dc_raw = np.empty((n_time, n_ch))
    ph_raw = np.empty((n_time, n_ch))
    for i, ch in enumerate(channel_map):
        ac_name, dc_name, ph_name = f'{det_letter}-AC{ch}', f'{det_letter}-DC{ch}', f'{det_letter}-Ph{ch}'
        for col_name in (ac_name, dc_name, ph_name):
            if col_name not in col_idx:
                raise ValueError(f"{datafile} has no data column {col_name!r} (mapped from layout channel {ch})")
        ac_raw[:, i] = data[:, col_idx[ac_name]]
        dc_raw[:, i] = data[:, col_idx[dc_name]]
        ph_raw[:, i] = data[:, col_idx[ph_name]]

        if apply_calibration:
            _, ac_factor = cal[ac_name]
            _, dc_factor = cal[dc_name]
            ph_term, _ = cal[ph_name]
            ac_raw[:, i] = ac_raw[:, i] * ac_factor
            dc_raw[:, i] = dc_raw[:, i] * dc_factor
            ph_raw[:, i] = ph_raw[:, i] + ph_term

    # --- Split the 8 mapped channels by wavelength (mirrors io.oxiplex) ---
    wavelengths = np.unique(lam[wl_idx - 1])
    unique_sources = np.unique(src_ids)
    n_wl = wavelengths.shape[0]
    n_channels = unique_sources.shape[0]

    ac_wl = np.empty((n_time, n_channels, n_wl))
    dc_wl = np.empty((n_time, n_channels, n_wl))
    ph_wl = np.empty((n_time, n_channels, n_wl))
    for wi, w in enumerate(wavelengths):
        mask = lam[wl_idx - 1] == w
        order = np.argsort(src_ids[mask])
        idx = np.flatnonzero(mask)[order]
        ac_wl[:, :, wi] = ac_raw[:, idx]
        dc_wl[:, :, wi] = dc_raw[:, idx]
        ph_wl[:, :, wi] = ph_raw[:, idx]

    phase_rad = np.unwrap((np.deg2rad(ph_wl) + np.pi) % (2 * np.pi) - np.pi, axis=0)
    complex_ac = ac_wl * np.exp(1j * phase_rad)
    dc_complex = dc_wl.astype(complex)
    data_complex = np.stack([dc_complex, complex_ac], axis=-1)  # (time, channel, wavelength, freq=2)
    freq_values = np.array([0.0, modulation_frequency])

    channel_dist = np.array([
        np.linalg.norm(src_pos_all[s - 1] - det_pos_all[det_ids[src_ids == s][0] - 1])
        for s in unique_sources
    ])
    detector_list = [int(det_ids[src_ids == s][0]) for s in unique_sources]
    channel_labels = [f"S{s}D{d}" for s, d in zip(unique_sources, detector_list)]

    channels_config = {
        'sources': list(unique_sources),
        'detectors': detector_list,
        'wavelengths': list(wavelengths),
    }
    probe = Probe(src_pos_all, det_pos_all, wavelengths,
                  lengthUnit=str(SD.SpatialUnit), channels=channels_config,
                  sc_threshold=sc_threshold)

    time_col = data[:, col_idx['time']]
    times = time_col - time_col[0]

    # One event per row where the instrument's 'mark' column is nonzero
    # (see the Notes above).
    events = Events()
    if 'mark' in col_idx:
        for t, m in zip(times, data[:, col_idx['mark']]):
            if m != 0:
                events.add_event(onset=t, duration=0.0, value=m, label='button_press')

    coords = {
        'time': times,
        'channel': channel_labels,
        'wavelength': wavelengths,
        'freq': freq_values,

        'channel_idx': ('channel', np.arange(n_channels)),
        'source': ('channel', list(unique_sources)),
        'detector': ('channel', detector_list),
        'distance': ('channel', channel_dist),
    }

    data_xr = xr.DataArray(
        data_complex,
        coords=coords,
        dims=['time', 'channel', 'wavelength', 'freq'],
        attrs={
            'modality': 'fd-nirs',
            'lengthUnit': str(SD.SpatialUnit),
            'timeUnit': 's',
            'frequencyUnit': 'Hz',
            'modulation_frequency_Hz': float(modulation_frequency),
            'sampling_rate': round(1.0 / np.diff(times).mean(), 2) if n_time > 1 else None,
            'freq_encoding': 'freq=0 -> DC (real); freq=f_mod -> AC*exp(i*phase_rad)',
            'calibration_applied': apply_calibration,
        }
    )

    stream_name = name or 'fd-nirs'
    return FD_Stream(data_xr, probe, name=stream_name, events=events)

