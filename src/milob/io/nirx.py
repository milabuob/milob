# io/nirx.py
"""
Reader for NIRx datasets.

Intensity is stored in one text file per wavelength alongside a header
describing the acquisition. Probe geometry is not embedded and must be
supplied separately as a .layout, .SD or digpts.txt file.

Header values are either plain or quoted scalars, or multi-line blocks
delimited by ``"#`` and ``#"``.
"""

import re
import warnings
import numpy as np
import scipy.io
import xarray as xr
from pathlib import Path


# ---------------------------------------------------------------------------
# Text decoding
# ---------------------------------------------------------------------------

def _read_text(path, encoding=None):
    """
    Read a NIRx text file as a string.

    Parameters
    ----------
    path : str or Path
        File to read.
    encoding : str, optional
        Text encoding. Decoded as UTF-8 and retried as latin-1 with a warning
        when omitted, since the acquisition software writes the recording
        machine's codepage. An explicit codec decodes strictly.

    Returns
    -------
    str
        File contents.
    """
    if encoding is not None:
        with open(path, 'r', encoding=encoding) as f:
            return f.read()

    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        warnings.warn(
            f"{Path(path).name} is not valid UTF-8; falling back to latin-1. "
            f"Pass encoding='...' to decode it explicitly.",
            stacklevel=2,
        )
        with open(path, 'r', encoding='latin-1') as f:
            return f.read()


# ---------------------------------------------------------------------------
# HDR low-level helpers
# ---------------------------------------------------------------------------

def _strip_scalar_value(line):
    """Return the value of a Key=Value line, stripped of surrounding quotes."""
    val = line.split('=', 1)[1].strip()
    return val.strip('"\'')


def _get_inline_block(lines, key_pattern):
    """
    Extract the data lines of a multi-line header block.

    Parameters
    ----------
    lines : list of str
        All lines of the header file.
    key_pattern : str
        Regular expression matching the key line.

    Returns
    -------
    list of str
        Content lines between the block delimiters, empty when the key is
        absent or the block holds no data.
    """
    data_lines = []
    in_block = False

    for line in lines:
        s = line.strip()

        if not in_block:
            if re.match(key_pattern, s):
                value_part = s.split('=', 1)[1].strip() if '=' in s else ''
                # Block opens when the value starts with "#  (with optional whitespace)
                if re.match(r'^"?#', value_part):
                    in_block = True
        else:
            # Block closes on #" or standalone #
            if s in ('#"', '#'):
                break
            if s:
                data_lines.append(s)

    return data_lines


# ---------------------------------------------------------------------------
# HDR file parsing
# ---------------------------------------------------------------------------

def _parse_nirx_hdr(hdr_path, encoding=None):
    """
    Parse a NIRx header file.

    Parameters
    ----------
    hdr_path : str or Path
        Path to the .hdr file.
    encoding : str, optional
        Text encoding. Decoded as UTF-8 and retried as latin-1 with a warning
        when omitted, since the acquisition software writes the recording
        machine's codepage. An explicit codec decodes strictly.

    Returns
    -------
    dict
        Keys ``n_sources``, ``n_detectors``, ``wavelengths``, ``fs``,
        ``markers`` (onset in seconds, trigger code and one-based sample, or
        None) and ``mask``, non-zero where a source-detector pair is active.
    """
    lines = _read_text(hdr_path, encoding).splitlines()

    n_sources = n_detectors = wavelengths = fs = None

    for line in lines:
        s = line.strip()
        if n_sources is None and re.match(r'Sources\s*=', s):
            n_sources = int(_strip_scalar_value(s))
        if n_detectors is None and re.match(r'Detectors\s*=', s):
            n_detectors = int(_strip_scalar_value(s))
        if wavelengths is None and re.match(r'Wavelengths\s*=', s):
            val = _strip_scalar_value(s).strip('[]')
            wavelengths = np.array([float(w) for w in re.split(r'[,\s\t]+', val) if w])
        if fs is None and re.match(r'SamplingRate\s*=', s):
            fs = round(float(_strip_scalar_value(s)), 1)

    missing = [k for k, v in {'Sources': n_sources, 'Detectors': n_detectors,
                               'Wavelengths': wavelengths, 'SamplingRate': fs}.items()
               if v is None]
    if missing:
        raise ValueError(f"Could not parse required fields from HDR: {missing}")

    markers = _parse_hdr_events(lines)
    mask = _parse_hdr_mask(lines, n_sources, n_detectors)

    return {
        'n_sources':   n_sources,
        'n_detectors': n_detectors,
        'wavelengths': wavelengths,
        'fs':          fs,
        'markers':     markers,
        'mask':        mask,
    }


def _parse_hdr_events(lines):
    """
    Extract event markers from the header's events block.

    Parameters
    ----------
    lines : list of str
        All lines of the header file.

    Returns
    -------
    np.ndarray or None
        Shape (n_events, 3): onset in seconds, trigger code and one-based
        sample.
    """
    data = _get_inline_block(lines, r'^Events\s*=')
    if not data:
        return None

    markers = []
    for row in data:
        parts = row.split()
        if len(parts) >= 3:
            try:
                markers.append([float(parts[0]), int(parts[1]), int(parts[2])])
            except ValueError:
                pass

    return np.array(markers) if markers else None


def _parse_hdr_mask(lines, n_sources, n_detectors):
    """
    Extract the source-detector mask from the header.

    Parameters
    ----------
    lines : list of str
        All lines of the header file.
    n_sources, n_detectors : int
        Expected mask dimensions.

    Returns
    -------
    np.ndarray
        Shape (n_sources, n_detectors), non-zero where a pair is active.
    """
    data = _get_inline_block(lines, r'^S-D-Mask\s*=')
    if not data:
        raise ValueError(
            "Could not locate 'S-D-Mask' in the HDR file. "
            "Ensure the file contains a [DataStructure] section with "
            "S-D-Mask=\"#...#\" format."
        )

    mask = []
    for row in data:
        parts = row.split()
        if len(parts) == n_detectors:
            try:
                mask.append([int(x) for x in parts])
            except ValueError:
                pass

    if len(mask) != n_sources:
        raise ValueError(
            f"S-D-Mask has {len(mask)} rows but HDR says Sources={n_sources}."
        )

    return np.array(mask, dtype=int)


# ---------------------------------------------------------------------------
# Wavelength file parsing
# ---------------------------------------------------------------------------

def _parse_nirx_wl(hdr_path, mask, n_sources, n_detectors, wavelengths):
    """
    Read the per-wavelength intensity files.

    Each file holds one row per time point, with columns spanning every
    possible source-detector pair in row-major order. Only pairs active in the
    mask are retained.

    Parameters
    ----------
    hdr_path : str or Path
        Path to the header; the wavelength files share its stem and directory.
    mask : np.ndarray
        Source-detector mask.
    n_sources, n_detectors : int
        Mask dimensions.
    wavelengths : np.ndarray
        Wavelengths in nm.

    Returns
    -------
    raw_data : np.ndarray
        Shape (n_times, n_valid_pairs * n_wavelengths), grouped by wavelength.
    n_timepoints : int
        Number of time points.
    """
    stem = str(hdr_path)
    if stem.lower().endswith('.hdr'):
        stem = stem[:-4]

    valid = mask.flatten() != 0  # row-major: src outer, det inner

    all_wl = []
    for wl_idx in range(len(wavelengths)):
        wl_path = f"{stem}.wl{wl_idx + 1}"
        raw = np.loadtxt(wl_path)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        all_wl.append(raw[:, valid])

    n_timepoints = all_wl[0].shape[0]
    data = np.concatenate(all_wl, axis=1)
    return data, n_timepoints


# ---------------------------------------------------------------------------
# Measurement list construction
# ---------------------------------------------------------------------------

def _build_measlist(mask, n_wavelengths):
    """
    Build a measurement list from the source-detector mask.

    Optode indices are one-based and wavelength indices zero-based. Column
    order matches the intensity files: every valid pair for the first
    wavelength, then for the next.

    Parameters
    ----------
    mask : np.ndarray
        Source-detector mask.
    n_wavelengths : int
        Number of wavelengths.

    Returns
    -------
    dict
        Sources, detectors, wavelength indices and data type per channel.
    """
    sources, detectors = [], []
    for src_i in range(mask.shape[0]):
        for det_j in range(mask.shape[1]):
            if mask[src_i, det_j] != 0:
                sources.append(src_i + 1)
                detectors.append(det_j + 1)

    n_pairs = len(sources)
    all_src, all_det, all_wl, all_dt = [], [], [], []
    for wl_i in range(n_wavelengths):
        all_src.extend(sources)
        all_det.extend(detectors)
        all_wl.extend([wl_i] * n_pairs)
        all_dt.extend([1] * n_pairs)

    return {
        'source':        np.array(all_src, dtype=int),
        'detector':      np.array(all_det, dtype=int),
        'wavelength_idx': np.array(all_wl, dtype=int),
        'dataType':      np.array(all_dt, dtype=int),
        'dataTypeIndex': np.ones(len(all_src), dtype=int),
        'dataUnit':      ['unknown'] * len(all_src),
    }


# ---------------------------------------------------------------------------
# Stimulus matrix construction
# ---------------------------------------------------------------------------

def _build_stim_matrix(markers, n_timepoints):
    """
    Build a binary stimulus matrix from event markers.

    The trigger code is used directly as a one-based column index.

    Parameters
    ----------
    markers : np.ndarray or None
        Shape (n_events, 3): onset in seconds, trigger code and one-based
        sample.
    n_timepoints : int
        Length of the recording in samples.

    Returns
    -------
    np.ndarray or None
        Shape (n_timepoints, max_trigger_code).
    """
    if markers is None or len(markers) == 0:
        return None

    n_conditions = int(markers[:, 1].max())
    s = np.zeros((n_timepoints, n_conditions), dtype=float)

    for row in markers:
        cond     = int(row[1]) - 1   # 0-based column
        t_sample = int(row[2]) - 1   # 0-based time sample
        if 0 <= t_sample < n_timepoints:
            s[t_sample, cond] = 1.0

    return s


# ---------------------------------------------------------------------------
# Probe file parsing
# ---------------------------------------------------------------------------

def _parse_digpts_txt(filepath, encoding=None):
    """
    Parse an AtlasViewer digpts.txt file.

    Each line holds a label followed by three coordinates. Sources are
    labelled sN and detectors dN; anything else is treated as a landmark.

    Parameters
    ----------
    filepath : str or Path
        Path to the file.
    encoding : str, optional
        Text encoding. Decoded as UTF-8 and retried as latin-1 with a warning
        when omitted, since the acquisition software writes the recording
        machine's codepage. An explicit codec decodes strictly.

    Returns
    -------
    s_pos_3d, d_pos_3d : np.ndarray
        Optode positions, shape (n, 3).
    landmarks : np.ndarray or None
        Landmark positions.
    l_labels : list of str or None
        Landmark labels.
    """
    s_dict, d_dict = {}, {}
    landmarks, l_labels = [], []

    pattern = re.compile(
        r'([a-zA-Z][a-zA-Z0-9]*)\s*:\s*(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)'
    )

    for line in _read_text(filepath, encoding).splitlines():
        m = pattern.search(line)
        if not m:
            continue
        label = m.group(1).lower()
        coords = np.array([float(m.group(2)), float(m.group(3)), float(m.group(4))])

        if label.startswith('s') and label[1:].isdigit():
            s_dict[int(label[1:])] = coords
        elif label.startswith('d') and label[1:].isdigit():
            d_dict[int(label[1:])] = coords
        else:
            landmarks.append(coords)
            l_labels.append(label.upper())

    if not s_dict:
        raise ValueError(f"No source positions (sN: x y z) found in {filepath}")
    if not d_dict:
        raise ValueError(f"No detector positions (dN: x y z) found in {filepath}")

    s_pos = np.array([s_dict[i] for i in sorted(s_dict)])
    d_pos = np.array([d_dict[i] for i in sorted(d_dict)])
    lm = np.array(landmarks) if landmarks else None
    ll = l_labels if l_labels else None

    return s_pos, d_pos, lm, ll


def _parse_probe_mat(filepath, n_sources, n_detectors):
    """
    Load probe positions from a MATLAB binary file.

    Recognises, in order, a NIRx nirsInfo export, a NIRx probeInfo export, a
    digitiser struct with two- and three-dimensional maps, a two-dimensional
    map alone, and a Homer2 SD struct.

    Parameters
    ----------
    filepath : str or Path
        Path to the .layout, .mat or .SD file.
    n_sources, n_detectors : int
        Expected optode counts.

    Returns
    -------
    s_pos_3d, d_pos_3d, s_pos_2d, d_pos_2d : np.ndarray or None
        Optode positions.
    landmarks, l_labels : None
        Never present in these formats.
    """
    try:
        mat = scipy.io.loadmat(str(filepath), squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        raise NotImplementedError(
            f"The file {filepath} appears to be a MATLAB v7.3 (HDF5) file, "
            "which is not yet supported for probe loading.  Resave it as v6 or v7 "
            "from MATLAB, or convert to digpts.txt format."
        )

    def _2d_to_3col(arr):
        arr = np.atleast_2d(arr)
        if arr.shape[1] == 2:
            return np.column_stack([arr, np.zeros(arr.shape[0])])
        return arr

    s2d = d2d = s3d = d3d = None

    if 'nirsInfo' in mat:
        probes = mat['nirsInfo'].probeInfo.probes
        s2d = _2d_to_3col(np.atleast_2d(probes.coords_s2))
        d2d = _2d_to_3col(np.atleast_2d(probes.coords_d2))
        s3d = np.atleast_2d(probes.coords_s3)
        d3d = np.atleast_2d(probes.coords_d3)

    elif 'probeInfo' in mat:
        probes = mat['probeInfo'].probes
        s2d = _2d_to_3col(np.atleast_2d(probes.coords_s2))
        d2d = _2d_to_3col(np.atleast_2d(probes.coords_d2))
        s3d = np.atleast_2d(probes.coords_s3)
        d3d = np.atleast_2d(probes.coords_d3)

    elif 'RS_MRI' in mat:
        rs = mat['RS_MRI']
        map2d = np.atleast_2d(rs.Map)
        map3d = np.atleast_2d(rs.Map3d)
        s2d = _2d_to_3col(map2d[:n_sources])
        d2d = _2d_to_3col(map2d[n_sources:n_sources + n_detectors])
        s3d = map3d[:n_sources]
        d3d = map3d[n_sources:n_sources + n_detectors]

    elif 'Map' in mat:
        map2d = np.atleast_2d(mat['Map'])
        s2d = _2d_to_3col(map2d[:n_sources])
        d2d = _2d_to_3col(map2d[n_sources:n_sources + n_detectors])
        s3d = s2d.copy()
        d3d = d2d.copy()

    elif 'SD' in mat:
        sd = mat['SD']
        s3d = _2d_to_3col(np.atleast_2d(sd.SrcPos))
        d3d = _2d_to_3col(np.atleast_2d(sd.DetPos))
        s2d = s3d.copy()
        d2d = d3d.copy()

    else:
        raise ValueError(
            f"Unrecognised probe file format in {filepath}. "
            "Expected one of: nirsInfo, probeInfo, RS_MRI, Map, or SD variables."
        )

    return s3d, d3d, s2d, d2d, None, None


def _parse_nirx_probe(probe_path, n_sources, n_detectors, encoding=None):
    """
    Dispatch to the probe parser matching the file extension.

    Parameters
    ----------
    probe_path : str or Path
        Probe geometry file.
    n_sources, n_detectors : int
        Expected optode counts.
    encoding : str, optional
        Text encoding. Decoded as UTF-8 and retried as latin-1 with a warning
        when omitted, since the acquisition software writes the recording
        machine's codepage. An explicit codec decodes strictly.

    Returns
    -------
    s_pos_3d, d_pos_3d, s_pos_2d, d_pos_2d : np.ndarray or None
        Optode positions.
    landmarks : np.ndarray or None
        Landmark positions.
    l_labels : list of str or None
        Landmark labels.
    """
    p = Path(probe_path)
    ext = p.suffix.lower()

    if ext == '.txt':
        s3d, d3d, lm, ll = _parse_digpts_txt(p, encoding=encoding)
        return s3d, d3d, None, None, lm, ll

    if ext in ('.layout', '.mat', '.sd'):
        return _parse_probe_mat(p, n_sources, n_detectors)

    raise ValueError(
        f"Unsupported probe file extension '{ext}'. "
        "Expected .txt (digpts), .layout, .mat, or .SD."
    )


# ---------------------------------------------------------------------------
# Short-channel detector remapping
# ---------------------------------------------------------------------------

def _remap_short_channel_detectors(ml, mask, n_detectors, n_pos_d):
    """
    Give bundled short-channel detectors their own indices.

    When the probe file holds more detector positions than the header
    declares, the surplus are treated as physically distinct short-channel
    detectors that the header bundles into one slot. The overloaded slot is
    identified by its source count, and each source connecting to it is
    remapped to its own detector index.

    Parameters
    ----------
    ml : dict
        Measurement list to remap.
    mask : np.ndarray
        Source-detector mask.
    n_detectors : int
        Detector count declared in the header.
    n_pos_d : int
        Detector positions available in the probe file.

    Returns
    -------
    dict
        The remapped measurement list, or the original when the overloaded
        slot cannot be identified unambiguously.
    """
    n_extra = n_pos_d - n_detectors

    # Count active sources per detector slot (1-based)
    det_src_count = {}
    for src_i in range(mask.shape[0]):
        for det_j in range(mask.shape[1]):
            if mask[src_i, det_j] != 0:
                d = det_j + 1
                det_src_count[d] = det_src_count.get(d, 0) + 1

    # The overloaded slot has exactly (1 + n_extra) source connections
    overloaded = [d for d, cnt in det_src_count.items() if cnt == 1 + n_extra]
    if len(overloaded) != 1:
        return ml  # ambiguous — return unchanged

    slot = overloaded[0]

    # Find which sources connect to the overloaded slot, sorted ascending
    short_sources = sorted(
        src_i + 1  # 1-based
        for src_i in range(mask.shape[0])
        if mask[src_i, slot - 1] != 0
    )
    # Map (src, old_det) → new_det
    remap = {(src, slot): slot + k for k, src in enumerate(short_sources)}

    new_det = np.array([
        remap.get((s, d), d)
        for s, d in zip(ml['source'], ml['detector'])
    ], dtype=int)

    remapped = dict(ml)
    remapped['detector'] = new_det
    return remapped


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _parse_nirx(hdr_path, *, sc_threshold, probe_path=None, length_unit=None,
                encoding=None):
    """
    Parse a complete NIRx dataset.

    Parameters
    ----------
    hdr_path : str or Path
        Path to the .hdr file. The per-wavelength files are located
        automatically from its stem and directory.
    sc_threshold : float or None
        Source-detector distance in mm below which a channel is classified as
        short. Pass None if the probe design has no short-separation channels.
    probe_path : str or Path, optional
        Probe geometry file. Without it, optode positions are zero-filled and
        distances and topography are unavailable.
    length_unit : {'cm', 'mm'}, optional
        Unit of the probe coordinates. Inferred from the probe file extension
        when omitted.
    encoding : str, optional
        Text encoding. Decoded as UTF-8 and retried as latin-1 with a warning
        when omitted, since the acquisition software writes the recording
        machine's codepage. An explicit codec decodes strictly.

    Returns
    -------
    dict
        Keys ``cw_xr`` (the data, with dims time, channel and wavelength),
        ``probe``, ``stim_matrix`` or None, and ``times`` in seconds.
    """
    from ..core.probe import Probe
    from .snirf import _reshape_measurement_list

    hdr_path = Path(hdr_path)

    # Infer length_unit from probe file extension when not explicitly set
    if length_unit is None:
        if probe_path is not None:
            ext = Path(probe_path).suffix.lower()
            length_unit = 'mm' if ext == '.txt' else 'cm'
        else:
            length_unit = 'mm'

    # 1. Parse header
    hdr         = _parse_nirx_hdr(hdr_path, encoding=encoding)
    n_sources   = hdr['n_sources']
    n_detectors = hdr['n_detectors']
    wavelengths = hdr['wavelengths']
    fs          = hdr['fs']
    mask        = hdr['mask']
    markers     = hdr['markers']

    # 2. Load probe geometry first — needed to reconcile mask with probe coverage
    expand_short_channels = False
    if probe_path is not None:
        s3d, d3d, s2d, d2d, landmarks, l_labels = _parse_nirx_probe(
            probe_path, n_sources, n_detectors, encoding=encoding)

        n_pos_s = s3d.shape[0]
        n_pos_d = d3d.shape[0]

        if n_pos_s < n_sources or n_pos_d < n_detectors:
            # Probe has FEWER positions than HDR — drop channels with no position
            filtered = mask.copy()
            filtered[n_pos_s:, :] = 0
            filtered[:, n_pos_d:] = 0
            n_dropped = int(mask.sum()) - int(filtered.sum())
            if n_dropped:
                print(
                    f"Warning: probe file has {n_pos_s} sources and {n_pos_d} detectors, "
                    f"but the HDR references up to source {n_sources} and detector "
                    f"{n_detectors}. Dropping {n_dropped} channel(s) with no probe position."
                )
            mask = filtered

        elif n_pos_d > n_detectors:
            # Probe has MORE detectors than HDR — likely short-channel expansion.
            # One NIRx detector slot is physically split into (n_pos_d - n_detectors + 1)
            # individual short-channel detectors, each with a unique probe position.
            n_extra = n_pos_d - n_detectors
            print(
                f"Note: probe file has {n_pos_d} detector positions but HDR declares "
                f"{n_detectors} detector slots ({n_extra} extra). Attempting to remap "
                f"the overloaded detector slot to unique short-channel positions."
            )
            expand_short_channels = True  # applied after building the measurement list

        if n_pos_s > n_sources:
            print(
                f"Note: probe file has {n_pos_s} source positions but HDR declares "
                f"{n_sources} sources. Extra probe source positions will be ignored."
            )
    else:
        s3d = np.zeros((n_sources,   3))
        d3d = np.zeros((n_detectors, 3))
        s2d = d2d = landmarks = l_labels = None
        n_pos_d = n_detectors

    # 3. Read intensity data from .wl files (uses filtered mask if probe was trimmed)
    raw_data, n_timepoints = _parse_nirx_wl(
        hdr_path, mask, n_sources, n_detectors, wavelengths)

    # 4. Build measurement list from mask
    ml = _build_measlist(mask, len(wavelengths))

    # 4b. Remap short-channel detector indices if the probe has extra positions
    if expand_short_channels:
        ml_remapped = _remap_short_channel_detectors(ml, mask, n_detectors, n_pos_d)
        if np.array_equal(ml_remapped['detector'], ml['detector']):
            print(
                "Warning: could not unambiguously identify the overloaded detector slot. "
                "Short-channel pairs will share a single probe position. "
                "Check that probe detector count minus HDR detector count equals the "
                "number of short channels minus one."
            )
        ml = ml_remapped

    # 5. Reshape to (time, channel, wavelength)
    data_reshaped, metadata = _reshape_measurement_list(raw_data, ml, wavelengths)

    # 6. Time vector (zero-based, seconds)
    t = np.arange(n_timepoints) / fs

    # 7. Build xarray DataArray
    coords = {
        'time':      t,
        'channel':   metadata['channel_labels'],
        'wavelength': metadata['wavelengths'],
        'source':    ('channel', metadata['sources']),
        'detector':  ('channel', metadata['detectors']),
    }
    if 'datatype' in metadata['dims']:
        coords['datatype']      = metadata['datatypes']
        coords['datatype_name'] = ('datatype', metadata['datatype_names'])

    data_xr = xr.DataArray(
        data_reshaped,
        coords=coords,
        dims=metadata['dims'],
        attrs={
            'modality':      'cw_nirs',
            'lengthUnit':    length_unit,
            'timeUnit':      's',
            'sampling_rate': fs,
        },
    )

    channels_config = {
        'sources':    metadata['sources'],
        'detectors':  metadata['detectors'],
        'wavelengths': [wavelengths[i] for i in ml['wavelength_idx']],
        'datatypes':  ml['dataType'].tolist(),
    }

    probe = Probe(
        s3d, d3d, wavelengths,
        channels=channels_config,
        s_pos_2d=s2d, d_pos_2d=d2d,
        landmark_pos=landmarks, landmark_labels=l_labels,
        lengthUnit=length_unit, sc_threshold=sc_threshold,
    )

    # 8. Stimulus matrix
    stim_matrix = _build_stim_matrix(markers, n_timepoints)

    return {
        'cw_xr':       data_xr,
        'probe':       probe,
        'stim_matrix': stim_matrix,
        'times':       t,
    }
