# import xarray as xr
import h5py
import re
import xarray as xr
import numpy as np
from typing import NamedTuple, Optional
from ..core.datastream import Datastream
from ..core.probe import Probe
from ..core.events import Events

# Constants for SNIRF dataType ranges (per specification)
SNIRF_DATATYPES = {
    'cw': (1, 100),
    'fd': (101, 200),
    'td_gated': (201, 300),
    'td_moments': (301, 400),
    'dcs': (401, 500),
}


#: Priority order for classifying a file from the *set* of dataType
#: categories present -- highest information-content modality wins. This
#: matters because FD acquisitions legitimately mix the CW-range DC term
#: (dataType=1) alongside their own 101/102 codes (see SNIRF spec Appendix),
#: so a file containing both 'CW' and 'FD' categories is FD, not CW.
_MODALITY_PRIORITY = ('DCS', 'TD_moments', 'TD_gated', 'FD', 'CW')


def get_snirf_modality(filepath):
    """
    Determine a SNIRF file's primary modality.

    Classifies from the full set of data-type codes present, since the first
    column alone would misclassify a frequency-domain file whose DC column is
    written first.

    Parameters
    ----------
    filepath : str
        Path to the .snirf file.

    Returns
    -------
    str
        Modality name, e.g. 'CW', 'TD_gated', 'FD' or 'DCS'.
    """
    with h5py.File(filepath, 'r') as f:
        try:
            ml = read_measurement_list(f, '/nirs/data1')
        except KeyError:
            return 'unknown'

        dt_codes = np.unique(np.array(ml['dataType']).flatten().astype(int))
        if dt_codes.size == 0:
            return 'unknown'

        present = {get_datatype_category(int(c)) for c in dt_codes}
        if 'processed' in present:
            # dataType=99999 ('processed') covers CW-derived OD/conc data
            # (see write_snirf()'s CW/chromophore branches) -- there's no
            # dedicated processed-data modality in the DCS/TD/FD/CW
            # dispatch, and CW is the only class currently reachable via
            # NirsStream.from_snirf() that consumes it.
            present.discard('processed')
            present.add('CW')

        for category in _MODALITY_PRIORITY:
            if category in present:
                if category == 'DCS':
                    return 'DCS_BFI' if 410 in dt_codes else 'DCS_G2'
                return category
        return 'unknown'


def read_measurement_list(h5file, data_path):
    """
    Read a SNIRF file's measurement list.

    Supports both the array encoding and the older indexed encoding.

    Parameters
    ----------
    h5file : h5py.File
        Open SNIRF file.
    data_path : str
        Path to the data group within the file.

    Returns
    -------
    dict
        Keys ``source`` and ``detector`` (one-based indices), ``wavelength_idx``
        (zero-based), ``dataType``, ``dataTypeIndex``, ``dataUnit`` and
        ``dataTypeLabel``, the last carrying a per-column label for processed
        data and None where absent.
    """
    if 'measurementLists' in h5file[data_path]:
        return _read_ml_array_format(h5file, data_path)
    return _read_ml_indexed_format(h5file, data_path)


def _read_ml_array_format(h5file, data_path):
    """Read a measurement list stored in the array encoding."""
    grp = h5file[f'{data_path}/measurementLists']
    n = len(grp['sourceIndex'][()])

    def _ints(key, default=1):
        if key in grp:
            return [int(x) for x in grp[key][()].flatten()]
        return [default] * n

    def _strs(key, default='unknown'):
        if key in grp:
            raw = grp[key][()]
            values = [v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in raw]
            if default is None:
                # write_snirf's array-format writer encodes a missing/None
                # dataTypeLabel as '' (HDF5 string datasets can't hold a
                # per-element None) -- map it back to None here so the
                # array-format read path matches the indexed-format one
                # (_read_ml_indexed_format returns None outright when the
                # dataset is absent), rather than leaking '' into the
                # labeled-component grouping path as a fake component name.
                values = [v if v != '' else None for v in values]
            return values
        return [default] * n

    return {
        'source':         _ints('sourceIndex'),
        'detector':       _ints('detectorIndex'),
        'wavelength_idx': [x - 1 for x in _ints('wavelengthIndex')],  # → 0-based
        'dataType':       _ints('dataType', default=1),
        'dataTypeIndex':  _ints('dataTypeIndex', default=1),
        'dataUnit':       _strs('dataUnit'),
        'dataTypeLabel':  _strs('dataTypeLabel', default=None),
    }


def _read_ml_indexed_format(h5file, data_path):
    """Read a measurement list stored in the indexed encoding."""
    ml_keys = [k for k in h5file[data_path].keys() if re.match(r'measurementList\d+$', k)]
    ml_keys.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    ml = {'source': [], 'detector': [], 'dataType': [], 'wavelength_idx': [], 'dataTypeIndex': [],
          'dataUnit': [], 'dataTypeLabel': []}

    for key in ml_keys:
        p = f'{data_path}/{key}'
        ml['source'].append(int(h5file[f'{p}/sourceIndex'][()].item()))
        ml['detector'].append(int(h5file[f'{p}/detectorIndex'][()].item()))
        ml['wavelength_idx'].append(int(h5file[f'{p}/wavelengthIndex'][()].item()) - 1)

        try:
            ml['dataType'].append(int(h5file[f'{p}/dataType'][()].item()))
        except KeyError:
            ml['dataType'].append(1)

        try:
            ml['dataTypeIndex'].append(int(h5file[f'{p}/dataTypeIndex'][()].item()))
        except KeyError:
            ml['dataTypeIndex'].append(1)

        try:
            u = h5file[f'{p}/dataUnit'][()]
            ml['dataUnit'].append(u.decode('utf-8') if isinstance(u, bytes) else str(u))
        except (KeyError, AttributeError):
            ml['dataUnit'].append('unknown')

        try:
            lbl = h5file[f'{p}/dataTypeLabel'][()]
            ml['dataTypeLabel'].append(lbl.decode('utf-8') if isinstance(lbl, bytes) else str(lbl))
        except (KeyError, AttributeError):
            ml['dataTypeLabel'].append(None)

    return ml



def _filter_measurement_list(measurement_list, dataType_code):
    """
    Restrict a measurement list to one data-type code.

    Used to split a file whose channels mix plain and fluorescence amplitude
    into independent groups before reshaping.

    Parameters
    ----------
    measurement_list : dict
        Measurement list to filter.
    dataType_code : int
        Code to keep.

    Returns
    -------
    dict
        The matching rows, in the same structure.
    """
    dt_array = np.array(measurement_list['dataType']).flatten().astype(int)
    mask = dt_array == dataType_code
    return {k: [v[i] for i in range(len(v)) if mask[i]] for k, v in measurement_list.items()}


def _reshape_measurement_list(raw_data, measurement_list, wavelengths):
    """
    Reshape flat measurement columns into a labelled array.

    Groups columns by source-detector pair, wavelength, and a fourth key
    chosen by modality: the data-type index alone for time-domain and DCS
    data, the data type and index together for frequency-domain data, where a
    per-frequency index overlays the DC, amplitude and phase split, and the
    data type alone otherwise.

    Parameters
    ----------
    raw_data : np.ndarray
        Flat data of shape (n_times, n_columns).
    measurement_list : dict
        Description of those columns.
    wavelengths : array-like
        Wavelengths in nm.

    Returns
    -------
    xr.DataArray
        Shape (time, channel, wavelength) with a fourth axis where the
        modality has one.
    """

    n_channels = len(measurement_list['source'])

    # Identify unique coordinates
    # We use a list for unique_sd_pairs to maintain the order they appear in the file
    sd_pairs_raw = list(zip(measurement_list['source'], measurement_list['detector']))
    unique_sd_pairs = []
    for sd in sd_pairs_raw:
        if sd not in unique_sd_pairs:
            unique_sd_pairs.append(sd)

    unique_wavelengths = np.unique(wavelengths[measurement_list['wavelength_idx']])

    # Check dataType categories to determine modality
    dt_array = np.array(measurement_list['dataType']).flatten().astype(int)
    dti_array = (np.array(measurement_list['dataTypeIndex']).flatten().astype(int)
                 if 'dataTypeIndex' in measurement_list else np.ones(n_channels, dtype=int))

    is_td_gated = int(201) in dt_array
    is_td_moments = int(301) in dt_array
    is_dcs = bool({401, 410} & set(dt_array.tolist()))
    is_fd = bool({101, 102} & set(dt_array.tolist()))
    # Processed per-component data (CW_Stream conc: HbO/HbR/HbT, ...) shares
    # dataType=99999 with OD ('dOD') and every other processed quantity, so
    # it can't be told apart by dataType alone the way TD/DCS/FD can.
    # dataTypeLabel -- not dataTypeIndex -- is the field the spec documents
    # for this: "[dataTypeLabel] is only required if dataType is
    # 'processed' (99999)". dataTypeIndex's one documented meaning for
    # processed data is different (e.g. a stimulus-condition index for
    # HRF-averaged data), so it's left alone (constant, see write_snirf())
    # rather than repurposed as a component counter. More than one distinct
    # dataTypeLabel under dataType=99999 -- e.g. 'HbO'/'HbR'/'HbT' vs OD's
    # constant 'dOD' -- is what signals per-component columns.
    label_array = measurement_list.get('dataTypeLabel', [None] * n_channels)
    non_null_labels = {lbl for lbl in label_array if lbl is not None}
    is_labeled_component = (
        set(dt_array.tolist()) == {99999}
        and len(non_null_labels) > 1
    )

    unique_datatypes = np.unique(dt_array)
    if len(unique_datatypes) > 1 and not is_fd:
        # FD legitimately mixes dataType 1 (DC term) with 101/102 (AC/phase)
        # -- see SNIRF spec Appendix -- so that combination alone isn't a
        # sign of anything wrong and is excluded from this check.
        import logging
        logging.getLogger('milob').warning(f"Multiple dataTypes found in measurement list: {unique_datatypes}. This may indicate a mixed-modality SNIRF file or multiple processed data types. Please verify the contents of the file.")

    # Select the (dataType, dataTypeIndex)-derived key for the 4th dimension,
    # per modality -- see docstring.
    if is_fd:
        keys_per_col = list(zip(dt_array.tolist(), dti_array.tolist()))
        unique_keys = sorted(set(keys_per_col), key=lambda k: (0 if k[0] == 1 else 1, k[1], k[0]))
        dim_label = "dataType"
        dt_for_coords = np.array([k[0] for k in unique_keys])
        dti_for_coords = np.array([k[1] for k in unique_keys])
    elif is_labeled_component:
        # Grouping key is the label itself (spec gives no uniqueness
        # guarantee on dataTypeLabel in general, but write_snirf() only
        # ever emits one column per (channel, component), so per-channel
        # labels are unique in practice). A missing label on an individual
        # column (malformed/foreign file) gets its own placeholder bucket
        # rather than colliding with a real component.
        keys_per_col = [lbl if lbl is not None else f'unlabeled_{i}'
                        for i, lbl in enumerate(label_array)]
        unique_keys = sorted(set(keys_per_col))
        dim_label = "chromophore"
        dt_for_coords = np.full(len(unique_keys), 99999)
        dti_for_coords = np.arange(1, len(unique_keys) + 1)
    elif is_dcs and 'dataTypeIndex' in measurement_list:
        keys_per_col = dti_array.tolist()
        unique_keys = sorted(np.unique(dti_array).tolist())
        dim_label = "tau"
        main_dt_code = 410 if 410 in dt_array else 401
        dt_for_coords = np.full(len(unique_keys), main_dt_code)
        dti_for_coords = np.array(unique_keys)
    elif (is_td_gated or is_td_moments) and 'dataTypeIndex' in measurement_list:
        keys_per_col = dti_array.tolist()
        unique_keys = sorted(np.unique(dti_array).tolist())
        dim_label = "bin" if is_td_gated else "moment"
        main_dt_code = 201 if is_td_gated else 301
        dt_for_coords = np.full(len(unique_keys), main_dt_code)
        dti_for_coords = np.array(unique_keys)
    else:
        keys_per_col = dt_array.tolist()
        unique_keys = sorted(np.unique(dt_array).tolist())
        dim_label = "dataType"
        dt_for_coords = np.array(unique_keys)
        dti_for_coords = np.ones(len(unique_keys), dtype=int)

    n_time = raw_data.shape[0]
    n_sd = len(unique_sd_pairs)
    n_wl = len(unique_wavelengths)
    n_4d = len(unique_keys)

    # Initialize 4D array: (time, sd, wl, dt)
    data_reshaped = np.full((n_time, n_sd, n_wl, n_4d), np.nan)

    # Create maps for fast indexing
    sd_map = {sd: i for i, sd in enumerate(unique_sd_pairs)}
    wl_map = {wl: i for i, wl in enumerate(unique_wavelengths)}
    coord_map = {k: i for i, k in enumerate(unique_keys)}
    col_to_slot = np.array([coord_map[k] for k in keys_per_col])

    for i in range(n_channels):
        sd = (measurement_list['source'][i], measurement_list['detector'][i])
        wl = wavelengths[measurement_list['wavelength_idx'][i]]
        data_reshaped[:, sd_map[sd], wl_map[wl], col_to_slot[i]] = raw_data[:, i]

    # Compute NaN counts here; added to metadata dict below after it is built.
    _n_nan = int(np.isnan(data_reshaped[0]).sum())
    _n_total = n_sd * n_wl * n_4d

    units_for_coords = None
    # If single-slot, reduce to 3D -- but never for FD/DCS/TD-gated/moments/
    # labeled-component, which must always keep their signature 4th
    # dimension even in a degenerate single-slot acquisition (e.g. exactly
    # one tau bin). Labeled-component data always keeps this 4D shape
    # (wavelength axis included, always size 1 -- see is_labeled_component's
    # docstring note) even for n_4d>1; read_snirf() drops the dummy
    # wavelength axis afterward, once the 'chromophore' coordinate itself
    # has been built the same way as DCS/FD/TD's 4th-dimension coordinates.
    if n_4d == 1 and not (is_td_gated or is_td_moments or is_dcs or is_fd or is_labeled_component):
        data_reshaped = data_reshaped[:, :, :, 0]
        dims = ["time", "channel", "wavelength"]
        if 'dataUnit' in measurement_list:
            units_for_coords = [measurement_list['dataUnit'][0]]
    else:
        dims = ["time", "channel", "wavelength", dim_label]
         # Identify units for each unique 4th dimension entry
        units_for_coords = [None] * n_4d
        for i in range(n_channels):
            idx_4d = col_to_slot[i]
            if units_for_coords[idx_4d] is None:
                units_for_coords[idx_4d] = measurement_list['dataUnit'][i]

    # unique_keys *are* the component names in the is_labeled_component
    # branch (the grouping key is the label itself -- see above).
    component_labels = unique_keys if is_labeled_component else None

    metadata = {
        'sources': [sd[0] for sd in unique_sd_pairs],
        'detectors': [sd[1] for sd in unique_sd_pairs],
        'wavelengths': unique_wavelengths,
        'dims': dims,
        'dim_label_4d': dim_label if len(dims) > 3 else None,
        'datatypes': dt_for_coords,
        'datatype_index_4d': dti_for_coords,
        'datatype_names': _get_datatype_names(dt_for_coords),
        'bins': dti_for_coords if is_td_gated else None,
        'moments': dti_for_coords if is_td_moments else None,
        'component_labels': component_labels,
        'channel_labels': [f"S{sd[0]}D{sd[1]}" for sd in unique_sd_pairs],
        'is_td_gated': is_td_gated,
        'is_td_moments': is_td_moments,
        'is_dcs': is_dcs,
        'is_fd': is_fd,
        'is_labeled_component': is_labeled_component,
        'data_units': units_for_coords,
        'n_nan_channels': _n_nan,
        'n_total_channels': _n_total,
    }

    return data_reshaped, metadata



def _get_datatype_names(datatypes):
    """
    Convert SNIRF data-type codes to readable names.

    Parameters
    ----------
    datatypes : list of int
        Data-type codes.

    Returns
    -------
    list of str
        One name per code.
    """
    datatype_map = {
        # 001-100: Raw - Continuous Wave (CW)
        1: 'CW_amplitude',
        51: 'CW_fluorescence_amplitude',

        # 101-200: Raw - Frequency Domain (FD)
        101: 'FD_AC_amplitude',
        102: 'FD_phase',
        151: 'FD_fluorescence_amplitude',
        152: 'FD_fluorescence_phase',

        # 201-300: Raw - Time Domain - Gated (TD Gated)
        201: 'TD_gated_amplitude',
        251: 'TD_gated_fluorescence_amplitude',

        # 301-400: Raw - Time Domain - Moments (TD Moments)
        301: 'TD_moments_amplitude',
        351: 'TD_moments_fluorescence_amplitude',

        # 401-500: Raw - Diffuse Correlation Spectroscopy (DCS)
        401: 'DCS_g2',
        410: 'DCS_BFi',

        # Processed
        99999: 'processed'
    }

    names = []
    for dt in datatypes:
        if dt in datatype_map:
            names.append(datatype_map[dt])
        else:
            # Handle unlisted codes by category
            if 1 <= dt <= 100:
                names.append(f'CW_raw_{dt}')
            elif 101 <= dt <= 200:
                names.append(f'FD_raw_{dt}')
            elif 201 <= dt <= 300:
                names.append(f'TD_gated_raw_{dt}')
            elif 301 <= dt <= 400:
                names.append(f'TD_moments_raw_{dt}')
            elif 401 <= dt <= 500:
                names.append(f'DCS_raw_{dt}')
            else:
                names.append(f'datatype_{dt}')

    return names


def get_datatype_category(datatype):
    """
    Return the broad category of a SNIRF data type.

    Parameters
    ----------
    datatype : int
        Data-type code.

    Returns
    -------
    str
        One of 'CW', 'FD', 'TD_gated', 'TD_moments', 'DCS', 'processed' or
        'unknown'.

    Examples
    --------
    >>> get_datatype_category(1)
    'CW'
    >>> get_datatype_category(410)
    'DCS'
    """
    if 1 <= datatype <= 100:
        return 'CW'
    elif 101 <= datatype <= 200:
        return 'FD'
    elif 201 <= datatype <= 300:
        return 'TD_gated'
    elif 301 <= datatype <= 400:
        return 'TD_moments'
    elif 401 <= datatype <= 500:
        return 'DCS'
    elif datatype == 99999:
        return 'processed'
    else:
        return 'unknown'


def identify_moment_from_units(unit_str):
    """
    Infer a moment order from a SNIRF data-unit string.

    For files whose moment orders are not recorded in the standard field.

    Parameters
    ----------
    unit_str : str
        The data unit.

    Returns
    -------
    str or None
        Moment label such as 'm0', 'm1' or 'm2'.
    """
    if not unit_str or unit_str.lower() in ['unitless', 'counts', 'au']:
        return 'm0'

    # Clean string
    u = unit_str.lower().strip()

    # Check for mean arrival time (Time units) for m1
    if u in ['s', 'ms', 'ns', 'ps']:
        return 'm1'

    # Check for variance (Time units squared) for m2
    if u in ['s^2', 'ms^2', 'ns^2', 'ps^2', 's2', 'ms2','ns2', 'ps2', 's**2', 'ms**2', 'ns**2', 'ps**2']:
        return 'm2'

    return None



def read_snirf(filepath, *, sc_threshold, name=None, **kwargs):
    """
    Read a SNIRF file into the matching stream subclass.

    Parameters
    ----------
    filepath : str
        Path to the .snirf file.
    sc_threshold : float or None
        Source-detector distance in mm below which a channel is classified as
        short. Pass None if the probe design has no short-separation
        channels.
    name : str, optional
        Stream name. Derived from the modality if omitted.
    **kwargs
        Forwarded to the stream constructor.

    Returns
    -------
    NirsStream
        A CW, TD, FD or DCS stream, according to the file's modality.
    """
    with h5py.File(filepath, 'r') as f:
        # Read metadata tags
        metadata_path = '/nirs/metaDataTags'

        def get_tag(tag_name, default='unknown'):
            if metadata_path not in f:
                return default
            if tag_name in f[metadata_path]:
                val = f[f'{metadata_path}/{tag_name}'][()]
                return val.decode('utf-8').strip() if isinstance(val, bytes) else str(val).strip()
            return default

        lengthUnit = get_tag('LengthUnit', 'cm')
        timeUnit = get_tag('TimeUnit', 's')
        # SNIRF has no native field for "raw vs. processed" -- unlike TD
        # gated/moments or CW od/conc, DC/AC/phase (FD) and g2/BFi (DCS)
        # use the same dataType whether or not a within-space transform
        # (filter, motion correction, ...) has been applied. write_snirf()
        # stashes stream.status in this custom metaDataTag (spec explicitly
        # allows additional vendor tags) so 'processed' survives the round
        # trip; files without it (e.g. real instrument output) default to
        # 'raw', matching prior behaviour.
        status_tag = get_tag('MILOB_Status', 'raw')

        # Instrument identification, where the file carries it. These are
        # optional SNIRF metaDataTags that real acquisition software commonly
        # writes; they are kept as flat string attrs (under their SNIRF names,
        # so their origin stays obvious) because downstream consumers want
        # them -- io/bids.py maps ManufacturerName/Model onto BIDS's
        # Manufacturer/ManufacturersModelName, which are recommended fields
        # nothing else in the recording can supply.
        instrument_tags = {
            tag: get_tag(tag, None)
            for tag in ('ManufacturerName', 'Model')
        }
        instrument_tags = {k: v for k, v in instrument_tags.items() if v}

        # Read Probe Geometry
        probe_path = '/nirs/probe'
        s_pos = f[f'{probe_path}/sourcePos3D'][()]
        d_pos = f[f'{probe_path}/detectorPos3D'][()]
        wavelengths = f[f'{probe_path}/wavelengths'][()].flatten()

        s_pos_2d = f[f'{probe_path}/sourcePos2D'][()]   if 'sourcePos2D'   in f[probe_path] else None
        d_pos_2d = f[f'{probe_path}/detectorPos2D'][()]  if 'detectorPos2D' in f[probe_path] else None

        landmarks = f[f'{probe_path}/landmarkPos3D'][()]  if 'landmarkPos3D'  in f[probe_path] else None
        if landmarks is not None and landmarks.shape[1] > 3:
            # SNIRF allows a 4th column (a 1-indexed selection/label index);
            # keep only the x,y,z spatial columns.
            landmarks = landmarks[:, :3]
        landmark_labels = (
            [v.decode('utf-8') if isinstance(v, bytes) else str(v)
             for v in f[f'{probe_path}/landmarkLabels'][()]]
            if 'landmarkLabels' in f[probe_path] else None
        )

        # Read Time-Domain (TD) Specific Timing Info
        time_delays = None
        time_delay_widths = None
        if 'timeDelays' in f[probe_path]:
            time_delays = f[f'{probe_path}/timeDelays'][()]
        if 'timeDelayWidths' in f[probe_path]:
            time_delay_widths = f[f'{probe_path}/timeDelayWidths'][()]

        # Read FD/DCS-specific probe fields -- the physical values that
        # dataTypeIndex indexes into (frequency index for FD, correlation-
        # delay-bin index for DCS); see _reshape_measurement_list().
        mod_frequencies = None
        if 'frequencies' in f[probe_path]:
            mod_frequencies = f[f'{probe_path}/frequencies'][()].flatten().astype(float)

        correlation_time_delays = None
        if 'correlationTimeDelays' in f[probe_path]:
            correlation_time_delays = f[f'{probe_path}/correlationTimeDelays'][()].flatten().astype(float)


        # Read Time-Series Data
        data_path = '/nirs/data1'
        raw_values = f['/nirs/data1/dataTimeSeries'][()]    # (time, channels)
        times = f[f'{data_path}/time'][()].flatten().astype(float)

        # Homer3 writes dataTimeSeries as (channels, time) rather than (time, channels).
        # Use the measurement list count as the ground truth for the channels dimension.
        n_ml = sum(1 for k in f[data_path] if k.startswith('measurementList'))
        if raw_values.shape[0] == n_ml and raw_values.shape[1] == len(times):
            raw_values = raw_values.T

        # Make sure the time dimensions match
        # (Added because LUMO .snirf conversion from .lufr files saves only 2 data points)
        if len(times) < raw_values.shape[0]:
            dt = times[1] - times[0]
            times = times[0] + np.arange(raw_values.shape[0]) * dt
        elif len(times) > raw_values.shape[0]:
            # Homer3 pre-allocates the time array; trim to actual data length
            times = times[:raw_values.shape[0]]

        # Force first time to be exactly 0 (standardizing relative time)
        times = times - times[0]
        fs = 1.0 / np.diff(times).mean() if len(times) > 1 else 1.0

        # Read measurement list and reshape
        # (time, channels*wavelengths, etc.) -> (time, sd_pairs, wavelengths, etc.)
        ml = read_measurement_list(f, data_path)

        # A CW file's channels may legitimately mix dataType 1 (plain
        # amplitude) with 51 (fluorescence amplitude) -- see SNIRF spec
        # Appendix, '001-100: Raw - Continuous Wave (CW)'. These are two
        # physically distinct measurements, not two slices of one signal, so
        # they must not be combined into a single array's 4th axis the way
        # e.g. FD's AC/phase (101/102) legitimately are. Restrict to the
        # primary (1) group here; the fluorescence (51) group is not read
        # by this release.
        present_dt = set(np.unique(np.array(ml['dataType']).flatten().astype(int)).tolist())
        if present_dt <= {1, 51} and len(present_dt) > 1:
            import logging, os
            logging.getLogger('milob').warning(
                f"{os.path.basename(filepath)}: found both plain CW amplitude (dataType=1) "
                "and fluorescence amplitude (dataType=51) channels. Only the plain-amplitude "
                "stream is returned here -- load via Session.from_snirf() to also get the "
                "fluorescence channels as a sibling stream."
            )
            ml = _filter_measurement_list(ml, 1)

        data_reshaped, metadata = _reshape_measurement_list(raw_values, ml, wavelengths)

        if metadata['n_nan_channels'] > 0:
            import logging, os
            logging.getLogger('milob').warning(
                f"{metadata['n_nan_channels']}/{metadata['n_total_channels']} channel combinations "
                f"have no data (filled with NaN) — {os.path.basename(filepath)}"
            )

        # Create Probe with channel configuration
        probe = Probe(s_pos, d_pos, wavelengths, lengthUnit=lengthUnit,
                      s_pos_2d=s_pos_2d, d_pos_2d=d_pos_2d,
                      landmark_pos=landmarks, landmark_labels=landmark_labels,
                      sc_threshold=sc_threshold)
        # Define the channel configuration based on the unique SD pairs
        probe._channels = {
            'sources': metadata['sources'],
            'detectors': metadata['detectors']
        }

        # Build coordinates dictionary
        n_channels = len(metadata['sources'])
        dims = metadata['dims'] # 'bin', 'moment', 'tau', or 'AC/phase/DC' depending on modality

        coords = {
            "time": times,
            "channel": metadata['channel_labels'],
            "wavelength": metadata['wavelengths'],

            # Helper coords for the 'channel' dimension
            "channel_idx": ("channel", np.arange(n_channels)),
            "source": ("channel", metadata['sources']),
            "detector": ("channel", metadata['detectors']),
            "distance": ("channel", probe.distances),
        }

        # Specific handling of 4th dimension (TD/FD/DCS) if present
        if len(dims) > 3:
            dim4_name = dims[3]
            coords["dataType"] = (dim4_name, metadata['datatypes']) # Add dataType codes as a coord for the 4th dim
            coords["datatype_name"] = (dim4_name, metadata['datatype_names'])   # Add readable names as a coord for the 4th dim

            if metadata.get('is_td_gated'):
                coords[dim4_name] = (dim4_name, metadata['bins']) # bin indices to 4th dim
                if time_delays is not None:
                    coords['timeDelays'] = (dim4_name, time_delays) # associated with 4th dim (bins)
                if time_delay_widths is not None:
                    coords['timeDelayWidths'] = (dim4_name, time_delay_widths)
            elif metadata.get('is_td_moments'):
                coords[dim4_name] = (dim4_name, metadata['moments'])
            elif metadata.get('is_labeled_component'):
                # Real component names (e.g. 'HbO'/'HbR'/'HbT') recovered
                # from dataTypeLabel, keyed by the per-component
                # dataTypeIndex -- see write_snirf()'s chromophore/component
                # branch and _reshape_measurement_list()'s is_labeled_component.
                coords[dim4_name] = (dim4_name, metadata['component_labels'])
            elif metadata.get('is_dcs'):
                # dataTypeIndex per slot (1-based correlation-delay-bin index)
                dti = np.asarray(metadata['datatype_index_4d'], dtype=int)
                if correlation_time_delays is not None:
                    coords[dim4_name] = (dim4_name, correlation_time_delays[dti - 1])
                else:
                    import logging, os
                    logging.getLogger('milob').warning(
                        f"{os.path.basename(filepath)}: DCS file has no "
                        "/nirs/probe/correlationTimeDelays -- 'tau' will hold "
                        "raw 1-based bin indices, not physical delay times in seconds."
                    )
                    coords[dim4_name] = (dim4_name, dti.astype(float))
            elif metadata.get('is_fd'):
                # dataType (1/101/102) and dataTypeIndex (1-based frequency
                # index, 1 for the DC term) per slot -- combine with
                # probe.frequencies to recover the real modulation frequency
                # for each slot; convert_snirf_fd_to_complex() uses this to
                # assemble the complex 'freq' axis, including for files with
                # more than one modulation frequency.
                dt_codes  = np.asarray(metadata['datatypes'], dtype=int)
                dti_codes = np.asarray(metadata['datatype_index_4d'], dtype=int)
                if mod_frequencies is not None:
                    coords['frequency_Hz'] = (
                        dim4_name,
                        np.where(dt_codes == 1, 0.0, mod_frequencies[dti_codes - 1])
                    )
                else:
                    import logging, os
                    logging.getLogger('milob').warning(
                        f"{os.path.basename(filepath)}: FD file has no "
                        "/nirs/probe/frequencies -- modulation frequency cannot "
                        "be recovered; downstream conversion will fall back to "
                        "a 0.0 Hz placeholder."
                    )
                coords[dim4_name] = (dim4_name, dt_codes)
            else:
                coords[dim4_name] = (dim4_name, metadata['datatypes'])
        else:
            # For 3D data (CW), we store dataType as an attribute
            coords["dataType"] = metadata['datatypes'][0]
            coords['datatype_name'] = metadata['datatype_names'][0]

        if metadata.get('is_labeled_component'):
            # Component data (CW_Stream conc: HbO/HbR/HbT, ...) has no real
            # wavelength axis -- write_snirf() always used a dummy
            # wavelengthIndex=1 for these columns (SNIRF requires the
            # field, but composition has already been unmixed across
            # wavelengths), so this axis is always size 1 here. Drop it
            # now that the 'chromophore' coordinate itself has been built
            # above, rather than carrying a spurious size-1 'wavelength' dim.
            wl_axis = dims.index('wavelength')
            data_reshaped = np.take(data_reshaped, 0, axis=wl_axis)
            dims = [d for d in dims if d != 'wavelength']
            coords.pop('wavelength', None)

        # Create xarray
        data_xr = xr.DataArray(
            data_reshaped,
            coords=coords,
            dims=dims,
            attrs={
                'sampling_rate': round(1.0 / np.diff(times).mean(), 1) if len(times) > 1 else None,
                'timeUnit': timeUnit,
                'lengthUnit': probe.lengthUnit,
                'status': status_tag,
                **instrument_tags,
            }
        )

        # --- Loading in moments.snirf files ---
        if metadata.get('is_td_moments'):
            data_xr.attrs['status'] = 'moment' # overwrite default status 'raw'

            # Scaling factors to reach seconds (s)
            TIME_TO_SECONDS = {
                's': 1.0,
                'ms': 1e-3,
                'ns': 1e-9,
                'ps': 1e-12,
                'unknown': 1.0
            }

            new_labels=[]

            # momentOrders is spec-legal to omit (SNIRF spec doesn't mark it
            # required), so this has to be a membership check, not just a
            # None check on the read value -- reading it unconditionally
            # raised KeyError on a genuinely momentOrders-less file instead
            # of falling through to the dataUnit-based fallback below.
            moment_orders = f[f'{probe_path}/momentOrders'][()] if 'momentOrders' in f[probe_path] else None

            if moment_orders is not None:
                mapping = {0: 'm0', 1: 'm1', 2: 'm2'}
                new_labels = [mapping.get(int(m), f"m{m}") for m in moment_orders]

            # Fallback to dataUnits if momentOrders is missing (useful for Kernel devices)
            elif metadata.get('data_units'):
                new_labels = [identify_moment_from_units(u) for u in metadata['data_units']]

            # Rescale moment data values to seconds
            if new_labels and metadata.get('data_units'):
                for i, u in enumerate(metadata['data_units']):
                    m_type = new_labels[i]
                    if m_type =='m0':
                        continue # skip as m0 is unitless (intensity)

                    base_unit = u.lower().replace('^2', '').replace('2', '').replace('**2', '').strip()
                    factor = TIME_TO_SECONDS.get(base_unit, 1.0)

                    if m_type=='m1':
                        data_xr.values[..., i] *= factor
                    elif m_type=='m2':
                        data_xr.values[..., i] *= (factor ** 2)

            if new_labels:
                data_xr.coords['moment'] = new_labels
                data_xr.attrs['moment_unit'] = 's'


            # Define canonical order (m0, m1, m2)
            MOM_ORDER = {'m0': 0, 'm1': 1, 'm2': 2}

            # Only reorder if moment dimension exists
            if 'moment' in data_xr.dims:
                moments = list(data_xr.coords['moment'].values)

                # Sort indices based on desired order
                sorted_idx = sorted(
                    range(len(moments)),
                    key=lambda i: MOM_ORDER.get(str(moments[i]), 99)
                )

                # Reorder data + coordinates together
                data_xr = data_xr.isel(moment=sorted_idx)

        # Read Events
        events = Events()
        stim_keys = [k for k in f['/nirs'].keys() if 'stim' in k]
        for s_key in stim_keys:
            group = f[f'/nirs/{s_key}']
            label = group['name'][()].decode('utf-8')
            data = group['data'][()] # [onset, duration, amplitude]
            for row in data:
                events.add_event(onset=row[0], duration=row[1], value=row[2], label=label)

        return data_xr, probe, events


# NEW VERSION
def read_snirf_aux(filepath, data_xr=None):
    """
    Read the auxiliary channels from a SNIRF file.

    Groups sharing a time vector are stacked into one array; groups with
    differing time vectors are returned separately.

    Parameters
    ----------
    filepath : str
        Path to the .snirf file.
    data_xr : xr.DataArray, optional
        The primary stream's data, used to align the time base.

    Returns
    -------
    dict
        Key ``generic`` holding an array or list of arrays, and ``irf``
        holding a time-domain instrument response or None. The instrument
        response is read from a vendor label convention rather than the SNIRF
        specification, which has no dedicated field for one; a response stored
        as its own sibling file is preferred.
    """
    with h5py.File(filepath, 'r') as f:
        if '/nirs' not in f:
            return None
        nirs_grp = f['/nirs']

        aux_keys = sorted(
            [k for k in nirs_grp.keys() if re.match(r'^aux\d+$', k)],
            key=lambda x: int(re.search(r'\d+', x).group())
        )

        if not aux_keys:
            return None

        generic_groups = []
        irf_entries = []

        # Loop over aux groups
        for key in aux_keys:
            grp = nirs_grp[key]
            label = grp['name'][()].decode('utf-8').strip() if 'name' in grp else key

            raw = grp['dataTimeSeries'][()]
            if raw.ndim == 1:
                raw = raw[:, np.newaxis]  # (time, 1)

            times = grp['time'][()].flatten().astype(float)
            if 'timeOffset' in grp:
                # SNIRF stores this as a scalar, but some writers emit a
                # length-1 array (e.g. shape (1,)) rather than a true 0-d
                # array -- float() on that raises TypeError on modern numpy.
                # .item() accepts any single-element array regardless of shape.
                times = times + grp['timeOffset'][()].item()
            times = times - times[0]

            # --- IRF detection ---
            # case for specific to Kernel DevKit moments.snirf files with IRF info included in aux
            if label.startswith("irf-moments"):

                match = re.match(r"irf-moments_(\d+)_(\w+)", label)

                if match:
                    wavelength = int(match.group(1))

                    moment_map = {
                        "sum": "m0",
                        "mean": "m1",
                        "var": "m2"
                    }
                    moment = moment_map.get(match.group(2))

                    irf_entries.append({
                        "data": raw,
                        "time": times,
                        "wavelength": wavelength,
                        "moment": moment,
                        "labels": grp['dataNames'][()] # optional
                    })
                continue


            # --- Generic groups --- (unchanged)
            unit = None
            if 'dataUnit' in grp:
                u = grp['dataUnit'][()]
                unit = u.decode('utf-8').strip() if isinstance(u, bytes) else str(u).strip()

            n_signals = raw.shape[1]
            if n_signals == 1:
                sig_labels = [label]
            else:
                sig_labels = [f"{label}_{i}" for i in range(n_signals)]

            generic_groups.append({
                'times': times,
                'data': raw,
                'labels': sig_labels,
                'unit': unit
            })


        # Build generic output
        generic_out = None

        if generic_groups:

            # Try to merge all groups that share the same time axis
            ref_times = generic_groups[0]['times']
            same_time = all(
                len(g['times']) == len(ref_times) and np.allclose(g['times'], ref_times)
                for g in generic_groups
            )

            if same_time:
                all_data = np.concatenate([g['data'] for g in generic_groups], axis=1)
                all_labels = [lbl for g in generic_groups for lbl in g['labels']]
                fs = 1.0 / np.diff(ref_times).mean() if len(ref_times) > 1 else None

                generic_out = xr.DataArray(
                    all_data,
                    coords={'time': ref_times, 'signal': all_labels},
                    dims=['time', 'signal'],
                    attrs={'sampling_rate': fs}
                )
        
            else:
            # Different time axes — return a list of DataArrays
                out = []
                for g in generic_groups:
                    fs = 1.0 / np.diff(g['times']).mean() if len(g['times']) > 1 else None
                    out.append(xr.DataArray(
                        g['data'],
                        coords={'time': g['times'], 'signal': g['labels']},
                        dims=['time', 'signal'],
                        attrs={'sampling_rate': fs}
                    ))
                generic_out = out


        # Build IRF output
        irf_out = None

        if irf_entries and data_xr is not None:

            # assume consistent time vector
            time = irf_entries[0]['time']

            # collect all channels
            channels = data_xr.coords["channel"].values
            sources = data_xr.coords["source"].values
            detectors = data_xr.coords["detector"].values

            wavelengths = sorted(set(e["wavelength"] for e in irf_entries))
            moments = ["m0", "m1", "m2"]

            irf_arr = np.zeros((len(time), len(channels), len(wavelengths), len(moments)))

            wl_idx = {w: i for i, w in enumerate(wavelengths)}
            mom_idx = {m: i for i, m in enumerate(moments)}

            for entry in irf_entries:
                wl_i = wl_idx[entry["wavelength"]]
                mom_i = mom_idx[entry["moment"]]
                data = entry["data"]
                
                # unit conversion ps -> s
                if entry["moment"] == "m1":
                    data = data * (1e-12)
                elif entry["moment"] == "m2":
                    data = data * (1e-12)**2

                irf_arr[:, :, wl_i, mom_i] = data # assume same ordering
            

            irf_out = xr.DataArray(
                irf_arr,
                dims=["time", "channel", "wavelength", "moment"],
                coords = {
                    "time": time,
                    "channel": channels,
                    "source": ("channel", sources),
                    "detector": ("channel", detectors),
                    "wavelength": wavelengths,
                    "moment": moments
                },
                name="irf_moments"
            )

        return {
            "generic": generic_out,
            "irf": irf_out
        }


def extract_ml_number(key):
    # Extract number from 'MeasurementList123' -> 123
    import re
    match = re.search(r'measurementList(\d+)', key)
    return int(match.group(1)) if match else 0


def convert_snirf_fd_to_complex(data_xr: xr.DataArray) -> xr.DataArray:
    """
    Pack real frequency-domain data types into a complex array.

    SNIRF stores the DC amplitude, the modulation amplitude and the phase as
    separate real data types, with distinct modulation frequencies
    distinguished by their index. These are combined into a single complex
    array carrying a frequency axis, with the DC entry real and each
    modulation entry the amplitude times its phase factor.

    Parameters
    ----------
    data_xr : xr.DataArray
        Shape (time, channel, wavelength, dataType), real. Must carry the
        amplitude and phase types, and may carry the DC type.

    Returns
    -------
    xr.DataArray
        Shape (time, channel, wavelength, freq), complex. Frequencies come
        from the file's own frequency list where present.
    """
    import warnings

    if 'dataType' not in data_xr.dims:
        raise ValueError(
            "convert_snirf_fd_to_complex expects a 'dataType' dimension. "
            f"Got dims: {data_xr.dims}"
        )

    dt_vals = data_xr.dataType.values
    has_dc    = 1   in dt_vals
    has_ac    = 101 in dt_vals
    has_phase = 102 in dt_vals

    if not (has_ac and has_phase):
        raise ValueError(
            "FD SNIRF data must contain both AC amplitude (dataType=101) "
            f"and phase (dataType=102). Found dataTypes: {dt_vals}"
        )

    if 'frequency_Hz' in data_xr.coords:
        freq_per_slot = np.asarray(data_xr.coords['frequency_Hz'].values, dtype=float)
    else:
        mod_freq = data_xr.attrs.get('modulation_frequency_Hz', None)
        if mod_freq is None:
            warnings.warn(
                "Modulation frequency not found in SNIRF attrs "
                "('modulation_frequency_Hz') or as a 'frequency_Hz' coordinate. "
                "freq coordinate will be set to 0.0 Hz as a placeholder — "
                "set data.attrs['modulation_frequency_Hz'] before using FD_Stream.",
                UserWarning,
                stacklevel=3
            )
            mod_freq = 0.0
        freq_per_slot = np.where(dt_vals == 1, 0.0, float(mod_freq))

    mod_freqs = sorted({float(f) for f in freq_per_slot[dt_vals != 1]})

    slices, freq_values = [], []

    if has_dc:
        dc_idx = int(np.flatnonzero(dt_vals == 1)[0])
        slices.append(data_xr.isel(dataType=dc_idx).values.astype(complex))
        freq_values.append(0.0)

    for freq in mod_freqs:
        ac_idx    = np.flatnonzero((dt_vals == 101) & (freq_per_slot == freq))
        phase_idx = np.flatnonzero((dt_vals == 102) & (freq_per_slot == freq))
        if len(ac_idx) != 1 or len(phase_idx) != 1:
            raise ValueError(
                f"Expected exactly one AC/phase pair at {freq} Hz, found "
                f"{len(ac_idx)} AC and {len(phase_idx)} phase entries."
            )
        ac_vals    = data_xr.isel(dataType=int(ac_idx[0])).values
        phase_vals = data_xr.isel(dataType=int(phase_idx[0])).values
        slices.append((ac_vals * np.exp(1j * phase_vals)).astype(complex))
        freq_values.append(freq)

    combined    = np.stack(slices, axis=-1)
    freq_values = np.array(freq_values)

    # Rebuild coords: drop dataType-related entries, add freq
    new_coords = {}
    for name, coord in data_xr.coords.items():
        if 'dataType' in getattr(coord, 'dims', ()):
            continue
        if name in ('dataType', 'datatype_name', 'frequency_Hz'):
            continue
        new_coords[name] = coord
    new_coords['freq'] = freq_values

    new_dims = [d if d != 'dataType' else 'freq' for d in data_xr.dims]

    new_attrs = data_xr.attrs.copy()
    new_attrs['modulation_frequency_Hz'] = float(mod_freqs[0]) if mod_freqs else 0.0
    new_attrs['freq_encoding'] = (
        'freq=0 → DC (real; imaginary=0);  '
        'freq=f_mod → AC_amplitude * exp(i * phase_rad)'
    )

    return xr.DataArray(
        combined,
        coords=new_coords,
        dims=new_dims,
        attrs=new_attrs
    )


# ─────────────────────────────────────────────────────────────────────────────
# SNIRF writer
# ─────────────────────────────────────────────────────────────────────────────

def _s(value):
    """Decode a value read from HDF5 into a plain Python string."""
    return np.bytes_(str(value))


def _get_channel_sd_indices(stream):
    """
    Return the one-based source and detector index for each channel.

    Parameters
    ----------
    stream : NirsStream
        Stream whose probe supplies the indices.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Source and detector indices.
    """
    data = stream.data
    if 'source' in data.coords and 'detector' in data.coords:
        return data.coords['source'].values.astype(int), data.coords['detector'].values.astype(int)
    sources, detectors = [], []
    for ch in data.channel.values:
        m = re.match(r'S(\d+)D(\d+)', str(ch))
        if m:
            sources.append(int(m.group(1)))
            detectors.append(int(m.group(2)))
        else:
            sources.append(1)
            detectors.append(1)
    return np.array(sources, dtype=int), np.array(detectors, dtype=int)


class MeasurementColumn(NamedTuple):
    """
    Description of one column of a SNIRF data time series.

    A stream's payload axis is flattened into these columns, so that the
    measurement list and any external per-measurement table describe the same
    columns in the same order.

    Attributes
    ----------
    channel_index : int
        Zero-based index into the stream's channels.
    source_index, detector_index : int
        One-based indices into the probe's optodes.
    wavelength_index : int
        One-based index into the probe's wavelengths. Fixed at 1 for
        label-carrying processed data, which is identified by ``label``
        instead.
    datatype : int
        SNIRF data-type code.
    datatype_index : int
        One-based position along the payload axis, such as a modulation
        frequency, time gate or correlation delay, or the moment order for
        time-domain moments. 1 where that axis is absent.
    unit : str or None
        SNIRF data unit for this column.
    label : str or None
        Per-column label for processed data.
    """
    channel_index: int
    source_index: int
    detector_index: int
    wavelength_index: int
    datatype: int
    datatype_index: int
    unit: Optional[str]
    label: Optional[str]


def _enumerate_columns(stream):
    """
    Flatten a stream into its SNIRF columns.

    Modality is determined from the stream's dimensions and status.

    Parameters
    ----------
    stream : NirsStream
        Stream to enumerate.

    Returns
    -------
    list of tuple of (np.ndarray, MeasurementColumn)
        Each column's time series, as a view rather than a copy, with its
        description.
    """
    data   = stream.data
    dims   = set(data.dims)
    status = stream.status

    src_idx, det_idx = _get_channel_sd_indices(stream)
    n_ch = len(data.channel)

    # wavelength value → 1-based probe index
    wavelengths = stream.probe.wavelengths if stream.probe is not None else np.array([])
    wl_to_idx = {float(wl): i + 1 for i, wl in enumerate(wavelengths)}

    # Each entry: (data_1d, src, det, wl_1based, dataType, dataTypeIndex, unit, label)
    cols = []

    # ── FD — complex (time, channel, wavelength, freq) ───────────────────────
    if 'freq' in dims and np.iscomplexobj(data.values):
        freqs     = data.freq.values
        wl_vals   = data.wavelength.values
        mod_freqs = freqs[freqs != 0]
        # 1-based index into probe.frequencies for each non-zero modulation freq
        freq_to_idx = {float(f): i + 1 for i, f in enumerate(mod_freqs)}

        for ch_i in range(n_ch):
            for wl_i, wl in enumerate(wl_vals):
                wl_1 = wl_to_idx.get(float(wl), wl_i + 1)
                for f_i, freq in enumerate(freqs):
                    z = data.values[:, ch_i, wl_i, f_i]
                    if freq == 0.0:
                        # DC component: CW amplitude (real part only)
                        cols.append((z.real, MeasurementColumn(
                            ch_i, src_idx[ch_i], det_idx[ch_i], wl_1, 1, 1, None, None)))
                    else:
                        fi = freq_to_idx.get(float(freq), f_i + 1)
                        cols.append((np.abs(z), MeasurementColumn(
                            ch_i, src_idx[ch_i], det_idx[ch_i], wl_1, 101, fi, None, None)))
                        cols.append((np.angle(z), MeasurementColumn(
                            ch_i, src_idx[ch_i], det_idx[ch_i], wl_1, 102, fi, None, None)))

    # ── DCS — g2 or BFi (time, channel, wavelength, tau) ────────────────────
    elif 'tau' in dims:
        wl_vals = data.wavelength.values
        n_tau   = len(data.tau)
        dt      = 410 if status == 'bfi' else 401

        for ch_i in range(n_ch):
            for wl_i, wl in enumerate(wl_vals):
                wl_1 = wl_to_idx.get(float(wl), wl_i + 1)
                for tau_i in range(n_tau):
                    cols.append((data.values[:, ch_i, wl_i, tau_i], MeasurementColumn(
                        ch_i, src_idx[ch_i], det_idx[ch_i], wl_1, dt, tau_i + 1, None, None)))

    # ── TD gated — (time, channel, wavelength, bin) ──────────────────────────
    elif 'bin' in dims:
        wl_vals = data.wavelength.values
        n_bins  = len(data.bin)

        for ch_i in range(n_ch):
            for wl_i, wl in enumerate(wl_vals):
                wl_1 = wl_to_idx.get(float(wl), wl_i + 1)
                for bin_i in range(n_bins):
                    cols.append((data.values[:, ch_i, wl_i, bin_i], MeasurementColumn(
                        ch_i, src_idx[ch_i], det_idx[ch_i], wl_1, 201, bin_i + 1, None, None)))

    # ── TD moments — (time, channel, wavelength, moment) ────────────────────
    elif 'moment' in dims:
        wl_vals  = data.wavelength.values
        moments  = data.moment.values
        MOM_ORD  = {'m0': 0, 'm1': 1, 'm2': 2}
        MOM_UNIT = {'m0': 'au', 'm1': 's', 'm2': 's^2'}

        for ch_i in range(n_ch):
            for wl_i, wl in enumerate(wl_vals):
                wl_1 = wl_to_idx.get(float(wl), wl_i + 1)
                for m_i, mom in enumerate(moments):
                    cols.append((data.values[:, ch_i, wl_i, m_i], MeasurementColumn(
                        ch_i, src_idx[ch_i], det_idx[ch_i], wl_1,
                        301, MOM_ORD.get(str(mom), m_i),
                        MOM_UNIT.get(str(mom), None), None)))

    # ── Chromophore / tissue components — (time, channel, chromophore) ──────
    # 'chromophore' is CW_Stream's conc dim (processing.mbll.od_to_concentration);
    # 'component' is TissueStream's (OptPropStream.to_concentration() /
    # FD_Stream.fit_to_conc(), core.tissue_stream.TissueStream) -- same SNIRF
    # layout either way, just a different xarray dim name.
    elif 'chromophore' in dims or 'component' in dims:
        CHROM_LABEL = {
            'HbO': 'HbO', 'HbR': 'HbR', 'HbT': 'HbT',
            'H2O': 'H2O', 'Lipid': 'Lipid', 'StO2': 'StO2',
            'BFi': 'BFi', 'mua': 'mua', 'musp': 'musp',
            'A': 'scattering_A', 'b': 'scattering_b',
        }
        component_dim = 'chromophore' if 'chromophore' in dims else 'component'
        chroms = data[component_dim].values

        for ch_i in range(n_ch):
            for c_i, chrom in enumerate(chroms):
                lbl = CHROM_LABEL.get(str(chrom), str(chrom))
                # dataTypeIndex stays constant (1), matching OD -- the spec's
                # one documented use of dataTypeIndex for processed data
                # (dataType=99999) is unrelated (e.g. a stimulus-condition
                # index for HRF-averaged data), not component disambiguation.
                # dataTypeLabel is what the spec actually designates for
                # this ("only required if dataType is 'processed'"), and is
                # what _reshape_measurement_list()'s labeled-component
                # branch groups on when reading this back.
                cols.append((data.values[:, ch_i, c_i], MeasurementColumn(
                    ch_i, src_idx[ch_i], det_idx[ch_i], 1, 99999, 1, None, lbl)))

    # ── CW / generic wavelength dim — (time, channel, wavelength) ────────────
    # ── Parameter-space streams — (time, channel, wavelength, op) ───────────
    # Caught explicitly so an OptPropStream doesn't fall through to the
    # 'wavelength' branch below, which would index past the 'op' axis and
    # write silently mangled data. SNIRF has no dataType for a recovered
    # parameter vector, and the accompanying uncertainty/obs_params sidecars
    # have nowhere to go either -- so refuse rather than lose them.
    elif 'op' in dims:
        raise NotImplementedError(
            "SNIRF export of a parameter-space stream (dims include 'op': "
            f"{list(data.op.values)}) isn't supported -- the format has no "
            "dataType for recovered optical properties, and it cannot carry "
            "the stream's uncertainty/obs_params. Export the derived "
            "concentrations instead (OptPropStream.to_concentration()), or "
            "save the DataArray directly with xarray (.to_netcdf())."
        )

    elif 'wavelength' in dims:
        wl_vals = data.wavelength.values
        if status in ('raw', 'processed'):
            # 'processed' (filtered/motion-corrected/quality-screened raw
            # intensity, see Datastream._mark_processed) is still
            # intensity-shaped data -- same SNIRF dataType as 'raw'.
            dt, lbl = 1, None
        elif status == 'od':
            dt, lbl = 99999, 'dOD'
        else:
            dt, lbl = 99999, None

        for ch_i in range(n_ch):
            for wl_i, wl in enumerate(wl_vals):
                wl_1 = wl_to_idx.get(float(wl), wl_i + 1)
                cols.append((data.values[:, ch_i, wl_i], MeasurementColumn(
                    ch_i, src_idx[ch_i], det_idx[ch_i], wl_1, dt, 1, None, lbl)))

    else:
        raise ValueError(
            f"Cannot determine SNIRF layout: dims={set(data.dims)}, status='{status}'. "
            "Expected one of: 'wavelength', 'chromophore', 'component', 'tau', 'bin', 'moment', "
            "or complex 'freq'."
        )

    return cols


def _build_measurement_columns(stream):
    """
    Decompose a stream into the flat layout SNIRF requires.

    Parameters
    ----------
    stream : NirsStream
        Stream to decompose.

    Returns
    -------
    flat_data : np.ndarray
        Shape (n_times, n_columns), real.
    ml : dict
        Measurement-list arrays: source, detector and wavelength indices, data
        types and indices, units and labels.
    """
    cols = _enumerate_columns(stream)

    flat_data = np.column_stack([payload for payload, _ in cols]).astype(np.float64)
    ml = {
        'sourceIndex':     np.array([c.source_index for _, c in cols], dtype=np.int32),
        'detectorIndex':   np.array([c.detector_index for _, c in cols], dtype=np.int32),
        'wavelengthIndex': np.array([c.wavelength_index for _, c in cols], dtype=np.int32),
        'dataType':        np.array([c.datatype for _, c in cols], dtype=np.int32),
        'dataTypeIndex':   np.array([c.datatype_index for _, c in cols], dtype=np.int32),
        'dataUnit':        [c.unit for _, c in cols],
        'dataTypeLabel':   [c.label for _, c in cols],
    }
    return flat_data, ml


def describe_measurement_columns(stream):
    """
    Describe the SNIRF columns this stream would be written as.

    Returns the measurement description without the data, for building an
    external per-measurement table that must line up with the file, such as a
    BIDS channels table.

    Parameters
    ----------
    stream : NirsStream
        Any stream :func:`write_snirf` accepts.

    Returns
    -------
    list of MeasurementColumn
        One entry per column, in write order.

    Raises
    ------
    ValueError
        If the stream cannot be represented in SNIRF.
    NotImplementedError
        If the modality has no SNIRF encoding.
    """
    return [meta for _, meta in _enumerate_columns(stream)]


def _write_probe_group(probe_grp, probe, stream):
    """Write the probe group, including optode positions, wavelengths and landmarks."""
    dims = set(stream.data.dims)

    # ── Required geometry ────────────────────────────────────────────────────
    probe_grp.create_dataset('wavelengths', data=probe.wavelengths.astype(np.float64))

    if probe.s_pos.shape[1] >= 3:
        probe_grp.create_dataset('sourcePos3D',   data=probe.s_pos.astype(np.float64))
        probe_grp.create_dataset('detectorPos3D', data=probe.d_pos.astype(np.float64))
    else:
        probe_grp.create_dataset('sourcePos2D',   data=probe.s_pos.astype(np.float64))
        probe_grp.create_dataset('detectorPos2D', data=probe.d_pos.astype(np.float64))

    # Optional 2D layout alongside 3D
    if probe.s_pos_2d is not None and probe.s_pos.shape[1] >= 3:
        probe_grp.create_dataset('sourcePos2D',   data=probe.s_pos_2d.astype(np.float64))
        probe_grp.create_dataset('detectorPos2D', data=probe.d_pos_2d.astype(np.float64))

    # ── Labels ────────────────────────────────────────────────────────────────
    if probe.source_labels:
        probe_grp.create_dataset('sourceLabels',   data=[_s(l) for l in probe.source_labels])
    if probe.detector_labels:
        probe_grp.create_dataset('detectorLabels', data=[_s(l) for l in probe.detector_labels])

    # ── Landmarks ─────────────────────────────────────────────────────────────
    if probe.landmarks is not None:
        probe_grp.create_dataset('landmarkPos3D', data=probe.landmarks.astype(np.float64))
        if probe.landmark_labels is not None:
            probe_grp.create_dataset('landmarkLabels',
                                     data=[_s(l) for l in probe.landmark_labels])

    # ── Modality-specific fields ──────────────────────────────────────────────
    data = stream.data

    if 'freq' in dims and np.iscomplexobj(data.values):
        mod_freqs = data.freq.values
        mod_freqs = mod_freqs[mod_freqs != 0]
        if len(mod_freqs):
            probe_grp.create_dataset('frequencies', data=mod_freqs.astype(np.float64))

    elif 'tau' in dims:
        probe_grp.create_dataset('correlationTimeDelays',
                                 data=data.tau.values.astype(np.float64))

    elif 'bin' in dims:
        if 'timeDelays' in data.coords:
            probe_grp.create_dataset('timeDelays',
                                     data=data.coords['timeDelays'].values.astype(np.float64))
        if 'timeDelayWidths' in data.coords:
            probe_grp.create_dataset('timeDelayWidths',
                                     data=data.coords['timeDelayWidths'].values.astype(np.float64))

    elif 'moment' in dims:
        MOM_ORD = {'m0': 0, 'm1': 1, 'm2': 2}
        orders = np.array([MOM_ORD.get(str(m), i)
                           for i, m in enumerate(data.moment.values)], dtype=np.float64)
        probe_grp.create_dataset('momentOrders', data=orders)


def _write_stim_groups(nirs_grp, events):
    """Write the stimulus groups from an event table."""
    if events is None or events._table.empty:
        return
    df = events._table.sort_values('onset').reset_index(drop=True)
    for i, label in enumerate(df['label'].unique()):
        grp = nirs_grp.create_group(f'stim{i + 1}')
        grp.create_dataset('name', data=_s(label))
        grp.create_dataset('data',
                           data=df[df['label'] == label][['onset', 'duration', 'value']]
                           .values.astype(np.float64))


def _write_aux_groups(nirs_grp, aux_streams, start_idx=1):
    """
    Write auxiliary streams as SNIRF auxiliary groups.

    Each signal trace becomes its own group, named by its signal label, as the
    specification allows one name per group.

    Parameters
    ----------
    nirs_grp : h5py.Group
        The parent group.
    aux_streams : list of AuxStream
        Streams to write.
    start_idx : int
        One-based index of the first group written.
    """
    j = start_idx
    for aux in aux_streams:
        data  = aux.data
        dim   = 'signal' if 'signal' in data.dims else 'channel'
        labels = list(data[dim].values) if dim in data.dims else [aux.name]
        time_vals = data.time.values.astype(np.float64)
        unit = data.attrs.get('units', data.attrs.get('unit', None))

        for sig_label in labels:
            trace = data.sel({dim: sig_label}).values if len(labels) > 1 else data.values
            if trace.ndim == 1:
                trace = trace[:, np.newaxis]

            grp = nirs_grp.create_group(f'aux{j}')
            grp.create_dataset('name',           data=_s(sig_label))
            grp.create_dataset('dataTimeSeries', data=trace.astype(np.float64))
            grp.create_dataset('time',           data=time_vals)
            if unit:
                grp.create_dataset('dataUnit', data=_s(unit))
            j += 1


def read_snirf_probe(filepath, *, sc_threshold):
    """
    Read only the probe geometry and channel configuration from a SNIRF file.

    Skips the time series, so it is much faster than a full read.

    Parameters
    ----------
    filepath : str
        Path to the .snirf file.
    sc_threshold : float or None
        Source-detector distance in mm below which a channel is classified as
        short. Pass None if the probe design has no short-separation
        channels.

    Returns
    -------
    Probe
        With optode positions, wavelengths, channel configuration, labels and,
        where present, landmarks.
    """
    with h5py.File(filepath, 'r') as f:
        metadata_path = '/nirs/metaDataTags'

        def get_tag(tag_name, default='unknown'):
            if metadata_path not in f:
                return default
            if tag_name in f[metadata_path]:
                val = f[f'{metadata_path}/{tag_name}'][()]
                return val.decode('utf-8').strip() if isinstance(val, bytes) else str(val).strip()
            return default

        length_unit = get_tag('LengthUnit', 'mm')

        # ── Geometry ──────────────────────────────────────────────────────────
        probe_path = '/nirs/probe'
        if 'sourcePos3D' in f[probe_path]:
            s_pos = f[f'{probe_path}/sourcePos3D'][()]
            d_pos = f[f'{probe_path}/detectorPos3D'][()]
        else:
            s_pos = f[f'{probe_path}/sourcePos2D'][()]
            d_pos = f[f'{probe_path}/detectorPos2D'][()]

        wavelengths = f[f'{probe_path}/wavelengths'][()].flatten()

        # ── Optional labels ───────────────────────────────────────────────────
        def _read_str_array(path):
            raw = f[path][()]
            return [v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in raw]

        source_labels   = _read_str_array(f'{probe_path}/sourceLabels')   if 'sourceLabels'   in f[probe_path] else None
        detector_labels = _read_str_array(f'{probe_path}/detectorLabels') if 'detectorLabels' in f[probe_path] else None

        # ── Optional landmarks ────────────────────────────────────────────────
        landmarks       = f[f'{probe_path}/landmarkPos3D'][()]             if 'landmarkPos3D'  in f[probe_path] else None
        if landmarks is not None and landmarks.shape[1] > 3:
            # SNIRF allows a 4th column (a 1-indexed selection/label index);
            # keep only the x,y,z spatial columns.
            landmarks = landmarks[:, :3]
        landmark_labels = _read_str_array(f'{probe_path}/landmarkLabels') if 'landmarkLabels' in f[probe_path] else None

        # ── Channel configuration from measurement list ────────────────────────
        ml = read_measurement_list(f, '/nirs/data1')

        sd_pairs_seen = []
        for sd in zip(ml['source'], ml['detector']):
            if sd not in sd_pairs_seen:
                sd_pairs_seen.append(sd)

        probe = Probe(
            s_pos, d_pos, wavelengths,
            source_labels=source_labels,
            detector_labels=detector_labels,
            landmark_pos=landmarks,
            landmark_labels=landmark_labels,
            lengthUnit=length_unit,
            sc_threshold=sc_threshold,
        )
        probe._channels = {
            'sources':   [sd[0] for sd in sd_pairs_seen],
            'detectors': [sd[1] for sd in sd_pairs_seen],
        }

    return probe


def write_snirf(stream, filepath, aux_streams=None, ml_format='indexed'):
    """
    Write a stream to a SNIRF file.

    Events are written as stimulus groups and any auxiliary streams as
    auxiliary groups. The format version is always 1.0; ``ml_format`` selects
    between two encodings of the measurement list within that version.

    Parameters
    ----------
    stream : NirsStream
        Stream to write. Must carry a probe.
    filepath : str or Path
        Destination path, created or overwritten.
    aux_streams : list of AuxStream, optional
        Auxiliary streams to embed in the same file.
    ml_format : {'indexed', 'array'}
        'indexed' (default) writes one subgroup per column and is the more
        widely supported. 'array' writes a single group of array datasets,
        which is more efficient for large channel counts but needs a reader
        that supports it.

    Raises
    ------
    ValueError
        If the stream cannot be represented in SNIRF.
    NotImplementedError
        If the modality has no SNIRF encoding.
    """
    import datetime

    supported_formats = ('indexed', 'array')
    if ml_format not in supported_formats:
        raise ValueError(f"Unsupported ml_format '{ml_format}'. Choose from {supported_formats}.")

    if stream.probe is None:
        raise ValueError("stream.probe is None — a valid Probe is required to write SNIRF.")

    flat_data, ml = _build_measurement_columns(stream)

    with h5py.File(filepath, 'w') as f:
        nirs = f.create_group('nirs')

        # Required by SNIRF spec -- always '1.0', the only value the spec
        # document defines (see docstring: ml_format is a within-version
        # structural choice, not a version bump).
        nirs.create_dataset('formatVersion', data=_s('1.0'))

        # ── metaDataTags ──────────────────────────────────────────────────────
        meta = nirs.create_group('metaDataTags')
        now  = datetime.datetime.now()
        length_unit = (getattr(stream.probe, 'lengthUnit', None)
                       or stream.data.attrs.get('lengthUnit')
                       or 'mm')
        meta.create_dataset('SubjectID',       data=_s(stream.name))
        meta.create_dataset('MeasurementDate', data=_s(now.strftime('%Y-%m-%d')))
        meta.create_dataset('MeasurementTime', data=_s(now.strftime('%H:%M:%S')))
        meta.create_dataset('LengthUnit',      data=_s(length_unit))
        meta.create_dataset('TimeUnit',        data=_s(stream.data.attrs.get('timeUnit', 's')))
        meta.create_dataset('FrequencyUnit',   data=_s('Hz'))
        # Custom vendor tag (spec explicitly allows additional metaDataTags)
        # -- see read_snirf()'s status_tag for why this is needed: SNIRF has
        # no native way to distinguish 'raw' from 'processed' FD/DCS data.
        meta.create_dataset('MILOB_Status',    data=_s(stream.status))

        # ── probe ─────────────────────────────────────────────────────────────
        _write_probe_group(nirs.create_group('probe'), stream.probe, stream)

        # ── data1 ─────────────────────────────────────────────────────────────
        data_grp = nirs.create_group('data1')
        data_grp.create_dataset('dataTimeSeries', data=flat_data,
                                chunks=True, compression='gzip', compression_opts=4)
        data_grp.create_dataset('time',
                                data=stream.data.time.values.astype(np.float64))

        # ── measurementList ───────────────────────────────────────────────────
        has_units  = any(u  is not None for u  in ml['dataUnit'])
        has_labels = any(lb is not None for lb in ml['dataTypeLabel'])

        if ml_format == 'array':
            # Compact array format — one group, arrays of length n_cols
            ml_grp = data_grp.create_group('measurementLists')
            ml_grp.create_dataset('sourceIndex',     data=ml['sourceIndex'])
            ml_grp.create_dataset('detectorIndex',   data=ml['detectorIndex'])
            ml_grp.create_dataset('wavelengthIndex', data=ml['wavelengthIndex'])
            ml_grp.create_dataset('dataType',        data=ml['dataType'])
            ml_grp.create_dataset('dataTypeIndex',   data=ml['dataTypeIndex'])
            if has_units:
                ml_grp.create_dataset('dataUnit',
                                      data=[_s(u or '') for u in ml['dataUnit']])
            if has_labels:
                ml_grp.create_dataset('dataTypeLabel',
                                      data=[_s(lb or '') for lb in ml['dataTypeLabel']])
        else:
            # Indexed format — one measurementList{k} subgroup per column.
            # Required by Homer3, FieldTrip, and other tools that don't
            # support the array encoding.
            n_cols = flat_data.shape[1]
            for k in range(n_cols):
                ml_grp = data_grp.create_group(f'measurementList{k + 1}')
                ml_grp.create_dataset('sourceIndex',     data=np.int32(ml['sourceIndex'][k]))
                ml_grp.create_dataset('detectorIndex',   data=np.int32(ml['detectorIndex'][k]))
                ml_grp.create_dataset('wavelengthIndex', data=np.int32(ml['wavelengthIndex'][k]))
                ml_grp.create_dataset('dataType',        data=np.int32(ml['dataType'][k]))
                ml_grp.create_dataset('dataTypeIndex',   data=np.int32(ml['dataTypeIndex'][k]))
                if has_units:
                    ml_grp.create_dataset('dataUnit',      data=_s(ml['dataUnit'][k] or ''))
                if has_labels:
                    ml_grp.create_dataset('dataTypeLabel', data=_s(ml['dataTypeLabel'][k] or ''))

        # ── stim ──────────────────────────────────────────────────────────────
        _write_stim_groups(nirs, stream.events)

        # ── aux ───────────────────────────────────────────────────────────────
        if aux_streams:
            _write_aux_groups(nirs, aux_streams, start_idx=1)
