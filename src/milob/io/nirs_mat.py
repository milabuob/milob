# io/nirs_mat.py

import numpy as np
import re
import scipy.io
import h5py
import xarray as xr
from .snirf import Probe, _reshape_measurement_list # Reuse 4D logic


def _parse_nirs_mat(filepath, *, sc_threshold, coord_file=None, length_unit="cm"):
    """
    Parse a .nirs MATLAB file into its raw components.

    No stream objects are built here.

    Parameters
    ----------
    filepath : str
        Path to the .nirs file.
    sc_threshold : float or None
        Source-detector distance in mm below which a channel is classified as
        short. Pass None if the probe design has no short-separation
        channels.
    coord_file : str, optional
        AtlasViewer .txt file supplying 3-D probe geometry.
    length_unit : {'cm', 'mm'}
        Unit of the probe coordinates.

    Returns
    -------
    dict
        Keys ``cw_xr`` (the data, with dims time, channel and wavelength),
        ``probe``, ``aux_xr`` or None, ``stim_matrix`` (the raw stimulus
        matrix, unparsed) or None, and ``times`` in seconds.
    """
    aux_raw = None

    try:
        # --- Attempt Legacy Load (SciPy) ---
        mat = scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)

        raw_data = mat['d']
        times = mat['t'].flatten()
        sd_struct = mat['SD']

        wavelengths = np.atleast_1d(sd_struct.Lambda)
        ml_mat = np.atleast_2d(sd_struct.MeasList)
        
        # Positions
        s_pos_2d = getattr(sd_struct, 'SrcPos', None)
        d_pos_2d = getattr(sd_struct, 'DetPos', None)
        s_pos_3d = getattr(sd_struct, 'SrcPos3D', s_pos_2d) # Fallback
        d_pos_3d = getattr(sd_struct, 'DetPos3D', d_pos_2d) # Fallback

        s_matrix = mat.get('s', None)
        if s_matrix is not None:
            s_matrix = np.atleast_2d(s_matrix)

        aux_raw = mat.get('aux', None)
        if aux_raw is not None:
            aux_raw = np.atleast_2d(aux_raw)
            if aux_raw.shape[0] == 1:
                aux_raw = aux_raw.T  # ensure (n_time, n_signals)

        # Device Labels (Brainsight)
        s_labels = sd_struct.SrcNameBrainsight.tolist() if hasattr(sd_struct, 'SrcNameBrainsight') else None
        d_labels = sd_struct.DetNameBrainsight.tolist() if hasattr(sd_struct, 'DetNameBrainsight') else None

    except (NotImplementedError, KeyError):
        # --- Fallback to HDF5 Load (h5py for v7.3) --- 
        with h5py.File(filepath, 'r') as f:
            raw_data = f['d'][()].T
            times = f['t'][()].flatten()
            sd_grp = f['SD']

            def get_val(item):
                val = item[()]
                return f[val][()] if isinstance(val, h5py.Reference) else val

            wavelengths = np.atleast_1d(get_val(sd_grp['Lambda']).flatten())
            ml_mat = np.atleast_2d(get_val(sd_grp['MeasList']).T)
            
            # Positions
            s_pos_2d = get_val(sd_grp['SrcPos']).T if 'SrcPos' in sd_grp else None
            d_pos_2d = get_val(sd_grp['DetPos']).T if 'DetPos' in sd_grp else None
            
            s_key_3d = 'SrcPos3D' if 'SrcPos3D' in sd_grp else 'SrcPos'
            d_key_3d = 'DetPos3D' if 'DetPos3D' in sd_grp else 'DetPos'
            s_pos_3d = get_val(sd_grp[s_key_3d]).T
            d_pos_3d = get_val(sd_grp[d_key_3d]).T
            
            s_matrix = np.atleast_2d(f['s'][()].T) if 's' in f else None

            aux_raw = None
            if 'aux' in f:
                aux_raw = np.atleast_2d(f['aux'][()].T)
                if aux_raw.shape[0] == 1:
                    aux_raw = aux_raw.T

            # Device Labels (Brainsight)
            s_labels = [get_val(ref) for ref in sd_grp['SrcNameBrainsight']] if 'SrcNameBrainsight' in sd_grp else None
            d_labels = [get_val(ref) for ref in sd_grp['DetNameBrainsight']] if 'DetNameBrainsight' in sd_grp else None
            
            
    # --- Geometry Padding ---
    def ensure_3d(pos):
        if pos is not None and pos.shape[1] == 2:
            return np.column_stack([pos, np.zeros(pos.shape[0])])
        return pos

    # Ensure 3D is actually 3D (Z=0 if falling back from 2D)
    s_pos_3d = ensure_3d(s_pos_3d)
    d_pos_3d = ensure_3d(d_pos_3d)
    
    # --- Common Processing ---
    
    # 1. Measurement List Mapping
    ml = {
        'source': ml_mat[:, 0].astype(int),
        'detector': ml_mat[:, 1].astype(int),
        'dataType': ml_mat[:, 2].astype(int),
        'wavelength_idx': ml_mat[:, 3].astype(int) - 1,
    }

    # --- Optional Geometry Overwrite (AtlasViewer) ---
    landmarks, l_labels = None, None
    if coord_file:
        s_pos_3d, d_pos_3d, landmarks, l_labels = _parse_atlas_viewer_txt(coord_file)

    # --- Reshape to (time, channel, wavelength) ---
    data_reshaped, metadata = _reshape_measurement_list(raw_data, ml, wavelengths)

    # --- Build CW xarray ---
    t0 = times[0]
    fs = round(1 / np.mean(np.diff(times)), 1) if len(times) > 1 else None
    coords = {
        'time': times - t0,
        'channel': metadata['channel_labels'],
        'wavelength': metadata['wavelengths'],
        'source': ('channel', metadata['sources']),
        'detector': ('channel', metadata['detectors']),
    }
    
    # If we have custom labels, add them to the xarray
    if s_labels and d_labels:
        coords["source_label"] = ("channel", [s_labels[i-1] for i in metadata['sources']])
        coords["detector_label"] = ("channel", [d_labels[i-1] for i in metadata['detectors']])
        
    if 'datatype' in metadata['dims']:
        coords["datatype"] = metadata['datatypes']
        coords["datatype_name"] = ("datatype", metadata['datatype_names'])
                
    data_xr = xr.DataArray(
        data_reshaped,
        coords=coords,
        dims=metadata['dims'],
        attrs={'modality': 'cw_nirs', 'lengthUnit': length_unit,
               'timeUnit': 's', 'sampling_rate': fs},
    )

    # --- Build Probe ---
    from ..core.probe import Probe

    channels_config = {
        'sources': metadata['sources'],
        'detectors': metadata['detectors'],
        'wavelengths': [wavelengths[i] for i in ml['wavelength_idx']],
        'datatypes': ml['dataType'],
    }
    probe = Probe(
        s_pos_3d, d_pos_3d, wavelengths, channels=channels_config,
        s_pos_2d=s_pos_2d, d_pos_2d=d_pos_2d,
        source_labels=s_labels, detector_labels=d_labels,
        landmark_pos=landmarks, landmark_labels=l_labels,
        lengthUnit=length_unit, sc_threshold=sc_threshold,
    )

    # --- Build Aux xarray (if present) ---
    aux_xr = None
    if aux_raw is not None:
        n_signals = aux_raw.shape[1]
        aux_xr = xr.DataArray(
            aux_raw,
            coords={
                'time': times - t0,
                'signal': [f'aux_{i}' for i in range(n_signals)],
            },
            dims=['time', 'signal'],
            attrs={'timeUnit': 's', 'sampling_rate': data_xr.attrs.get('sampling_rate')},
        )

    return {
        'cw_xr': data_xr,
        'aux_xr': aux_xr,
        'probe': probe,
        'stim_matrix': s_matrix,
        'times': times - t0,
    }


def _parse_atlas_viewer_txt(filepath):
    """
    Parse an AtlasViewer .txt file for 3-D optode coordinates.

    Parameters
    ----------
    filepath : str
        Path to the file.

    Returns
    -------
    tuple
        Source positions, detector positions, landmark positions and landmark
        labels.
    """
    s_dict = {}
    d_dict = {}
    landmarks = []
    l_labels = []

    pattern = re.compile(r'([a-zA-Z0-9]+):\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)')

    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                label = match.group(1).lower()
                coords = np.array([float(match.group(2)), float(match.group(3)), float(match.group(4))])

                if label.startswith('s'):
                    idx = int(label[1:])
                    s_dict[idx] = coords
                elif label.startswith('d'):
                    idx = int(label[1:])
                    d_dict[idx] = coords
                else:
                    landmarks.append(coords)
                    l_labels.append(label.upper())

    s_pos = np.array([s_dict[i] for i in sorted(s_dict.keys())])
    d_pos = np.array([d_dict[i] for i in sorted(d_dict.keys())])

    return s_pos, d_pos, np.array(landmarks), l_labels
