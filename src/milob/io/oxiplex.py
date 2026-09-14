import numpy as np
import xarray as xr
from ..core.probe import Probe
from ..core.events import Events
from ..core.fd_nirs import FD_Stream


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

#Keeping for now, but use below function
# def _unwrap_phase_rad(phase_deg: np.ndarray) -> np.ndarray:
#     """Standardise degrees → radians and unwrap for temporal continuity."""
#     standardised = phase_deg % 360
#     return np.unwrap(np.deg2rad(standardised))

def _unwrap_phase_rad(phase_deg: np.ndarray) -> np.ndarray:
    """
    Convert phase from degrees to radians and unwrap it.

    Parameters
    ----------
    phase_deg : np.ndarray
        Phase in degrees.

    Returns
    -------
    np.ndarray
        Unwrapped phase in radians.
    """
    # Wrap to [-180, 180] instead of [0, 360] to keep small phase shifts near zero
    standardised = (phase_deg + 180) % 360 - 180
    return np.unwrap(np.deg2rad(standardised))


def _build_complex_fd(dc: np.ndarray,
                      ac: np.ndarray,
                      phase_deg: np.ndarray,
                      modulation_frequency: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the complex frequency-domain representation from raw measurements.

    Parameters
    ----------
    dc, ac, phase_deg : np.ndarray
        Shape (n_times, n_channels) each.
    modulation_frequency : float
        Instrument modulation frequency in Hz.

    Returns
    -------
    freq_values : np.ndarray
        Shape (2,): the DC and modulation frequencies.
    complex_data : np.ndarray
        Shape (n_times, n_channels, 2), complex. The DC entry is real; the
        modulation entry is the amplitude times the phase factor.
    """
    phase_rad = _unwrap_phase_rad(phase_deg)
    complex_ac = ac * np.exp(1j * phase_rad)          # AC * e^(i*phi)
    dc_complex = dc.astype(complex)                    # imaginary = 0

    complex_data = np.stack([dc_complex, complex_ac], axis=-1)  # (..., 2)
    freq_values = np.array([0.0, modulation_frequency])
    return freq_values, complex_data


# ---------------------------------------------------------------------------
# Public reader
# ---------------------------------------------------------------------------

def read_oxiplex(filepath: str,
                 detector: str,
                 wavelength_mask: np.ndarray,
                 *, sc_threshold) -> FD_Stream:
    """
    Load an ISS OxiplexTS text file into a frequency-domain stream.

    The instrument's manual marker column becomes rows in the stream's event
    table, with zero duration and the label 'button_press'.

    Parameters
    ----------
    filepath : str
        Path to the exported text file.
    detector : {'A', 'B'}
        Which detector array to read.
    wavelength_mask : np.ndarray
        Wavelength in nm for each raw channel column.
    sc_threshold : float or None
        Source-detector distance in mm below which a channel is classified as
        short. Pass None if the probe design has no short-separation
        channels.

    Returns
    -------
    FD_Stream
        Shape (time, channel, wavelength, freq), complex, with the DC and
        modulation frequencies on the last axis.
    """
    name = 'fd-nirs'
    wavelengths = np.unique(wavelength_mask)

    # -----------------------------------------------------------------------
    # Parse the text file
    # -----------------------------------------------------------------------
    with open(filepath, 'r') as f:
        lines = f.readlines()

    start_line = cal_line = dist_line = modfreq_line = None
    coef_found = dist_found = False

    for i, line in enumerate(lines):
        if line.startswith('Time Stamp'):
            start_line = i
        if line.startswith('Coef') and not coef_found:
            cal_line = i
            coef_found = True
        if line.startswith('Distance') and not dist_found:
            dist_line = i
            dist_found = True
        if line.startswith('Modulation Frequency:'):
            modfreq_line = i

    modulation_frequency = float(lines[modfreq_line].split()[2])  # Hz

    # Header row (shared by both detectors): 'Time Stamp', 'Elapsed Time',
    # AC/DC/PH x8 for A, AUX x4, 'Marker', then AC/DC/PH x8 for B.
    header_cols = [c for c in lines[start_line].rstrip('\n').split('\t') if c]
    marker_col = header_cols.index('Marker')

    # Raw data block (rows = time points)
    data_block = np.array(
        [line.split() for line in lines[start_line + 1:]],
        dtype=float
    )

    # -----------------------------------------------------------------------
    # Calibration coefficients
    # -----------------------------------------------------------------------
    det_a_coefs = np.array(
        [lines[cal_line + 1 + k].split()[1:] for k in range(3)], dtype=float
    )
    det_b_coefs = np.array(
        [lines[cal_line + 1 + k + 4].split()[1:] for k in range(3)], dtype=float
    )
    cal_factors = det_a_coefs if detector == 'A' else det_b_coefs

    # -----------------------------------------------------------------------
    # Distances and probe geometry
    # -----------------------------------------------------------------------
    dist_row_offset = 0 if detector == 'A' else 3
    raw_distances = np.array(
        lines[dist_line + dist_row_offset].split()[1:], dtype=float
    )

    # Average distances per wavelength (channels are interleaved by wavelength)
    split_distances = np.array([
        [raw_distances[i]
         for i in range(len(wavelength_mask))
         if wavelength_mask[i] == w]
        for w in wavelengths
    ])
    distances = np.mean(split_distances, axis=0)

    # 1-D probe: sources at different distances, single detector at origin
    source_pos = np.column_stack([distances, np.zeros(len(distances)), np.zeros(len(distances))])
    detector_pos = np.array([[0.0, 0.0, 0.0]])

    n_channels = len(distances)
    source_list = list(range(1, n_channels + 1))
    detector_list = [1] * n_channels
    channel_labels = [f"S{s}D1" for s in source_list]

    channels_config = {
        'sources': source_list,
        'detectors': detector_list,
        'wavelengths': list(wavelengths),
    }
    probe = Probe(source_pos, detector_pos, wavelengths,
                  lengthUnit='cm', channels=channels_config,
                  sc_threshold=sc_threshold)

    # -----------------------------------------------------------------------
    # Extract and calibrate raw measurements
    # -----------------------------------------------------------------------
    times = data_block[:, 1]

    # One event per row where the 'Marker' column is nonzero (see the Notes
    # above). Shared by both detectors, so this doesn't depend on `detector`.
    events = Events()
    for t, m in zip(times, data_block[:, marker_col]):
        if m != 0:
            events.add_event(onset=t, duration=0.0, value=m, label='button_press')

    col_slice = slice(2, 26) if detector == 'A' else slice(31, None)
    raw = data_block[:, col_slice]

    # Layout: (n_time, n_raw_channels=8, n_datatypes=3) in Fortran order
    reshaped = raw.reshape(-1, 8, 3, order='F')

    ac_raw    = reshaped[:, :, 0]   # AC amplitude
    dc_raw    = reshaped[:, :, 1]   # DC intensity
    phase_raw = reshaped[:, :, 2]   # Phase (degrees)

    cal_dc    = dc_raw    * cal_factors[1, :]
    cal_ac    = ac_raw    * cal_factors[0, :]
    cal_phase = phase_raw + cal_factors[2, :]
    
    

    # -----------------------------------------------------------------------
    # Reshape by wavelength: (n_time, n_channels, n_wavelengths)
    # -----------------------------------------------------------------------
    slices_dc    = [cal_dc   [:, wavelength_mask == w] for w in wavelengths]
    slices_ac    = [cal_ac   [:, wavelength_mask == w] for w in wavelengths]
    slices_phase = [cal_phase[:, wavelength_mask == w] for w in wavelengths]

    # Stack → (n_time, n_channels, n_wavelengths)
    dc_wl    = np.stack(slices_dc,    axis=2)
    ac_wl    = np.stack(slices_ac,    axis=2)
    phase_wl = np.stack(slices_phase, axis=2)

    # -----------------------------------------------------------------------
    # Build complex freq representation
    # -----------------------------------------------------------------------
    

    #operate on per-wavelength arrays then stack
    complex_stack = []
    for wi, w in enumerate(wavelengths):
        dc_i    = dc_wl[:, :, wi].astype(complex)
        ac_i    = ac_wl[:, :, wi]
        phi_i   = _unwrap_phase_rad(phase_wl[:, :, wi])
        cac_i   = ac_i * np.exp(1j * phi_i)
        complex_stack.append(np.stack([dc_i, cac_i], axis=-1))   # (t, ch, 2)

    # (n_time, n_channels, n_wavelengths, 2)  complex
    data_complex = np.stack(complex_stack, axis=2)

    freq_values = np.array([0.0, modulation_frequency])

    # -----------------------------------------------------------------------
    # Build xarray DataArray
    # -----------------------------------------------------------------------
    fs = round(1.0 / np.diff(times).mean(), 1) if len(times) > 1 else None

    coords = {
        'time':        times,
        'channel':     channel_labels,
        'wavelength':  wavelengths,
        'freq':        freq_values,
        # per-channel helper coords
        'channel_idx': ('channel', np.arange(n_channels)),
        'source':      ('channel', source_list),
        'detector':    ('channel', detector_list),
        'distance':    ('channel', distances),
    }

    data_xr = xr.DataArray(
        data_complex,
        coords=coords,
        dims=['time', 'channel', 'wavelength', 'freq'],
        attrs={
            'modality':      'fd-nirs',
            'sampling_rate': fs,
            'timeUnit':      's',
            'lengthUnit':    'cm',
            'frequencyUnit': 'Hz',
            'freq_encoding': 'freq=0 → DC (real);  freq=f_mod → AC*exp(i*phase_rad)',
        }
    )

    return FD_Stream(data_xr, probe, name=name, events=events)
