# processing/physics.py
import numpy as np
import xarray as xr
from typing import Union, Optional
import pandas as pd
import os


def intensity_to_od(data: Union[np.ndarray, xr.DataArray], 
                    baseline: Optional[np.ndarray] = None,
                    baseline_window: Optional[tuple] = None) -> Union[np.ndarray, xr.DataArray]:
    """
    Convert intensity to optical density.

    Computes OD = -log(I / I0). NaNs are ignored when forming the baseline,
    and infinities arising from zero intensity become NaN.

    Parameters
    ----------
    data : np.ndarray or xr.DataArray
        Raw intensity, with time on the first axis.
    baseline : np.ndarray, optional
        Pre-computed baseline intensity per channel. Computed from the data if
        omitted.
    baseline_window : tuple of (int, int), optional
        Sample indices over which to compute the baseline. Defaults to the
        whole time series.

    Returns
    -------
    np.ndarray or xr.DataArray
        Optical density, of the same type as the input.

    Examples
    --------
    >>> od = intensity_to_od(intensity)
    >>> od = intensity_to_od(intensity, baseline_window=(0, 100))
    """
    # Determine input type and extract data
    is_xarray = isinstance(data, xr.DataArray)

    if is_xarray:
        # Work with underlying numpy array
        values = data.values
        coords = data.coords
        dims = data.dims
        attrs = data.attrs.copy()
    else:
        values = data
    
    # =============================================
    # Shape Management (Universal 4D: T, C, W, D)
    # =============================================
    # Added to handle different data types before sub-class separation; 
    # consider removing this part since it's on cw_nirs now
    has_dummy_dim = False
    if values.ndim == 3:
        values = values[..., np.newaxis]
        has_dummy_dim = True
    # values shape is now (T, C, W, D)
    
    
    # =============================================
    # Optical Density Computation
    # =============================================
    import warnings
    
    # Use np.errstate to keep the console clean
    with np.errstate(divide='ignore', invalid='ignore'), \
        warnings.catch_warnings():
            
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        
        # Compute baseline if not provided
        if baseline is None:
            if baseline_window is not None:
                start, end = baseline_window
                # Mean across Time (axis 0). Result shape: (C, W, D)
                baseline = np.nanmean(np.abs(values[start:end, ...]), axis=0)
            else:
                baseline = np.nanmean(np.abs(values), axis=0)

        # Ensure baseline broadcasts correctly
        # baseline is (C, W, D). Add Time dim to make it (1, C, W, D)
        if baseline.ndim == 3:
            baseline_ready = baseline[np.newaxis, ...]
        elif baseline.ndim == 2: 
            # Case for legacy 3D input where user provided a 2D baseline
            baseline_ready = baseline[np.newaxis, ..., np.newaxis]
        else:
            baseline_ready = baseline

        # Convert to OD
        epsilon = 1e-10 # small ofset to avoid taking log(0) - especially late TD bins
        od_values = -np.log((np.abs(values) + epsilon) / baseline_ready + epsilon)
        
    # Clean up numerical artifacts
    # Replace inf with NaN (occurs when intensity is 0)
    od_values[np.isinf(od_values)] = np.nan    
    
    
    # =============================================
    # Return appropriate type
    # =============================================
    # Add this to enforce that final xarray is 3D for CW types
    # can be removed with the Universal 4D above later on...
    if has_dummy_dim:
        od_values = od_values.squeeze(axis=-1)
    
    # return correct format if xarray was provided as input
    if is_xarray:
        od = xr.DataArray(
            od_values,
            coords=coords,
            dims=dims,
            attrs=attrs
        )
        od.attrs['units'] = 'OD'
        od.attrs['transformation'] = 'intensity_to_od'
        return od
    
    return od_values



def od_to_concentration(od: Union[np.ndarray, xr.DataArray],
                       dpf: np.ndarray,
                       wavelengths: Optional[np.ndarray] = None,
                       distances: Optional[np.ndarray] = None) -> Union[np.ndarray, xr.DataArray]:
    """
    Convert optical density to haemoglobin concentration changes.

    Applies the modified Beer-Lambert law, dC = (eps * L * DPF)^-1 * dOD.

    Parameters
    ----------
    od : np.ndarray or xr.DataArray
        Optical density changes.
    dpf : float or np.ndarray
        Differential pathlength factor. A scalar applies to every wavelength;
        an array gives one value per wavelength. Default 6.0.
    wavelengths : np.ndarray
        Wavelength of each channel in nm.
    distances : np.ndarray
        Source-detector distance of each channel in mm.

    Returns
    -------
    np.ndarray or xr.DataArray
        Concentration changes, with a trailing axis of [HbO, HbR].

    Examples
    --------
    >>> conc = od_to_concentration(od, wavelengths=wl, distances=d)
    """
    is_xarray = isinstance(od, xr.DataArray)
    
    if is_xarray:
        values = od.values
        lambdas = wavelengths if wavelengths is not None else od.coords['wavelength'].values
        distances = distances if distances is not None else od.coords['distance'].values        
        if od.attrs.get('lengthUnit') == 'mm': # transform to cm
            distances = distances / 10.0
        # Capture dimension names and coordinates for reconstruction
        original_dims = list(od.dims) 
        original_coords = dict(od.coords)
        attrs = od.attrs.copy()
    else:
        values = od
        lambdas = wavelengths
        if lambdas is None:
            raise ValueError("Wavelengths must be provided if input is a numpy array.")
        if distances is None:
            raise ValueError("Source-detector distances must be provided if input is a numpy array.")
    
    # =============================================
    # Shape Management
    # =============================================
    # We standardise to 4D: (Time, Channel, Wavelength, Datatype)
    # If input is 3D (T, C, W), we add a dummy 4th dim to make logic universal
    has_dummy_dim = False
    if values.ndim == 3:
        values = values[..., np.newaxis]
        has_dummy_dim = True
    # values shape is now (T, C, W, D)
    T, C, W, D = values.shape

    # A scalar dpf means "use this value for every wavelength" -- broadcast
    # it to shape (W,) so the pathlength normalization below can always
    # assume a per-wavelength array.
    dpf = np.asarray(dpf, dtype=float)
    if dpf.ndim == 0:
        dpf = np.full(W, dpf.item())

    # =============================================
    # Concenctration Computation (Modified Beer-Lambert)
    # =============================================

    # Physics setup
    # e = _get_extinction_coefficients(lambdas)   # in /cm units
    constants = get_extinction_coefficients_Prahl(lambdas) # Use new function with larger table
    e = constants['E_matrix'] # in /cm units
    einv = np.linalg.pinv(e)    # Shape (2, nWav)
    
    # Pathlength normalization: OD / (dist * dpf)
    # dist: (C,), dpf: (W,)
    # Broadcast to (1, C, W, 1) to match (T, C, W, D)
    pathlength = distances[None, :, None, None] * dpf[None, None, :, None]
    norm_od = values / pathlength  # (T, C, W, D)
    
    # Matrix multiplication
    # We want to multiply across the Wavelength dimension (axis 2)
    # Swap axes to put Wavelength at the end for matmul: (T, C, D, W) @ (W, 2)
    #
    # einv is (2, nWav) -- transpose to (nWav, 2) so the contraction lands
    # on the wavelength axis (norm_od_swapped's last dim, nWav) against
    # einv's wavelength axis, producing chromophore (2) as the output's
    # last dim, matching the "(T,C,D,W) @ (W,2)" this comment always
    # described. Without the transpose this only avoided a shape error by
    # coincidence when nWav==2 (a (D,2)@(2,2) matmul is always shape-valid
    # regardless of which axis is "really" being contracted) -- and even
    # then it silently returned wrong values: it contracted OD's
    # wavelength axis against einv's CHROMOPHORE axis instead of its
    # wavelength axis, equivalent to using einv.T's transpose of itself,
    # correct only if einv happened to be symmetric (it isn't, since HbO
    # and HbR extinction coefficients differ across wavelengths). Verified
    # against a synthetic round-trip (forward Beer-Lambert then invert):
    # this transposed form recovers the true [HbO, HbR] exactly; the
    # untransposed form was off by roughly 2x with HbO/HbR partially
    # conflated, for standard 760/850nm coefficients.
    norm_od_swapped = np.swapaxes(norm_od, 2, 3)
    conc_values = np.matmul(norm_od_swapped, einv.T) * 1e6  # (T, C, D, 2)
    
    # Calculate HbT and concatenate
    # Result: (T, C, D, 3) where last dim is [HbO, HbR, HbT]
    hbt = np.sum(conc_values, axis=-1, keepdims=True)
    final_values = np.concatenate([conc_values, hbt], axis=-1)
    
    # Swap back so chromophore replaces wavelength: (T, C, 3, D)
    final_values = np.swapaxes(final_values, 2, 3)
    # ----------------

    # Replacement --------
    # # Use einsum for the matrix multiplication:
    # # i=Time, j=Channel, k=Wavelength, l=Datatype, m=Chromophore
    # # We contract over 'k' (Wavelength) using einv (m, k)
    # conc_values = np.einsum('ijkl,mk->ijml', norm_od, einv) * 1e6 # Result: (T, C, m, D)

    # # Calculate HbT (Total Hemoglobin)
    # # conc_values is (T, C, 2, D). Summing over axis 2 (m)
    # hbt = np.sum(conc_values, axis=2, keepdims=True) # Result: (T, C, 1, D)
    
    # # Concatenate HbO, HbR, and HbT along the chromophore axis (axis 2)
    # final_values = np.concatenate([conc_values, hbt], axis=2) # Result: (T, C, 3, D)
    #--------------------
    
    # If it was originally 3D, remove the dummy 4th dimension
    if has_dummy_dim:
        final_values = final_values.squeeze(-1)
    
    # =============================================
    # Return appropriate type
    # =============================================
    if is_xarray:
        # Update dimensions: 'wavelength' is gone, 'chromophore' is new
        new_dims = [d if d != 'wavelength' else 'chromophore' for d in original_dims]
        
        # Clean up old coords that belonged to the wavelength dimension
        new_coords = {k: v for k, v in original_coords.items() if k != 'wavelength'}
        new_coords['chromophore'] = ['HbO', 'HbR', 'HbT']

        conc_final = xr.DataArray(
            final_values,
            coords=new_coords,
            dims=new_dims,
            attrs=attrs
        )
        conc_final.attrs['units'] = 'uM'
        return conc_final    
    else:
        return final_values




def _get_extinction_coefficients(wavelengths: np.ndarray) -> np.ndarray:
    """
    Return tabulated HbO and HbR extinction coefficients.

    Returns
    -------
    np.ndarray
        Shape (2, n_wavelengths), in cm^-1 per mole per litre, with HbO on
        row 0 and HbR on row 1.
    """
    # Standard extinction coefficients (Wavelength, HbO, HbR)
    table = np.array([
        [670, 427, 3013], [680, 423, 2610], [690, 415, 2141],
        [700, 419, 1827], [750, 600, 1532], [760, 645, 1669],
        [780, 754, 1107], [800, 863, 838],  [810, 914, 798],
        [830, 1008, 778], [850, 1097, 781], [900, 1241, 883]
    ])
    ext_hbo = np.interp(wavelengths, table[:,0], table[:,1])
    ext_hbr = np.interp(wavelengths, table[:,0], table[:,2])
    
    # Convert base 10 to base e
    return np.vstack([ext_hbo, ext_hbr]).T * 2.303


def get_extinction_coefficients_Prahl(wavelengths, include_water=False):
    """
    Interpolate Prahl's haemoglobin extinction coefficients.

    Parameters
    ----------
    wavelengths : np.ndarray
        Wavelengths in nm.
    include_water : bool
        Also interpolate the water absorption coefficient, for modalities that
        use it. Default False; the water column is sampled at fewer
        wavelengths than the haemoglobin data.

    Returns
    -------
    np.ndarray
        Shape (2, n_wavelengths) in cm^-1 per mole per litre, with HbO on
        row 0 and HbR on row 1. A third row carries water absorption in
        cm^-1 when ``include_water`` is True.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, "master_optical_coefficients.csv")

    df_constants = pd.read_csv(filepath)
    df_constants.columns = df_constants.columns.str.strip()

    # force lambda column to numeric - turns units row 'nm' to NaN
    df_constants['lambda'] = pd.to_numeric(df_constants['lambda'], errors='coerce')
    df_constants = df_constants.dropna(subset=['lambda'])

    for col in ['hbo', 'hbr']:
        df_constants[col] = pd.to_numeric(df_constants[col], errors='coerce')

    # Interpolate HbO and HbR (base 10 values) and convert to base e
    ext_hbo = np.interp(wavelengths, df_constants['lambda'], df_constants['hbo']) * 2.303
    ext_hbr = np.interp(wavelengths, df_constants['lambda'], df_constants['hbr']) * 2.303

    result = {'E_matrix': np.vstack([ext_hbo, ext_hbr]).T}  # shape (n_wl, 2)

    if include_water:
        if 'water_mua' not in df_constants.columns:
            raise ValueError("'water_mua' column not found in master_optical_coefficients.csv.")
        df_constants['water_mua'] = pd.to_numeric(df_constants['water_mua'], errors='coerce')
        df_constants['water_mua'] = df_constants['water_mua'].interpolate()
        result['water_mua'] = np.interp(
            wavelengths, df_constants['lambda'], df_constants['water_mua']
        )

    return result