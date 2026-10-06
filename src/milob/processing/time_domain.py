import numpy as np
import xarray as xr
import copy
import re
from ..core.units import cm_per_unit

# ------------------------------------------------------------------
# Modality conversion
# ------------------------------------------------------------------

def convert_to_cw(datastream):
    """
    Convert a time-domain stream to its continuous-wave equivalent.

    Gated data is summed across bins; moment data selects m0.

    Parameters
    ----------
    datastream : TD_Stream
        Stream in gated or moment form.

    Returns
    -------
    xr.DataArray
        Total intensity, with the fourth dimension standardised to 'dataType'.
    """

    all_dims = list(datastream.data.dims)
    dim4_name = all_dims[3] 
    
    print(f"Integrating {dim4_name} dimension into CW...")

    # If it's moments, m0 selcted as CW equivalent
    if dim4_name == 'moment':
        if 'm0' in datastream.data.coords.get('moment', []):
            cw_values = datastream.data.sel(moment='m0')
        elif 'M0' in datastream.data.coords.get('moment', []):
            cw_values = datastream.data.sel(moment='M0')
        elif 3 in datastream.data.coords.get('moment', []):
            cw_values = datastream.data.sel(moment=3)
        else:
            print("Error: M0 moment not found. Cannot convert moments to CW. Aborting...")
            return datastream

    # If it's bins, we sum them all to get total intensity
    elif dim4_name == 'bin':
        cw_values = datastream.data.sum(dim=dim4_name, skipna=True, min_count=1)

    # Clean up the coordinates
    cw_values = cw_values.drop_vars([dim4_name], errors='ignore')
    if 'timeDelays' in cw_values.coords:
        cw_values = cw_values.drop_vars(['timeDelays', 'timeDelayWidths'])

    # Prepare metadata dictionary to pass to new constructor
    from ..core.datastream import Datastream
    metadata = {
        'name': f"cw_stream",
        'events': copy.copy(datastream.events) if datastream.events is not None else None,
        'status': f"raw",
        'history': datastream.history + [Datastream._history_entry('to_cw', {'source_dim': dim4_name})]
    }
    
    return cw_values, metadata


# ------------------------------------------------------------------
# Moment computation
# ------------------------------------------------------------------

def calculate_moments(datastream, sheppards_corr=True):
    """
    Compute the moments m0, m1 and m2 from gated TPSF data.

    Parameters
    ----------
    datastream : TD_Stream
        Gated stream with dims (time, channel, wavelength, bin).
    sheppards_corr : bool
        Apply Sheppard's correction for the finite gate width.

    Returns
    -------
    xr.DataArray
        Shape (time, channel, wavelength, moment) with moment = [m0, m1, m2],
        holding total intensity, mean arrival time and variance.
    """
    # Select data xarray from datastream
    data_xr = datastream.data

    # Ensure datastream is TD gated
    if 'bin' not in data_xr.dims:
        raise ValueError(f"calculate_moments requires data with a 'bin' dimension. Dimensions found: {data_xr.dims}")
    
    if 'timeDelays' in data_xr.coords:
        t_bin = data_xr.coords['timeDelays'] # center of each bin (units: s)
    else:
        print("Warning: timeDelays not found. Using bin indices.")
        t_bin = data_xr.coords['bin']
    
    # 0th Moment (M0): Total Intensity - shape: (nT, nCh, nWl)
    m0 = data_xr.sum(dim='bin', skipna=True, min_count=1)
    
    # 1st Moment (M1): Mean Arrival Time - sum(t * I) / M0
    m1 = (data_xr * t_bin).sum(dim='bin') / (m0 + 1e-20)
    
    # 2nd centralised Moment (M2): Variance - sum((t - M1)^2 * I) / M0
    m2 = ((t_bin - m1)**2 * data_xr).sum(dim='bin') / (m0 + 1e-20)

    if sheppards_corr:
        if 'timeDelayWidths' in data_xr.coords:
            t_bin_width = data_xr.coords["timeDelayWidths"][0]
        else:
            raise ValueError("timeDelayWidths now found. Need for Sheppard's correction application.")

        m2 = m2 - (t_bin_width)**2 / 12
    
    # Package into a single DataArray ---
    moments_xr = xr.concat([m0, m1, m2], dim='moment')
    moments_xr.coords['moment'] = ['m0', 'm1', 'm2']

    # Transpose so 'moment' is the 4th dim
    moments_xr = moments_xr.transpose('time', 'channel', 'wavelength', 'moment')

    # Prepare metadata for datastream
    from ..core.datastream import Datastream
    metadata = {
        'name': copy.copy(datastream.name) if datastream.name is not None else None,
        'events': copy.copy(datastream.events) if datastream.events is not None else None,
        'status': copy.deepcopy(datastream.status) if datastream.status is not None else None,
        'history': datastream.history + [Datastream._history_entry('to_moments', {})]
    }
    
    return moments_xr, metadata


# ------------------------------------------------------------------
# Moment IRF Correction
# ------------------------------------------------------------------
def apply_irf_correction(datastream_mom, datastream_irf):
    """
    Correct measured moments for the instrument response function.

    Parameters
    ----------
    datastream_mom : TD_Stream
        Stream with ``status='moment'``.
    datastream_irf : TD_Stream
        Stream holding the IRF moments, either time-resolved with dims
        (time, channel, wavelength, moment) or static with dims
        (channel, wavelength, moment).

    Returns
    -------
    corrected_xr : xr.DataArray
        IRF-corrected moments.
    metadata : dict
        Details of the correction applied.
    """

    mom_xr = datastream_mom.data

    irf_xr = datastream_irf.data if hasattr(datastream_irf, "data") else datastream_irf

    # --- sanity checks ---
    required_dims = {"moment", "channel", "wavelength"}
    if not required_dims.issubset(mom_xr.dims):
        raise ValueError("Moment data must contain 'moment', 'channel', 'wavelength'")
    if not required_dims.issubset(irf_xr.dims):
        raise ValueError("IRF data must contain 'moment', 'channel', 'wavelength'")
    

    # --- extract moments ---
    m0 = mom_xr.sel(moment="m0")
    m1 = mom_xr.sel(moment="m1")
    m2 = mom_xr.sel(moment="m2")

    m0_irf = irf_xr.sel(moment="m0")
    m1_irf = irf_xr.sel(moment="m1")
    m2_irf = irf_xr.sel(moment="m2")

    # --- handle time broadcasting ---
    if "time" not in m1_irf.dims:
        # static IRF -> expand to match moment data array time
        m0_irf = m0_irf.expand_dims(time=mom_xr.coords["time"])
        m1_irf = m1_irf.expand_dims(time=mom_xr.coords["time"])
        m2_irf = m2_irf.expand_dims(time=mom_xr.coords["time"])

    # --- apply correction ---
    # m0_corr = m0 / m0_irf
    m0_corr = m0
    m1_corr = m1 - m1_irf
    m2_corr = m2 - m2_irf

    # ensure variance is non-negative
    m2_corr = xr.where(m2_corr < 0, 0, m2_corr)

    # --- reconstruct data array ---
    corrected_xr = xr.concat([m0_corr, m1_corr, m2_corr], dim="moment")
    corrected_xr.coords['moment'] = ['m0', 'm1', 'm2']

    # Prepare metadata for datastream
    metadata = {
        'name': copy.copy(datastream_mom.name) if datastream_mom.name is not None else None,
        'events': copy.copy(datastream_mom.events) if datastream_mom.events is not None else None,
        'status': copy.deepcopy(datastream_mom.status) if datastream_mom.status is not None else None,
        'history': datastream_mom.history + ["Applied IRF correction to moments"]
    }

    return corrected_xr, metadata


# ------------------------------------------------------------------
# Optical property inversion
# ------------------------------------------------------------------

# --- MOMENT METHOD ---
def calculate_optical_properties_moments(datastream_mom, n=1.37):
    """
    Derive optical properties from TD moments in closed form.

    Assumes a semi-infinite homogeneous medium in the diffusion regime, and
    IRF-corrected windowed moments.

    Parameters
    ----------
    datastream_mom : TD_Stream
        Stream with ``status='moment'``, carrying at least m1 and m2.
    n : float
        Refractive index of the medium. Default 1.37.

    Returns
    -------
    optical_xr : xr.DataArray
        Shape (time, channel, wavelength, optical_property) with
        optical_property = ['mua', 'musp'], in cm^-1.
    metadata : dict
        Details of the calculation.

    References
    ----------
    Liebert, A. et al. (2003). Applied Optics, 42(28), 5785-5792.
    """

    if datastream_mom.status != "moment":
        raise ValueError(
            "moments_to_optical_params() requires a TD stream with "
            "status='moment' -- call to_moments() first."
        )

    data = datastream_mom.data

    # --- Extract moments ---
    m1 = data.sel(moment="m1")   # mean time of flight [s]
    m2 = data.sel(moment="m2")   # variance [s^2]

    # --- Geometry ---
    # distance is only in mm when explicitly tagged as such (see e.g.
    # FD_Stream.fit_to_op) -- unconditionally dividing by 10 silently
    # produced a 10x-wrong rho (and therefore a badly wrong musp) for any
    # stream already storing distances in cm, e.g. every
    # forward.dos.simulate_*_td_stream output.
    distances = data.distance
    distances = distances * cm_per_unit(data.attrs.get('lengthUnit'), default='cm')
    rho_cm = xr.DataArray(distances, dims=["channel"])

    # --- Constants ---
    c_cm = (3e10 / n)                     # speed of light in medium [cm/s]

    # --- Liebert moment formulas ---
    # Absorption coefficient μa [cm^-1]
    mu_a = (m1**3) / (2 * c_cm * m2 * (m1**2 + m2))

    # Reduced scattering coefficient μs' [cm^-1]
    mu_sp = (2 * m1 * c_cm * (m1**2 + m2)) / (3 * rho_cm**2 * m2)

    # --- Stack output ---
    optical_xr = xr.concat(
        [mu_a, mu_sp],
        dim="op"
    )

    optical_xr.coords["op"] = ["mua", "musp"]

    optical_xr = optical_xr.transpose(
        "time", "channel", "wavelength", "op"
    )

    # --- Metadata ---
    from ..core.datastream import Datastream
    metadata = {
        "name": datastream_mom.name,
        "events": datastream_mom.events,
        "status": "optical",
        "history": datastream_mom.history + [
            Datastream._history_entry('moments_to_optical_params', {})
        ]
    }

    optical_xr.attrs["units"] = "cm^-1"
    # The closed-form relations above work in cm throughout, so record
    # that explicitly rather than leaving it implicit -- see
    # core.opt_prop_stream.fit_attrs for the metadata contract.
    optical_xr.attrs["lengthUnit"] = "cm"
    optical_xr.attrs["fitting_model"] = "liebert_moments_semi_infinite"
    optical_xr.attrs["transformation"] = "moments_to_optical_params"

    return optical_xr, metadata


# Calculate tissue optical parameters using moments (semi-infinite medium)
def calculate_optical_params(datastream_mom):
    """
    Derive optical properties from the normalised TD moments.

    Assumes a semi-infinite homogeneous medium and uses the mean time of
    flight and the variance.

    Parameters
    ----------
    datastream_mom : TD_Stream
        Stream with ``status='moment'``.

    Returns
    -------
    xr.DataArray
        Absorption and reduced scattering, in cm^-1.

    References
    ----------
    Liebert, A. et al. (2003). Applied Optics, 42(28), 5785-5792.
    """
    mom1 = datastream_mom.data.sel(moment='m1') # DTOF (measured) ~ TPSF Kernel accounts for IRF
    mom2 = datastream_mom.data.sel(moment='m2')

    rho = (datastream_mom.data.distance) / 10 # convert mm to cm
    n = 1.37
    c = 3e10 / n # speed of light in units cm/s and corrected for n

    mu_a = (mom1)**3 / (2 * c * mom2 * (mom1**2 + mom2))

    # Reduced scattering coefficient μs' [cm^-1]
    mu_sp = (2 * mom1 * c * (mom1**2 + mom2)) / (3 * rho**2 * mom2)

    # --- Stack output ---
    optical_xr = xr.concat(
        [mu_a, mu_sp],
        dim="op"
    )

    optical_xr.coords["op"] = ["mua", "musp"]

    optical_xr = optical_xr.transpose(
        "time", "channel", "wavelength", "op"
    )

    # --- Metadata ---
    metadata = {
        "name": datastream_mom.name,
        "events": datastream_mom.events,
        "status": "optical",
        "history": datastream_mom.history + [
            "Computed mua and musp using TD moment relations."
        ]
    }

    optical_xr.attrs["units"] = "cm^-1"
    # The closed-form relations above work in cm throughout, so record
    # that explicitly rather than leaving it implicit -- see
    # core.opt_prop_stream.fit_attrs for the metadata contract.
    optical_xr.attrs["lengthUnit"] = "cm"
    optical_xr.attrs["fitting_model"] = "liebert_moments_semi_infinite"
    optical_xr.attrs["transformation"] = "moments_to_optical_params"

    return optical_xr, metadata


# ------------------------------------------------------------------
# Optical property to chromophore concentration conversion
# ------------------------------------------------------------------

def opt_params_to_conc(mua, water_corr, water_frac, attrs=None,
                       component_dim='component',
                       status='absolute concentration',
                       uncertainty=None):
    """
    Unmix absorption into chromophore concentrations.

    Inverts the extinction-coefficient matrix wavelength by wavelength. Every
    dimension other than wavelength is carried through, so the same call
    serves channel- and voxel-indexed data.

    Parameters
    ----------
    mua : xr.DataArray
        Absorption in cm^-1, carrying a 'wavelength' dimension. May equally be
        a perturbation, in which case ``water_corr`` should be False.
    water_corr : bool
        Subtract water's contribution before unmixing.
    water_frac : float
        Water volume fraction assumed when ``water_corr`` is True.
    attrs : dict, optional
        Attributes to carry onto the result.
    component_dim : str
        Name of the payload dimension on the result: 'component' (default) for
        absolute composition, or 'chromophore' for relative changes.
    status : str
        Status attribute written onto the result.
    uncertainty : xr.DataArray, optional
        Per-wavelength variance of ``mua``, same shape. Propagated through the
        linear unmixing, treating wavelengths as independent.

    Returns
    -------
    xr.DataArray or tuple of (xr.DataArray, xr.DataArray)
        Concentrations, with their variance alongside when ``uncertainty`` is
        given.
    """
    # Define extinction - units cm^-1 / (moles / litre)
    from .mbll import get_extinction_coefficients_Prahl

    wls = mua.wavelength.values

    if len(wls) < 2:
        raise ValueError(f"Concentration derivation requires at least 2 wavelengths."
                         f"Found only {len(wls)}: {wls}")

    # include_water=water_corr -- only request the water column (sparser than
    # the Hb columns, see get_extinction_coefficients_Prahl's docstring) when
    # it's actually going to be used, not unconditionally.
    constants = get_extinction_coefficients_Prahl(wavelengths=wls, include_water=water_corr)

    mua_measured = mua  # shape (time, channel, wavelength)

    if water_corr:
        # Subtract the water contribution from measured mu_a
        # mua_corr = mua_tot - (mu_water * water_frac), e.g. water_frac=0.75
        # assumes tissue is 75% water
        mua_water = xr.DataArray(constants['water_mua'], coords={'wavelength': wls})
        mua = mua_measured - (mua_water * water_frac)
    else:
        mua = mua_measured

    E = constants['E_matrix']

    # Solve for HbO and HbR using the corrected signal
    E_inv = np.linalg.pinv(E)

    # Perform the linear solve
    # hbo = sum over wavelengths (mua_i * E_inv[0, i]) where i wl_idx
    # hbr = sum over wavelengths (mua_i * E_inv[1, i])
    hbo = (mua * xr.DataArray(E_inv[0, :], coords={'wavelength': wls})).sum(dim='wavelength') *1e6 # change units for M to \mu M
    hbr = (mua * xr.DataArray(E_inv[1, :], coords={'wavelength': wls})).sum(dim='wavelength') *1e6

    # Rebuild DataArray
    # Dimensions are taken from `mua` rather than hardcoded, so any spatial
    # index (channel, voxel, ...) survives the unmixing untouched.
    keep_dims = [d for d in mua.dims if d not in ('wavelength', 'op', 'layer')]
    conc_data = np.stack([hbo.transpose(*keep_dims).values,
                          hbr.transpose(*keep_dims).values], axis=-1)

    new_coords = {k: v for k, v in mua.coords.items()
                  if k not in ('wavelength', 'op', 'layer')}
    new_coords[component_dim] = ['HbO', 'HbR']
    new_attrs = dict(attrs or {})
    new_attrs['status'] = status
    new_attrs['units'] = r'uM/L'

    out = xr.DataArray(
        conc_data,
        coords=new_coords,
        dims=keep_dims + [component_dim],
        attrs=new_attrs
    )
    if uncertainty is None:
        return out

    # Linear propagation: c = E^+ mua, so var(c) = (E^+)^2 var(mua) with
    # wavelengths independent -- an available rule, so it is applied rather
    # than discarded (milob-package-design invariant 7). 1e12 = (1e6)^2,
    # matching the M -> uM scaling applied to the concentrations above.
    var_hbo = (uncertainty * xr.DataArray(E_inv[0, :] ** 2, coords={'wavelength': wls})
               ).sum(dim='wavelength') * 1e12
    var_hbr = (uncertainty * xr.DataArray(E_inv[1, :] ** 2, coords={'wavelength': wls})
               ).sum(dim='wavelength') * 1e12
    var_data = np.stack([var_hbo.transpose(*keep_dims).values,
                         var_hbr.transpose(*keep_dims).values], axis=-1)
    var_out = xr.DataArray(
        var_data, coords=new_coords, dims=keep_dims + [component_dim],
        attrs={'description': f'variance of {component_dim} concentrations '
                              '(uM^2), propagated through the linear unmixing'},
    )
    return out, var_out


def calculate_dpf_moments(td_stream, n=1.44, mode='static'):
    """
    Estimate the differential pathlength factor from the first TD moment.

    Uses the semi-infinite homogeneous approximation
    ``DPF = (c / n) * m1 / rho``, with ``m1`` the mean time of flight and
    ``rho`` the source-detector separation.

    Parameters
    ----------
    td_stream : TD_Stream
        Stream with ``status='raw'`` or ``status='moment'``. Raw gated data is
        converted to moments first.
    n : float
        Refractive index of the medium. Default 1.44.
    mode : {'static', 'timeseries'}
        'static' averages over time and returns dims (channel, wavelength).
        'timeseries' keeps the time axis when present, returning
        (time, channel, wavelength).

    Returns
    -------
    xarray.DataArray
        DPF values named 'dpf', with the refractive index and mode in
        ``attrs``.

    Raises
    ------
    ValueError
        If ``mode`` is unknown, the stream status is not 'raw' or 'moment', or
        the moment data lacks ``m1`` or a 'distance' coordinate.
    """
    if mode not in {'static', 'timeseries'}:
        raise ValueError(f"Mode must be 'static' or 'timeseries'. Got {mode!r}.")

    if td_stream.status == 'raw':
        moments = td_stream.to_moments()
    elif td_stream.status == 'moment':
        moments = td_stream
    else:
        raise ValueError(
            f"calculate_dpf() requires TD data with status='raw' or 'moment'. Got status='{td_stream.status}'."
        )

    if 'moment' not in moments.data.dims:
        raise ValueError("Moment data is required: 'moment' dimension was not found.")

    moment_names = np.asarray(moments.data.coords['moment'].values)
    if 'm1' not in moment_names:
        raise ValueError(
            "The first temporal moment 'm1' is required to compute DPF. "
            f"Available moments: {moment_names.tolist()}"
        )

    # Select the mean-arrival-time moment (m1)
    m1 = moments.data.sel(moment='m1')

    # Ensure we have the channel distance coordinate
    if 'distance' not in moments.data.coords:
        raise ValueError("TD data does not have a 'distance' coordinate for source-detector separation.")

    rho = moments.data.coords['distance']
    rho = rho * cm_per_unit(moments.data.attrs.get('lengthUnit'), default='cm')

    c_cm = 2.99792458e10        # Speed of light  [cm/s]

    # Compute DPF = (c / n) * m1 / rho
    dpf = (c_cm / float(n)) * m1 / rho

    if mode == 'static':
        if 'time' in dpf.dims:
            dpf = dpf.mean(dim='time', skipna=True)
        dpf = dpf.transpose('channel', 'wavelength')

    elif mode == 'timeseries':
        # Keep time axis when present; otherwise leave as channel/wavelength
        if 'time' in dpf.dims:
            dpf = dpf.transpose('time', 'channel', 'wavelength')

    dpf.name = 'dpf'
    dpf.attrs.update({
        'description': 'differential pathlength factor',
        'units': 'dimensionless',
        'refractive_index': float(n),
        'formula': 'DPF = (c / n) * m1 / rho',
        'mode': mode,
    })

    return dpf
