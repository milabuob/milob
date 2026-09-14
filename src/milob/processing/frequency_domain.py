import numpy as np
import xarray as xr
from typing import Optional, Union


def fd_slopefitting(data: Union[np.ndarray, xr.DataArray],
                    distances: Optional[np.ndarray] = None,
                    n: float = 1.33,
                    modulation_frequency: Optional[float] = None
                    ) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Derive optical properties from multi-distance frequency-domain data.

    Fits the slopes of log(r^2 * AC) and of phase against source-detector
    distance, which are linear in the diffusion regime.

    Parameters
    ----------
    data : xr.DataArray or np.ndarray
        Shape (time, channel, wavelength, freq), complex, with the channel
        dimension spanning source-detector distances. DC sits at freq 0 and
        the complex AC at the modulation frequency. For a plain array,
        ``distances`` and ``modulation_frequency`` are required.
    distances : np.ndarray, optional
        Source-detector distances in cm. Read from ``data.coords['distance']``
        if omitted.
    n : float
        Tissue refractive index. Default 1.33.
    modulation_frequency : float, optional
        Modulation frequency in Hz. Read from the largest non-zero freq
        coordinate if omitted.

    Returns
    -------
    op_da : xr.DataArray
        Shape (time, 1, wavelength, op) with op = ['mua', 'musp'], in cm^-1.
        The single channel is the fitted location.
    error_da : xr.DataArray
        Propagated fit uncertainty, same shape as ``op_da``.

    References
    ----------
    Fantini, S., Franceschini, M. A., & Gratton, E. (1994). Journal of the
    Optical Society of America B, 11(10), 2128-2138.
    """
    is_xarray = isinstance(data, xr.DataArray)

    if is_xarray:
        if distances is None:
            distances = data.coords['distance'].values
            if data.attrs.get('lengthUnit') == 'mm':
                distances = distances / 10.0

        if modulation_frequency is None:
            freqs = data.freq.values
            mod_freqs = freqs[freqs != 0]
            if len(mod_freqs) == 0:
                raise ValueError("No non-zero frequency found in 'freq' coordinate.")
            modulation_frequency = float(mod_freqs[-1])

        # Extract AC and phase from complex DataArray
        ac_da    = np.abs(data.sel(freq=modulation_frequency))    # (t, ch, wl)
        phase_da = xr.apply_ufunc(
            np.angle,
            data.sel(freq=modulation_frequency)
        ) #Required to keep output as an xarray since np.angle not a ufunc. (t, ch, wl) radians


        ac_vals    = ac_da.values
        phase_vals = phase_da.values
        original_coords = dict(data.coords)
        attrs = data.attrs.copy()

    else:
        # Legacy numpy path: expect (..., 2) where [..., 0]=DC, [..., 1]=complex AC
        if distances is None:
            raise ValueError(
                "Source-detector distances must be provided when input is a numpy array."
            )
        if modulation_frequency is None:
            raise ValueError(
                "modulation_frequency must be provided when input is a numpy array."
            )
        ac_vals    = np.abs(data[..., 1])
        phase_vals = np.angle(data[..., 1])
        original_coords = None
        attrs = {}

    # -----------------------------------------------------------------------
    # Slope fitting
    # -----------------------------------------------------------------------
    n_time, n_channels, n_wavelengths = ac_vals.shape
    omega = 2 * np.pi * modulation_frequency
    v = 2.99792458e10 / n   # speed of light in tissue (cm/s)

    mua_ts    = np.empty((n_time, n_wavelengths))
    musp_ts   = np.empty((n_time, n_wavelengths))
    mua_err   = np.empty((n_time, n_wavelengths))
    musp_err  = np.empty((n_time, n_wavelengths))

    for wi in range(n_wavelengths):
        for ti in range(n_time):
            ac_t    = ac_vals   [ti, :, wi]
            phase_t = phase_vals[ti, :, wi]

            # Real slope: fit log(r² · AC) vs distance
            real_y = np.log(distances**2 * ac_t)
            real_reg, real_cov = np.polyfit(distances, real_y,    1, cov=True)
            img_reg,  img_cov  = np.polyfit(distances, phase_t,   1, cov=True)

            k_r =  real_reg[0]
            k_i = -img_reg[0]   # phase decreases with distance → flip sign

            # Optical properties
            mua  = (omega / (2 * v)) * (k_r / k_i - k_i / k_r)
            musp = (2 * v / (3 * omega)) * k_r * k_i - mua

            mua_ts [ti, wi] = mua
            musp_ts[ti, wi] = musp

            # Uncertainty propagation (first-order)
            A = omega / (2 * v)
            B = 2 * v / (3 * omega)

            d_mua_dkr  =  A * ( 1 / k_i  +  k_i / k_r**2)
            d_mua_dki  =  A * (-k_r / k_i**2 - 1 / k_r)
            d_musp_dkr =  B * k_i - d_mua_dkr
            d_musp_dki =  B * k_r - d_mua_dki

            var_kr = real_cov[0, 0]
            var_ki = img_cov [0, 0]

            mua_err [ti, wi] = np.sqrt(d_mua_dkr**2  * var_kr + d_mua_dki**2  * var_ki)
            musp_err[ti, wi] = np.sqrt(d_musp_dkr**2 * var_kr + d_musp_dki**2 * var_ki)

    # -----------------------------------------------------------------------
    # Pack into (time, 1, wavelength, op) DataArrays
    # -----------------------------------------------------------------------
    # Insert singleton channel axis
    op_array  = np.stack([mua_ts,  musp_ts], axis=-1)[:, np.newaxis, :, :]   # (t,1,wl,2)
    err_array = np.stack([mua_err, musp_err], axis=-1)[:, np.newaxis, :, :]

    if is_xarray:
        wavelength_vals = data.wavelength.values
        time_vals       = data.time.values

        op_coords = {
            'time':       time_vals,
            'channel':    ['fitted_location'],
            'wavelength': wavelength_vals,
            'op':         ['mua', 'musp'],
        }

        op_da = xr.DataArray(
            op_array,
            coords=op_coords,
            dims=['time', 'channel', 'wavelength', 'op'],
            attrs={**attrs,
                   'units': 'cm^-1',
                   # Distances are converted to cm before fitting, so this
                   # is what the fitter actually worked in -- see
                   # core.opt_prop_stream.fit_attrs.
                   'lengthUnit': 'cm',
                   'fitting_model': 'fd_slopefitting',
                   'transformation': 'fd_slopefitting',
                   'modulation_frequency_Hz': modulation_frequency,
                   'refractive_index': n}
        )

        err_da = xr.DataArray(
            err_array,
            coords=op_coords,
            dims=['time', 'channel', 'wavelength', 'op'],
            attrs={**attrs,
                   'units': 'cm^-1',
                   'description': 'first-order propagated uncertainty'}
        )
        return op_da, err_da

    return op_array, err_array
