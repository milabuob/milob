"""Time-domain waveform plots: photon-time moments and TPSFs."""
import warnings

import numpy as np
import matplotlib.pyplot as plt

from . import theme


def plot_td_moments(moments_xr, channel_name=None, wl_indices=None):
    """
    Plot the first three photon-time moments over time for one channel.

    Parameters
    ----------
    moments_xr : xarray.DataArray
        Moments with time, channel, wavelength and moment dimensions.
    channel_name : str, optional
        Channel label of the form ``'S1D2'``. Defaults to the first channel.
    wl_indices : int or list of int, optional
        Wavelength indices to plot. Default is ``[0, 1]``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the panels.
    """
    if wl_indices is None:
        wl_indices = [0, 1]
    elif isinstance(wl_indices, (int, np.integer)):
        wl_indices = [wl_indices]
    if channel_name is None:
        channel_name = moments_xr.coords['channel'].values[0]

    subset = moments_xr.sel(channel=channel_name).isel(wavelength=wl_indices)
    time = subset.coords['time'].values
    wavelengths = subset.coords['wavelength'].values

    fig, axes = plt.subplots(1, 3, figsize=theme.signal_panel_figsize(1, ncols=3))

    plot_configs = [
        {'moment': 'm0', 'title': 'Total Intensity ($M_0$)', 'ylabel': 'Counts'},
        {'moment': 'm1', 'title': 'Mean Time of Flight ($M_1$)', 'ylabel': 'Time (s)'},
        {'moment': 'm2', 'title': 'Variance ($M_2$)', 'ylabel': r'Variance (s$^2$)'},
    ]

    for i, (ax, config) in enumerate(zip(axes, plot_configs)):
        for idx, wl in enumerate(wavelengths):
            y_values = subset.sel(moment=config['moment'], wavelength=wl).values
            ax.plot(time, y_values, color=theme.categorical_color(idx),
                    linewidth=1.4, alpha=0.9, label=f'{wl} nm')

        ax.set_title(config['title'], pad=12)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(config['ylabel'])
        theme.style_quantitative_axes(ax)

        if i == len(plot_configs) - 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, title='Wavelength')

    fig.suptitle(f"Time-Domain Analysis: Channel {channel_name}", y=1.02)
    fig.tight_layout()
    return fig


def _get_meta(datastream, key):
    if hasattr(datastream.probe, key):
        return getattr(datastream.probe, key)
    if isinstance(datastream.probe, dict) and key in datastream.probe:
        return datastream.probe[key]
    if key in datastream.data.coords:
        return datastream.data.coords[key].values
    if key in datastream.data.attrs:
        return datastream.data.attrs[key]
    return None


def plot_tpsf(datastream, channel="S1D1", wl=690, time_pt=100, normalise=False,
               y_max=None, ax=None):
    """
    Plot the temporal point spread function for one channel and wavelength.

    Parameters
    ----------
    datastream : Datastream
        Stream carrying the gated histograms and probe.
    channel : str, optional
        Channel label. Default is ``'S1D1'``.
    wl : int, optional
        Wavelength in nm. Default is 690.
    time_pt : int, optional
        Index of the sampling time to plot. Default is 100.
    normalise : bool, optional
        Normalise the counts to sum to one. Default is False.
    y_max : float, optional
        Fix the y-axis limit, so several plots can be compared directly.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the curve.
    """
    try:
        subset = datastream.data.sel(channel=channel, wavelength=wl)
    except KeyError:
        raise ValueError(
            f"Channel {channel} or wavelength {wl} not found. Check "
            "datastream.data.channel.values / datastream.data.wavelength.values"
        )

    n_time = subset.sizes['time']
    if not -n_time <= time_pt < n_time:
        raise ValueError(f"time_pt={time_pt} out of range for {n_time} time points")

    subset = subset.isel(time=time_pt)
    tpsf_counts = subset.values

    bin_centres = _get_meta(datastream, 'timeDelays')
    bin_widths = _get_meta(datastream, 'timeDelayWidths')

    if bin_centres is None or bin_widths is None:
        print("Warning: timeDelays/Widths not found in metadata. Using defaults (0, 0.25) ns.")
        n_bins = len(tpsf_counts)
        bin_centres = np.arange(n_bins) * 0.25 + 0.125
        width = 0.25
    else:
        width = bin_widths[0] if isinstance(bin_widths, (list, np.ndarray)) else bin_widths

    if normalise:
        # nansum, not sum: a stray NaN in tpsf_counts would otherwise make
        # `total` NaN and silently normalise every bin (not just the NaN
        # one) to NaN.
        total = np.nansum(tpsf_counts)
        if total != 0:
            tpsf_counts = tpsf_counts / total
        else:
            warnings.warn("TPSF sums to zero; skipping normalisation.")

    if ax is None:
        fig, ax = plt.subplots(figsize=theme.FIGSIZE['single'])
    else:
        fig = ax.figure

    ax.bar(bin_centres, tpsf_counts, width=width * 0.9,
           color=theme.categorical_color(0), alpha=0.85, edgecolor='black', linewidth=0.5)

    ax.set_xlabel("Time of Flight (s)")
    ax.set_ylabel("Photon Counts" if not normalise else "Normalised Counts")

    wavelength_nm = subset.coords['wavelength'].values
    ax.set_title(f"TPSF: {channel} ({wavelength_nm} nm)\nTime: {subset.time.values:.2f} s")

    if y_max is not None:
        ax.set_ylim(0, y_max)
    else:
        ax.set_ylim(0, np.nanmax(tpsf_counts) * 1.1)

    theme.style_quantitative_axes(ax)
    fig.tight_layout()
    return fig, ax
