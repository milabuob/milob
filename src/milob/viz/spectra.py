import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import xarray as xr
from typing import Union, List, Optional

from . import theme
from .events import overlay_events, event_legend_handles


def plot_timeseries_psd(data: xr.DataArray,
                        channels: Union[str, List[str]],
                        types: Optional[Union[str, List[str]]] = None,
                        fmin: float = 0.0,
                        fmax: Optional[float] = None,
                        events=None,
                        event_style: str = "auto",
                        event_alpha: float = 0.15):
    """
    Plot time series and their power spectra side by side.

    Parameters
    ----------
    data : xarray.DataArray
        Array with ``time``, ``channel`` and either ``wavelength`` or
        ``chromophore`` dimensions.
    channels : str or list of str
        Channel labels to plot.
    types : str or list of str, optional
        Wavelengths or chromophores to plot. All available are used if
        omitted.
    fmin, fmax : float, optional
        Frequency range of the spectrum panels in Hz.
    events : Events or pandas.DataFrame, optional
        Markers to overlay on the time-series panels only. Each condition
        keeps one colour across every row.
    event_style : {'auto', 'span', 'line'}, optional
        How events are drawn; see :func:`milob.viz.events.overlay_events`.
        Default is ``'auto'``.
    event_alpha : float, optional
        Opacity of shaded event blocks. Default is 0.15.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the panels.
    """
    # Handle inputs
    if isinstance(channels, str):
        channels = [channels]
    channels = list(np.unique(channels))

    # Identify dimension (wavelength or chromophore)
    type_dim = 'chromophore' if 'chromophore' in data.dims else 'wavelength'
    if types is None:
        types = list(data.coords[type_dim].values)
    elif isinstance(types, (str, int, float)):
        types = [types]

    fs = 1.0 / np.nanmean(np.diff(data.time.values))
    if fmax is None:
        fmax = fs / 2.0

    # Setup Figure
    n_rows = len(channels)
    fig, axes = plt.subplots(n_rows, 2, figsize=theme.signal_panel_figsize(n_rows), squeeze=False)
    plt.subplots_adjust(hspace=0.4, wspace=0.25)

    event_colors = None

    for i, ch in enumerate(channels):
        ax_time = axes[i, 0]
        ax_psd = axes[i, 1]

        for j, t in enumerate(types):
            # HbO/HbR/HbT use the fixed field colors + linestyles; anything
            # else (raw wavelengths, other types) falls back to the
            # categorical palette by position.
            line_color = theme.CHROMOPHORE_COLORS.get(t, theme.categorical_color(j))
            linestyle = theme.CHROMOPHORE_LINESTYLES.get(t, '-')

            try:
                sel_dict = {'channel': ch, type_dim: t}
                signal = data.sel(**sel_dict).values
                time = data.time.values

                # Plot Time Series
                ax_time.plot(time, signal, label=f"{t}", color=line_color,
                             linestyle=linestyle, alpha=0.9, lw=1.4)

                # PSD Calculation
                mask = np.isfinite(signal)
                if np.any(mask):
                    sig_clean = signal[mask]
                    # Welch estimate: use a window of 60s for better low-freq resolution
                    nperseg = min(len(sig_clean), int(fs * 60))
                    f, pxx = welch(sig_clean, fs=fs, nperseg=nperseg)

                    ax_psd.semilogy(f, pxx, label=f"{t}", color=line_color,
                                     linestyle=linestyle, lw=1.5)
            except KeyError:
                continue

        # Event markers: time panel only -- the PSD panel's x-axis is
        # frequency, so an onset time means nothing there. One shared
        # color_map keeps a condition the same color down every row.
        if events is not None:
            event_colors = overlay_events(ax_time, events, style=event_style,
                                          color_map=event_colors,
                                          alpha=event_alpha,
                                          start_index=len(types))

        # Formatting
        ax_time.set_title(f"Channel: {ch}", fontweight='bold')
        ax_time.set_ylabel("Amplitude")
        theme.style_quantitative_axes(ax_time)
        type_legend = ax_time.legend(title=type_dim, fontsize='small', loc='upper right')

        # A second legend for the conditions, on the first row only -- keep
        # the type legend alive by promoting it to an artist first.
        if event_colors and i == 0:
            ax_time.add_artist(type_legend)
            ax_time.legend(
                handles=event_legend_handles(event_colors, event_style, events=events),
                title='condition', fontsize='small', loc='upper left')

        ax_psd.set_title(f"Power Spectrum: {ch}", fontweight='bold')
        ax_psd.set_ylabel("Power (V²/Hz)")
        ax_psd.set_xlim(fmin, fmax)
        theme.style_quantitative_axes(ax_psd)

        if i == n_rows - 1:
            ax_time.set_xlabel("Time (s)")
            ax_psd.set_xlabel("Frequency (Hz)")

    return fig
