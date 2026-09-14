"""Plots of channel quality metrics such as SCI and SNR."""
import numpy as np
import matplotlib.pyplot as plt

from . import theme


def plot_quality_histogram(values: np.ndarray, threshold: float,
                            metric_name: str = "Metric", ax=None, **kwargs):
    """
    Plot a histogram of a quality metric with its threshold marked.

    Parameters
    ----------
    values : numpy.ndarray
        Metric value for each channel.
    threshold : float
        Quality threshold, drawn as a vertical line.
    metric_name : str, optional
        Name of the metric, used in the labels and title.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.
    **kwargs
        Passed to :meth:`matplotlib.axes.Axes.hist`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the histogram.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=theme.FIGSIZE['single'])
    else:
        fig = ax.figure

    valid_mask = np.isfinite(values)
    vals_valid = values[valid_mask]

    if len(vals_valid) == 0:
        ax.text(0.5, 0.5, f"No valid {metric_name} data",
                ha='center', va='center', transform=ax.transAxes)
        return fig, ax

    hist_params = {
        'bins': 'auto',
        'edgecolor': 'white',
        'alpha': 0.85,
        'color': theme.categorical_color(0),
    }
    hist_params.update(kwargs)
    ax.hist(vals_valid, **hist_params)

    ax.axvline(threshold, color=theme.CATEGORICAL_PALETTE[3], linestyle='--', linewidth=2.5,
               label=f'Threshold ({threshold})')

    ax.set_xlabel(f'{metric_name} Value')
    ax.set_ylabel('Number of Channels')
    ax.set_title(f'{metric_name} Distribution Across Channels')

    n_good = np.sum(vals_valid >= threshold)
    n_total = len(vals_valid)
    pct_good = 100.0 * n_good / n_total

    stats_text = (
        "Summary:\n"
        f"Pass: {n_good}/{n_total} ({pct_good:.1f}%)\n"
        f"Mean: {np.nanmean(vals_valid):.3f}\n"
        f"Median: {np.nanmedian(vals_valid):.3f}"
    )
    ax.text(0.05, 0.95, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='0.7'),
            fontsize=9, family='monospace')

    ax.legend(loc='upper right')
    theme.style_quantitative_axes(ax, grid=True)

    return fig, ax
