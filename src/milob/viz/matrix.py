"""Channel-by-channel matrix heatmaps, used for connectivity matrices."""
import numpy as np
import matplotlib.pyplot as plt

from . import theme


def plot_matrix(data, cmap=None, vmin=None, vmax=None, center=None, ax=None,
                 title=None, colorbar_label='Value', annot=False, fmt='.2f',
                 tick_labels=None, show_axis=True):
    """
    Plot a matrix as a heatmap.

    Parameters
    ----------
    data : array-like or pandas.DataFrame
        Two-dimensional matrix to plot.
    cmap : str, optional
        Colormap. Defaults to the theme's diverging map.
    vmin, vmax : float, optional
        Colour scale limits. Taken from the data if omitted.
    center : float, optional
        Make the colour scale symmetric about this value, the usual choice
        for a signed metric such as correlation.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.
    annot : bool, optional
        Print each cell's value on the heatmap. Default is False.
    tick_labels : sequence of str or tuple, optional
        Labels for both axes, or a ``(row_labels, col_labels)`` pair to set
        them separately, which a rectangular matrix requires. Defaults to
        the index and columns when ``data`` is a DataFrame.
    colorbar_label : str, optional
        Label for the colorbar.
    title : str, optional
        Title for the plot.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the heatmap.
    """
    cmap = cmap or theme.DIVERGING_CMAP

    if hasattr(data, 'values'):
        row_labels, col_labels = list(data.index), list(data.columns)
        values = np.asarray(data.values, dtype=float)
    else:
        row_labels = col_labels = None
        values = np.asarray(data, dtype=float)

    if tick_labels is not None:
        if isinstance(tick_labels, tuple) and len(tick_labels) == 2:
            row_labels, col_labels = tick_labels
        else:
            row_labels = col_labels = tick_labels

    if row_labels is not None and len(row_labels) != values.shape[0]:
        raise ValueError(
            f"row_labels has {len(row_labels)} entries, but data has "
            f"{values.shape[0]} rows."
        )
    if col_labels is not None and len(col_labels) != values.shape[1]:
        raise ValueError(
            f"col_labels has {len(col_labels)} entries, but data has "
            f"{values.shape[1]} columns."
        )

    n_labels = max(
        len(row_labels) if row_labels is not None else 0,
        len(col_labels) if col_labels is not None else 0,
        max(values.shape),
    )

    if center is not None and vmin is None and vmax is None:
        vlim = np.nanmax(np.abs(values - center))
        vmin, vmax = center - vlim, center + vlim

    if ax is None:
        # Fixed figsize crushes tick labels once there are more than a
        # handful of channels -- scale the figure (and the tick font, in the
        # opposite direction) with the number of rows/columns instead.
        side = float(np.clip(0.09 * n_labels + 5.0, theme.FIGSIZE['spatial'][0], 16.0))
        fig, ax = plt.subplots(figsize=(side, side))
    else:
        fig = ax.figure

    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)

    if row_labels is not None or col_labels is not None:
        tick_fontsize = float(np.clip(500.0 / max(n_labels, 1), 5.0, 8.0))
        if col_labels is not None:
            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels, rotation=90, fontsize=tick_fontsize)
        if row_labels is not None:
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=tick_fontsize)
    elif not show_axis:
        ax.axis('off')

    if annot:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, format(v, fmt), ha='center', va='center', fontsize=6)

    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig, ax
