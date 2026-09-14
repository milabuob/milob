"""
Shared visual style for every plot in MILOB.

Holds the colours, markers, colormaps and figure sizes the plotting
functions draw on, so that none of them hardcodes a literal. Source,
detector and chromophore colours are fixed conventions; the categorical
palette and the sequential and diverging colormaps are defaults chosen to
remain legible with colour-vision deficiency.

Examples
--------
>>> from milob.viz import theme
>>> fig, ax = plt.subplots(figsize=theme.FIGSIZE['single'])
>>> ax.scatter(x, y, c=theme.SOURCE_COLOR, marker=theme.SOURCE_MARKER)
>>> theme.style_spatial_axes(ax)
"""

from pathlib import Path

import matplotlib.pyplot as plt

_STYLE_PATH = Path(__file__).parent / "milob.mplstyle"

# --- Fixed field conventions (do not change without updating the physical
# convention itself: sources = red circles, detectors = black squares,
# HbO = red, HbR = blue, HbT = green) ---------------------------------------
SOURCE_COLOR = "#D62728"
SOURCE_MARKER = "o"
DETECTOR_COLOR = "#000000"
DETECTOR_MARKER = "s"

# 3-D anatomy surfaces (imaging/surface.py). Override these to restyle every
# 3-D plot at once; pass surface_color=... to restyle a single call.
HEAD_COLOR = "bisque"
BRAIN_COLOR = "lightgrey"

HBO_COLOR = "#D62728"
HBR_COLOR = "#1F77B4"
HBT_COLOR = "#2CA02C"
HBO_LINESTYLE = "-"
HBR_LINESTYLE = "--"
HBT_LINESTYLE = ":"
CHROMOPHORE_COLORS = {"HbO": HBO_COLOR, "HbR": HBR_COLOR, "HbT": HBT_COLOR}
CHROMOPHORE_LINESTYLES = {"HbO": HBO_LINESTYLE, "HbR": HBR_LINESTYLE, "HbT": HBT_LINESTYLE}

# --- Colormaps ---------------------------------------------------------------
SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "RdBu_r"

# --- Categorical palette (ROI/channel/graph-node identity; no field
# convention applies). Fixed order -- never cycle/re-sort per plot. Chosen to
# stay clear of the reserved red/blue/green and validated CVD-safe. ---------
CATEGORICAL_PALETTE = [
    "#eb6834",  # orange
    "#1baf7a",  # aqua/teal
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#4a3aa7",  # violet
]
UNASSIGNED_COLOR = "#7F7F7F"

# Flag maps (is_significant and friends) drawn on anatomy. A boolean has no
# magnitude and no sign, so it must not borrow the diverging red/blue scale a
# signed statistic uses -- red would read as an increase and blue as a decrease
# where the value only says yes. Purple keeps it clear of all three reserved
# hues, which CATEGORICAL_PALETTE's violet (#4a3aa7) does not manage in
# practice: once the 3-D shading darkens it against grey cortex it reads as
# plain blue, i.e. as the negative end of the very scale this avoids.
SIGNIFICANT_COLOR = "#8e3fbf"  # purple

# --- Figure size presets (inches), tied to a single-column / double-column
# journal layout rather than ad hoc per-function tuples. --------------------
FIGSIZE = {
    "single": (4.5, 4.0),
    "wide": (7.0, 4.0),
    "spatial": (6.0, 6.0),
}


def panel_figsize(nrows, ncols):
    """
    Figure size for a grid of compact panels, such as topographic maps.

    Parameters
    ----------
    nrows, ncols : int
        Shape of the panel grid.

    Returns
    -------
    tuple of float
        Width and height in inches.
    """
    return (3.2 * ncols, 3.0 * nrows)


def signal_panel_figsize(nrows, ncols=2):
    """
    Figure size for a grid of wide signal panels, such as a time series
    beside its spectrum.

    Parameters
    ----------
    nrows : int
        Number of rows.
    ncols : int, optional
        Number of columns. Default is 2.

    Returns
    -------
    tuple of float
        Width and height in inches.
    """
    return (6.0 * ncols, 4.0 * nrows)


def spatial_panel_figsize(nrows, ncols):
    """
    Figure size for a grid of spatial panels, which stay roughly square.

    Parameters
    ----------
    nrows, ncols : int
        Shape of the panel grid.

    Returns
    -------
    tuple of float
        Width and height in inches.
    """
    return (4.5 * ncols, 4.5 * nrows)


def graph_figsize(n_nodes, labeled=True):
    """
    Figure size for a node-link graph diagram.

    Grows with the node count when labels are drawn, since labels crowd as
    the graph gets denser.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the graph.
    labeled : bool, optional
        Whether node labels will be drawn. Default is True.

    Returns
    -------
    tuple of float
        Width and height in inches.
    """
    if not labeled:
        return FIGSIZE["spatial"]
    side = min(max(0.12 * n_nodes + 5.0, FIGSIZE["spatial"][0]), 14.0)
    return (side, side)


def timeline_figsize(n_rows):
    """
    Figure size for an event timeline.

    Wide, since the informative axis is time, and short, growing only
    modestly with the number of condition rows.

    Parameters
    ----------
    n_rows : int
        Number of condition rows.

    Returns
    -------
    tuple of float
        Width and height in inches.
    """
    return (FIGSIZE["wide"][0], min(max(0.55 * n_rows + 1.3, 2.0), 8.0))


_style_applied = False


def apply_style():
    """Apply the shared MILOB matplotlib style. Safe to call repeatedly."""
    global _style_applied
    plt.style.use(str(_STYLE_PATH))
    _style_applied = True


def style_spatial_axes(ax, show_axis=True, xlabel="X (mm)", ylabel="Y (mm)"):
    """
    Apply the standard styling for a spatial plot.

    Sets an equal aspect ratio, as probe maps, topographic maps and head
    renders all require.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to style.
    show_axis : bool, optional
        Keep ticks, spines and labels visible so scale can be judged from
        the figure. Pass False for a clean thumbnail. Default is True.
    xlabel, ylabel : str, optional
        Axis labels. Default is ``'X (mm)'`` and ``'Y (mm)'``.
    """
    ax.set_aspect("equal")
    if show_axis:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.axis("off")
    return ax


def style_quantitative_axes(ax, grid=True):
    """
    Apply the standard styling for a quantitative plot.

    Suits time series, spectra, histograms and box plots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to style.
    grid : bool, optional
        Draw a light grid on the major ticks. Default is True.
    """
    if grid:
        ax.grid(True, which="major", axis="both")
    else:
        ax.grid(False)
    return ax


def categorical_color(index):
    """
    Return the categorical colour for a zero-based index.

    Cycles through the palette when there are more categories than colours.

    Parameters
    ----------
    index : int
        Category index.

    Returns
    -------
    str
        Hex colour.
    """
    return CATEGORICAL_PALETTE[index % len(CATEGORICAL_PALETTE)]


def categorical_colors_for(labels, unassigned_value=None, start_index=0):
    """
    Assign a colour to each category label, in first-seen order.

    Parameters
    ----------
    labels : iterable
        Category labels, which may repeat.
    unassigned_value : optional
        Label treated as unassigned and given the reserved colour rather
        than a palette slot.
    start_index : int, optional
        Number of palette slots to skip, so a second set of categories
        sharing one axes does not repeat the first set's colours. Default
        is 0.

    Returns
    -------
    dict
        Label to hex colour.
    """
    mapping = {}
    next_idx = start_index
    for label in labels:
        if label == unassigned_value:
            mapping[label] = UNASSIGNED_COLOR
            continue
        if label not in mapping:
            mapping[label] = categorical_color(next_idx)
            next_idx += 1
    return mapping
