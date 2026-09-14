# viz/topo.py -- 2D probe-space plots: geometry, topographic heatmaps,
# ROI maps, and connectivity diagrams. All spatial (probe-space) plotting
# in the package lives here.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Ellipse
from scipy.interpolate import griddata
from typing import Optional

from . import theme

_NODE_COLOR = "#333333"


def draw_head_outline(ax, scale: float):
    """Draw a schematic nose and ears on the axes for orientation."""
    head = Circle((0, 0), scale, color='black', fill=False, lw=2, zorder=2)
    ax.add_patch(head)

    # Nose (pointing UP)
    nose_width = scale * 0.2
    nose_tip = scale * 1.15
    nose = Polygon([
        (-nose_width/2, scale * 0.95),
        (0, nose_tip),
        (nose_width/2, scale * 0.95)
    ], color='black', fill=False, lw=2, zorder=2)
    ax.add_patch(nose)

    # Ears
    ear_h, ear_w = scale * 0.3, scale * 0.1
    l_ear = Ellipse((-scale * 1.02, 0), ear_w, ear_h, color='black', fill=False, lw=2, zorder=2)
    r_ear = Ellipse((scale * 1.02, 0), ear_w, ear_h, color='black', fill=False, lw=2, zorder=2)
    ax.add_patch(l_ear)
    ax.add_patch(r_ear)


def _project_3d_to_2d(pos):
    """Flatten 3-D optode positions onto the plane by stereographic projection."""
    # If z is all zeros or nearly zero, assume already 2D
    if np.all(np.abs(pos[:, 2]) < 1e-5):
        return pos[:, 0], pos[:, 1]

    # Stereographic Projection
    # Assumes points are on a sphere. We project from the "South Pole" [0,0,-R]
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    r_max = np.max(r)

    # Stereographic Projection from South Pole
    # Ensuring top-down view (Nose up)
    # Standard formula: X = x / (1 + z/r), Y = y / (1 + z/r)
    x_proj = x / (1 + z/r_max)
    y_proj = y / (1 + z/r_max)
    return x_proj, y_proj


def _channel_midpoints(probe, s_x, s_y, d_x, d_y):
    """Return the 2-D midpoint between each channel's source and detector."""
    ch_x, ch_y = [], []
    for s_i, d_i in zip(probe._channels['sources'], probe._channels['detectors']):
        ch_x.append((s_x[s_i - 1] + d_x[d_i - 1]) / 2)
        ch_y.append((s_y[s_i - 1] + d_y[d_i - 1]) / 2)
    return np.array(ch_x), np.array(ch_y)


def _interpolate_topo(ch_x, ch_y, values, scale, grid_n=200, mask_factor=0.9):
    """Interpolate per-channel values onto a square grid masked to a circle."""
    valid = ~np.isnan(values)
    xi = np.linspace(-scale, scale, grid_n)
    yi = np.linspace(-scale, scale, grid_n)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((ch_x[valid], ch_y[valid]), values[valid], (xi, yi), method='cubic')
    mask = (xi**2 + yi**2) <= (scale * mask_factor)**2
    zi[~mask] = np.nan
    return xi, yi, zi


def _draw_optodes(ax, s_x, s_y, d_x, d_y, size=80, alpha=1.0, label=False, zorder=3):
    """Draw the source and detector markers in the package's fixed convention."""
    ax.scatter(s_x, s_y, c=theme.SOURCE_COLOR, marker=theme.SOURCE_MARKER, s=size,
               alpha=alpha, label='Sources' if label else None,
               edgecolors='white', linewidths=0.5, zorder=zorder)
    ax.scatter(d_x, d_y, c=theme.DETECTOR_COLOR, marker=theme.DETECTOR_MARKER, s=size,
               alpha=alpha, label='Detectors' if label else None,
               edgecolors='white', linewidths=0.5, zorder=zorder)


def plot_probe_2d(probe, ax: Optional[plt.Axes] = None, show_labels: bool = False,
                  show_axis: bool = True):
    """
    Plot the probe layout in two dimensions.

    Parameters
    ----------
    probe : Probe
        Probe whose optode positions are drawn.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.
    show_labels : bool, optional
        Annotate each optode with its label. Default is False.
    show_axis : bool, optional
        Keep labelled axes visible so scale can be judged. Pass False for a
        clean thumbnail. Default is True.

    Returns
    -------
    tuple
        The figure and axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=theme.FIGSIZE['spatial'])
    else:
        fig = ax.figure

    # Coordinate Projection (Flattening)
    s_x, s_y = _project_3d_to_2d(probe.s_pos)
    d_x, d_y = _project_3d_to_2d(probe.d_pos)

    # Calculate scale for outline
    all_x = np.concatenate([s_x, d_x])
    all_y = np.concatenate([s_y, d_y])
    max_dist = np.max(np.sqrt(all_x**2 + all_y**2))
    scale = max_dist * 1.1

    draw_head_outline(ax, scale)

    # Draw Channels (gold lines)
    if probe.has_channels:
        drawn_pairs = set()
        for s_idx_1, d_idx_1 in zip(probe._channels['sources'], probe._channels['detectors']):
            pair = (s_idx_1, d_idx_1)
            if pair not in drawn_pairs:
                s_i, d_i = s_idx_1 - 1, d_idx_1 - 1
                ax.plot([s_x[s_i], d_x[d_i]], [s_y[s_i], d_y[d_i]],
                        color='gold', lw=2, alpha=0.5, zorder=1)
                drawn_pairs.add(pair)

    # Draw Optodes
    _draw_optodes(ax, s_x, s_y, d_x, d_y, size=80, label=True)

    # Optional labels, offset from each marker
    if show_labels:
        label_props = dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray', lw=0.5)
        offset = scale * 0.05
        for i, (x, y) in enumerate(zip(s_x, s_y)):
            ax.text(x, y + offset, f'S{i+1}', fontsize=6, fontweight='bold',
                    ha='center', va='bottom', bbox=label_props, zorder=5)
        for i, (x, y) in enumerate(zip(d_x, d_y)):
            ax.text(x, y - offset, f'D{i+1}', fontsize=6, fontweight='bold',
                    ha='center', va='top', bbox=label_props, zorder=5)

    theme.style_spatial_axes(ax, show_axis=show_axis)
    ax.set_xlim(-scale * 1.3, scale * 1.3)
    ax.set_ylim(-scale * 1.2, scale * 1.4)

    return fig, ax


def plot_rois_2d(probe, ax: Optional[plt.Axes] = None, show_labels: bool = False,
              show_axis: bool = True):
    """
    Plot the probe's regions of interest, one colour per region.

    Channels belonging to no region are drawn in the neutral unassigned
    colour. Only channel midpoints are shown, not optodes.

    Parameters
    ----------
    probe : Probe
        Probe carrying the ROI definitions.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.
    show_labels : bool, optional
        Annotate each channel with its label. Default is False.
    show_axis : bool, optional
        Keep labelled axes visible. Default is True.

    Returns
    -------
    tuple
        The figure and axes.
    """
    if not probe.rois:
        raise ValueError("No ROIs defined. Use add_roi() or add_roi_by_*() first.")

    coords = probe.get_2d_coords()          # {label: (x, y)}
    roi_names = list(probe.rois.keys())
    colors = theme.categorical_colors_for(roi_names)
    roi_set = {ch for chs in probe.rois.values() for ch in chs}

    if ax is None:
        fig, ax = plt.subplots(figsize=theme.FIGSIZE['spatial'])
    else:
        fig = ax.figure

    # Unassigned channels
    unassigned_xy = [(x, y) for lbl, (x, y) in coords.items() if lbl not in roi_set]
    if unassigned_xy:
        ux, uy = zip(*unassigned_xy)
        ax.scatter(ux, uy, c=theme.UNASSIGNED_COLOR, s=55, zorder=2,
                   edgecolors='grey', linewidths=0.5, label='unassigned')

    # ROI channels
    for roi_name, chs in probe.rois.items():
        xs = [coords[ch][0] for ch in chs if ch in coords]
        ys = [coords[ch][1] for ch in chs if ch in coords]
        ax.scatter(xs, ys, c=colors[roi_name], s=75, label=roi_name,
                   zorder=3, edgecolors='white', linewidths=0.5)
        if show_labels:
            for ch in chs:
                if ch in coords:
                    ax.text(coords[ch][0], coords[ch][1], ch,
                            fontsize=6, ha='center', va='bottom', zorder=4)

    ax.legend(loc='upper right', framealpha=0.85, fontsize=8)
    ax.set_title("ROI Assignment")
    theme.style_spatial_axes(ax, show_axis=show_axis)
    fig.tight_layout()
    return fig, ax


def plot_topo_map(probe, values: np.ndarray, ax: Optional[plt.Axes] = None,
                   cmap=None, vmin=None, vmax=None, show_labels=False, title=None,
                   colorbar_label='Value', show_axis=True, show_cbar=True, **kwargs):
    """
    Plot per-channel values as a topographic heatmap.

    Parameters
    ----------
    probe : Probe
        Probe supplying the channel positions.
    values : numpy.ndarray
        One value per channel, in probe channel order.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.
    cmap : str, optional
        Colormap. Defaults to the theme's diverging map.
    vmin, vmax : float, optional
        Colour scale limits. Taken from the data if omitted.
    show_labels : bool, optional
        Annotate each channel with its label. Default is False.
    title : str, optional
        Axes title.
    colorbar_label : str, optional
        Label for the colorbar. Default is ``'Value'``.
    show_axis : bool, optional
        Keep labelled axes visible. Default is True.
    show_cbar : bool, optional
        Draw the colorbar. Default is True.
    **kwargs
        Passed to :meth:`matplotlib.axes.Axes.imshow`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the map.
    """
    cmap = cmap or theme.DIVERGING_CMAP

    if ax is None:
        fig, ax = plt.subplots(figsize=theme.FIGSIZE['spatial'])
    else:
        fig = ax.figure

    s_x, s_y = _project_3d_to_2d(probe.s_pos)
    d_x, d_y = _project_3d_to_2d(probe.d_pos)
    ch_x, ch_y = _channel_midpoints(probe, s_x, s_y, d_x, d_y)

    # Alignment check
    if len(ch_x) != len(values):
        print(f"Warning: Data size ({len(values)}) mismatch with Probe size ({len(ch_x)}).")
        min_len = min(len(ch_x), len(values))
        ch_x, ch_y, values = ch_x[:min_len], ch_y[:min_len], values[:min_len]

    if not np.any(~np.isnan(values)):
        print("Warning: all values are NaN; skipping interpolation.")
        return plot_probe_2d(probe, ax=ax, show_labels=show_labels, show_axis=show_axis)

    scale = np.max(np.sqrt(s_x**2 + s_y**2)) * 1.1
    xi, yi, zi = _interpolate_topo(ch_x, ch_y, values, scale)

    im = ax.imshow(zi, extent=(-scale, scale, -scale, scale), origin='lower',
                    cmap=cmap, vmin=vmin, vmax=vmax, zorder=0)
    ax.set_title(title)
    if show_cbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)

    draw_head_outline(ax, scale)
    _draw_optodes(ax, s_x, s_y, d_x, d_y, size=20, alpha=0.6, zorder=2)
    if show_labels:
        label_props = dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray', lw=0.5)
        offset = scale * 0.05
        for i, label in enumerate(probe.channel_labels[:len(ch_x)]):
            ax.text(ch_x[i], ch_y[i] + offset, label, fontsize=6,
                    ha='center', va='bottom', bbox=label_props, zorder=5)

    theme.style_spatial_axes(ax, show_axis=show_axis)
    ax.set_xlim(-scale * 1.3, scale * 1.3)
    ax.set_ylim(-scale * 1.2, scale * 1.4)

    return fig, ax


def align_group_values_to_probe(probe, channel_labels, values):
    """
    Align labelled values to the probe's full channel order.

    Channels absent from the input are filled with NaN.

    Parameters
    ----------
    probe : Probe
        Probe defining the target channel order.
    channel_labels : sequence of str
        Labels the values belong to.
    values : array-like
        Values to align.

    Returns
    -------
    numpy.ndarray
        Values in probe channel order, NaN where a channel was missing.
    """
    full_labels = [f"S{s}D{d}" for s, d in zip(probe._channels['sources'],
                                              probe._channels['detectors'])]
    aligned = np.full(len(full_labels), np.nan)
    label_to_idx = {lab: i for i, lab in enumerate(channel_labels)}
    for i, lab in enumerate(full_labels):
        if lab in label_to_idx:
            aligned[i] = values[label_to_idx[lab]]
    return aligned


def plot_topo_with_significance(probe, values: np.ndarray, sig_mask: np.ndarray = None,
                                 cmap=None, vmin=None, vmax=None,
                                 show_labels=False, title=None, ax=None, show_cbar=True,
                                 colorbar_label='Value', show_axis=True, marker='x', **kwargs):
    """
    Plot a topographic map with significant channels marked.

    Parameters
    ----------
    probe : Probe
        Probe supplying the channel positions.
    values : numpy.ndarray
        One value per channel, in probe channel order.
    sig_mask : numpy.ndarray, optional
        Boolean array marking the channels to mark. No markers are drawn if
        omitted.
    cmap : str, optional
        Colormap. Defaults to the theme's diverging map.
    vmin, vmax : float, optional
        Colour scale limits. Taken from the data if omitted.
    show_labels : bool, optional
        Annotate each channel with its label. Default is False.
    title : str, optional
        Axes title.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.
    show_cbar : bool, optional
        Draw the colorbar. Default is True.
    colorbar_label : str, optional
        Label for the colorbar. Default is ``'Value'``.
    show_axis : bool, optional
        Keep labelled axes visible. Default is True.
    marker : str, optional
        Marker style for significant channels. Default is ``'x'``.
    **kwargs
        Passed to :meth:`matplotlib.axes.Axes.imshow`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the map.
    """
    cmap = cmap or theme.DIVERGING_CMAP

    if ax is None:
        fig, ax = plt.subplots(figsize=theme.FIGSIZE['spatial'])
    else:
        fig = ax.figure

    s_x, s_y = _project_3d_to_2d(probe.s_pos)
    d_x, d_y = _project_3d_to_2d(probe.d_pos)
    ch_x, ch_y = _channel_midpoints(probe, s_x, s_y, d_x, d_y)

    min_len = min(len(values), len(ch_x))
    values = values[:min_len]
    ch_x, ch_y = ch_x[:min_len], ch_y[:min_len]
    if sig_mask is not None:
        sig_mask = sig_mask[:min_len]

    scale = np.max(np.sqrt(s_x**2 + s_y**2)) * 1.1
    xi, yi, zi = _interpolate_topo(ch_x, ch_y, values, scale)

    im = ax.imshow(zi, extent=(-scale, scale, -scale, scale), origin='lower',
                    cmap=cmap, vmin=vmin, vmax=vmax, zorder=0)
    if show_cbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)

    draw_head_outline(ax, scale)

    if probe.has_channels:
        drawn_pairs = set()
        for s_idx_1, d_idx_1 in zip(probe._channels['sources'], probe._channels['detectors']):
            pair = (s_idx_1, d_idx_1)
            if pair not in drawn_pairs:
                s_i, d_i = s_idx_1 - 1, d_idx_1 - 1
                ax.plot([s_x[s_i], d_x[d_i]], [s_y[s_i], d_y[d_i]],
                        color='gold', lw=1, alpha=0.5, zorder=1)
                drawn_pairs.add(pair)

    _draw_optodes(ax, s_x, s_y, d_x, d_y, size=20, alpha=0.6, zorder=2)

    if sig_mask is not None:
        for i, sig in enumerate(sig_mask):
            if sig:
                ax.scatter(ch_x[i], ch_y[i], marker=marker, s=30, color='black', lw=0.75, zorder=10)

    if show_labels:
        label_props = dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray', lw=0.5)
        offset = scale * 0.05
        for i, label in enumerate(probe.channel_labels[:len(ch_x)]):
            ax.text(ch_x[i], ch_y[i] + offset, label, fontsize=6, ha='center', va='bottom', bbox=label_props)

    theme.style_spatial_axes(ax, show_axis=show_axis)
    ax.set_xlim(-scale * 1.3, scale * 1.3)
    ax.set_ylim(-scale * 1.2, scale * 1.4)
    ax.set_title(title)

    return fig, ax


def plot_tstat_conjunction_multi(probe, hbo_tstats, hbo_p, hbr_p, hbo_betas, hbr_betas,
                                  p_threshold=0.05, condition='hbo_and_hbr',
                                  cmap=None, vmax=None, ax=None, title=None,
                                  show_labels=False, show_axis=True, **kwargs):
    """
    Plot HbO t-statistics, marking channels significant under a conjunction.

    Parameters
    ----------
    probe : Probe
        Probe supplying the channel positions.
    hbo_tstats : numpy.ndarray
        HbO t-statistic per channel, the quantity mapped.
    hbo_p, hbr_p : numpy.ndarray
        p-value per channel for each chromophore.
    hbo_betas, hbr_betas : numpy.ndarray
        Beta estimate per channel for each chromophore, used for the sign
        conditions.
    p_threshold : float, optional
        Significance threshold. Default is 0.05.
    condition : {'hbo_and_hbr', 'hbo', 'hbr'}, optional
        Which criterion marks a channel. ``'hbo_and_hbr'`` requires an HbO
        increase and an HbR decrease, both significant, the canonical
        activation pattern; the others test one chromophore alone. Default
        is ``'hbo_and_hbr'``.
    cmap : str, optional
        Colormap. Defaults to the theme's diverging map.
    vmax : float, optional
        Symmetric colour limit, with ``vmin`` set to its negation. Taken
        from the largest absolute t-statistic in the panel if omitted.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.
    title : str, optional
        Axes title.
    show_labels : bool, optional
        Annotate each channel with its label. Default is False.
    show_axis : bool, optional
        Keep labelled axes visible. Default is True.
    **kwargs
        Passed to :func:`plot_topo_with_significance`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the map.
    """
    hbo_sig = (hbo_p < p_threshold) & (hbo_betas > 0)
    hbr_sig = (hbr_p < p_threshold) & (hbr_betas < 0)
    if condition == 'hbo_and_hbr':
        sig_mask = hbo_sig & hbr_sig
    elif condition == 'hbo':
        sig_mask = hbo_sig
    elif condition == 'hbr':
        sig_mask = hbr_sig
    else:
        raise ValueError(f"Unknown condition: {condition!r} (expected 'hbo_and_hbr', 'hbo', or 'hbr')")

    if vmax is None:
        finite = hbo_tstats[np.isfinite(hbo_tstats)]
        vmax = np.nanmax(np.abs(finite)) if finite.size else 1.0

    return plot_topo_with_significance(
        probe, values=hbo_tstats, sig_mask=sig_mask,
        cmap=cmap or theme.DIVERGING_CMAP, vmin=-vmax, vmax=vmax,
        title=title, ax=ax, show_labels=show_labels, show_axis=show_axis,
        colorbar_label='t-statistic (HbO)', **kwargs
    )


def plot_connectivity_map(probe, matrix, chromophore='HbO', threshold=0.5,
                           show_labels=True, label_fontsize=8, ax=None, show_axis=True):
    """
    Plot a connectivity matrix as a network over the probe layout.

    An edge is drawn between every channel pair whose absolute value
    reaches ``threshold``, coloured by sign.

    Parameters
    ----------
    probe : Probe
        Probe supplying the channel positions.
    matrix : pandas.DataFrame
        Square channel-by-channel matrix, indexed by channel label.
    chromophore : str, optional
        Chromophore named in the title. Default is ``'HbO'``.
    threshold : float, optional
        Minimum absolute value for an edge to be drawn. Default is 0.5.
    show_labels : bool, optional
        Annotate each node with its channel label. Default is True.
    label_fontsize : int, optional
        Font size for the node labels. Default is 8.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.
    show_axis : bool, optional
        Keep labelled axes visible. Default is True.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the map.
    """
    pos = probe.get_2d_coords()
    valid_channels = matrix.index.tolist()

    if ax is None:
        fig, ax = plt.subplots(figsize=theme.FIGSIZE['spatial'])
    else:
        fig = ax.figure

    for i, ch_i in enumerate(valid_channels):
        for ch_j in valid_channels[i + 1:]:
            val = matrix.loc[ch_i, ch_j]
            if abs(val) >= threshold:
                p1, p2 = pos[ch_i], pos[ch_j]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        color=theme.HBO_COLOR if val > 0 else theme.HBR_COLOR,
                        alpha=np.clip(abs(val), 0.2, 0.8), linewidth=abs(val) * 5, zorder=1)

    for ch in valid_channels:
        x, y = pos[ch]
        ax.scatter(x, y, c=_NODE_COLOR, s=40, zorder=5, edgecolors='white')
        if show_labels:
            ax.text(x, y + 0.005, ch, fontsize=label_fontsize,
                    ha='center', va='bottom', zorder=6,
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.5))

    ax.set_title(f"Connectivity Map ({chromophore})\nThreshold |r| ≥ {threshold}")
    theme.style_spatial_axes(ax, show_axis=show_axis)

    return fig, ax
