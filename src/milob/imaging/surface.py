import os

import warnings

import numpy as np

from ..viz import theme



#: Default opacity per named surface layer. Colours come from ``viz.theme``,
#: read at call time so overriding the theme takes effect without a reimport.
_LAYER_ALPHA = {'head': 0.20, 'brain': 0.10, 'ellipsoid': 0.10}


def _layer_style(name):
    """Return the default colour and alpha for a named surface layer."""
    colour = theme.BRAIN_COLOR if name == 'brain' else theme.HEAD_COLOR
    return colour, _LAYER_ALPHA[name]


def _is_rgb(spec):
    """Return True for a single colour tuple, as opposed to a list of colours."""
    return (isinstance(spec, (tuple, list)) and len(spec) in (3, 4)
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in spec))

#: Layer stack for each named `surface` value. 'both' deliberately overrides
#: the per-layer defaults: a faint scalp over a more solid brain is the only
#: combination in which you can read either one.
_LAYER_SETS = {
    'head':      [('head', None)],
    'brain':     [('brain', None)],
    'ellipsoid': [('ellipsoid', None)],
    'both':      [('head', 0.10), ('brain', 0.45)],
}


#: Anatomical view presets as (elev, azim) in degrees, matplotlib convention,
#: for a probe in MNI RAS (+x right, +y anterior, +z superior).
_VIEW_PRESETS = {
    'right':     (0, 0),     'left':      (0, 180),
    'front':     (0, 90),    'anterior':  (0, 90),
    'back':      (0, -90),   'posterior': (0, -90),
    'top':       (89, -90),  'superior':  (89, -90),
    'bottom':    (-89, -90), 'inferior':  (-89, -90),
    'oblique':   (20, -60),
}


def _resolve_view(view):
    """
    Normalise a view specification to elevation, azimuth and roll in degrees.

    Parameters
    ----------
    view : str, tuple or None
        A preset name, an (elev, azim) or (elev, azim, roll) tuple, or a
        PyVista camera position.

    Returns
    -------
    tuple or None
        Angles in degrees, or None to leave the camera alone.
    """
    if view is None:
        return None

    if isinstance(view, str):
        key = view.strip().lower()
        if key not in _VIEW_PRESETS:
            raise ValueError(
                f"Unknown view {view!r}. Expected one of {sorted(_VIEW_PRESETS)}, "
                f"or (elev, azim[, roll]) in degrees."
            )
        elev, azim = _VIEW_PRESETS[key]
        return float(elev), float(azim), 0.0

    seq = list(view)
    # pyvista camera_position: [(px,py,pz), (fx,fy,fz), (ux,uy,uz)]
    if len(seq) == 3 and all(np.ndim(item) == 1 and len(item) == 3 for item in seq):
        return _camera_to_view(seq)

    if len(seq) not in (2, 3):
        raise ValueError(
            f"view must be a preset name, (elev, azim), (elev, azim, roll), or a "
            f"pyvista camera_position; got {view!r}."
        )
    elev, azim = float(seq[0]), float(seq[1])
    roll = float(seq[2]) if len(seq) == 3 else 0.0
    return elev, azim, roll


def _view_to_camera(elev, azim, roll, focal, radius):
    """Convert elevation, azimuth and roll to a PyVista camera position."""
    e, a = np.radians(elev), np.radians(azim)
    direction = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    position = np.asarray(focal, float) + direction * radius

    # View-up: +z projected perpendicular to the view direction, degenerate
    # straight up/down (hence the 89 deg presets, which avoid the singularity).
    up = np.array([0.0, 0.0, 1.0])
    up = up - direction * (up @ direction)
    if np.linalg.norm(up) < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
        up = up - direction * (up @ direction)
    up /= np.linalg.norm(up)

    if roll:
        r = np.radians(roll)
        right = np.cross(direction, up)
        up = up * np.cos(r) + right * np.sin(r)
        up /= np.linalg.norm(up)

    return [tuple(position), tuple(np.asarray(focal, float)), tuple(up)]


def _camera_to_view(camera):
    """Convert a PyVista camera position to elevation, azimuth and roll."""
    position, focal, up = (np.asarray(v, dtype=float) for v in camera)
    direction = position - focal
    norm = np.linalg.norm(direction)
    if norm == 0:
        return 0.0, 0.0, 0.0
    direction /= norm

    elev = np.degrees(np.arcsin(np.clip(direction[2], -1.0, 1.0)))
    azim = np.degrees(np.arctan2(direction[1], direction[0]))

    reference = np.array([0.0, 0.0, 1.0])
    reference = reference - direction * (reference @ direction)
    if np.linalg.norm(reference) < 1e-6:
        reference = np.array([0.0, 1.0, 0.0])
        reference = reference - direction * (reference @ direction)
    reference /= np.linalg.norm(reference)
    right = np.cross(direction, reference)
    roll = np.degrees(np.arctan2(up @ right, up @ reference))
    return float(elev), float(azim), float(roll)


def _view_label(elev, azim, roll=0.0):
    """Format a camera angle as a string that can be passed back as ``view``."""
    if abs(roll) < 0.5:
        return f"view=({elev:.0f}, {azim:.0f})"
    return f"view=({elev:.0f}, {azim:.0f}, {roll:.0f})"


def _resolve_surface(surface, coreg, surface_alpha, surface_color=None):
    """
    Normalise the surface argument into a list of layers.

    Parameters
    ----------
    surface : str, Surface, or None
        A named layer stack, a mesh, a path to an OBJ file, or None to draw no
        surface.
    coreg : Coregistration
        Used to fit an ellipsoid where one is requested.
    surface_alpha : float, sequence or dict
        Opacity per layer.
    surface_color : colour, sequence or dict
        Colour per layer.

    Returns
    -------
    list of tuple
        Each layer as a mesh, colour and alpha, outermost first.
    """
    from .head_model import Surface, BrainSurface, HeadSurface

    alpha_seq = (list(surface_alpha)
                 if isinstance(surface_alpha, (tuple, list)) else None)

    def alpha_for(name, default, position=0):
        if isinstance(surface_alpha, dict):
            return surface_alpha.get(name, default)
        if alpha_seq is not None:
            return alpha_seq[position] if position < len(alpha_seq) else default
        return default if surface_alpha is None else surface_alpha

    def colour_for(name, default, position=0):
        if surface_color is None:
            return default
        if isinstance(surface_color, dict):
            return surface_color.get(name, default)
        if isinstance(surface_color, str) or _is_rgb(surface_color):
            return surface_color
        seq = list(surface_color)
        return seq[position] if position < len(seq) else default

    if surface is None or surface is False:
        return []

    if isinstance(surface, Surface):
        colour, default = _layer_style('head')
        return [(surface, colour_for('head', colour), alpha_for('head', default))]

    # A string is a file path only if it looks like one; anything else must be
    # a known layer name, so a typo raises instead of becoming a missing file.
    if isinstance(surface, str) and (os.sep in surface
                                     or surface.lower().endswith('.obj')):
        colour, default = _layer_style('head')
        return [(Surface.from_file(surface), colour_for('head', colour),
                 alpha_for('head', default))]

    name = surface
    if name not in _LAYER_SETS:
        raise ValueError(
            f"Unknown surface {surface!r}. Expected one of "
            f"{sorted(_LAYER_SETS)}, None, an OBJ path, or a Surface instance."
        )

    layers = []
    for position, (layer, override) in enumerate(_LAYER_SETS[name]):
        default_colour, default = _layer_style(layer)
        colour = colour_for(layer, default_colour, position)
        alpha = alpha_for(layer, default if override is None else override, position)
        if alpha <= 0:
            continue
        if layer == 'head':
            mesh = HeadSurface.auto(coreg)
        elif layer == 'ellipsoid':
            mesh = HeadSurface.ellipsoid(coreg)
        else:
            mesh = BrainSurface.auto() or HeadSurface.ellipsoid(coreg)
        if mesh is not None:
            layers.append((mesh, colour, alpha))
    return layers


def _cue_layer(layers, show_nose_ears, coreg):
    """
    Build the nose and ear cue layer.

    Parameters
    ----------
    layers : list of tuple
        The resolved surface layers.
    show_nose_ears : bool or None
        None draws the cues whenever a head-shaped surface is present and
        omits them for a bare cortex.
    coreg : Coregistration
        Supplies the landmarks the cues are pinned to.

    Returns
    -------
    tuple or None
        The cue layer, or None when no cues are drawn.
    """
    from .head_model import landmark_cues, HeadSurface

    head_shown = any(isinstance(mesh, HeadSurface) for mesh, _, _ in layers)
    if show_nose_ears is False or (show_nose_ears is None and not head_shown):
        return None

    reference = getattr(coreg, 'reference_landmarks', None)
    head_mesh = next((mesh for mesh, _, _ in layers
                      if isinstance(mesh, HeadSurface)), None)
    cues = landmark_cues(reference, surface=head_mesh)
    if cues is None:
        return None
    # Match the head it is attached to, a touch more solid so it reads as shape.
    colour, alpha = _layer_style('head')
    for mesh, layer_colour, layer_alpha in layers:
        if isinstance(mesh, HeadSurface):
            colour, alpha = layer_colour, layer_alpha
            break
    return (cues, colour, min(1.0, alpha * 1.6))


def plot_probe_3d(
    coregistration,
    values=None,
    connectivity_matrix=None,
    channel_labels=None,
    mode='nodes',
    threshold=0.3,
    surface='head',
    cmap=None,
    edge_cmap=None,
    vmin=None,
    vmax=None,
    show_labels=False,
    show_sources=False,
    show_detectors=False,
    surface_alpha=None,
    surface_color=None,
    show_nose_ears=None,
    view=None,
    show_view=True,
    title='fNIRS Probe – MNI space',
    colorbar_label='Value',
    max_edge_width=4.0,
    backend='matplotlib',
    **kwargs,
):
    """
    Draw a coregistered probe in three dimensions.

    Parameters
    ----------
    coregistration : Coregistration
        A fitted coregistration.
    values : array-like, optional
        One value per channel, colouring the nodes. Numeric values use a
        continuous scale with a colorbar; string values are treated as
        categories, drawn from a qualitative palette with a legend, and are
        supported only in 'nodes' mode.
    connectivity_matrix : np.ndarray, optional
        Shape (n_channels, n_channels). Required for 'connectome' mode.
    channel_labels : list of str, optional
        Channel names, used for labels and the colorbar.
    mode : {'connectome', 'nodes'}
        'connectome' draws edges between channel pairs above the threshold,
        with nodes coloured by mean connection strength. 'nodes' draws
        coloured markers at the channel midpoints alone.
    threshold : float
        Minimum absolute value for an edge to be drawn.
    surface : str, Surface, or None
        Anatomy rendered behind the probe. 'head' (default) draws the MNI152
        scalp, where optodes sit, and is the appropriate backdrop for
        channel-space values. 'brain' draws the cortical surface, appropriate
        for reconstructed quantities, above which a probe floats. 'both'
        draws a faint scalp over the cortex. 'ellipsoid' fits a crude shape
        to the optode cloud. None draws nothing. A mesh or OBJ path renders
        that mesh directly.
    cmap : str, optional
        Colormap for node values. Defaults to the theme's sequential map, or
        its categorical palette for labels.
    edge_cmap : str, optional
        Colormap for edges. Defaults to the theme's diverging map.
    vmin, vmax : float, optional
        Colour limits for node values.
    show_labels : bool
        Annotate channel midpoints with their labels.
    show_sources, show_detectors : bool
        Draw the optodes. Both default to False in connectome mode.
    surface_alpha : float, sequence or dict, optional
        Opacity of the rendered surfaces. A float applies to every layer, a
        sequence gives one value per layer outermost first, and a dict is
        keyed by layer name. An alpha of zero skips a layer. Two stacked
        translucent surfaces z-fight in matplotlib, which does not depth-sort
        across collections; PyVista composites them correctly.
    surface_color : colour, sequence or dict, optional
        Colour of the rendered surfaces, in the same three forms. Defaults to
        the theme's head and brain colours. The cues follow the head's colour.
    show_nose_ears : bool, optional
        Draw schematic nose and ears pinned to the landmarks, making the
        head's orientation readable. These are rendering cues rather than
        anatomy, the template being defaced. None draws them whenever a
        head-shaped surface is rendered.
    title : str
        Figure title.
    colorbar_label : str
        Label for the node colorbar.
    max_edge_width : float
        Maximum line width for edges.
    view : str, tuple, or None
        Camera angle, so a figure can be reproduced rather than rotated by
        hand. Accepts an anatomical preset, an (elev, azim) or
        (elev, azim, roll) tuple in degrees, or a PyVista camera position.
        None leaves each backend's default camera alone, and the same value
        means the same angle in both.
    show_view : bool
        Annotate the corner with the current camera in a form that can be
        passed back as ``view``. Default True. It updates after each drag in
        an interactive backend.
    backend : {'matplotlib', 'pyvista'}
        'matplotlib' draws a static figure; 'pyvista' opens an interactive
        window and requires the ``threed`` extra.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes) or pyvista.Plotter
    """
    layers = _resolve_surface(surface, coregistration, surface_alpha, surface_color)
    cues = _cue_layer(layers, show_nose_ears, coregistration)
    if cues is not None:
        layers.append(cues)
    cmap = cmap or theme.SEQUENTIAL_CMAP
    edge_cmap = edge_cmap or theme.DIVERGING_CMAP
    if backend == 'pyvista':
        return _plot_pyvista(
            coregistration, values, connectivity_matrix, channel_labels,
            mode, threshold, layers, cmap, edge_cmap, vmin, vmax, show_labels,
            show_sources, show_detectors, title, colorbar_label,
            max_edge_width, view, show_view, **kwargs
        )
    return _plot_matplotlib(
        coregistration, values, connectivity_matrix, channel_labels,
        mode, threshold, layers, cmap, edge_cmap, vmin, vmax, show_labels,
        show_sources, show_detectors, title, colorbar_label,
        max_edge_width, view, show_view, **kwargs
    )


# ──────────────────────────────────────────────────────────────────────────── #
# Shared helpers                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

def _build_edges(connectivity_matrix, mid_pos, threshold):
    """
    Build the edge segments above a threshold.

    Parameters
    ----------
    connectivity_matrix : np.ndarray
        Shape (n_channels, n_channels).
    mid_pos : np.ndarray
        Channel midpoints, shape (n_channels, 3).
    threshold : float
        Minimum absolute value for an edge.

    Returns
    -------
    list of tuple
        Each edge's endpoint pair and its value.
    """
    n = len(mid_pos)
    segments, r_vals = [], []
    for i in range(n):
        for j in range(i + 1, n):
            r = connectivity_matrix[i, j]
            if np.isnan(r) or abs(r) < threshold:
                continue
            segments.append([mid_pos[i], mid_pos[j]])
            r_vals.append(r)
    return segments, np.array(r_vals)


def _node_degree(connectivity_matrix):
    """Return each channel's mean absolute connection strength, ignoring NaNs."""
    mat = connectivity_matrix.copy().astype(float)
    np.fill_diagonal(mat, np.nan)
    return np.nanmean(np.abs(mat), axis=1)


def _is_categorical(values):
    """Return True if the values are category labels rather than numbers."""
    return np.asarray(values).dtype.kind in ('U', 'S', 'O')


def _categorical_palette(values, cmap_name):
    """
    Map each unique value to a colour, in first-seen order.

    Parameters
    ----------
    values : array-like
        Category labels.
    cmap_name : str, optional
        A qualitative colormap. Defaults to the theme's categorical palette.

    Returns
    -------
    tuple of (list, dict)
        The categories in order, and their colours.
    """
    import matplotlib.colors as mcolors

    categories = list(dict.fromkeys(values))

    if cmap_name in (None, 'tab10'):
        colors = theme.categorical_colors_for(categories, unassigned_value='unassigned')
        return categories, {c: mcolors.to_rgba(colors[c]) for c in categories}

    import matplotlib as mpl
    try:
        palette = mpl.colormaps[cmap_name].resampled(max(len(categories), 1))
    except (KeyError, ValueError):
        colors = theme.categorical_colors_for(categories, unassigned_value='unassigned')
        return categories, {c: mcolors.to_rgba(colors[c]) for c in categories}
    return categories, {c: palette(i) for i, c in enumerate(categories)}


# ──────────────────────────────────────────────────────────────────────────── #
# Matplotlib backend                                                           #
# ──────────────────────────────────────────────────────────────────────────── #

def _plot_matplotlib(coreg, values, connectivity_matrix, channel_labels,
                     mode, threshold, layers, cmap, edge_cmap, vmin, vmax,
                     show_labels, show_sources, show_detectors,
                     title, colorbar_label, max_edge_width, view, show_view,
                     **kwargs):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as mcm
    from mpl_toolkits.mplot3d import Axes3D          # noqa: F401

    fig = kwargs.get('fig') or plt.figure(figsize=(10, 9))
    ax  = kwargs.get('ax')  or fig.add_subplot(111, projection='3d')

    s_pos   = coreg.mni_s_pos
    d_pos   = coreg.mni_d_pos
    mid_pos = coreg.mni_channel_midpoints

    # ── Anatomy layers (outermost first) ──────────────────────────────────
    for mesh, colour, alpha in layers:
        v = mesh.vertices
        ax.plot_trisurf(
            v[:, 0], v[:, 1], v[:, 2],
            triangles=mesh.faces,
            alpha=alpha, color=colour,
            linewidth=0, shade=True, zorder=0,
        )

    cat_handles = None
    if mode == 'connectome':
        _mpl_connectome(ax, fig, mid_pos, connectivity_matrix, values,
                        threshold, cmap, edge_cmap, vmin, vmax,
                        colorbar_label, max_edge_width)
    else:
        cat_handles = _mpl_nodes(ax, fig, mid_pos, values, cmap, vmin, vmax, colorbar_label)

    # ── Sources & detectors ───────────────────────────────────────────────
    if show_sources:
        ax.scatter(s_pos[:, 0], s_pos[:, 1], s_pos[:, 2],
                   c=theme.SOURCE_COLOR, s=60, marker=theme.SOURCE_MARKER, zorder=6,
                   depthshade=False, label='Sources')
    if show_detectors:
        ax.scatter(d_pos[:, 0], d_pos[:, 1], d_pos[:, 2],
                   c=theme.DETECTOR_COLOR, s=60, marker=theme.DETECTOR_MARKER, zorder=6,
                   depthshade=False, label='Detectors')

    # ── Channel labels ────────────────────────────────────────────────────
    if show_labels and channel_labels is not None:
        for (x, y, z), lbl in zip(mid_pos, channel_labels):
            ax.text(x, y, z, f'  {lbl}', fontsize=6, zorder=7)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(title, pad=12)
    if show_sources or show_detectors or cat_handles:
        handles, _ = ax.get_legend_handles_labels()
        if cat_handles:
            handles = cat_handles + handles
        ax.legend(handles=handles, fontsize=8, loc='upper left')

    resolved = _resolve_view(view)
    if resolved is not None:
        ax.view_init(elev=resolved[0], azim=resolved[1], roll=resolved[2])

    if show_view:
        _attach_view_readout_mpl(fig, ax)

    plt.tight_layout()
    return fig, ax


def _attach_view_readout_mpl(fig, ax):
    """
    Annotate a matplotlib figure's corner with the current camera angle.

    Refreshed after each drag, so an angle found interactively can be read off
    and passed back as ``view``.
    """
    label = ax.text2D(0.99, 0.01, '', transform=ax.transAxes, ha='right',
                      va='bottom', fontsize=8, color='0.45', family='monospace')

    def refresh(_event=None):
        label.set_text(_view_label(ax.elev, ax.azim, getattr(ax, 'roll', 0.0) or 0.0))

    refresh()
    # Kept on the figure so the callback is not garbage-collected.
    fig._milob_view_cid = fig.canvas.mpl_connect('button_release_event', refresh)
    return label


def _mpl_nodes(ax, fig, mid_pos, values, cmap, vmin, vmax, colorbar_label):
    if values is not None and _is_categorical(values):
        import matplotlib.patches as mpatches
        categories, cat_color = _categorical_palette(values, cmap)
        colors = [cat_color[v] for v in values]
        ax.scatter(mid_pos[:, 0], mid_pos[:, 1], mid_pos[:, 2],
                  c=colors, s=70, zorder=5, depthshade=False,
                  edgecolors='white', linewidths=0.5)
        return [mpatches.Patch(color=cat_color[c], label=str(c)) for c in categories]

    if values is not None:
        values = np.asarray(values, dtype=float)
        _vmin  = vmin if vmin is not None else np.nanmin(values)
        _vmax  = vmax if vmax is not None else np.nanmax(values)
        sc = ax.scatter(mid_pos[:, 0], mid_pos[:, 1], mid_pos[:, 2],
                        c=values, cmap=cmap, vmin=_vmin, vmax=_vmax,
                        s=70, zorder=5, depthshade=False)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.45, pad=0.1)
        cbar.set_label(colorbar_label, fontsize=9)
    else:
        ax.scatter(mid_pos[:, 0], mid_pos[:, 1], mid_pos[:, 2],
                   c=theme.UNASSIGNED_COLOR, s=70, zorder=5, depthshade=False,
                   label='Channels')


def _mpl_connectome(ax, fig, mid_pos, connectivity_matrix, node_values,
                    threshold, node_cmap, edge_cmap, vmin, vmax,
                    colorbar_label, max_edge_width):
    import matplotlib as mpl
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    segments, r_vals = _build_edges(connectivity_matrix, mid_pos, threshold)

    if len(segments):
        # Edge colours mapped to r value (positive = warm, negative = cool)
        e_norm = mcolors.Normalize(vmin=-1, vmax=1)
        e_cm   = mpl.colormaps[edge_cmap]
        colors     = [e_cm(e_norm(r)) for r in r_vals]
        linewidths = np.clip(np.abs(r_vals), 0, 1) * max_edge_width
        # Alpha proportional to |r| so weak edges fade out
        for seg, col, lw, r in zip(segments, colors, linewidths, r_vals):
            xs = [seg[0][0], seg[1][0]]
            ys = [seg[0][1], seg[1][1]]
            zs = [seg[0][2], seg[1][2]]
            alpha = float(np.clip(abs(r), 0.15, 1.0))
            ax.plot(xs, ys, zs, color=col, linewidth=lw, alpha=alpha, zorder=3)

        # Colorbar for edges
        sm = mcm.ScalarMappable(cmap=edge_cmap, norm=e_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.35, pad=0.12, label='r value')
        cbar.ax.tick_params(labelsize=8)

    # Nodes coloured by degree (or supplied values)
    if node_values is None and connectivity_matrix is not None:
        node_values = _node_degree(connectivity_matrix)
    _vmin = vmin if vmin is not None else (np.nanmin(node_values) if node_values is not None else 0)
    _vmax = vmax if vmax is not None else (np.nanmax(node_values) if node_values is not None else 1)

    sc = ax.scatter(mid_pos[:, 0], mid_pos[:, 1], mid_pos[:, 2],
                    c=node_values if node_values is not None else theme.UNASSIGNED_COLOR,
                    cmap=node_cmap, vmin=_vmin, vmax=_vmax,
                    s=40, zorder=5, depthshade=False,
                    edgecolors='white', linewidths=0.5)
    if node_values is not None:
        cbar2 = fig.colorbar(sc, ax=ax, shrink=0.35, pad=0.02)
        cbar2.set_label(colorbar_label or 'Mean |r|', fontsize=9)
        cbar2.ax.tick_params(labelsize=8)


# ──────────────────────────────────────────────────────────────────────────── #
# pyvista backend                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

def _plot_pyvista(coreg, values, connectivity_matrix, channel_labels,
                  mode, threshold, layers, cmap, edge_cmap, vmin, vmax,
                  show_labels, show_sources, show_detectors,
                  title, colorbar_label, max_edge_width, view, show_view,
                  **kwargs):
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "pyvista is required for the 'pyvista' backend. "
            "Install it with:  pip install pyvista"
        ) from exc

    s_pos   = coreg.mni_s_pos
    d_pos   = coreg.mni_d_pos
    mid_pos = coreg.mni_channel_midpoints

    plotter = pv.Plotter(title=title)

    # ── Anatomy layers (outermost first) ──────────────────────────────────
    for mesh, colour, alpha in layers:
        poly = pv.PolyData(mesh.vertices, np.hstack([
            np.full((len(mesh.faces), 1), 3, dtype=int), mesh.faces
        ]))
        plotter.add_mesh(poly, opacity=alpha, color=colour, smooth_shading=True)

    if mode == 'connectome' and connectivity_matrix is not None:
        _pv_connectome(plotter, mid_pos, connectivity_matrix, values,
                       threshold, cmap, edge_cmap, vmin, vmax,
                       colorbar_label, max_edge_width)
    else:
        pts = pv.PolyData(mid_pos.astype(float))
        if values is not None and _is_categorical(values):
            categories, cat_color = _categorical_palette(values, cmap)
            rgba = np.array([cat_color[v] for v in values])
            pts['rgba'] = (rgba * 255).astype(np.uint8)
            plotter.add_mesh(pts, scalars='rgba', rgba=True, point_size=14,
                             render_points_as_spheres=True)
            plotter.add_legend([[str(c), cat_color[c][:3]] for c in categories],
                               bcolor='white')
        elif values is not None:
            pts['value'] = np.asarray(values, dtype=float)
            plotter.add_mesh(pts, scalars='value', cmap=cmap,
                             clim=[vmin, vmax], point_size=14,
                             render_points_as_spheres=True,
                             scalar_bar_args={'title': colorbar_label})
        else:
            plotter.add_mesh(pts, color=theme.UNASSIGNED_COLOR, point_size=14,
                             render_points_as_spheres=True)

    if show_sources:
        plotter.add_mesh(pv.PolyData(s_pos.astype(float)),
                         color=theme.SOURCE_COLOR, point_size=10,
                         render_points_as_spheres=True, label='Sources')
    if show_detectors:
        plotter.add_mesh(pv.PolyData(d_pos.astype(float)),
                         color=theme.DETECTOR_COLOR, point_size=10,
                         render_points_as_spheres=True, label='Detectors')

    if show_labels and channel_labels is not None:
        for pos, lbl in zip(mid_pos, channel_labels):
            plotter.add_point_labels([pos.tolist()], [lbl],
                                     font_size=8, point_color='black',
                                     point_size=1)
    if plotter.renderer._labels:
        plotter.add_legend()

    resolved = _resolve_view(view)
    if resolved is not None:
        focal = np.vstack([m.vertices for m, _, _ in layers]).mean(axis=0) \
            if layers else mid_pos.mean(axis=0)
        radius = float(np.linalg.norm(mid_pos - focal, axis=1).max()) * 6.0
        plotter.camera_position = _view_to_camera(*resolved, focal, radius)
        plotter.reset_camera(render=False)   # keep the angle, refit the zoom

    if show_view:
        _attach_view_readout_pv(plotter)

    plotter.show()
    # PyVista never auto-closes plotters inside Jupyter (see Plotter.show),
    # so a static/non-interactive render leaks its off-screen render window
    # (and everything it holds: mesh buffers, textures, framebuffer) unless
    # we close it ourselves once the screenshot has been captured. Skip this
    # for genuinely interactive backends (e.g. 'trame'), where the caller
    # still needs the live view.
    if plotter.notebook and plotter._theme.jupyter_backend == 'static':
        plotter.close()
    return plotter


def _attach_view_readout_pv(plotter):
    """
    Annotate a PyVista plotter's corner with the current camera angle.

    Refreshed whenever a drag ends, so an angle found interactively can be
    read off and passed back as ``view``.
    """
    # An explicit viewport position returns a plain text actor; the corner
    # keywords return a vtkCornerAnnotation, which has a different setter.
    actor = plotter.add_text('', position=(0.68, 0.015), viewport=True,
                             font_size=9, color='grey', font='courier')

    def refresh(*_args):
        elev, azim, roll = _camera_to_view(plotter.camera_position)
        actor.SetInput(_view_label(elev, azim, roll))

    refresh()
    interactor = getattr(plotter, 'iren', None)
    if interactor is not None:
        # EndInteractionEvent fires on mouse-up, so the text settles on the
        # angle you actually chose rather than flickering through the drag.
        try:
            interactor.add_observer('EndInteractionEvent', refresh)
        except Exception:          # non-interactive/off-screen plotters
            pass
    return actor


def _pv_connectome(plotter, mid_pos, connectivity_matrix, node_values,
                   threshold, node_cmap, edge_cmap, vmin, vmax,
                   colorbar_label, max_edge_width):
    import pyvista as pv

    segments, r_vals = _build_edges(connectivity_matrix, mid_pos, threshold)

    if len(segments):
        import matplotlib as mpl
        import matplotlib.colors as mcolors
        e_norm = mcolors.Normalize(vmin=-1, vmax=1)
        e_cm   = mpl.colormaps[edge_cmap]
        for (p1, p2), r in zip(segments, r_vals):
            line = pv.Line(p1.tolist(), p2.tolist())
            rgba = e_cm(e_norm(r))
            plotter.add_mesh(
                line,
                color=rgba[:3],
                opacity=float(np.clip(abs(r), 0.15, 1.0)),
                line_width=float(np.clip(abs(r), 0, 1) * max_edge_width),
            )

    if node_values is None and connectivity_matrix is not None:
        node_values = _node_degree(connectivity_matrix)

    pts = pv.PolyData(mid_pos.astype(float))
    if node_values is not None:
        pts['degree'] = node_values.astype(float)
        plotter.add_mesh(pts, scalars='degree', cmap=node_cmap,
                         clim=[vmin or float(np.nanmin(node_values)),
                               vmax or float(np.nanmax(node_values))],
                         point_size=10, render_points_as_spheres=True,
                         scalar_bar_args={'title': colorbar_label or 'Mean |r|'})
    else:
        plotter.add_mesh(pts, color=theme.UNASSIGNED_COLOR, point_size=10,
                         render_points_as_spheres=True)


# ---------------------------------------------------------------------------
# Voxel-space statistical maps on a cortical surface
# ---------------------------------------------------------------------------

def _shade(face_colours, normals, ambient=0.55, light=(0.4, 0.3, 0.86)):
    """
    Apply Lambertian shading so that a curved mesh reads as curved.

    Once face colours are set explicitly the renderer's own shading is lost
    and the geometry appears flat. Brightness is scaled by the angle between
    each face normal and a fixed light, leaving hue and saturation, and so the
    statistic, untouched.

    Parameters
    ----------
    face_colours : np.ndarray
        Per-face RGBA colours.
    normals : np.ndarray
        Per-face normals.
    ambient : float
        Minimum brightness.
    light : array-like
        Direction the light comes from.

    Returns
    -------
    np.ndarray
        Shaded face colours.
    """
    lightdir = np.asarray(light, dtype=float)
    lightdir = lightdir / np.linalg.norm(lightdir)
    intensity = np.abs(normals @ lightdir)
    scale = ambient + (1.0 - ambient) * intensity
    out = face_colours.copy()
    out[:, :3] = np.clip(out[:, :3] * scale[:, None], 0.0, 1.0)
    return out


def _face_normals(vertices, faces):
    tri = vertices[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(norms > 0, norms, 1.0)


def plot_surface_map(values, voxel_grid, surface=None, *, views=('left', 'right'),
                     hemi='auto', backend='matplotlib', cmap=None, vmin=None,
                     vmax=None, symmetric=True, threshold=None, tail='both',
                     radius=12.0,
                     method='gaussian', fwhm=10.0, min_voxels=1, voxel_mask=None,
                     channels=None, channel_values=None, channel_size=60,
                     unpainted_color='0.82', title=None, colorbar_label=None,
                     figsize=None, show_colorbar=True, zoom=1.45,
                     resolution='fsaverage5', ax=None):
    """
    Render a voxel-space statistical map on a cortical surface.

    Values are resampled onto the mesh, and vertices with no voxel within the
    gather radius are left unpainted, so the field of view is drawn honestly:
    the unpainted colour means the array said nothing there, not that nothing
    happened there.

    Parameters
    ----------
    values : array-like
        Shape (n_voxels,), aligned with the grid's positions.
    voxel_grid : VoxelGrid
        The domain the values live on.
    surface : Surface, optional
        Target mesh, which must be in the same frame as the grid. Defaults to
        an fsaverage pial surface in MNI space, so a grid built with a
        coregistration lines up and one left in digitiser space will not.
    views : str or sequence of str
        One panel per view, each a preset name or an (elev, azim) pair.
    hemi : {'auto', 'left', 'right', 'both'}
        Which hemisphere to draw. 'auto' (default) shows only the near one for
        a lateral view. This is not cosmetic: matplotlib does not depth-sort a
        mesh this size reliably, so a far hemisphere bleeds through the near
        one and blobs appear on the wrong surface.
    backend : {'matplotlib', 'pyvista'}
        'matplotlib' returns a static figure; 'pyvista' opens an interactive
        window and uses only the first view.
    cmap : str, optional
        Colormap. Defaults to the theme's diverging map.
    vmin, vmax : float, optional
        Colour limits. Default to symmetric limits at the 99th percentile of
        the absolute values.
    symmetric : bool
        Whether automatic limits straddle zero. Default True. Set False for a
        one-sided quantity, where the limits become the 1st and 99th
        percentiles of what reaches the surface. Ignored once limits are
        given.
    threshold : float, optional
        Hide vertices that do not clear this value.
    tail : {'both', 'positive', 'negative'}
        Which side the threshold keeps. 'both' (default) keeps either sign, so
        negative regions survive; pair it with a diverging colormap and
        symmetric limits, or they read as positive.
    radius : float
        Gather window in mm, forwarded to the projection.
    method : {'gaussian', 'idw', 'peak', 'nearest'}
        How voxels within the window are combined.
    fwhm : float
        Kernel width in mm, for the Gaussian method.
    min_voxels : int
        Vertices supported by fewer voxels than this are left unpainted.
    voxel_mask : array-like of bool, optional
        Restrict the projection to these voxels, typically a grey-matter mask.
        A depth-defined domain is mostly scalp and skull, which a generous
        radius would otherwise pull onto the cortical surface.
    channels : array-like, optional
        Channel midpoints to draw over the map, in the same frame as the grid.
        Worth showing on a backprojected map, where a smooth blob otherwise
        reads as a focal result. They are drawn at their true scalp positions,
        so they float above the cortex.
    channel_values : array-like, optional
        Colour the channel markers on the same scale as the surface. By
        default they are a single neutral colour.
    channel_size : float
        Marker size.
    unpainted_color : colour
        Colour used where no value is supported.
    title : str, optional
        Figure title.
    colorbar_label : str, optional
        Label for the colorbar.
    figsize : tuple, optional
        Figure size in inches.
    show_colorbar : bool
        Draw the colorbar.
    zoom : float
        How much of each panel the surface fills. Values above one enlarge it.
    resolution : str
        Mesh density used when ``surface`` is None.
    ax : matplotlib.axes.Axes, optional
        Axes to draw a single view on.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, np.ndarray) or pyvista.Plotter
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as mcm
    from ..viz import theme
    from .head_model import BrainSurface

    if surface is None:
        # fsaverage5 (~10k vertices/hemisphere) rather than BrainSurface.auto()'s
        # fsaverage3 default: a statistical map needs enough vertices to resolve
        # a blob's edge, and 640 per hemisphere renders it as a few triangles.
        surface = BrainSurface.auto(resolution)
        if surface is None:
            raise ImportError(
                "plot_surface_map needs a cortical surface. Install nilearn "
                "(pip install 'milob[threed]') so BrainSurface.auto() can "
                "fetch fsaverage, or pass surface=... explicitly."
            )

    values = np.asarray(values, dtype=float).ravel()
    if voxel_mask is not None:
        voxel_mask = np.asarray(voxel_mask, dtype=bool)
        if voxel_mask.shape != values.shape:
            raise ValueError(
                f"voxel_mask must match values, got {voxel_mask.shape} vs "
                f"{values.shape}.")
        # NaN is already "no information" to project_to_surface, so masking
        # needs no separate code path.
        values = np.where(voxel_mask, values, np.nan)

    vertex_vals = voxel_grid.project_to_surface(
        values, surface, radius=radius, method=method, fwhm=fwhm,
        min_voxels=min_voxels)

    finite = np.isfinite(vertex_vals)
    if not finite.any():
        raise ValueError(
            "No surface vertex has a supported value. Usually this means the "
            "grid and the surface are in different coordinate frames (was the "
            "grid built with coreg=?), or `radius` is smaller than the voxel "
            "spacing."
        )

    auto_limits = vmax is None and vmin is None
    if symmetric:
        if vmax is None:
            vmax = float(np.nanpercentile(np.abs(vertex_vals[finite]), 99)) or 1.0
        if vmin is None:
            vmin = -vmax
    else:
        # Limits from the vertices, not the voxels: on a depth-defined domain
        # the extremes live in scalp, which never reaches the cortical mesh,
        # so voxel-derived limits leave the visible range crushed into one end.
        if vmax is None:
            vmax = float(np.nanpercentile(vertex_vals[finite], 99))
        if vmin is None:
            vmin = float(np.nanpercentile(vertex_vals[finite], 1))
        if not vmax > vmin:
            vmax = vmin + 1.0
    # plt.get_cmap, not cm.get_cmap: the latter was removed in matplotlib 3.9.
    cmap = cmap if hasattr(cmap, '__call__') else plt.get_cmap(cmap or theme.DIVERGING_CMAP)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    verts, faces = np.asarray(surface.vertices, float), np.asarray(surface.faces, int)
    # A face is painted only if all three of its vertices are supported --
    # otherwise colour bleeds one triangle past the true field of view.
    face_vals = vertex_vals[faces].mean(axis=1)
    paint = np.isfinite(vertex_vals[faces]).all(axis=1)
    if threshold is not None:
        if tail == 'positive':
            paint &= face_vals >= threshold
        elif tail == 'negative':
            paint &= face_vals <= -threshold
        elif tail == 'both':
            paint &= np.abs(face_vals) >= threshold
        else:
            raise ValueError(
                f"tail must be 'both', 'positive' or 'negative', got {tail!r}."
            )

    # Matplotlib's Normalize clips silently, so a value outside [vmin, vmax]
    # is drawn in the end colour and becomes indistinguishable from one that
    # genuinely sits there. With tail='both' and a one-sided scale that is
    # actively misleading -- a strongly negative region renders as though it
    # were part of a positive map. Say so rather than let the figure lie.
    if paint.any():
        shown = face_vals[paint]
        # The dangerous case, always worth saying: values clamped across zero,
        # i.e. negatives drawn on a non-negative scale. Those do not merely
        # saturate, they change apparent sign.
        # The clamped value must actually cross zero to change apparent sign.
        # Without the `shown` half of each test, any saturation on a scale
        # that happens not to straddle zero -- a log-sensitivity map, a map of
        # a positive quantity -- is reported as a sign flip it is not.
        wrong_sign = int(((shown < vmin) & (vmin >= 0) & (shown < 0)).sum()
                         + ((shown > vmax) & (vmax <= 0) & (shown > 0)).sum())
        n_out = int(((shown < vmin) | (shown > vmax)).sum())
        if wrong_sign:
            warnings.warn(
                f"{wrong_sign} of {paint.sum()} painted faces have the "
                f"opposite sign to the colour scale [{vmin:g}, {vmax:g}] and "
                f"were clamped to its end colour -- they will read as though "
                f"they were on the other side of zero. Either pass "
                f"tail='positive'/'negative' so only one side is shown, or "
                f"use a diverging colormap with symmetric limits.",
                UserWarning, stacklevel=2,
            )
        elif n_out and not auto_limits:
            # Saturation only; harmless with auto limits (the 99th-percentile
            # default clips ~1% by construction), worth flagging when the
            # caller chose the limits and may not expect it.
            warnings.warn(
                f"{n_out} of {paint.sum()} painted faces fall outside "
                f"[vmin={vmin:g}, vmax={vmax:g}] and were clamped to the end "
                f"colours.", UserWarning, stacklevel=2,
            )

    colours = np.tile(mcolors.to_rgba(unpainted_color), (len(faces), 1))
    colours[paint] = cmap(norm(face_vals[paint]))
    colours = _shade(colours, _face_normals(verts, faces))

    if backend == 'pyvista':
        return _plot_surface_map_pyvista(verts, faces, colours, views, title,
                                        channels=channels,
                                        channel_values=channel_values,
                                        cmap=cmap, norm=norm)

    face_x = verts[faces][:, :, 0].mean(axis=1)

    view_list = [views] if isinstance(views, str) or (
        len(views) == 2 and all(isinstance(v, (int, float)) for v in views)) else list(views)

    if ax is not None:
        axes = [ax]
        fig = ax.get_figure()
        view_list = view_list[:1]
    else:
        figsize = figsize or (5.2 * len(view_list), 5.0)
        fig = plt.figure(figsize=figsize)
        axes = [fig.add_subplot(1, len(view_list), i + 1, projection='3d')
                for i in range(len(view_list))]

    for axis, view in zip(axes, view_list):
        side = hemi
        if hemi == 'auto':
            side = view if (isinstance(view, str) and view in ('left', 'right')) else 'both'
        if side == 'left':
            keep_f = face_x < 0
        elif side == 'right':
            keep_f = face_x > 0
        else:
            keep_f = np.ones(len(faces), dtype=bool)

        tri = axis.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                                triangles=faces[keep_f], linewidth=0,
                                antialiased=False, shade=False)
        tri.set_fc(colours[keep_f])

        if channels is not None:
            pts = np.asarray(channels, dtype=float)
            # Same hemisphere rule as the mesh: a channel on the far side
            # would otherwise punch through the cortex, matplotlib's 3-D
            # renderer not being able to depth-sort it against the surface.
            if side == 'left':
                keep_c = pts[:, 0] < 0
            elif side == 'right':
                keep_c = pts[:, 0] > 0
            else:
                keep_c = np.ones(len(pts), dtype=bool)
            if keep_c.any():
                shared = dict(s=channel_size, edgecolors='white',
                              linewidths=0.6, depthshade=False, zorder=6)
                if channel_values is None:
                    axis.scatter(*pts[keep_c].T, c=theme.UNASSIGNED_COLOR, **shared)
                else:
                    axis.scatter(*pts[keep_c].T,
                                 c=np.asarray(channel_values, float)[keep_c],
                                 cmap=cmap, norm=norm, **shared)

        resolved = _resolve_view(view)
        if resolved is not None:
            axis.view_init(elev=resolved[0], azim=resolved[1], roll=resolved[2])
        axis.set_axis_off()

        # Frame the drawn hemisphere tightly. Matplotlib pads 3-D axes
        # generously and equalises nothing, so without explicit limits the
        # brain sits small in a sea of whitespace.
        shown = verts[np.unique(faces[keep_f])]
        centre = (shown.min(axis=0) + shown.max(axis=0)) / 2.0
        half = np.ptp(shown, axis=0).max() / 2.0 * 1.02
        axis.set_xlim(centre[0] - half, centre[0] + half)
        axis.set_ylim(centre[1] - half, centre[1] + half)
        axis.set_zlim(centre[2] - half, centre[2] + half)
        # zoom fills the panel: an equal-limit cube leaves the brain small,
        # since the cube is sized by the longest axis (anterior-posterior)
        # while a lateral view only ever shows the y-z face.
        axis.set_box_aspect((1, 1, 1), zoom=zoom)
        if isinstance(view, str):
            axis.set_title(view, fontsize=10)

    if show_colorbar:
        sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02,
                     label=colorbar_label or '')
    if title:
        fig.suptitle(title, y=0.97)
    return fig, (axes[0] if len(axes) == 1 else axes)


def _plot_surface_map_pyvista(verts, faces, face_colours, views, title,
                              channels=None, channel_values=None,
                              cmap=None, norm=None):
    """Render the surface map interactively, the PyVista counterpart of the matplotlib backend."""
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "backend='pyvista' needs pyvista (pip install 'milob[threed]')."
        ) from exc

    cells = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
    mesh = pv.PolyData(verts, cells)
    mesh.cell_data['colour'] = (face_colours[:, :3] * 255).astype(np.uint8)

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars='colour', rgb=True)

    if channels is not None:
        pts = pv.PolyData(np.asarray(channels, dtype=float))
        if channel_values is None:
            plotter.add_mesh(pts, color=theme.UNASSIGNED_COLOR, point_size=14,
                             render_points_as_spheres=True)
        else:
            # Colours resolved here rather than handed over as scalars, so the
            # dots and the surface cannot end up on two different norms.
            rgba = cmap(norm(np.asarray(channel_values, dtype=float)))
            pts['rgba'] = (rgba * 255).astype(np.uint8)
            plotter.add_mesh(pts, scalars='rgba', rgba=True, point_size=14,
                             render_points_as_spheres=True)

    view = views[0] if not isinstance(views, str) else views
    resolved = _resolve_view(view)
    if resolved is not None:
        plotter.camera_position = _view_to_camera(
            *resolved, verts.mean(axis=0),
            float(np.linalg.norm(verts - verts.mean(axis=0), axis=1).max()) * 3.0)
    if title:
        plotter.add_text(title, font_size=10)
    return plotter
