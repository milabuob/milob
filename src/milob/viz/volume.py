"""
Voxel-space maps drawn over anatomical slices.

The 2-D counterpart of :func:`milob.imaging.surface.plot_surface_map`. A
surface rendering shows only what reaches the cortical mesh, whereas a
reconstruction domain defined by depth below the scalp is largely
extracerebral, so slices are what show whether the signal sits in brain
or in scalp and skull.
"""
import numpy as np

from . import theme


def load_anatomical(background):
    """
    Normalise a background specification into a volume and its affine.

    Parameters
    ----------
    background : str, Path, nibabel image, or tuple
        A NIfTI path or image, which requires NiBabel, or an array already
        paired with its 4x4 voxel-to-world affine.

    Returns
    -------
    tuple
        The 3-D volume and its 4x4 affine.
    """
    if isinstance(background, tuple) and len(background) == 2:
        vol, affine = background
        return np.asarray(vol).squeeze(), np.asarray(affine, dtype=float)

    if hasattr(background, 'affine') and hasattr(background, 'dataobj'):
        return np.asarray(background.dataobj).squeeze(), np.asarray(background.affine, float)

    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "Reading a NIfTI background needs nibabel (pip install nibabel, "
            "or 'milob[threed]'). Alternatively pass (volume, affine) directly."
        ) from exc
    img = nib.load(str(background))
    return np.asarray(img.dataobj).squeeze(), np.asarray(img.affine, float)


_AXES = {'x': 0, 'y': 1, 'z': 2}


def plot_voxel_slices(values, voxel_grid, background=None, *, axis='z',
                      slices=None, n_slices=6, voxel_mask=None,
                      cmap=None, vmin=None, vmax=None, threshold=None,
                      tail='both', slab=None, fill_radius=None,
                      bg_cmap='gray', overlay_alpha=1.0, title=None,
                      colorbar_label=None, figsize=None, ncols=None,
                      orientation_labels=True):
    """
    Draw a per-voxel map over anatomical slices, in world millimetres.

    Parameters
    ----------
    values : array-like
        One scalar per voxel, aligned with ``voxel_grid.positions``.
    voxel_grid : VoxelGrid
        Grid supplying the voxel positions.
    background : str, nibabel image, or tuple, optional
        Structural volume drawn underneath, which must be in the same frame
        as the grid. The slices are drawn on a plain background if omitted.
    axis : {'x', 'y', 'z'}, optional
        Slice orientation; ``'z'`` gives axial slices. Default is ``'z'``.
    slices : sequence of float, optional
        World coordinates in mm to slice at. Defaults to ``n_slices``
        positions spread over the range the voxels occupy.
    n_slices : int, optional
        Number of slices when ``slices`` is omitted. Default is 6.
    voxel_mask : array-like of bool, optional
        Restrict the overlay to these voxels, for instance grey matter only.
    cmap : str, optional
        Colormap for the overlay. Defaults to the theme's map.
    vmin, vmax : float, optional
        Colour scale limits. Taken from the data if omitted.
    threshold : float, optional
        Hide values that do not clear this magnitude.
    tail : {'both', 'positive', 'negative'}, optional
        Side of the threshold to keep. Default is ``'both'``.
    slab : float, optional
        Half-thickness in mm of the voxel slab collapsed into each slice.
        Defaults to half the voxel spacing, so no voxel is counted twice.
    fill_radius : float, optional
        In-plane radius in mm each voxel paints. Defaults to the voxel
        spacing, so the map renders at its true resolution rather than as
        points.
    bg_cmap : str, optional
        Colormap for the background volume. Default is ``'gray'``.
    overlay_alpha : float, optional
        Opacity of the overlay. Default is 1.0.
    title : str, optional
        Figure title.
    colorbar_label : str, optional
        Label for the colorbar.
    figsize : tuple, optional
        Figure size in inches. Chosen from the grid shape if omitted.
    ncols : int, optional
        Panels per row.
    orientation_labels : bool, optional
        Annotate each panel with its anatomical directions, which
        distinguishes radiological from neurological convention. Default is
        True.

    Returns
    -------
    tuple
        The figure and its axes.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from scipy.spatial import cKDTree

    if axis not in _AXES:
        raise ValueError(f"axis must be one of {sorted(_AXES)}, got {axis!r}.")
    ax_i = _AXES[axis]
    in_plane = [i for i in range(3) if i != ax_i]

    values = np.asarray(values, dtype=float).ravel()
    if values.shape != (voxel_grid.n_voxels,):
        raise ValueError(
            f"values must have shape ({voxel_grid.n_voxels},), got {values.shape}.")
    if voxel_mask is not None:
        values = np.where(np.asarray(voxel_mask, dtype=bool), values, np.nan)
    if threshold is not None:
        if tail == 'positive':
            values = np.where(values >= threshold, values, np.nan)
        elif tail == 'negative':
            values = np.where(values <= -threshold, values, np.nan)
        elif tail == 'both':
            values = np.where(np.abs(values) >= threshold, values, np.nan)
        else:
            raise ValueError(f"tail must be 'both'/'positive'/'negative', got {tail!r}.")

    pos = voxel_grid.positions
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("No finite values left to plot (check threshold / voxel_mask).")

    slab = voxel_grid.spacing / 2.0 if slab is None else float(slab)
    fill_radius = voxel_grid.spacing if fill_radius is None else float(fill_radius)
    if slices is None:
        lo, hi = pos[finite, ax_i].min(), pos[finite, ax_i].max()
        slices = np.linspace(lo, hi, n_slices) if hi > lo else [lo]
    slices = list(np.atleast_1d(slices))

    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(values[finite]), 99)) or 1.0
    if vmin is None:
        vmin = -vmax
    cmap = cmap if callable(cmap) else plt.get_cmap(cmap or theme.DIVERGING_CMAP)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    bg = load_anatomical(background) if background is not None else None

    # One common in-plane window for every panel, from the background's own
    # extent trimmed to where there is actually anatomy. Without this each
    # panel auto-scales to its own overlay and the slices stop being
    # comparable -- and a template with a neck wastes most of the frame.
    window = _head_window(bg, in_plane) if bg is not None else None
    if window is None:
        pad = fill_radius * 2
        pts_all = pos[finite][:, in_plane]
        window = (pts_all[:, 0].min() - pad, pts_all[:, 0].max() + pad,
                  pts_all[:, 1].min() - pad, pts_all[:, 1].max() + pad)

    ncols = ncols or min(len(slices), 4)
    nrows = int(np.ceil(len(slices) / ncols))
    figsize = figsize or (3.6 * ncols, 3.9 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for panel, coord in enumerate(slices):
        a = axes_flat[panel]
        if bg is not None:
            _draw_background(a, bg, ax_i, in_plane, coord, bg_cmap)

        near = finite & (np.abs(pos[:, ax_i] - coord) <= slab)
        if near.any():
            pts = pos[near][:, in_plane]
            vals = values[near]
            # Paint each voxel as a disc of the grid's own spacing, sampled on
            # a fine lattice, so the overlay is contiguous and honest about
            # resolution instead of a scatter of dots.
            u = np.arange(pts[:, 0].min() - fill_radius, pts[:, 0].max() + fill_radius, 1.0)
            v = np.arange(pts[:, 1].min() - fill_radius, pts[:, 1].max() + fill_radius, 1.0)
            if u.size and v.size:
                gu, gv = np.meshgrid(u, v, indexing='ij')
                q = np.column_stack([gu.ravel(), gv.ravel()])
                dist, idx = cKDTree(pts).query(q)
                img = np.where(dist <= fill_radius, vals[idx], np.nan).reshape(gu.shape)
                a.imshow(img.T, origin='lower', cmap=cmap, norm=norm,
                         extent=(u[0], u[-1], v[0], v[-1]),
                         alpha=overlay_alpha, interpolation='nearest', zorder=2)
        a.set_title(f"{axis} = {coord:.0f} mm", fontsize=10)
        a.set_xlim(window[0], window[1])
        a.set_ylim(window[2], window[3])
        a.set_aspect('equal')
        if orientation_labels:
            _annotate_orientation(a, ax_i)
        a.set_xticks([]); a.set_yticks([])
        for spine in a.spines.values():
            spine.set_visible(False)

    for extra in range(len(slices), len(axes_flat)):
        axes_flat[extra].set_visible(False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.75, pad=0.02, label=colorbar_label or '')
    if title:
        fig.suptitle(title, y=0.99)
    return fig, axes


def _draw_background(a, bg, ax_i, in_plane, coord, bg_cmap):
    """Draw one anatomical slice, placed in world mm so the overlay lines up."""
    vol, affine = bg
    inv = np.linalg.inv(affine)
    centre = np.zeros(3)
    centre[ax_i] = coord
    idx = int(round((inv @ np.append(centre, 1.0))[ax_i]))
    if not (0 <= idx < vol.shape[ax_i]):
        return
    sl = np.take(vol, idx, axis=ax_i)

    # World extent of the two in-plane axes. The affines here are diagonal
    # (ICBM152 templates are axis-aligned), so a corner-to-corner mapping is
    # exact; a rotated affine would need resampling instead.
    corners = []
    for c in ([0, 0, 0], np.array(vol.shape) - 1):
        p = np.append(np.asarray(c, float), 1.0)
        corners.append((affine @ p)[:3])
    lo, hi = np.minimum(*corners), np.maximum(*corners)
    extent = (lo[in_plane[0]], hi[in_plane[0]], lo[in_plane[1]], hi[in_plane[1]])
    a.imshow(sl.T, origin='lower', cmap=bg_cmap, extent=extent,
             interpolation='bilinear', zorder=1)


def _head_window(bg, in_plane, percentile=1.0, pad=6.0):
    """Return the in-plane bounding box of the non-empty anatomy in a volume."""
    vol, affine = bg
    thresh = np.percentile(vol[vol > 0], percentile) if (vol > 0).any() else 0.0
    occupied = np.argwhere(vol > thresh)
    if not len(occupied):
        return None
    lo_idx, hi_idx = occupied.min(axis=0), occupied.max(axis=0)
    lo = (affine @ np.append(lo_idx.astype(float), 1.0))[:3]
    hi = (affine @ np.append(hi_idx.astype(float), 1.0))[:3]
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    return (lo[in_plane[0]] - pad, hi[in_plane[0]] + pad,
            lo[in_plane[1]] - pad, hi[in_plane[1]] + pad)


#: Anatomical direction at each edge of a slice panel, per slice axis, in
#: RAS world coordinates (+x right, +y anterior, +z superior). Panels are
#: drawn with the first in-plane axis horizontal and the second vertical,
#: origin lower -- so these read (left edge, right edge, bottom, top).
#: Note the axial panels put the subject's right on the right: neurological
#: convention, not radiological.
_ORIENT = {
    0: ('P', 'A', 'I', 'S'),   # sagittal slice: y horizontal, z vertical
    1: ('L', 'R', 'I', 'S'),   # coronal:        x horizontal, z vertical
    2: ('L', 'R', 'P', 'A'),   # axial:          x horizontal, y vertical
}


def _annotate_orientation(a, ax_i):
    """Mark the edges of a slice panel with their anatomical directions."""
    import matplotlib.patheffects as pe

    left, right, bottom, top = _ORIENT[ax_i]
    # Outlined glyphs: a slice panel is black at the edges and white in the
    # middle of the head, so no single ink colour stays legible on its own.
    style = dict(transform=a.transAxes, fontsize=9, color='white',
                 ha='center', va='center', fontweight='bold',
                 path_effects=[pe.withStroke(linewidth=2.0, foreground='black')])
    a.text(0.035, 0.5, left, **style)
    a.text(0.965, 0.5, right, **style)
    a.text(0.5, 0.035, bottom, **style)
    a.text(0.5, 0.965, top, **style)
