"""
International 10-20 and 10-10 scalp positions, and optode labelling against
them.

Builds the standard position system from its definition -- proportional
distances along scalp arcs between anatomical fiducials -- evaluated on a
real scalp mesh, rather than from a table of coordinates.

The labels produced are scalp positions: they say where an optode sits on
the head, not what cortex it interrogates. They are also derived
quantities carrying the coregistration's error, so they belong alongside
an optode's own label rather than in place of it, not least because two
optodes can share a nearest position.

References
----------
Jurcak, V., Tsuzuki, D., & Dan, I. (2007). NeuroImage, 34(4), 1600-1611.
"""

import numpy as np

#: Positions along the sagittal curve, as a fraction of the Nz -> Iz arc.
_MIDLINE = [
    ('Nz', 0.0), ('Fpz', 0.1), ('AFz', 0.2), ('Fz', 0.3), ('FCz', 0.4),
    ('Cz', 0.5), ('CPz', 0.6), ('Pz', 0.7), ('POz', 0.8), ('Oz', 0.9),
    ('Iz', 1.0),
]

#: The circumference, as two quarters either side of the ear, each divided
#: into five equal parts: (left label, right label) at 1/5 .. 4/5 of the
#: quarter. Dividing each quarter separately, rather than taking 10% steps of
#: the whole half-arc, is what keeps the construction well defined when the
#: front and back quarters do not come out exactly equal -- which on a real
#: head they do not (see ``build_scalp_positions``'s circumference_balance
#: diagnostic). The two readings coincide when they are equal.
_CIRC_FRONT = [('Fp1', 'Fp2'), ('AF7', 'AF8'), ('F7', 'F8'), ('FT7', 'FT8')]
_CIRC_BACK = [('TP7', 'TP8'), ('P7', 'P8'), ('PO7', 'PO8'), ('O1', 'O2')]

#: Each row: (left anchor, midline position, right anchor, left labels, right
#: labels). The row is the arc from anchor to anchor through the midline
#: position, divided into eight equal parts; it is traced and divided as two
#: halves about the midline position, each into four, for the same reason the
#: circumference is done by quarters -- the halves are not exactly equal on a
#: real head, and dividing each on its own length keeps every label on the
#: arc that defines it. Left labels run outward-to-inward (F5, F3, F1), right
#: labels inward-to-outward (F2, F4, F6).
_ROWS = [
    ('AF7', 'AFz', 'AF8', ['AF5', 'AF3', 'AF1'], ['AF2', 'AF4', 'AF6']),
    ('F7',  'Fz',  'F8',  ['F5',  'F3',  'F1'],  ['F2',  'F4',  'F6']),
    ('FT7', 'FCz', 'FT8', ['FC5', 'FC3', 'FC1'], ['FC2', 'FC4', 'FC6']),
    ('T7',  'Cz',  'T8',  ['C5',  'C3',  'C1'],  ['C2',  'C4',  'C6']),
    ('TP7', 'CPz', 'TP8', ['CP5', 'CP3', 'CP1'], ['CP2', 'CP4', 'CP6']),
    ('P7',  'Pz',  'P8',  ['P5',  'P3',  'P1'],  ['P2',  'P4',  'P6']),
    ('PO7', 'POz', 'PO8', ['PO5', 'PO3', 'PO1'], ['PO2', 'PO4', 'PO6']),
]

#: The 19 scalp positions of the classical 10-20 system (the ear electrodes
#: A1/A2 are not scalp positions and are not produced).
_TEN_TWENTY = {
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2',
}

#: Legacy 10-20 spellings, accepted on input and offered as an output style.
_LEGACY_NAMES = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}

SYSTEMS = ('10-20', '10-10')

#: How far a traced arc may land from the endpoints it was asked for, in mm.
#: Set by mesh resolution -- the template scalp's vertices are ~3 mm apart, so
#: a ray cast at an endpoint's own bearing lands within about that -- not by
#: any tolerance on the anatomy.
_ENDPOINT_TOLERANCE_MM = 5.0

#: Built position sets for the default (template) construction, keyed by system.
_POSITION_CACHE = {}


# ---------------------------------------------------------------------------
# Ray casting on a star-shaped surface
# ---------------------------------------------------------------------------

def _ray_surface_hits(origin, directions, vertices, faces):
    """
    Cast rays from an interior origin and return where they leave the surface.

    Uses a vectorised Moller-Trumbore intersection. The scalp mesh is
    star-shaped about an interior point, so a ray exits once; where numerical
    slivers give several hits the outermost is taken.

    Parameters
    ----------
    origin : np.ndarray
        Shape (3,), a point inside the head.
    directions : np.ndarray
        Shape (n, 3), ray directions, not necessarily normalised.
    vertices, faces : np.ndarray
        Surface mesh.

    Returns
    -------
    np.ndarray
        Shape (n, 3), intersection points. Rays that miss give NaN.
    """
    origin = np.asarray(origin, dtype=float)
    directions = np.atleast_2d(np.asarray(directions, dtype=float))
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    v0 = vertices[faces[:, 0]]
    edge1 = vertices[faces[:, 1]] - v0
    edge2 = vertices[faces[:, 2]] - v0
    to_origin = origin - v0                                   # (n_faces, 3)

    hits = np.full((len(directions), 3), np.nan)

    for i, direction in enumerate(directions):
        pvec = np.cross(direction, edge2)
        det = np.einsum('ij,ij->i', edge1, pvec)

        parallel = np.abs(det) < 1e-12
        inv_det = np.where(parallel, 0.0, 1.0 / np.where(parallel, 1.0, det))

        u = np.einsum('ij,ij->i', to_origin, pvec) * inv_det
        qvec = np.cross(to_origin, edge1)
        v = np.dot(qvec, direction) * inv_det
        t = np.einsum('ij,ij->i', edge2, qvec) * inv_det

        inside = (~parallel) & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-6)
        if inside.any():
            hits[i] = origin + direction * t[inside].max()

    return hits


def _plane_basis(normal):
    """
    Return an orthonormal pair spanning the plane with a given normal.

    Parameters
    ----------
    normal : array-like
        Plane normal.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Two orthonormal vectors spanning the plane.
    """
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, normal)) > 0.9:
        seed = np.array([0.0, 0.0, 1.0])

    u = seed - np.dot(seed, normal) * normal
    u /= np.linalg.norm(u)
    w = np.cross(normal, u)

    return u, w


def _arc_points(surface, origin, start, end, normal=None, n_samples=361,
                prefer='vertex'):
    """Sample the scalp arc between two points on a given plane, choosing which of the two possible sweeps to return."""
    origin = np.asarray(origin, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    if normal is None:
        normal = np.cross(start - origin, end - origin)

    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    u, w = _plane_basis(normal)

    # The contour must be traced in the arc's own plane, not in a parallel one
    # through the interior origin: ``start`` and ``end`` lie on the former, and
    # sweeping the latter would return points from a different section of the
    # head entirely. ``start`` is on the plane by construction, so drop the
    # origin onto it along the normal.
    origin = origin - np.dot(origin - start, normal) * normal

    def angle_of(point):
        offset = np.asarray(point, dtype=float) - origin
        return np.arctan2(np.dot(offset, w), np.dot(offset, u))

    theta_start = angle_of(start)
    theta_end = angle_of(end)

    # The two ways round: increasing angle, or decreasing.
    forward = (theta_end - theta_start) % (2 * np.pi)
    sweeps = (forward, forward - 2 * np.pi)

    candidates = []
    for sweep in sweeps:
        thetas = theta_start + sweep * np.linspace(0.0, 1.0, n_samples)
        directions = (np.cos(thetas)[:, None] * u + np.sin(thetas)[:, None] * w)
        points = _ray_surface_hits(origin, directions, surface.vertices, surface.faces)

        # A sweep that leaves the mesh (through the neck opening, say) is not
        # a scalp arc at all -- drop it rather than measuring along a gap.
        if np.isnan(points).any():
            continue

        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(steps)])
        candidates.append((points, cumulative))

    if not candidates:
        raise ValueError(
            "Could not trace a scalp arc between the given points: every ray "
            "sweep left the surface. This usually means the mesh has a hole, "
            "or the interior origin used for ray casting is outside it."
        )

    if prefer == 'short':
        points, cumulative = min(candidates, key=lambda c: c[1][-1])
    else:
        points, cumulative = max(candidates, key=lambda c: c[0][n_samples // 2, 2])

    # The traced contour should start and end where it was asked to. A large
    # error means the endpoints are not on this plane's section of the scalp,
    # which is a construction bug rather than a property of the head -- catch
    # it here instead of returning a plausible-looking arc through nowhere.
    drift = max(np.linalg.norm(points[0] - start),
                np.linalg.norm(points[-1] - end))
    if drift > _ENDPOINT_TOLERANCE_MM:
        raise ValueError(
            f"Traced scalp arc misses its endpoints by {drift:.1f} mm "
            f"(tolerance {_ENDPOINT_TOLERANCE_MM} mm). The endpoints are not "
            "on this plane's section of the surface."
        )

    return points, cumulative


def _at_fractions(points, cumulative, fractions):
    """
    Interpolate arc points at given fractions of the total arc length.

    Parameters
    ----------
    points : np.ndarray
        Arc points.
    cumulative : np.ndarray
        Cumulative arc length along them.
    fractions : array-like
        Fractions of the total length, in [0, 1].

    Returns
    -------
    np.ndarray
        One point per fraction.
    """
    targets = np.asarray(fractions, dtype=float) * cumulative[-1]
    return np.column_stack([
        np.interp(targets, cumulative, points[:, axis]) for axis in range(3)
    ])


# ---------------------------------------------------------------------------
# Building the position set
# ---------------------------------------------------------------------------

def _resolve_fiducials(surface, fiducials, origin):
    """Fill in any missing reference fiducials and project all four onto the scalp."""
    from .coregistration import MNI152_LANDMARKS

    resolved = {
        'Nz': MNI152_LANDMARKS['NAS'],
        'LPA': MNI152_LANDMARKS['LPA'],
        'RPA': MNI152_LANDMARKS['RPA'],
    }

    if 'Iz' not in (fiducials or {}):
        midsagittal = surface.vertices[np.abs(surface.vertices[:, 0]) < 3.0]
        if len(midsagittal) == 0:
            raise ValueError(
                "Cannot locate Iz: the surface has no mid-sagittal vertices. "
                "Pass it explicitly via fiducials={'Iz': (x, y, z)}."
            )
        resolved['Iz'] = midsagittal[midsagittal[:, 1].argmin()]

    resolved.update({k: np.asarray(v, dtype=float)
                     for k, v in (fiducials or {}).items()})

    # The reference landmarks sit a few mm off the mesh; put them on it, so
    # every arc endpoint lies on the surface the arcs are measured along.
    labels = list(resolved)
    directions = np.array([resolved[label] - origin for label in labels])
    projected = _ray_surface_hits(origin, directions, surface.vertices, surface.faces)

    for label, point in zip(labels, projected):
        if not np.isnan(point).any():
            resolved[label] = point

    return resolved


def build_scalp_positions(surface=None, *, system='10-10', fiducials=None,
                          origin=None, return_diagnostics=False):
    """
    Construct the 10-20 or 10-10 positions on a scalp surface.

    The sagittal curve runs between the nasion and inion over the vertex, and
    the coronal curve between the preauricular points through the vertex.
    Positions sit at proportional intervals along these and along the
    circumference, with each row divided into eight equal parts.

    Parameters
    ----------
    surface : HeadSurface, optional
        Scalp to measure along. Defaults to the bundled MNI152 template, which
        puts the result in the same frame as a coregistered probe.
    system : {'10-10', '10-20'}
        Which position set to return. '10-20' is the 19-position subset.
    fiducials : dict, optional
        Overrides for any of the four reference fiducials.
    origin : array-like, optional
        Interior point the arcs are ray-cast from. Defaults to the surface
        centroid.
    return_diagnostics : bool
        Also return arc lengths, the discrepancy between the sagittal and
        coronal readings of the vertex, and the fiducials used.

    Returns
    -------
    positions : dict of {str: np.ndarray}
        Position name to coordinate, in the surface's own frame.
    diagnostics : dict, optional
        Returned when ``return_diagnostics`` is True.
    """
    if system not in SYSTEMS:
        raise ValueError(f"Unknown system {system!r}; choose from {SYSTEMS}.")

    # The default construction is deterministic and takes a few seconds of ray
    # casting, so a study labelling many probes gets it once. Only the fully
    # default call is cached; any custom surface, fiducials or origin is built
    # fresh, since none of those is safely hashable as a cache key.
    default = surface is None and not fiducials and origin is None
    if default and system in _POSITION_CACHE:
        cached, cached_diagnostics = _POSITION_CACHE[system]
        positions = {label: point.copy() for label, point in cached.items()}
        return (positions, cached_diagnostics) if return_diagnostics else positions

    if surface is None:
        from .head_model import HeadSurface
        surface = HeadSurface.from_template()

    if origin is None:
        origin = surface.vertices.mean(axis=0)
    origin = np.asarray(origin, dtype=float)

    fids = _resolve_fiducials(surface, fiducials, origin)
    positions = {}

    # 1. Sagittal curve, Nz -> Iz over the vertex.
    sag_points, sag_length = _arc_points(
        surface, origin, fids['Nz'], fids['Iz']
    )
    midline = _at_fractions(sag_points, sag_length,
                            [fraction for _, fraction in _MIDLINE])
    for (label, _), point in zip(_MIDLINE, midline):
        positions[label] = point

    # 2. Coronal curve, LPA -> RPA through Cz.
    coronal_normal = np.cross(fids['LPA'] - positions['Cz'],
                              fids['RPA'] - positions['Cz'])
    cor_points, cor_length = _arc_points(
        surface, origin, fids['LPA'], fids['RPA'], coronal_normal
    )

    # 3. Circumference through Fpz, T7, Oz, T8. T7/T8 come from the coronal
    #    curve, which is where the system defines them.
    temporals = _at_fractions(cor_points, cor_length, [0.1, 0.9])
    positions['T7'], positions['T8'] = temporals

    # Fpz, T7, Oz and T8 are nowhere near coplanar on a real head, so there is
    # no single circumference plane to fit -- and forcing one puts its section
    # of the scalp tens of millimetres from the points it is meant to join.
    # Each quarter therefore gets the plane through its own two endpoints and
    # the head interior, which is where a tape would lie between them.
    fifths = np.arange(1, 5) / 5.0
    quarter_lengths = {}

    for index, anchor in ((0, positions['T7']), (1, positions['T8'])):
        # Trace each quarter separately, via the ear, so the sweep is pinned to
        # the correct side of the head rather than inferred -- and so each
        # quarter is divided on its own measured length.
        front, front_len = _arc_points(surface, origin, positions['Fpz'], anchor,
                                       prefer='short')
        back, back_len = _arc_points(surface, origin, anchor, positions['Oz'],
                                     prefer='short')

        for labels, point in zip(_CIRC_FRONT, _at_fractions(front, front_len, fifths)):
            positions[labels[index]] = point
        for labels, point in zip(_CIRC_BACK, _at_fractions(back, back_len, fifths)):
            positions[labels[index]] = point

        quarter_lengths['left' if index == 0 else 'right'] = (
            float(front_len[-1]), float(back_len[-1])
        )

    # 4. Rows: each anchor-to-anchor arc through its midline position, traced
    #    as two halves about that position so the sweep is pinned to the arc
    #    the row is defined by, then each half divided into four.
    quarters = np.arange(1, 4) / 4.0
    for left, centre, right, left_labels, right_labels in _ROWS:
        row_normal = np.cross(positions[left] - positions[centre],
                              positions[right] - positions[centre])

        for anchor, labels in ((left, left_labels), (right, right_labels)):
            start, end = ((positions[anchor], positions[centre])
                          if labels is left_labels
                          else (positions[centre], positions[anchor]))
            half, half_len = _arc_points(surface, origin, start, end,
                                         row_normal, prefer='short')
            for label, point in zip(labels, _at_fractions(half, half_len, quarters)):
                positions[label] = point

    if system == '10-20':
        positions = {label: point for label, point in positions.items()
                     if label in _TEN_TWENTY}

    coronal_midpoint = _at_fractions(cor_points, cor_length, [0.5])[0]
    diagnostics = {
        'fiducials': fids,
        'sagittal_arc_mm': float(sag_length[-1]),
        'coronal_arc_mm': float(cor_length[-1]),
        # How far apart the two definitions of Cz land. A few mm is normal;
        # a large value means the fiducials are inconsistent with the mesh.
        'cz_discrepancy_mm': float(
            np.linalg.norm(coronal_midpoint - positions['Cz'])
        ),
        # Front quarter / back quarter of each circumference half. The system
        # assumes 1.0; a departure means the coronal curve puts T7/T8 off the
        # circumference's own midpoint, and the two readings of F7, P7 and
        # their mirrors would disagree by roughly that much.
        'circumference_balance': {
            side: front / back for side, (front, back) in quarter_lengths.items()
        },
        'circumference_quarters_mm': quarter_lengths,
        'origin': origin,
    }

    if default:
        # Store a private copy, so a caller mutating what it is handed back
        # cannot corrupt what the next caller receives.
        _POSITION_CACHE[system] = (
            {label: point.copy() for label, point in positions.items()},
            diagnostics,
        )

    return (positions, diagnostics) if return_diagnostics else positions


# ---------------------------------------------------------------------------
# Labelling a probe
# ---------------------------------------------------------------------------

def label_optodes(probe, *, system='10-10', reference_landmarks=None,
                  surface=None, naming='modern', positions=None):
    """
    Name each optode by the nearest standard scalp position.

    The probe is coregistered to MNI space using its own digitised landmarks,
    then each optode matched to the nearest constructed position. The distance
    is returned alongside, since it is what makes the label judgeable: the
    grid's own spacing is only 25 to 35 mm.

    Parameters
    ----------
    probe : Probe
        Must carry digitised landmarks and their labels.
    system : {'10-10', '10-20'}
        Which position set to match against.
    reference_landmarks : dict, optional
        Forwarded to the probe's coregistration.
    surface : HeadSurface, optional
        Forwarded to :func:`build_scalp_positions`.
    naming : {'modern', 'legacy'}
        'legacy' returns the older T3, T4, T5 and T6 names.
    positions : dict, optional
        A prebuilt position set, to avoid reconstructing it per probe.

    Returns
    -------
    pandas.DataFrame
        Columns name, type, position, distance_mm and the optode's MNI
        coordinates, ordered sources first.

    Raises
    ------
    ValueError
        If the probe carries no landmarks to coregister with.
    """
    import pandas as pd

    if probe.landmarks is None or probe.landmark_labels is None:
        raise ValueError(
            "10-20 labelling needs digitised anatomical landmarks: this probe "
            "has none, so there is nothing to coregister to MNI space. Load a "
            "SNIRF file carrying digpts, or set landmark_pos/landmark_labels "
            "on the Probe."
        )

    if positions is None:
        positions = build_scalp_positions(surface, system=system)

    coreg = probe.coreg(reference_landmarks=reference_landmarks)

    labels = list(positions)
    grid = np.array([positions[label] for label in labels])

    rows = []
    for optode_type, names, mni in (
        ('source', probe.source_labels, coreg.mni_s_pos),
        ('detector', probe.detector_labels, coreg.mni_d_pos),
    ):
        for name, point in zip(names, mni):
            distances = np.linalg.norm(grid - point, axis=1)
            nearest = int(distances.argmin())
            position = labels[nearest]

            if naming == 'legacy':
                position = _LEGACY_NAMES.get(position, position)

            rows.append({
                'name': str(name),
                'type': optode_type,
                'position': position,
                'distance_mm': float(distances[nearest]),
                'mni_x': float(point[0]),
                'mni_y': float(point[1]),
                'mni_z': float(point[2]),
            })

    return pd.DataFrame(rows)
