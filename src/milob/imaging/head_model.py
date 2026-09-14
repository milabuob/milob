import numpy as np


def _parse_obj(lines):
    """Parse OBJ text lines into vertex and face arrays."""
    vertices, faces = [], []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'v':
            vertices.append([float(x) for x in parts[1:4]])
        elif parts[0] == 'f':
            # handles 'f v1 v2 v3' and 'f v1/vt1/vn1 ...' forms
            faces.append([int(x.split('/')[0]) - 1 for x in parts[1:4]])
    return np.array(vertices, dtype=float), np.array(faces, dtype=int)


def _load_obj(path: str):
    """Read an OBJ file into vertex and face arrays."""
    with open(path) as f:
        return _parse_obj(f)


class Surface:
    """
    A triangular surface mesh in MNI space, for 3-D visualisation.

    Base class for :class:`BrainSurface` and :class:`HeadSurface`. These are
    visualisation meshes, not the segmented volume that image reconstruction
    would need.

    Parameters
    ----------
    vertices : np.ndarray
        Shape (n, 3), vertex positions in mm.
    faces : np.ndarray
        Shape (m, 3), zero-based triangular face indices.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces    = np.asarray(faces,    dtype=int)

    @classmethod
    def from_file(cls, path: str):
        """
        Load a surface from an OBJ file.

        Parameters
        ----------
        path : str
            Path to the mesh. It must be in mm and in the same frame as the
            reference landmarks.

        Returns
        -------
        Surface
        """
        vertices, faces = _load_obj(path)
        return cls(vertices=vertices, faces=faces)

    def __repr__(self):
        return (f"<{type(self).__name__} | {len(self.vertices):,} vertices, "
                f"{len(self.faces):,} faces>")


class BrainSurface(Surface):
    """
    A cortical surface mesh for 3-D visualisation.

    Optodes sit on the scalp, roughly 8 to 19 mm above this surface over the
    vault and further at the fiducials, so a probe rendered against it appears
    to float. Use :class:`HeadSurface` to show where a probe is, and this to
    show where reconstructed signal is thought to originate.

    Build one with :meth:`auto`, :meth:`from_nilearn` or
    :meth:`Surface.from_file`.
    """

    # ------------------------------------------------------------------ #
    # Constructors                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_nilearn(cls, resolution: str = 'fsaverage3'):
        """
        Load an fsaverage pial surface.

        Parameters
        ----------
        resolution : {'fsaverage3', 'fsaverage4', 'fsaverage5'}
            Mesh density, from about 640 to about 10,000 vertices per hemisphere.
            The coarsest renders acceptably in matplotlib; the finest suits
            PyVista.

        Returns
        -------
        BrainSurface

        Raises
        ------
        ImportError
            If nilearn is not installed.
        """
        try:
            from nilearn import datasets
            from nilearn.surface import load_surf_mesh
        except ImportError as exc:
            raise ImportError(
                "nilearn is required to auto-load a brain surface. "
                "Install it with:  pip install nilearn\n"
                "Alternatively pass a BrainSurface.from_file(path) or "
                "pass surface='head' to render the MNI152 scalp instead."
            ) from exc

        fsaverage = datasets.fetch_surf_fsaverage(resolution)
        coords_l, faces_l = load_surf_mesh(fsaverage.pial_left)
        coords_r, faces_r = load_surf_mesh(fsaverage.pial_right)

        n_left   = len(coords_l)
        vertices = np.vstack([coords_l, coords_r])
        faces    = np.vstack([faces_l, faces_r + n_left])
        return cls(vertices=vertices, faces=faces)

    @classmethod
    def auto(cls, resolution: str = 'fsaverage3'):
        """
        Load an fsaverage surface if nilearn is available.

        Parameters
        ----------
        resolution : str
            Mesh density, as for :meth:`from_nilearn`.

        Returns
        -------
        BrainSurface or None
            None when nilearn is not installed, with a warning explaining why.
        """
        try:
            return cls.from_nilearn(resolution)
        except ImportError:
            import warnings
            warnings.warn(
                "nilearn is not installed — falling back to a head ellipsoid. "
                "Install nilearn (pip install nilearn) for a real brain surface, "
                "or supply a BrainSurface.from_file(path).",
                stacklevel=3,
            )
            return None


def _uv_sphere(centre, half_axes, n_pol: int = 40, n_azi: int = 60):
    """
    Build a triangulated ellipsoid.

    Poles are collapsed to single points so that the mesh stays manifold.

    Parameters
    ----------
    centre : array-like
        Centre in mm.
    half_axes : array-like
        Semi-axis lengths in mm.
    n_pol, n_azi : int
        Polar and azimuthal subdivision counts.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Vertices and faces.
    """
    centre    = np.asarray(centre, dtype=float)
    half_axes = np.asarray(half_axes, dtype=float)

    theta = np.linspace(0.0, np.pi, n_pol)
    phi   = np.linspace(0.0, 2 * np.pi, n_azi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    grid = centre + half_axes * np.stack(
        [np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], axis=-1
    )

    verts = [grid[0, 0]]
    ring = {i: [1 + (i - 1) * n_azi + j for j in range(n_azi)]
            for i in range(1, n_pol - 1)}
    for i in range(1, n_pol - 1):
        verts.extend(grid[i])
    pole_bot = len(verts)
    verts.append(grid[-1, 0])

    faces = [[0, ring[1][(j + 1) % n_azi], ring[1][j]] for j in range(n_azi)]
    for i in range(1, n_pol - 2):
        for j in range(n_azi):
            j2 = (j + 1) % n_azi
            faces.append([ring[i][j], ring[i][j2], ring[i + 1][j2]])
            faces.append([ring[i][j], ring[i + 1][j2], ring[i + 1][j]])
    faces += [[pole_bot, ring[n_pol - 2][j], ring[n_pol - 2][(j + 1) % n_azi]]
              for j in range(n_azi)]

    return np.array(verts, dtype=float), np.array(faces, dtype=int)


class HeadSurface(Surface):
    """
    A scalp surface mesh for 3-D visualisation.

    The surface optodes sit on, so a probe rendered against it lies where it
    belongs. It is also the appropriate backdrop for channel-space results,
    which are boundary measurements carrying no depth resolution;
    :class:`BrainSurface` suits reconstructed voxel-level quantities instead.

    Build one with :meth:`from_template`, :meth:`ellipsoid`, :meth:`auto` or
    :meth:`Surface.from_file`.
    """

    #: Shipped mesh, relative to this package's ``data/`` directory.
    TEMPLATE = 'mni152_scalp.obj'

    @classmethod
    def from_template(cls):
        """
        Load the MNI152 scalp surface bundled with the package.

        Derived from the ICBM152 non-linear asymmetric 2009a whole-head template
        by thresholding to an outer-skin mask and ray-casting a star-shaped
        surface, giving about 4,200 vertices. It shares the frame of the default
        reference landmarks, which fall 3.5 to 4.2 mm from it; the residual at the
        preauricular points is anatomical, those being depressions a star-shaped
        surface bridges over.

        Returns
        -------
        HeadSurface

        Raises
        ------
        FileNotFoundError
            If the bundled mesh is missing.
        """
        from importlib.resources import files
        path = files('milob.imaging') / 'data' / cls.TEMPLATE
        if not path.is_file():
            raise FileNotFoundError(
                f"milob's packaged scalp mesh ({cls.TEMPLATE}) was not found at "
                f"{path}. This usually means an incomplete install; reinstall "
                f"milob, or pass your own via HeadSurface.from_file(path)."
            )
        with path.open() as fh:
            return cls(*_parse_obj(fh))

    @classmethod
    def ellipsoid(cls, coreg=None):
        """
        Build a crude ellipsoid standing in for the head.

        Given a coregistration, the ellipsoid is fitted to that probe's optode
        cloud, which resembles a head only when the probe covers most of one; a
        patch over a single region yields a ball centred on that patch. Without
        one, a canonical head ellipsoid is used.

        Parameters
        ----------
        coreg : Coregistration, optional
            Fitted coregistration whose optode positions define the fit.

        Returns
        -------
        HeadSurface
        """
        if coreg is None:
            centre, half = np.array([0.0, -18.0, 15.0]), np.array([80.0, 100.0, 85.0])
        else:
            pts    = np.vstack([coreg.mni_s_pos, coreg.mni_d_pos])
            centre = pts.mean(axis=0)
            half   = np.maximum(
                np.percentile(np.abs(pts - centre), 90, axis=0) * 1.2, 40.0
            )
        return cls(*_uv_sphere(centre, half))

    @classmethod
    def auto(cls, coreg=None):
        """
        Load the bundled template, falling back to an ellipsoid.

        Never raises; warns when it falls back.

        Parameters
        ----------
        coreg : Coregistration, optional
            Used for the fallback ellipsoid.

        Returns
        -------
        HeadSurface
        """
        try:
            return cls.from_template()
        except FileNotFoundError as exc:
            import warnings
            warnings.warn(f"{exc} Falling back to a head ellipsoid.", stacklevel=3)
            return cls.ellipsoid(coreg)


def _cone(apex, base_centre, radius, n_side: int = 20):
    """
    Build a closed cone from an apex, a base rim and a base centre.

    Parameters
    ----------
    apex : array-like
        Apex position in mm.
    base_centre : array-like
        Base centre in mm.
    radius : float
        Base radius in mm.
    n_side : int
        Number of rim vertices.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Vertices and faces.
    """
    apex = np.asarray(apex, float); base_centre = np.asarray(base_centre, float)
    axis = apex - base_centre
    axis = axis / np.linalg.norm(axis)
    # any two directions spanning the base plane
    tmp = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, tmp); u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    ang = np.linspace(0.0, 2 * np.pi, n_side, endpoint=False)
    rim = base_centre + radius * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v)
    verts = np.vstack([apex, base_centre, rim])
    faces = []
    for j in range(n_side):
        a, b = 2 + j, 2 + (j + 1) % n_side
        faces.append([0, a, b])       # side
        faces.append([1, b, a])       # base
    return verts, np.array(faces, dtype=int)


def landmark_cues(landmarks=None, surface=None, nose_length=26.0,
                  nose_radius=13.0, ear_axes=(9.0, 16.0, 19.0),
                  centre=(0.0, -18.0, 15.0)):
    """
    Build schematic nose and ear cues for orientation.

    These are rendering cues rather than anatomy: the template is defaced, so
    neither feature can be recovered from it. Drawing them anchored to the
    fiducials answers which way the head faces, and doubles as a coarse check
    on the coregistration.

    Parameters
    ----------
    landmarks : dict, optional
        Landmarks as {label: (x, y, z)} in mm. Defaults to the MNI152
        fiducials. The nasion gives the nose and the preauricular points the
        ears; whichever are present are drawn.
    surface : Surface, optional
        Head mesh to attach the cues to. Each landmark is pushed along its own
        radius until it meets the mesh, so the cues sit on the scalp.
    nose_length, nose_radius : float
        Cone dimensions in mm.
    ear_axes : tuple of (float, float, float)
        Ellipsoid semi-axes in mm, thin in x so each ear reads as a flap.
    centre : tuple of (float, float, float)
        Head centre defining the nose's outward direction.

    Returns
    -------
    Surface or None
        None when no usable landmark was given.
    """
    from .coregistration import MNI152_LANDMARKS

    landmarks = MNI152_LANDMARKS if landmarks is None else landmarks
    centre = np.asarray(centre, dtype=float)

    def on_surface(point):
        """Slide a point along its own radius onto the surface, if one was given."""
        if surface is None:
            return point
        ray = point - centre
        length = np.linalg.norm(ray)
        if length == 0:
            return point
        ray = ray / length
        # radial distance of each vertex, weighted toward those the ray points at
        rel = surface.vertices - centre
        norms = np.linalg.norm(rel, axis=1)
        cos = (rel @ ray) / np.maximum(norms, 1e-9)
        near = cos > np.cos(np.radians(12.0))
        if not near.any():
            near = cos >= np.sort(cos)[-8]
        return centre + ray * np.median(norms[near])

    parts = []

    nas = landmarks.get('NAS')
    if nas is not None:
        nas = on_surface(np.asarray(nas, dtype=float))
        out = nas - centre
        # Damp the downward tilt: the raw radius to the nasion dips ~20 deg, and
        # a nose angled into the jaw reads as a growth rather than a nose.
        out[2] *= 0.35
        out /= np.linalg.norm(out)
        # base sunk into the scalp so the cone merges with it
        parts.append(_cone(nas + out * nose_length, nas - out * 10.0, nose_radius))

    for label in ('LPA', 'RPA'):
        pos = landmarks.get(label)
        if pos is None:
            continue
        pos = on_surface(np.asarray(pos, dtype=float))
        # centred just proud of the scalp, so roughly half the ellipsoid shows
        outward = 2.0 if pos[0] > 0 else -2.0
        parts.append(_uv_sphere(pos + np.array([outward, 0.0, 0.0]),
                                np.asarray(ear_axes, float), n_pol=16, n_azi=24))

    if not parts:
        return None

    verts, faces, offset = [], [], 0
    for v, f in parts:
        verts.append(v); faces.append(f + offset); offset += len(v)
    return Surface(np.vstack(verts), np.vstack(faces))
