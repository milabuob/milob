import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree


def fit_boundary_plane(optode_positions, inward_point=None):
    """
    Fit a plane to a set of points by least squares.

    Parameters
    ----------
    optode_positions : np.ndarray
        Shape (n, 3), points to fit in mm, typically the source and detector
        positions.
    inward_point : array-like, optional
        A point known to lie on the tissue side, used only to orient the
        normal. Defaults to the coordinate origin, which lies inside the head
        in MNI space.

    Returns
    -------
    boundary_point : np.ndarray
        Shape (3,), the plane's centroid in mm.
    boundary_normal : np.ndarray
        Shape (3,), unit normal oriented toward ``inward_point``.
    """
    pts = np.asarray(optode_positions, dtype=float)
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid)
    normal = Vt[-1]
    normal = normal / np.linalg.norm(normal)

    ref = np.zeros(3) if inward_point is None else np.asarray(inward_point, dtype=float)
    if np.dot(normal, ref - centroid) < 0:
        normal = -normal
    return centroid, normal


class VoxelGrid:
    """
    A volumetric lattice serving as the spatial domain for voxel-indexed work.

    Holds positions only; optical properties are supplied separately.

    Parameters
    ----------
    positions : np.ndarray
        Shape (n_voxels, 3), voxel centres in mm. In MNI space when built with
        a coregistration, otherwise in the probe's own frame.
    spacing : float
        Isotropic voxel pitch in mm.
    boundary_point, boundary_normal : np.ndarray, optional
        The flat plane each voxel's depth is measured against. Both may be
        None when ``depths`` is supplied instead, which is the case for a
        montage no single plane describes.
    grid_shape : tuple of int, optional
        Shape of the full lattice before pruning, so a flat voxel list can be
        reshaped back to a volume.
    mask : np.ndarray of bool, optional
        Which points of the full lattice survived pruning. Its True entries
        stay aligned, in order, with ``positions``.
    origin : np.ndarray, optional
        Lower corner of the full lattice in mm. With ``grid_shape`` it
        reconstructs the full axis vectors.
    depths : np.ndarray, optional
        Per-voxel depth below the tissue surface in mm, measured against a
        segmented head rather than a plane. Takes precedence over the plane
        projection, and must stay aligned with ``positions``.
    """

    def __init__(self, positions, spacing, boundary_point=None, boundary_normal=None,
                 grid_shape=None, mask=None, origin=None, depths=None):
        self.positions = np.asarray(positions, dtype=float)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(
                f"positions must have shape (n_voxels, 3), got {self.positions.shape}."
            )
        self.spacing = float(spacing)
        if boundary_point is None:
            self.boundary_point = None
        else:
            self.boundary_point = np.asarray(boundary_point, dtype=float)
        if boundary_normal is None:
            self.boundary_normal = None
        else:
            boundary_normal = np.asarray(boundary_normal, dtype=float)
            self.boundary_normal = boundary_normal / np.linalg.norm(boundary_normal)
        if depths is None:
            self._depths = None
        else:
            self._depths = np.asarray(depths, dtype=float)
            if self._depths.shape != (self.positions.shape[0],):
                raise ValueError(
                    f"depths must have shape (n_voxels,) = ({self.positions.shape[0]},), "
                    f"got {self._depths.shape}."
                )
        if self._depths is None and (self.boundary_point is None
                                      or self.boundary_normal is None):
            raise ValueError(
                "VoxelGrid needs a way to measure voxel depth: either a boundary "
                "plane (boundary_point + boundary_normal) or per-voxel `depths`. "
                "Both are None."
            )
        self.grid_shape = grid_shape
        self.mask = mask
        self.origin = None if origin is None else np.asarray(origin, dtype=float)

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def n_voxels(self):
        return self.positions.shape[0]

    @property
    def voxel_volume_mm3(self):
        return self.spacing ** 3

    @property
    def voxel_volume_cm3(self):
        """Volume of one voxel, in cubic centimetres."""
        return self.voxel_volume_mm3 / 1000.0

    def depths_mm(self):
        """
        Depth of each voxel below the tissue surface, in mm.

        Returns the anatomical depths when they were supplied, otherwise the
        perpendicular distance below the boundary plane, which is negative for
        voxels on the air side.

        Returns
        -------
        np.ndarray
            Shape (n_voxels,).
        """
        if self._depths is not None:
            return self._depths
        return (self.positions - self.boundary_point) @ self.boundary_normal

    def depths_cm(self):
        """Depth of each voxel below the tissue surface, in cm."""
        return self.depths_mm() / 10.0

    def full_axes(self):
        """
        Reconstruct the full lattice's axis vectors before pruning.

        Returns
        -------
        list of np.ndarray
            The x, y and z axis vectors in mm, as a solver's interpolation grid
            expects.
        """
        if self.origin is None or self.grid_shape is None:
            raise ValueError(
                "full_axes() requires both origin and grid_shape, set by "
                "from_probe_fov() -- not available on a hand-built VoxelGrid."
            )
        return [self.origin[k] + self.spacing * np.arange(self.grid_shape[k])
                for k in range(3)]

    # ------------------------------------------------------------------ #
    # Construction                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_probe_fov(cls, probe, coreg=None, spacing=10.0, lateral_margin=15.0,
                        depth_range=(0.0, 30.0)):
        """
        Build a grid covering a field of view around the probe.

        Restricts a lattice inside the probe's bounding box to voxels within a
        depth range below the fitted boundary plane, then drops those too far
        laterally from every channel midpoint to carry sensitivity.

        Parameters
        ----------
        probe : Probe
            Must carry a channel configuration.
        coreg : Coregistration, optional
            When given, positions are built in MNI space. Otherwise they stay in
            the probe's own frame.
        spacing : float
            Isotropic voxel pitch in mm.
        lateral_margin : float
            Extra margin in mm around the channel footprint.
        depth_range : tuple of (float, float)
            Minimum and maximum depth in mm below the boundary plane to keep.

        Returns
        -------
        VoxelGrid
        """
        s_pos = probe.s_pos.astype(float)
        d_pos = probe.d_pos.astype(float)
        distances = np.asarray(probe.distances, dtype=float)
        if probe.lengthUnit == 'cm':
            s_pos = s_pos * 10.0
            d_pos = d_pos * 10.0
            distances = distances * 10.0
        optodes = np.vstack([s_pos, d_pos])

        if coreg is not None:
            optodes = coreg.apply(optodes)
            channel_mids = coreg.mni_channel_midpoints
        else:
            channel_mids = probe.channel_midpoints.astype(float)
            if probe.lengthUnit == 'cm':
                channel_mids = channel_mids * 10.0

        boundary_point, boundary_normal = fit_boundary_plane(optodes)

        pad = max(lateral_margin, depth_range[1])
        lo = optodes.min(axis=0) - pad
        hi = optodes.max(axis=0) + pad

        axes = [np.arange(lo[k], hi[k] + spacing, spacing) for k in range(3)]
        grid_shape = tuple(len(a) for a in axes)
        gx, gy, gz = np.meshgrid(*axes, indexing='ij')
        full_positions = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)

        depths = (full_positions - boundary_point) @ boundary_normal
        depth_ok = (depths >= depth_range[0]) & (depths <= depth_range[1])

        # Lateral distance to the nearest channel midpoint, measured within
        # the boundary plane (project out the depth component first so a
        # deep voxel directly under a channel isn't unfairly penalised).
        in_plane = full_positions - np.outer(depths, boundary_normal)
        mid_depths = (channel_mids - boundary_point) @ boundary_normal
        in_plane_mids = channel_mids - np.outer(mid_depths, boundary_normal)
        nearest_dist = cdist(in_plane, in_plane_mids).min(axis=1)
        lateral_radius = distances.max() / 2.0 + lateral_margin
        lateral_ok = nearest_dist <= lateral_radius

        keep = depth_ok & lateral_ok
        mask = keep.reshape(grid_shape)
        positions = full_positions[keep]

        return cls(positions=positions, spacing=spacing,
                    boundary_point=boundary_point, boundary_normal=boundary_normal,
                    grid_shape=grid_shape, mask=mask, origin=lo)

    @classmethod
    def from_head_volume(cls, volume, probe, volume_offset=None, *, coreg=None,
                          spacing=4.0, depth_range=(0.0, 30.0),
                          lateral_margin=15.0, volume_spacing=1.0,
                          outside_label=0, channel_mask=None):
        """
        Build a grid whose depth is measured against a segmented head volume.

        The anatomy-driven counterpart to :meth:`from_probe_fov`, for a montage no
        single plane describes: two separated patches have opposing surface
        normals, so a fitted plane would cut through the head rather than lying
        under each patch. Depth is the distance from each voxel to the nearest
        voxel outside the head, valid however the surface curves, and the lateral
        distance is decomposed from the three-dimensional distance rather than
        projected onto a plane.

        Parameters
        ----------
        volume : HeadModel or np.ndarray
            The segmented head. A HeadModel supplies its own offset, spacing and
            outside label, so the grid and the probe cannot end up in different
            frames.
        probe : Probe
            Must carry a channel configuration.
        volume_offset : array-like, optional
            Physical position in mm of the volume's first voxel. Required only for
            a bare array.
        coreg : Coregistration, optional
            When given, optode and channel positions are mapped through it first.
        spacing : float
            Isotropic voxel pitch of the grid, in mm.
        depth_range : tuple of (float, float)
            Minimum and maximum depth in mm below the scalp to keep.
        lateral_margin : float
            Extra margin in mm around the channel footprint.
        volume_spacing : float
            Physical size in mm of one volume voxel.
        outside_label : int
            Value in ``volume`` marking voxels outside the head.
        channel_mask : array-like of bool, optional
            Which channels the reconstruction will use. Pass the same mask used
            for the sensitivity matrix: the lateral radius follows the longest
            separation, so omitting it lets a pair that is never reconstructed
            inflate the radius and effectively disable the prune.

        Returns
        -------
        VoxelGrid
            With depths set from the anatomy and no boundary plane.
        """
        from scipy.ndimage import distance_transform_edt
        from .head import HeadModel

        if isinstance(volume, HeadModel):
            head = volume
            if volume_offset is None:
                volume_offset = head.offset
            volume_spacing = head.spacing
            outside_label = head.outside_label
            volume = head.volume
        elif volume_offset is None:
            raise ValueError(
                "from_head_volume needs volume_offset when given a bare array "
                "-- the offset is what places the volume in the same frame as "
                "the coregistered probe, and getting it wrong misplaces the "
                "whole reconstruction without raising. Pass a "
                "imaging.head.HeadModel instead and it comes with its own."
            )

        volume = np.asarray(volume)
        volume_offset = np.asarray(volume_offset, dtype=float)
        if volume.ndim != 3:
            raise ValueError(f"volume must be 3-D, got shape {volume.shape}.")

        s_pos = probe.s_pos.astype(float)
        d_pos = probe.d_pos.astype(float)
        distances = np.asarray(probe.distances, dtype=float)
        if probe.lengthUnit == 'cm':
            s_pos, d_pos, distances = s_pos * 10.0, d_pos * 10.0, distances * 10.0
        optodes = np.vstack([s_pos, d_pos])

        if coreg is not None:
            optodes = coreg.apply(optodes)
            channel_mids = coreg.mni_channel_midpoints
        else:
            channel_mids = probe.channel_midpoints.astype(float)
            if probe.lengthUnit == 'cm':
                channel_mids = channel_mids * 10.0

        if channel_mask is not None:
            channel_mask = np.asarray(channel_mask, dtype=bool)
            if channel_mask.shape != (len(distances),):
                raise ValueError(
                    f"channel_mask must have one entry per probe channel "
                    f"({len(distances)}), got {channel_mask.shape}."
                )
            if not channel_mask.any():
                raise ValueError("channel_mask excludes every channel.")
            channel_mids = channel_mids[channel_mask]
            distances = distances[channel_mask]

        # Depth field over the whole volume: distance (in volume voxels,
        # then mm) from each head voxel to the nearest non-head voxel.
        head = volume != outside_label
        depth_field = distance_transform_edt(head) * float(volume_spacing)

        pad = max(lateral_margin, depth_range[1])
        lo = optodes.min(axis=0) - pad
        hi = optodes.max(axis=0) + pad

        axes = [np.arange(lo[k], hi[k] + spacing, spacing) for k in range(3)]
        grid_shape = tuple(len(a) for a in axes)
        gx, gy, gz = np.meshgrid(*axes, indexing='ij')
        full_positions = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)

        # Sample the depth field at each lattice point (nearest volume voxel).
        idx = np.rint((full_positions - volume_offset) / float(volume_spacing)).astype(int)
        inside_volume = np.all((idx >= 0) & (idx < np.array(volume.shape)), axis=1)
        depths = np.zeros(len(full_positions))
        safe = idx[inside_volume]
        depths[inside_volume] = depth_field[safe[:, 0], safe[:, 1], safe[:, 2]]

        # depth == 0 means "outside the head" (the EDT is 0 off the mask), so
        # an in-head test is implied by depth > 0 and doesn't need repeating.
        depth_ok = (depths >= depth_range[0]) & (depths <= depth_range[1]) & (depths > 0)

        d3d = cdist(full_positions, channel_mids).min(axis=1)
        in_surface = np.sqrt(np.clip(d3d ** 2 - depths ** 2, 0.0, None))
        lateral_ok = in_surface <= distances.max() / 2.0 + lateral_margin

        keep = depth_ok & lateral_ok
        mask = keep.reshape(grid_shape)

        return cls(positions=full_positions[keep], spacing=spacing,
                    boundary_point=None, boundary_normal=None,
                    grid_shape=grid_shape, mask=mask, origin=lo,
                    depths=depths[keep])

    def restrict(self, keep):
        """
        Return a new grid holding only the selected voxels.

        The single place pruning is implemented, so that positions, depths and the
        lattice mask cannot drift apart.

        Parameters
        ----------
        keep : np.ndarray of bool
            Shape (n_voxels,), which voxels to keep.

        Returns
        -------
        VoxelGrid
            A new grid; this one is unchanged.
        """
        keep = np.asarray(keep, dtype=bool)
        if keep.shape != (self.n_voxels,):
            raise ValueError(
                f"keep must have shape (n_voxels,) = ({self.n_voxels},), "
                f"got {keep.shape}."
            )

        mask = None
        if self.mask is not None:
            mask = self.mask.copy()
            flat_mask = mask.ravel()
            true_idx = np.flatnonzero(flat_mask)
            flat_mask[true_idx[~keep]] = False
            mask = flat_mask.reshape(self.grid_shape)

        return VoxelGrid(
            positions=self.positions[keep], spacing=self.spacing,
            boundary_point=self.boundary_point, boundary_normal=self.boundary_normal,
            grid_shape=self.grid_shape, mask=mask, origin=self.origin,
            depths=None if self._depths is None else self._depths[keep],
        )

    def restrict_to_coverage(self, jacobian, threshold=1e-3):
        """
        Prune to the voxels the probe can actually see.

        Judged by the sensitivity matrix's column norms. A geometrically sensible
        domain is typically far larger than the measurable one, since sensitivity
        falls off steeply with depth; the unmeasured voxels cannot absorb signal,
        but every reconstructed array and every voxel-wise analysis pays for them.

        Apply this after building the sensitivity matrix, and slice its columns
        with the returned mask so the two stay aligned.

        Parameters
        ----------
        jacobian : np.ndarray
            Shape (n_channels, n_voxels). With several wavelengths, pass the least
            sensitive, or combine them.
        threshold : float
            Column norm, relative to the largest, below which a voxel is dropped.
            The default is conservative; inspect the distribution before
            tightening it.

        Returns
        -------
        grid : VoxelGrid
            The pruned grid.
        keep : np.ndarray of bool
            The selection applied, for slicing the matrix's columns.
        """
        jacobian = np.asarray(jacobian)
        if jacobian.shape[1] != self.n_voxels:
            raise ValueError(
                f"jacobian has {jacobian.shape[1]} columns but this grid has "
                f"{self.n_voxels} voxels."
            )
        coverage = (jacobian.astype(float) ** 2).sum(axis=0)
        peak = coverage.max()
        if not np.isfinite(peak) or peak <= 0:
            raise ValueError("jacobian has no positive sensitivity anywhere.")
        keep = coverage > threshold * peak
        return self.restrict(keep), keep

    def tissue_labels(self, volume, volume_offset=None, volume_spacing=1.0,
                       outside_label=0):
        """
        Look up the segmented-volume label at each voxel of this grid.

        Answers which tissue each reconstructed voxel sits in. A domain defined by
        depth below the scalp is mostly extracerebral by construction, so masking
        by label is the honest way to restrict a figure to brain, rather than
        widening a projection radius until distant scalp voxels reach the cortical
        surface.

        Parameters
        ----------
        volume : HeadModel or np.ndarray
            The segmented head the grid was built from. A HeadModel supplies its
            own offset, spacing and outside label.
        volume_offset : array-like, optional
            Physical position in mm of the volume's first voxel. Required only for
            a bare array.
        volume_spacing : float
            Physical size in mm of one volume voxel.
        outside_label : int
            Value returned for grid voxels falling outside the volume.

        Returns
        -------
        np.ndarray of int
            Shape (n_voxels,).
        """
        from .head import HeadModel

        if isinstance(volume, HeadModel):
            return volume.labels_at(self.positions)
        if volume_offset is None:
            raise ValueError(
                "tissue_labels needs volume_offset when given a bare array; "
                "pass the imaging.head.HeadModel instead and it comes with its "
                "own."
            )
        volume = np.asarray(volume)
        volume_offset = np.asarray(volume_offset, dtype=float)
        idx = np.rint((self.positions - volume_offset) / float(volume_spacing)).astype(int)
        inside = np.all((idx >= 0) & (idx < np.array(volume.shape)), axis=1)
        out = np.full(self.n_voxels, outside_label, dtype=int)
        safe = idx[inside]
        out[inside] = volume[safe[:, 0], safe[:, 1], safe[:, 2]]
        return out

    def project_to_surface(self, values, surface, radius=12.0, method='gaussian',
                            fwhm=10.0, power=2.0, min_voxels=1):
        """
        Resample a per-voxel scalar field onto a surface mesh's vertices.

        A reconstruction lattice is coarse and covers only what the array sees,
        while a cortical mesh is dense and curved, so the two share no points and
        an explicit resampling is needed. Vertices with no voxel within ``radius``
        come back as NaN, letting a renderer leave them unpainted rather than
        extrapolating into tissue that was never measured.

        ``radius`` is the parameter that matters. Voxels sit some distance from
        the cortical surface, so a tight window silently drops much of the image,
        while the intrinsic spatial resolution of the measurement is coarser than
        any small window would imply.

        Parameters
        ----------
        values : array-like
            Shape (n_voxels,), the field to project. NaN voxels are excluded.
        surface : Surface
            Target mesh, which must be in the same coordinate frame as this grid.
            A frame mismatch cannot be detected here and would project onto the
            wrong anatomy.
        radius : float
            Gather window in mm. A vertex with no voxel inside it stays NaN.
        method : {'gaussian', 'idw', 'peak', 'nearest'}
            How voxels within the window are combined. 'gaussian' (default)
            weights by a kernel of width ``fwhm``. 'idw' weights by inverse
            distance and is much peakier. 'peak' takes the value of largest
            magnitude, preserving extrema but dilating blobs and biasing away from
            zero. 'nearest' takes the closest voxel with no smoothing. The
            averaging methods let opposite-signed neighbours cancel, which shrinks
            peak amplitudes.
        fwhm : float
            Kernel width in mm, for the Gaussian method.
        power : float
            Exponent, for the inverse-distance method.
        min_voxels : int
            Vertices supported by fewer voxels than this are left NaN, which trims
            ragged edges at the field of view's border.

        Returns
        -------
        np.ndarray
            Shape (n_vertices,), NaN where unsupported.
        """
        methods = ('gaussian', 'idw', 'peak', 'nearest')
        if method not in methods:
            raise ValueError(f"method must be one of {methods}, got {method!r}.")
        values = np.asarray(values, dtype=float).ravel()
        if values.shape != (self.n_voxels,):
            raise ValueError(
                f"values must have shape (n_voxels,) = ({self.n_voxels},), "
                f"got {values.shape}."
            )
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}.")
        if method == 'gaussian' and fwhm <= 0:
            raise ValueError(f"fwhm must be positive, got {fwhm}.")

        verts = np.asarray(surface.vertices, dtype=float)
        good = np.isfinite(values)
        out = np.full(len(verts), np.nan)
        if not good.any():
            return out

        pos, vals = self.positions[good], values[good]
        tree = cKDTree(pos)
        neighbours = tree.query_ball_point(verts, r=radius)
        # exp(-4 ln2 d^2 / fwhm^2): the standard FWHM parameterisation.
        gauss_k = 4.0 * np.log(2.0) / (fwhm ** 2)

        for i, idx in enumerate(neighbours):
            if len(idx) < min_voxels:
                continue
            idx = np.asarray(idx)
            v = vals[idx]
            if method == 'peak':
                out[i] = v[np.argmax(np.abs(v))]
                continue
            d = np.linalg.norm(pos[idx] - verts[i], axis=1)
            if method == 'nearest':
                out[i] = v[np.argmin(d)]
                continue
            if method == 'gaussian':
                w = np.exp(-gauss_k * d ** 2)
            else:
                # A vertex sitting on a voxel centre takes that value
                # outright rather than dividing by zero.
                exact = d <= 1e-9
                if exact.any():
                    out[i] = v[exact].mean()
                    continue
                w = 1.0 / d ** power
            total = w.sum()
            if total > 0:
                out[i] = float(np.dot(w, v) / total)
        return out

    def restrict_to_surface(self, surface, max_distance=15.0):
        """
        Prune to voxels near a surface's vertices.

        Parameters
        ----------
        surface : Surface
            Target mesh, in the same frame as this grid.
        max_distance : float
            Maximum distance in mm from a vertex for a voxel to be kept.

        Returns
        -------
        VoxelGrid
            A new grid; this one is unchanged.
        """
        tree = cKDTree(surface.vertices)
        dist, _ = tree.query(self.positions)
        return self.restrict(dist <= max_distance)

    def __repr__(self):
        return (f"<VoxelGrid | {self.n_voxels:,} voxels | "
                f"spacing: {self.spacing:.1f}mm>")
