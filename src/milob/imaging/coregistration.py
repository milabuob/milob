import numpy as np

# Standard fiducial positions on the MNI152 (fsaverage) template, in mm (RAS).
# From MNE-Python fsaverage-fiducials, converted from metres.
MNI152_LANDMARKS = {
    'NAS': np.array([  1.0,  89.0, -25.0]),
    'LPA': np.array([-78.0, -15.0, -52.0]),
    'RPA': np.array([ 78.0, -15.0, -52.0]),
}

# Normalise the wide variety of landmark label conventions to NAS / LPA / RPA.
_LABEL_ALIASES = {
    'nas': 'NAS', 'nz': 'NAS', 'nasion': 'NAS', 'fpz': 'NAS',
    'lpa': 'LPA', 'al':  'LPA', 'lfe': 'LPA', 'lear': 'LPA', 'lm': 'LPA',
    'rpa': 'RPA', 'ar':  'RPA', 'rfe': 'RPA', 'rear': 'RPA', 'rm': 'RPA',
}


def _normalise_label(label: str) -> str:
    return _LABEL_ALIASES.get(label.strip().lower(), label.strip().upper())


def _rigid_transform_svd(src_pts: np.ndarray, dst_pts: np.ndarray):
    """
    Compute the rigid-body transform between two corresponding point sets.

    Uses an SVD-based Procrustes alignment, giving rotation and translation
    with no scaling.

    Parameters
    ----------
    src_pts, dst_pts : np.ndarray
        Corresponding points of shape (n, 3), in source and destination
        spaces.

    Returns
    -------
    R : np.ndarray
        Rotation matrix of shape (3, 3).
    t : np.ndarray
        Translation vector of shape (3,), such that a row vector ``p`` maps to
        ``p @ R.T + t``.
    """
    c_src = src_pts.mean(axis=0)
    c_dst = dst_pts.mean(axis=0)
    H = (src_pts - c_src).T @ (dst_pts - c_dst)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:      # correct for reflection
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = c_dst - c_src @ R.T
    return R, t


class Coregistration:
    """
    Rigid-body coregistration of probe positions to MNI space.

    Uses three anatomical landmarks to compute a rotation and translation from
    the digitiser frame onto MNI space. The transform is rigid, so distances
    are preserved, and probe coordinates are converted to mm first, since a
    rigid transform cannot absorb a unit mismatch.

    Parameters
    ----------
    probe : Probe
        Probe carrying digitised landmarks, their labels, and 3-D optode
        positions.
    reference_landmarks : dict, optional
        Target landmarks as {label: (x, y, z)} in mm. Labels are matched
        case-insensitively after alias resolution. Defaults to the MNI152
        fiducials.

    Examples
    --------
    >>> coreg = Coregistration(probe).fit()
    >>> mni_positions = coreg.mni_channel_midpoints
    """

    def __init__(self, probe, reference_landmarks=None):
        self.probe = probe
        self.reference_landmarks = (
            {k: np.asarray(v, dtype=float) for k, v in reference_landmarks.items()}
            if reference_landmarks is not None
            else MNI152_LANDMARKS
        )
        self._R = None
        self._t = None

    # ------------------------------------------------------------------ #
    # Fitting                                                              #
    # ------------------------------------------------------------------ #

    def fit(self):
        """
        Compute the transform from probe space to MNI space.

        Returns
        -------
        Coregistration
            This object, for chaining.
        """
        if self.probe.landmarks is None or self.probe.landmark_labels is None:
            raise ValueError(
                "Probe must have 'landmarks' (Nx3 positions) and "
                "'landmark_labels' set before coregistration."
            )
        probe_pts, ref_pts = self._match_landmarks()
        if len(probe_pts) < 3:
            normed = [_normalise_label(l) for l in self.probe.landmark_labels]
            raise ValueError(
                f"Need at least 3 matching landmarks but found {len(probe_pts)}.\n"
                f"  Probe labels (normalised): {normed}\n"
                f"  Reference labels:          {list(self.reference_landmarks.keys())}"
            )
        self._R, self._t = _rigid_transform_svd(probe_pts, ref_pts)
        return self

    # ------------------------------------------------------------------ #
    # Transform                                                            #
    # ------------------------------------------------------------------ #

    def apply(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points from probe space to MNI space.

        Parameters
        ----------
        points : np.ndarray
            Shape (n, 3), in probe space and already in mm.

        Returns
        -------
        np.ndarray
            Shape (n, 3), in MNI space in mm.
        """
        self._check_fitted()
        return np.asarray(points, dtype=float) @ self._R.T + self._t

    # ------------------------------------------------------------------ #
    # Convenience properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def mni_s_pos(self) -> np.ndarray:
        """Source positions in MNI space, shape (n_sources, 3), in mm."""
        return self.apply(self._to_mm(self.probe.s_pos))

    @property
    def mni_d_pos(self) -> np.ndarray:
        """Detector positions in MNI space, shape (n_detectors, 3), in mm."""
        return self.apply(self._to_mm(self.probe.d_pos))

    @property
    def mni_channel_midpoints(self) -> np.ndarray:
        """
        Source-detector midpoints in MNI space.

        Returns
        -------
        np.ndarray
            Shape (n_channels, 3) in mm, ordered to match the probe's channel
            labels.
        """
        ch = self.probe._channels
        pts = np.array([
            (self.probe.s_pos[int(s) - 1] + self.probe.d_pos[int(d) - 1]) / 2
            for s, d in zip(ch['sources'], ch['detectors'])
        ])
        return self.apply(self._to_mm(pts))

    @property
    def mni_landmark_pos(self) -> np.ndarray:
        """Probe landmarks in MNI space, shape (n, 3), in mm."""
        return self.apply(self._to_mm(self.probe.landmarks))

    @property
    def transform(self) -> np.ndarray:
        """Homogeneous 4x4 transform from probe space to MNI space."""
        self._check_fitted()
        T = np.eye(4)
        T[:3, :3] = self._R
        T[:3,  3] = self._t
        return T

    @property
    def residuals(self) -> dict:
        """
        Per-landmark registration error.

        Returns
        -------
        np.ndarray
            Distance in mm between each transformed landmark and its reference.
        """
        self._check_fitted()
        probe_pts, ref_pts = self._match_landmarks()
        transformed = self.apply(probe_pts)
        errors = np.linalg.norm(transformed - ref_pts, axis=1)
        labels = list(self.reference_landmarks.keys())[:len(errors)]
        return {lbl: float(e) for lbl, e in zip(labels, errors)}

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    @property
    def _probe_scale(self) -> float:
        """
        Return the factor taking the probe's own coordinates to mm.

        The fit is rigid and the reference landmarks are in mm, so a probe built
        in other units must be scaled before it is fitted or transformed.
        """
        return 10.0 if getattr(self.probe, 'lengthUnit', None) == 'cm' else 1.0

    def _to_mm(self, points) -> np.ndarray:
        """Convert probe-space points from the probe's own unit to mm."""
        return np.asarray(points, dtype=float) * self._probe_scale

    def _match_landmarks(self):
        probe_pts, ref_pts = [], []
        for ref_lbl, ref_xyz in self.reference_landmarks.items():
            norm_ref = _normalise_label(ref_lbl)
            for i, probe_lbl in enumerate(self.probe.landmark_labels):
                if _normalise_label(probe_lbl) == norm_ref:
                    probe_pts.append(self.probe.landmarks[i])
                    ref_pts.append(ref_xyz)
                    break
        return self._to_mm(probe_pts), np.array(ref_pts)

    def _check_fitted(self):
        if self._R is None:
            raise RuntimeError(
                "Coregistration has not been fitted yet. Call .fit() first."
            )

    def __repr__(self):
        fitted = self._R is not None
        if fitted:
            res = self.residuals
            rms = float(np.sqrt(np.mean(list(res.values()) ** 2 if False
                                         else [v**2 for v in res.values()])))
            return (f"<Coregistration | fitted | "
                    f"RMS residual: {rms:.2f} mm | "
                    f"landmarks: {list(res.keys())}>")
        return "<Coregistration | not fitted>"
