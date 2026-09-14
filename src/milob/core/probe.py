import numpy as np
from scipy.spatial.distance import cdist


class Probe:
    def __init__(self, s_pos, d_pos, wavelengths, *, sc_threshold,
                 channels=None, landmark_pos=None, landmark_labels=None,
                 source_labels=None, detector_labels=None,
                 s_pos_2d=None, d_pos_2d=None,
                 lengthUnit=None):
        """
        Optode geometry and channel configuration.

        Parameters
        ----------
        s_pos : np.ndarray
            Source positions, shape (n_sources, 3).
        d_pos : np.ndarray
            Detector positions, shape (n_detectors, 3).
        wavelengths : np.ndarray
            Wavelengths in nm.
        sc_threshold : float or None
            Source-detector distance in mm below which a channel is classified as
            short. Fixed for the lifetime of the probe and exposed read-only. Pass
            None when the design has no short-separation channels, in which case
            every channel is classified as long.
        channels : dict, optional
            Channel configuration with keys 'sources' and 'detectors' (1-based
            optode IDs), 'wavelengths', and optionally 'datatypes'. Without it the
            probe carries geometry only.
        landmark_pos : np.ndarray, optional
            Positions of digitised anatomical landmarks.
        landmark_labels : list of str, optional
            Label for each landmark.
        source_labels, detector_labels : list of str, optional
            Names for the optodes.
        s_pos_2d, d_pos_2d : np.ndarray, optional
            Two-dimensional layout positions, for flat probe drawings.
        lengthUnit : {'mm', 'cm'}, optional
            Unit of the position arrays.

        Examples
        --------
        >>> probe = Probe(s_pos, d_pos, wavelengths, sc_threshold=12)
        >>> channels = {'sources': [1, 1, 2, 2], 'detectors': [1, 2, 1, 2],
        ...             'wavelengths': [760, 850, 760, 850]}
        >>> probe = Probe(s_pos, d_pos, wavelengths, channels=channels,
        ...               sc_threshold=12)
        """
        # 3D Physical geometry (Primary)
        self.s_pos = np.array(s_pos)
        self.d_pos = np.array(d_pos)

        # 2D Geometry (Optional Layout)
        self.s_pos_2d = np.array(s_pos_2d) if s_pos_2d is not None else None
        self.d_pos_2d = np.array(d_pos_2d) if d_pos_2d is not None else None

        self.lengthUnit = lengthUnit
        self.wavelengths = np.array(wavelengths)

        # Store labels if provided, else generate defaults
        self.source_labels = source_labels if source_labels is not None else [f"S{i+1}" for i in range(len(s_pos))]
        self.detector_labels = detector_labels if detector_labels is not None else [f"D{i+1}" for i in range(len(d_pos))]

        # Channel configuration
        self._channels = channels

        # Landmarks
        self.landmarks = landmark_pos
        self.landmark_labels = landmark_labels

        # Cache computed properties
        self._channel_labels = None
        self._distances = None

        # User-defined ROIs: name → list of channel labels
        self.rois = {}

        # Short/long classification threshold (mm) -- fixed for this Probe's
        # lifetime, see the `sc_threshold` property. None is a genuine,
        # permanent value (this probe design has no short-separation
        # channels), not a placeholder for "not decided yet" -- there is no
        # default, so every Probe is forced to make this choice explicitly.
        self._sc_threshold = sc_threshold
        



    # ============================================
    # Properties
    # ============================================
    
    @property
    def sc_threshold(self):
        """Distance in mm below which a channel counts as short. Read-only."""
        return self._sc_threshold

    @property
    def has_channels(self):
        """True if a channel configuration is present."""
        return self._channels is not None
    
    @property
    def n_sources(self):
        """Number of sources."""
        return len(self.s_pos)
    
    @property
    def n_detectors(self):
        """Number of detectors."""
        return len(self.d_pos)
    
    @property
    def n_channels(self):
        """Number of channels."""
        if not self.has_channels:
            raise ValueError("No channel configuration available")
        return len(self._channels['sources'])
    
    @property
    def channel_labels(self):
        """Channel labels, e.g. 'S1D1'."""
        if not self.has_channels:
            raise ValueError("No channel configuration available")
        
        if self._channel_labels is None:
            self._channel_labels = [
                f"S{s}D{d}" 
                for s, d in zip(self._channels['sources'], self._channels['detectors'])
            ]
        return self._channel_labels
    
    @property
    def distances(self):
        """Source-detector distance for each channel, in :attr:`lengthUnit`."""
        if not self.has_channels:
            raise ValueError("No channel configuration available")
        
        if self._distances is None:
            self._distances = [
                self.get_distance(s - 1, d - 1)  # Convert to 0-based
                for s, d in zip(self._channels['sources'], self._channels['detectors'])
            ]
        return self._distances
    

    
    # ============================================
    # ROI Management
    # ============================================

    def add_roi(self, name, channels, show_stats=True):
        """
        Define a named ROI from a list of channels.

        Calling this with an existing name replaces that ROI.

        Parameters
        ----------
        name : str
            ROI name.
        channels : list of str
            Channel labels belonging to the ROI. Labels absent from the probe are
            skipped with a warning.
        show_stats : bool
            Print a one-line channel-count summary, including the short and long
            breakdown. Default True.
        """
        if not self.has_channels:
            raise ValueError("Probe has no channel configuration — cannot define ROIs.")

        valid = set(self.channel_labels)
        accepted = [ch for ch in channels if ch in valid]
        skipped  = [ch for ch in channels if ch not in valid]

        if skipped:
            import logging
            logging.getLogger('milob').warning(
                f"ROI '{name}': {len(skipped)} label(s) not in probe and were skipped: {skipped}"
            )
        if not accepted:
            raise ValueError(f"ROI '{name}': no valid channel labels provided.")

        self.rois[name] = accepted

        if show_stats:
            if self.sc_threshold is None:
                print(f"ROI '{name}': {len(accepted)} channel(s) "
                      f"(no short-channel concept for this probe)")
            else:
                label_to_dist = dict(zip(self.channel_labels, self.distances))
                dists = np.array([label_to_dist[ch] for ch in accepted])
                if self.lengthUnit == 'cm':
                    dists = dists * 10
                n_short = int(np.sum(dists < self.sc_threshold))
                n_long = len(accepted) - n_short
                print(f"ROI '{name}': {len(accepted)} channel(s) — "
                      f"{n_short} short (<{self.sc_threshold}mm), {n_long} long")

    def get_roi_mask(self, name):
        """
        Boolean mask selecting a named ROI's channels.

        Parameters
        ----------
        name : str
            ROI name.

        Returns
        -------
        np.ndarray of bool
            Length n_channels, in ``channel_labels`` order.
        """
        if name not in self.rois:
            raise KeyError(f"ROI '{name}' not defined. Available: {list(self.rois.keys())}")
        roi_set = set(self.rois[name])
        return np.array([ch in roi_set for ch in self.channel_labels])

    def rename_roi(self, old_name, new_name):
        """
        Rename an ROI, keeping its position in the ROI order.

        Parameters
        ----------
        old_name : str
            Current name.
        new_name : str
            Replacement name.

        Examples
        --------
        >>> probe.add_roi_by_clustering(4)
        >>> probe.rename_roi('ROI_1', 'frontal_left')
        """
        if old_name not in self.rois:
            raise KeyError(f"ROI '{old_name}' not found. Available: {list(self.rois.keys())}")
        if new_name in self.rois and new_name != old_name:
            raise ValueError(f"ROI '{new_name}' already exists.")
        self.rois = {(new_name if k == old_name else k): v for k, v in self.rois.items()}

    def list_rois(self):
        """
        Summarise the defined ROIs.

        Returns
        -------
        pandas.DataFrame
            Indexed by ROI name, with columns n_channels and channels.
        """
        import pandas as pd
        rows = [
            {'roi': name, 'n_channels': len(chs), 'channels': ', '.join(chs)}
            for name, chs in self.rois.items()
        ]
        if not rows:
            return pd.DataFrame(columns=['n_channels', 'channels'])
        return pd.DataFrame(rows).set_index('roi')

    @property
    def channel_roi_map(self):
        """
        Map every channel to its ROI.

        Returns
        -------
        dict of {str: str or None}
            Channel label to ROI name, with None for unassigned channels.
        """
        mapping = {ch: None for ch in self.channel_labels}
        for roi_name, channels in self.rois.items():
            for ch in channels:
                mapping[ch] = roi_name
        return mapping

    # ============================================
    # ROI Construction helpers
    # ============================================

    @property
    def channel_midpoints(self):
        """Midpoint of each channel, the mean of its source and detector positions."""
        if not self.has_channels:
            raise ValueError("No channel configuration available.")
        return np.array([
            (self.s_pos[s - 1] + self.d_pos[d - 1]) / 2
            for s, d in zip(self._channels['sources'], self._channels['detectors'])
        ])

    def channel_distance_matrix(self):
        """
        Pairwise distance between channel midpoints.

        Returns
        -------
        np.ndarray
            Symmetric matrix of shape (n_channels, n_channels) in
            :attr:`lengthUnit`, ordered to match ``channel_labels``.
        """
        mids = self.channel_midpoints
        diff = mids[:, None, :] - mids[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    def _channel_optode_positions_mm(self):
        """Return {channel label: (source position, detector position)} in mm."""
        scale = 10.0 if self.lengthUnit == 'cm' else 1.0
        s = np.asarray(self.s_pos, dtype=float) * scale
        d = np.asarray(self.d_pos, dtype=float) * scale
        return {
            label: (s[si - 1], d[di - 1])
            for label, si, di in zip(self.channel_labels,
                                     self._channels['sources'],
                                     self._channels['detectors'])
        }

    def compare(self, other, *, position_tol=1.0):
        """
        Compare this probe's channels, geometry and wavelengths with another.

        Geometry is compared per channel and keyed by label, so the two probes
        need not number their optodes identically. Both are converted to mm first.

        Parameters
        ----------
        other : Probe
            Probe to compare against. Both need a channel configuration.
        position_tol : float
            Largest optode displacement in mm still counted as the same montage.
            Default 1.0.

        Returns
        -------
        dict
            Keys ``match``, ``same_channels``, ``same_order``,
            ``wavelengths_match``, ``missing_from_other``, ``extra_in_other``,
            ``n_shared``, ``max_optode_offset_mm``, ``position_tol_mm`` and
            ``summary``. ``match`` is True when the channels, wavelengths and
            geometry all agree; channel order is reported but does not affect it.

        Examples
        --------
        >>> probe_a.compare(probe_b)['match']
        True
        """
        if not (self.has_channels and other.has_channels):
            raise ValueError(
                "Probe.compare needs channel configuration on both probes; "
                "bare geometry has no channel labels to line up."
            )

        mine, theirs = list(self.channel_labels), list(other.channel_labels)
        mine_set, theirs_set = set(mine), set(theirs)
        missing = [c for c in mine if c not in theirs_set]
        extra = [c for c in theirs if c not in mine_set]
        shared = [c for c in mine if c in theirs_set]
        same_channels = not missing and not extra

        wl_a = np.sort(np.asarray(self.wavelengths, dtype=float))
        wl_b = np.sort(np.asarray(other.wavelengths, dtype=float))
        wavelengths_match = (wl_a.shape == wl_b.shape
                             and bool(np.allclose(wl_a, wl_b, atol=1.0)))

        max_offset = None
        if shared:
            a_pos = self._channel_optode_positions_mm()
            b_pos = other._channel_optode_positions_mm()
            max_offset = max(
                max(float(np.linalg.norm(a_pos[c][0] - b_pos[c][0])),
                    float(np.linalg.norm(a_pos[c][1] - b_pos[c][1])))
                for c in shared
            )

        geometry_ok = max_offset is not None and max_offset <= position_tol
        match = same_channels and wavelengths_match and geometry_ok

        if match:
            summary = (f"same montage: {len(shared)} channels, max optode "
                       f"offset {max_offset:.2f} mm")
        else:
            bits = []
            if missing:
                bits.append(f"{len(missing)} channel(s) absent from the other "
                            f"probe (e.g. {missing[:3]})")
            if extra:
                bits.append(f"{len(extra)} extra channel(s) in the other probe "
                            f"(e.g. {extra[:3]})")
            if not wavelengths_match:
                bits.append(f"wavelengths {list(wl_a)} vs {list(wl_b)}")
            if max_offset is None:
                bits.append("no channels in common")
            elif not geometry_ok:
                bits.append(f"max optode offset {max_offset:.2f} mm "
                            f"(tolerance {position_tol:g} mm)")
            summary = "different montage: " + "; ".join(bits)

        return {
            'match': match,
            'same_channels': same_channels,
            'same_order': mine == theirs,
            'wavelengths_match': wavelengths_match,
            'missing_from_other': missing,
            'extra_in_other': extra,
            'n_shared': len(shared),
            'max_optode_offset_mm': max_offset,
            'position_tol_mm': float(position_tol),
            'summary': summary,
        }

    @staticmethod
    def _scale_to_mm(probe):
        """Return the optode positions converted to mm."""
        return 10.0 if probe.lengthUnit == 'cm' else 1.0

    @classmethod
    def union(cls, probes, *, position_tol=1.0):
        """
        Build one probe carrying every channel recorded by any of the inputs.

        Channel order follows first appearance across the sequence. Optode
        numbering is assumed shared, and verified: every optode and shared channel
        must agree on position within ``position_tol``.

        Parameters
        ----------
        probes : sequence of Probe
            Probes to merge. All need a channel configuration.
        position_tol : float
            Largest optode displacement in mm still counted as agreement.

        Returns
        -------
        Probe
            Geometry from the largest optode table, channels from every input, and
            landmarks from the first probe carrying them.

        Raises
        ------
        ValueError
            If the probes disagree on wavelengths, length unit, ``sc_threshold``
            or optode positions.

        Examples
        --------
        >>> montage = Probe.union([s.get_probe('task') for s in study.sessions])
        """
        probes = list(probes)
        if not probes:
            raise ValueError("Probe.union needs at least one probe.")
        without = [i for i, p in enumerate(probes) if not p.has_channels]
        if without:
            raise ValueError(
                f"Probe.union needs channel configuration on every probe; "
                f"probe(s) at index {without} have bare geometry only."
            )
        if len(probes) == 1:
            return probes[0]

        ref = probes[0]

        # ── The three things that must simply agree ──────────────────────
        wl_ref = np.sort(np.asarray(ref.wavelengths, dtype=float))
        for i, p in enumerate(probes[1:], start=1):
            wl = np.sort(np.asarray(p.wavelengths, dtype=float))
            if wl.shape != wl_ref.shape or not np.allclose(wl, wl_ref, atol=1.0):
                raise ValueError(
                    f"Probe.union: probe {i} records wavelengths {list(wl)}, "
                    f"probe 0 records {list(wl_ref)}. A single sensitivity "
                    "operator is built per wavelength, so these are different "
                    "montages, not one to be unioned."
                )
            if p.lengthUnit != ref.lengthUnit:
                raise ValueError(
                    f"Probe.union: probe {i} is in {p.lengthUnit!r}, probe 0 in "
                    f"{ref.lengthUnit!r}. Convert before unioning rather than "
                    "letting the result inherit one arbitrarily."
                )
            if p.sc_threshold != ref.sc_threshold:
                raise ValueError(
                    f"Probe.union: probe {i} classifies short channels at "
                    f"{p.sc_threshold} mm, probe 0 at {ref.sc_threshold} mm. "
                    "That is a different probe design assumption (see Probe's "
                    "class docstring), not a montage difference."
                )

        # ── Optode tables: take the largest, verify the rest against it ──
        base_s = max(probes, key=lambda p: len(p.s_pos))
        base_d = max(probes, key=lambda p: len(p.d_pos))
        s_pos = np.asarray(base_s.s_pos, dtype=float) * cls._scale_to_mm(base_s)
        d_pos = np.asarray(base_d.d_pos, dtype=float) * cls._scale_to_mm(base_d)

        for i, p in enumerate(probes):
            for name, table, theirs in (('source', s_pos, p.s_pos),
                                        ('detector', d_pos, p.d_pos)):
                theirs = np.asarray(theirs, dtype=float) * cls._scale_to_mm(p)
                n = len(theirs)
                if n > len(table):                      # pragma: no cover
                    raise AssertionError("largest table was not the largest")
                offset = np.linalg.norm(table[:n] - theirs, axis=1)
                if offset.size and offset.max() > position_tol:
                    worst = int(np.argmax(offset))
                    raise ValueError(
                        f"Probe.union: probe {i} puts {name} {worst + 1} "
                        f"{offset[worst]:.2f} mm from where the largest optode "
                        f"table does (tolerance {position_tol:g} mm). These "
                        "probes do not share one optode numbering, so their "
                        "channel labels do not name the same pairs. Build one "
                        "operator per montage instead of unioning."
                    )

        # ── Channels: first appearance across the sequence ───────────────
        keysets = {frozenset(p._channels) for p in probes}
        if len(keysets) != 1:
            raise ValueError(
                f"Probe.union: the probes carry different channel fields "
                f"{[sorted(k) for k in keysets]}; cannot merge them without "
                "inventing values for the missing ones."
            )
        fields = list(probes[0]._channels)

        seen, merged = set(), {k: [] for k in fields}
        for p in probes:
            for j, label in enumerate(p.channel_labels):
                if label in seen:
                    continue
                seen.add(label)
                for k in fields:
                    merged[k].append(p._channels[k][j])

        landmark_src = next((p for p in probes if p.landmarks is not None), ref)
        scale = cls._scale_to_mm(landmark_src)
        landmarks = (None if landmark_src.landmarks is None
                     else np.asarray(landmark_src.landmarks, dtype=float) * scale)

        out = cls(
            s_pos, d_pos, ref.wavelengths,
            channels=merged,
            landmark_pos=landmarks,
            landmark_labels=landmark_src.landmark_labels,
            source_labels=base_s.source_labels,
            detector_labels=base_d.detector_labels,
            lengthUnit='mm',
            sc_threshold=ref.sc_threshold,
        )

        # Shared channels must name the same physical pair in every probe.
        # The optode check above already implies this, but a channel whose
        # source or detector index is out of range in one probe would slip
        # through it, and a silently mispaired channel is the failure this
        # whole class of check exists to prevent.
        out_pos = out._channel_optode_positions_mm()
        for i, p in enumerate(probes):
            p_pos = p._channel_optode_positions_mm()
            for label, (sp, dp) in p_pos.items():
                op_s, op_d = out_pos[label]
                gap = max(float(np.linalg.norm(op_s - sp)),
                          float(np.linalg.norm(op_d - dp)))
                if gap > position_tol:                  # pragma: no cover
                    raise ValueError(
                        f"Probe.union: channel {label} sits {gap:.2f} mm from "
                        f"where probe {i} puts it (tolerance {position_tol:g} mm)."
                    )
        return out

    def add_roi_by_position(self, name, center, radius, show_stats=True):
        """
        Define an ROI from the channels whose midpoint lies within a sphere.

        Parameters
        ----------
        name : str
            ROI name.
        center : array-like, shape (3,)
            Centre coordinate in :attr:`lengthUnit`.
        radius : float
            Inclusion radius in the same unit.
        show_stats : bool
            Forwarded to :meth:`add_roi`.
        """
        center = np.array(center)
        dists  = np.linalg.norm(self.channel_midpoints - center, axis=1)
        included = [ch for ch, d in zip(self.channel_labels, dists) if d <= radius]
        if not included:
            raise ValueError(
                f"No channels found within radius {radius} of {center}. "
                f"Check units — probe length unit is '{self.lengthUnit}'."
            )
        self.add_roi(name, included, show_stats=show_stats)

    def add_roi_by_landmark(self, name, landmark_name, radius, show_stats=True):
        """
        Define an ROI from the channels within a sphere around a landmark.

        Parameters
        ----------
        name : str
            ROI name.
        landmark_name : str
            Label matching an entry in ``landmark_labels``.
        radius : float
            Inclusion radius in :attr:`lengthUnit`.
        show_stats : bool
            Forwarded to :meth:`add_roi`.
        """
        if self.landmarks is None or not self.landmark_labels:
            raise ValueError("No landmarks defined on this probe.")
        if landmark_name not in self.landmark_labels:
            raise KeyError(
                f"Landmark '{landmark_name}' not found. "
                f"Available: {self.landmark_labels}"
            )
        idx    = list(self.landmark_labels).index(landmark_name)
        center = self.landmarks[idx]
        self.add_roi_by_position(name, center, radius, show_stats=show_stats)

    def add_roi_by_mni_position(self, name, mni_center, radius, reference_landmarks=None,
                                 show_stats=True):
        """
        Define an ROI from the channels near a standard MNI coordinate.

        Coregisters the probe to MNI space using its own digitised landmarks, so
        the coordinate selects anatomically corresponding channels whatever the cap
        placement. Registration is rigid, so ``radius`` keeps its physical meaning.
        Selection is by source-detector midpoint on the scalp, which approximates
        rather than resolves cortical correspondence.

        Parameters
        ----------
        name : str
            ROI name.
        mni_center : array-like, shape (3,)
            Target coordinate in MNI152 space, in mm.
        radius : float
            Inclusion radius in mm.
        reference_landmarks : dict, optional
            Forwarded to :meth:`coreg`. Defaults to the MNI152 fiducials.
        show_stats : bool
            Forwarded to :meth:`add_roi`.

        Examples
        --------
        >>> probe.add_roi_by_mni_position('dlPFC_L', (-42, 26, 24), radius=15)
        """
        coreg = self.coreg(reference_landmarks=reference_landmarks)
        mni_mid = coreg.mni_channel_midpoints
        center = np.asarray(mni_center, dtype=float)
        dists = np.linalg.norm(mni_mid - center, axis=1)
        included = [ch for ch, d in zip(self.channel_labels, dists) if d <= radius]
        if not included:
            raise ValueError(
                f"No channels found within radius {radius}mm of MNI {tuple(center)}."
            )
        self.add_roi(name, included, show_stats=show_stats)

    def add_roi_by_clustering(self, n_rois, method='kmeans', names=None, random_state=42,
                               show_stats=True):
        """
        Partition channels into spatial clusters and store them as ROIs.

        Clusters channel midpoints in three dimensions. The result is spatially
        compact but carries no anatomical labelling.

        Parameters
        ----------
        n_rois : int
            Number of clusters.
        method : {'kmeans', 'agglomerative'}
            Clustering algorithm. Default 'kmeans'.
        names : list of str, optional
            ROI names, of length ``n_rois``. Defaults to 'ROI_1', 'ROI_2', ...
        random_state : int
            Seed for k-means. Ignored for agglomerative clustering.
        show_stats : bool
            Forwarded to :meth:`add_roi` for each cluster.
        """
        from sklearn.cluster import KMeans, AgglomerativeClustering

        if names is not None and len(names) != n_rois:
            raise ValueError(f"'names' must have exactly {n_rois} entries.")

        midpoints = self.channel_midpoints

        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_rois, random_state=random_state, n_init=10)
        elif method == 'agglomerative':
            clusterer = AgglomerativeClustering(n_clusters=n_rois)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'kmeans' or 'agglomerative'.")

        cluster_ids   = clusterer.fit_predict(midpoints)
        channel_arr   = np.array(self.channel_labels)
        roi_names_out = names if names is not None else [f"ROI_{k + 1}" for k in range(n_rois)]

        for k, roi_name in enumerate(roi_names_out):
            roi_chs = channel_arr[cluster_ids == k].tolist()
            self.add_roi(roi_name, roi_chs, show_stats=show_stats)

    def plot_rois_2d(self, ax=None, show_labels=False, show_axis=True):
        """
        Draw a two-dimensional map of the defined ROIs.

        Channel midpoints are coloured by ROI; unassigned channels take a neutral
        colour. Sources and detectors are not drawn.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Created if omitted.
        show_labels : bool
            Annotate each midpoint with its channel label. Default False.
        show_axis : bool
            Draw the axes. Default False.

        Returns
        -------
        tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        from ..viz.topo import plot_rois_2d
        return plot_rois_2d(self, ax=ax, show_labels=show_labels, show_axis=show_axis)

    # ============================================
    # Distance Calculations
    # ============================================

    def get_distance(self, s_idx, d_idx):
        """
        Distance between one source and one detector.

        Parameters
        ----------
        s_idx : int
            Source index.
        d_idx : int
            Detector index.

        Returns
        -------
        float
            Euclidean distance in :attr:`lengthUnit`.
        """
        # Force to integer to prevent IndexError
        s_idx, d_idx = int(s_idx), int(d_idx)
        
        src = self.s_pos[s_idx]
        det = self.d_pos[d_idx]
        return np.linalg.norm(src - det)

    
    def _resolve_sc_threshold(self, threshold):
        """Return the explicit threshold if given, else this probe's own."""
        return threshold if threshold is not None else self.sc_threshold

    def get_short_channels(self, threshold=None):
        """
        Boolean mask selecting the short-separation channels.

        Parameters
        ----------
        threshold : float, optional
            Distance in mm to use for this call only. Defaults to the probe's own
            ``sc_threshold``.

        Returns
        -------
        np.ndarray of bool
            All False when ``sc_threshold`` is None.
        """
        if not self.has_channels:
            raise ValueError("No channel configuration available")

        threshold = self._resolve_sc_threshold(threshold)
        if threshold is None:
            return np.zeros(self.n_channels, dtype=bool)

        distances = np.array(self.distances)
        if self.lengthUnit == 'cm':  # transform to mm
            distances = 10*distances

        return distances < threshold

    # ============================================
    # Channel Classification
    # ============================================
    
    def list_channels(self):
        """
        Tabulate the channel configuration.

        Returns
        -------
        pandas.DataFrame
            One row per channel, with source, detector, distance and wavelength.
        """
        if not self.has_channels:
            raise ValueError("No channel configuration available")
        
        
        import pandas as pd
        
        data = {
            'channel': self.channel_labels,
            'source': self._channels['sources'],
            'detector': self._channels['detectors'],
            'distance': self._distances,
        }
        
        # Add wavelength info if available
        if 'wavelengths' in self._channels:
            data['wavelength'] = self._channels['wavelengths']
        
        # Add datatype info if available
        if 'datatypes' in self._channels:
            data['datatype'] = self._channels['datatypes']
        
        return pd.DataFrame(data)
    
    
    def list_short_channels(self, threshold=None):
        """
        List the short-separation channel labels.

        Parameters
        ----------
        threshold : float, optional
            Distance in mm to use for this call only. Defaults to the probe's own
            ``sc_threshold``.

        Returns
        -------
        list of str
            Empty when ``sc_threshold`` is None.
        """
        threshold = self._resolve_sc_threshold(threshold)
        if threshold is None:
            return []

        distances = np.array(self.distances)
        if self.lengthUnit == 'cm':  # transform to mm
            distances = 10*distances

        labels = np.array(self.channel_labels)
        mask = distances < threshold
        return labels[mask].tolist()


    def list_long_channels(self, threshold=None):
        """
        List the long-separation channel labels.

        A channel is long when it is not short; there is no separate cutoff.

        Parameters
        ----------
        threshold : float, optional
            Distance in mm to use for this call only. Defaults to the probe's own
            ``sc_threshold``.

        Returns
        -------
        list of str
            Every channel when ``sc_threshold`` is None.
        """
        threshold = self._resolve_sc_threshold(threshold)
        if threshold is None:
            return list(self.channel_labels)

        distances = np.array(self.distances)
        if self.lengthUnit == 'cm':  # transform to mm
            distances = 10*distances

        labels = np.array(self.channel_labels)
        mask = distances >= threshold
        return labels[mask].tolist()


    # ============================================
    # Channel Visualization
    # ============================================
    def get_2d_coords(self, projection='orthographic'):
        """
        Two-dimensional coordinates for each channel.

        Parameters
        ----------
        projection : str, optional
            Projection used to flatten the geometry.

        Returns
        -------
        dict of {str: tuple}
            Channel label to (x, y).
        """
        if not self.has_channels:
            raise ValueError("No channel configuration available")

        if self.s_pos_2d is not None and self.d_pos_2d is not None:
            src_arr, det_arr = self.s_pos_2d, self.d_pos_2d
        else:
            src_arr, det_arr = self.s_pos[:, :2], self.d_pos[:, :2]

        coords_2d = {}
        for i, label in enumerate(self.channel_labels):
            s_idx = int(self._channels['sources'][i]) - 1
            d_idx = int(self._channels['detectors'][i]) - 1
            midpoint = (src_arr[s_idx] + det_arr[d_idx]) / 2
            coords_2d[label] = (float(midpoint[0]), float(midpoint[1]))

        return coords_2d
    
    def get_channel_label(self, ch_idx):
        """Descriptive label for a channel, including its ROI where one is defined."""
        s_id = self._channels['sources'][ch_idx]
        d_id = self._channels['detectors'][ch_idx]
        
        base_name = f"S{s_id}D{d_id}"
        
        # If we have Brainsight/Custom labels, append them
        if self.source_labels and self.detector_labels:
            s_lab = self.source_labels[s_id-1]
            d_lab = self.detector_labels[d_id-1]
            return f"{base_name} ({s_lab}-{d_lab})"
        
        return base_name
    
    def plot_2d(self, ax=None, show_labels=False, show_axis=True):
        from ..viz.topo import plot_probe_2d
        return plot_probe_2d(self, ax=ax, show_labels=show_labels, show_axis=show_axis)

    def plot_3d(self, show_labels=False, reference_landmarks=None, backend='matplotlib', **kwargs):
        """
        Draw the sources and detectors in three dimensions.

        Coregisters the probe to MNI space and renders the optodes and channel
        midpoints on a head surface. No data values are shown.

        Parameters
        ----------
        show_labels : bool
            Annotate each channel midpoint with its label. Default False.
        reference_landmarks : dict, optional
            MNI reference landmarks as {label: (x, y, z)} in mm. Defaults to the
            MNI152 fiducials.
        backend : {'matplotlib', 'pyvista'}
            'matplotlib' draws a static figure; 'pyvista' opens an interactive
            window and requires the ``threed`` extra.
        **kwargs
            Forwarded to :func:`~milob.imaging.surface.plot_probe_3d`, including
            ``surface``, ``surface_alpha``, ``surface_color``, ``view`` and
            ``title``.

        Returns
        -------
        tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes) or pyvista.Plotter

        Examples
        --------
        >>> probe.plot_3d(show_labels=True)
        """
        from ..imaging.surface import plot_probe_3d

        coreg = self.coreg(reference_landmarks=reference_landmarks)

        return plot_probe_3d(
            coreg,
            values=None,
            channel_labels=self.channel_labels if show_labels else None,
            mode='nodes',
            backend=backend,
            show_sources=True,
            show_detectors=True,
            show_labels=show_labels,
            title=kwargs.pop('title', 'fNIRS Probe – 3D Source/Detector Positions'),
            **kwargs
        )

    def plot_rois_3d(self, show_labels=False, reference_landmarks=None,
                      backend='matplotlib', cmap='tab10', **kwargs):
        """
        Draw the channels in three dimensions, coloured by ROI.

        Uses the same coregistration and renderer as :meth:`plot_3d`. Channels
        without an ROI are drawn in grey.

        Parameters
        ----------
        show_labels : bool
            Annotate each channel midpoint with its label. Default False.
        reference_landmarks : dict, optional
            MNI reference landmarks as {label: (x, y, z)} in mm.
        backend : {'matplotlib', 'pyvista'}
            Renderer to use.
        cmap : str
            Qualitative colormap for the ROI colours. Default 'tab10'; use
            'tab20' for more than ten ROIs.
        **kwargs
            Forwarded to :func:`~milob.imaging.surface.plot_probe_3d`.

        Returns
        -------
        tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes) or pyvista.Plotter

        Examples
        --------
        >>> probe.add_roi_by_clustering(4)
        >>> probe.plot_rois_3d(backend='pyvista')
        """
        if not self.rois:
            raise ValueError("No ROIs defined. Use add_roi() or add_roi_by_*() first.")

        from ..imaging.surface import plot_probe_3d

        coreg = self.coreg(reference_landmarks=reference_landmarks)
        roi_map = self.channel_roi_map
        roi_values = [roi_map[ch] if roi_map[ch] is not None else 'unassigned'
                      for ch in self.channel_labels]

        return plot_probe_3d(
            coreg,
            values=roi_values,
            channel_labels=self.channel_labels if show_labels else None,
            mode='nodes',
            backend=backend,
            cmap=cmap,
            show_sources=kwargs.pop('show_sources', True),
            show_detectors=kwargs.pop('show_detectors', True),
            show_labels=show_labels,
            title=kwargs.pop('title', 'fNIRS Probe – ROI Assignment'),
            **kwargs
        )

    def info(self):
        """Print a summary of the probe configuration and metadata."""
        lines = [
            "=== Probe Configuration ===",
            f"Sources:    {self.n_sources}",
            f"Detectors:  {self.n_detectors}",
            f"Wavelengths: {self.wavelengths.tolist()} nm",
            f"Channels:    {self.n_channels if self.has_channels else 'Not Configured'}",
            "",
            "--- Geometry Status ---",
            f"3D Positions: Available ({self.s_pos.shape[1]}D matrix)",
            f"2D Positions: {'Available' if self.s_pos_2d is not None else 'Missing'}",
            f"Landmarks:    {len(self.landmarks) if self.landmarks is not None else 'None'}",
            f"Length Unit:  {self.lengthUnit if self.lengthUnit else 'Not Set'}",
            "",
            "--- Metadata ---",
            f"Source Labels:   {'Custom (e.g., ' + self.source_labels[0] + ')' if self.source_labels else 'Default'}",
            f"Detector Labels: {'Custom (e.g., ' + self.detector_labels[0] + ')' if self.detector_labels else 'Default'}",
        ]
        
        if self.rois:
            lines += ["", "--- ROIs ---"]
            for roi_name, chs in self.rois.items():
                lines.append(f"  {roi_name}: {len(chs)} channel(s)")

        print("\n".join(lines))



    @classmethod
    def from_distances(cls, distances, wavelengths, sc_threshold=None):
        """
        Build a probe for multi-distance simulation.

        Places one source at the origin and detectors along the x-axis at the given
        separations.

        Parameters
        ----------
        distances : array-like
            Source-detector separations in cm.
        wavelengths : array-like
            Wavelengths in nm.
        sc_threshold : float or None, optional
            Distance in mm below which a channel is short. Default None, so every
            channel is classified as long.

        Returns
        -------
        Probe
            One source, N detectors and N channels, with ``lengthUnit='cm'``.

        Examples
        --------
        >>> probe = Probe.from_distances([1.5, 2.0, 2.5, 3.0], [690, 830])
        """
        distances = np.asarray(distances, dtype=float)
        n = len(distances)

        s_pos = np.array([[0.0, 0.0, 0.0]])
        d_pos = np.column_stack([distances, np.zeros(n), np.zeros(n)])

        channels = {
            'sources': [1] * n,
            'detectors': list(range(1, n + 1)),
            'datatypes': [1] * n,
        }

        return cls(
            s_pos=s_pos,
            d_pos=d_pos,
            wavelengths=np.asarray(wavelengths),
            channels=channels,
            lengthUnit='cm',
            sc_threshold=sc_threshold,
        )

    def coreg(self, reference_landmarks=None):
        """
        Fit a coregistration of this probe to MNI space.

        Requires ``landmarks`` and ``landmark_labels``.

        Parameters
        ----------
        reference_landmarks : dict, optional
            Target landmarks as {label: (x, y, z)} in mm. Defaults to the MNI152
            fiducials.

        Returns
        -------
        Coregistration
            Already fitted.

        Examples
        --------
        >>> coreg = probe.coreg()
        >>> coreg.residuals
        """
        from ..imaging.coregistration import Coregistration
        return Coregistration(self, reference_landmarks).fit()

    def label_optodes_10_20(self, *, system='10-10', reference_landmarks=None,
                            surface=None, naming='modern', positions=None):
        """
        Name each optode by the nearest standard 10-20 or 10-10 scalp position.

        Coregisters the probe to MNI space and matches each optode against the
        standard position set. Two optodes may share a nearest position, and each
        result carries the distance to it, so these are descriptions rather than
        identifiers; the join to a stream's channel axis stays on
        ``source_labels`` and ``detector_labels``.

        Parameters
        ----------
        system : {'10-10', '10-20'}
            Position set to match against. '10-20' is the 19-position subset.
        reference_landmarks : dict, optional
            Forwarded to :meth:`coreg`.
        surface : HeadSurface, optional
            Scalp on which to construct the positions. Defaults to the MNI152
            template.
        naming : {'modern', 'legacy'}
            'legacy' returns T3, T4, T5 and T6 rather than T7, T8, P7 and P8.
        positions : dict, optional
            Prebuilt position set from
            :func:`~milob.imaging.ten_twenty.build_scalp_positions`.

        Returns
        -------
        pandas.DataFrame
            Columns name, type, position, distance_mm and mni_x, mni_y, mni_z.

        Examples
        --------
        >>> labels = probe.label_optodes_10_20()
        >>> labels[labels.distance_mm > 15]
        """
        from ..imaging.ten_twenty import label_optodes
        return label_optodes(self, system=system,
                             reference_landmarks=reference_landmarks,
                             surface=surface, naming=naming, positions=positions)


    def __repr__(self):
        has_2d = "Yes" if self.s_pos_2d is not None else "No"
        return (f"<Probe | {self.n_sources}S, {self.n_detectors}D, {len(self.wavelengths)}λ | "
                f"3D: Yes, 2D: {has_2d}>")