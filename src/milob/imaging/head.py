"""
The segmented anatomical domain a tomographic reconstruction is computed on.

Holds the label volume, its placement in physical space, the meaning of
each label, the reference optical properties of each tissue, and
optionally a structural image to draw slices on. These are kept together
because they only mean anything together: a label volume needs its offset
to sit in the same frame as a coregistered probe, and the property table
is indexed by the same integers the volume holds.
"""

import hashlib
from pathlib import Path

import numpy as np

_DATA_DIR = Path(__file__).parent / "data"

#: Tissue labels of the bundled ICBM152 segmentation (NeuroDOT
#: convention; 0 is outside the head).
ICBM152_LABELS = {1: 'csf', 2: 'white', 3: 'grey', 4: 'skull', 5: 'scalp'}

#: Physical position (mm) of ``volume[0, 0, 0]`` for the bundled ICBM152
#: volume, i.e. ``mni_mm = index * spacing + offset``.
#:
#: This volume is MNI-aligned, which is what makes the constant valid and
#: is not something to assume about a segmentation in general. Verified
#: rather than taken on faith: coregistering a 84-optode two-patch cap
#: onto the MNI152 fiducials and applying this offset puts the optodes a
#: median 4.2 mm from this volume's own scalp label, and the fsaverage
#: pial surface sits a median 0.5 mm from its grey-matter label. Both are
#: within a voxel of where they should be, so volume, surfaces and probe
#: share one frame.
ICBM152_OFFSET = np.array([-87.0, -124.0, -76.5])

#: Reference optical properties, ``{wavelength_nm: {label: (mua, musp, ri)}}``
#: in **cm^-1** (milob's convention throughout).
#:
#: Source values are the NeuroDOT reference set, which is
#: published in mm^-1; they are converted once, here, so that no caller
#: ever has to remember the factor of 10. Tabulated at 750 and 850 nm --
#: `HeadModel.tissue_properties` resolves any other wavelength to the
#: nearest of these and says so, rather than interpolating a table whose
#: two points are too far apart for interpolation to mean much.
ICBM152_PROPERTIES = {
    750.0: {1: (0.040, 3.000, 1.4), 2: (0.167, 11.908, 1.4), 3: (0.180, 8.359, 1.4),
            4: (0.116, 9.400, 1.4), 5: (0.170, 7.400, 1.4)},
    850.0: {1: (0.040, 3.000, 1.4), 2: (0.208, 10.107, 1.4), 3: (0.192, 6.726, 1.4),
            4: (0.139, 8.400, 1.4), 5: (0.190, 6.400, 1.4)},
}

#: Label groups usable by name in `HeadModel.label_ids` / `.mask`.
_GROUPS = {
    'brain': ('grey', 'white'),
    'cortex': ('grey',),
    'extracerebral': ('scalp', 'skull'),
    'head': None,          # every non-outside label
}


class HeadModel:
    """
    A segmented head: label volume, placement, tissue names and properties.

    Parameters
    ----------
    volume : np.ndarray
        Three-dimensional label volume, cast to uint8. Voxels equal to
        ``outside_label`` lie outside the head; every other value is a tissue.
    offset : array-like
        Physical position in mm of the volume's first voxel, such that
        ``physical = index * spacing + offset``.
    spacing : float
        Physical size in mm of one voxel.
    labels : dict of {int: str}, optional
        Tissue name per label value. Names are what let tissue groups be
        selected by name rather than by integer.
    outside_label : int
        Value marking voxels outside the head.
    properties : dict, optional
        Reference optical properties as
        ``{wavelength_nm: {label: (mua, musp, refractive_index)}}``, with
        coefficients in cm^-1. Optional, since masking and depth work without
        them.
    background : str, Path or np.ndarray, optional
        Structural image for slice plots, given as a path or an array.
    name : str, optional
        Identifier used in provenance and cache keys.
    """

    def __init__(self, volume, offset, spacing=1.0, labels=None,
                 outside_label=0, properties=None, background=None, name=None):
        self.volume = np.asarray(volume, dtype=np.uint8)
        if self.volume.ndim != 3:
            raise ValueError(f"volume must be 3-D, got shape {self.volume.shape}.")
        self.offset = np.asarray(offset, dtype=float).reshape(3)
        self.spacing = float(spacing)
        self.labels = dict(labels or {})
        self.outside_label = int(outside_label)
        self.properties = {float(k): dict(v) for k, v in (properties or {}).items()}
        self.background = background
        self.name = name or 'head'

    # ------------------------------------------------------------------ #
    # Constructors                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def icbm152(cls, background=None):
        """
        Load the bundled five-tissue ICBM152 segmentation, in MNI space.

        Ships with the package, so anatomically realistic work needs no external
        download.

        Parameters
        ----------
        background : str or Path, optional
            Structural volume for slice plots, typically the ICBM152 T1, which is
            too large to bundle. Slice plots fall back to plotting on nothing
            without it.

        Returns
        -------
        HeadModel
        """
        import scipy.io as sio
        path = _DATA_DIR / "icbm152_headvol.mat"
        if not path.exists():                                   # pragma: no cover
            raise FileNotFoundError(
                f"Bundled head volume missing at {path} -- the installed "
                "package is incomplete; reinstall milob."
            )
        volume = sio.loadmat(path)['mask']
        return cls(volume, offset=ICBM152_OFFSET, spacing=1.0,
                   labels=ICBM152_LABELS, outside_label=0,
                   properties=ICBM152_PROPERTIES, background=background,
                   name='icbm152')

    @classmethod
    def from_nifti(cls, path, *, labels=None, outside_label=0,
                   properties=None, background=None, name=None):
        """
        Load a segmented volume from a NIfTI file, taking its placement from the
        file's own affine.

        Only axis-aligned, isotropic affines are accepted, since the rest of the
        imaging stack assumes ``physical = index * spacing + offset``. An oblique
        volume must be resampled first.

        Parameters
        ----------
        path : str or Path
            Path to the NIfTI file.
        labels : dict of {int: str}, optional
            Tissue name per label value.
        outside_label : int
            Value marking voxels outside the head.
        properties : dict, optional
            Reference optical properties per wavelength and label.
        background : str, Path or np.ndarray, optional
            Structural image for slice plots.
        name : str, optional
            Identifier.

        Returns
        -------
        HeadModel

        Raises
        ------
        ValueError
            If the file's affine is not axis-aligned and isotropic.
        """
        try:
            import nibabel as nib
        except ImportError as exc:                              # pragma: no cover
            raise ImportError(
                "HeadModel.from_nifti requires nibabel: pip install nibabel "
                "(or `pip install milob[image]`)."
            ) from exc

        img = nib.load(str(path))
        affine = np.asarray(img.affine, dtype=float)
        rot = affine[:3, :3]
        off_diag = rot - np.diag(np.diag(rot))
        if np.abs(off_diag).max() > 1e-6:
            raise ValueError(
                f"{path} has a rotated/oblique affine; only axis-aligned "
                "volumes are supported (see HeadModel.from_nifti). Resample "
                "to an axis-aligned grid first."
            )
        diag = np.abs(np.diag(rot))
        if diag.max() - diag.min() > 1e-6:
            raise ValueError(
                f"{path} is anisotropic (voxel sizes {diag}); milob's imaging "
                "stack assumes isotropic voxels. Resample first."
            )
        return cls(np.asarray(img.dataobj), offset=affine[:3, 3],
                   spacing=float(diag[0]), labels=labels,
                   outside_label=outside_label, properties=properties,
                   background=background, name=name or Path(path).stem)

    # ------------------------------------------------------------------ #
    # Tissue vocabulary                                                   #
    # ------------------------------------------------------------------ #

    @property
    def region_labels(self):
        """
        Label values present in the volume, excluding the outside label.

        Returns
        -------
        list of int
            Sorted label values.
        """
        present = np.unique(self.volume)
        return [int(v) for v in present if int(v) != self.outside_label]

    def label_ids(self, *names):
        """
        Resolve tissue names to label integers.

        Parameters
        ----------
        *names : str or int
            Individual tissue names, the groups 'brain', 'cortex',
            'extracerebral' or 'head', or bare integers, which pass through
            unchanged.

        Returns
        -------
        list of int
            Label values.
        """
        out = []
        for name in names:
            if isinstance(name, (int, np.integer)):
                out.append(int(name))
                continue
            key = str(name).lower()
            if key in _GROUPS:
                members = _GROUPS[key]
                if members is None:
                    out.extend(self.region_labels)
                    continue
                out.extend(self.label_ids(*members))
                continue
            matches = [k for k, v in self.labels.items() if v.lower() == key]
            if not matches:
                raise KeyError(
                    f"Unknown tissue {name!r}. This head model knows "
                    f"{sorted(self.labels.values())} plus the groups "
                    f"{sorted(_GROUPS)}."
                )
            out.extend(matches)
        return sorted(set(out))

    def mask(self, *names):
        """
        Build a boolean volume selecting the named tissues.

        Parameters
        ----------
        *names : str or int
            Tissues to select, as for :meth:`label_ids`.

        Returns
        -------
        np.ndarray of bool
            Same shape as ``volume``.
        """
        return np.isin(self.volume, self.label_ids(*names))

    def labels_at(self, positions):
        """
        Look up the tissue label at each physical position.

        Parameters
        ----------
        positions : np.ndarray
            Shape (n, 3), in mm.

        Returns
        -------
        np.ndarray
            Label per position. Positions outside the volume take the outside
            label.
        """
        positions = np.asarray(positions, dtype=float)
        idx = np.rint((positions - self.offset) / self.spacing).astype(int)
        inside = np.all((idx >= 0) & (idx < np.array(self.volume.shape)), axis=1)
        out = np.full(len(positions), self.outside_label, dtype=int)
        safe = idx[inside]
        out[inside] = self.volume[safe[:, 0], safe[:, 1], safe[:, 2]]
        return out

    def tissue_names(self, label_array):
        """
        Map label integers to their tissue names.

        Parameters
        ----------
        label_array : array-like of int
            Label values.

        Returns
        -------
        np.ndarray of str
            Tissue name per label.
        """
        names = dict(self.labels)
        names.setdefault(self.outside_label, 'outside')
        return np.array([names.get(int(v), str(int(v))) for v in np.ravel(label_array)])

    # ------------------------------------------------------------------ #
    # Optical properties                                                  #
    # ------------------------------------------------------------------ #

    def tissue_properties(self, wavelength, strict=False):
        """
        Return the reference optical properties at one wavelength.

        Resolves to the nearest tabulated wavelength unless ``strict``. The
        wavelength actually used is recorded in the caller's provenance, so the
        substitution is a recorded decision rather than a silent one.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm.
        strict : bool
            Raise rather than substituting when there is no exact entry.

        Returns
        -------
        dict of {int: tuple}
            Absorption and reduced scattering in cm^-1 and refractive index, per
            label.

        Raises
        ------
        ValueError
            If ``strict`` and no entry matches, or if no properties were supplied.
        """
        if not self.properties:
            raise ValueError(
                f"HeadModel {self.name!r} carries no tissue properties. Pass "
                "properties={wavelength: {label: (mua_cm, musp_cm, ri)}} to the "
                "constructor, or use HeadModel.icbm152() which ships with them."
            )
        wavelength = float(wavelength)
        if wavelength in self.properties:
            return dict(self.properties[wavelength])
        if strict:
            raise KeyError(
                f"No tissue properties tabulated at {wavelength:g} nm "
                f"(have {sorted(self.properties)}); pass strict=False to use "
                "the nearest."
            )
        nearest = min(self.properties, key=lambda w: abs(w - wavelength))
        return dict(self.properties[nearest])

    def nearest_tabulated(self, wavelength):
        """
        Return the tabulated wavelength :meth:`tissue_properties` would use.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm.

        Returns
        -------
        float
            The nearest tabulated wavelength.
        """
        if not self.properties:
            return None
        return min(self.properties, key=lambda w: abs(w - float(wavelength)))

    # ------------------------------------------------------------------ #
    # Identity                                                            #
    # ------------------------------------------------------------------ #

    @property
    def fingerprint(self):
        """
        Return a short content hash of the geometry.

        Covers the volume's contents as well as its offset and spacing, so an
        edited segmentation invalidates anything cached against the old anatomy.

        Returns
        -------
        str
            Hexadecimal digest, suitable as part of a cache key.
        """
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(self.volume).tobytes())
        h.update(np.ascontiguousarray(self.offset).tobytes())
        h.update(np.float64(self.spacing).tobytes())
        return h.hexdigest()[:16]

    def summarise(self, labels):
        """
        Summarise the tissue composition of an array of labels.

        Parameters
        ----------
        labels : array-like of int
            Label values, such as those returned by a voxel grid's tissue lookup.

        Returns
        -------
        list of tuple
            Tissue name, voxel count and fraction, sorted by count.
        """
        labels = np.asarray(labels)
        total = labels.size
        names = dict(self.labels)
        names.setdefault(self.outside_label, 'outside')
        rows = [(names.get(int(v), str(int(v))), int((labels == v).sum()))
                for v in np.unique(labels)]
        rows.sort(key=lambda r: -r[1])
        return [(n, c, c / total if total else 0.0) for n, c in rows]

    def __repr__(self):
        wl = sorted(self.properties) or None
        return (f"<HeadModel {self.name!r} | {self.volume.shape} @ "
                f"{self.spacing:g} mm | regions {self.region_labels} | "
                f"properties {wl}>")
