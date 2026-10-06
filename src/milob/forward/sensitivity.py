"""
Linearised forward operator for image reconstruction.

:class:`SensitivityOperator` holds one sensitivity matrix ``J[channel, voxel]``
per wavelength together with the channel labels and voxel grid that index
it, so that pruning, channel matching and caching keep the three in step.
:meth:`CW_Stream.reconstruct` inverts it, and
:meth:`SensitivityOperator.backproject` projects channel values onto the
voxels without inversion.
"""

import hashlib
import pickle
import warnings
from pathlib import Path

import numpy as np

from ..imaging import IMAGING_DTYPE

#: File-format version for `SensitivityOperator.save` and `.load`.
_FORMAT_VERSION = 1


class SensitivityOperator:
    """
    Sensitivity matrices, one per wavelength, with their channel and voxel axes.

    Usually created with :meth:`build`.

    Parameters
    ----------
    matrices : dict
        ``{wavelength_nm: ndarray (n_channels, n_voxels)}``, stored as
        ``IMAGING_DTYPE``.
    channel_labels : sequence of str
        Channel per row, in row order.
    voxel_grid : VoxelGrid
        Voxel per column.
    channel_mask : array-like of bool, optional
        Active rows. Inactive rows are set to zero, so those channels cannot
        contribute to a reconstruction. Defaults to all active.
    op : str
        Parameter the matrices are derivatives with respect to. Default 'mua'.
    backend : str
        Name of the method that assembled the matrices.
    baseline : dict, optional
        Linearisation point, e.g. ``{'mua0': ..., 'musp0': ..., 'n': ...}``.
    provenance : dict, optional
        Build record, written into the history of reconstructed streams.

    Raises
    ------
    ValueError
        If no matrix is given, the matrices differ in shape, or their shape does
        not match the channel labels, voxel grid or ``channel_mask``.
    """

    def __init__(self, matrices, channel_labels, voxel_grid, *,
                 channel_mask=None, op='mua', backend='unknown',
                 baseline=None, provenance=None):
        self.matrices = {float(wl): np.asarray(m, dtype=IMAGING_DTYPE)
                         for wl, m in matrices.items()}
        if not self.matrices:
            raise ValueError("SensitivityOperator needs at least one matrix.")
        self.channel_labels = [str(c) for c in channel_labels]
        self.voxel_grid = voxel_grid
        self.op = str(op)
        self.backend = str(backend)
        self.baseline = dict(baseline or {})
        self.provenance = dict(provenance or {})

        shapes = {m.shape for m in self.matrices.values()}
        if len(shapes) != 1:
            raise ValueError(
                f"All wavelengths must share one shape, got {sorted(shapes)}."
            )
        n_ch, n_vox = shapes.pop()
        if n_ch != len(self.channel_labels):
            raise ValueError(
                f"Matrices have {n_ch} rows but {len(self.channel_labels)} "
                "channel labels were given."
            )
        if n_vox != voxel_grid.n_voxels:
            raise ValueError(
                f"Matrices have {n_vox} columns but the voxel grid has "
                f"{voxel_grid.n_voxels} voxels."
            )

        if channel_mask is None:
            self.channel_mask = np.ones(n_ch, dtype=bool)
        else:
            self.channel_mask = np.asarray(channel_mask, dtype=bool)
            if self.channel_mask.shape != (n_ch,):
                raise ValueError(
                    f"channel_mask must have shape ({n_ch},), got "
                    f"{self.channel_mask.shape}."
                )

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def wavelengths(self):
        """Built wavelengths in nm, sorted."""
        return sorted(self.matrices)

    @property
    def n_channels(self):
        """Number of channels (rows)."""
        return len(self.channel_labels)

    @property
    def n_voxels(self):
        """Number of voxels (columns)."""
        return self.voxel_grid.n_voxels

    #: Largest gap (nm) over which `matrix` substitutes the nearest built
    #: wavelength.
    WAVELENGTH_TOL = 25.0

    def matrix(self, wavelength):
        """
        Return the matrix at one wavelength.

        A wavelength within ``WAVELENGTH_TOL`` nm of a built one returns the
        nearest matrix, with a warning when they differ.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm.

        Returns
        -------
        np.ndarray, shape (n_channels, n_voxels)

        Raises
        ------
        KeyError
            If the nearest built wavelength is further than ``WAVELENGTH_TOL``.
        """
        wavelength = float(wavelength)
        if wavelength in self.matrices:
            return self.matrices[wavelength]
        nearest = min(self.matrices, key=lambda w: abs(w - wavelength))
        gap = abs(nearest - wavelength)
        if gap > self.WAVELENGTH_TOL:
            raise KeyError(
                f"No sensitivity matrix at {wavelength:g} nm; this operator has "
                f"{self.wavelengths} and the nearest is {gap:g} nm away (tolerance "
                f"{self.WAVELENGTH_TOL:g} nm). Build the operator at the "
                "wavelengths the data actually carries -- "
                "wavelengths=stream.probe.wavelengths -- or raise "
                "SensitivityOperator.WAVELENGTH_TOL if the substitution is "
                "deliberate."
            )
        if gap > 1e-6:
            warnings.warn(
                f"No sensitivity matrix at {wavelength:g} nm; using the nearest "
                f"built, {nearest:g} nm ({gap:g} nm away).",
                UserWarning,
                stacklevel=2,
            )
        return self.matrices[nearest]

    # ------------------------------------------------------------------ #
    # Construction                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def build(cls, probe, voxel_grid, *, wavelengths, mua0, musp0, n=1.4,
              backend='analytical', channel_mask=None, cache=None,
              verbose=True, **backend_kwargs):
        """
        Assemble one sensitivity matrix per wavelength.

        Uses the semi-infinite Green's function of
        :func:`~milob.forward.jacobian.build_jacobian` around a homogeneous
        baseline.

        Parameters
        ----------
        probe : Probe
            Probe with a channel configuration. Defines the rows.
        voxel_grid : VoxelGrid
            Voxels that define the columns.
        wavelengths : sequence of float
            Wavelengths in nm, typically ``stream.data.wavelength.values``.
        mua0, musp0 : float
            Baseline absorption and reduced scattering in cm^-1.
        n : float
            Refractive index. Default 1.4.
        backend : {'analytical'}
            Assembly method. Only 'analytical' is available.
        channel_mask : array-like of bool, optional
            Channels to include. Use the same mask as for the voxel grid.
        cache : str or Path, optional
            Directory or file to read a matching operator from and write this one
            to. The cache key covers the channels, grid, wavelengths, baseline and
            mask.
        verbose : bool
            Print cache activity. Default True.
        **backend_kwargs
            Passed to :func:`~milob.forward.jacobian.build_jacobian`, e.g.
            ``phi0_source``.

        Returns
        -------
        SensitivityOperator

        Raises
        ------
        ValueError
            If ``backend`` is not 'analytical'.
        """
        if backend != 'analytical':
            raise ValueError(f"Unknown backend {backend!r}; expected 'analytical'.")
        wavelengths = [float(w) for w in np.atleast_1d(wavelengths)]
        channel_labels = [str(c) for c in probe.channel_labels]
        if channel_mask is not None:
            channel_mask = np.asarray(channel_mask, dtype=bool)

        key = None
        if cache is not None:
            key = _cache_key(channel_labels, voxel_grid, wavelengths, backend,
                             channel_mask, mua0, musp0, n)
            path = _cache_path(cache, key)
            if path.exists():
                op = cls.load(path)
                if verbose:
                    print(f"SensitivityOperator: loaded cached {path.name}")
                return op

        operator = cls._build_analytical(
            probe, voxel_grid, wavelengths, channel_labels, channel_mask,
            mua0=mua0, musp0=musp0, n=n, **backend_kwargs)

        if cache is not None:
            path = _cache_path(cache, key)
            operator.provenance['cache_key'] = key
            operator.save(path)
            if verbose:
                print(f"SensitivityOperator: cached to {path}")
        return operator

    @classmethod
    def _build_analytical(cls, probe, voxel_grid, wavelengths, channel_labels,
                          channel_mask, *, mua0, musp0, n, **kwargs):
        """Build the matrices with the semi-infinite Green's function."""
        from .jacobian import build_jacobian
        if mua0 is None or musp0 is None:
            raise ValueError(
                "The analytical backend needs an explicit homogeneous baseline: "
                "pass mua0 and musp0 (cm^-1)."
            )
        matrices = {}
        for wl in wavelengths:
            jac = build_jacobian(probe, voxel_grid, mua0=mua0, musp0=musp0, n=n,
                                 wavelength=wl, channel_mask=channel_mask, **kwargs)
            matrices[wl] = jac['matrix']
        return cls(matrices, channel_labels, voxel_grid,
                   channel_mask=channel_mask, op='mua', backend='analytical',
                   baseline={'mua0': mua0, 'musp0': musp0, 'n': n},
                   provenance={'geometry': 'semi-infinite'})

    # ------------------------------------------------------------------ #
    # Operations -- each moves J, grid and labels together                #
    # ------------------------------------------------------------------ #

    def _replace(self, **changes):
        """Return a copy with some fields replaced."""
        fields = {
            'matrices': self.matrices, 'channel_labels': self.channel_labels,
            'voxel_grid': self.voxel_grid, 'channel_mask': self.channel_mask,
            'op': self.op, 'backend': self.backend,
            'baseline': dict(self.baseline), 'provenance': dict(self.provenance),
        }
        fields.update(changes)
        matrices = fields.pop('matrices')
        channel_labels = fields.pop('channel_labels')
        voxel_grid = fields.pop('voxel_grid')
        return SensitivityOperator(matrices, channel_labels, voxel_grid, **fields)

    def coverage(self, wavelength=None):
        """
        Return the squared column norm of each voxel.

        Parameters
        ----------
        wavelength : float, optional
            Wavelength in nm. If None, the minimum across wavelengths.

        Returns
        -------
        np.ndarray, shape (n_voxels,)
        """
        if wavelength is not None:
            return (self.matrix(wavelength).astype(np.float64) ** 2).sum(axis=0)
        stack = [(m.astype(np.float64) ** 2).sum(axis=0)
                 for m in self.matrices.values()]
        return np.min(np.vstack(stack), axis=0)

    def restrict_to_voxels(self, keep, *, reason='explicit'):
        """
        Keep a subset of voxels in every matrix and in the voxel grid.

        Parameters
        ----------
        keep : array-like of bool
            One entry per voxel.
        reason : str
            Recorded in the provenance of the returned operator.

        Returns
        -------
        SensitivityOperator

        Raises
        ------
        ValueError
            If ``keep`` has the wrong length or removes every voxel.

        Examples
        --------
        >>> seen = np.mean([op.coverage() > 1e-3 * op.coverage().max()
        ...                 for op in per_subject], axis=0)
        >>> shared = reference.restrict_to_voxels(seen >= 0.8, reason='group coverage')
        """
        keep = np.asarray(keep, dtype=bool)
        if keep.shape != (self.n_voxels,):
            raise ValueError(
                f"keep must have one entry per voxel ({self.n_voxels},), got "
                f"{keep.shape}."
            )
        if not keep.any():
            raise ValueError("keep would remove every voxel.")

        grid = self.voxel_grid.restrict(keep)
        matrices = {wl: m[:, keep] for wl, m in self.matrices.items()}
        prov = dict(self.provenance)
        prov['voxel_prune'] = {'reason': reason,
                               'kept': int(keep.sum()),
                               'from': int(keep.size)}
        return self._replace(matrices=matrices, voxel_grid=grid, provenance=prov)

    def restrict_to_coverage(self, threshold=1e-3, verbose=False):
        """
        Keep the voxels whose coverage exceeds a fraction of the peak.

        Parameters
        ----------
        threshold : float
            Fraction of the largest :meth:`coverage` a voxel must exceed.
            Default 1e-3.
        verbose : bool
            Print the number of voxels kept. Default False.

        Returns
        -------
        SensitivityOperator

        Raises
        ------
        ValueError
            If the operator has no positive sensitivity, or no voxel passes.
        """
        coverage = self.coverage()
        peak = coverage.max()
        if not np.isfinite(peak) or peak <= 0:
            raise ValueError("This operator has no positive sensitivity anywhere.")
        keep = coverage > threshold * peak
        if not keep.any():
            raise ValueError(
                f"threshold={threshold:g} would remove every voxel; the "
                f"largest relative coverage is 1.0 by construction, so try a "
                "smaller threshold."
            )
        out = self.restrict_to_voxels(keep, reason=f'coverage > {threshold:g} of peak')
        out.provenance['coverage_prune'] = {'threshold': threshold,
                                            'kept': int(keep.sum()),
                                            'from': int(keep.size)}
        if verbose:
            print(f"coverage prune: {keep.size:,} -> {int(keep.sum()):,} voxels")
        return out

    # ------------------------------------------------------------------ #
    # Display projections -- J^T, never J^-1                              #
    # ------------------------------------------------------------------ #

    def normalised(self, wavelength=None):
        """
        Return the sensitivity with each channel's row scaled to its own peak.

        Parameters
        ----------
        wavelength : float, optional
            Wavelength in nm. If None, the normalised rows are averaged across
            wavelengths. The result is for display and back-projection, not for a
            solver.

        Returns
        -------
        np.ndarray, shape (n_channels, n_voxels)
        """
        if wavelength is not None:
            mats = [self.matrix(wavelength)]
        else:
            mats = [self.matrices[wl] for wl in self.wavelengths]

        out = np.zeros((self.n_channels, self.n_voxels), dtype=np.float64)
        for m in mats:
            m = np.abs(np.asarray(m, dtype=np.float64))
            peak = m.max(axis=1, keepdims=True)
            out += np.divide(m, peak, out=np.zeros_like(m), where=peak > 0)
        return out / len(mats)

    def backproject(self, values, channel_labels=None, *, mode='weighted',
                    threshold=1e-2, wavelength=None):
        """
        Spread per-channel values over the voxels each channel is sensitive to.

        Applies the transpose of the normalised sensitivity; nothing is inverted.
        The result is a topographic map. For a tomographic image use
        :meth:`CW_Stream.reconstruct`.

        Parameters
        ----------
        values : array-like, shape (n_channels,)
            One value per channel, e.g. a beta or t-statistic. NaN channels are
            left out.
        channel_labels : sequence of str, optional
            Labels for ``values``, matched to the operator's channels by name.
            Required unless ``values`` follows ``self.channel_labels`` exactly.
        mode : {'weighted', 'winner'}
            How overlapping channels combine at a voxel. 'weighted' takes a mean
            weighted by ``log10(sensitivity / threshold)``; 'winner' takes the value
            of the most sensitive channel.
        threshold : float
            Normalised sensitivity a channel must exceed to contribute to a voxel.
            Default 1e-2. Voxels no channel reaches are NaN.
        wavelength : float, optional
            Use one wavelength instead of the average (see :meth:`normalised`).

        Returns
        -------
        np.ndarray, shape (n_voxels,)
            Values aligned with ``voxel_grid.positions``.

        Raises
        ------
        ValueError
            If ``values`` and the labels do not match the operator's channels, no
            channel has a finite value, or ``mode`` is unknown.
        """
        values = np.asarray(values, dtype=float).ravel()

        if channel_labels is None:
            if values.size != self.n_channels:
                raise ValueError(
                    f"values has {values.size} entries but this operator has "
                    f"{self.n_channels} channels. Pass channel_labels=... so "
                    "they can be matched by name rather than by position."
                )
            order = np.arange(self.n_channels)
        else:
            labels = [str(c) for c in channel_labels]
            if len(labels) != values.size:
                raise ValueError(
                    f"channel_labels has {len(labels)} entries but values has "
                    f"{values.size}."
                )
            index = {lab: i for i, lab in enumerate(labels)}
            missing = [c for c in self.channel_labels if c not in index]
            if missing:
                raise ValueError(
                    f"{len(missing)} of this operator's channels are absent "
                    f"from channel_labels (e.g. {missing[:5]}). Build the "
                    "operator for the channels you are plotting -- "
                    "SensitivityOperator.for_probe() slices it to match."
                )
            order = np.array([index[c] for c in self.channel_labels])

        x = values[order]
        keep = np.isfinite(x) & self.channel_mask
        if not keep.any():
            raise ValueError(
                "No channel survives: every value is NaN, or the operator's "
                "channel_mask excludes them all. With sig_mask=True this "
                "means nothing reached significance."
            )

        threshold = float(threshold)
        sens = self.normalised(wavelength)[keep]
        x = x[keep]

        out = np.full(self.n_voxels, np.nan, dtype=np.float64)
        if mode == 'weighted':
            # Weight = decades above the threshold, zero below it.
            ratio = np.divide(sens, threshold, out=np.zeros_like(sens),
                              where=sens > 0)
            weights = np.zeros_like(sens)
            np.log10(ratio, out=weights, where=ratio > 1.0)
            denom = weights.sum(axis=0)
            np.divide(x @ weights, denom, out=out, where=denom > 0)
        elif mode == 'winner':
            sens = np.where(sens > threshold, sens, 0.0)
            reached = sens.max(axis=0) > 0
            out[reached] = x[sens.argmax(axis=0)[reached]]
        else:
            raise ValueError(
                f"Unknown mode {mode!r}; expected 'weighted' or 'winner'."
            )
        return out

    def plot_coverage_3d(self, *, wavelength=None, threshold=1e-3, log=True,
                         surface=None, views=('left', 'right'),
                         backend='matplotlib', **kwargs):
        """
        Plot the array's coverage on the cortex.

        Shows :meth:`coverage` relative to its peak.

        Parameters
        ----------
        wavelength : float, optional
            Wavelength in nm. If None, the minimum across wavelengths.
        threshold : float
            Hide voxels below this fraction of peak coverage. Default 1e-3.
        log : bool
            Plot log10 of the relative coverage. Default True.
        surface, views, backend, **kwargs
            Passed to :func:`~milob.imaging.surface.plot_surface_map`.

        Returns
        -------
        tuple or pyvista.Plotter
            ``(fig, axes)`` for matplotlib, or the plotter for pyvista.

        Raises
        ------
        ValueError
            If the operator has no positive sensitivity, or no voxel passes
            ``threshold``.
        """
        from ..imaging.surface import plot_surface_map
        from ..viz import theme

        cov = np.asarray(self.coverage(wavelength), dtype=float)
        peak = cov.max()
        if not np.isfinite(peak) or peak <= 0:
            raise ValueError("This operator has no positive sensitivity anywhere.")

        rel = np.where(cov > float(threshold) * peak, cov / peak, np.nan)
        vals = np.log10(rel) if log else rel
        if not np.isfinite(vals).any():
            raise ValueError(
                f"threshold={threshold:g} leaves no voxel; the largest "
                "relative coverage is 1.0 by construction, so try smaller."
            )

        kwargs.setdefault('cmap', theme.SEQUENTIAL_CMAP)
        kwargs.setdefault('symmetric', False)
        kwargs.setdefault('colorbar_label',
                          'log10 sensitivity / peak' if log else 'sensitivity / peak')
        kwargs.setdefault('title',
                          f"Array coverage (above {threshold:g} of peak)")
        return plot_surface_map(vals, self.voxel_grid, surface=surface,
                                views=views, backend=backend, **kwargs)

    def for_probe(self, target, *, require_all=True, mask=None):
        """
        Reorder the rows to match another channel list, by label.

        Channels in ``target`` that the operator does not have get zero rows and
        are inactive.

        Parameters
        ----------
        target : CW_Stream, Probe or sequence of str
            Channel list to match.
        require_all : bool
            Raise if ``target`` has channels the operator does not have.
            Default True.
        mask : array-like of bool, optional
            Extra mask per target channel, combined with the operator's own.

        Returns
        -------
        SensitivityOperator
            Same voxel grid, rows in the order of ``target``.

        Raises
        ------
        ValueError
            If channels are missing and ``require_all`` is True, ``mask`` has the
            wrong length, or no channel remains active.
        """
        labels = _channel_labels_of(target)
        index = {lab: i for i, lab in enumerate(self.channel_labels)}
        missing = [lab for lab in labels if lab not in index]
        if missing and require_all:
            raise ValueError(
                f"{len(missing)} channel(s) of the target probe are absent from "
                f"this operator, e.g. {missing[:5]}. That is a genuine geometry "
                "mismatch -- the operator was built for a different montage. "
                "Pass require_all=False to zero those rows instead."
            )

        rows = np.array([index.get(lab, -1) for lab in labels])
        known = rows >= 0
        safe_rows = np.where(known, rows, 0)

        matrices = {}
        for wl, m in self.matrices.items():
            picked = m[safe_rows]
            picked[~known] = 0.0
            matrices[wl] = picked

        new_mask = self.channel_mask[safe_rows] & known
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != new_mask.shape:
                raise ValueError(
                    f"mask must have one entry per target channel "
                    f"({new_mask.shape[0]}), got {mask.shape}."
                )
            new_mask = new_mask & mask
        for wl in matrices:
            matrices[wl][~new_mask] = 0.0

        # Fail here rather than in tikhonov_solve with an unrelated message.
        if not new_mask.any():
            why = ("this operator has no active channels to begin with"
                   if not self.channel_mask.any() else
                   f"none of the target's {len(labels)} channels is both known "
                   f"to this operator ({int(known.sum())} are) and active in "
                   f"the mask")
            raise ValueError(
                f"for_probe(): no channel survives the match -- {why}. The "
                "usual cause is a per-subject quality mask that removed "
                "everything (check how many channels passed screening), or a "
                "mask built from a different channel band than the operator's. "
                "Reconstructing from zero measurements is not defined."
            )

        prov = dict(self.provenance)
        prov['matched_to'] = {'n_channels': len(labels),
                              'n_active': int(new_mask.sum()),
                              'n_unknown': int((~known).sum())}
        return self._replace(matrices=matrices, channel_labels=labels,
                             channel_mask=new_mask, provenance=prov)

    def resolution_diag(self, wavelength=None, alpha=None, lambda1=None,
                        lambda2=None, method='svd'):
        """
        Return the diagonal of the resolution matrix.

        Values near 1 mean a voxel is recovered as itself; values near 0 mean it
        is blurred into its neighbours by the regularisation.

        Parameters
        ----------
        wavelength : float, optional
            Wavelength in nm. Defaults to the first built.
        alpha, lambda1, lambda2, method
            Passed to :func:`~milob.processing.fitting.tikhonov_solve`.

        Returns
        -------
        np.ndarray, shape (n_voxels,)
        """
        from ..processing.fitting import tikhonov_solve
        wl = self.wavelengths[0] if wavelength is None else wavelength
        J = self.matrix(wl)
        y = np.zeros((J.shape[0], 1))
        _, _, info = tikhonov_solve(J, y, alpha=alpha, lambda1=lambda1,
                                    lambda2=lambda2, method=method,
                                    return_uncertainty=True)
        return info['resolution_diag']

    # ------------------------------------------------------------------ #
    # Bridge to the existing jacobian_fn contract                         #
    # ------------------------------------------------------------------ #

    def as_jacobian_fn(self):
        """
        Return a ``jacobian_fn`` callable for :meth:`CW_Stream.reconstruct`.

        Returns
        -------
        callable
            ``(probe, voxel_grid, *, wavelength, **kwargs) -> dict`` in the format
            of :func:`~milob.forward.jacobian.build_jacobian`.
        """
        def jacobian_fn(probe, voxel_grid, *, wavelength, **_ignored):
            return {
                'matrix': self.matrix(wavelength),
                'channel_labels': self.channel_labels,
                'channel_mask': self.channel_mask,
                'voxel_grid': self.voxel_grid,
                'phi0': None,
                'wavelength': float(wavelength),
                'mua0': self.baseline.get('mua0'),
                'musp0': self.baseline.get('musp0'),
            }
        jacobian_fn.__name__ = f"SensitivityOperator[{self.backend}]"
        return jacobian_fn

    def history_entry(self):
        """Return the build record written into a reconstructed stream's history."""
        return {
            'backend': self.backend,
            'op': self.op,
            'wavelengths': self.wavelengths,
            'n_channels': self.n_channels,
            'n_active_channels': int(self.channel_mask.sum()),
            'n_voxels': self.n_voxels,
            'baseline': dict(self.baseline),
            **{f'jacobian_{k}': v for k, v in self.provenance.items()},
        }

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path):
        """
        Write the operator to disk.

        Parameters
        ----------
        path : str or Path
            Output file. Parent directories are created.

        Returns
        -------
        Path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'version': _FORMAT_VERSION,
            'matrices': self.matrices,
            'channel_labels': self.channel_labels,
            'voxel_grid': self.voxel_grid,
            'channel_mask': self.channel_mask,
            'op': self.op,
            'backend': self.backend,
            'baseline': self.baseline,
            'provenance': self.provenance,
        }
        with open(path, 'wb') as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path):
        """
        Read an operator written by :meth:`save`.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        SensitivityOperator

        Raises
        ------
        ValueError
            If the file was written by a different format version.
        """
        with open(path, 'rb') as fh:
            payload = pickle.load(fh)
        version = payload.get('version')
        if version != _FORMAT_VERSION:
            raise ValueError(
                f"{path} was written by format version {version}, this milob "
                f"reads version {_FORMAT_VERSION}. Rebuild the operator."
            )
        return cls(payload['matrices'], payload['channel_labels'],
                   payload['voxel_grid'], channel_mask=payload['channel_mask'],
                   op=payload['op'], backend=payload['backend'],
                   baseline=payload['baseline'], provenance=payload['provenance'])

    def __repr__(self):
        wl = ', '.join(f"{w:g}" for w in self.wavelengths)
        return (f"<SensitivityOperator {self.backend} d/d({self.op}) | "
                f"{self.n_channels} channels "
                f"({int(self.channel_mask.sum())} active) x "
                f"{self.n_voxels:,} voxels | {wl} nm>")


# ---------------------------------------------------------------------- #
# Helpers                                                                 #
# ---------------------------------------------------------------------- #

def _channel_labels_of(target):
    """Return the channel labels of a stream, a probe or a sequence."""
    if hasattr(target, 'data') and hasattr(target.data, 'channel'):
        return [str(c) for c in target.data.channel.values]
    if hasattr(target, 'channel_labels'):
        return [str(c) for c in target.channel_labels]
    return [str(c) for c in target]


def _cache_key(channel_labels, voxel_grid, wavelengths, backend, channel_mask,
               mua0, musp0, n):
    """Return a content hash of everything that determines the matrices."""
    h = hashlib.sha1()
    h.update('|'.join(channel_labels).encode())
    h.update(np.ascontiguousarray(voxel_grid.positions, dtype=np.float64).tobytes())
    h.update(np.float64(voxel_grid.spacing).tobytes())
    if voxel_grid.origin is not None:
        h.update(np.ascontiguousarray(voxel_grid.origin, dtype=np.float64).tobytes())
    h.update(str(voxel_grid.grid_shape).encode())
    h.update(','.join(f"{w:g}" for w in sorted(wavelengths)).encode())
    h.update(backend.encode())
    h.update(np.array([mua0, musp0, n], dtype=np.float64).tobytes())
    if channel_mask is not None:
        h.update(np.ascontiguousarray(channel_mask, dtype=bool).tobytes())
    return h.hexdigest()[:16]


def _cache_path(cache, key):
    cache = Path(cache)
    if cache.suffix:
        return cache
    return cache / f"sensitivity_{key}.pkl"
