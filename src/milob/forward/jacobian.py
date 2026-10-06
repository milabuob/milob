"""
Sensitivity (Jacobian) matrix for linear CW image reconstruction.

Built from the semi-infinite Green's function. Under the first-order Rytov
approximation around a homogeneous baseline ``(mua0, musp0)``::

    J[i, j] = G(s_i, voxel_j) * G(voxel_j, d_i) * voxel_volume_cm3 / phi0_i
    Delta_OD_i ~= sum_j J[i, j] * Delta_mua_j

References
----------
Boas, D. A. (1997). A fundamental limitation of linearized algorithms for
diffuse optical tomography. Optics Express, 1(13), 404-413.

Arridge, S. R., & Schotland, J. C. (2009). Optical tomography: forward and
inverse problems. Inverse Problems, 25(12), 123010.
"""

import numpy as np
from ..imaging import IMAGING_DTYPE

from . import kernels
from . import dispersion


def si_greens_adapter(source_pos, target_pos, *, mua, musp, n, wavelength,
                       R_eff=None, boundary_point, boundary_normal):
    """
    Evaluate the semi-infinite CW Green's function from one optode to targets.

    Each target is split into a lateral distance ``rho`` along the boundary
    plane and a depth ``z`` below it, and passed to
    :func:`~milob.forward.kernels.si_kernel`.

    Parameters
    ----------
    source_pos : array-like, shape (3,)
        Optode position in mm.
    target_pos : array-like, shape (M, 3) or (3,)
        Target positions in mm, e.g. ``VoxelGrid.positions``.
    mua, musp : float
        Baseline absorption and reduced scattering in cm^-1.
    n : float
        Refractive index.
    wavelength : float
        Wavelength in nm.
    R_eff : float, optional
        Effective boundary reflectance. Computed from ``n`` against air if None.
    boundary_point, boundary_normal : array-like, shape (3,)
        Point on, and normal to, the flat boundary plane, in mm.

    Returns
    -------
    np.ndarray, shape (M,)
        Green's function value at each target.
    """
    source_pos = np.asarray(source_pos, dtype=float)
    target_pos = np.atleast_2d(np.asarray(target_pos, dtype=float))
    boundary_point = np.asarray(boundary_point, dtype=float)
    boundary_normal = np.asarray(boundary_normal, dtype=float)
    boundary_normal = boundary_normal / np.linalg.norm(boundary_normal)

    source_depth = (source_pos - boundary_point) @ boundary_normal
    source_in_plane = source_pos - source_depth * boundary_normal

    target_depth = (target_pos - boundary_point) @ boundary_normal
    target_in_plane = target_pos - np.outer(target_depth, boundary_normal)

    rho_mm = np.linalg.norm(target_in_plane - source_in_plane, axis=1)
    rho = rho_mm / 10.0    # mm -> cm, forward.kernels' native unit
    z = target_depth / 10.0

    if R_eff is None:
        R_eff = kernels.Reff(n, 1.0)
    K2 = dispersion.k2(mua, musp, n, wavelength)   # CW: freq=0, tau=0
    K = complex(np.sqrt(K2))

    return kernels.si_kernel(rho, K, mua, musp, R_eff, z=z)


def build_jacobian(probe, voxel_grid, mua0, musp0, n, wavelength, *,
                    greens_fn=si_greens_adapter, R_eff=None,
                    phi0_source="model", phi0_measured=None, channel_mask=None):
    """
    Assemble the sensitivity matrix for one wavelength.

    Parameters
    ----------
    probe : Probe
        Probe with a channel configuration.
    voxel_grid : VoxelGrid
        Voxels the matrix columns index.
    mua0, musp0 : float
        Homogeneous baseline absorption and reduced scattering in cm^-1.
    n : float
        Refractive index.
    wavelength : float
        Wavelength in nm.
    greens_fn : callable, optional
        Green's function with the signature of :func:`si_greens_adapter`.
        Default :func:`si_greens_adapter`.
    R_eff : float, optional
        Effective boundary reflectance. Computed from ``n`` against air if None.
    phi0_source : {"model", "measured"}
        Normalisation of each channel's row: the modelled source-detector
        fluence, or ``phi0_measured``.
    phi0_measured : array-like, shape (n_channels,), optional
        Measured baseline per channel. Required when ``phi0_source="measured"``.
    channel_mask : array-like of bool, shape (n_channels,), optional
        Channels to assemble. Rows outside the mask are zero. Defaults to all.

    Returns
    -------
    dict
        ``matrix`` (n_channels, n_voxels) in OD per cm^-1 of absorption change,
        ``channel_labels``, ``channel_mask``, ``voxel_grid``, ``phi0`` (complex,
        per channel), and the ``wavelength``, ``mua0`` and ``musp0`` used.

    Raises
    ------
    ValueError
        If the probe has no channel configuration, or ``phi0_source`` is
        unknown or lacks ``phi0_measured``.
    """
    if not probe.has_channels:
        raise ValueError("Probe must have channel configuration to build a Jacobian.")

    s_pos = probe.s_pos.astype(float)
    d_pos = probe.d_pos.astype(float)
    s_pos = s_pos * probe.mm_per_unit
    d_pos = d_pos * probe.mm_per_unit

    sources = probe._channels['sources']
    detectors = probe._channels['detectors']
    channel_labels = probe.channel_labels
    n_channels = len(sources)
    n_voxels = voxel_grid.n_voxels

    if channel_mask is None:
        channel_mask = np.ones(n_channels, dtype=bool)
    else:
        channel_mask = np.asarray(channel_mask, dtype=bool)

    if R_eff is None:
        R_eff = kernels.Reff(n, 1.0)

    if phi0_source == "measured":
        if phi0_measured is None:
            raise ValueError("phi0_measured must be given when phi0_source='measured'.")
        phi0 = np.asarray(phi0_measured, dtype=complex).copy()
    elif phi0_source == "model":
        phi0 = np.full(n_channels, np.nan, dtype=complex)
    else:
        raise ValueError(f"phi0_source must be 'model' or 'measured', got {phi0_source!r}.")

    J = np.zeros((n_channels, n_voxels), dtype=float)

    for i in range(n_channels):
        if not channel_mask[i]:
            continue
        s = s_pos[sources[i] - 1]
        d = d_pos[detectors[i] - 1]

        g_sv = greens_fn(s, voxel_grid.positions, mua=mua0, musp=musp0, n=n,
                          wavelength=wavelength, R_eff=R_eff,
                          boundary_point=voxel_grid.boundary_point,
                          boundary_normal=voxel_grid.boundary_normal)
        g_vd = greens_fn(d, voxel_grid.positions, mua=mua0, musp=musp0, n=n,
                          wavelength=wavelength, R_eff=R_eff,
                          boundary_point=voxel_grid.boundary_point,
                          boundary_normal=voxel_grid.boundary_normal)

        if phi0_source == "model":
            g_sd = greens_fn(s, d, mua=mua0, musp=musp0, n=n, wavelength=wavelength,
                              R_eff=R_eff, boundary_point=voxel_grid.boundary_point,
                              boundary_normal=voxel_grid.boundary_normal)
            phi0[i] = g_sd[0]

        J[i, :] = (g_sv * g_vd * voxel_grid.voxel_volume_cm3 / phi0[i]).real

    return {
        # Stored as IMAGING_DTYPE; tikhonov_solve upcasts to float64.
        'matrix': J.astype(IMAGING_DTYPE),
        'channel_labels': channel_labels,
        'channel_mask': channel_mask,
        'voxel_grid': voxel_grid,
        'phi0': phi0,
        'wavelength': wavelength,
        'mua0': mua0,
        'musp0': musp0,
    }
