"""
Cluster-level inference on voxel maps by non-stationary random field theory.

Cluster size is measured in resels using a local resels-per-voxel (RPV) map
estimated from residual images, and each cluster receives a family-wise
error corrected p-value. The residual images must share the noise structure
of the t-map: weighted subject deviations for a group model, or temporal
residuals for a single subject.

References
----------
Worsley, K. J., Marrett, S., Neelin, P., Vandal, A. C., Friston, K. J., &
Evans, A. C. (1996). A unified statistical approach for determining
significant signals in images of cerebral activation. Human Brain Mapping,
4(1), 58-73.

Hayasaka, S., Phan, K. L., Liberzon, I., Worsley, K. J., & Nichols, T. E.
(2004). Nonstationary cluster-size inference with random field and
permutation methods. NeuroImage, 22(2), 676-687.

Hassanpour, M. S., White, B. R., Eggebrecht, A. T., Ferradal, S. L.,
Snyder, A. Z., & Culver, J. P. (2014). Statistical analysis of high density
diffuse optical tomography. NeuroImage, 85, 104-116.
"""

import itertools
import warnings

import numpy as np
from scipy import sparse, stats
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.special import gamma, gammaln

_FOUR_LN2 = 4.0 * np.log(2.0)
_TWO_LN2 = 2.0 * np.log(2.0)


# ---------------------------------------------------------------------- #
# Lattice geometry                                                        #
# ---------------------------------------------------------------------- #

def lattice_index(voxel_grid, tol=1e-3):
    """
    Return the integer lattice coordinates of a grid's voxels.

    Parameters
    ----------
    voxel_grid : VoxelGrid
        Grid whose positions lie on a lattice of ``voxel_grid.spacing``.
    tol : float
        Largest allowed offset from the lattice, as a fraction of a voxel.
        Default 1e-3.

    Returns
    -------
    np.ndarray, shape (n_voxels, 3)

    Raises
    ------
    ValueError
        If the positions are not on a common lattice.
    """
    pos = np.asarray(voxel_grid.positions, dtype=float)
    h = float(voxel_grid.spacing)
    frac = (pos - pos.min(axis=0)) / h
    ijk = np.rint(frac).astype(np.int64)
    off = float(np.abs(frac - ijk).max()) if len(frac) else 0.0
    if off > tol:
        raise ValueError(
            f"Voxel positions are not on a common {h:g} mm lattice (max offset "
            f"{off:.3f} voxel). Cluster inference needs lattice neighbours.")
    return ijk


def _connectivity_offsets(connectivity):
    if connectivity not in (6, 18, 26):
        raise ValueError(f"connectivity must be 6, 18 or 26, got {connectivity!r}.")
    order = {6: 1, 18: 2, 26: 3}[connectivity]
    return [d for d in itertools.product((-1, 0, 1), repeat=3)
            if 0 < sum(map(abs, d)) <= order]


def _neighbour(ijk, offset, lut):
    """Return the index of each voxel's neighbour at ``offset``, or -1."""
    nb = ijk + np.asarray(offset)
    ok = np.all((nb >= 0) & (nb < lut.shape), axis=1)
    out = np.full(len(ijk), -1, dtype=np.int64)
    out[ok] = lut[tuple(nb[ok].T)]
    return out


def _lookup(ijk):
    lut = np.full(ijk.max(axis=0) + 1, -1, dtype=np.int64)
    lut[tuple(ijk.T)] = np.arange(len(ijk))
    return lut


def lattice_adjacency(voxel_grid, connectivity=18):
    """
    Return the adjacency matrix of a voxel grid.

    Parameters
    ----------
    voxel_grid : VoxelGrid
        Grid on a regular lattice.
    connectivity : {6, 18, 26}
        Faces; faces and edges; or faces, edges and corners. Default 18.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (n_voxels, n_voxels)
        Symmetric boolean adjacency.

    Raises
    ------
    ValueError
        If ``connectivity`` is not 6, 18 or 26, or the grid is not a lattice.
    """
    ijk = lattice_index(voxel_grid)
    lut = _lookup(ijk)
    rows, cols = [], []
    for d in _connectivity_offsets(connectivity):
        j = _neighbour(ijk, d, lut)
        keep = j >= 0
        rows.append(np.flatnonzero(keep))
        cols.append(j[keep])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    n = len(ijk)
    return sparse.csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)),
                             shape=(n, n))


def find_clusters(supra, adjacency):
    """
    Label the connected components of a boolean mask.

    Parameters
    ----------
    supra : array-like of bool, shape (n_voxels,)
        Voxels above threshold.
    adjacency : scipy.sparse matrix
        Neighbourhood graph.

    Returns
    -------
    np.ndarray of int, shape (n_voxels,)
        0 outside the mask, 1 to K inside.
    """
    supra = np.asarray(supra, dtype=bool)
    labels = np.zeros(supra.shape, dtype=np.int64)
    idx = np.flatnonzero(supra)
    if idx.size:
        _, lab = connected_components(adjacency[idx][:, idx], directed=False)
        labels[idx] = lab + 1
    return labels


# ---------------------------------------------------------------------- #
# Smoothness                                                              #
# ---------------------------------------------------------------------- #

_SMOOTHING_CACHE = {}


def _smoothing_matrix(positions, fwhm_mm):
    """
    Return Gaussian weights between voxels within 3 sigma, cached by geometry and FWHM.
    """
    key = (hash(np.ascontiguousarray(positions).tobytes()), round(float(fwhm_mm), 2))
    if key not in _SMOOTHING_CACHE:
        if len(_SMOOTHING_CACHE) > 8:
            _SMOOTHING_CACHE.clear()
        _SMOOTHING_CACHE[key] = _build_smoothing_matrix(positions, fwhm_mm)
    return _SMOOTHING_CACHE[key]


def _build_smoothing_matrix(positions, fwhm_mm):
    sigma = fwhm_mm / np.sqrt(8.0 * np.log(2.0))
    tree = cKDTree(positions)
    D = tree.sparse_distance_matrix(tree, 3.0 * sigma, output_type='coo_matrix')
    W = sparse.coo_matrix((np.exp(-0.5 * (D.data / sigma) ** 2), (D.row, D.col)),
                          shape=D.shape).tocsr()
    return W + sparse.identity(len(positions), format='csr')


def auto_rpv_kernel(global_fwhm_mm, df):
    """
    Return the default FWHM of the kernel that smooths the local RPV map.

    The FWHM is ``max(1, 2 * (9 / df)**(1/3))`` times the global FWHM, rounded
    to 0.5 mm.

    Parameters
    ----------
    global_fwhm_mm : float or array-like
        Global FWHM in mm, scalar or per axis.
    df : float
        Residual degrees of freedom.

    Returns
    -------
    float
        Kernel FWHM in mm.
    """
    g = float(np.prod(np.asarray(global_fwhm_mm, dtype=float)) ** (1 / 3))
    factor = max(1.0, 2.0 * (9.0 / float(df)) ** (1 / 3))
    return max(0.5, round(2.0 * factor * g) / 2.0)


def estimate_rpv(resid, voxel_grid, df, *, valid=None, smooth_fwhm='auto',
                 bias_correct=True):
    """
    Estimate the local smoothness of a noise field from residual images.

    Smoothness comes from the correlation between neighbouring voxels across
    the residual images, assuming a Gaussian autocorrelation.

    Parameters
    ----------
    resid : np.ndarray, shape (n_images, n_voxels)
        Residual images. Voxels with NaN in any image are left out.
    voxel_grid : VoxelGrid
        Grid on a regular lattice.
    df : float
        Residual degrees of freedom, e.g. ``n_subjects - 1``.
    valid : array-like of bool, shape (n_voxels,), optional
        Further restricts the search region.
    smooth_fwhm : 'auto', float or None
        FWHM in mm of the Gaussian that smooths the local estimate. 'auto'
        uses :func:`auto_rpv_kernel`; None disables smoothing.
    bias_correct : bool
        Apply the first-order Olkin-Pratt correction to the sample
        correlations. Default True.

    Returns
    -------
    dict
        ``rpv`` (resels per voxel, NaN outside the region), ``fwhm_mm`` (local
        FWHM per axis), ``global_fwhm_mm`` (per axis), ``valid`` (region used)
        and ``smooth_fwhm_mm`` (kernel used).

    Raises
    ------
    ValueError
        If the region has no lattice edges along some axis.
    """
    R = np.asarray(resid, dtype=float)
    n = R.shape[1]
    ssq = np.nansum(R ** 2, axis=0)
    ok = np.isfinite(R).all(axis=0) & (ssq > 0)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    U = np.where(ok, R / np.sqrt(np.where(ssq > 0, ssq, 1.0)), 0.0)

    ijk = lattice_index(voxel_grid)
    lut = _lookup(ijk)
    h = float(voxel_grid.spacing)

    # Per axis and voxel: sum and count of neighbour correlations, averaged
    # before the log.
    esum = np.zeros((3, n))
    ecnt = np.zeros((3, n))
    for a in range(3):
        e = np.zeros(3, dtype=int); e[a] = 1
        j = _neighbour(ijk, e, lut)
        has = j >= 0
        has[has] &= ok[j[has]]
        has &= ok
        v, w = np.flatnonzero(has), j[has]
        rho = np.einsum('iv,iv->v', U[:, v], U[:, w])
        for idx in (v, w):
            esum[a] += np.bincount(idx, rho, minlength=n)
            ecnt[a] += np.bincount(idx, minlength=n)

    if (ecnt[:, ok].sum(axis=1) == 0).any():
        raise ValueError("Search region has no lattice edges along some axis; "
                         "smoothness cannot be estimated.")

    def to_L(rho_bar):
        # Olkin-Pratt correction, then lambda h^2 / 2 = -ln(rho) for a
        # Gaussian autocorrelation; rho is floored at 0.01.
        if bias_correct:
            rho_bar = rho_bar * (1.0 + (1.0 - rho_bar ** 2) / (2.0 * df))
        return -np.log(np.clip(rho_bar, 0.01, 1.0 - 1e-12))

    L_glob = to_L(esum[:, ok].sum(axis=1) / ecnt[:, ok].sum(axis=1))
    global_fwhm = h * np.sqrt(_TWO_LN2 / L_glob)

    if smooth_fwhm == 'auto':
        smooth_fwhm = auto_rpv_kernel(global_fwhm, df)
    if smooth_fwhm:
        W = _smoothing_matrix(voxel_grid.positions, float(smooth_fwhm))
        W = W.multiply(ok[np.newaxis, :]).tocsr()   # only in-region voxels contribute
        esum = (W @ esum.T).T
        ecnt = (W @ ecnt.T).T

    with np.errstate(invalid='ignore', divide='ignore'):
        rho_bar = esum / ecnt
    has_edges = np.isfinite(rho_bar)
    L = to_L(np.where(has_edges, rho_bar, 0.5))
    # Voxels with no usable edge along an axis take the global value.
    L = np.where(has_edges, L, L_glob[:, np.newaxis])

    rpv = np.prod(np.sqrt(L / _TWO_LN2), axis=0)
    fwhm = h * np.sqrt(_TWO_LN2 / L).T
    rpv[~ok] = np.nan
    fwhm[~ok] = np.nan
    return {'rpv': rpv, 'fwhm_mm': fwhm, 'global_fwhm_mm': global_fwhm,
            'valid': ok, 'smooth_fwhm_mm': smooth_fwhm}


# ---------------------------------------------------------------------- #
# Random field theory                                                     #
# ---------------------------------------------------------------------- #

def resel_counts(voxel_grid, valid, fwhm_mm):
    """
    Return the resel counts R0 to R3 of a search region.

    Uses the lattice formula of Worsley et al. (1996): counts of points, edges,
    faces and cubes combined with ``spacing / FWHM`` per axis.

    Parameters
    ----------
    voxel_grid : VoxelGrid
        Grid on a regular lattice.
    valid : array-like of bool, shape (n_voxels,)
        Search region.
    fwhm_mm : array-like, shape (3,)
        FWHM in mm per axis.

    Returns
    -------
    np.ndarray, shape (4,)
    """
    ijk = lattice_index(voxel_grid)[np.asarray(valid, dtype=bool)]
    if not len(ijk):
        return np.zeros(4)
    ijk = ijk - ijk.min(axis=0)
    B = np.zeros(ijk.max(axis=0) + 1, dtype=bool)
    B[tuple(ijk.T)] = True

    def count(axes):
        core = tuple(slice(0, B.shape[k] - 1) if k in axes else slice(None)
                     for k in range(3))
        acc = np.ones(B[core].shape, dtype=bool)
        for corner in itertools.product(*[(0, 1) if k in axes else (0,)
                                          for k in range(3)]):
            acc &= B[tuple(slice(c, B.shape[k] - 1 + c) if k in axes else slice(None)
                           for k, c in enumerate(corner))]
        return int(acc.sum())

    r = float(voxel_grid.spacing) / np.asarray(fwhm_mm, dtype=float)
    P = int(B.sum())
    E = [count((a,)) for a in range(3)]
    F = {(a, b): count((a, b)) for a, b in itertools.combinations(range(3), 2)}
    C = count((0, 1, 2))
    Fab = lambda a, b: F[(min(a, b), max(a, b))]

    R0 = P - sum(E) + sum(F.values()) - C
    R1 = sum((E[a] - sum(Fab(a, b) for b in range(3) if b != a) + C) * r[a]
             for a in range(3))
    R2 = sum((F[(a, b)] - C) * r[a] * r[b] for a, b in F)
    R3 = C * r[0] * r[1] * r[2]
    return np.array([R0, R1, R2, R3], dtype=float)


def ec_density_t(u, df):
    """
    Return the Euler characteristic densities of a t-field.

    Parameters
    ----------
    u : float
        Threshold.
    df : float
        Degrees of freedom.

    Returns
    -------
    np.ndarray, shape (4,)
        Densities rho_0 to rho_3.
    """
    v = float(df)
    a = _FOUR_LN2
    b = np.exp(gammaln((v + 1) / 2) - gammaln(v / 2))
    c = (1 + u ** 2 / v) ** ((1 - v) / 2)
    return np.array([
        stats.t.sf(u, v),
        a ** 0.5 / (2 * np.pi) * c,
        a / (2 * np.pi) ** 1.5 * c * u / np.sqrt(v / 2) * b,
        a ** 1.5 / (2 * np.pi) ** 2 * c * ((v - 1) * u ** 2 / v - 1),
    ])


def cluster_p_fwe(k_resels, u, df, R):
    """
    Return the family-wise corrected p-value of a cluster.

    Uses the Poisson clumping heuristic: the expected number of clusters from
    the EC densities and ``P(size >= k) = exp(-beta k**(2/3))``.

    Parameters
    ----------
    k_resels : float or array-like
        Cluster size in resels.
    u : float
        Cluster-forming threshold.
    df : float
        Degrees of freedom.
    R : array-like, shape (4,)
        Resel counts of the search region.

    Returns
    -------
    float or np.ndarray
    """
    EC = np.maximum(ec_density_t(u, df), np.finfo(float).eps)
    EM = np.asarray(R, dtype=float) * EC
    Ec = EM.sum()
    Ek = EC[0] * R[3] / EM[3]
    beta = (gamma(2.5) / Ek) ** (2.0 / 3.0)
    p_size = np.exp(-beta * np.asarray(k_resels, dtype=float) ** (2.0 / 3.0))
    return 1.0 - np.exp(-Ec * p_size)


def grf_cluster_inference(t, resid, df, voxel_grid, *, cluster_p=0.001,
                          sign='both', connectivity=18, smooth_fwhm='auto',
                          adjacency=None):
    """
    Apply non-stationary random field cluster inference to one t-map.

    Parameters
    ----------
    t : np.ndarray, shape (n_voxels,)
        t-map. NaN voxels are outside the search region.
    resid : np.ndarray, shape (n_images, n_voxels)
        Residual images for the smoothness estimate.
    df : float
        Degrees of freedom of ``t``.
    voxel_grid : VoxelGrid
        Grid on a regular lattice.
    cluster_p : float
        Uncorrected voxelwise p-value that forms clusters. Default 0.001.
    sign : {+1, -1, 'both'}
        Which excursions form clusters. 'both' uses the two-sided threshold and
        Bonferroni-corrects the cluster p-values over the two signs.
    connectivity : {6, 18, 26}
        Voxel neighbourhood. Default 18.
    smooth_fwhm : 'auto', float or None
        See :func:`estimate_rpv`.
    adjacency : scipy.sparse matrix, optional
        Precomputed :func:`lattice_adjacency`.

    Returns
    -------
    dict
        ``cluster_id`` (0 outside clusters), ``cluster_p_fwe`` (per voxel, NaN
        outside clusters), ``clusters`` (list of dicts with ``id``, ``sign``,
        ``n_voxels``, ``resels``, ``peak_t``, ``peak_voxel``, ``p_fwe``),
        ``threshold``, ``resels`` (R0 to R3) and ``rpv`` (the
        :func:`estimate_rpv` result).

    Raises
    ------
    ValueError
        If ``sign`` is not +1, -1 or 'both'.
    """
    t = np.asarray(t, dtype=float)
    if sign == 'both':
        signs, u = (+1, -1), stats.t.isf(cluster_p / 2.0, df)
        n_tails = 2
    elif sign in (+1, -1):
        signs, u = (sign,), stats.t.isf(cluster_p, df)
        n_tails = 1
    else:
        raise ValueError(f"sign must be +1, -1 or 'both', got {sign!r}.")

    region = np.isfinite(t)
    sm = estimate_rpv(resid, voxel_grid, df, valid=region, smooth_fwhm=smooth_fwhm)
    region = sm['valid']
    R = resel_counts(voxel_grid, region, sm['global_fwhm_mm'])
    # R3 as the sum of RPV over the region, the same measure as cluster size;
    # R0-R2 stay lattice-based.
    R[3] = float(np.nansum(sm['rpv'][region]))
    if adjacency is None:
        adjacency = lattice_adjacency(voxel_grid, connectivity)

    n = len(t)
    cluster_id = np.zeros(n, dtype=np.int64)
    cluster_pv = np.full(n, np.nan)
    clusters = []
    next_id = 1
    for s in signs:
        supra = region & (s * np.nan_to_num(t, nan=-np.inf) > u)
        lab = find_clusters(supra, adjacency)
        for k in range(1, lab.max() + 1):
            members = np.flatnonzero(lab == k)
            resels = float(np.nansum(sm['rpv'][members]))
            p = min(1.0, n_tails * float(cluster_p_fwe(resels, u, df, R)))
            peak = members[np.argmax(s * t[members])]
            clusters.append({'id': next_id, 'sign': s, 'n_voxels': len(members),
                             'resels': resels, 'peak_t': float(t[peak]),
                             'peak_voxel': int(peak), 'p_fwe': p})
            cluster_id[members] = next_id
            cluster_pv[members] = p
            next_id += 1

    if R[3] < 5:
        warnings.warn(
            f"Search region is only {R[3]:.1f} resels; random field theory "
            f"is an asymptotic approximation and is unreliable this small.",
            stacklevel=2)

    return {'cluster_id': cluster_id, 'cluster_p_fwe': cluster_pv,
            'clusters': clusters, 'threshold': float(u), 'resels': R, 'rpv': sm}
