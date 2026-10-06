"""
Max-statistic permutation inference on statistic maps.

:func:`permutation_inference` turns an observed map and a stack of null maps
into family-wise error corrected p-values for clusters (extent or mass),
TFCE and single voxels (max-t). Null maps come from subject-wise sign flips
for a group model (:func:`sign_flip_t`) or from circularly shifted task
regressors for a single subject (:meth:`GLM.circular_shift_null`). The
adjacency can be a voxel lattice or a channel neighbourhood
(:func:`channel_adjacency`).

References
----------
Nichols, T. E., & Holmes, A. P. (2002). Nonparametric permutation tests for
functional neuroimaging: a primer with examples. Human Brain Mapping,
15(1), 1-25.
"""

import numpy as np
from scipy import sparse, stats
from scipy.spatial import cKDTree

from .cluster import find_clusters

_STATISTICS = ('mass', 'extent', 'tfce')


# ---------------------------------------------------------------------- #
# Containers and adjacency                                                #
# ---------------------------------------------------------------------- #

class PermutationNull:
    """
    Null t-maps for a set of contrasts, with their labels and provenance.

    Parameters
    ----------
    t : array-like, shape (n_perm, n_contrast, n_space, n_payload)
        Null t-maps, stored as float32.
    contrasts : sequence of str
        Contrast names.
    space_coords : array-like
        Channel or voxel labels.
    payload_levels : sequence of str
        Payload levels, e.g. ['HbO', 'HbR'].
    space_dim, payload_dim : str
        Dimension names.
    df : float
        Degrees of freedom of the t-maps, used only to set the cluster-forming
        threshold.
    kind : str
        'sign_flip' or 'circular_shift'.
    params : dict
        Settings needed to reproduce the null.
    observed_t : array-like, shape (n_contrast, n_space, n_payload), optional
        Unpermuted map, compared with the output being tested.
    exhaustive : bool
        True if the maps are every permutation, identity included.
    """

    def __init__(self, t, contrasts, space_coords, payload_levels, space_dim,
                 payload_dim, df, kind, params, observed_t=None, exhaustive=False):
        self.t = np.asarray(t, dtype=np.float32)
        self.contrasts = list(contrasts)
        self.space_coords = np.asarray(space_coords)
        self.payload_levels = list(payload_levels)
        self.space_dim, self.payload_dim = space_dim, payload_dim
        self.df = float(df)
        self.kind = kind
        self.params = dict(params)
        self.observed_t = None if observed_t is None else np.asarray(observed_t)
        self.exhaustive = bool(exhaustive)

    @property
    def n_perm(self):
        """Number of null maps."""
        return self.t.shape[0]

    def __repr__(self):
        return (f"<PermutationNull {self.kind} | {self.n_perm} maps | "
                f"{len(self.contrasts)} contrasts x {len(self.space_coords)} "
                f"{self.space_dim} x {self.payload_levels}>")


def channel_adjacency(positions_mm, max_mm):
    """
    Return the adjacency between channels whose midpoints are within a distance.

    Parameters
    ----------
    positions_mm : array-like, shape (n_channels, 3)
        Channel midpoints in mm.
    max_mm : float
        Largest neighbour distance in mm.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (n_channels, n_channels)
    """
    pos = np.asarray(positions_mm, dtype=float)
    pairs = cKDTree(pos).query_pairs(float(max_mm), output_type='ndarray')
    n = len(pos)
    if not len(pairs):
        return sparse.csr_matrix((n, n), dtype=bool)
    r = np.r_[pairs[:, 0], pairs[:, 1]]
    c = np.r_[pairs[:, 1], pairs[:, 0]]
    return sparse.csr_matrix((np.ones(len(r), dtype=bool), (r, c)), shape=(n, n))


# ---------------------------------------------------------------------- #
# Group null: sign flips                                                  #
# ---------------------------------------------------------------------- #

def sign_flips(n_subjects, n_perm, seed=0):
    """
    Return a matrix of subject-wise sign flips.

    All ``2**n_subjects`` flips (identity first) are returned when that is at
    most ``n_perm``; otherwise the identity and ``n_perm - 1`` random flips.

    Parameters
    ----------
    n_subjects : int
        Number of subjects.
    n_perm : int
        Largest number of flips.
    seed : int
        Seed for the random flips.

    Returns
    -------
    flips : np.ndarray, shape (n_flips, n_subjects)
        Entries of +1 and -1.
    exhaustive : bool
        True if every flip was enumerated.
    """
    if 2 ** n_subjects <= n_perm:
        k = np.arange(2 ** n_subjects)[:, None]
        bits = (k >> np.arange(n_subjects)[None, :]) & 1
        return 1.0 - 2.0 * bits, True
    rng = np.random.default_rng(seed)
    F = rng.choice([-1.0, 1.0], size=(n_perm, n_subjects))
    F[0] = 1.0
    return F, False


def sign_flip_t(Y, V, flips, method='simple', batch=64):
    """
    Return the group t-map under each sign flip.

    Computes the statistics of ``Study.get_group_stats`` for every flip. For
    'weighted', the DerSimonian-Laird tau^2 is re-estimated under each flip.

    Parameters
    ----------
    Y : array-like, shape (n_subjects, n_space)
        Subject estimates. NaN excludes a cell.
    V : array-like, shape (n_subjects, n_space)
        Within-subject variances. Used by 'weighted' only, where V <= 0 also
        excludes a cell.
    flips : array-like, shape (n_flips, n_subjects)
        Sign flips, e.g. from :func:`sign_flips`.
    method : {'simple', 'weighted'}
        Group model. Default 'simple'.
    batch : int
        Flips per block for 'weighted'. Default 64.

    Returns
    -------
    np.ndarray, shape (n_flips, n_space)
        NaN where fewer than two subjects contribute.

    Raises
    ------
    ValueError
        If ``method`` is unknown.
    """
    Y = np.asarray(Y, dtype=float)
    F = np.asarray(flips, dtype=float)
    if method == 'weighted':
        V = np.asarray(V, dtype=float)
        ok = np.isfinite(Y) & np.isfinite(V) & (V > 0)
    else:
        ok = np.isfinite(Y)
    y = np.where(ok, Y, 0.0)
    n_v = ok.sum(axis=0).astype(float)
    out = np.full((F.shape[0], Y.shape[1]), np.nan)

    with np.errstate(invalid='ignore', divide='ignore'):
        if method == 'simple':
            ss = (y ** 2).sum(axis=0)
            mean = (F @ y) / n_v
            var = (ss - n_v * mean ** 2) / (n_v - 1)
            out = mean / np.sqrt(np.maximum(var, 0) / n_v)
        elif method == 'weighted':
            v = np.where(ok, V, np.inf)
            w0 = np.where(ok, 1.0 / v, 0.0)
            sw0 = w0.sum(axis=0)
            C = sw0 - (w0 ** 2).sum(axis=0) / sw0
            wy2 = (w0 * y ** 2).sum(axis=0)
            for s in range(0, F.shape[0], batch):
                Fb = F[s:s + batch]
                A = Fb @ (w0 * y)
                Q = wy2 - A ** 2 / sw0
                tau = np.where(C > 0, np.maximum(0.0, (Q - (n_v - 1)) / C), 0.0)
                tau = np.nan_to_num(tau, nan=0.0)
                W = np.where(ok[None], 1.0 / (v[None] + tau[:, None, :]), 0.0)
                num = np.einsum('bn,bnv,nv->bv', Fb, W, y)
                out[s:s + batch] = num / np.sqrt(W.sum(axis=1))
        else:
            raise ValueError(f"method must be 'simple' or 'weighted', got {method!r}.")
    out[:, n_v < 2] = np.nan
    return out


# ---------------------------------------------------------------------- #
# Statistics                                                              #
# ---------------------------------------------------------------------- #

def _cluster_values(st, adjacency, u, statistic):
    """Return the clusters of ``st > u`` and their extent or mass."""
    lab = find_clusters(np.nan_to_num(st, nan=-np.inf) > u, adjacency)
    if not lab.max():
        return lab, np.zeros(0)
    idx = lab[lab > 0] - 1
    if statistic == 'extent':
        vals = np.bincount(idx).astype(float)
    else:
        vals = np.bincount(idx, weights=st[lab > 0] - u)
    return lab, vals


def tfce(st, adjacency, E=0.5, H=2.0, dh=0.1):
    """
    Return the threshold-free cluster enhancement of the positive part of a map.

    Computes ``sum_h extent(h)**E * h**H * dh`` over heights ``h``, with extent
    in voxels.

    Parameters
    ----------
    st : array-like, shape (n_space,)
        Signed statistic map.
    adjacency : scipy.sparse matrix
        Neighbourhood graph.
    E, H : float
        Extent and height exponents. Defaults 0.5 and 2.
    dh : float
        Height step. Default 0.1.

    Returns
    -------
    np.ndarray, shape (n_space,)

    References
    ----------
    Smith, S. M., & Nichols, T. E. (2009). Threshold-free cluster enhancement.
    NeuroImage, 44(1), 83-98.
    """
    st = np.nan_to_num(np.asarray(st, dtype=float), nan=0.0)
    out = np.zeros_like(st)
    top = st.max()
    if top <= 0:
        return out
    for h in np.arange(dh, top + dh, dh):
        lab = find_clusters(st >= h, adjacency)
        if not lab.max():
            break
        ext = np.bincount(lab)[lab]
        m = lab > 0
        out[m] += ext[m] ** E * h ** H * dh
    return out


def permutation_inference(t_obs, t_null, adjacency, *, df, cluster_p=0.001,
                          sign='both', statistic='mass', exhaustive=False,
                          tfce_params=None, alpha=0.05):
    """
    Apply max-statistic permutation inference to one statistic map.

    Parameters
    ----------
    t_obs : array-like, shape (n_space,)
        Observed map.
    t_null : array-like, shape (n_perm, n_space)
        Null maps. If ``exhaustive``, they are every permutation including the
        identity; otherwise p-values are ``(1 + n_exceed) / (1 + n_perm)``.
    adjacency : scipy.sparse matrix, shape (n_space, n_space)
        Neighbourhood graph.
    df : float
        Degrees of freedom, used only to set the cluster-forming threshold.
    cluster_p : float
        Uncorrected p-value that forms clusters. Not used by 'tfce'.
        Default 0.001.
    sign : {+1, -1, 'both'}
        Which excursions are tested. 'both' takes the maximum over the two
        signs in every permutation.
    statistic : {'mass', 'extent', 'tfce'}
        Cluster statistic. 'mass' sums the height above threshold. Default
        'mass'.
    exhaustive : bool
        Whether ``t_null`` holds every permutation. Default False.
    tfce_params : dict, optional
        ``E``, ``H`` and ``dh`` for :func:`tfce`.
    alpha : float
        'tfce' only: level at which voxels are grouped into reported clusters.

    Returns
    -------
    dict
        ``cluster_id``, ``cluster_p_fwe``, ``clusters`` (as
        :func:`~milob.stats.cluster.grf_cluster_inference`, with ``statistic``
        in place of ``resels``), ``p_fwe_voxel`` (max-t), ``threshold``,
        ``null_max`` and, for 'tfce', ``tfce`` and ``p_fwe_tfce``.

    Raises
    ------
    ValueError
        If ``statistic`` or ``sign`` is unknown.
    """
    if statistic not in _STATISTICS:
        raise ValueError(f"statistic must be one of {_STATISTICS}, got {statistic!r}.")
    t_obs = np.asarray(t_obs, dtype=float)
    t_null = np.asarray(t_null, dtype=float)
    if sign == 'both':
        signs, u = (1, -1), stats.t.isf(cluster_p / 2.0, df)
    elif sign in (1, -1):
        signs, u = (sign,), stats.t.isf(cluster_p, df)
    else:
        raise ValueError(f"sign must be +1, -1 or 'both', got {sign!r}.")
    tp = {'E': 0.5, 'H': 2.0, 'dh': 0.1, **(tfce_params or {})}

    def score(tmap):
        """Per-sign cluster labels/values or TFCE, and the max over signs."""
        res, best, vmax = {}, 0.0, -np.inf
        for s in signs:
            st = s * tmap
            if statistic == 'tfce':
                val = tfce(st, adjacency, **tp)
                res[s] = val
                best = max(best, float(val.max()) if val.size else 0.0)
            else:
                lab, vals = _cluster_values(st, adjacency, u, statistic)
                res[s] = (lab, vals)
                best = max(best, float(vals.max()) if vals.size else 0.0)
            vmax = max(vmax, float(np.nanmax(st)) if np.isfinite(st).any() else -np.inf)
        return res, best, vmax

    obs, _, _ = score(t_obs)
    null_max = np.empty(len(t_null))
    null_tmax = np.empty(len(t_null))
    for i, tm in enumerate(t_null):
        _, null_max[i], null_tmax[i] = score(tm)

    def p_of(x, null):
        srt = np.sort(null)
        exceed = len(srt) - np.searchsorted(srt, np.asarray(x, dtype=float) - 1e-12,
                                            side='left')
        return exceed / len(srt) if exhaustive else (exceed + 1) / (len(srt) + 1)

    n = len(t_obs)
    signed_obs = np.full(n, -np.inf)
    for s in signs:
        signed_obs = np.maximum(signed_obs, np.nan_to_num(s * t_obs, nan=-np.inf))
    p_vox = np.where(np.isfinite(t_obs), p_of(signed_obs, null_tmax), np.nan)

    cluster_id = np.zeros(n, dtype=np.int64)
    cluster_pv = np.full(n, np.nan)
    clusters, next_id = [], 1
    out = {'threshold': float(u) if statistic != 'tfce' else np.nan,
           'null_max': null_max, 'p_fwe_voxel': p_vox}

    if statistic == 'tfce':
        enh = np.zeros(n)
        for s in signs:
            enh = np.where(obs[s] > enh, obs[s], enh)
        p_tfce = p_of(enh, null_max)
        out['tfce'] = enh
        # Connected TFCE-significant voxels form clusters, with the best p.
        for s in signs:
            sig = (obs[s] > 0) & (obs[s] >= enh) & (p_tfce < alpha)
            lab = find_clusters(sig, adjacency)
            for k in range(1, lab.max() + 1):
                m = np.flatnonzero(lab == k)
                peak = m[np.argmax(s * t_obs[m])]
                p = float(p_tfce[m].min())
                clusters.append({'id': next_id, 'sign': s, 'n_voxels': len(m),
                                 'statistic': float(enh[m].max()),
                                 'peak_t': float(t_obs[peak]),
                                 'peak_voxel': int(peak), 'p_fwe': p})
                cluster_id[m], cluster_pv[m] = next_id, p
                next_id += 1
        out['p_fwe_tfce'] = np.where(np.isfinite(t_obs), p_tfce, np.nan)
    else:
        for s in signs:
            lab, vals = obs[s]
            for k in range(1, lab.max() + 1):
                m = np.flatnonzero(lab == k)
                peak = m[np.argmax(s * t_obs[m])]
                p = float(p_of(vals[k - 1], null_max))
                clusters.append({'id': next_id, 'sign': s, 'n_voxels': len(m),
                                 'statistic': float(vals[k - 1]),
                                 'peak_t': float(t_obs[peak]),
                                 'peak_voxel': int(peak), 'p_fwe': p})
                cluster_id[m], cluster_pv[m] = next_id, p
                next_id += 1

    out.update({'cluster_id': cluster_id, 'cluster_p_fwe': cluster_pv,
                'clusters': clusters})
    return out
