import warnings

import numpy as np


# ---------------------------------------------------------------------------
# aggregate_channels() group-decision helpers
#
# Pure array/DataArray math, no Probe dependency. The Probe-facing rebuild
# (synthesising virtual optode geometry for merged channels) is
# core-object-model bookkeeping, not a swappable numerical algorithm, so it
# stays in core/dcs_stream.py (same split as Datastream.drop_channels(),
# which also has no processing/ counterpart).
# ---------------------------------------------------------------------------

def _resolve_manual_groups(data, channel_labels, groups, position_tol):
    """Normalise a user-supplied groups dict to {name: [0-based channel index]}."""
    label_to_idx = {lbl: i for i, lbl in enumerate(channel_labels)}
    n_channels = len(channel_labels)

    resolved = {}
    seen = set()
    for new_name, members in groups.items():
        if len(members) == 0:
            raise ValueError(f"Group '{new_name}' has no members.")

        idxs = []
        for m in members:
            if isinstance(m, str):
                if m not in label_to_idx:
                    raise ValueError(
                        f"Channel '{m}' (group '{new_name}') not found. "
                        f"Available (first 10): {channel_labels[:10]}"
                    )
                idxs.append(label_to_idx[m])
            else:
                idx = int(m)
                if not (0 <= idx < n_channels):
                    raise ValueError(
                        f"Channel index {idx} (group '{new_name}') out of "
                        f"range [0, {n_channels})."
                    )
                idxs.append(idx)

        if len(set(idxs)) != len(idxs):
            raise ValueError(f"Group '{new_name}' lists the same channel more than once.")
        overlap = seen & set(idxs)
        if overlap:
            dupes = sorted(channel_labels[i] for i in overlap)
            raise ValueError(f"Channel(s) {dupes} appear in more than one group.")
        seen.update(idxs)

        if len(idxs) > 1 and 'distance' in data.coords:
            dists = data['distance'].values[idxs].astype(float)
            if np.ptp(dists) > position_tol:
                warnings.warn(
                    f"Group '{new_name}' members span SDS "
                    f"{dists.min():.2f}-{dists.max():.2f} "
                    f"(> position_tol={position_tol}); averaging anyway.",
                    UserWarning,
                )

        resolved[new_name] = idxs

    return resolved


def cluster_channels_by_source(source_ids, detector_positions_mm, position_tol):
    """
    Cluster each source's detectors by position, giving co-located groups.

    Channels are partitioned by source first, so clustering on position alone
    already implies an equal source-detector separation.

    Parameters
    ----------
    source_ids : array-like
        Source ID per channel, shape (n_channels,).
    detector_positions_mm : np.ndarray
        Detector position per channel in mm, shape (n_channels, 3), aligned
        row for row with ``source_ids``.
    position_tol : float
        Clustering distance threshold in mm.

    Returns
    -------
    dict of {str: list of int}
        Group name to member channel indices. Singleton clusters are omitted,
        so channels absent from the result pass through unchanged.
    """
    from scipy.cluster.hierarchy import linkage, fcluster

    source_ids = np.asarray(source_ids)
    resolved = {}
    for src in np.unique(source_ids):
        idxs = np.where(source_ids == src)[0]
        if len(idxs) == 1:
            continue  # only channel on this source -> passthrough

        Z = linkage(detector_positions_mm[idxs], method='complete')
        labels = fcluster(Z, t=position_tol, criterion='distance')

        for k, cl in enumerate(np.unique(labels), start=1):
            member_idxs = idxs[labels == cl].tolist()
            if len(member_idxs) > 1:
                resolved[f"S{src}_agg{k}"] = member_idxs

    return resolved






