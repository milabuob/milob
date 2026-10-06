"""Probe geometry helpers for the forward simulators."""

import numpy as np

from ..core.units import mm_per_unit


def probe_distances_cm(probe, untagged):
    """
    Return the probe's source-detector distances in cm.

    Parameters
    ----------
    probe : Probe
        Probe with a channel configuration. Its ``lengthUnit`` is used when
        set.
    untagged : {'cm', 'mm'}
        Unit assumed when ``probe.lengthUnit`` is None.

    Returns
    -------
    np.ndarray
        Distance per channel, in cm.
    """
    return (np.array(probe.distances, dtype=float)
            * mm_per_unit(probe.lengthUnit, default=untagged) / 10.0)
