"""
Metadata describing the value space of each functional-connectivity method.

Every FC method declares a ``units`` string, and everything that depends on
what kind of number a connectivity value is -- plot range and colormap,
which variance-stabilising transform makes it averageable, whether negative
values are meaningful -- is looked up here.
"""

import numpy as np

FC_UNITS_META = {
    'r': {
        # Pearson correlation coefficient. Signed, bounded, and derived from
        # a sampling distribution that Fisher's z-transform is designed for.
        'range': (-1.0, 1.0),
        'cmap': 'RdBu_r',
        'z_transform': 'fisher_r',
        'signed': True,
    },
    'z': {
        # Already a Fisher-Z transform of r. Signed and unbounded; averaging
        # it again through arctanh/tanh would be a second, unjustified
        # transform, so no transform here (average directly).
        'range': None,
        'cmap': 'RdBu_r',
        'z_transform': None,
        'signed': True,
    },
    'coherence': {
        # Magnitude-squared coherence. Non-negative and bounded [0, 1].
        # Fisher's z of the VALUE is wrong here -- C is not an r and does
        # not come from the bivariate-normal sampling theory arctanh(r)
        # assumes. Fisher's z of the coherence MAGNITUDE, arctanh(sqrt(C)),
        # is the correct variance-stabilising transform (Enochson &
        # Goodman 1965), so this space does have one -- it is simply not
        # the same equation as 'r'. Applies to every coherence-valued
        # method alike (Welch 'coherence' and 'wavelet_coherence'), since
        # the transform is a property of the value space, not the method.
        'range': (0.0, 1.0),
        'cmap': 'viridis',
        'z_transform': 'fisher_msc',
        'signed': False,
    },
    'null_z': {
        # A statistic normalized against a null/surrogate distribution --
        # observed coherence or Pearson r vs. a phase-randomized null, see
        # FC.correct_bias(). Deliberately NOT named 'z' or 'z_score': this
        # codebase's 'z' already means Fisher's z-transform of a Pearson r,
        # a specific, unrelated concept -- reusing that name for "distance
        # from a null-model mean, in null-model standard deviations" would
        # collide with it. Signed (can occasionally be negative, if the
        # observed statistic happens to sit below the null mean) and
        # unbounded, and not an r-value, so no Fisher-Z step either.
        'range': None,
        'cmap': 'RdBu_r',
        'z_transform': None,
        'signed': True,
    },
}


# Variance-stabilising transforms, keyed by the name each FC_UNITS_META entry
# declares in its 'z_transform' field. Every consumer that aggregates FC values
# (FCOutput.average, FCOutput.roi_average) averages in the transformed space and
# maps back through 'inverse'; to_dataframe() emits the transformed value as an
# extra column named by 'column'.
#
# Why a registry rather than the old `fisher_z_valid` boolean: a boolean can
# only say "arctanh(value) or nothing", which forced coherence into the "or
# nothing" branch even though it HAS a correct transform -- just a different
# equation (arctanh of the magnitude, not of the value). Keying the transform to
# the value space rather than to the FC method also means Welch coherence and
# wavelet coherence are handled identically without either one being named here,
# since both declare units='coherence'.
#
# Same name-dispatch convention as processing.fitting's
# _FD_RESIDUAL_TRANSFORMS / forward.dynamics.msd.
FC_Z_TRANSFORMS = {
    'fisher_r': {
        # Fisher's z for a Pearson r. Clipped off +/-1 so a perfect
        # correlation (a channel against itself, or a degenerate slice)
        # gives a large finite z instead of an inf that would poison the
        # mean of every other pair in the block.
        'forward': lambda v: np.arctanh(np.clip(v, -0.999, 0.999)),
        'inverse': np.tanh,
        'column': 'z',
    },
    'fisher_msc': {
        # Enochson & Goodman (1965): for magnitude-squared coherence C
        # estimated from n independent averages, arctanh(sqrt(C)) is
        # approximately normal with a variance set by n alone, independent
        # of the underlying coherence -- the property that makes values
        # averageable, exactly as Fisher's z does for r.
        #
        # NOTE the variance depends on the effective number of averages,
        # which differs between Welch coherence (segment count) and wavelet
        # coherence (set by the scale/time smoothing) and is not currently
        # recorded on the output. This transform therefore makes values
        # averageable WITHIN a method; it does not by itself make a WTC
        # value comparable to a Welch one.
        'forward': lambda c: np.arctanh(np.sqrt(np.clip(c, 0.0, 1.0 - 1e-12))),
        'inverse': lambda z: np.tanh(z) ** 2,
        'column': 'z',
    },
}


def resolve_z_transform(units):
    """
    Return the variance-stabilising transform for an FC value space.

    Parameters
    ----------
    units : str or None
        An FCOutput's ``attrs['units']``, e.g. 'r' or 'coherence'.

    Returns
    -------
    dict or None
        Keys 'forward', 'inverse' and 'column'. None when the space needs no
        transform, as for already-transformed values, or when the units are
        unknown, in which case callers aggregate the raw values.
    """
    name = FC_UNITS_META.get(units, {}).get('z_transform')
    if name is None:
        return None
    try:
        return FC_Z_TRANSFORMS[name]
    except KeyError:
        raise ValueError(
            f"FC_UNITS_META[{units!r}] declares z_transform={name!r}, which is "
            f"not in FC_Z_TRANSFORMS; choose one of {sorted(FC_Z_TRANSFORMS)}."
        )


# Per-channel quality coords (indexed on 'channel' in the source Datastream)
# that FC.fit() propagates onto channel_i/channel_j in the FCOutput, if
# present, so downstream aggregation (FCOutput.roi_average(weights=...)) can
# use them. Shared here (not in analysis/connectivity.py or
# outputs/output_conn.py) for the same circular-import reason as
# FC_UNITS_META above -- both modules need this list.
CHANNEL_QUALITY_COORDS = ('sci', 'snr', 'sci_motion_burden')
