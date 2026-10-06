"""
Length units for probes and streams.

Probes (``Probe.lengthUnit``) and streams (``data.attrs['lengthUnit']``)
carry a length-unit tag. Every conversion between units goes through the
functions here, which accept 'mm', 'cm' and 'm' in any case and raise on
anything else.
"""

#: Millimetres per supported length unit.
MM_PER_UNIT = {'mm': 1.0, 'cm': 10.0, 'm': 1000.0}


def normalize_length_unit(unit):
    """
    Return the canonical spelling of a length unit.

    Parameters
    ----------
    unit : str or None
        Unit to normalise. Surrounding spaces and case are ignored.

    Returns
    -------
    str or None
        'mm', 'cm' or 'm', or None when ``unit`` is None or empty.

    Raises
    ------
    ValueError
        If ``unit`` is not a supported length unit.
    """
    if unit is None:
        return None
    key = str(unit).strip().lower()
    if key == '':
        return None
    if key not in MM_PER_UNIT:
        raise ValueError(
            f"Unsupported length unit {unit!r}; expected one of "
            f"{sorted(MM_PER_UNIT)} (case-insensitive) or None.")
    return key


def mm_per_unit(unit, default='mm'):
    """
    Return the number of millimetres in one ``unit``.

    Parameters
    ----------
    unit : str or None
        Length unit.
    default : str
        Unit assumed when ``unit`` is None or empty. Default 'mm'.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If ``unit`` or ``default`` is not a supported length unit.
    """
    key = normalize_length_unit(unit)
    return MM_PER_UNIT[key if key is not None else normalize_length_unit(default)]


def cm_per_unit(unit, default='cm'):
    """
    Return the number of centimetres in one ``unit``.

    Parameters
    ----------
    unit : str or None
        Length unit.
    default : str
        Unit assumed when ``unit`` is None or empty. Default 'cm'.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If ``unit`` or ``default`` is not a supported length unit.
    """
    return mm_per_unit(unit, default=default) / 10.0
