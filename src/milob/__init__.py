# pyoptical/src/pyoptical/__init__.py
"""
MILOB: a Python library for diffuse optical analysis.

Covers continuous-wave, time-domain and frequency-domain NIRS and
diffuse correlation spectroscopy, from raw recordings through to
group-level statistics.

An analysis is organised into four nested containers: a :class:`Study`
holds subjects, a :class:`Session` holds one recording's streams, a
:class:`Datastream` holds one time series with its probe geometry and
events, and an ``Output`` holds the statistical result. Measurement
streams hold what the instrument recorded, parameter streams what was
recovered from it.
"""

__version__ = "1.0.0"

import logging as _logging

# Package-level logger. All milob modules use logging.getLogger('milob').
# The logger itself is always at INFO so messages reach all handlers.
# Verbosity is controlled per-handler: the console handler level changes
# with set_verbose(); the file handler always captures INFO and above.
_logger = _logging.getLogger('milob')
_logger.setLevel(_logging.INFO)
_logger.propagate = False  # don't bubble up to the root logger

if not _logger.handlers:
    _console_handler = _logging.StreamHandler()
    _console_handler.setFormatter(_logging.Formatter('%(message)s'))
    _console_handler.setLevel(_logging.INFO)
    _logger.addHandler(_console_handler)


def set_verbose(verbose, log_file=None):
    """
    Control the package's informational output.

    Parameters
    ----------
    verbose : bool, optional
        Show progress messages on the console. Genuine warnings appear
        either way. Default is True.
    log_file : str, optional
        Path to a log file, which receives every message regardless of the
        console setting. Pass None to remove a log file set earlier.

    Examples
    --------
    >>> milob.set_verbose(False, log_file="batch_run.log")
    >>> milob.set_verbose(True)
    """
    # Adjust the console handler level — the logger itself stays at INFO
    for h in _logger.handlers:
        if isinstance(h, _logging.StreamHandler) and not isinstance(h, _logging.FileHandler):
            h.setLevel(_logging.INFO if verbose else _logging.WARNING)

    # Remove any existing file handlers
    for h in _logger.handlers[:]:
        if isinstance(h, _logging.FileHandler):
            h.close()
            _logger.removeHandler(h)

    if log_file is not None:
        file_handler = _logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(_logging.INFO)
        file_handler.setFormatter(
            _logging.Formatter('%(asctime)s  %(levelname)-8s  %(message)s',
                               datefmt='%H:%M:%S')
        )
        _logger.addHandler(file_handler)


# Core base classes
from .core.datastream import Datastream, MeasurementStream, ParameterStream
from .core.probe import Probe
from .core.events import Events

# Optical measurement streams (modality-specific)
from .core.nirs import NirsStream
from .core.cw_nirs import CW_Stream
from .core.td_nirs import TD_Stream
from .core.fd_nirs import FD_Stream

# DCS streams
from .core.dcs_stream import DCS_Stream

# Intermediate / processed streams (modality-independent)
from .core.opt_prop_stream import OptPropStream
from .core.tissue_stream import TissueStream

# Auxiliary / physiological streams
from .core.auxiliary import AuxStream

# Group-level containers
from .core.session import Session
from .core.study import Study

# Simulation module
from . import forward


__all__ = [
    "set_verbose",
    # Base
    "Datastream", "MeasurementStream", "ParameterStream", "Probe", "Events",
    # Optical measurement
    "NirsStream", "CW_Stream", "TD_Stream", "FD_Stream",
    # DCS
    "DCS_Stream",
    # Processed
    "OptPropStream", "TissueStream",
    # Auxiliary
    "AuxStream",
    # Group
    "Session", "Study",
    # Simulation
    "forward",
]
