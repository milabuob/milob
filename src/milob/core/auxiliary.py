import numpy as np
import xarray as xr
from .datastream import MeasurementStream


class AuxStream(MeasurementStream):
    """
    Auxiliary physiological time series, such as PPG, ECG or accelerometer.

    Data has dims (time, signal), where 'signal' indexes independent traces
    and carries no spatial meaning. A probe is not required, and there are no
    wavelength or channel-quality coordinates.

    Attributes
    ----------
    data.attrs['signal_type'] : str
        Category of the recording: 'ppg', 'ecg', 'accel', 'co2', 'gsr',
        'respiration', 'hr' or 'other'.
    data.attrs['units'] : str
        Physical units of the signal, e.g. 'mV', 'g', 'mmHg', '%'.
    data.attrs['sampling_rate'] : float
        Samples per second.

    Examples
    --------
    >>> ppg = AuxStream(data, signal=['ppg'])
    >>> acc = AuxStream(data, signal=['acc_x', 'acc_y', 'acc_z'])
    """

    def __init__(self, data, probe=None, signal_type: str = 'other', **kwargs):
        super().__init__(data, probe=probe, **kwargs)
        self.data.attrs.setdefault('signal_type', signal_type)

        # Soft validation: warn if 'signal' dim is missing but don't raise —
        # some aux data may use a single unnamed axis.
        if 'signal' not in self.data.dims and 'channel' not in self.data.dims:
            import warnings
            warnings.warn(
                "AuxStream data has no 'signal' or 'channel' dimension. "
                "Consider labelling signal axes for clarity."
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def signal_type(self) -> str:
        return self.data.attrs.get('signal_type', 'other')

    @property
    def signal_labels(self) -> list:
        """Labels of the individual signal traces."""
        dim = 'signal' if 'signal' in self.data.dims else 'channel'
        return list(self.data[dim].values) if dim in self.data.dims else []

    @property
    def sampling_rate(self) -> float:
        return self.data.attrs.get('sampling_rate', None)

    # ------------------------------------------------------------------
    # Signal-specific helpers
    # ------------------------------------------------------------------

    def magnitude(self) -> xr.DataArray:
        """
        Vector magnitude of a three-axis sensor.

        Returns
        -------
        xr.DataArray
            Magnitude sqrt(x^2 + y^2 + z^2), with dims (time,).
        """
        dim = 'signal' if 'signal' in self.data.dims else 'channel'
        labels = self.signal_labels

        xyz_candidates = [
            [l for l in labels if 'x' in l.lower()],
            [l for l in labels if 'y' in l.lower()],
            [l for l in labels if 'z' in l.lower()],
        ]
        if not all(xyz_candidates):
            raise ValueError(
                "Cannot compute magnitude: expected signal labels containing "
                "'x', 'y', 'z' (e.g. 'acc_x', 'acc_y', 'acc_z'). "
                f"Found: {labels}"
            )
        x = self.data.sel({dim: xyz_candidates[0][0]})
        y = self.data.sel({dim: xyz_candidates[1][0]})
        z = self.data.sel({dim: xyz_candidates[2][0]})
        return np.sqrt(x**2 + y**2 + z**2)

    def select_signal(self, label: str) -> xr.DataArray:
        """
        Return a single signal trace.

        Parameters
        ----------
        label : str
            Label of the trace to select.

        Returns
        -------
        xr.DataArray
            The selected trace, with dims (time,).
        """
        dim = 'signal' if 'signal' in self.data.dims else 'channel'
        return self.data.sel({dim: label})

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self):
        fs = self.sampling_rate
        fs_str = f"{fs:.1f} Hz" if fs is not None else "unknown fs"
        return (
            f"<AuxStream | {self.name} | {self.signal_type} | "
            f"Signals: {self.signal_labels} | "
            f"{fs_str} | "
            f"Duration: {len(self.data.time)} frames>"
        )

