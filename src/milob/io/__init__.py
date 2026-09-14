# pyoptical/io/__init__.py
import os
from .snirf import read_snirf
# from .snirf import read_csv

def load_data(filepath, *, sc_threshold, **kwargs):
    """
    Load a data file, dispatching on its extension.

    Parameters
    ----------
    filepath : str
        Path to the file. Only SNIRF is currently recognised.
    sc_threshold : float or None
        Source-detector distance in mm below which a channel is classified as
        short. Pass None if the probe design has no short-separation
        channels.
    **kwargs
        Forwarded to the format's reader.

    Returns
    -------
    NirsStream
        The loaded stream.

    Raises
    ------
    ValueError
        If the file extension is not recognised.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.snirf':
        return read_snirf(filepath, sc_threshold=sc_threshold, **kwargs)
    # elif ext in ['.csv', '.txt']:
    #     return read_csv(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {ext}")