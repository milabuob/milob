import numpy as np

#: Storage dtype for every voxel-indexed imaging array -- Jacobians,
#: reconstructed images, their uncertainty and QA maps, and the voxel
#: streams built from them.
#:
#: Reconstruction arrays are the largest objects in the package by a wide
#: margin: a whole-session reconstruction is (n_frames, n_voxels,
#: n_wavelengths), which for a 40-minute run at 4 mm resolution is billions
#: of bytes before anything else is allocated. float32 halves that at no
#: cost to the result, because the precision that matters is in the *solve*,
#: not the storage: a solver upcasts its inputs to float64, so a float32
#: array is still inverted in double precision, and voxel-space values
#: themselves carry nothing like 7 significant digits of real information.
#:
#: Producers cast on the way out; nothing downstream should assume float64.
IMAGING_DTYPE = np.float32

from .coregistration import Coregistration
from .head import HeadModel, ICBM152_LABELS, ICBM152_OFFSET, ICBM152_PROPERTIES
from .head_model import Surface, BrainSurface, HeadSurface
from .surface import plot_probe_3d
from .voxel_grid import VoxelGrid, fit_boundary_plane

__all__ = [
    "IMAGING_DTYPE",
    "Coregistration",
    "HeadModel", "ICBM152_LABELS", "ICBM152_OFFSET", "ICBM152_PROPERTIES",
    "Surface", "BrainSurface", "HeadSurface",
    "plot_probe_3d",
    "VoxelGrid", "fit_boundary_plane",
]
