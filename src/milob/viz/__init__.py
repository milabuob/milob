"""
Plotting for the MILOB package.

Every 2-D plot is matplotlib. 3-D probe and head rendering uses PyVista,
an optional extra installed with ``pip install milob[threed]``.

Shared style -- colours, fonts, spines and figure sizing -- lives in
:mod:`milob.viz.theme` and is applied once at import time. Field
conventions are fixed: sources are red circles, detectors black squares,
HbO red, HbR blue and HbT green.
"""

from . import theme
theme.apply_style()

from .spectra import plot_timeseries_psd
from .topo import (
    plot_probe_2d,
    plot_rois_2d,
    plot_topo_map,
    plot_topo_with_significance,
    plot_tstat_conjunction_multi,
    plot_connectivity_map,
    align_group_values_to_probe,
)
from .network import (
    plot_network,
    plot_circular_network,
    plot_network_clusters,
    plot_small_world_comparison,
    plot_degree_distribution,
)
from .quality import plot_quality_histogram
from .events import plot_events, overlay_events
from .timedomain import plot_td_moments, plot_tpsf
from .matrix import plot_matrix
from .interactive import build_widgets, build_widget_browser

__all__ = [
    "theme",
    "plot_timeseries_psd",
    "plot_probe_2d",
    "plot_rois_2d",
    "plot_topo_map",
    "plot_topo_with_significance",
    "plot_tstat_conjunction_multi",
    "plot_connectivity_map",
    "align_group_values_to_probe",
    "plot_network",
    "plot_circular_network",
    "plot_network_clusters",
    "plot_small_world_comparison",
    "plot_degree_distribution",
    "plot_quality_histogram",
    "plot_events",
    "overlay_events",
    "plot_td_moments",
    "plot_tpsf",
    "plot_matrix",
    "build_widgets",
    "build_widget_browser",
]
