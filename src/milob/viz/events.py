"""
Event timelines: the experimental design plotted against sampling time.

Two entry points. :func:`plot_events` draws a standalone timeline that
owns its axes; :func:`overlay_events` marks events onto an existing
time-series axes and returns the label-to-colour mapping, so a condition
keeps one colour across panels.

The axis throughout is sampling time, never photon time of flight, which
belongs to :mod:`milob.viz.timedomain`.
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import theme

#: Label used on the legend/rows for events whose 'label' is missing. BIDS
#: files without a trial_type column land here (Events.from_bids fills the
#: column with NaN rather than inventing a condition name), as do .nirs stim
#: matrices with unnamed columns.
UNLABELED = "unlabeled"

_STYLES = ("auto", "span", "line")
_LAYOUTS = ("raster", "overlay")


def _as_table(events):
    """Normalise an Events object or DataFrame into a clean, sorted table."""
    if isinstance(events, pd.DataFrame):
        table = events.copy()
    elif hasattr(events, "to_dataframe"):
        table = events.to_dataframe()
    else:
        raise TypeError(
            f"events must be an Events object or a DataFrame, got {type(events).__name__}"
        )

    for col in ("onset", "duration", "value", "label"):
        if col not in table.columns:
            table[col] = np.nan

    table["label"] = table["label"].where(~table["label"].isna(), UNLABELED)
    table["label"] = table["label"].astype(str)
    table["onset"] = pd.to_numeric(table["onset"], errors="coerce")
    table["duration"] = pd.to_numeric(table["duration"], errors="coerce").fillna(0.0)

    # An event with no onset has no place on a time axis at all.
    dropped = int(table["onset"].isna().sum())
    if dropped:
        warnings.warn(f"plot_events: dropped {dropped} event(s) with no onset time.")
        table = table[table["onset"].notna()]

    return table.sort_values("onset").reset_index(drop=True)


def _select(table, labels, tmin, tmax):
    """Filter events by condition label and time window."""
    if labels is not None:
        if isinstance(labels, str):
            labels = [labels]
        labels = [str(l) for l in labels]
        missing = set(labels) - set(table["label"].unique())
        if missing:
            warnings.warn(f"plot_events: no events with label(s) {sorted(missing)}.")
        table = table[table["label"].isin(labels)]
    if tmin is not None:
        table = table[table["onset"] + table["duration"] >= tmin]
    if tmax is not None:
        table = table[table["onset"] <= tmax]
    return table.reset_index(drop=True)


def _resolve_colors(table, color_map, start_index=0):
    """Assign a colour per condition, extending a supplied mapping."""
    conditions = list(dict.fromkeys(table["label"]))
    if color_map is None:
        return theme.categorical_colors_for(conditions, unassigned_value=UNLABELED,
                                            start_index=start_index)

    resolved = dict(color_map)
    # UNLABELED is pinned to UNASSIGNED_COLOR and never consumes a palette
    # slot (theme.categorical_colors_for does the same), so counting it here
    # would skip a color and shift every label resolved after it.
    n_used = sum(1 for label in resolved if label != UNLABELED) + start_index
    for label in conditions:
        if label in resolved:
            continue
        if label == UNLABELED:
            resolved[label] = theme.UNASSIGNED_COLOR
            continue
        resolved[label] = theme.categorical_color(n_used)
        n_used += 1
    return resolved


def _is_block(duration, style):
    """Whether an event is drawn as a shaded block rather than a line."""
    if style == "span":
        return True
    if style == "line":
        return False
    return bool(np.isfinite(duration)) and duration > 0  # 'auto'


def _legend_handles(colors, table, style):
    """Build one legend handle per condition, matching how it was drawn."""
    handles = []
    for label, color in colors.items():
        durations = table.loc[table["label"] == label, "duration"]
        if len(durations):
            blocky = any(_is_block(d, style) for d in durations)
        else:
            # Carried in from a supplied color_map but not drawn on this
            # panel: nothing to infer from, so follow the style default.
            blocky = style != "line"
        if blocky:
            handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.5, label=label))
        else:
            handles.append(Line2D([0], [0], color=color, lw=1.8, label=label))
    return handles


def event_legend_handles(color_map, style="span", events=None):
    """
    Build legend handles for a label-to-colour mapping.

    Intended for axes that already carry a legend of their own and need a
    second one for the event colours; add the first with
    :meth:`~matplotlib.axes.Axes.add_artist` before calling
    :meth:`~matplotlib.axes.Axes.legend` again with these.

    Parameters
    ----------
    color_map : dict
        Mapping of label to colour, as returned by :func:`overlay_events`.
    style : {'auto', 'span', 'line'}, optional
        The style the events were drawn with. Default is ``'span'``.
    events : Events or pandas.DataFrame, optional
        The events that were drawn. Required with ``style='auto'``, where
        the marker a condition received depends on its own durations;
        without it every condition falls back to a block.

    Returns
    -------
    list
        Legend handles, one per condition.

    Raises
    ------
    ValueError
        If ``style`` is not one of the accepted values.
    """
    if style not in _STYLES:
        raise ValueError(f"style must be one of {_STYLES}, got {style!r}")
    if events is not None:
        return _legend_handles(color_map, _as_table(events), style)
    if style == "line":
        return [Line2D([0], [0], color=c, lw=1.8, label=l) for l, c in color_map.items()]
    return [Patch(facecolor=c, edgecolor=c, alpha=0.5, label=l)
            for l, c in color_map.items()]


def plot_events(events, ax=None, labels=None, style="auto", layout="raster",
                color_map=None, legend=True, tmin=None, tmax=None, alpha=0.45,
                title=None):
    """
    Plot experimental events against time, coloured by condition.

    Parameters
    ----------
    events : Events or pandas.DataFrame
        Event table with onset, duration, value and label columns.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.
    labels : str or list of str, optional
        Restrict to these conditions. All are drawn if omitted.
    style : {'auto', 'span', 'line'}, optional
        How each event is drawn. ``'auto'`` shades a block where the event
        has a positive duration and draws a vertical line where it does
        not; the other values force one or the other. Default is ``'auto'``.
    layout : {'raster', 'overlay'}, optional
        ``'raster'`` gives each condition its own row; ``'overlay'`` stacks
        every condition on one full-height row. Default is ``'raster'``.
    color_map : dict, optional
        Existing label-to-colour mapping to reuse, so a condition keeps its
        colour across panels. Unseen labels extend it.
    legend : bool, optional
        Draw the condition legend. Default is True.
    tmin, tmax : float, optional
        Time window in seconds. Events outside it are dropped.
    alpha : float, optional
        Opacity of shaded blocks; lines are drawn opaque.
    title : str, optional
        Axes title.

    Returns
    -------
    tuple
        The figure and axes.

    Raises
    ------
    ValueError
        If ``style`` or ``layout`` is not one of the accepted values.

    Examples
    --------
    >>> fig, ax = plot_events(stream.events, layout='overlay', style='line')
    """
    if style not in _STYLES:
        raise ValueError(f"style must be one of {_STYLES}, got {style!r}")
    if layout not in _LAYOUTS:
        raise ValueError(f"layout must be one of {_LAYOUTS}, got {layout!r}")

    table = _select(_as_table(events), labels, tmin, tmax)
    conditions = list(dict.fromkeys(table["label"]))
    n_rows = max(len(conditions), 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=theme.timeline_figsize(n_rows))
    else:
        fig = ax.figure

    if table.empty:
        warnings.warn("plot_events: no events to plot.")
        ax.set_xlabel("Time (s)")
        ax.set_yticks([])
        theme.style_quantitative_axes(ax, grid=False)
        ax.text(0.5, 0.5, "no events", ha="center", va="center",
                transform=ax.transAxes, color=theme.UNASSIGNED_COLOR)
        return fig, ax

    colors = _resolve_colors(table, color_map)
    row_of = {label: i for i, label in enumerate(conditions)}

    for _, ev in table.iterrows():
        color = colors[ev["label"]]
        if layout == "raster":
            centre = row_of[ev["label"]]
            lo, hi = centre - 0.4, centre + 0.4
        else:
            lo, hi = 0.0, 1.0

        if _is_block(ev["duration"], style):
            ax.fill_betweenx([lo, hi], ev["onset"], ev["onset"] + ev["duration"],
                             color=color, alpha=alpha, linewidth=0)
        else:
            ax.vlines(ev["onset"], lo, hi, color=color, lw=1.8)

    if layout == "raster":
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels(conditions)
        ax.set_ylim(-0.6, len(conditions) - 0.4)
    else:
        ax.set_yticks([])
        ax.set_ylim(0, 1)

    ax.set_xlabel("Time (s)")
    ax.set_xlim(tmin if tmin is not None else min(0.0, table["onset"].min()),
                tmax if tmax is not None else (table["onset"] + table["duration"]).max())
    if title:
        ax.set_title(title, fontweight="bold")
    theme.style_quantitative_axes(ax, grid=False)
    ax.grid(True, which="major", axis="x")

    if legend and (layout == "overlay" or len(conditions) > 1):
        ax.legend(handles=_legend_handles(colors, table, style),
                  title="condition", fontsize="small", loc="upper right")

    return fig, ax


def overlay_events(ax, events, labels=None, style="auto", color_map=None,
                   alpha=0.15, legend=False, tmin=None, tmax=None,
                   start_index=0):
    """
    Mark events on an existing time-series axes, coloured by condition.

    Draws onto the given axes and leaves its y-limits alone, so it can be
    layered onto any plot whose x-axis is sampling time. It returns the
    colour mapping rather than the axes, since it does not own them and the
    caller needs the mapping to stay consistent across panels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on, with time in seconds on the x-axis.
    events : Events or pandas.DataFrame
        Event table.
    labels : str or list of str, optional
        Restrict to these conditions. All are drawn if omitted.
    style : {'auto', 'span', 'line'}, optional
        How each event is drawn; as in :func:`plot_events`. Default is
        ``'auto'``.
    color_map : dict, optional
        Existing label-to-colour mapping to reuse and extend.
    alpha : float, optional
        Opacity of shaded blocks, low by default so the signal stays
        readable through them.
    tmin, tmax : float, optional
        Time window in seconds. Events outside it are dropped.
    legend : bool, optional
        Add a condition legend to the axes. Default is False, since the
        host axes usually carries its own.
    start_index : int, optional
        Palette slots to skip, so the event colours do not repeat those
        already used by the host axes. Default is 0.

    Returns
    -------
    dict
        Mapping of label to colour, for reuse on the next panel.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot(t, hbo)
    >>> overlay_events(ax, stream.events)
    """
    if style not in _STYLES:
        raise ValueError(f"style must be one of {_STYLES}, got {style!r}")

    table = _select(_as_table(events), labels, tmin, tmax)
    if table.empty:
        warnings.warn("overlay_events: no events to overlay.")
        return dict(color_map or {})

    colors = _resolve_colors(table, color_map, start_index=start_index)

    for _, ev in table.iterrows():
        color = colors[ev["label"]]
        if _is_block(ev["duration"], style):
            ax.axvspan(ev["onset"], ev["onset"] + ev["duration"],
                       color=color, alpha=alpha, linewidth=0, zorder=0)
        else:
            ax.axvline(ev["onset"], color=color, lw=1.2, alpha=0.8, zorder=0)

    if legend:
        ax.legend(handles=_legend_handles(colors, table, style),
                  title="condition", fontsize="small", loc="upper right")

    return colors
