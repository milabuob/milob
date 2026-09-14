"""
Shared ipywidgets scaffolding for the interactive Jupyter viewers.

Keeps the widget construction, wiring and display boilerplate in one
place, so each viewer supplies only its widget specification and its
render callback. What to draw stays with the caller, being genuinely
different per viewer.
"""

from ipywidgets import widgets, HBox, VBox, interactive_output
from IPython.display import display

_DEFAULT_LAYOUT = widgets.Layout(padding="10px", border="1px solid #ddd")

_WIDGET_CTORS = {
    "dropdown": widgets.Dropdown,
    "select_multiple": widgets.SelectMultiple,
    "toggle_buttons": widgets.ToggleButtons,
    "slider": widgets.FloatSlider,
    "range_slider": widgets.FloatRangeSlider,
    "bounded_float_text": widgets.BoundedFloatText,
    "button": widgets.Button,
    "html": widgets.HTML,
}


def build_widgets(specs):
    """
    Build a mapping of name to widget from a list of specifications.

    Parameters
    ----------
    specs : list of dict
        Each entry gives ``'name'`` and ``'type'``, with any further keys
        passed to the widget constructor. An optional ``'watch'`` key is
        consumed here rather than forwarded; see :func:`build_widget_browser`.

    Returns
    -------
    dict
        Widget name to widget instance.

    Raises
    ------
    ValueError
        If a specification names an unsupported widget type.
    """
    made = {}
    for spec in specs:
        spec = dict(spec)
        name = spec.pop("name")
        kind = spec.pop("type")
        spec.pop("watch", None)
        made[name] = _WIDGET_CTORS[kind](**spec)
    return made


def build_widget_browser(widget_specs, render_fn, columns=None, observers=None,
                          layout=None, assemble_ui=None):
    """
    Assemble and display an interactive widget browser.

    Parameters
    ----------
    widget_specs : list of dict
        Widget specifications; see :func:`build_widgets`. Widgets omitted
        from ``columns`` are still built and bound, which is useful for
        hidden state.
    render_fn : callable
        Called with the current value of every interactive widget whenever
        any of them changes. It is responsible for closing, building and
        showing its own figure, since ipywidgets redraws in place.
    columns : list of list of str, optional
        Widget names grouped into rows. Defaults to a single row in
        specification order, and is ignored when ``assemble_ui`` is given.
    observers : list of tuple, optional
        ``(widget_name, callback)`` pairs, where the callback receives the
        ipywidgets change dict and the full widget mapping. Use it for
        cross-widget behaviour, such as one control enabling another.
    layout : ipywidgets.Layout, optional
        Layout applied to the assembled container.
    assemble_ui : callable, optional
        Called with the widget mapping to build the container, replacing the
        default row-and-column arrangement.

    Returns
    -------
    dict
        The widget mapping, so the caller can inspect or update the widgets.
    """
    built = build_widgets(widget_specs)

    for name, callback in observers or []:
        built[name].observe(lambda change, cb=callback: cb(change, built), names="value")

    interactive_names = [
        spec["name"] for spec in widget_specs
        if spec["type"] not in ("button", "html") and spec.get("watch", True)
    ]

    out = interactive_output(
        render_fn,
        {name: built[name] for name in interactive_names},
    )

    if assemble_ui is not None:
        ui = assemble_ui(built)
    else:
        if columns is None:
            columns = [[spec["name"] for spec in widget_specs]]
        ui = VBox(
            [HBox([built[name] for name in col]) for col in columns],
            layout=layout or _DEFAULT_LAYOUT,
        )

    display(ui, out)
    return built
