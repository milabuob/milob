"""
Graph and network plots built on NetworkX layouts.

Covers spatial network diagrams, circular connectograms, spring-layout
cluster diagrams, and the small-world and degree-distribution summaries.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from . import theme


def plot_network(G, probe, node_values=None, ax=None, show_labels=False, title=None,
                  node_size=50, edge_alpha=0.2, node_alpha=0.8, font_size=6,
                  show_axis=True):
    """
    Draw a graph with each node at its real probe position.

    Parameters
    ----------
    G : networkx.Graph
        Graph whose node names are probe channel labels.
    probe : Probe
        Probe supplying the node positions.
    node_values : array-like, optional
        Per-node value used to size and colour the nodes on a sequential
        colormap. All nodes share one flat colour if omitted.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.
    show_labels : bool, optional
        Annotate each node with its channel label. Default is False.
    title : str, optional
        Axes title.
    node_size : int, optional
        Base marker size for the nodes. Default is 50.
    edge_alpha, node_alpha : float, optional
        Opacity of the edges and nodes. Defaults are 0.2 and 0.8.
    font_size : int, optional
        Font size for the node labels. Default is 6.
    show_axis : bool, optional
        Keep labelled axes visible. Default is True.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the diagram.
    """
    all_pos_2d = probe.get_2d_coords()
    node_list = list(G.nodes())

    if all(n in all_pos_2d for n in node_list):
        pos_2d = {node: all_pos_2d[node] for node in node_list}
    else:
        # ROI/mesoscale fallback: index-based matching
        coords_list = list(all_pos_2d.values())
        pos_2d = {node: coords_list[i] for i, node in enumerate(node_list) if i < len(coords_list)}

    if node_values is not None:
        node_sizes = np.asarray(node_values) * (node_size / 2)
        node_color = node_sizes
    else:
        node_sizes = node_size
        node_color = theme.categorical_color(1)

    if ax is None:
        fig, ax = plt.subplots(figsize=theme.graph_figsize(len(node_list), labeled=show_labels))
    else:
        fig = ax.figure

    nx.draw_networkx_edges(G, pos_2d, ax=ax, edge_color='gray', alpha=edge_alpha, width=0.5)
    nx.draw_networkx_nodes(G, pos_2d, ax=ax, node_size=node_sizes, node_color=node_color,
                            cmap=theme.SEQUENTIAL_CMAP, alpha=node_alpha,
                            edgecolors='white', linewidths=0.3)
    if show_labels:
        nx.draw_networkx_labels(G, pos_2d, ax=ax, font_size=font_size, font_color='black')

    ax.set_title(f"Network Topology: {title if title is not None else ''}".strip())
    theme.style_spatial_axes(ax, show_axis=show_axis)
    return fig, ax


def plot_circular_network(matrix, channels, threshold=None, chromophore='', ax=None):
    """
    Draw a connectogram with the nodes arranged on a ring.

    Edges are coloured by sign, warm for positive and cool for negative.

    Parameters
    ----------
    matrix : array-like
        Square connectivity matrix.
    channels : sequence of str
        Channel labels, in the same order as the matrix rows.
    threshold : float, optional
        Minimum absolute weight for an edge to be drawn. All edges are drawn
        if omitted.
    chromophore : str, optional
        Chromophore named in the title.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the connectogram.
    """
    G = nx.from_numpy_array(np.asarray(matrix))
    G = nx.relabel_nodes(G, {i: name for i, name in enumerate(channels)})

    if threshold is not None:
        drop = [(u, v) for u, v, d in G.edges(data=True) if abs(d.get('weight', 0)) < threshold]
        G.remove_edges_from(drop)

    if ax is None:
        fig, ax = plt.subplots(figsize=theme.graph_figsize(G.number_of_nodes(), labeled=True))
    else:
        fig = ax.figure

    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=300, node_color=theme.UNASSIGNED_COLOR)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    for u, v, d in G.edges(data=True):
        w = d['weight']
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, width=abs(w) * 2,
                                edge_color=theme.HBO_COLOR if w > 0 else theme.HBR_COLOR, alpha=0.5)

    title = f"Circular Network: {chromophore}" if chromophore else "Circular Network"
    if threshold is not None:
        title += f" (|r| ≥ {threshold})"
    ax.set_title(title)
    theme.style_spatial_axes(ax, show_axis=False)
    return fig, ax


def plot_network_clusters(G, chromophore='', weight_threshold=0.3, show_labels=False, ax=None,
                           k=None, iterations=100, node_alpha=0.9, font_size=8):
    """
    Draw a graph with a force-directed layout, so clusters separate.

    Edges below ``weight_threshold`` are dropped before the layout is
    computed.

    Parameters
    ----------
    G : networkx.Graph
        Graph to lay out.
    chromophore : str, optional
        Chromophore named in the title.
    weight_threshold : float, optional
        Edges weaker than this are ignored when computing the layout.
        Default is 0.3.
    show_labels : bool, optional
        Annotate each node with its channel label. Default is False.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.
    k : float, optional
        Target distance between nodes in the spring layout. Chosen by
        NetworkX if omitted.
    iterations : int, optional
        Layout iterations. Default is 100.
    node_alpha : float, optional
        Opacity of the nodes. Default is 0.9.
    font_size : int, optional
        Font size for the node labels. Default is 8.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the diagram.
    """
    G = G.copy()
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True)
                        if abs(d.get('weight', 0)) < weight_threshold]
    G.remove_edges_from(edges_to_remove)

    if k is None:
        k = 1 / np.sqrt(max(len(G.nodes()), 1))
    pos = nx.spring_layout(G, k=k, iterations=iterations)

    if ax is None:
        fig, ax = plt.subplots(figsize=theme.graph_figsize(G.number_of_nodes(), labeled=show_labels))
    else:
        fig = ax.figure

    d = dict(G.degree(weight='weight' if nx.is_weighted(G) else None))
    node_sizes = [v * 100 for v in d.values()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=theme.categorical_color(0),
                            edgecolors='white', linewidths=0.5, alpha=node_alpha)
    if show_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size, font_weight='bold')

    for u, v, d in G.edges(data=True):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax,
                                width=abs(d.get('weight', 1)) * 2, alpha=0.2, edge_color='gray')

    ax.set_title(f"Functional Clusters (Spring Layout)\n{chromophore} | r > {weight_threshold}")
    theme.style_spatial_axes(ax, show_axis=False)
    return fig, ax


def plot_small_world_comparison(gamma, lambd, sigma, chromophore='', ax=None):
    """
    Plot the small-world metrics normalised to random-graph nulls.

    Parameters
    ----------
    gamma : float
        Normalised clustering coefficient.
    lambd : float
        Normalised characteristic path length.
    sigma : float
        Small-world index, the ratio of the two.
    chromophore : str, optional
        Chromophore named in the title.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the chart.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=theme.FIGSIZE['single'])
    else:
        fig = ax.figure

    metrics = ['Gamma (C_norm)', 'Lambda (L_norm)', 'Sigma (SW Index)']
    values = [gamma, lambd, sigma]
    colors = [theme.categorical_color(i) for i in range(3)]

    ax.bar(metrics, values, color=colors)
    ax.axhline(1.0, linestyle='--', color='black', alpha=0.6, label='Random Level')

    ax.set_ylabel('Ratio to Random Graph')
    ax.set_title(f"Small-World Metrics: {chromophore}" if chromophore else "Small-World Metrics")
    ax.legend()
    theme.style_quantitative_axes(ax)
    return fig, ax


def plot_degree_distribution(degrees, chromophore='', ax=None):
    """
    Plot the distribution of nodal degree, with a hub threshold marked.

    The threshold is drawn two standard deviations above the mean, and a
    density curve is overlaid where there are enough nodes to support one.

    Parameters
    ----------
    degrees : array-like
        Degree of each node.
    chromophore : str, optional
        Chromophore named in the title.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the histogram.
    """
    degrees = np.asarray(degrees)

    if ax is None:
        fig, ax = plt.subplots(figsize=theme.FIGSIZE['wide'])
    else:
        fig = ax.figure

    bins = max(int(np.sqrt(len(degrees))), 1)
    counts, bin_edges, _ = ax.hist(degrees, bins=bins, color=theme.categorical_color(1),
                                    alpha=0.7, edgecolor='white', density=False)

    if len(degrees) >= 5 and np.std(degrees) > 0:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(degrees)
        xs = np.linspace(degrees.min(), degrees.max(), 200)
        bin_width = bin_edges[1] - bin_edges[0]
        ax.plot(xs, kde(xs) * len(degrees) * bin_width, color=theme.categorical_color(1),
                linewidth=1.8)

    ax.set_title(f"Degree Distribution: {chromophore} (N={len(degrees)})" if chromophore
                 else f"Degree Distribution (N={len(degrees)})")
    ax.set_xlabel("Degree (Number of Connections)")
    ax.set_ylabel("Number of Channels")

    mean_d, std_d = np.mean(degrees), np.std(degrees)
    ax.axvline(mean_d + 2 * std_d, color=theme.CATEGORICAL_PALETTE[0], linestyle='--',
               label='Potential Hub Threshold')
    ax.legend()
    theme.style_quantitative_axes(ax)
    return fig, ax
