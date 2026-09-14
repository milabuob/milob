from .output import BaseOutput
import networkx as nx 
import numpy as np


class GraphOutput(BaseOutput):
    """
    Graph-theoretic analysis of a connectivity result.

    Holds one NetworkX graph per chromophore, together with the global and
    nodal metrics computed from them.

    Parameters
    ----------
    graphs : dict
        Mapping of chromophore name to :class:`networkx.Graph`.
    data : xarray.Dataset, optional
        Metrics already computed for these graphs.
    probe : Probe, optional
        Probe geometry, required for the spatial plotting methods.
    analysis_type : str, optional
        Label stored with the output. Default is ``'Graph'``.
    source_conn : FCOutput, optional
        Connectivity output the graphs were built from, used by
        :meth:`plot_circular_network`.
    """
    def __init__(self, graphs, data=None, probe=None, analysis_type="Graph", source_conn=None):
        # Initialize the base class properties
        super().__init__(data=data, probe=probe, analysis_type=analysis_type)
        self.graphs = graphs
        # Store the parent ConnectivityOutput object
        self.source_conn = source_conn



    def compute_network_metrics(self, global_metrics=None, nodal_metrics=None,
                            compute_small_world=False, n_nulls=20):
        """
        Compute global and nodal network metrics for every graph.

        Parameters
        ----------
        global_metrics : list of str, optional
            Whole-network metrics to compute, from ``'density'``,
            ``'clustering'``, ``'efficiency'`` and ``'path_length'``.
            Default is ``['density', 'clustering', 'efficiency']``.
        nodal_metrics : list of str, optional
            Per-channel metrics to compute, from ``'degree'``,
            ``'betweenness'``, ``'clustering'`` and ``'efficiency'``.
            Default is ``['degree']``.
        compute_small_world : bool, optional
            If True, also compute the small-world index. Default is False.
        n_nulls : int, optional
            Number of null graphs used for the small-world index. Default is 20.

        Returns
        -------
        GraphOutput
            Self, with the metrics stored in ``output``.
        """
        import networkx as nx
        import xarray as xr
        import numpy as np

        global_metrics = global_metrics or ['density', 'clustering', 'efficiency']
        nodal_metrics = nodal_metrics or ['degree']

        chroms = list(self.graphs.keys())
        # Use the full probe channel list from the source to ensure 
        # the output xarray matches the original geometry.
        full_channel_list = list(self.source_conn.output.channel_i.values)
        
        data_vars = {}

        # --- Compute Global Metrics ---
        for metric in global_metrics:
            values = []
            for chrom in chroms:
                G = self.graphs[chrom]
                is_w = nx.is_weighted(G)
                w_attr = 'weight' if is_w else None

                if metric == 'efficiency':
                    values.append(nx.global_efficiency(G)) 
                elif metric == 'clustering':
                    values.append(nx.average_clustering(G, weight=w_attr))
                elif metric == 'density':
                    values.append(nx.density(G))
                elif metric == 'path_length':
                    if nx.is_connected(G):
                        values.append(nx.average_shortest_path_length(G, weight=w_attr))
                    else:
                        values.append(np.nan)   # if graph is disconnected, path length is undefined/infinite
            
            data_vars[f"global_{metric}"] = (['chromophore'], values)

        # --- Compute Nodal Metrics ---
        # These map back to the full probe (Bad channels = NaN)
        for metric in nodal_metrics:
            matrix = np.full((len(full_channel_list), len(chroms)), np.nan)
            
            for i, chrom in enumerate(chroms):
                G = self.graphs[chrom]
                is_w = nx.is_weighted(G)
                w_attr = 'weight' if is_w else None
                
                if metric == 'degree':
                    d = dict(G.degree(weight=w_attr))
                elif metric == 'efficiency':
                    # Nodal efficiency is the average of the inverse shortest path lengths 
                    # between node 'n' and all other nodes 'm'.
                    d = {}
                    for n in G.nodes:
                        # Calculate the average efficiency between node n and every other node individually
                        efficiencies = [nx.efficiency(G, n, m) for m in G.nodes if m != n]
                        # Nodal efficiency is defined as the mean of these pairwise values
                        d[n] = np.mean(efficiencies) if efficiencies else 0
                elif metric == 'betweenness':
                    d = nx.betweenness_centrality(G, weight=w_attr)
                elif metric == 'clustering':
                    d = nx.clustering(G, weight=w_attr)
                
                # Map values back to the full probe array using node names
                for j, node_name in enumerate(full_channel_list):
                    if node_name in d:
                        matrix[j, i] = d[node_name]
                    
            data_vars[f"nodal_{metric}"] = (['channel', 'chromophore'], matrix)

        # Small Worldness 
        if compute_small_world:
            sw_res = {'sigma': [], 'gamma': [], 'lambda': []}
            for chrom in chroms:
                stats = self.compute_small_worldness(chromophore=chrom, n_nulls=n_nulls)
                for key in sw_res:
                    sw_res[key].append(stats[key])
            
            for key, vals in sw_res.items():
                data_vars[f"global_{key}"] = (['chromophore'], vals)

        # Package into the Output's .data (xarray Dataset)
        self.output = xr.Dataset(
            data_vars=data_vars, 
            coords={
                'channel': full_channel_list, 
                'chromophore': chroms
            }
        )
        
        return self
    
    
    
    def compute_small_worldness(self, chromophore='HbO', n_nulls=10):
        """
        Compute the small-world index against rewired null graphs.

        Parameters
        ----------
        chromophore : str, optional
            Chromophore whose graph is analysed. Default is ``'HbO'``.
        n_nulls : int, optional
            Number of null graphs averaged over. Default is 10.

        Returns
        -------
        dict
            Normalised clustering ``'gamma'``, normalised path length
            ``'lambda'``, and their ratio ``'sigma'``.

        References
        ----------
        .. [1] Humphries, M.D. and Gurney, K. (2008). Network 'small-world-ness':
               a quantitative method for determining canonical network equivalence.
               PLoS ONE, 3(4), e0002051.
        """
        G = self.graphs[chromophore]
        
        C_obs = nx.average_clustering(G)
        if nx.is_connected(G):
            L_obs = nx.average_shortest_path_length(G)
        else:
            L_obs = 1 / nx.global_efficiency(G) if nx.global_efficiency(G) > 0 else np.nan
                    
        # Compute Null Metrics
        C_nulls = []
        L_nulls = []
        
        for _ in range(n_nulls):
            # generate_null_network using 'rewire' for Maslov-Sneppen model
            R = self.generate_null_network(chromophore, model='rewire')
            C_nulls.append(nx.average_clustering(R))
        
            if nx.is_connected(R):
                L_nulls.append(nx.average_shortest_path_length(R))
            else:
                eff_rand = nx.global_efficiency(R)
                if eff_rand > 0:
                    L_nulls.append(1 / eff_rand)
                else:
                    L_nulls.append(np.nan)
    
        C_rand = np.nanmean(C_nulls)
        L_rand = np.nanmean(L_nulls)
                
        # Calculate Normalized Metrics
        gamma = C_obs / C_rand if C_rand > 0 else np.nan
        lambd = L_obs / L_rand if (L_rand > 0 and not np.isnan(L_rand)) else np.nan
        sigma = gamma / lambd if (not np.isnan(gamma) and not np.isnan(lambd) and lambd > 0) else np.nan
        
        return {'sigma': sigma, 'gamma': gamma, 'lambda': lambd}        
        


    # ****************************
    #  GENERATIVE MODELS
    # ****************************        
    # This can be later reorganised into a GraphsModel class as engine to allow parallelization
    
    def generate_null_network(self, chromophore='HbO', model='rewire', iterations=10):
        """
        Generate a random null graph for comparison.

        Parameters
        ----------
        chromophore : str, optional
            Chromophore whose graph is randomised. Default is ``'HbO'``.
        model : {'rewire', 'random'}, optional
            ``'rewire'`` applies degree-preserving edge swaps; ``'random'``
            draws an Erdos-Renyi graph with the same number of nodes and
            edges. Default is ``'rewire'``.
        iterations : int, optional
            Edge swaps per edge, used by ``'rewire'`` only. Default is 10.

        Returns
        -------
        networkx.Graph
            The null graph.

        References
        ----------
        .. [1] Maslov, S. and Sneppen, K. (2002). Specificity and stability in
               topology of protein networks. Science, 296(5569), 910-913.
        """
        G = self.graphs[chromophore].copy()
        
        if model == 'random':
            # Erdős-Rényi: Same nodes (n) and same number of edges (m)
            n = G.number_of_nodes()
            m = G.number_of_edges()
            # Generate a new random graph
            R = nx.gnm_random_graph(n, m)
            # Re-map the labels to match the original channels
            mapping = {i: name for i, name in enumerate(G.nodes())}
            null_G = nx.relabel_nodes(R, mapping)
            
        elif model == 'rewire':
            # Maslov-Sneppen: Swaps edges (u-v, x-y -> u-y, x-v) 
            # This preserves the degree of every single node.
            # swap_attempts: total swaps = m * iterations
            null_G = G.copy()
            # We use the standard swap instead of the connected_swap
            # to handle fNIRS graphs that often have disconnected nodes.
            try:
                nx.double_edge_swap(
                    null_G, 
                    nswap=G.number_of_edges() * iterations, 
                    max_tries=G.number_of_edges() * iterations * 10,
                    seed=None
                )
            except nx.NetworkXError:
                # Fallback if the graph is too sparse to even swap
                print(f"Warning: Graph for {chromophore} is too sparse for rewiring.")                
        return null_G
    
    
    
    def generate_network_model(self, chromophore='HbO', model='small_world', p=0.1, m=2):
        """
        Generate a theoretical graph with the same number of nodes.

        Parameters
        ----------
        chromophore : str, optional
            Chromophore whose graph sets the node count. Default is ``'HbO'``.
        model : {'small_world', 'scale_free'}, optional
            ``'small_world'`` uses the Watts-Strogatz model, ``'scale_free'``
            the Barabasi-Albert model. Default is ``'small_world'``.
        p : float, optional
            Rewiring probability for the Watts-Strogatz model. Default is 0.1.
        m : int, optional
            Edges attached per new node in the Barabasi-Albert model.
            Default is 2.

        Returns
        -------
        networkx.Graph
            The generated graph, relabelled with the channel names.

        References
        ----------
        .. [1] Watts, D.J. and Strogatz, S.H. (1998). Collective dynamics of
               'small-world' networks. Nature, 393(6684), 440-442.
        .. [2] Barabasi, A.-L. and Albert, R. (1999). Emergence of scaling in
               random networks. Science, 286(5439), 509-512.
        """
        G_obs = self.graphs[chromophore]
        n = G_obs.number_of_nodes()
        # Calculate average degree k (each edge is counted twice)
        k = int(round(2 * G_obs.number_of_edges() / n))

        if model == 'small_world':
            # Watts-Strogatz requires k to be even
            if k % 2 != 0: k += 1
            G_gen = nx.watts_strogatz_graph(n, k, p)
            
        elif model == 'scale_free':
            # Barabási-Albert: Starts with m nodes and adds more
            # Note: m must be >= 1 and < n
            G_gen = nx.barabasi_albert_graph(n, m)
            
        # Re-map the labels to match the original channels for consistency
        mapping = {i: name for i, name in enumerate(G_obs.nodes())}
        return nx.relabel_nodes(G_gen, mapping)
    
    
    def benchmark_against_model(self, chromophore='HbO', models=['random', 'small_world'], n_iterations=10):
        """
        Compare observed clustering and efficiency against several models.

        Parameters
        ----------
        chromophore : str, optional
            Chromophore whose graph is compared. Default is ``'HbO'``.
        models : list of str, optional
            Models to benchmark against, from ``'random'``, ``'small_world'``
            and ``'scale_free'``. Default is ``['random', 'small_world']``.
        n_iterations : int, optional
            Graphs generated per model. Default is 10.

        Returns
        -------
        dict
            For each model, the observed-to-model ratios
            ``'clustering_ratio'`` and ``'efficiency_ratio'``.
        """
        G_obs = self.graphs[chromophore]
        obs_metrics = {
            'C': nx.average_clustering(G_obs),
            'E': nx.global_efficiency(G_obs)
        }
        
        results = {}
        for model in models:
            c_list, e_list = [], []
            for _ in range(n_iterations):
                if model == 'random':
                    G_null = self.generate_null_network(chromophore, model='rewire')
                else:
                    G_null = self.generate_network_model(chromophore, model=model)
                
                c_list.append(nx.average_clustering(G_null))
                e_list.append(nx.global_efficiency(G_null))
            
            # Calculate ratios (Observed / Model)
            results[model] = {
                'clustering_ratio': obs_metrics['C'] / np.mean(c_list),
                'efficiency_ratio': obs_metrics['E'] / np.mean(e_list)
            }
        return results

        

    # ****************************
    #  VISUALIZATION TOOLS
    # ****************************
    
    def plot_network(self, chromophore='HbT', ax=None, show_labels=False, title=None, **kwargs):
        """
        Plot the network over the probe layout.

        Parameters
        ----------
        chromophore : str, optional
            Chromophore whose graph is drawn. Default is ``'HbT'``.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created if omitted.
        show_labels : bool, optional
            If True, annotate each node with its channel label. Default is False.
        title : str, optional
            Title for the plot.
        **kwargs
            Passed to :func:`milob.viz.network.plot_network`, which accepts
            ``node_size`` among others.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the network.

        Raises
        ------
        KeyError
            If no graph exists for the requested chromophore.
        """
        from ..viz.network import plot_network

        if chromophore not in self.graphs:
            raise KeyError(f"Graph for {chromophore} not found.")

        G = self.graphs[chromophore]
        node_values = None
        if 'nodal_degree' in self.output:
            node_values = [self.output.nodal_degree.sel(chromophore=chromophore, channel=node).values
                            for node in G.nodes()]

        return plot_network(
            G, self.probe, node_values=node_values, ax=ax, show_labels=show_labels,
            title=f"{chromophore} ({title if title is not None else 'n/d'})",
            node_size=kwargs.get('node_size', 50), edge_alpha=kwargs.get('edge_alpha', 0.2),
            node_alpha=kwargs.get('node_alpha', 0.8), font_size=kwargs.get('font_size', 6),
        )
    
    
    
            
    def plot_nodal_metric(self, metric='nodal_degree', chromophore='HbO', **kwargs):
        """
        Plot a nodal metric as a topographic map.

        Parameters
        ----------
        metric : str, optional
            Name of the nodal metric to map. Default is ``'nodal_degree'``.
        chromophore : str, optional
            Chromophore to display. Default is ``'HbO'``.
        **kwargs
            Passed to :func:`milob.viz.topo.plot_topo_map`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the map.
        """
        # Since self.output is an xarray with 'channel' and 'chromophore', 
        # it is 100% compatible with our existing viz engine.
        from ..viz.topo import plot_topo_map
        
        values = self.output[metric].sel(chromophore=chromophore).values
        title = kwargs.pop('title', f"Nodal Map: {metric} ({chromophore})")
        
        return plot_topo_map(self.probe, values=values, title=title, **kwargs)
    
    
    def plot_circular_network(self, chromophore='HbT', threshold=None, ax=None, **kwargs):
        """
        Plot the connections as a circular graph.

        Parameters
        ----------
        chromophore : str, optional
            Chromophore to display. Default is ``'HbT'``.
        threshold : float, optional
            Minimum absolute weight for an edge to be drawn. All edges are
            drawn if omitted.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created if omitted.
        **kwargs
            Passed to :func:`milob.viz.network.plot_circular_network`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        from ..viz.network import plot_circular_network

        # Get the matrix from the source ConnectivityOutput
        # (We access the parent connectivity object that created this network)
        data = self.source_conn.output.value.sel(chromophore=chromophore)
        return plot_circular_network(
            data.values, data.channel_i.values, threshold=threshold,
            chromophore=chromophore, ax=ax,
        )
    
    
    
    def plot_network_clusters(self, chromophore='HbO', weight_threshold=0.3, show_labels=False, ax=None, **kwargs):
        """
        Plot the network with a force-directed layout that groups clusters.

        Parameters
        ----------
        chromophore : str, optional
            Chromophore whose graph is drawn. Default is ``'HbO'``.
        weight_threshold : float, optional
            Edges weaker than this are ignored when computing the layout.
            Default is 0.3.
        show_labels : bool, optional
            If True, annotate each node with its channel label. Default is False.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created if omitted.
        **kwargs
            Passed to :func:`milob.viz.network.plot_network_clusters`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        from ..viz.network import plot_network_clusters

        if chromophore not in self.graphs:
            raise KeyError(f"Graph for {chromophore} not found.")

        return plot_network_clusters(
            self.graphs[chromophore], chromophore=chromophore,
            weight_threshold=weight_threshold, show_labels=show_labels, ax=ax,
            k=kwargs.get('k'), iterations=kwargs.get('iterations', 100),
            node_alpha=kwargs.get('node_alpha', 0.9), font_size=kwargs.get('font_size', 8),
        )
    
    
    
    
    def plot_small_world_comparison(self, chromophore='HbO', ax=None):
        """
        Plot the small-world metrics against their null distributions.

        Parameters
        ----------
        chromophore : str, optional
            Chromophore to display. Default is ``'HbO'``.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created if omitted.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        from ..viz.network import plot_small_world_comparison

        gamma = self.output.global_gamma.sel(chromophore=chromophore).values
        lambd = self.output.global_lambda.sel(chromophore=chromophore).values
        sigma = self.output.global_sigma.sel(chromophore=chromophore).values
        return plot_small_world_comparison(gamma, lambd, sigma, chromophore=chromophore, ax=ax)

    def plot_degree_distribution(self, chromophore='HbO', ax=None, **kwargs):
        """
        Plot the degree distribution of the network.

        Parameters
        ----------
        chromophore : str, optional
            Chromophore to display. Default is ``'HbO'``.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created if omitted.
        **kwargs
            Passed to :func:`milob.viz.network.plot_degree_distribution`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.

        Raises
        ------
        ValueError
            If nodal degree has not been computed.
        """
        from ..viz.network import plot_degree_distribution

        if 'nodal_degree' not in self.output:
            raise ValueError("Nodal degree not computed. Run compute_metrics first.")

        degrees = self.output.nodal_degree.sel(chromophore=chromophore).values
        return plot_degree_distribution(degrees, chromophore=chromophore, ax=ax)