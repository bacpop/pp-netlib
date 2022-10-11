########################
####   .CONSTRUCT   ####
########################
import scipy
from scipy.stats import rankdata
import numpy as np
import pandas as pd
import os, sys

def construct_with_graphtool(network_data, vertex_labels, weights = None):
    """Construct a graph with graph-tool

    Args:
        network_data (dataframe OR edge list OR sparse coordinate matrix): Data containing record of edges in the graph.
        vertex_labels (list): List of vertex/node labels to apply to graph vertices
        weights (list, optional): List of weights associated with edges in network_data.
                                      Weights must be in the same order as edges in network_data. Defaults to None.

    Returns:
        graph (gt.Graph): Graph-tool graph object populated with network data
    """
    import graph_tool.all as gt

    graph = gt.Graph(directed = False) ## initialise graph_tool graph object

    ########################
    ####    DF INPUT    ####
    ########################
    if isinstance(network_data, pd.DataFrame):
        network_data.columns = ["source", "destination"]

        graph.add_vertex(len(vertex_labels)) ## add vertices

        ## add weights column if weights provided as list (add error catching?)
        if weights is not None:
            network_data["weights"] = weights
            eweight = graph.new_ep("float")
            graph.add_edge_list(network_data.values, eprops = [eweight]) ## add weighted edges
            graph.edge_properties["weight"] = eweight
        else:
            graph.add_edge_list(network_data.values) ## add edges

    ##########################
    #### SPARSE MAT INPUT ####
    ##########################
    elif isinstance(network_data, scipy.sparse.coo_matrix):
        graph_data_df = pd.DataFrame()
        graph_data_df["source"] = network_data.row
        graph_data_df["destination"] =  network_data.col
        graph_data_df["weights"] = network_data.data

        graph.add_vertex(len(vertex_labels)) ## add vertices
        eweight = graph.new_ep("float")
        graph.add_edge_list(list(map(tuple, graph_data_df.values)), eprops = [eweight]) ## add weighted edges
        graph.edge_properties["weight"] = eweight

    ########################
    ####   LIST INPUT   ####
    ########################
    elif isinstance(network_data, list):
        graph.add_vertex(len(vertex_labels)) ## add vertices

        if weights is not None:
            weighted_edges = []

            for i in range(len(network_data)):
                weighted_edges.append(network_data[i] + (weights[i],))

            eweight = graph.new_ep("float")
            graph.add_edge_list(weighted_edges, eprops = [eweight]) ## add weighted edges
            graph.edge_properties["weight"] = eweight

        else:
            graph.add_edge_list(network_data) ## add edges

    v_name_prop = graph.new_vp("string")
    graph.vertex_properties["id"] = v_name_prop
    for i in range(len([v for v in graph.vertices()])):
        v_name_prop[graph.vertex(i)] = vertex_labels[i]

    return graph

def construct_with_networkx(network_data, vertex_labels, weights = None):
    """Construct a graph with networkx

    Args:
        network_data (dataframe OR edge list OR sparse coordinate matrix): Data containing record of edges in the graph.
        vertex_labels (list): List of vertex/node labels to apply to graph vertices
        weights (list, optional): List of weights associated with edges in network_data.
                                      Weights must be in the same order as edges in network_data. Defaults to None.

    Returns:
        graph (nx.Graph): Graph-tool graph object populated with network data
    """
    import networkx as nx
    
    ## initialise nx graph and add nodes
    graph = nx.Graph()

    nodes_list = [(i, dict(id=vertex_labels[i])) for i in range(len(vertex_labels))]
    graph.add_nodes_from(nodes_list)

    ########################
    ####    DF INPUT    ####
    ########################
    if isinstance(network_data, pd.DataFrame):
        network_data.columns = ["source", "destination"]
        if weights is not None:
            network_data["weights"] = weights
            graph.add_weighted_edges_from(network_data.values)
        else:
            graph.add_edges_from(network_data.values)

    ##########################
    #### SPARSE MAT INPUT ####
    ##########################
    elif isinstance(network_data, scipy.sparse.coo_matrix):
        weighted_edges = list(zip(list(network_data.row), list(network_data.col), list(network_data.data)))
        graph.add_weighted_edges_from(weighted_edges)

    ########################
    ####   LIST INPUT   ####
    ########################
    elif isinstance(network_data, list):
        if weights is not None:
            src, dest = zip(*network_data)
            weighted_edges = list(zip(src, dest, weights))
            graph.add_weighted_edges_from(weighted_edges)
        else:
            graph.add_edges_from(network_data)

    return graph

########################
####   .SUMMARISE   ####
########################
def summarise(graph, backend):
    """Get graph metrics and format into soutput string.

    Args:
        graph (Network object): The graph for which to obtain summary metrics
        backend (str): The tool used to build graph ("GT", "NX", or "CU"(TODO))

    Returns:
        summary_contents (formatted str): Graph summary metrics formatted to print to stderr
    """
    if backend == "GT":
        import graph_tool.all as gt
        component_assignments, component_frequencies = gt.label_components(graph)
        components = len(component_frequencies)
        density = len(list(graph.edges()))/(0.5 * len(list(graph.vertices())) * (len(list(graph.vertices())) - 1))
        transitivity = gt.global_clustering(graph)[0]

        mean_bt = 0
        weighted_mean_bt = 0
        betweenness = []
        sizes = []
        for component, size in enumerate(component_frequencies):
            if size > 3:
                vfilt = component_assignments.a == component
                subgraph = gt.GraphView(graph, vfilt=vfilt)
                betweenness.append(max(gt.betweenness(subgraph, norm = True)[0].a))
                sizes.append(size)

        if len(betweenness) > 1:
            mean_bt = np.mean(betweenness)
            weighted_mean_bt = np.average(betweenness, weights=sizes)
        elif len(betweenness) == 1:
            mean_bt = betweenness[0]
            weighted_mean_bt = betweenness[0]

    elif backend == "NX":
        import networkx as nx
        components = nx.number_connected_components(graph)
        density = nx.density(graph)
        transitivity = nx.transitivity(graph)

        betweenness = []
        sizes = []
        for c in nx.connected_components(graph):
            betweenness.append(max((nx.betweenness_centrality(graph.subgraph(c))).values()))
            sizes.append(len(graph.subgraph(c)))

        if len(betweenness) > 1:
            mean_bt = np.mean(betweenness)
            weighted_mean_bt = np.average(betweenness, weights=sizes)
        elif len(betweenness) == 1:
            mean_bt = betweenness[0]
            weighted_mean_bt = betweenness[0]

    metrics = [components, density, transitivity, mean_bt, weighted_mean_bt]
    base_score = transitivity * (1 - density)
    scores = [base_score, base_score * (1 - metrics[3]), base_score * (1 - metrics[4])]
    
    return metrics, scores

########################
####      .SAVE     ####
########################
def save_graph(graph, backend, outdir, file_name, file_format):
    if backend == "GT":
        import graph_tool.all as gt
        if file_format is None:
            graph.save(os.path.join(outdir, file_name+".gt"))
        elif file_format is not None:
            if file_format not in [".gt", ".graphml"]:
                raise NotImplementedError("Supported file formats to save a graph-tools graph are .gt or .graphml")
            else:
                graph.save(os.path.join(outdir, file_name+file_format))

    if backend == "NX":
        import networkx as nx
        nx.write_graphml(graph, os.path.join(outdir, file_name+".graphml"))

########################
####   .VISUALISE   ####
########################
def gt_generate_mst(graph):
    import graph_tool.all as gt

    mst_edge_prop_map = gt.min_spanning_tree(graph, weights = graph.ep["weight"])
    mst_network = gt.GraphView(graph, efilt = mst_edge_prop_map)
    mst_network = gt.Graph(mst_network, prune = True)

    num_components = 1
    seed_vertices = set()

    component_assignments, component_frequencies = gt.label_components(mst_network)
    num_components = len(component_frequencies)
    if num_components > 1:
        for component_index in range(len(component_frequencies)):
            component_members = component_assignments.a == component_index
            component = gt.GraphView(mst_network, vfilt = component_members)
            component_vertices = component.get_vertices()
            out_degrees = component.get_out_degrees(component_vertices)
            seed_vertex = list(component_vertices[np.where(out_degrees == np.amax(out_degrees))])
            seed_vertices.add(seed_vertex[0]) # Can only add one otherwise not MST

    if num_components > 1:
        # With graph-tool look to retrieve edges in larger graph
        connections = []
        max_weight = float(np.max(graph.edge_properties["weight"].a))

        # Identify edges between seeds to link components together
        for ref in seed_vertices:
            seed_edges = graph.get_all_edges(ref, [graph.ep["weight"]])
            found = False  # Not all edges may be in graph
            for seed_edge in seed_edges:
                if seed_edge[1] in seed_vertices:
                    found = True
                    connections.append((seed_edge))
            # TODO: alternative would be to requery the DB (likely quick)
            if found == False:
                for query in seed_vertices:
                    if query != ref:
                        connections.append((ref, query, max_weight))
        # Construct graph
        seed_G = gt.Graph(directed = False)
        seed_G.add_vertex(len(seed_vertices))
        eweight = seed_G.new_ep("float")
        seed_G.add_edge_list(connections, eprops = [eweight])
        seed_G.edge_properties["weight"] = eweight
        seed_mst_edge_prop_map = gt.min_spanning_tree(seed_G, weights = seed_G.ep["weight"])
        seed_mst_network = gt.GraphView(seed_G, efilt = seed_mst_edge_prop_map)
        # Insert seed MST into original MST - may be possible to use graph_union with include=True & intersection
        deep_edges = seed_mst_network.get_edges([seed_mst_network.ep["weight"]])
        mst_network.add_edge_list(deep_edges)


    return mst_network

def nx_generate_mst(graph):
    import networkx as nx

    mst_network = nx.minimum_spanning_tree(graph)

    num_components = 1
    seed_vertices = set()

    num_components = nx.number_connected_components(mst_network)
    if num_components > 1:
        for component in nx.connected_components(mst_network):
            # comp_graphview = nx.subgraph(mst_network, component)
            comp_vertices = list(component)
            out_degrees = np.array(j for (i, j) in mst_network.out_degree(comp_vertices))
            seed_vertex = list(comp_vertices[np.where(out_degrees == np.amax(out_degrees))])
            seed_vertices.add(seed_vertex[0])

    if num_components >1:
        connections = []
        weights = list(k for (i, j, k) in graph.edges.data("weight"))
        max_weight = float(np.max(weights))

        for ref in seed_vertices:
            seed_edges = graph.edges(ref).data("weight")
            found = False
            for seed_edge in seed_edges:
                if seed_edge[1] in seed_vertices:
                    found = True
                    connections.append((seed_edge))
    
            if found == False:
                for query in seed_vertices:
                    if query != ref:
                        connections.append((ref, query, max_weight))

        seed_G = nx.Graph()
        seed_G.add_nodes_from(len(seed_vertex))
        seed_G.add_weighted_edges_from(connections)

        seed_mst_network = nx.minimum_spanning_tree(seed_G)
        deep_edges = list(seed_mst_network.edges(data = "weight"))
        mst_network.add_weighted_edges_from(deep_edges)

    return mst_network

def cu_generate_mst(graph):
    import cugraph, cudf
    """Generate a minimum spanning tree from a network
    Args:
       G (network)
           Graph tool network
       from_cugraph (bool)
            If a pre-calculated MST from cugraph
            [default = False]
    Returns:
       mst_network (str)
           Minimum spanning tree (as graph-tool graph)
    """
    #
    # Create MST
    #

    mst_network = graph

    # Find seed nodes as those with greatest outdegree in each component
    num_components = 1
    seed_vertices = set()
    
    mst_df = cugraph.components.connectivity.connected_components(mst_network)
    num_components_idx = mst_df["labels"].max()
    num_components = mst_df.iloc[num_components_idx]["labels"]
    if num_components > 1:
        mst_df["degree"] = mst_network.in_degree()["degree"]
        # idxmax only returns first occurrence of maximum so should maintain
        # MST - check cuDF implementation is the same
        max_indices = mst_df.groupby(["labels"])["degree"].idxmax()
        seed_vertices = mst_df.iloc[max_indices]["vertex"]

    # If multiple components, add distances between seed nodes
    if num_components > 1:

        # Extract edges and maximum edge length - as DF for cugraph
        # list of tuples for graph-tool
        
        # With cugraph the MST is already calculated
        # so no extra edges can be retrieved from the graph
        G_df = graph.view_edge_list()
        max_weight = G_df["weights"].max()
        first_seed = seed_vertices.iloc[0]
        G_seed_link_df = cudf.DataFrame()
        G_seed_link_df["dst"] = seed_vertices.iloc[1:seed_vertices.size]
        G_seed_link_df["src"] = seed_vertices.iloc[0]
        G_seed_link_df["weights"] = seed_vertices.iloc[0]
        G_df = G_df.append(G_seed_link_df)

    # Construct graph
    mst_network = cugraph.Graph()
    G_df.rename(columns={"src": "source","dst": "destination"}, inplace=True)
    mst_network.from_cudf_edgelist(G_df, edge_attr="weights", renumber=False)

    sys.stderr.write("Completed calculation of minimum-spanning tree with CU.\n")

    return mst_network


def get_gt_clusters(graph):
    import graph_tool.all as gt

    vertex_labels = list(str(graph.vp["id"][v]) for v in graph.vertices())

    # get a sorted list of component assignments
    component_assignments, component_frequencies = gt.label_components(graph)
    component_frequency_ranks = len(component_frequencies) - rankdata(component_frequencies, method = 'ordinal').astype(int)

    # use components to determine new clusters
    new_clusters = [set() for rank in range(len(component_frequency_ranks))]

    for isolate_index, isolate_name in enumerate(vertex_labels):
        component = component_assignments.a[isolate_index]
        component_rank = component_frequency_ranks[component]
        new_clusters[component_rank].add(isolate_name)

    clustering = {}
    for new_cls_idx, new_cluster in enumerate(new_clusters):
        cls_id = new_cls_idx + 1
        for cluster_member in new_cluster:
            clustering[cluster_member] = cls_id

    return clustering

def get_nx_clusters(graph):
    import networkx as nx

    new_clusters = sorted(nx.connected_components(graph), key=len, reverse=True)
    clustering = {}
    for new_cls_idx, new_cluster in enumerate(new_clusters):
        cls_id = new_cls_idx + 1
        for cluster_member in new_cluster:
            clustering[cluster_member] = cls_id

    return clustering

def draw_gt_mst(mst, out_prefix, isolate_clustering, overwrite):
    """Plot a layout of the minimum spanning tree
    Args:
        mst (graph_tool.Graph)
            A minimum spanning tree
        outPrefix (str)
            Output prefix for save files
        isolate_clustering (dict)
            Dictionary of ID: cluster, used for colouring vertices
        clustering_name (str)
            Name of clustering scheme to be used for colouring
        overwrite (bool)
            Overwrite existing output files
    """
    import graph_tool.all as gt
    
    graph1_file_name = out_prefix + "_mst_stress_plot.png"
    graph2_file_name = out_prefix + "_mst_cluster_plot.png"
    if overwrite or not os.path.isfile(graph1_file_name) or not os.path.isfile(graph2_file_name):
        sys.stderr.write("Drawing MST\n")
        pos = gt.sfdp_layout(mst)
        if overwrite or not os.path.isfile(graph1_file_name):
            deg = mst.degree_property_map("total")
            deg.a = 4 * (np.sqrt(deg.a) * 0.5 + 0.4)
            ebet = gt.betweenness(mst)[1]
            #print(list(ebet))
            ebet.a /= ebet.a.max() / 50.
            eorder = ebet.copy()
            eorder.a *= -1
            gt.graph_draw(mst, pos=pos, vertex_size=gt.prop_to_size(deg, mi=20, ma=50),
                            vertex_fill_color=deg, vorder=deg,
                            edge_color=ebet, eorder=eorder, edge_pen_width=ebet,
                            output=graph1_file_name, output_size=(3000, 3000))
        if overwrite or not os.path.isfile(graph2_file_name):
            cluster_fill = {}
            for cluster in set(isolate_clustering.values()):
                cluster_fill[cluster] = list(np.random.rand(3)) + [0.9]
            plot_color = mst.new_vertex_property('vector<double>')
            mst.vertex_properties['plot_color'] = plot_color
            for v in mst.vertices():
                plot_color[v] = cluster_fill[isolate_clustering[mst.vp.id[v]]]

            gt.graph_draw(mst, pos=pos, vertex_fill_color=mst.vertex_properties['plot_color'],
                    output=graph2_file_name, output_size=(3000, 3000))

def draw_nx_mst(mst, out_prefix, isolate_clustering, overwrite):
    import networkx as nx
    
    import matplotlib.pyplot as plt

    graph1_file_name = out_prefix + "_mst_stress_plot.png"
    graph2_file_name = out_prefix + "_mst_cluster_plot.png"

    if overwrite or not os.path.isfile(graph1_file_name) or not os.path.isfile(graph2_file_name):
        sys.stderr.write("Drawing MST\n")
        pos = nx.spring_layout(mst, seed=13)
        if overwrite or not os.path.isfile(graph1_file_name):
            deg = list(degree for (node, degree) in mst.degree(mst.nodes()))
            deg = 4 * (np.sqrt(deg) * 0.5 + 0.4)
            ebet = np.array(list((nx.edge_betweenness_centrality(mst)).values()))
            ebet /= ebet.max() / 10.
            plt.figure(figsize=(10,7))
            nx.draw_networkx_nodes(mst, pos=pos, node_size=10*deg, node_color=deg)
            nx.draw_networkx_edges(mst, pos=pos, edge_color=ebet, width=ebet)
            plt.axis("off")
            # plt.show()
            plt.savefig(graph1_file_name)
            plt.close()

        if overwrite or not os.path.isfile(graph2_file_name):
            cluster_fill = {}
            for cluster in set(isolate_clustering.values()):
                cluster_fill[cluster] = list(np.random.rand(3)) + [0.9]
            for v in mst.nodes():
                mst.nodes[v]["plot_color"] = cluster_fill[isolate_clustering[v]]

            plt.figure(figsize=(10,7))
            nx.draw_networkx_nodes(mst, pos=pos, node_size=100, node_color=[color for (node, color) in mst.nodes(data="plot_color")])
            nx.draw_networkx_edges(mst, pos=pos)
            plt.axis("off")
            # plt.show()
            plt.savefig(graph2_file_name)
            plt.close()