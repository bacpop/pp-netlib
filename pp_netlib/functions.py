########################
####   .CONSTRUCT   ####
########################
import scipy
import numpy as np
import pandas as pd
import graph_tool.all as gt
import networkx as nx

def construct_with_graphtool(network_data, vertex_labels, use_gpu, weights = None):
    """Construct a graph with graph-tool

    Args:
        network_data (dataframe OR edge list OR sparse coordinate matrix): Data containing record of edges in the graph.
        weights (list, optional): List of weights associated with edges in network_data.
                                      Weights must be in the same order as edges in network_data. Defaults to None.

    Returns:
        graph (gt.Graph): Graph-tool graph object populated with network data
    """
    graph = gt.Graph(directed = False) ## initialise graph_tool graph object

    ########################
    ####    DF INPUT    ####
    ########################
    if isinstance(network_data, pd.DataFrame):
        # if use_gpu:
        #     network_data = cudf.from_pandas(network_data) ## convert to cudf if use_gpu
        ## add column names
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
        if not use_gpu:
            graph_data_df = pd.DataFrame()
        # else:
        #     graph_data_df = cudf.DataFrame()
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

def construct_with_networkx(network_data, vertex_labels, use_gpu, weights = None):
    ## initialise nx graph and add nodes
    graph = nx.Graph()
    graph.add_nodes_from(range(len(vertex_labels)))

    ## add node labels
    for i in range(len(vertex_labels)):
        graph.nodes[i]["id"] = vertex_labels[i]

    ########################
    ####    DF INPUT    ####
    ########################
    if isinstance(network_data, pd.DataFrame):
        network_data.columns = ["source", "destination"]
        if weights is not None:
            network_data["weights"] = weights
            for i in range(len(network_data)):
                graph.add_edge(network_data["source"][i], network_data["destination"][i], weight=network_data["weights"][i])
        else:
            graph.add_edges_from(network_data.values)

    ##########################
    #### SPARSE MAT INPUT ####
    ##########################
    elif isinstance(network_data, scipy.sparse.coo_matrix):
        for i in range((network_data.shape[0])):
            graph.add_edge(network_data.row[i], network_data.col[i], weight=network_data.data[i])

    ########################
    ####   LIST INPUT   ####
    ########################
    elif isinstance(network_data, list):
        if weights is not None:
            src, dest = zip(*network_data)
            for i in range(len(network_data)):
                graph.add_edge(src[i], dest[i], weight=weights[i])
        else:
            graph.add_edges_from(network_data)

    return graph

########################
####   .SUMMARISE   ####
########################
def summarise(graph, backend):

    if backend == "GT":
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
        components = nx.number_connected_components(graph)
        density = nx.density(graph)
        transitivity = nx.transitivity(graph)

        betweenness = []
        sizes = []
        subgraphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
        for subgraph in subgraphs:
            betweenness.append(max((nx.betweenness_centrality(subgraph)).values()))
            sizes.append(len(subgraph))

        print("sizes=", sizes)
        if len(betweenness) > 1:
            mean_bt = np.mean(betweenness)
            weighted_mean_bt = np.average(betweenness, weights=sizes)
        elif len(betweenness) == 1:
            mean_bt = betweenness[0]
            weighted_mean_bt = betweenness[0]

    metrics = [components, density, transitivity, mean_bt, weighted_mean_bt]
    base_score = transitivity * (1 - density)
    scores = [base_score, base_score * (1 - metrics[3]), base_score * (1 - metrics[4])]

    summary_contents = ("Network summary:\n" + "\n".join(["\tComponents\t\t\t\t" + str(metrics[0]),
                                                    "\tDensity\t\t\t\t\t" + "{:.4f}".format(metrics[1]),
                                                    "\tTransitivity\t\t\t\t" + "{:.4f}".format(metrics[2]),
                                                    "\tMean betweenness\t\t\t" + "{:.4f}".format(metrics[3]),
                                                    "\tWeighted-mean betweenness\t\t" + "{:.4f}".format(metrics[4]),
                                                    "\tScore\t\t\t\t\t" + "{:.4f}".format(scores[0]),
                                                    "\tScore (w/ betweenness)\t\t\t" + "{:.4f}".format(scores[1]),
                                                    "\tScore (w/ weighted-betweenness)\t\t" + "{:.4f}".format(scores[2])])
                                                    + "\n")

    return summary_contents
