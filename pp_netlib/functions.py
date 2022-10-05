########################
####   .CONSTRUCT   ####
########################
import scipy
import numpy as np
import pandas as pd
import os

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

def construct_with_cugraph(network_data, vertex_labels, weights = None):
    import cugraph
    import cudf

    graph = cugraph.Graph()
    graph.add_nodes_from(vertex_labels)

    if isinstance(network_data, cudf.DataFrame):
        if weights is not None:
            graph.from_cudf_edgelist(network_data, source="0", destination="1", edge_attr="2", renumber=False)
        else:
            graph.from_cudf_edgelist(network_data, source="0", destination="1", edge_attr=None)

    if isinstance(network_data, pd.DataFrame):
        if weights is not None:
            graph.from_pandas_edgelist(network_data, source="0", destination="1", edge_attr="2", renumber=False)
        else:
            graph.from_pandas_edgelist(network_data, source="0", destination="1", edge_attr=None)

    return graph
########################
####   .SUMMARISE   ####
########################
def get_cugraph_triangles(graph):
    """Counts the number of triangles in a cugraph
    network. Can be removed when the cugraph issue
    https://github.com/rapidsai/cugraph/issues/1043 is fixed.

    ## DEPENDS ON Fns: {none}

    Args:
        graph (cugraph network)
            Network to be analysed
    Returns:
        triangle_count (int)
            Count of triangles in graph
    """
    import cupy as cp
    num_vertices = graph.number_of_vertices()
    edge_df = graph.view_edge_list()
    A = cp.full((num_vertices, num_vertices), 0, dtype = cp.int32)
    A[edge_df.src.values, edge_df.dst.values] = 1
    A = cp.maximum( A, A.transpose() )
    triangle_count = int(cp.around(cp.trace(cp.matmul(A, cp.matmul(A, A)))/6,0))
    return triangle_count

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

    elif backend == "CU":
        import cugraph
        component_assignments = cugraph.components.connectivity.connected_components(graph)
        component_nums = component_assignments['labels'].unique().astype(int)
        components = len(component_nums)
        density = graph.number_of_edges()/(0.5 * graph.number_of_vertices() * graph.number_of_vertices() - 1)
        # consistent with graph-tool for small graphs - triangle counts differ for large graphs
        # could reflect issue https://github.com/rapidsai/cugraph/issues/1043
        # this command can be restored once the above issue is fixed - scheduled for cugraph 0.20
        # triangle_count = cugraph.community.triangle_count.triangles(G)/3
        triangle_count = 3*get_cugraph_triangles(graph)
        degree_df = graph.in_degree()
        # consistent with graph-tool
        triad_count = 0.5 * sum([d * (d - 1) for d in degree_df[degree_df['degree'] > 1]['degree'].to_pandas()])
        if triad_count > 0:
            transitivity = triangle_count/triad_count
        else:
            transitivity = 0.0

        component_frequencies = component_assignments['labels'].value_counts(sort = True, ascending = False)
        for component in component_nums.to_pandas():
            size = component_frequencies[component_frequencies.index == component].iloc[0].astype(int)
            if size > 3:
                component_vertices = component_assignments['vertex'][component_assignments['labels']==component]
                subgraph = cugraph.subgraph(graph, component_vertices)
                if len(component_vertices) >= 100:
                    component_betweenness = cugraph.betweenness_centrality(subgraph, k = 100, normalized = True)
                else:
                    component_betweenness = cugraph.betweenness_centrality(subgraph, normalized = True)
                betweenness.append(component_betweenness['betweenness_centrality'].max())
                sizes.append(size)

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
