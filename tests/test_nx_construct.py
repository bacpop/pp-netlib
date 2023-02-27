import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import coo_matrix

from pp_netlib.network import Network

def __init__():
    vertex_labels = ["s1", "s2", "s3", "s4", "s5"]
    weights = [0.1, 0.1, 0.5, 0.7, 0.2]

    ## setting up edge list
    sample_edge_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]

    ## setting up test dataframe
    sample_df_data = {"source": [0, 1, 2, 3, 4], "target": [1, 2, 3, 4, 0]}
    sample_df = pd.DataFrame(sample_df_data)
    
    ## setting up test coo_matrix
    sample_row = np.array([0, 1, 2, 3, 4])
    sample_column = np.array([1, 2, 3, 4, 0])
    sample_weights = np.array([0.1, 0.1, 0.5, 0.7, 0.2])

    sample_coo = coo_matrix((sample_weights, (sample_row, sample_column)), shape=(5,5))

    ## setting up "reference" graph
    graph = nx.Graph()
    graph.add_nodes_from(range(len(vertex_labels)))
    ## add node labels
    for i in range(len(vertex_labels)):
        graph.nodes[i]["id"] = vertex_labels[i]

    graph.add_edges_from(sample_edge_list)

    ## setting up "reference" *weighted* graph
    weighted_graph = nx.Graph()
    weighted_graph.add_nodes_from(range(len(vertex_labels)))
    ## add node labels
    for i in range(len(vertex_labels)):
        weighted_graph.nodes[i]["id"] = vertex_labels[i]

    src, dest = zip(*sample_edge_list)
    for i in range(len(vertex_labels)):
        weighted_graph.add_edge(src[i], dest[i], weight=weights[i])

    ## initiaising network
    testing_network = Network(ref_list=["s1", "s2", "s3", "s4", "s5"], backend="NX", outdir="tests/")
    testing_weighted_network = Network(ref_list=["s1", "s2", "s3", "s4", "s5"], backend="NX", outdir="tests/")

    return graph, weighted_graph, sample_edge_list, sample_df, sample_coo, weights, testing_network, testing_weighted_network

def nx_test_from_df(data_df, graph, weighted_graph, network, weighted_network, weights):

    network.construct(network_data = data_df)
    weighted_network.construct(network_data = data_df, weights = weights)

    edges = list(graph.edges())
    test_edges = list(network.graph.edges())

    weighted_edges = list(weighted_graph.edges.data())
    weighted_test_edges = list(weighted_network.graph.edges.data())

    nodes = list(graph.nodes.data("id"))
    test_nodes = list(network.graph.nodes.data("id"))

    w_nodes = list(weighted_graph.nodes.data("id"))
    w_test_nodes = list(weighted_network.graph.nodes.data("id"))

    try:
        assert edges == test_edges
        assert weighted_edges == weighted_test_edges
        assert nodes == test_nodes
        assert w_nodes == w_test_nodes
        print("nx_test_from_df passed.")
    except AssertionError as ae:
        print("nx_test_from_df failed.")
        raise ae

def nx_test_from_list(data_list, graph, weighted_graph, network, weighted_network, weights):

    network.construct(network_data = data_list)
    weighted_network.construct(network_data = data_list, weights = weights)

    edges = list(graph.edges())
    test_edges = list(network.graph.edges())

    weighted_edges = list(weighted_graph.edges.data())
    weighted_test_edges = list(weighted_network.graph.edges.data())

    nodes = list(graph.nodes.data("id"))
    test_nodes = list(network.graph.nodes.data("id"))

    w_nodes = list(weighted_graph.nodes.data("id"))
    w_test_nodes = list(weighted_network.graph.nodes.data("id"))

    try:
        assert edges == test_edges
        assert weighted_edges == weighted_test_edges
        assert nodes == test_nodes
        assert w_nodes == w_test_nodes
        print("nx_test_from_list passed.")
    except AssertionError as ae:
        print("nx_test_from_list failed.")
        raise ae

def nx_test_from_coo(data_coo, weighted_graph, weighted_network):

    weighted_network.construct(network_data = data_coo, weights = weights)

    weighted_edges = list(weighted_graph.edges.data())
    weighted_test_edges = list(weighted_network.graph.edges.data())

    w_nodes = list(weighted_graph.nodes.data("id"))
    w_test_nodes = list(weighted_network.graph.nodes.data("id"))

    try:
        assert weighted_edges == weighted_test_edges
        assert w_nodes == w_test_nodes
        print("nx_test_from_coo passed.")
    except AssertionError as ae:
        print("nx_test_from_coo failed.")
        raise ae

if __name__ == "__main__":
    graph, weighted_graph, sample_edge_list, sample_df, sample_coo, weights, testing_network, testing_weighted_network = __init__()
    nx_test_from_df(sample_df, graph, weighted_graph, testing_network, testing_weighted_network, weights)
    nx_test_from_list(sample_edge_list, graph, weighted_graph, testing_network, testing_weighted_network, weights)
    nx_test_from_coo(sample_coo, weighted_graph, testing_weighted_network)