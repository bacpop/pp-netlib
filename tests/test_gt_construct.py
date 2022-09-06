import numpy as np
import pandas as pd
import graph_tool.all as gt
from scipy.sparse import coo_matrix

from pp_netlib.network import Network

def __init__():
    """Initialise test data, create (un)weighted graph-tool graphs, and (un)weighted Network objects.

    Returns:
        graph (gt.Graph): Unweighted graph-tool graph
        weighted_graph (gt.Graph): Weighted graph-tool graph
        sample_df (pd.DataFrame): Edge data as dataframe, input for fn `gt_test_from_df`
        sample_edge_list (list): Edge data as list of tuples, input for fn `gt_test_from_list`
        sample_coo (scipy.sparse.coo_matrix): Edge data as sparse matrix in coordinate format, input for fn `gt_test_from_sparse_matrix`
        testing_network (network.Network): Initialised Network object, will be used to build unweighted graphs in tests
        testing_weighted_network (network.Network): Initialised Network object, will be used to build weighted graphs in tests
        weights (list): List of weights to be used for building weighted graphs from Network objects in tests
    """

    ## setting up edge list and weighted edge list
    vertex_labels = ["s1", "s2", "s3", "s4", "s5"]
    sample_edge_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    sample_weighted_edge_list = [(0, 1, 0.1), (1, 2, 0.1), (2, 3, 0.5), (3, 4, 0.7), (4, 0, 0.2)]

    ## setting up test dataframe
    sample_df_data = {0: [0, 1, 2, 3, 4], 1: [1, 2, 3, 4, 0]}
    sample_df = pd.DataFrame(sample_df_data)

    weights = [0.1, 0.1, 0.5, 0.7, 0.2]

    ## setting up test coo_matrix
    sample_row = np.array([0, 1, 2, 3, 4])
    sample_column = np.array([1, 2, 3, 4, 0])
    sample_weights = np.array([0.1, 0.1, 0.5, 0.7, 0.2])

    sample_coo = coo_matrix((sample_weights, (sample_row, sample_column)), shape=(5,5))

    ## setting up "reference" graph
    graph = gt.Graph(directed = False)
    graph.add_vertex(5)
    graph.add_edge_list(sample_edge_list) ## ordinary gt graph

    v_name_prop = graph.new_vp("string")
    graph.vertex_properties["id"] = v_name_prop
    for i in range(len([v for v in graph.vertices()])):
        v_name_prop[graph.vertex(i)] = vertex_labels[i]

    ## setting up "reference" weighted graph
    weighted_graph = gt.Graph(directed = False)
    weighted_graph.add_vertex(5)
    eweight = weighted_graph.new_ep("float")
    weighted_graph.add_edge_list(sample_weighted_edge_list, eprops = [eweight]) ## weighted gt graph
    weighted_graph.edge_properties["weight"] = eweight

    v_weighted_name_prop = weighted_graph.new_vp("string")
    weighted_graph.vertex_properties["id"] = v_weighted_name_prop
    for i in range(len([v for v in weighted_graph.vertices()])):
        v_weighted_name_prop[weighted_graph.vertex(i)] = vertex_labels[i]


    ## intialising network
    testing_network = Network(ref_list=["s1", "s2", "s3", "s4", "s5"], query_list=["s1", "s2", "s3", "s4", "s5"], outdir="tests/")
    testing_weighted_network = Network(ref_list=["s1", "s2", "s3", "s4", "s5"], query_list=["s1", "s2", "s3", "s4", "s5"], outdir="tests/")

    return graph, weighted_graph, sample_df, sample_edge_list, sample_coo, testing_network, testing_weighted_network, weights

def gt_test_from_df(data_df, graph, weighted_graph, network, weighted_network, weights):
    """Test construct method from Network class with dataframe input and graph-tool

    Args:
        data_df (pd.DataFrame): Dataframe of edge lists generated in __init__
        graph (gt.Graph): "Reference" graph-tool graph object pre-generated in __init__
        weighted_graph (gt.Graph): "Reference" *weighted* graph-tool graph object pre-generated in __init__
        network (network.Network): Initialised Network graph object to be tested
        weighted_network (network.Network): Initialised *weighted* Network graph object to be tested
        weights (list): List of weights to use to construct weighted graph

    Raises:
        ae: AssertionError if any tests fail.
    """

    network.construct(network_data = data_df)
    weighted_network.construct(network_data = data_df, weights = weights)

    edges = [e for e in graph.edges()]
    test_edges = [te for te in network.graph.edges()]

    weighted_edges = [we for we in weighted_graph.edges()]
    test_weighted_edges = [twe for twe in weighted_network.graph.edges()]

    ref_weights = [weighted_graph.ep["weight"][e] for e in weighted_graph.edges()]
    test_weights = [weighted_network.graph.ep["weight"][e] for e in weighted_network.graph.edges()]

    ref_ids = [graph.vp["id"][v] for v in graph.vertices()]
    test_ids = [network.graph.vp["id"][v] for v in network.graph.vertices()]

    ref_weighted_ids = [weighted_graph.vp["id"][v] for v in weighted_graph.vertices()]
    test_weighted_ids = [weighted_network.graph.vp["id"][v] for v in weighted_network.graph.vertices()]

    try:
        assert edges == test_edges
        assert weighted_edges == test_weighted_edges
        assert ref_weights == test_weights
        assert len([graph.get_vertices()]) == len([network.graph.get_vertices()])
        assert len([weighted_graph.get_vertices()]) == len([weighted_network.graph.get_vertices()])
        assert ref_ids == test_ids
        assert ref_weighted_ids == test_weighted_ids
        print("gt_test_from_df passed.")
    except AssertionError as ae:
        print("gt_test_from_df failed.")
        raise ae


def gt_test_from_list(data_list, graph, weighted_graph, network, weighted_network, weights):
    """Test construct method from Network class with list input and graph-tool

    Args:
        data_list (list): Edge list; list of tuples describing edge source and destination nodes; generated in __init__
        graph (gt.Graph): "Reference" graph-tool graph object pre-generated in __init__
        weighted_graph (gt.Graph): "Reference" *weighted* graph-tool graph object pre-generated in __init__
        network (network.Network): Initialised Network graph object to be tested
        weighted_network (network.Network): Initialised *weighted* Network graph object to be tested
        weights (list): List of weights to use to construct weighted graph

    Raises:
        ae: AssertionError if any tests fail.
    """

    network.construct(network_data = data_list)
    weighted_network.construct(network_data = data_list, weights = weights)

    edges = [e for e in graph.edges()]
    test_edges = [te for te in network.graph.edges()]

    weighted_edges = [we for we in weighted_graph.edges()]
    test_weighted_edges = [twe for twe in weighted_network.graph.edges()]

    ref_weights = [weighted_graph.ep["weight"][e] for e in weighted_graph.edges()]
    test_weights = [weighted_network.graph.ep["weight"][e] for e in weighted_network.graph.edges()]

    ref_ids = [graph.vp["id"][v] for v in graph.vertices()]
    test_ids = [network.graph.vp["id"][v] for v in network.graph.vertices()]

    ref_weighted_ids = [weighted_graph.vp["id"][v] for v in weighted_graph.vertices()]
    test_weighted_ids = [weighted_network.graph.vp["id"][v] for v in weighted_network.graph.vertices()]

    try:
        assert edges == test_edges
        assert weighted_edges == test_weighted_edges
        assert ref_weights == test_weights
        assert len([v for v in graph.vertices()]) == len([tv for tv in network.graph.vertices()])
        assert len([v for v in weighted_graph.vertices()]) == len([tv for tv in weighted_network.graph.vertices()])
        assert ref_ids == test_ids
        assert ref_weighted_ids == test_weighted_ids
        print("gt_test_from_list passed.")
    except AssertionError as ae:
        print("gt_test_from_list failed.")
        raise ae


def gt_test_from_sparse_matrix(data_matrix, weighted_graph, weighted_network):
    """Test construct method from Network class with dataframe input and graph-tool

    Args:
        data_matrix (scipy.sparse.coo_matrix): Sparse matrix of edge lists generated in __init__
        weighted_graph (gt.Graph): "Reference" *weighted* graph-tool graph object pre-generated in __init__
        weighted_network (network.Network): Initialised *weighted* Network graph object to be tested
        weights (list): List of weights to use to construct weighted graph

    Raises:
        ae: AssertionError if any tests fail.
    """

    weighted_network.construct(network_data = data_matrix)

    weighted_edges = [we for we in weighted_graph.edges()]
    test_weighted_edges = [twe for twe in weighted_network.graph.edges()]

    ref_weights = [weighted_graph.ep["weight"][e] for e in weighted_graph.edges()]
    test_weights = [weighted_network.graph.ep["weight"][e] for e in weighted_network.graph.edges()]

    ref_weighted_ids = [weighted_graph.vp["id"][v] for v in weighted_graph.vertices()]
    test_weighted_ids = [weighted_network.graph.vp["id"][v] for v in weighted_network.graph.vertices()]

    try:
        assert weighted_edges == test_weighted_edges
        assert ref_weights == test_weights
        assert len([v for v in weighted_graph.vertices()]) == len([tv for tv in weighted_network.graph.vertices()])
        assert ref_weighted_ids == test_weighted_ids
        print("gt_test_from_sparse_matrix passed.")
    except AssertionError as ae:
        print("gt_test_from_sparse_matrix failed.")
        raise ae


if __name__ == "__main__":
    graph, weighted_graph, sample_df, sample_edge_list, sample_coo, testing_network, testing_weighted_network, weights = __init__()
    gt_test_from_df(sample_df, graph, weighted_graph, testing_network, testing_weighted_network, weights)
    gt_test_from_list(sample_edge_list, graph, weighted_graph, testing_network, testing_weighted_network, weights)
    gt_test_from_sparse_matrix(sample_coo, weighted_graph, testing_weighted_network)




