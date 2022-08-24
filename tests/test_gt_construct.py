import numpy as np
import pandas as pd
import graph_tool.all as gt
from scipy.sparse import coo_matrix

from pp_netlib.network import Network

## setting up edge list and weighted edge list
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

## setting up "reference" weighted graph
weighted_graph = gt.Graph(directed = False)
weighted_graph.add_vertex(5)
eweight = weighted_graph.new_ep("float")
weighted_graph.add_edge_list(sample_weighted_edge_list, eprops = [eweight]) ## weighted gt graph
weighted_graph.edge_properties["weight"] = eweight

## setting up "previous" graph
prev_edge_list = [(1, 5), (2, 6), (3, 8), (5, 7), (6, 8), (4, 9), (8, 9)]
prev_weighted_edge_list = []

prev_graph = gt.Graph(directed = False)
prev_graph.add_vertex(5)
prev_graph.add_edge_list(prev_edge_list)

## intialising network
testing_network = Network(ref_list="tests/ref_list.txt", query_list="tests/ref_list.txt", outdir="tests/")
testing_weighted_network = Network(ref_list="tests/ref_list.txt", query_list="tests/ref_list.txt", outdir="tests/")

def gt_test_from_df(graph, data_df, network, weights):

    network.construct(network_data = data_df)
    testing_weighted_network.construct(network_data = data_df, weights = weights)

    edges = [e for e in graph.edges()]
    test_edges = [te for te in network.graph.edges()]

    weighted_edges = [we for we in weighted_graph.edges()]
    test_weighted_edges = [twe for twe in testing_weighted_network.graph.edges()]

    ref_weights = [weighted_graph.ep["weight"][e] for e in weighted_graph.edges()]
    test_weights = [testing_weighted_network.graph.ep["weight"][e] for e in testing_weighted_network.graph.edges()]

    # print([(edge.source(), edge.target()) for edge in weighted_graph.edges()])
    # print(ref_weights)

    try:
        assert edges == test_edges
        assert weighted_edges == test_weighted_edges
        assert ref_weights == test_weights
        print("gt_test_from_df passed.")
    except AssertionError as ae:
        print("gt_test_from_df failed.")
        raise ae


def gt_test_from_list(graph, data_list, network, weights):

    network.construct(network_data = data_list)
    testing_weighted_network.construct(network_data = data_list, weights = weights)

    edges = [e for e in graph.edges()]
    test_edges = [te for te in network.graph.edges()]

    weighted_edges = [we for we in weighted_graph.edges()]
    test_weighted_edges = [twe for twe in testing_weighted_network.graph.edges()]

    ref_weights = [weighted_graph.ep["weight"][e] for e in weighted_graph.edges()]
    test_weights = [testing_weighted_network.graph.ep["weight"][e] for e in testing_weighted_network.graph.edges()]

    try:
        assert edges == test_edges
        assert weighted_edges == test_weighted_edges
        assert ref_weights == test_weights
        print("gt_test_from_list passed.")
    except AssertionError as ae:
        print("gt_test_from_list failed.")


def gt_test_from_sparse_matrix(weighted_graph, data_matrix, weighted_network):

    weighted_network.construct(network_data = data_matrix)

    weighted_edges = [we for we in weighted_graph.edges()]
    test_weighted_edges = [twe for twe in weighted_network.graph.edges()]

    ref_weights = [weighted_graph.ep["weight"][e] for e in weighted_graph.edges()]
    test_weights = [weighted_network.graph.ep["weight"][e] for e in weighted_network.graph.edges()]

    try:
        assert weighted_edges == test_weighted_edges
        assert ref_weights == test_weights
        print("gt_test_from_sparse_matrix passed.")
    except:
        print("gt_test_from_sparse_matrix failed.")
    

if __name__ == "__main__":
    gt_test_from_df(graph, sample_df, testing_network, weights)
    gt_test_from_list(graph, sample_edge_list, testing_network, weights)
    gt_test_from_sparse_matrix(weighted_graph, sample_coo, testing_weighted_network)




