import os
import graph_tool
import numpy as np
import pandas as pd
import argparse
import graph_tool.all as gt
import networkx as nx
from scipy.sparse import coo_matrix

from pp_netlib.network import Network

def get_args():
    parser = argparse.ArgumentParser(description="Testing backend selector.")
    parser.add_argument("--set_with_python", action="store_true", help="Use flag to set backend with od.environ.")
    args = parser.parse_args()
    set_with_py = args.set_with_python
    return set_with_py

def __init__():
    vertex_labels = ["s1", "s2", "s3", "s4", "s5"]
    sample_edge_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    weights = [0.1, 0.1, 0.5, 0.7, 0.2]
    sample_weighted_edge_list = [(0, 1, 0.1), (1, 2, 0.1), (2, 3, 0.5), (3, 4, 0.7), (4, 0, 0.2)]

    ## setting up "reference" graph-tool weighted graph
    gt_weighted_graph = gt.Graph(directed = False)
    gt_weighted_graph.add_vertex(5)
    eweight = gt_weighted_graph.new_ep("float")
    gt_weighted_graph.add_edge_list(sample_weighted_edge_list, eprops = [eweight]) ## weighted gt graph
    gt_weighted_graph.edge_properties["weight"] = eweight

    v_weighted_name_prop = gt_weighted_graph.new_vp("string")
    gt_weighted_graph.vertex_properties["id"] = v_weighted_name_prop
    for i in range(len([v for v in gt_weighted_graph.vertices()])):
        v_weighted_name_prop[gt_weighted_graph.vertex(i)] = vertex_labels[i]

    ## setting up "reference" networkx weighted graph
    nx_weighted_graph = nx.Graph()
    nx_weighted_graph.add_nodes_from(range(len(vertex_labels)))
    ## add node labels
    for i in range(len(vertex_labels)):
        nx_weighted_graph.nodes[i]["id"] = vertex_labels[i]

    src, dest = zip(*sample_edge_list)
    for i in range(len(vertex_labels)):
        nx_weighted_graph.add_edge(src[i], dest[i], weight=weights[i])

    return gt_weighted_graph, nx_weighted_graph, sample_edge_list, weights

def set_with_python_gt(premade_gt_graph, sample_edge_list, weights):
    os.environ["GRAPH_BACKEND"] = "GT"
    print(os.getenv("GRAPH_BACKEND"))
    testing_weighted_network = Network(ref_list=["s1", "s2", "s3", "s4", "s5"], query_list=["s1", "s2", "s3", "s4", "s5"], outdir="tests/")
    testing_weighted_network.construct(sample_edge_list, weights)
    gt_edges = [e for e in premade_gt_graph.edges()]
    gt_test_edges = [e for e in testing_weighted_network.graph.edges()]
    try:
        assert gt_edges == gt_test_edges
        print("Setting GT backend with os.environ works as expected.")
    except AssertionError as ae:
        print("Setting GT backend with os.environ failed to work as expected.")
        raise ae

def set_with_python_nx(premade_nx_graph, sample_edge_list, weights):
    os.environ["GRAPH_BACKEND"] = "NX"
    print(os.getenv("GRAPH_BACKEND"))
    testing_weighted_network = Network(ref_list=["s1", "s2", "s3", "s4", "s5"], query_list=["s1", "s2", "s3", "s4", "s5"], outdir="tests/")
    testing_weighted_network.construct(sample_edge_list, weights)
    nx_edges = list(premade_nx_graph.edges())
    nx_test_edges = list(testing_weighted_network.graph.edges())
    try:
        assert nx_edges == nx_test_edges
        print("Setting NX backend with os.environ works as expected.")
    except AssertionError as ae:
        print("Setting NX backend with os.environ failed to work as expected.")
        raise ae

if __name__ == "__main__":
    set_with_py = get_args()
    gt_weighted_graph, nx_weighted_graph, sample_edge_list, weights = __init__()
    if set_with_py:
        set_with_python_gt(gt_weighted_graph, sample_edge_list, weights)
        set_with_python_nx(nx_weighted_graph, sample_edge_list, weights)
    else:
        print(os.getenv("GRAPH_BACKEND"))
        testing_weighted_network = Network(ref_list=["s1", "s2", "s3", "s4", "s5"], query_list=["s1", "s2", "s3", "s4", "s5"], outdir="tests/")
        testing_weighted_network.construct(sample_edge_list, weights)
        try:
            assert testing_weighted_network.backend == os.getenv("GRAPH_BACKEND")
            print("Network backend selected correctly.")
        except AssertionError as ae:
            print("Network backend not selected correctly.")
            raise ae

        if os.getenv("GRAPH_BACKEND") == "NX":
            try:
                assert str(type(testing_weighted_network.graph)) == "<class 'networkx.classes.graph.Graph'>"
                print("Network successfully selects NX.")
            except AssertionError as ae:
                print("Network fails to select NX.")
                raise ae

        elif os.getenv("GRAPH_BACKEND") == "GT":
            try:
                assert type(testing_weighted_network.graph) == graph_tool.Graph
                print("Network successfully selects GT.")
            except AssertionError as ae:
                print("Network fails to select GT.")
                raise ae