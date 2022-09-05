import numpy as np
import pandas as pd
import graph_tool.all as gt
import networkx as nx

from pp_netlib.network import Network

def __init__():
    sample_edge_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 5), (2, 5), (5, 0), (4, 6), (1, 6)]
    weights = [0.1, 0.1, 0.5, 0.7, 0.2, 0.3, 0.6, 0.8, 0.4, 0.3]

    gt_graph = Network(ref_list=["s1", "s2", "s3", "s4", "s5", "s6", "s7"], query_list=["s1", "s2", "s3", "s4", "s5", "s6", "s7"], backend="GT", outdir="tests/")
    nx_graph = Network(ref_list=["s1", "s2", "s3", "s4", "s5", "s6", "s7"], query_list=["s1", "s2", "s3", "s4", "s5", "s6", "s7"], backend="NX", outdir="tests/")

    gt_graph.construct(sample_edge_list, weights)
    nx_graph.construct(sample_edge_list, weights)

    return gt_graph, nx_graph

def test_summarise(graph, backend):
    try:
        assert backend == graph.backend
        print("Graph backend looks okay.")
    except AssertionError as ae:
        print("Something went wrong.")
        raise ae

    graph.get_summary()

if __name__ == "__main__":
    gt_graph, nx_graph = __init__()
    test_summarise(gt_graph, "GT")
    test_summarise(nx_graph, "NX")