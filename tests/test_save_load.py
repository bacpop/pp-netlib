import os
from pp_netlib.network import Network

def __init__():
    sample_edge_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 5), (2, 5), (5, 0), (4, 6), (1, 6)]
    weights = [0.1, 0.1, 0.5, 0.7, 0.2, 0.3, 0.6, 0.8, 0.4, 0.3]

    gt_graph = Network(ref_list=["s1", "s2", "s3", "s4", "s5", "s6", "s7"], backend="GT", outdir="tests/")
    nx_graph = Network(ref_list=["s1", "s2", "s3", "s4", "s5", "s6", "s7"], backend="NX", outdir="tests/")

    gt_graph.construct(sample_edge_list, weights)
    nx_graph.construct(sample_edge_list, weights)

    gt_edges = [str(e) for e in gt_graph.graph.edges()]
    nx_edges = []
    for (s,d) in nx_graph.graph.edges():
        nx_edges.append((str(s), str(d)))

    gt_graph.save("test_gt_graph", ".gt")
    nx_graph.save("test_nx_graph", ".graphml")

    return gt_graph, nx_graph, gt_edges, nx_edges

def check_if_saved(file_path):
    try:
        assert os.path.isfile(file_path)
        print(f"{file_path} exists!")
    except AssertionError as ae:
        print(f"{file_path} does not seem to exist!")
        raise ae

def check_loading(gt_edges, nx_edges):
    loaded_gt_graph = Network(ref_list=[], backend="GT", outdir="tests/")
    loaded_gt_graph.load_network("tests/test_gt_graph.gt")
    loaded_gt_edges = [str(e) for e in loaded_gt_graph.graph.edges()]

    loaded_nx_graph = Network(ref_list=[], backend="NX", outdir="tests/")
    loaded_nx_graph.load_network("tests/test_nx_graph.graphml")
    loaded_nx_edges = list(loaded_nx_graph.graph.edges())

    try:
        assert loaded_gt_edges == gt_edges
        print("Graph-tools successfully loads file.")
        assert loaded_nx_edges == nx_edges
        print("Networkx successfully loads file.")
    except AssertionError as ae:
        print("Something went wrong with loading the network file in.")
        raise ae

if __name__ == "__main__":
    gt_graph, nx_graph, gt_edges, nx_edges = __init__()
    check_if_saved("tests/test_gt_graph.gt")
    check_if_saved("tests/test_nx_graph.graphml")
    check_loading(gt_edges, nx_edges)
    os.remove("tests/test_gt_graph.gt")
    os.remove("tests/test_nx_graph.graphml")

    print("tests/test_gt_graph.gt, tests/test_nx_graph.graphml removed, cleanup done.")

    assert os.path.isfile("tests/test_gt_graph.gt") is False
    assert os.path.isfile("tests/test_nx_graph.graphml") is False