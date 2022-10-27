import pandas as pd

from pp_netlib.network import Network

def __init__():
    sample_edge_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 5), (2, 5), (5, 0), (4, 6), (1, 6)]
    weights = [0.1, 0.1, 0.5, 0.7, 0.2, 0.3, 0.6, 0.8, 0.4, 0.3]
    sample_ids = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]

    test_edge_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    test_weights = [0.1, 0.1, 0.5, 0.7, 0.2]

    new_data = pd.DataFrame()
    new_data["source"] = [1, 2, 5, 4, 1]
    new_data["target"] = [5, 5, 0, 6, 6]
    new_data["weights"] = [0.3, 0.6, 0.8, 0.4, 0.3]

    gt_graph = Network(ref_list=["s1", "s2", "s3", "s4", "s5", "s6", "s7"], backend="GT", outdir="tests/")
    nx_graph = Network(ref_list=["s1", "s2", "s3", "s4", "s5", "s6", "s7"], backend="NX", outdir="tests/")

    gt_graph.construct(sample_edge_list, weights)
    nx_graph.construct(sample_edge_list, weights)

    gt_test_graph = Network(ref_list=["s1", "s2", "s3", "s4", "s5"], backend="GT", outdir="tests/")
    nx_test_graph = Network(ref_list=["s1", "s2", "s3", "s4", "s5"], backend="NX", outdir="tests/")

    gt_test_graph.construct(test_edge_list, test_weights)
    nx_test_graph.construct(test_edge_list, test_weights)

    return gt_graph, nx_graph, gt_test_graph, nx_test_graph, new_data

def test_add(ref_graph, test_graph, new_data):

    test_graph.add_to_network(new_data, ["s6", "s7"])
    if test_graph.backend == "GT":
        try:
            assert list(str(e) for e in test_graph.graph.edges()) == list(str(e) for e in ref_graph.graph.edges())
            print("GT Edge data match.")
            assert sorted(list(test_graph.graph.ep["weight"][e] for e in test_graph.graph.edges())) == sorted(list(ref_graph.graph.ep["weight"][e] for e in ref_graph.graph.edges()))
            print("GT Weight data associated with edges match.")
            assert list(test_graph.graph.vp["id"][v] for v in test_graph.graph.vertices()) == list(ref_graph.graph.vp["id"][v] for v in ref_graph.graph.vertices())
            print("GT Vertex labels match.\n\n")
        except AssertionError as ae:
            print("Something went wrong when adding with GT\n\n")
            raise ae
    elif test_graph.backend == "NX":
        try:
            assert list(test_graph.graph.edges.data()) == list(ref_graph.graph.edges.data())
            print("NX Edge data match.")
            assert list(test_graph.graph.nodes.data("id")) == list(ref_graph.graph.nodes.data("id"))
            print("NX Vertex data match.\n\n")
        except AssertionError as ae:
            print(f"Something went wrong when adding with NX\n\n")
            raise ae
        
if __name__ == "__main__":
    gt_graph, nx_graph, gt_test_graph, nx_test_graph, new_data = __init__()
    print("Running tests for GT add_to_network...\n")
    test_add(gt_graph, gt_test_graph, new_data)
    print("Running tests for NX add_to_network...\n")
    test_add(nx_graph, nx_test_graph, new_data)
    
