import os
import graph_tool.all as gt
import networkx as nx

from pp_netlib.network import Network

sample_file = "tests/labelled_random.graphml"

def test_gt_mst(sample_file):
    sample_gt_graph = Network([], outdir="tests/", backend = "GT")
    sample_gt_graph.load_network(sample_file)

    sample_gt_graph.visualize("mst", "test_gt")

    num_vertices = sample_gt_graph.mst_network.num_vertices()
    num_edges = sample_gt_graph.mst_network.num_edges()

    print(f"sample_gt_graph mst has {num_vertices} nodes and {num_edges} edges.\n")

    expected_outfiles = ["tests/test_gt_mst_cluster_plot.png", "tests/test_gt_mst_network_data.graphml", "tests/test_gt_mst_stress_plot.png"]
    for outfile in expected_outfiles:
        try:
            assert os.path.isfile(outfile)
        except AssertionError as ae:
            print(f"{outfile} not found when making mst with GT.\n")
            raise ae

    for outfile in expected_outfiles:
        os.remove(outfile) 
    print("Cleanup after GT mst done.\n")

    return num_vertices, num_edges

def test_nx_mst(sample_file):
    sample_nx_graph = Network([], outdir="tests/", backend = "NX")
    sample_nx_graph.load_network(sample_file)
    sample_nx_graph.visualize("mst", "test_nx")
    num_vertices = sample_nx_graph.mst_network.number_of_nodes()
    num_edges = sample_nx_graph.mst_network.number_of_edges()

    print(f"sample_nx_graph mst has {num_vertices} nodes and {num_edges} edges.")

    expected_outfiles = ["tests/test_nx_mst_cluster_plot.png", "tests/test_nx_mst_network_data.graphml", "tests/test_nx_mst_stress_plot.png"]
    for outfile in expected_outfiles:
        try:
            assert os.path.isfile(outfile)
        except AssertionError as ae:
            print(f"{outfile} not found when making mst with NX.\n")
            raise ae

    for outfile in expected_outfiles:
        os.remove(outfile) 
    print("Cleanup after NX mst done.\n")

    return num_vertices, num_edges

if __name__ == "__main__":
    gt_num_nodes, gt_num_edges = test_gt_mst(sample_file)
    nx_num_nodes, nx_num_edges = test_nx_mst(sample_file)
    try:
        assert gt_num_nodes == nx_num_nodes
        assert gt_num_edges == nx_num_edges
        print("MST created with GT and NX has the same number of nodes and edges as expected.\n")
    except AssertionError as ae:
        print("MST created with GT and NX does not have the same number of nodes and edges as expected.\n")
        raise ae

