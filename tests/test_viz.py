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

    expected_outfiles = ["tests/mst/test_gt_mst_cluster_plot.png", "tests/mst/test_gt_mst_network_data.graphml", "tests/mst/test_gt_mst_stress_plot.png"]
    for outfile in expected_outfiles:
        try:
            assert os.path.isfile(outfile)
        except AssertionError as ae:
            print(f"{outfile} not found when making mst with GT.\n")
            raise ae

    for outfile in expected_outfiles:
        os.remove(outfile) 
    os.rmdir("./tests/mst")
    print("Cleanup after GT mst done.\n")

    return num_vertices, num_edges

def test_nx_mst(sample_file):
    sample_nx_graph = Network([], outdir="tests/", backend = "NX")
    sample_nx_graph.load_network(sample_file)
    sample_nx_graph.visualize("mst", "test_nx")
    num_vertices = sample_nx_graph.mst_network.number_of_nodes()
    num_edges = sample_nx_graph.mst_network.number_of_edges()

    print(f"sample_nx_graph mst has {num_vertices} nodes and {num_edges} edges.")

    expected_outfiles = ["tests/mst/test_nx_mst_cluster_plot.png", "tests/mst/test_nx_mst_network_data.graphml", "tests/mst/test_nx_mst_stress_plot.png"]
    for outfile in expected_outfiles:
        try:
            assert os.path.isfile(outfile)
        except AssertionError as ae:
            print(f"{outfile} not found when making mst with NX.\n")
            raise ae

    for outfile in expected_outfiles:
        os.remove(outfile) 
    os.rmdir("./tests/mst")
    print("Cleanup after NX mst done.\n")

    return num_vertices, num_edges

def test_gt_cytoscape(sample_file):
    sample_gt_graph = Network([], outdir="tests/", backend = "GT")
    sample_gt_graph.load_network(sample_file)

    sample_gt_graph.visualize("cytoscape", "test_gt")
    expected_outfiles = ["tests/cytoscape/test_gt_mst.graphml", "tests/cytoscape/test_gt_cytoscape.graphml", "tests/cytoscape/test_gt.csv"]

    num_components = len(set(gt.label_components(sample_gt_graph.graph)[0].a))
    for i in range (num_components):
        expected_outfiles.append("tests/cytoscape/test_gt_component_"+str(i+1)+".graphml")

    for outfile in expected_outfiles:
        try:
            assert os.path.isfile(outfile)
        except AssertionError as ae:
            print(f"{outfile} not found when making cytoscape outputs with GT.\n")
            raise ae

    for outfile in expected_outfiles:
        os.remove(outfile)

    os.rmdir("./tests/cytoscape")
    print("Cleanup after GT cytoscape test done.\n")

def test_nx_cytoscape(sample_file):
    sample_nx_graph = Network([], outdir="tests/", backend = "NX")
    sample_nx_graph.load_network(sample_file)

    sample_nx_graph.visualize("cytoscape", "test_nx")
    expected_outfiles = ["tests/cytoscape/test_nx_mst.graphml", "tests/cytoscape/test_nx_cytoscape.graphml", "tests/cytoscape/test_nx.csv"]

    num_components = nx.number_connected_components(sample_nx_graph.graph)
    for i in range (num_components):
        expected_outfiles.append("tests/cytoscape/test_nx_component_"+str(i+1)+".graphml")

    for outfile in expected_outfiles:
        try:
            assert os.path.isfile(outfile)
        except AssertionError as ae:
            print(f"{outfile} not found when making cytoscape outputs with NX.\n")
            raise ae

    for outfile in expected_outfiles:
        os.remove(outfile)

    os.rmdir("./tests/cytoscape")
    print("Cleanup after NX cytoscape test done.\n")

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

    test_gt_cytoscape(sample_file)
    test_nx_cytoscape(sample_file)

