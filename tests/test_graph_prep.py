import os

from pp_netlib.network import Network

sample_file = "tests/test_graph.graphml"
sample_2 = "tests/random_weighted.graphml"

def test_prep_with_gt(graph_file):
    sample_gt_graph = Network([], outdir="tests/", backend = "GT")
    sample_gt_graph.load_network(graph_file)
    sample_gt_graph.write_metadata("tests", "test_gt_summary")

    expected_outfiles = ["tests/test_gt_summary_edge_data.tsv", "tests/test_gt_summary_node_data.tsv"]

    for outfile in expected_outfiles:
        try:
            assert os.path.isfile(outfile)
        except AssertionError as ae:
            print(f"{outfile} not found when making mst with GT.\n")
            raise ae

    for outfile in expected_outfiles:
        os.remove(outfile)
    print("Cleanup done after testing metadata output with GT.\n")

def test_prep_with_nx(graph_file):
    sample_nx_graph = Network([], outdir="tests/", backend = "NX")
    sample_nx_graph.load_network(graph_file)
    sample_nx_graph.write_metadata("tests", "test_nx_summary")

    expected_outfiles = ["tests/test_nx_summary_edge_data.tsv", "tests/test_nx_summary_node_data.tsv"]

    for outfile in expected_outfiles:
        try:
            assert os.path.isfile(outfile)
        except AssertionError as ae:
            print(f"{outfile} not found when making mst with NX.\n")
            raise ae

    for outfile in expected_outfiles:
        os.remove(outfile)
    print("Cleanup done after testing metadata output with NX.\n")

if __name__ == "__main__":
    test_prep_with_gt(sample_file)
    test_prep_with_nx(sample_file)
    print("\n\nNow testing with weighted graph\n\n")
    test_prep_with_gt(sample_2)
    test_prep_with_nx(sample_2)