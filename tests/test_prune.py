import graph_tool.all as gt
import networkx as nx

from pp_netlib.network import Network

sample_file = "tests/test_graph.graphml"
sample_pruned_file = "tests/test_pruned_graph.graphml"

def test_gt_prune():

    sample_gt_graph = Network([], backend = "GT")
    sample_gt_graph.load_network(sample_file)
    sample_pruned_gt_graph = Network([], backend = "GT")
    sample_pruned_gt_graph.load_network(sample_pruned_file)

    sample_pruned_edges = []
    for s, t in sample_pruned_gt_graph.graph.iter_edges():
        sv = sample_pruned_gt_graph.graph.vp.id[s]
        tv = sample_pruned_gt_graph.graph.vp.id[t]
        sample_pruned_edges.append((sv, tv))

    test_initial_components = len(set(gt.label_components(sample_gt_graph.graph)[0].a))
    print(f"{test_initial_components} components in sample unpruned graph.")

    sample_pruned_components = len(set(gt.label_components(sample_pruned_gt_graph.graph)[0].a))

    print(f"{sample_pruned_components} components in sample pruned graph.")
    sample_gt_graph.prune("6925_1_57")
    test_pruned_components = len(set(gt.label_components(sample_gt_graph.graph)[0].a))

    test_pruned_edges = []
    for s, t in sample_gt_graph.graph.iter_edges():
        sv = sample_gt_graph.graph.vp.id[s]
        tv = sample_gt_graph.graph.vp.id[t]
        test_pruned_edges.append((sv, tv))

    assert test_pruned_edges == sample_pruned_edges

    print(f"{test_pruned_components} components in test pruned graph.")

test_gt_prune()