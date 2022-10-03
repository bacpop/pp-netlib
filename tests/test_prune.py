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
    test_pruned_components = len(set(gt.label_components(sample_gt_graph.ref_graph)[0].a))
    print(f"{test_pruned_components} components in test pruned graph.")

    test_pruned_edges = []
    for s, t in sample_gt_graph.ref_graph.iter_edges():
        sv = sample_gt_graph.ref_graph.vp.id[s]
        tv = sample_gt_graph.ref_graph.vp.id[t]
        test_pruned_edges.append((sv, tv))

    try:
        assert test_pruned_edges == sample_pruned_edges
        print("Pruning with GT backend worked as excepted\n\n")
    except AssertionError as ae:
        print("Something went wrong while pruning with GT backend...\n\n")
        raise ae

def test_nx_prune(type_isolate):
    sample_nx_graph = Network([], backend = "NX")
    sample_nx_graph.load_network(sample_file)


    type_idx = [i[0] for i in list(sample_nx_graph.graph.nodes(data="id")) if i[1] == type_isolate][0]

    for idx, c in enumerate(nx.connected_components(sample_nx_graph.graph)):
        for v in c:
            sample_nx_graph.graph.nodes[v]["comp_membership"] = idx

    original_comp_memberships = list(sample_nx_graph.graph.nodes(data="comp_membership"))
    orig_nodes, orig_comps = zip(*original_comp_memberships)
    original_comp_memberships_dict = dict(zip(orig_nodes, orig_comps))

    sample_nx_graph.prune("6925_1_57")
    num_pruned_nodes = sample_nx_graph.ref_graph.number_of_nodes()
    num_pruned_edges = sample_nx_graph.ref_graph.number_of_edges()
    print(f"\nsample_nx_graph now pruned, contains {num_pruned_nodes} nodes, {num_pruned_edges} edges.\n")

    pruned_comp_memberships = list(sample_nx_graph.ref_graph.nodes(data="comp_membership"))
    pruned_nodes, pruned_comps = zip(*pruned_comp_memberships)
    pruned_comp_memberships_dict = dict(zip(pruned_nodes, pruned_comps))

    try:
        assert set(sorted(orig_comps)) == set(sorted(pruned_comps))
        print("All original components represented in pruned graph.")

        assert type_idx in sample_nx_graph.graph
        print("Type isolate included in pruned graph.")

        for v in sample_nx_graph.ref_graph.nodes():
            assert original_comp_memberships_dict[v] == pruned_comp_memberships_dict[v]
        print("Component memberships preserved after pruning.")

        print("Pruning with NX backend worked as excepted (see caveat)\n\n")

    except AssertionError as ae:
        print("Something went wrong while pruning with NX backend...\n\n")
        raise ae

if __name__ == "__main__":
    test_gt_prune()
    test_nx_prune("6925_1_57")