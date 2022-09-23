import networkx as nx
from pp_netlib.network import Network

sample_file = "tests/test_graph.graphml"

sample_nx_graph = Network([], backend = "NX")
sample_nx_graph.load_network(sample_file)



for idx, c in enumerate(nx.connected_components(sample_nx_graph.graph)):
    for v in c:
        print(v)
        sample_nx_graph.graph.nodes[v]["comp_membership"] = idx
    print(c, idx)

print(list((nx.get_node_attributes(sample_nx_graph.graph, "id")).values()))
print(list((nx.get_node_attributes(sample_nx_graph.graph, "comp_membership")).values()))
print(list((nx.get_node_attributes(sample_nx_graph.graph, "index")).values()))

