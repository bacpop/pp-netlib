import graph_tool.all as gt
import networkx as nx

from pp_netlib.network import Network
from pp_netlib.functions import gt_get_graph_data, nx_get_graph_data, prepare_graph

sample_file = "tests/test_graph.graphml"

sample_gt_graph = Network([], outdir="tests/", backend = "GT")
sample_gt_graph.load_network(sample_file)

prepped_gt_graph = prepare_graph(sample_gt_graph.graph, "GT")

gt_edges, gt_nodes = gt_get_graph_data(prepped_gt_graph)
print(gt_edges)
print(gt_nodes)


sample_nx_graph = Network([], outdir="tests/", backend = "NX")
sample_nx_graph.load_network(sample_file)

prepped_nx_graph = prepare_graph(sample_nx_graph.graph, "NX")

nx_edges, nx_nodes = nx_get_graph_data(prepped_nx_graph)
print(nx_edges)
print(nx_nodes)
