import numpy as np

from pp_netlib.network import Network

ref_list = "/Users/bruhad/Desktop/Code/pop_net_utils/tests/Mass_hdbscan.refs"
assignments = np.load("/Users/bruhad/Desktop/Code/pop_net_utils/tests/test_assignments.npy")

test_graph = Network(ref_list, ref_list, assignments, "tests/test_construct", "hdbscan", use_gpu=False)

test_graph.construct(assignments)

test_graph.summarize("test_summary")