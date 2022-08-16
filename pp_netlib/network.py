import os, sys
import numpy as np
import pandas as pd
import graph_tool.all as gt

from pp_netlib.functions import initial_graph_properties, convert_data_to_df, process_previous_network

class Network:
    def __init__(self, ref_list, query_list, assignments, outdir, model_type, use_gpu = False):
        self.ref_list = ref_list
        self.query_list = query_list
        self.assignments = assignments
        self.outdir = outdir
        self.use_gpu = use_gpu
        self.model_type = model_type
        self.graph = None

        if use_gpu:
            try:
                import cupyx
                import cugraph
                import cudf
                import cupy as cp
                from numba import cuda
                import rmm
                use_gpu = True
            except ImportError as e:
                sys.stderr.write("Unable to load GPU libraries; using CPU libraries instead\n")
                use_gpu = False

        print(ref_list, assignments, outdir, use_gpu, model_type)

    def construct(self, network_data, weights = None, previous_network = None, adding_qq_dists = False, old_ids = None, previous_pkl = None): ## construct from edge_list
        ## Check whether GPU to be used
        use_gpu = self.use_gpu

        # data structures
        vertex_labels, self_comparison = initial_graph_properties(self.ref_list, self.query_list)

        ## force input data into a dataframe using convert_data_to_df
        network_data_df = convert_data_to_df(network_data, weights, use_gpu)

        # Create new network
        # if not use_gpu:
        #     if weights is not None:
        #         edges = list(network_data_df.itertuples(index=False, name = None))
        #         # Load previous network
        #         if previous_network is not None:
        #             extra_sources, extra_targets, extra_weights = process_previous_network(previous_network = previous_network, adding_qq_dists = adding_qq_dists, old_ids = old_ids, previous_pkl = previous_pkl, vertex_labels = vertex_labels, weights = (weights is not None), use_gpu = use_gpu)
        #             for (src, dest, weight) in zip(extra_sources, extra_targets, extra_weights):
        #                     edges.append((src, dest, weight))
        #     else:
        #         edges = list(network_data_df.itertuples(index=False, name = None))
        #         if previous_network is not None:
        #             extra_sources, extra_targets = process_previous_network(previous_network = previous_network, adding_qq_dists = adding_qq_dists, old_ids = old_ids, previous_pkl = previous_pkl, vertex_labels = vertex_labels, weights = False, use_gpu = use_gpu)
        #             for (src, dest) in zip(extra_sources, extra_targets):
        #                 edges.append((src, dest))
        
        #     self.graph = gt.Graph(directed = False)
        #     self.graph.add_vertex(len(vertex_labels))
        #     if weights is not None:
        #         eweight = self.graph.new_ep("float")
        #         self.graph.add_edge_list(edges, eprops = [eweight])
        #         self.graph.edge_properties["weight"] = eweight
        #     else:
        #         self.graph.add_edge_list(edges)

            #### ALTERNATE IMPLEMENTATION ####

        if weights is not None:
            network_data_df["weights"] = weights

        if not use_gpu:
            edges = list(network_data_df.itertuples(index=False, name = None))
            try:
                check_weights = edges[0][2]
                data_has_weights = True
            except IndexError as ie:
                data_has_weights = False

            if previous_network is not None:
                if data_has_weights:
                    extra_sources, extra_targets, extra_weights = process_previous_network(previous_network = previous_network, adding_qq_dists = adding_qq_dists, old_ids = old_ids, previous_pkl = previous_pkl, vertex_labels = vertex_labels, weights = (weights is not None), use_gpu = use_gpu)
                    for (src, dest, weight) in zip(extra_sources, extra_targets, extra_weights):
                            edges.append((src, dest, weight))
                else:
                    extra_sources, extra_targets = process_previous_network(previous_network = previous_network, adding_qq_dists = adding_qq_dists, old_ids = old_ids, previous_pkl = previous_pkl, vertex_labels = vertex_labels, weights = False, use_gpu = use_gpu)
                    for (src, dest) in zip(extra_sources, extra_targets):
                        edges.append((src, dest))

            self.graph = gt.Graph(directed = False)
            self.graph.add_vertex(len(vertex_labels))
            if weights is not None:
                eweight = self.graph.new_ep("float")
                self.graph.add_edge_list(edges, eprops = [eweight])
                self.graph.edge_properties["weight"] = eweight
            else:
                self.graph.add_edge_list(edges)

        else:
            # benchmarking concurs with https://stackoverflow.com/questions/55922162/recommended-cudf-dataframe-construction
            if len(network_data_df) > 1:
                edge_array = cp.array(network_data_df, dtype = np.int32)
                edge_gpu_matrix = cuda.to_device(edge_array)
                gpu_graph_df = cudf.DataFrame(edge_gpu_matrix, columns = ["source","destination"])
            else:
                # Cannot generate an array when one edge
                gpu_graph_df = cudf.DataFrame(columns = ["source","destination"])
                gpu_graph_df["source"] = [network_data_df[0][0]]
                gpu_graph_df["destination"] = [network_data_df[0][1]]


