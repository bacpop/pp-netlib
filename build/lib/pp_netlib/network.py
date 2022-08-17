## internal imports
# from .utils import *
# from .cliques import *
# from .vertices import *
# from .load_network import *
# from .indices_refs_clusters import *
# from .construct_network import *

import os, sys
import numpy as np
import pandas as pd
import graph_tool.all as gt
from pp_netlib.cliques import prune_cliques

from pp_netlib.construct_network import construct_network_from_assignments, convert_data_to_df, initial_graph_properties, network_summary, process_previous_network
from pp_netlib.indices_refs_clusters import add_self_loop


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
        # Check GPU library use
        use_gpu = self.use_gpu

        if isinstance(network_data, np.ndarray):
            construct_network_from_assignments(ref_list, ref_list, assignments)

        # data structures
        vertex_labels, self_comparison = initial_graph_properties(self.ref_list, self.query_list)

        network_data_df = convert_data_to_df(network_data, weights, use_gpu)

        ################################
        #   TODO TODO TODO TODO TODO   #
        #   TODO TODO TODO TODO TODO   #
        ################################

        # Create new network
        if not use_gpu:
            if weights is not None:
                edges = list(network_data_df.itertuples(index=False, name = None))
                # Load previous network
                if previous_network is not None:
                    extra_sources, extra_targets, extra_weights = process_previous_network(previous_network = previous_network, adding_qq_dists = adding_qq_dists, old_ids = old_ids, previous_pkl = previous_pkl, vertex_labels = vertex_labels, weights = (weights is not None), use_gpu = use_gpu)
                    for (src, dest, weight) in zip(extra_sources, extra_targets, extra_weights):
                            edges.append((src, dest, weight))
            else:
                edges = list(network_data_df.itertuples(index=False, name = None))
                if previous_network is not None:
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

            if weights is not None:
                gpu_graph_df["weights"] = weights

            if previous_network is not None:
                G_extra_df = cudf.DataFrame()
                G_extra_df["source"] = extra_sources
                G_extra_df["destination"] = extra_targets
                if extra_weights is not None:
                    G_extra_df["weights"] = extra_weights
                gpu_graph_df = cudf.concat([gpu_graph_df,G_extra_df], ignore_index = True)

            # direct conversion
            # ensure the highest-integer node is included in the edge list
            # by adding a self-loop if necessary; see https://github.com/rapidsai/cugraph/issues/1206
            max_in_vertex_labels = len(vertex_labels)-1
            use_weights = False
            if weights:
                use_weights = True
            self.graph = add_self_loop(gpu_graph_df, max_in_vertex_labels, weights = use_weights, renumber = False)
                
        return self.graph

        ################################
        #   TODO TODO TODO TODO TODO   #
        #   TODO TODO TODO TODO TODO   #
        ################################

    def prune(self):
        prune_cliques() ###TODO populate this functino call with arguments
        return

    def summarize(self, summary_file_prefix):
        """Wrapper function for printing network information

        ## DEPENDS ON Fns: {.:[network_summary]}

        Args:
            graph (graph)
                List of reference sequence labels
            betweenness_sample (int)
                Number of sequences per component used to estimate betweenness using
                a GPU. Smaller numbers are faster but less precise [default = 100]
            use_gpu (bool)
                Whether to use GPUs for network construction
        """
        # print some summaries
        if not self.graph:
            raise RuntimeError("Graph not set")
        
        (metrics, scores) = network_summary(self.graph, betweenness_sample = 100, use_gpu = self.use_gpu)

        summary_contents = ("Network summary:\n" + "\n".join(["\tComponents\t\t\t\t" + str(metrics[0]),
                                                    "\tDensity\t\t\t\t\t" + "{:.4f}".format(metrics[1]),
                                                    "\tTransitivity\t\t\t\t" + "{:.4f}".format(metrics[2]),
                                                    "\tMean betweenness\t\t\t" + "{:.4f}".format(metrics[3]),
                                                    "\tWeighted-mean betweenness\t\t" + "{:.4f}".format(metrics[4]),
                                                    "\tScore\t\t\t\t\t" + "{:.4f}".format(scores[0]),
                                                    "\tScore (w/ betweenness)\t\t\t" + "{:.4f}".format(scores[1]),
                                                    "\tScore (w/ weighted-betweenness)\t\t" + "{:.4f}".format(scores[2])])
                                                    + "\n")
        sys.stderr.write(summary_contents)
        summary_file_name = os.path.join(self.outdir, (str(summary_file_prefix) + ".txt"))

        #################################
        #   extra little thing to       #
        #  write summary to plain text  #
        #################################

        with open(summary_file_name, "w") as summary:
            summary.write(summary_contents)
        summary.close()
        return

    def visualize(self):
        # code for calling viz functions
        print("visualizing network")
        return #files associated with viz

    def load_network(self, network_file):
        """Load the network based on input options
       Returns the network as a graph-tool format graph, and sets
       the slope parameter of the passed model object.

       ## DEPENDS ON Fns: {none}

       Args:
            network_file (str)
                Network file name
            use_gpu (bool)
                Use cugraph library to load graph
       Returns:
            genome_network (graph)
                The loaded network
    """
        # Load the network from the specified file

        if not self.use_gpu:
            genome_network = gt.load_graph(network_file)
            sys.stderr.write("Network loaded: " + str(len(list(genome_network.vertices()))) + " samples\n")
        else:
            graph_df = cudf.read_csv(network_file, compression = "gzip")
            if "src" in graph_df.columns:
                graph_df.rename(columns={"src": "source", "dst": "destination"}, inplace=True)
            genome_network = cugraph.Graph()
            if "weights" in graph_df.columns:
                graph_df = graph_df[["source", "destination", "weights"]]
                genome_network.from_cudf_edgelist(graph_df, edge_attr = "weights", renumber = False)
            else:
                genome_network.from_cudf_edgelist(graph_df, renumber = False)
            sys.stderr.write("Network loaded: " + str(genome_network.number_of_vertices()) + " samples\n")

        return genome_network

    def add_to_network(self, datapoint):
        # calls functions which load a preexisting network, or work with a newly built one, and add data to it?
        print(f"adding {datapoint} to network")
        return

    def _convert(self, intial_format, target_format):
        ### TODO call load_network, use network_to_edges, then call construct, add check to prevent computation in case of missing imports

        if intial_format == "cugraph":
            cugraph_dataframe = cugraph.to_pandas_edgelist(self.graph)



        if target_format == "cugraph" and not self.use_gpu:
            sys.stderr.write("You have asked for your gaph to be converted to cugraph format, but your system/environment seems to be missing gpu related imports. Converting anyway...")

        

        print(f"converting from {intial_format} to {target_format}")
        return

    def save(self, prefix, suffix, use_graphml):
        """Save a network to disk

        ## DEPENDS ON Fns: {None}

        Args:
        graph (network)
            Graph tool network
        prefix (str)
            Prefix for output file
        use_graphml (bool)
            Whether to output a graph-tool file
            in graphml format
        use_gpu (bool)
            Whether graph is a cugraph or not
            [default = False]
        """

        file_name = prefix + "/" + os.path.basename(prefix)
        if suffix is not None:
            file_name = file_name + suffix
        if not self.use_gpu:
            if use_graphml:
                self.graph.save(file_name + ".graphml", fmt = "graphml")
            else:
                self.graph.save(file_name + ".gt", fmt = "gt")
        else:
            self.graph.to_pandas_edgelist().to_csv(file_name + ".csv.gz", compression="gzip", index = False)


# a = Network("refs_list", "assignments.npy", "/Home/Desktop/", "dbscan", True)

# a.load_network()
# a.construct()
# a.prune()
# a.summarize()
# a.visualize()
# a.convert("cugraph", "graphtool")
# a.add_to_network("random_data_point")
# a.save()

ref_list = "tests/Mass_hdbscan.refs"
assignments = np.load("tests/test_assignments.npy")

test_graph = Network(ref_list, ref_list, assignments, "tests/test_construct", "hdbscan", use_gpu=False)

test_graph.construct(assignments)