from functools import partial
from multiprocessing import Pool
import os, sys
import numpy as np
import pandas as pd
import scipy

from pp_netlib.functions import construct_with_graphtool, construct_with_networkx, summarise
from pp_netlib.prune_functions import gt_clique_prune, gt_get_ref_graph, nx_clique_prune

class Network:
    def __init__(self, ref_list, query_list = [], outdir = "./", backend = None, use_gpu = False):
        """Initialises a graph object (based on graph-tool, networkx (TODO), or cugraph (TODO)). Produces a Network object.

        Args:
            ref_list (list): List of sequence names/identifiers (names/identifiers should be strings) which will be vertices in the graph
            query_list (list): List of sequence names/identifiers (names/identifiers) (TODO not used/necessary?)
            outdir (str): Path to output directory where graph files will be stored. Defaults to "./" (i.e. working directory) (TODO not used currently)
            backend (str, optional): Which graphing module to use. Can be specified here (valid options: "GT", "NX", "CU") or as an environment variable. (TODO)
            use_gpu (bool, optional): Whether to use GPU and GPU python modules. Defaults to False.
                                      If set to True and if ImportErrors are raised due to missing moduiles, will revert to False.

        Usage:
        your_script.py or interactive python terminal
            ```
            from pp_netlib.network import Network

            samples_list = ["sample1", "sample2", "sample3"]

            #initialise a graph object, supplying three labels, outdir as Desktop,
            # graphing module as graph-tool, and not using gpu and related modules
            example_graph = Network(ref_list = samples_list, outdir = "/Users/user/Desktop", backend = "GT", use_gpu = False)


            example_graph.construct(*construct_args) ## call the construct method to populate graph object with your data
            example_graph.save(*save_args) ## save your graph to file in outdir
            ```

            Graphing backend can alternatively be set as follows: (TODO To be discussed)
            ```
            os.environ["GRAPH_BACKEND"] = "GT" ## inside of your_script.py or in interactive python terminal

            *OR*

            export GRAPH_BACKEND=GT ## inside bash/shell terminal/environment
            ```
        """
        self.ref_list = ref_list
        self.query_list = query_list
        self.outdir = outdir
        if backend is None:
            self.backend = os.getenv("GRAPH_BACKEND") ## read os env variable "GRAPH_BACKEND"
        else:
            self.backend = backend ## else use backend if specified
        self.use_gpu = use_gpu
        self.graph = None

        if self.backend == "GT":
            import graph_tool.all as gt
            self.gt = gt
        elif self.backend == "NX":
            import networkx as nx
            self.nx = nx
        elif self.backend == "CU":
            raise NotImplementedError("GPU graph not yet implemented")
            # import cupyx
            # import cugraph
            # import cudf
            # import cupy as cp
            # from numba import cuda
            # import rmm
            # use_gpu = True

    def construct(self, network_data, weights = None): #, previous_network = None, adding_qq_dists = False, old_ids = None, previous_pkl = None):
        """Method called on Network object. Constructs a graph using either graph-tool, networkx (TODO), or cugraph(TODO)

        Args:
            network_data (dataframe OR edge list OR sparse coordinate matrix): Data containing record of edges in the graph.
            weights (list, optional): List of weights associated with edges in network_data.
                                      Weights must be in the same order as edges in network_data. Defaults to None.

        This method is called on a Network object and produces a graph populated with edges.

        The number of ref_list elements (used in initialising the Network object) is assumed to be equal to the number of edges in any of
        the following data types. Additionally, the orders of elements in the ref_list and network_data are also assumed to correspond exactly.

        network_data can be a dataframe, sparse matrix, or a list of tuples where each tuple contains source and destination node indices.
        The following data generate identical graphs:
        ## Edge List
        ```
        >> edge_list
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        ```

        ## Dataframe
        >> edge_df
        ```
        ## column [0] is source nodes
        ## column [1] is destination nodes
            0   1
        0   0   1
        1   1   2
        2   2   3
        3   3   4
        4   4   0
        ```

        ## Sparse matrix
        ## edge_matrix.row should return source nodes
        ## edge_matrix.col should return destination nodes
        ## edge_matrix.data should return weights
        >> edge_matrix
        ```
        (0, 1)	0.1
        (1, 2)	0.1
        (2, 3)	0.5
        (3, 4)	0.7
        (4, 0)	0.2

        ## the above matrix as a numpy array for reference
        >> edge_matrix.toarray()
        [[0.0  0.1 0.0  0.0  0.0 ]
        [0.0  0.0  0.1 0.0  0.0 ]
        [0.0  0.0  0.0  0.5 0.0 ]
        [0.0  0.0  0.0  0.0  0.7]
        [0.2 0.0  0.0  0.0  0.0 ]]
        ```
        """
        ########################
        ####     INITIAL    ####
        ########################

        # Check GPU library use
        use_gpu = self.use_gpu

        # data structures
        if self.ref_list != self.query_list:
            vertex_labels = self.ref_list + self.query_list
            self_comparison = False
        else:
            vertex_labels = self.ref_list
            self_comparison = True
        self.vertex_labels = vertex_labels
        self.weights = weights
        # initialise a graph object
        if self.backend == "GT":
            self.graph = construct_with_graphtool(network_data=network_data, vertex_labels=self.vertex_labels, weights=weights)
        elif self.backend == "NX":
            self.graph = construct_with_networkx(network_data=network_data, vertex_labels=self.vertex_labels, weights=weights)

        ## keeping this section here for now; might be useful in add_to_network method
        # ########################
        # ####  PREV NETWORK  ####
        # ########################
        # if previous_network is not None:
        #     prev_edges = []
        #     if weights is not None:
        #         extra_sources, extra_targets, extra_weights = process_previous_network(previous_network = previous_network, adding_qq_dists = adding_qq_dists, old_ids = old_ids, previous_pkl = previous_pkl, vertex_labels = vertex_labels, weights = (weights is not None), use_gpu = use_gpu)
        #         for (src, dest, weight) in zip(extra_sources, extra_targets, extra_weights):
        #                 prev_edges.append((src, dest, weight))

        #     else:
        #         extra_sources, extra_targets = process_previous_network(previous_network = previous_network, adding_qq_dists = adding_qq_dists, old_ids = old_ids, previous_pkl = previous_pkl, vertex_labels = vertex_labels, weights = (weights is not None), use_gpu = use_gpu)
        #         for (src, dest, weight) in zip(extra_sources, extra_targets):
        #                 prev_edges.append((src, dest))


        #     self.graph.add_edge_list(prev_edges) ## add previous edge list to newly made graph

        #     self.edges = [edge for edge in self.graph.edges()]


    def prune(self, type_isolate = None):

        if self.backend == "GT":
            reference_vertices = set()
            components = self.gt.label_components(self.graph)[0].a
            #reference_vertices = gt_prune_cliques(graph=self.graph, reference_indices=set(), components_list=(components))

            for component in set(components):
                reference_indices = gt_clique_prune(component, self.graph, set(), components)
                reference_vertices.update(reference_indices)
                #print(reference_vertices)

            # with Pool as pool:
            #     ref_verts = pool.map([component for component in set(components)], clique_prune, graph=self.graph, reference_indices=set(), component_list=components)

        if self.backend == "NX":
            for idx, c in enumerate(self.nx.connected_components(self.graph)):
                for v in c:
                    self.graph.nodes[v]["comp_membership"] = idx
            reference_vertices = set()
            components = list(c for c in self.nx.connected_components(self.graph))

            for component in self.nx.connected_components(self.graph):
                reference_indices = nx_clique_prune(component, self.graph, set(), components)
                reference_vertices.update(reference_indices)

        if self.backend == "GT":
            labels = list(self.graph.vp["id"][v] for v in self.graph.vertices())
            self.graph = gt_get_ref_graph(self.graph, reference_vertices, labels, type_isolate, backend=self.backend)
            num_nodes = self.graph.num_vertices()
            num_edges = self.graph.num_edges()
        elif self.backend == "NX":
            
            labels = list((self.nx.get_node_attributes(self.graph, "id")).values())
            self.graph = gt_get_ref_graph(self.graph, reference_vertices, labels, type_isolate, backend=self.backend)
            num_nodes = self.graph.number_of_nodes()
            num_edges = self.graph.number_of_edges()

        sys.stderr.write(f"Pruned network has {num_nodes} nodes and {num_edges} edges.\n\n")

    def get_summary(self, print_to_std = True, summary_file_prefix = None):
        """Method called on initialised and populated Network object. Prints summary of network properties to stderr and optionally to plain text file.

        Args:
            summary_file_prefix (str, optional): File name of summary file to which graph property summaries should be written.
            Defaults to None. If None, no summary file is written.

        Raises:
            RuntimeError: RuntimeError raised if no graph initialised.
        """
        # print some summaries
        if self.graph is None:
            raise RuntimeError("Graph not constructed or loaded.")

        self.metrics, self.scores = summarise(self.graph, self.backend)

        summary_contents = ("Network summary:\n" + "\n".join(["\tComponents\t\t\t\t" + str(self.metrics[0]),
                                                    "\tDensity\t\t\t\t\t" + "{:.4f}".format(self.metrics[1]),
                                                    "\tTransitivity\t\t\t\t" + "{:.4f}".format(self.metrics[2]),
                                                    "\tMean betweenness\t\t\t" + "{:.4f}".format(self.metrics[3]),
                                                    "\tWeighted-mean betweenness\t\t" + "{:.4f}".format(self.metrics[4]),
                                                    "\tScore\t\t\t\t\t" + "{:.4f}".format(self.scores[0]),
                                                    "\tScore (w/ betweenness)\t\t\t" + "{:.4f}".format(self.scores[1]),
                                                    "\tScore (w/ weighted-betweenness)\t\t" + "{:.4f}".format(self.scores[2])])
                                                    + "\n")
        if print_to_std:
            sys.stderr.write(f"{summary_contents}\n\n")

        #################################
        #  write summary to plain text  #
        #################################
        if summary_file_prefix is not None:
            summary_file_name = os.path.join(self.outdir, (str(summary_file_prefix) + ".txt"))

            with open(summary_file_name, "w") as summary:
                summary.write(summary_contents)
            summary.close()

    def visualize(self):
        # code for calling viz functions
        print("visualizing network")
        return #files associated with viz

    def load_network(self, network_file):
        """Load a premade graph from a network file.

        Args:
            network_file (str/path): The file in which the prebuilt graph is stored. Must be a .gt file, .graphml file or .xml file.
        """
        if self.graph is not None:
            sys.stderr.write("Network instance already contains a graph. Cannot load another graph.\n\n")
            sys.exit(1)
        
        # Load the network from the specified file
        file_name, file_extension = os.path.splitext(network_file)
        if file_extension == ".gt":
            self.graph = self.gt.load_graph(network_file)
            num_nodes = len(list(self.graph.vertices()))
            num_edges = len(list(self.graph.edges()))
            sys.stderr.write(f"Loaded network with {num_nodes} nodes and {num_edges} edges with {self.backend}.\n\n")
            if self.backend == "NX":
                sys.stderr.write("Network file provided is in .gt format and cannot be opened with networkx. Quitting.\n\n")
                sys.exit(1)

        elif file_extension in [".graphml", ".xml"]:
            if self.backend == "GT":
                self.graph = self.gt.load_graph(network_file)
                num_nodes = len(list(self.graph.vertices()))
                num_edges = len(list(self.graph.edges()))
            if self.backend == "NX":
                self.graph = self.nx.read_graphml(network_file)
                num_nodes = len(self.graph.nodes())
                num_edges = len(self.graph.edges())
        
            sys.stderr.write(f"Loaded network with {num_nodes} nodes and {num_edges} edges with {self.backend}.\n\n")

        # useful for cugraph, to be added in later
        # elif file_extension in [".csv", ".tsv", ".txt"]:
        #     sys.stderr.write("The network file appears to be in tabular format, please load it as a dataframe and use the construct method to build a graph.\n")
        #     sys.exit(1)

        else:
            sys.stderr.write("File format not recognised. Quitting...\n\n")
            sys.exit(1)
        
        # graph_df = cudf.read_csv(network_file, compression = "gzip")
        # if "src" in graph_df.columns:
        #     graph_df.rename(columns={"src": "source", "dst": "destination"}, inplace=True)
        # genome_network = cugraph.Graph()
        # if "weights" in graph_df.columns:
        #     graph_df = graph_df[["source", "destination", "weights"]]
        #     genome_network.from_cudf_edgelist(graph_df, edge_attr = "weights", renumber = False)
        # else:
        #     genome_network.from_cudf_edgelist(graph_df, renumber = False)
        # sys.stderr.write("Network loaded: " + str(genome_network.number_of_vertices()) + " samples\n")

    def add_to_network(self, new_data_df, new_vertex_labels):

        if self.graph is None:
            sys.stderr.write("No network found, cannot add data. Please load a network to update or construct a network with this data.\n\n")

        if self.backend == "GT":
            self.graph.add_vertex(len(new_vertex_labels)) ## add new vertices

            if self.weights is not None:
                if "weights" not in new_data_df.columns: ## if existing graph has edge weights, but new data does not
                    new_data_df["weights"] = 0.0 ## add a dummy column
                ## update graph
                eweight = self.graph.ep["weight"]
                self.graph.add_edge_list(new_data_df.values, eprops = [eweight])
            
            elif self.weights is None: 
                if "weights" in new_data_df.columns: ## if existing graph does not have edge weights, but new data does
                    new_data_df.drop(columns=["weights"]) ## remove the weights column
                ## update graph
                self.graph.add_edge_list(new_data_df.values)

            ## add vertex labels to new data
            for idx, vertex_label in enumerate(new_vertex_labels):
                self.graph.vp.id[idx + len(self.vertex_labels)] = vertex_label

        if self.backend == "NX":
            new_nodes_list = [(i+len(self.vertex_labels), dict(id=new_vertex_labels[i])) for i in range(len(new_vertex_labels))]
            self.graph.add_nodes_from(new_nodes_list) ## add new nodes

            if self.weights is not None:
                if "weights" not in new_data_df.columns: ## if existing graph has edge weights, but new data does not
                    new_data_df["weights"] = 0.0 ## add a dummy column
                self.graph.add_weighted_edges_from(new_data_df.values)

            elif self.weights is None:
                if "weights" in new_data_df.columns: ## if existing graph does not have edge weights, but new data does
                    new_data_df.drop(columns=["weights"]) ## remove the weights column
                self.graph.add_edges_from(new_data_df.values)

        self.vertex_labels += new_vertex_labels

    def _convert(self, intial_format, target_format):
        ### TODO call load_network, use network_to_edges, then call construct, add check to prevent computation in case of missing imports

        if intial_format == "cugraph":
            cugraph_dataframe = cugraph.to_pandas_edgelist(self.graph)



        if target_format == "cugraph" and not self.use_gpu:
            sys.stderr.write("You have asked for your graph to be converted to cugraph format, but your system/environment seems to be missing gpu related imports. Converting anyway...")

        

        print(f"converting from {intial_format} to {target_format}")
        return

    def save(self, file_name, file_format):
        """Save graph to file.

        Args:
            file_name (str): Name to be given to the graph file
            file_format (str): File extenstion to be used with graph file

            Example:
            ```
            graph.save("sample_graph", ".graphml")
            ```

        Raises:
            NotImplementedError: If graph_tool is selected a backend,
        """
        if self.graph is None:
            raise RuntimeError("Graph not constructed or loaded.")

        outdir = self.outdir
        if self.backend == "GT":
            if file_format is None:
                self.graph.save(os.path.join(outdir, file_name+".gt"))
            elif file_format is not None:
                if file_format not in [".gt", ".graphml"]:
                    raise NotImplementedError("Supported file formats to save a graph-tools graph are .gt or .graphml")
                else:
                    self.graph.save(os.path.join(outdir, file_name+file_format))

        if self.backend == "NX":
            self.nx.write_graphml(self.graph, os.path.join(outdir, file_name+".graphml"))

        # useful with cugraph, to be added in later
        # if self.backend == "CU":
        #     self.graph.to_pandas_edgelist().to_csv(file_name + ".csv.gz", compression="gzip", index = False)
