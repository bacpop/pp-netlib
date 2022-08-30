import os, sys
import numpy as np
import pandas as pd
import scipy
import graph_tool.all as gt


class Network:
    def __init__(self, ref_list, query_list = None, outdir = "./", backend = None, use_gpu = False):
        """Initialises a graph object (based on graph-tool, networkx (TODO), or cugraph (TODO)). Produces a Network object.

        Args:
            ref_list (list): List of sequence names/identifiers (names/identifiers should be strings) which will be vertices in the graph
            query_list (list): List of sequence names/identifiers (names/identifiers) (TODO not used/necessary?)
            outdir (str): Path to output directory where graph files will be stored. Defaults to "./" (i.e. working directory) (TODO not used currently)
            backend (str, optional): Which graphing module to use. Can be specified here (valid options: "GT", "NX", "CG") or as an environment variable. (TODO)
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
        """
        self.ref_list = ref_list
        self.query_list = query_list
        self.outdir = outdir
        if backend is None:
            self.backend = os.getenv("GRAPH_BACKEND")
        else:
            self.backend = backend ## read os env variable "GRAPH_BACKEND"
        self.use_gpu = use_gpu
        self.graph = None

        if use_gpu:
            raise NotImplementedError("GPU graph not yet implemented")
            use_gpu = False
            # try:
            #     import cupyx
            #     import cugraph
            #     import cudf
            #     import cupy as cp
            #     from numba import cuda
            #     import rmm
            #     use_gpu = True
            # except ImportError or ModuleNotFoundError as e:
            #     sys.stderr.write("Unable to load GPU libraries; using CPU libraries instead\n")
            #     use_gpu = False


    def construct(self, network_data, weights = None): #, previous_network = None, adding_qq_dists = False, old_ids = None, previous_pkl = None):
        """Method called on Network object. Constructs a graph using either graph-tool, networkx (TODO), or cugraph(TODO)

        Args:
            network_data (dataframe OR edge list OR sparse coordinate matrix): Data containing record of edges in the graph.
            weights (list, optional): List of weights associated with edges in network_data.
                                      Weights must be in the same order as edges in network_data. Defaults to None.

        This method is called on a Network object and produces a graph populated with edges.

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

        # initialise a graph object
        self.graph = gt.Graph(directed = False) ## initialise graph_tool graph object

        ########################
        ####    DF INPUT    ####
        ########################
        if isinstance(network_data, pd.DataFrame):
            # if use_gpu:
            #     network_data = cudf.from_pandas(network_data) ## convert to cudf if use_gpu
            ## add column names
            network_data.columns = ["source", "destination"]

            self.graph.add_vertex(len(network_data)) ## add vertices

            ## add weights column if weights provided as list (add error catching?)
            if weights is not None:
                network_data["weights"] = weights
                eweight = self.graph.new_ep("float")
                self.graph.add_edge_list(network_data.values, eprops = [eweight]) ## add weighted edges
                self.graph.edge_properties["weight"] = eweight
            else:
                self.graph.add_edge_list(network_data.values) ## add edges

        ##########################
        #### SPARSE MAT INPUT ####
        ##########################
        elif isinstance(network_data, scipy.sparse.coo_matrix):
            if not use_gpu:
                graph_data_df = pd.DataFrame()
            # else:
            #     graph_data_df = cudf.DataFrame()
            graph_data_df["source"] = network_data.row
            graph_data_df["destination"] =  network_data.col
            graph_data_df["weights"] = network_data.data

            self.graph.add_vertex(len(graph_data_df)) ## add vertices
            eweight = self.graph.new_ep("float")
            self.graph.add_edge_list(list(map(tuple, graph_data_df.values)), eprops = [eweight]) ## add weighted edges
            self.graph.edge_properties["weight"] = eweight

        ########################
        ####   LIST INPUT   ####
        ########################
        elif isinstance(network_data, list):
            self.graph.add_vertex(len(network_data)) ## add vertices

            if weights is not None:
                weighted_edges = []

                for i in range(len(network_data)):
                    weighted_edges.append(network_data[i] + (weights[i],))

                eweight = self.graph.new_ep("float")
                self.graph.add_edge_list(weighted_edges, eprops = [eweight]) ## add weighted edges
                self.graph.edge_properties["weight"] = eweight

            else:
                self.graph.add_edge_list(network_data) ## add edges

        v_name_prop = self.graph.new_vp("string")
        self.graph.vertex_properties["id"] = v_name_prop
        for i in range(len([v for v in self.graph.vertices()])):
            v_name_prop[self.graph.vertex(i)] = vertex_labels[i]

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


    def prune(self):
        #prune_cliques() ###TODO populate this function call with arguments
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
        
        # (metrics, scores) = network_summary(self.graph, betweenness_sample = 100, use_gpu = self.use_gpu)

        # summary_contents = ("Network summary:\n" + "\n".join(["\tComponents\t\t\t\t" + str(metrics[0]),
        #                                             "\tDensity\t\t\t\t\t" + "{:.4f}".format(metrics[1]),
        #                                             "\tTransitivity\t\t\t\t" + "{:.4f}".format(metrics[2]),
        #                                             "\tMean betweenness\t\t\t" + "{:.4f}".format(metrics[3]),
        #                                             "\tWeighted-mean betweenness\t\t" + "{:.4f}".format(metrics[4]),
        #                                             "\tScore\t\t\t\t\t" + "{:.4f}".format(scores[0]),
        #                                             "\tScore (w/ betweenness)\t\t\t" + "{:.4f}".format(scores[1]),
        #                                             "\tScore (w/ weighted-betweenness)\t\t" + "{:.4f}".format(scores[2])])
        #                                             + "\n")
        # sys.stderr.write(summary_contents)
        # summary_file_name = os.path.join(self.outdir, (str(summary_file_prefix) + ".txt"))

        # #################################
        # #   extra little thing to       #
        # #  write summary to plain text  #
        # #################################

        # with open(summary_file_name, "w") as summary:
        #     summary.write(summary_contents)
        # summary.close()
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
