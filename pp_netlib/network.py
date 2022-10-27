from functools import partial
from multiprocessing import Pool
import os, sys
import pandas as pd


from pp_netlib.functions import construct_with_graphtool, construct_with_networkx, prepare_graph, summarise, save_graph

class Network:
    def __init__(self, ref_list, outdir = "./", backend = None):
        """Initialises a graph object (based on graph-tool, networkx, or cugraph (TODO)). Produces a Network object.

        Args:
            ref_list (list): List of sequence names/identifiers (names/identifiers should be strings) which will be vertices in the graph
            outdir (str): Path to output directory where graph files will be stored. Defaults to "./" (i.e. working directory)
            backend (str, optional): Which graphing module to use. Can be specified here (valid options: "GT", "NX", "CU") or as an environment variable.

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

            Graphing backend can alternatively be set as follows:
            ```
            os.environ["GRAPH_BACKEND"] = "GT" ## inside of your_script.py or in interactive python terminal

            *OR*

            export GRAPH_BACKEND=GT ## inside bash/shell terminal/environment
            ```
        """
        ## initialising class attributes
        self.outdir = outdir # directory to which all outputs are written

        self.ref_list = ref_list # list of sample names
        self.graph = None # empty graph
        self.ref_graph = None # empty pruned graph
        self.mst_network = None # empty mst
        
        ## if backend not provided when Network is initialised, try to read backend from env
        ## if backend is provided during init AND set as env var, value provided during init will be used.
        if backend is None:
            self.backend = os.getenv("GRAPH_BACKEND") ## read os env variable "GRAPH_BACKEND"
        else:
            self.backend = backend ## else use backend if specified

        ## if no backend provided in either way, try to import graphtool first, then networkx, and quit if neither found (TODO: is this a good idea?)
        if backend is None and os.getenv("GRAPH_BACKEND") is None:
            print("No graph module specified, checking for available modules...")
            try:
                import graph_tool.all as gt
                self.gt = gt
                print("Using graph-tool...")
                self.backend = "GT"
            except ImportError:
                print("Graph-tool not found...")

                try:
                    import networkx as nx
                    self.nx = nx
                    print("Using networkx...")
                    self.backend = "NX"
                except ImportError as ie:
                    print("Networkx not found...\nQuitting.\n")
                    raise ie

        ## perform imports
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

    def construct(self, network_data, weights = None):
        """Method called on Network object. Constructs a graph using either graph-tool, networkx, or cugraph(TODO)

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

        ## clean up problem characters in sample names, store as vertex_labels attribute
        vertex_labels = [sample.replace('.','_').replace(':','').replace('(','_').replace(')','_') for sample in self.ref_list]
        self.vertex_labels = vertex_labels
        self.weights = weights

        ## initialise a graph object
        if self.backend == "GT":
            self.graph = construct_with_graphtool(network_data=network_data, vertex_labels=self.vertex_labels, weights=weights)
        elif self.backend == "NX":
            self.graph = construct_with_networkx(network_data=network_data, vertex_labels=self.vertex_labels, weights=weights)
        elif self.backend == "CU":
            raise NotImplementedError("GPU graph not yet implemented")

        prepare_graph(self.graph, backend = self.backend, labels = self.vertex_labels) # call to prepare_graph to add component_membership and edge weight if required

    def prune(self, type_isolate = None, threads = 4):
        """Method to prune full graph and produce a reference graph

        Args:
            type_isolate (str, optional): Sample name of type isolate, as calcualted by poppunk. Defaults to None.
            threads (int, optional): Number of threads to use when pruuning. Defaults to 4.
        """

        if self.graph is None:
            raise RuntimeError("Graph not constructed or loaded.")

        if self.backend == "GT":
            ## get pruning fns
            from pp_netlib.gt_prune import gt_clique_prune, gt_get_ref_graph

            ## set up iterables
            reference_vertices = set()
            components = self.gt.label_components(self.graph)[0].a

            ## run prune in parallel
            with Pool(threads) as pool:
                ref_lists = pool.map(partial(gt_clique_prune, graph=self.graph, reference_indices=set(), components_list=components), set(components))

            ## flatten prune results
            reference_vertices = set([entry for sublist in ref_lists for entry in sublist])

            ## create pruned ref graph
            labels = list(self.graph.vp["id"][v] for v in self.graph.vertices())
            self.ref_graph = gt_get_ref_graph(self.graph, reference_vertices, labels, type_isolate)

            num_nodes = self.ref_graph.num_vertices()
            num_edges = self.ref_graph.num_edges()

        if self.backend == "NX":
            ## get pruning fns
            from pp_netlib.nx_prune import nx_get_clique_refs, nx_get_connected_refs

            ## set up iterable over subgraphs
            subgraphs = [self.graph.subgraph(c) for c in self.nx.connected_components(self.graph)]

            ## run prune in parallel
            with Pool(threads) as pool:
                ref_lists = pool.map(partial(nx_get_clique_refs, references=set()), subgraphs)

            ## flatten prune results
            reference_vertices = set([entry for sublist in ref_lists for entry in sublist])

            ## get shortest paths between prune results so that component assignments match the original
            updated_refs = nx_get_connected_refs(self.graph, reference_vertices)

            ## add type_isolate to pruned list of samples
            type_idx = [i[0] for i in list(self.graph.nodes(data="id")) if i[1] == type_isolate]
            updated_refs.add(type_idx[0])

            ## create pruned ref graph
            self.ref_graph = self.graph.copy()
            self.ref_graph.remove_nodes_from([node for node in self.graph.nodes() if node not in updated_refs])

            num_nodes = self.ref_graph.number_of_nodes()
            num_edges = self.ref_graph.number_of_edges()

        sys.stderr.write(f"Pruned network has {num_nodes} nodes and {num_edges} edges.\n\n")

    def get_summary(self, print_to_std = True, summary_file_prefix = None):
        """Method called on initialised and populated Network object. Prints summary of network properties to stderr and optionally to plain text file.

        Args:
            summary_file_prefix (str, optional): File name of summary file to which graph property summaries should be written.
            Defaults to None. If None, no summary file is written.

        Raises:
            RuntimeError: RuntimeError raised if no graph initialised.
        """
        
        if self.graph is None:
            raise RuntimeError("Graph not constructed or loaded.")

        ## store metrics and stores as attributes
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

        ## print some summaries
        if print_to_std:
            sys.stderr.write(f"{summary_contents}\n\n")

        ##  write summary to plain text
        if summary_file_prefix is not None:
            summary_file_name = os.path.join(self.outdir, (str(summary_file_prefix) + ".txt"))

            with open(summary_file_name, "w") as summary:
                summary.write(summary_contents)
            summary.close()

    def visualize(self, viz_type, out_prefix, external_data = None):
        """Generate visualisations from graph.

        Args:
            viz_type (str): can be either "mst" or "cytoscape". If "mst", minimum-spanning tree visualisations are produced as .png files, mst saved as .graphml. 
                    If "cytoscape", the graph, individual components, and minimum spanning tree are saved as .graphml and a csv file containing cluster info is also produced.
            out_prefix (str): the prefix to apply to all output files (not a path)
            external_data (path/str *OR* pd.DataFrame, optional): Epidemiological or other data to be associated with each sample. Defaults to None.
        """
        from pp_netlib.functions import generate_mst_network, save_graph

        if viz_type == "mst":
            self.mst_network = generate_mst_network(self.graph, self.backend)
            mst_outdir = os.path.join(self.outdir, "mst")
            if not os.path.exists(mst_outdir):
                os.makedirs(mst_outdir)

            file_name = out_prefix+"_mst_network_data"
            save_graph(self.mst_network, backend = self.backend, outdir = mst_outdir, file_name = file_name, file_format = ".graphml")

            ## visualise minimum-spanning tree
            if self.backend == "GT":
                from pp_netlib.functions import draw_gt_mst, get_gt_clusters

                isolate_clustering = get_gt_clusters(self.graph)
                draw_gt_mst(mst = self.mst_network, out_prefix = os.path.join(mst_outdir, out_prefix), isolate_clustering=isolate_clustering, overwrite=True)
            
            if self.backend == "NX":
                from pp_netlib.functions import draw_nx_mst, get_nx_clusters

                isolate_clustering = get_nx_clusters(self.graph)
                draw_nx_mst(mst=self.mst_network, out_prefix=os.path.join(mst_outdir, out_prefix), isolate_clustering=isolate_clustering, overwrite=True)

        if viz_type == "cytoscape":
            from pp_netlib.functions import write_cytoscape_csv
        
            ## set up output directory as {self.outdir}/cytoscape; create it if it does not exist
            cytoscape_outdir = os.path.join(self.outdir, "cytoscape")
            if not os.path.exists(cytoscape_outdir):
                os.makedirs(cytoscape_outdir)

            ## save mst as out_prefix_mst.graphml, save full graph as out_prefix_cytoscape.graphml
            save_graph(self.graph, self.backend, cytoscape_outdir, out_prefix+"_cytoscape", ".graphml")
            if self.mst_network is not None:
                save_graph(self.mst_network, self.backend, cytoscape_outdir, out_prefix+"_mst", ".graphml")

            ## call functions to save individual graph components and write csv for cytoscape
            if self.backend == "GT":
                from pp_netlib.functions import gt_save_graph_components, get_gt_clusters
                gt_save_graph_components(self.graph, out_prefix, cytoscape_outdir)
                clustering = get_gt_clusters(self.graph)

                write_cytoscape_csv(os.path.join(cytoscape_outdir, out_prefix+".csv"), clustering.keys(), clustering, external_data)

            if self.backend == "NX":
                from pp_netlib.functions import nx_save_graph_components, get_nx_clusters
                nx_save_graph_components(self.graph, out_prefix, cytoscape_outdir)
                clustering = get_nx_clusters(self.graph)

                write_cytoscape_csv(os.path.join(cytoscape_outdir, out_prefix+".csv"), clustering.keys(), clustering, external_data)

    def load_network(self, network_file):
        """Load a premade graph from a network file.

        Args:
            network_file (str/path): The file in which the prebuilt graph is stored. Must be a .gt file, .graphml file or .xml file.
        """
        if self.graph is not None:
            raise RuntimeError("Network instance already contains a graph. Cannot load another graph.\n\n")
        
        # Load the network from the specified file
        file_name, file_extension = os.path.splitext(network_file)

        if file_extension == ".gt":
            self.graph = self.gt.load_graph(network_file)
            num_nodes = len(list(self.graph.vertices()))
            num_edges = len(list(self.graph.edges()))
            sys.stderr.write(f"Loaded network with {num_nodes} nodes and {num_edges} edges with {self.backend}.\n\n")

            ## NX backend cannot open .gt file
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

        else:
            raise RuntimeError("File format not recognised.\n\n")

        prepare_graph(self.graph, backend = self.backend) # call to prepare_graph to add component_membership and edge weight if required

    def add_to_network(self, new_data_df, new_vertex_labels):
        """Add data to network

        Args:
            new_data_df (pd.DataFrame): Dataframe of edges with source, destination and weights columns
            new_vertex_labels (list): List of new sample names to add to graph
        """

        if self.graph is None:
            raise RuntimeError("No network found, cannot add data. Please load a network to update or construct a network with this data.\n\n")

        new_vertex_labels = [new_sample.replace('.','_').replace(':','').replace('(','_').replace(')','_') for new_sample in new_vertex_labels]

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

        ## update vertex labels attribute
        self.vertex_labels += new_vertex_labels

    def write_metadata(self, meta_outdir, out_prefix, external_data = None):
        
        if meta_outdir is None:
            meta_outdir = self.outdir

        if self.backend == "GT":
            from pp_netlib.functions import gt_get_graph_data
            self.edge_data, self.sample_metadata = gt_get_graph_data(self.graph)

        if self.backend == "NX":
            from pp_netlib.functions import nx_get_graph_data
            self.edge_data, self.sample_metadata = nx_get_graph_data(self.graph)

        
        edge_data = pd.DataFrame.from_dict(self.edge_data, orient="index")
        if len(edge_data.columns) == 2:
            edge_data.columns = ["source", "target"]
        else:
            edge_data.columns = ["source", "target", "weight"]

        edge_data.to_csv(os.path.join(meta_outdir, out_prefix+"_edge_data.tsv"), sep="\t", index=False)
        
        if external_data is None:
            pd.DataFrame.from_dict(self.sample_metadata, orient="index", columns=["sample_id", "sample_component"]).to_csv(os.path.join(meta_outdir, out_prefix+"_node_data.tsv"), sep="\t", index=False)

        else:
            node_data_df = pd.DataFrame.from_dict(self.sample_metadata, orient="index", columns=["sample_id", "sample_component"])
            if isinstance(external_data, str):
                external_data_df = pd.read_csv(external_data, sep="\t", header=0)
                sample_metadata = pd.merge(node_data_df, external_data_df, on="sample_id")
            elif isinstance(external_data, pd.DataFrame):
                sample_metadata = pd.merge(node_data_df, external_data, on="sample_id")

            sample_metadata.to_csv(os.path.join(meta_outdir, out_prefix), sep="\t", index=False)

    def save(self, file_name, file_format, to_save=None):
        """Save graph to file.

        Args:
            file_name (str): Name to be given to the graph file
            file_format (str): File extenstion to be used with graph file
            to_save (str): Which graph to save. Allowed values are "full_graph", "pruned_graph", "both"

            Example:
            ```
            graph.save("sample_graph", ".graphml")
            ```

        Raises:
            NotImplementedError: If graph_tool is selected a backend,
        """
        if self.graph is None:
            raise RuntimeError("Graph not constructed or loaded.")

        if self.ref_graph is None and to_save == "both":
            sys.stderr.write("Pruned graph not found, only saving full graph.\n")
            to_save = "full_graph"

        save_graph(graph=self.graph, backend=self.backend, outdir = self.outdir, file_name=file_name, file_format=file_format)
        
        if to_save == "full_graph" or to_save == "both":
            save_graph(graph=self.graph, backend=self.backend, outdir = self.outdir, file_name=file_name, file_format=file_format)
        if to_save == "ref_graph" or to_save == "both":
            save_graph(graph=self.ref_graph, backend=self.backend, outdir = self.outdir, file_name=file_name+".pruned", file_format=file_format)

