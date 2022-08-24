########################
####   .CONSTRUCT   ####
########################
def initial_graph_properties(ref_list, query_list):
    """Initial processing of sequence names for
    network construction.

    ## DEPENDS ON Fns: {None}

    Args:
        ref_list (list)
            List of reference sequence labels
        query_list (list)
            List of query sequence labels
    Returns:
        vertex_labels (list)
            Ordered list of sequences in network
        self_comparison (bool)
            Whether the network is being constructed from all-v-all distances or
            reference-v-query information
    """
    if ref_list == query_list:
        self_comparison = True
        vertex_labels = ref_list
    else:
        self_comparison = False
        vertex_labels = ref_list +  query_list
    return vertex_labels, self_comparison

def load_network_file(network_file, use_gpu = False):
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
            genomeNetwork (graph)
                The loaded network
    """
    # Load the network from the specified file
    if not use_gpu:
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

def network_to_edges(prev_G_fn, rlist, adding_qq_dists = False, old_ids = None, previous_pkl = None, weights = False, use_gpu = False):
    """Load previous network, extract the edges to match the
    vertex order specified in rlist, and also return weights if specified.

    ## DEPENDS ON Fns: {.: [load_network_file]}

    Args:
        prev_G_fn (str or graph object)
            Path of file containing existing network, or already-loaded
            graph object
        adding_qq_dists (bool)
            Boolean specifying whether query-query edges are being added
            to an existing network, such that not all the sequence IDs will
            be found in the old IDs, which should already be correctly ordered
        rlist (list)
            List of reference sequence labels in new network
        old_ids (list)
            List of IDs of vertices in existing network
        previous_pkl (str)
            Path of pkl file containing names of sequences in
            previous network
        weights (bool)
            Whether to return edge weights
            (default = False)
        use_gpu (bool)
            Whether to use cugraph for graph analyses
    Returns:
        source_ids (list)
            Source nodes for each edge
        target_ids (list)
            Target nodes for each edge
        edge_weights (list)
            Weights for each new edge
    """
    # Load graph from file if passed string; else use graph object passed in
    # as argument
    if isinstance(prev_G_fn, str):
        prev_graph = load_network_file(prev_G_fn, use_gpu = use_gpu)
    else:
        prev_graph = prev_G_fn

    # load list of names in previous network if pkl name supplied
    if previous_pkl is not None:
        with open(previous_pkl, 'rb') as pickle_file:
            old_rlist, old_qlist, self = pickle.load(pickle_file)
        if self:
            old_ids = old_rlist
        else:
            old_ids = old_rlist + old_qlist
    elif old_ids is None:
        sys.stderr.write('Missing .pkl file containing names of sequences in '
                         'previous network\n')
        sys.exit(1)

    # Get edges as lists of source,destination,weight using original IDs
    if not use_gpu:
        # get the source and target nodes
        old_source_ids = gt.edge_endpoint_property(prev_graph, prev_graph.vertex_index, "source")
        old_target_ids = gt.edge_endpoint_property(prev_graph, prev_graph.vertex_index, "target")
        # get the weights
        if weights:
            if prev_graph.edge_properties.keys() is None or "weight" not in prev_graph.edge_properties.keys():
                sys.stderr.write("Loaded network does not have edge weights; try a different "
                                    "network or turn off graph weights\n")
                exit(1)
            edge_weights = list(prev_graph.ep["weight"])
    else:
        G_df = prev_graph.view_edge_list()
        if weights:
            if len(G_df.columns) < 3:
                sys.stderr.write("Loaded network does not have edge weights; try a different "
                                    "network or turn off graph weights\n")
                exit(1)
            if "src" in G_df.columns:
                G_df.rename(columns={"source": "src","destination": "dst"}, inplace=True)
            edge_weights = G_df["weights"].to_arrow().to_pylist()
        G_df.rename(columns={"src": "source","dst": "destination"}, inplace=True)
        old_source_ids = G_df["source"].astype("int32").to_arrow().to_pylist()
        old_target_ids = G_df["destination"].astype("int32").to_arrow().to_pylist()

    # If appending queries to an existing network, then the recovered links can be left
    # unchanged, as the new IDs are the queries, and the existing sequences will not be found
    # in the list of IDs
    if adding_qq_dists:
        source_ids = old_source_ids
        target_ids = old_target_ids
    else:
        # Update IDs to new versions
        old_id_indices = [rlist.index(x) for x in old_ids]
        # translate to indices
        source_ids = [old_id_indices[x] for x in old_source_ids]
        target_ids = [old_id_indices[x] for x in old_target_ids]

    # return values
    if weights:
        return source_ids, target_ids, edge_weights
    else:
        return source_ids, target_ids

def process_previous_network(previous_network = None, adding_qq_dists = False, old_ids = None, previous_pkl = None, vertex_labels = None, weights = False, use_gpu = False):
    """Extract edge types from an existing network

    ## DEPENDS ON Fns: {.load_network: [network_to_edges]}

    Args:
        previous_network (str or graph object)
            Name of file containing a previous network to be integrated into this new
            network, or already-loaded graph object
        adding_qq_dists (bool)
            Boolean specifying whether query-query edges are being added
            to an existing network, such that not all the sequence IDs will
            be found in the old IDs, which should already be correctly ordered
        old_ids (list)
            Ordered list of vertex names in previous network
        previous_pkl (str)
            Name of file containing the names of the sequences in the previous_network
            ordered based on the original network construction
        vertex_labels (list)
            Ordered list of sequence labels
        weights (bool)
            Whether weights should be extracted from the previous network
        use_gpu (bool)
            Whether to use GPUs for network construction
    Returns:
        extra_sources (list)
            List of source node identifiers
        extra_targets (list)
            List of destination node identifiers
        extra_weights (list or None)
            List of edge weights
    """
    if previous_pkl is not None or old_ids is not None:
        if weights:
            # Extract from network
            extra_sources, extra_targets, extra_weights = network_to_edges(previous_network, vertex_labels, adding_qq_dists = adding_qq_dists, old_ids = old_ids, previous_pkl = previous_pkl, weights = True, use_gpu = use_gpu)
        else:
            # Extract from network
            extra_sources, extra_targets = network_to_edges(previous_network, vertex_labels, adding_qq_dists = adding_qq_dists, old_ids = old_ids, previous_pkl = previous_pkl, weights = False, use_gpu = use_gpu)
            extra_weights = None
    else:
        sys.stderr.write("A distance pkl corresponding to " + previous_pkl + " is required for loading\n")
        sys.exit(1)

    return extra_sources, extra_targets, extra_weights

def read_in_previous_network(previous_network: str or object, vprops: None, eprops: None):
    if isinstance(previous_network, str):
        prev_graph = gt.Graph()
        prev_graph.load(previous_network)
    else:
        prev_graph = previous_network

    if vprops is not None:
        vprop_dict = {}
        for v in prev_graph.vertices():
            for vprop in vprops:
                vprop_dict[prev_graph.vertex_index[v]] = (vprop, prev_graph.vertex_properties[vprop][v])

    ## get edge lists
    edge_data = []
    # sources = gt.edge_endpoint_property(prev_graph, prev_graph.vertex_index, "source")
    # targets = gt.edge_endpoint_property(prev_graph, prev_graph.vertex_index, "target")
    edge_data.append(gt.edge_endpoint_property(prev_graph, prev_graph.vertex_index, "source"))
    edge_data.append(gt.edge_endpoint_property(prev_graph, prev_graph.vertex_index, "target"))
    
    if eprops is not None:
        for eprop in eprops:
            edge_data.append([prev_graph.edge_properties[eprop][edge] for edge in prev_graph.edges()])

    prev_edge_list = list(zip(*edge_data))

    


    return prev_edge_list, vprop_dict
