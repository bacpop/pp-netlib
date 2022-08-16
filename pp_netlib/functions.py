#### FUNCTIONS FOR Graph.construct ####

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

def convert_data_to_df(network_data, weights:(bool or list), use_gpu:bool):
    if isinstance(network_data, np.array) or isinstance(network_data, scipy.sparse.coo_matrix):
        if not use_gpu:
            graph_data_df = pd.DataFrame()
        else:
            graph_data_df = cudf.DataFrame()
        graph_data_df["source"] = network_data.row
        graph_data_df["destination"] =  network_data.col
        
        graph_data_df["weights"] = network_data.data

        return graph_data_df

    elif isinstance(network_data, pd.DataFrame) or isinstance(network_data, cudf.DataFrame):
        if weights:
            network_data.columns = ["source","destination","weights"]
        elif isinstance(weights, list):
            network_data.columns = ["source","destination"]
            network_data["weights"] = weights
        else:
            network_data.columns = ["source","destination"]

        return network_data

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

