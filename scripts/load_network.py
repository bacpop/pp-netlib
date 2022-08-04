import os, sys
import pickle

import graph_tool.all as gt

import cudf
import cugraph

from .vertices import *
from .utils import *

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

#TODO: refactor
def fetch_network(network_dir, model, ref_list, ref_graph = False, core_only = False, accessory_only = False, use_gpu = False):
    """Load the network based on input options
       Returns the network as a graph-tool format graph, and sets
       the slope parameter of the passed model object.

       ## DEPENDS ON Fns: {.analyse_network: [check_network_vertex_count], .: load_network_file, poppunk.utils: [check_and_set_gpu]}

       Args:
            network_dir (str)
                A network used to define clusters
            model (ClusterFit)
                A fitted model object
            ref_list (list)
                Names of references that should be in the network
            ref_graph (bool)
                Use ref only graph, if available
                [default = False]
            core_only (bool)
                Return the network created using only core distances
                [default = False]
            accessory_only (bool)
                Return the network created using only accessory distances
                [default = False]
            use_gpu (bool)
                Use cugraph library to load graph
       Returns:
            genomeNetwork (graph)
                The loaded network
            cluster_file (str)
                The CSV of cluster assignments corresponding to this network
    """
    # If a refined fit, may use just core or accessory distances
    dir_prefix = network_dir + "/" + os.path.basename(network_dir)

    # load CUDA libraries - here exit without switching to CPU libraries
    # to avoid loading an unexpected file
    use_gpu = check_and_set_gpu(use_gpu, gpu_lib, quit_on_fail = True)

    if use_gpu:
        graph_suffix = ".csv.gz"
    else:
        graph_suffix = ".gt"

    if core_only and model.type == "refine":
        if ref_graph:
            network_file = dir_prefix + "_core.refs_graph" + graph_suffix
        else:
            network_file = dir_prefix + "_core_graph" + graph_suffix
        cluster_file = dir_prefix + "_core_clusters.csv"
    elif accessory_only and model.type == "refine":
        if ref_graph:
            network_file = dir_prefix + "_accessory.refs_graph" + graph_suffix
        else:
            network_file = dir_prefix + "_accessory_graph" + graph_suffix
        cluster_file = dir_prefix + "_accessory_clusters.csv"
    else:
        if ref_graph and os.path.isfile(dir_prefix + ".refs_graph" + graph_suffix):
            network_file = dir_prefix + ".refs_graph" + graph_suffix
        else:
            network_file = dir_prefix + "_graph" + graph_suffix
        cluster_file = dir_prefix + "_clusters.csv"
        if core_only or accessory_only:
            sys.stderr.write("Can only do --core or --accessory fits from a refined fit. Using the combined distances.\n")

    # Load network file
    sys.stderr.write("Loading network from " + network_file + "\n")
    genome_network = load_network_file(network_file, use_gpu = use_gpu)

    # Ensure all in dists are in final network
    check_network_vertex_count(ref_list, genome_network, use_gpu)

    return genome_network, cluster_file

def save_network(graph, prefix = None, suffix = None, use_graphml = False, use_gpu = False):
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
    if not use_gpu:
        if use_graphml:
            graph.save(file_name + ".graphml", fmt = "graphml")
        else:
            graph.save(file_name + ".gt", fmt = "gt")
    else:
        graph.to_pandas_edgelist().to_csv(file_name + ".csv.gz", compression="gzip", index = False)
