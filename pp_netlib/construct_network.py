import pandas as pd
import numpy as np
#from numba import cuda
import graph_tool.all as gt

#import cudf
#import cupy as cp

from .utils import *
from .load_network import *
from .indices_refs_clusters import *

import poppunk_refine

def get_cugraph_triangles(graph):
    """Counts the number of triangles in a cugraph
    network. Can be removed when the cugraph issue
    https://github.com/rapidsai/cugraph/issues/1043 is fixed.

    ## DEPENDS ON Fns: {none}

    Args:
        graph (cugraph network)
            Network to be analysed
    Returns:
        triangle_count (int)
            Count of triangles in graph
    """
    num_vertices = graph.number_of_vertices()
    edge_df = graph.view_edge_list()
    A = cp.full((num_vertices, num_vertices), 0, dtype = cp.int32)
    A[edge_df.src.values, edge_df.dst.values] = 1
    A = cp.maximum( A, A.transpose() )
    triangle_count = int(cp.around(cp.trace(cp.matmul(A, cp.matmul(A, A)))/6,0))
    return triangle_count

#TODO: can fn network_summary and fn print_network summary be combined?
def print_network_summary(graph, betweenness_sample = betweenness_sample_default, use_gpu = False):
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
    (metrics, scores) = network_summary(graph, betweenness_sample = betweenness_sample, use_gpu = use_gpu)
    sys.stderr.write("Network summary:\n" + "\n".join(["\tComponents\t\t\t\t" + str(metrics[0]),
                                                   "\tDensity\t\t\t\t\t" + "{:.4f}".format(metrics[1]),
                                                   "\tTransitivity\t\t\t\t" + "{:.4f}".format(metrics[2]),
                                                   "\tMean betweenness\t\t\t" + "{:.4f}".format(metrics[3]),
                                                   "\tWeighted-mean betweenness\t\t" + "{:.4f}".format(metrics[4]),
                                                   "\tScore\t\t\t\t\t" + "{:.4f}".format(scores[0]),
                                                   "\tScore (w/ betweenness)\t\t\t" + "{:.4f}".format(scores[1]),
                                                   "\tScore (w/ weighted-betweenness)\t\t" + "{:.4f}".format(scores[2])])
                                                   + "\n")

def process_weights(dist_matrix, weights_type):
    """Calculate edge weights from the distance matrix

    ## DEPENDS ON Fns: {none}

    Args:
        dist_matrix (2 column ndarray)
            Numpy array of pairwise distances
        weights_type (str)
            Measure to calculate from the distMat to use as edge weights in network
            - options are core, accessory or euclidean distance
    Returns:
        processed_weights (list)
            Edge weights
    """
    processed_weights = []
    if weights_type is not None and dist_matrix is not None:
        # Check weights type is valid
        if weights_type not in accepted_weights_types:
            sys.stderr.write("Unable to calculate distance type " + str(weights_type) + "; "
                             "accepted types are " + str(accepted_weights_types) + "\n")
        if weights_type == "euclidean":
            processed_weights = np.linalg.norm(dist_matrix, axis = 1).tolist()
        elif weights_type == "core":
            processed_weights = dist_matrix[:, 0].tolist()
        elif weights_type == "accessory":
            processed_weights = dist_matrix[:, 1].tolist()
    else:
        sys.stderr.write("Require distance matrix to calculate distances\n")
    return processed_weights

def construct_network_from_edge_list(ref_list, query_list, edge_list, weights = None, dist_matrix = None, previous_network = None, adding_qq_dists = False, old_ids = None, previous_pkl = None, betweenness_sample = betweenness_sample_default, summarise = True, use_gpu = False):
    """Construct an undirected network using a data frame of edges. Nodes are samples and
    edges where samples are within the same cluster
    Will print summary statistics about the network to ``STDERR``

    ## DEPENDS ON Fns: {poppunk.utils: [check_and_set_gpu], .: [construct_network_from_df], .analyse_network: [process_previous_network], .network_io: [print_network_summary]}

    Args:
        ref_list (list)
            List of reference sequence labels
        query_list (list)
            List of query sequence labels
        edge_list (cudf or pandas data frame) ####TODO UNUSED???
            Data frame in which the first two columns are the nodes linked by edges
        weights (list)
            List of edge weights
        dist_matrix (2 column ndarray)
            Numpy array of pairwise distances
        previous_network (str or graph object)
            Name of file containing a previous network to be integrated into this new
            network, or the already-loaded graph object
        adding_qq_dists (bool)
            Boolean specifying whether query-query edges are being added
            to an existing network, such that not all the sequence IDs will
            be found in the old IDs, which should already be correctly ordered
        old_ids (list)
            Ordered list of vertex names in previous network
        previous_pkl (str)
            Name of file containing the names of the sequences in the previous_network
        betweenness_sample (int)
            Number of sequences per component used to estimate betweenness using
            a GPU. Smaller numbers are faster but less precise [default = 100]
        summarise (bool)
            Whether to calculate and print network summaries with :func:`~networkSummary`
            (default = True)
        use_gpu (bool)
            Whether to use GPUs for network construction
    Returns:
        graph (graph)
            The resulting network
    """

    # Check GPU library use
    use_gpu = check_and_set_gpu(use_gpu, gpu_lib, quit_on_fail = True)

    # data structures
    vertex_labels, self_comparison = initial_graph_properties(ref_list, query_list)

    # Create new network
    if not use_gpu:
        # Load previous network
        if previous_network is not None:
            extra_sources, extra_targets, extra_weights = process_previous_network(previous_network = previous_network, adding_qq_dists = adding_qq_dists, old_ids = old_ids, previous_pkl = previous_pkl, vertex_labels = vertex_labels, weights = (weights is not None), use_gpu = use_gpu)
        # Construct list of tuples for graph-tool
        # Include information from previous graph if supplied
        if weights is not None:
            weighted_edges = []
            for ((src, dest), weight) in zip(edge_list, weights):
                weighted_edges.append((src, dest, weight))
            if previous_network is not None:
                for (src, dest, weight) in zip(extra_sources, extra_targets, extra_weights):
                    weighted_edges.append((src, dest, weight))
            edge_list = weighted_edges
        else:
            if previous_network is not None:
                for (src, dest) in zip(extra_sources, extra_targets):
                    edge_list.append((src, dest))
        # build the graph
        graph = gt.Graph(directed = False)
        graph.add_vertex(len(vertex_labels))
        if weights is not None:
            eweight = graph.new_ep("float")
            graph.add_edge_list(edge_list, eprops = [eweight])
            graph.edge_properties["weight"] = eweight
        else:
            graph.add_edge_list(edge_list)

    else:
        # benchmarking concurs with https://stackoverflow.com/questions/55922162/recommended-cudf-dataframe-construction
        if len(edge_list) > 1:
            edge_array = cp.array(edge_list, dtype = np.int32)
            edge_gpu_matrix = cuda.to_device(edge_array)
            G_df = cudf.DataFrame(edge_gpu_matrix, columns = ['source','destination'])
        else:
            # Cannot generate an array when one edge
            G_df = cudf.DataFrame(columns = ['source','destination'])
            G_df['source'] = [edge_list[0][0]]
            G_df['destination'] = [edge_list[0][1]]
        if weights is not None:
            G_df['weights'] = weights
        graph = construct_network_from_df(ref_list, query_list, G_df, weights = (weights is not None), dist_matrix = dist_matrix, adding_qq_dists = adding_qq_dists, old_ids = old_ids, previous_network = previous_network, previous_pkl = previous_pkl, summarise = False, use_gpu = use_gpu)

    if summarise:
        print_network_summary(graph, betweenness_sample = betweenness_sample, use_gpu = use_gpu)

    return graph

def construct_network_from_df(ref_list, query_list, G_df, weights = False, dist_matrix = None, previous_network = None, adding_qq_dists = False, old_ids = None, previous_pkl = None, betweenness_sample = betweenness_sample_default, summarise = True, use_gpu = False):
    """Construct an undirected network using a data frame of edges. Nodes are samples and
    edges where samples are within the same cluster
    Will print summary statistics about the network to ``STDERR``

    ## DEPENDS ON Fns: {.utils: [check_and_set_gpu], .: [counstruct_network_from_edge_list, print_network_summary], .: [process_previous network], .indices_refs_clusters: [add_self_loop]}

    Args:
        ref_list (list)
            List of reference sequence labels
        query_list (list)
            List of query sequence labels
        G_df (cudf or pandas data frame)
            Data frame in which the first two columns are the nodes linked by edges
        weights (bool)
            Whether weights in the G_df data frame should be included in the network
        dist_matrix (2 column ndarray)
            Numpy array of pairwise distances
        previous_network (str or graph object)
            Name of file containing a previous network to be integrated into this new
            network, or the already-loaded graph object
        adding_qq_dists (bool)
            Boolean specifying whether query-query edges are being added
            to an existing network, such that not all the sequence IDs will
            be found in the old IDs, which should already be correctly ordered
        old_ids (list)
            Ordered list of vertex names in previous network
        previous_pkl (str)
            Name of file containing the names of the sequences in the previous_network
        betweenness_sample (int)
            Number of sequences per component used to estimate betweenness using
            a GPU. Smaller numbers are faster but less precise [default = 100]
        summarise (bool)
            Whether to calculate and print network summaries with :func:`~networkSummary`
            (default = True)
        use_gpu (bool)
            Whether to use GPUs for network construction
    Returns:
        graph (graph)
            The resulting network
    """

    # Check GPU library use
    use_gpu = check_and_set_gpu(use_gpu, gpu_lib, quit_on_fail = True)

    # data structures
    vertex_labels, self_comparison = initial_graph_properties(ref_list, query_list)

    # Check df format is correct
    if weights:
        G_df.columns = ['source','destination','weights']
    else:
        G_df.columns = ['source','destination']

    # Load previous network
    if previous_network is not None:
        extra_sources, extra_targets, extra_weights = process_previous_network(previous_network = previous_network,
                                                                                adding_qq_dists = adding_qq_dists,
                                                                                old_ids = old_ids,
                                                                                previous_pkl = previous_pkl,
                                                                                vertex_labels = vertex_labels,
                                                                                weights = weights,
                                                                                use_gpu = use_gpu)
        if use_gpu:
            G_extra_df = cudf.DataFrame()
        else:
            G_extra_df = pd.DataFrame()
        G_extra_df['source'] = extra_sources
        G_extra_df['destination'] = extra_targets
        if extra_weights is not None:
            G_extra_df['weights'] = extra_weights
        G_df = cudf.concat([G_df,G_extra_df], ignore_index = True)

    if use_gpu:
        # direct conversion
        # ensure the highest-integer node is included in the edge list
        # by adding a self-loop if necessary; see https://github.com/rapidsai/cugraph/issues/1206
        max_in_vertex_labels = len(vertex_labels)-1
        use_weights = False
        if weights:
            use_weights = True
        graph = add_self_loop(G_df, max_in_vertex_labels, weights = use_weights, renumber = False)
    else:
        # Convert bool to list of weights or None
        if weights:
            weights = G_df['weights']
        else:
            weights = None
        # Convert data frame to list of tuples
        connections = list(zip(*[G_df[c].values.tolist() for c in G_df[['source','destination']]]))
        graph = construct_network_from_edge_list(ref_list, query_list, connections,
                                            weights = weights,
                                            distMat = dist_matrix,
                                            previous_network = previous_network,
                                            old_ids = old_ids,
                                            previous_pkl = previous_pkl,
                                            summarise = False,
                                            use_gpu = use_gpu)
    if summarise:
        print_network_summary(graph, betweenness_sample = betweenness_sample, use_gpu = use_gpu)
    return graph

def construct_network_from_sparse_matrix(ref_list, query_list, sparse_input, weights = None, previous_network = None, previous_pkl = None, betweenness_sample = betweenness_sample_default, summarise = True, use_gpu = False):
    """Construct an undirected network using a sparse matrix. Nodes are samples and
    edges where samples are within the same cluster
    Will print summary statistics about the network to ``STDERR``

    ## DEPENDS ON Fns: {.utils: [check_and_set_gpu], .:[construct_network_from_df, print_network_summary]}

    Args:
        ref_listlist (list)
            List of reference sequence labels
        query_list (list)
            List of query sequence labels
        sparse_input (numpy.array)
            Sparse distance matrix from lineage fit
        weights (list)
            List of weights for each edge in the network
        distMat (2 column ndarray)
            Numpy array of pairwise distances
        previous_network (str)
            Name of file containing a previous network to be integrated into this new
            network
        previous_pkl (str)
            Name of file containing the names of the sequences in the previous_network
        betweenness_sample (int)
            Number of sequences per component used to estimate betweenness using
            a GPU. Smaller numbers are faster but less precise [default = 100]
        summarise (bool)
            Whether to calculate and print network summaries with :func:`~networkSummary`
            (default = True)
        use_gpu (bool)
            Whether to use GPUs for network construction
    Returns:
        graph (graph)
            The resulting network
    """

    # Check GPU library use
    use_gpu = check_and_set_gpu(use_gpu, gpu_lib, quit_on_fail = True)

    if not use_gpu:
        G_df = pd.DataFrame()
    else:
        G_df = cudf.DataFrame()
    G_df['source'] = sparse_input.row
    G_df['destination'] =  sparse_input.col
    G_df['weights'] = sparse_input.data
    graph = construct_network_from_df(ref_list, query_list, G_df, weights = True, previous_network = previous_network, previous_pkl = previous_pkl, betweenness_sample = betweenness_sample, summarise = False, use_gpu = use_gpu)
    if summarise:
        print_network_summary(graph, betweenness_sample = betweenness_sample, use_gpu = use_gpu)
    return graph

def construct_dense_weighted_network(ref_list, dist_matrix, weights_type = None, use_gpu = False):
    """Construct an undirected network using sequence lists, assignments of pairwise distances
    to clusters, and the identifier of the cluster assigned to within-strain distances.
    Nodes are samples and edges where samples are within the same cluster
    Will print summary statistics about the network to ``STDERR``

    ## DEPENDS ON Fns: {.utils: [check_and_set_gpu], .:[initial_graph_properties, process_weights], poppunk_refine: [generateAllTuples]}

    Args:
        ref_list (list)
            List of reference sequence labels
        dist_matrix (2 column ndarray)
            Numpy array of pairwise distances
        weights_type (str)
            Type of weight to use for network
        use_gpu (bool)
            Whether to use GPUs for network construction
    Returns:
        graph (graph)
            The resulting network
    """
    # Check GPU library use
    use_gpu = check_and_set_gpu(use_gpu, gpu_lib, quit_on_fail = True)

    # data structures
    vertex_labels, self_comparison = initial_graph_properties(ref_list, ref_list)

    # Filter weights to only the relevant edges
    if weights is None:
        sys.stderr.write("Need weights to construct weighted network\n")
        sys.exit(1)

    # Process weights
    weights = process_weights(dist_matrix, weights_type)

    # Convert edge indices to tuples
    edge_list = generate_all_tuples(num_ref = len(ref_list),
                                                self = True,
                                                int_offset = 0)

    if not use_gpu:
        # Construct network with CPU via edge list
        weighted_edges = []
        for ((src, dest), weight) in zip(edge_list, weights):
            weighted_edges.append((src, dest, weight))
        # build the graph
        graph = gt.Graph(directed = False)
        graph.add_vertex(len(vertex_labels))
        eweight = graph.new_ep("float")
        # Could alternatively assign weights through eweight.a = weights
        graph.add_edge_list(weighted_edges, eprops = [eweight])
        graph.edge_properties["weight"] = eweight

    else:
        # Construct network with GPU via data frame
        G_df = cudf.DataFrame(columns = ['source','destination'])
        G_df['source'] = [edge_list[0][0]]
        G_df['destination'] = [edge_list[0][1]]
        G_df['weights'] = weights
        max_in_vertex_labels = len(vertex_labels)-1
        graph = add_self_loop(G_df, max_in_vertex_labels, weights = True, renumber = False)

    return graph

def construct_network_from_assignments(ref_list, query_list, assignments, within_label = 1, int_offset = 0, weights = None, dist_matrix = None, weights_type = None, previous_network = None, old_ids = None, adding_qq_dists = False, previous_pkl = None, betweenness_sample = betweenness_sample_default, summarise = True, use_gpu = False):
    """Construct an undirected network using sequence lists, assignments of pairwise distances
    to clusters, and the identifier of the cluster assigned to within-strain distances.
    Nodes are samples and edges where samples are within the same cluster
    Will print summary statistics about the network to ``STDERR``

    ## DEPENDS ON Fns: {.utils: [check_and_set_gpu], .:[construct_network_from_edge_list], poppunk_refine: [generateTuples]}

    Args:
        ref_list (list)
            List of reference sequence labels
        query_list (list)
            List of query sequence labels
        assignments (numpy.array or int)
            Labels of most likely cluster assignment
        within_label (int)
            The label for the cluster representing within-strain distances
        int_offset (int)
            Constant integer to add to each node index
        weights (list)
            List of weights for each edge in the network
        distMat (2 column ndarray)
            Numpy array of pairwise distances
        weights_type (str)
            Measure to calculate from the distMat to use as edge weights in network
            - options are core, accessory or euclidean distance
        previous_network (str)
            Name of file containing a previous network to be integrated into this new
            network
        old_ids (list)
            Ordered list of vertex names in previous network
        adding_qq_dists (bool)
            Boolean specifying whether query-query edges are being added
            to an existing network, such that not all the sequence IDs will
            be found in the old IDs, which should already be correctly ordered
        previous_pkl (str)
            Name of file containing the names of the sequences in the previous_network
        betweenness_sample (int)
            Number of sequences per component used to estimate betweenness using
            a GPU. Smaller numbers are faster but less precise [default = 100]
        summarise (bool)
            Whether to calculate and print network summaries with :func:`~networkSummary`
            (default = True)
        use_gpu (bool)
            Whether to use GPUs for network construction
    Returns:
        graph (graph)
            The resulting network
    """

    # Check GPU library use
    #use_gpu = check_and_set_gpu(use_gpu, gpu_lib, quit_on_fail = True)

    # Filter weights to only the relevant edges
    if weights is not None:
        weights = weights[assignments == within_label]
    elif dist_matrix is not None and weights_type is not None:
        if isinstance(assignments, list):
            assignments = np.array(assignments)
        dist_matrix = dist_matrix[assignments == within_label,:]
        weights = process_weights(dist_matrix, weights_type)

    # Convert edge indices to tuples
    connections = poppunk_refine.generateTuples(assignments, within_label, self = (ref_list == query_list), num_ref = len(ref_list), int_offset = int_offset)

    # Construct network using edge list
    graph = construct_network_from_edge_list(ref_list, query_list, connections, weights = weights, dist_matrix = dist_matrix, previous_network = previous_network, adding_qq_dists = adding_qq_dists, old_ids = old_ids, previous_pkl = previous_pkl, summarise = False, use_gpu = use_gpu)
    if summarise:
        print_network_summary(graph, betweenness_sample = betweenness_sample, use_gpu = use_gpu)

    return graph

#TODO: refactor
def construct_network_from_lineage_rank_fit(sparse_mat, ref_list, use_gpu = False):
    #TODO (was previously fn. sparse_mat_to_network())
    """Generate a network from a lineage rank fit

    ## DEPENDS ON Fns: {.indices_refs_clusters: [add_self_loop], .:[construct_network_from_edge_list]}

    Args:
       sparse_mat (scipy or cupyx sparse matrix)
         Sparse matrix of kNN from lineage fit
       ref_list (list)
         List of sequence names
       use_gpu (bool)
         Whether GPU libraries should be used
    Returns:
      graph (network)
          Graph tool or cugraph network
    """
    if not use_gpu:
        connections = []
        for (src,dst) in zip(sparse_mat.row,sparse_mat.col):
            connections.append(src,dst)
        graph = construct_network_from_edge_list(ref_list, ref_list, connections, weights=sparse_mat.data, summarise=False)

    else:
        G_df = cudf.DataFrame(columns = ['source','destination','weights'])
        G_df['source'] = sparse_mat.row
        G_df['destination'] = sparse_mat.col
        G_df['weights'] = sparse_mat.data
        max_in_vertex_labels = len(ref_list)-1
        graph = add_self_loop(G_df, max_in_vertex_labels, weights = True, renumber = False)

    return graph

#TODO: refactor, careful
def add_query_to_network(db_funcs, ref_list, query_list, graph, kmers, assignments, model, query_DB, distances = None, distance_type = "euclidean", query_query_dists = False, strand_preserved = False, weights = None, threads = 1, use_gpu = False):
    """Finds edges between queries and items in the reference database,
    and modifies the network to include them.

    ## DEPENDS ON Fns: {.: [construct_network_from_assignments], poppunk.sketchlib: [addRandom], poppunk.utils: [iterDistRows]}

    Args:
        dbFuncs (list)
            List of backend functions from :func:`~PopPUNK.utils.setupDBFuncs`
        ref_list (list)
            List of reference names
        query_list (list)
            List of query names
        graph (graph)
            Network to add to (mutated)
        kmers (list)
            List of k-mer sizes
        assignments (numpy.array)
            Cluster assignment of items in qlist
        model (ClusterModel)
            Model fitted to reference database
        queryDB (str)
            Query database location
        distances (str)
            Prefix of distance files for extending network
        distance_type (str)
            Distance type to use as weights in network
        query_query_dists (bool)
            Add in all query-query distances
            (default = False)
        strand_preserved (bool)
            Whether to treat strand as known (i.e. ignore rc k-mers)
            when adding random distances. Only used if queryQuery = True
            [default = False]
        weights (numpy.array)
            If passed, the core,accessory distances for each assignment, which will
            be annotated as an edge attribute
        threads (int)
            Number of threads to use if new db created
        use_gpu (bool)
            Whether to use cugraph for analysis
            (default = 1)
    Returns:
        dist_matrix (numpy.array)
            Query-query distances
    """
    # initalise functions
    query_database = db_funcs["queryDatabase"]

    # do not calculate weights unless specified
    if weights is None:
        distance_type = None

    # initialise links data structure
    new_edges = []
    assigned = set()

    # These are returned
    qq_dist_matrix = None

    # store links for each query in a list of edge tuples
    ref_count = len(ref_list)

    # Add queries to network
    graph = construct_network_from_assignments(ref_list, query_list, assignments, within_label = model.within_label, previous_network = graph, old_ids = ref_list, dist_matrix = weights, weights_type = distance_type, summarise = False, use_gpu = use_gpu)

    # Calculate all query-query distances too, if updating database
    if query_query_dists:
        if len(query_list) == 1:
            qq_dist_matrix = np.zeros((0, 2), dtype=np.float32)
        else:
            sys.stderr.write("Calculating all query-query distances\n")
            add_random(query_DB, query_list, kmers, strand_preserved, threads = threads)
            qq_dist_matrix = query_database(rNames = query_list,
                                      qNames = query_list,
                                      dbPrefix = query_DB,
                                      queryPrefix = query_DB,
                                      klist = kmers,
                                      self = True,
                                      number_plot_fits = 0,
                                      threads = threads)

            if distance_type == 'core':
                query_assignation = model.assign(qq_dist_matrix, slope = 0)
            elif distance_type == 'accessory':
                query_assignation = model.assign(qq_dist_matrix, slope = 1)
            else:
                query_assignation = model.assign(qq_dist_matrix)

            # Add queries to network
            graph = construct_network_from_assignments(query_list, query_list, query_assignation, int_offset = ref_count, within_label = model.within_label, previous_network = graph, old_ids = ref_list, adding_qq_dists = True, dist_matrix = qq_dist_matrix, weights_type = distance_type, summarise = False, use_gpu = use_gpu)

    # Otherwise only calculate query-query distances for new clusters
    else:
        # identify potentially new lineages in list: unassigned is a list of queries with no hits
        unassigned = set(query_list).difference(assigned)
        query_indices = {k:v+ref_count for v,k in enumerate(query_list)}
        # process unassigned query sequences, if there are any
        if len(unassigned) > 1:
            sys.stderr.write("Found novel query clusters. Calculating distances between them.\n")

            # use database construction methods to find links between unassigned queries
            add_random(query_DB, query_list, kmers, strand_preserved, threads = threads)
            qq_dist_matrix = query_database(rNames = list(unassigned),
                                      qNames = list(unassigned),
                                      dbPrefix = query_DB,
                                      queryPrefix = query_DB,
                                      klist = kmers,
                                      self = True,
                                      number_plot_fits = 0,
                                      threads = threads)

            if distance_type == 'core':
                query_assignation = model.assign(qq_dist_matrix, slope = 0)
            elif distance_type == 'accessory':
                query_assignation = model.assign(qq_dist_matrix, slope = 1)
            else:
                query_assignation = model.assign(qq_dist_matrix)

            # identify any links between queries and store in the same links dict
            # links dict now contains lists of links both to original database and new queries
            # have to use names and link to query list in order to match to node indices
            for row_idx, (assignment, (query1, query2)) in enumerate(zip(query_assignation, iter_dist_rows(query_list, query_list, self = True))):
                if assignment == model.within_label:
                    if weights is not None:
                        if distance_type == 'core':
                            dist = weights[row_idx, 0]
                        elif distance_type == 'accessory':
                            dist = weights[row_idx, 1]
                        else:
                            dist = np.linalg.norm(weights[row_idx, :])
                        edge_tuple = (query_indices[query1], query_indices[query2], dist)
                    else:
                        edge_tuple = (query_indices[query1], query_indices[query2])
                    new_edges.append(edge_tuple)

            graph = construct_network_from_assignments(query_list, query_list, query_assignation, int_offset = ref_count, within_label = model.within_label, previous_network = graph, old_ids = ref_list + query_list, adding_qq_dists = True, dist_matrix = qq_dist_matrix, weights_type = distance_type, summarise = False, use_gpu = use_gpu)

    return graph, qq_dist_matrix

def cugraph_to_graph_tool(graph, ref_list):
    """Save a network to disk

    ## DEPENDS ON Fns: {.construct_network: [construct_network_from_edge_list]} 
    
    Args:
       G (cugraph network)
         Cugraph network
       rlist (list)
         List of sequence names
    Returns:
      G (graph-tool network)
          Graph tool network
    """
    edge_df = graph.view_edge_list()
    edge_tuple = edge_df[["src", "dst"]].values.tolist()
    edge_weights = None
    if "weights" in edge_df.columns:
        edge_weights = edge_df["weights"].values_host
    graph = construct_network_from_edge_list(ref_list, ref_list, edge_tuple, weights = edge_weights, summarise=False)
    vid = graph.new_vertex_property("string", vals = ref_list)
    graph.vp.id = vid

    return graph



##################
#                #
#    Fns USED    #
#                #
##################

def convert_data_to_df(network_data, weights:(bool or list), use_gpu:bool):
    if isinstance(network_data, scipy.sparse.coo_matrix):
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

def network_summary(graph, calc_betweenness=True, betweenness_sample = betweenness_sample_default, use_gpu = False):
    """Provides summary values about the network

    ## DEPENDS ON Fns: {None}

    Args:
        graph (graph)
            The network of strains
        calc_betweenness (bool)
            Whether to calculate betweenness stats
        use_gpu (bool)
            Whether to use cugraph for graph analysis
    Returns:
        metrics (list)
            List with # components, density, transitivity, mean betweenness
            and weighted mean betweenness
        scores (list)
            List of scores
    """
    if not use_gpu:

        component_assignments, component_frequencies = gt.label_components(graph)
        components = len(component_frequencies)
        density = len(list(graph.edges()))/(0.5 * len(list(graph.vertices())) * (len(list(graph.vertices())) - 1))
        transitivity = gt.global_clustering(graph)[0]

    else:
        use_gpu = check_and_set_gpu(use_gpu, gpu_lib, quit_on_fail = True)

        component_assignments = cugraph.components.connectivity.connected_components(graph)
        component_nums = component_assignments['labels'].unique().astype(int)
        components = len(component_nums)
        density = graph.number_of_edges()/(0.5 * graph.number_of_vertices() * graph.number_of_vertices() - 1)
        # consistent with graph-tool for small graphs - triangle counts differ for large graphs
        # could reflect issue https://github.com/rapidsai/cugraph/issues/1043
        # this command can be restored once the above issue is fixed - scheduled for cugraph 0.20
#        triangle_count = cugraph.community.triangle_count.triangles(G)/3
        triangle_count = 3*get_cugraph_triangles(graph)
        degree_df = graph.in_degree()
        # consistent with graph-tool
        triad_count = 0.5 * sum([d * (d - 1) for d in degree_df[degree_df['degree'] > 1]['degree'].to_pandas()])
        if triad_count > 0:
            transitivity = triangle_count/triad_count
        else:
            transitivity = 0.0

    mean_bt = 0
    weighted_mean_bt = 0
    if calc_betweenness:
        betweenness = []
        sizes = []

        if not use_gpu:
            for component, size in enumerate(component_frequencies):
                if size > 3:
                    vfilt = component_assignments.a == component
                    subgraph = gt.GraphView(graph, vfilt=vfilt)
                    betweenness.append(max(gt.betweenness(subgraph, norm = True)[0].a))
                    sizes.append(size)
        else:
            component_frequencies = component_assignments['labels'].value_counts(sort = True, ascending = False)
            for component in component_nums.to_pandas():
                size = component_frequencies[component_frequencies.index == component].iloc[0].astype(int)
                if size > 3:
                    component_vertices = component_assignments['vertex'][component_assignments['labels']==component]
                    subgraph = cugraph.subgraph(graph, component_vertices)
                    if len(component_vertices) >= betweenness_sample:
                        component_betweenness = cugraph.betweenness_centrality(subgraph,
                                                                                k = betweenness_sample,
                                                                                normalized = True)
                    else:
                        component_betweenness = cugraph.betweenness_centrality(subgraph,
                                                                                normalized = True)
                    betweenness.append(component_betweenness['betweenness_centrality'].max())
                    sizes.append(size)

        if len(betweenness) > 1:
            mean_bt = np.mean(betweenness)
            weighted_mean_bt = np.average(betweenness, weights=sizes)
        elif len(betweenness) == 1:
            mean_bt = betweenness[0]
            weighted_mean_bt = betweenness[0]

    # Calculate scores
    metrics = [components, density, transitivity, mean_bt, weighted_mean_bt]
    base_score = transitivity * (1 - density)
    scores = [base_score, base_score * (1 - metrics[3]), base_score * (1 - metrics[4])]
    return(metrics, scores)

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

