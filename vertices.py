import sys

def get_vertex_list(graph, use_gpu = False):
    """Generate a list of node indices

    ## DEPENDS ON Fns: {None}

    Args:
       graph (network)
           Graph tool network
       use_gpu (bool)
            Whether graph is a cugraph or not
            [default = False]
    Returns:
       vertex_list (list)
           List of integers corresponding to nodes
    """

    if not use_gpu:
        vertex_list = list(graph.vertices())
    else:
        vertex_list = range(graph.number_of_vertices())

    return vertex_list

def check_network_vertex_count(seq_list, graph, use_gpu):
    """Checks the number of network vertices matches the number
    of sequence names.

    ## DEPENDS ON Fns: {.: [get_vertex_list]}

    Args:
        seq_list (list)
            The list of sequence names
        graph (network)
            The network of sequences
        use_gpu (bool)
            Whether to use cugraph for graph analyses
    """
    
    vertex_list = set(get_vertex_list(graph, use_gpu = use_gpu))
    num_missing_vertices = set(set(range(len(seq_list))).difference(vertex_list))
    if len(num_missing_vertices) > 0:
        sys.stderr.write("ERROR: " + str(len(num_missing_vertices)) + " samples are missing from the final network\n")
        sys.exit(1)
