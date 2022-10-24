import graph_tool.all as gt


def get_clique_refs(graph, reference_indices = set()):
    """Recursively prune a network of its cliques. Returns one vertex from
    a clique at each stage

    ## DEPENDS ON Fns: {none}

    Args:
        graph (graph)
            The graph to get clique representatives from
        reference_indices (set)
            The unique list of vertices being kept, to add to
    """
    cliques = gt.max_cliques(graph)
    try:
        # Get the first clique, and see if it has any members already
        # contained in the vertex list
        clique = frozenset(next(cliques))
        if clique.isdisjoint(reference_indices):
            reference_indices.add(list(clique)[0])

        # Remove the clique, and prune the resulting subgraph (recursively)
        subgraph = gt.GraphView(graph, vfilt=[v not in clique for v in graph.vertices()])
        if subgraph.num_vertices() > 1:
            get_clique_refs(subgraph, reference_indices)
        elif subgraph.num_vertices() == 1:
            reference_indices.add(subgraph.get_vertices()[0])
    except StopIteration:
        pass
    return reference_indices

def prune_cliques(component, graph, reference_indices, components_list):
    """Wrapper function around :func:`~getCliqueRefs` so it can be
       called by a multiprocessing pool

        ## DEPENDS ON Fns: {.:[get_clique_refs]}

    """
    if gt.openmp_enabled():
        gt.openmp_set_num_threads(1)
    subgraph = gt.GraphView(graph, vfilt=components_list == component)
    refs = reference_indices.copy()
    if subgraph.num_vertices() <= 2:
        refs.add(subgraph.get_vertices()[0])
        ref_list = refs
    else:
        ref_list = get_clique_refs(subgraph, refs)
    return(list(ref_list))

