import networkx as nx

def nx_get_clique_refs(graph, references):
    """Get reference sample from each clique

    Args:
        graph (nx.Graph): networkx graph to be pruned
        references (set): set of known reference samples, if available.

    Returns:
        references (set): set of reference samples from each clique
    """
    cliques = list(nx.find_cliques(graph))
    # order list by size of clique
    cliques.sort(key = len, reverse=True)
    # iterate through cliques
    for clique in cliques:
        alreadyRepresented = 0
        for node in clique: ## sorted each clique makig the result consistent over multiple function calls. Possibly slow.
            if node in references:
                alreadyRepresented = 1
                break
        if alreadyRepresented == 0:
            references.add(sorted(clique)[0])

    return references

def nx_get_connected_refs(graph, references):
    """Add nodes connecting references found from each clique

    Args:
        graph (nx.Graph): graph to be pruned
        references (set): set of reference samples

    Returns:
        references: *updated* set, now containing connecting nodes
    """

    new_clusters = sorted(nx.connected_components(graph), key=len, reverse=True)
    clustering = {}
    for new_cls_idx, new_cluster in enumerate(new_clusters):
        cls_id = new_cls_idx + 1
        for cluster_member in new_cluster:
            clustering[cluster_member] = cls_id

    ref_clusters = set()
    multi_ref_clusters = set()
    for reference in references:
        if clustering[reference] in ref_clusters:
            multi_ref_clusters.add(clustering[reference])
        else:
            ref_clusters.add(clustering[reference])

    if len(multi_ref_clusters) > 0:
        # Initial reference graph
        ref_G = graph.copy()
        ref_G.remove_nodes_from([node for node in graph.nodes() if node not in references])

        for multi_ref_cluster in multi_ref_clusters:
            # Get a list of nodes that need to be in the same component
            check = []
            for reference in references:
                if clustering[reference] == multi_ref_cluster:
                    check.append(reference)

            # Pairwise check that nodes are in same component
            for i in range(len(check)):
                component = nx.node_connected_component(ref_G, check[i])
                for j in range(i+1, len(check)):
                    # Add intermediate nodes
                    if check[j] not in component:
                        new_path = nx.shortest_path(graph, check[i], check[j])
                        for node in new_path:
                            references.add(node)

    return references
