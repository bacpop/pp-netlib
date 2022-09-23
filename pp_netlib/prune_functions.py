import sys
from collections import Counter, defaultdict
from turtle import back

def get_gt_clique_refs(graph, reference_indices:set()):
    """Recursively prune a network of its cliques. Returns one vertex from
    a clique at each stage

    ## DEPENDS ON Fns: {none}

    Args:
        graph (graph)
            The graph to get clique representatives from
        reference_indices (set)
            The unique list of vertices being kept, to add to
    """
    import graph_tool.all as gt
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
            get_gt_clique_refs(subgraph, reference_indices)
        elif subgraph.num_vertices() == 1:
            reference_indices.add(subgraph.get_vertices()[0])
    except StopIteration:
        pass

    return reference_indices

def gt_clique_prune(component, graph, reference_indices, components_list):
    import graph_tool.all as gt
    if gt.openmp_enabled():
        gt.openmp_set_num_threads(1)
    sys.setrecursionlimit(3000)
    subgraph = gt.GraphView(graph, vfilt=components_list == component)
    refs = reference_indices.copy()
    if subgraph.num_vertices() <= 2:
        refs.add(subgraph.get_vertices()[0])
        ref_list = refs
    else:
        ref_list = get_gt_clique_refs(subgraph, refs)

    return ref_list

def get_nx_clique_refs(graph, reference_indices = set()):
    import networkx as nx
    cliques = nx.find_cliques(graph)

    try:
        clique = frozenset(next(cliques))
        if clique.isdisjoint(reference_indices):
            reference_indices.add(list(clique)[0])
        
        subgraph = graph.subgraph([v not in clique for v in graph.nodes()])
        if subgraph.number_of_nodes() > 1:
            get_nx_clique_refs(subgraph, reference_indices)
        elif subgraph.number_of_nodes() == 1:
            reference_indices.add(subgraph.nodes()[0])
    except StopIteration:
        pass

    return reference_indices

def nx_clique_prune(component, graph, reference_indices, components_list):

    import networkx as nx

    sys.setrecursionlimit(3000)
    subgraph = graph.subgraph(component)
    refs = reference_indices.copy()
    if len(subgraph.nodes()) <= 2:
        refs.add(list(subgraph.nodes())[0])
        ref_list = refs
    else:
        ref_list = get_nx_clique_refs(subgraph, refs)

    refs_list = set(int(ref.replace("n", "")) for ref in ref_list)

    return refs_list

def print_clusters(graph, vertex_labels, backend, out_prefix=None, old_cluster_file=None, external_cluster_csv=None, print_ref=True, print_csv=True, clustering_type="combined", write_unwords=True, use_gpu = False):

    import operator
    import numpy as np
    from scipy.stats import rankdata
    from pp_netlib.utils import gen_unword, read_isolate_type_from_csv, print_external_clusters
    if backend == "GT":
        import graph_tool.all as gt
    elif backend == "NX":
        import networkx as nx
    

    if old_cluster_file == None and print_ref == False:
        raise RuntimeError("Trying to print query clusters with no query sequences")
    if write_unwords and not print_csv:
        write_unwords = False

    # get a sorted list of component assignments
    if use_gpu:
        raise NotImplementedError("GPU graph not yet implemented")
        # use_gpu = check_and_set_gpu(use_gpu, gpu_lib, quit_on_fail = True)
        # component_assignments = cugraph.components.connectivity.connected_components(G)
        # component_frequencies = component_assignments['labels'].value_counts(sort = True, ascending = False)
        # newClusters = [set() for rank in range(component_frequencies.size)]
        # for isolate_index, isolate_name in enumerate(rlist): # assume sorted at the moment
        #     component = component_assignments['labels'].iloc[isolate_index].item()
        #     component_rank_bool = component_frequencies.index == component
        #     component_rank = np.argmax(component_rank_bool.to_array())
        #     newClusters[component_rank].add(isolate_name)
    else:
        if backend == "GT":
            component_assignments, component_frequencies = gt.label_components(graph)
            component_assignments = list(component_assignments)
            component_frequencies = list(component_frequencies)

        elif backend == "NX":
            component_assignments = list((nx.get_node_attributes(graph, "comp_membership")).values())
            component_frequencies = list(len(c) for c in nx.connected_components(graph))

        component_frequency_ranks = len(component_frequencies) - rankdata(component_frequencies, method = "ordinal").astype(int)
        # use components to determine new clusters
        new_clusters = [set() for rank in range(len(component_frequency_ranks))]

        for isolate_index, isolate_name in enumerate(vertex_labels):
            component = component_assignments[isolate_index]
            component_rank = component_frequency_ranks[component]
            new_clusters[component_rank].add(isolate_name)


    old_names = set()

    if old_cluster_file != None:
        old_all_clusters = read_isolate_type_from_csv(old_cluster_file, mode = "external", return_dict = False)
        old_clusters = old_all_clusters[list(old_all_clusters.keys())[0]]
        new_id = len(old_clusters.keys()) + 1 # 1-indexed
        while new_id in old_clusters:
            new_id += 1 # in case clusters have been merged

        # Samples in previous clustering
        for prev_cluster in old_clusters.values():
            for prev_sample in prev_cluster:
                old_names.add(prev_sample)

    # Assign each cluster a name
    clustering = {}
    found_old_clusters = []
    cluster_unword = {}
    if write_unwords:
        unword_generator = gen_unword()

    for new_cls_idx, new_cluster in enumerate(new_clusters):
        needs_unword = False
        # Ensure consistency with previous labelling
        if old_cluster_file != None:
            merge = False
            cls_id = None

            # Samples in this cluster that are not queries
            ref_only = old_names.intersection(new_cluster)

            # A cluster with no previous observations
            if len(ref_only) == 0:
                cls_id = str(new_id)    # harmonise data types; string flexibility helpful
                new_id += 1
                needs_unword = True
            else:
                # Search through old cluster IDs to find a match
                for old_cluster_name, old_cluster_members in old_clusters.items():
                    join = ref_only.intersection(old_cluster_members)
                    if len(join) > 0:
                        # Check cluster is consistent with previous definitions
                        if old_cluster_name in found_old_clusters:
                            sys.stderr.write("WARNING: Old cluster " + old_cluster_name + " split"
                                             " across multiple new clusters\n")
                        else:
                            found_old_clusters.append(old_cluster_name)

                        # Query has merged clusters
                        if len(join) < len(ref_only):
                            merge = True
                            needs_unword = True
                            if cls_id == None:
                                cls_id = old_cluster_name
                            else:
                                cls_id += "_" + old_cluster_name
                        # Exact match -> same name as before
                        elif len(join) == len(ref_only):
                            assert merge == False # should not have already been part of a merge
                            cls_id = old_cluster_name
                            break

            # Report merges
            if merge:
                merged_ids = cls_id.split("_")
                sys.stderr.write("Clusters " + ",".join(merged_ids) + " have merged into " + cls_id + "\n")

        # Otherwise just number sequentially starting from 1
        else:
            cls_id = new_cls_idx + 1
            needs_unword = True

        if write_unwords and needs_unword:
            unword = next(unword_generator)
        else:
            unword = None

        for cluster_member in new_cluster:
            clustering[cluster_member] = cls_id
            if unword is not None:
                cluster_unword[cluster_member] = unword

    # print clustering to file
    if print_csv:
        out_filename = out_prefix + "_clusters.csv"
        with open(out_filename, 'w') as cluster_file:
            cluster_file.write("Taxon,Cluster\n")
            if write_unwords:
                unword_file = open(out_prefix + "_unword_clusters.csv", 'w')
                unword_file.write("Taxon,Cluster_name\n")

            # sort the clusters by frequency - define a list with a custom sort order
            # first line gives tuples e.g. (1, 28), (2, 17) - cluster 1 has 28 members, cluster 2 has 17 members
            # second line takes first element - the cluster IDs sorted by frequency
            freq_order = sorted(dict(Counter(clustering.values())).items(), key=operator.itemgetter(1), reverse=True)
            freq_order = [x[0] for x in freq_order]

            # iterate through cluster dictionary sorting by value using above custom sort order
            for cluster_member, cluster_name in sorted(clustering.items(), key=lambda i:freq_order.index(i[1])):
                if print_ref or cluster_member not in old_names:
                    cluster_file.write(",".join((cluster_member, str(cluster_name))) + "\n")
                if write_unwords and cluster_member in cluster_unword:
                    unword_file.write(",".join((cluster_member, cluster_unword[cluster_member])) + "\n")

            if write_unwords:
                unword_file.close()

        if external_cluster_csv is not None:
            print_external_clusters(new_clusters, external_cluster_csv, out_prefix, old_names, print_ref)

    return clustering 

def gt_get_ref_graph(graph, ref_indices, vertex_labels, type_isolate, backend):

    if backend == "GT":
        import graph_tool.all as gt
    elif backend == "NX":
        import networkx as nx

    if type_isolate is not None:
        type_isolate_index = vertex_labels.index(type_isolate)
    else:
        type_isolate_index = None
    
    if type_isolate_index is not None and type_isolate_index not in ref_indices:
            ref_indices.add(type_isolate_index)

    if backend == "GT":
        reference_vertex = graph.new_vertex_property('bool')
        for n, vertex in enumerate(graph.vertices()):
            if n in ref_indices:
                reference_vertex[vertex] = True
            else:
                reference_vertex[vertex] = False

        ref_graph = gt.GraphView(graph, vfilt = reference_vertex)
        ref_graph = gt.Graph(ref_graph, prune = True)
    
    elif backend == "NX":
        ref_graph = graph.subgraph(["n"+str(ri) for ri in ref_indices])
    ###
    clusters_in_full_graph = print_clusters(graph, vertex_labels, backend = backend, print_csv=False)
    reference_clusters_in_full_graph = defaultdict(set)
    for reference_index in ref_indices:
        try:
            reference_clusters_in_full_graph[clusters_in_full_graph[vertex_labels[reference_index]]].add(reference_index)
        except IndexError:
            pass

    # Calculate the component membership within the reference graph
    ref_order = [name for idx, name in enumerate(vertex_labels) if idx in frozenset(ref_indices)]
    clusters_in_reference_graph = print_clusters(ref_graph, ref_order, backend = backend, print_csv=False)
    # Record the components/clusters the references are in the reference graph
    # dict: name: ref_cluster
    reference_clusters_in_reference_graph = {}
    for reference_name in ref_order:
        try:
            reference_clusters_in_reference_graph[reference_name] = clusters_in_reference_graph[reference_name]
        except IndexError:
            pass

    # Check if multi-reference components have been split as a validation test
    # First iterate through clusters
    network_update_required = False
    for cluster_id, ref_idxs in reference_clusters_in_full_graph.items():
        # Identify multi-reference clusters by this length
        if len(ref_idxs) > 1:
            check = list(ref_idxs)
            # check if these are still in the same component in the reference graph
            for i in range(len(check)):
                component_i = reference_clusters_in_reference_graph[vertex_labels[check[i]]]
                for j in range(i + 1, len(check)):
                    # Add intermediate nodes
                    component_j = reference_clusters_in_reference_graph[vertex_labels[check[j]]]
                    if component_i != component_j:
                        network_update_required = True
                        if backend == "GT":
                            vertex_list, edge_list = gt.shortest_path(graph, check[i], check[j])
                            for vertex in vertex_list:
                                reference_vertex[vertex] = True
                                ref_indices.add(int(vertex))
                        elif backend == "NX":
                            print(check[i], check[j])
                            vertex_list = nx.shortest_path(graph, "n"+str(check[i]), "n"+str(check[j]))
                            print(vertex_list)
                            for vertex in vertex_list:
                                print(vertex)
                                vert_idx = vertex.replace("n", "")
                                print(ref_indices)
                                ref_indices.add(int(vert_idx))
                                print(ref_indices)


    # update reference graph if vertices have been added
    if network_update_required:
        if backend == "GT":
            ref_graph = gt.GraphView(graph, vfilt = reference_vertex)
            ref_graph = gt.Graph(ref_graph, prune = True) # https://stackoverflow.com/questions/30839929/graph-tool-graphview-object
        elif backend == "NX":
            ref_graph = graph.subgraph(["n"+str(ri) for ri in ref_indices])

    # Order found references as in sketch files
    #reference_names = [vertex_labels[int(x)] for x in sorted(ref_indices)]
    ###

    return ref_graph