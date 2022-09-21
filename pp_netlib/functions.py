########################
####   .CONSTRUCT   ####
########################
from ast import Index
from collections import defaultdict, Counter
from functools import partial
from multiprocessing import Pool, freeze_support
import sys
import scipy
import numpy as np
import pandas as pd

def construct_with_graphtool(network_data, vertex_labels, weights = None):
    """Construct a graph with graph-tool

    Args:
        network_data (dataframe OR edge list OR sparse coordinate matrix): Data containing record of edges in the graph.
        vertex_labels (list): List of vertex/node labels to apply to graph vertices
        weights (list, optional): List of weights associated with edges in network_data.
                                      Weights must be in the same order as edges in network_data. Defaults to None.

    Returns:
        graph (gt.Graph): Graph-tool graph object populated with network data
    """
    import graph_tool.all as gt

    graph = gt.Graph(directed = False) ## initialise graph_tool graph object

    ########################
    ####    DF INPUT    ####
    ########################
    if isinstance(network_data, pd.DataFrame):
        network_data.columns = ["source", "destination"]

        graph.add_vertex(len(vertex_labels)) ## add vertices

        ## add weights column if weights provided as list (add error catching?)
        if weights is not None:
            network_data["weights"] = weights
            eweight = graph.new_ep("float")
            graph.add_edge_list(network_data.values, eprops = [eweight]) ## add weighted edges
            graph.edge_properties["weight"] = eweight
        else:
            graph.add_edge_list(network_data.values) ## add edges

    ##########################
    #### SPARSE MAT INPUT ####
    ##########################
    elif isinstance(network_data, scipy.sparse.coo_matrix):
        graph_data_df = pd.DataFrame()
        graph_data_df["source"] = network_data.row
        graph_data_df["destination"] =  network_data.col
        graph_data_df["weights"] = network_data.data

        graph.add_vertex(len(vertex_labels)) ## add vertices
        eweight = graph.new_ep("float")
        graph.add_edge_list(list(map(tuple, graph_data_df.values)), eprops = [eweight]) ## add weighted edges
        graph.edge_properties["weight"] = eweight

    ########################
    ####   LIST INPUT   ####
    ########################
    elif isinstance(network_data, list):
        graph.add_vertex(len(vertex_labels)) ## add vertices

        if weights is not None:
            weighted_edges = []

            for i in range(len(network_data)):
                weighted_edges.append(network_data[i] + (weights[i],))

            eweight = graph.new_ep("float")
            graph.add_edge_list(weighted_edges, eprops = [eweight]) ## add weighted edges
            graph.edge_properties["weight"] = eweight

        else:
            graph.add_edge_list(network_data) ## add edges

    v_name_prop = graph.new_vp("string")
    graph.vertex_properties["id"] = v_name_prop
    for i in range(len([v for v in graph.vertices()])):
        v_name_prop[graph.vertex(i)] = vertex_labels[i]

    return graph

def construct_with_networkx(network_data, vertex_labels, weights = None):
    """Construct a graph with networkx

    Args:
        network_data (dataframe OR edge list OR sparse coordinate matrix): Data containing record of edges in the graph.
        vertex_labels (list): List of vertex/node labels to apply to graph vertices
        weights (list, optional): List of weights associated with edges in network_data.
                                      Weights must be in the same order as edges in network_data. Defaults to None.

    Returns:
        graph (nx.Graph): Graph-tool graph object populated with network data
    """
    import networkx as nx
    
    ## initialise nx graph and add nodes
    graph = nx.Graph()

    nodes_list = [(i, dict(id=vertex_labels[i])) for i in range(len(vertex_labels))]
    graph.add_nodes_from(nodes_list)

    ########################
    ####    DF INPUT    ####
    ########################
    if isinstance(network_data, pd.DataFrame):
        network_data.columns = ["source", "destination"]
        if weights is not None:
            network_data["weights"] = weights
            graph.add_weighted_edges_from(network_data.values)
        else:
            graph.add_edges_from(network_data.values)

    ##########################
    #### SPARSE MAT INPUT ####
    ##########################
    elif isinstance(network_data, scipy.sparse.coo_matrix):
        weighted_edges = list(zip(list(network_data.row), list(network_data.col), list(network_data.data)))
        graph.add_weighted_edges_from(weighted_edges)

    ########################
    ####   LIST INPUT   ####
    ########################
    elif isinstance(network_data, list):
        if weights is not None:
            src, dest = zip(*network_data)
            weighted_edges = list(zip(src, dest, weights))
            graph.add_weighted_edges_from(weighted_edges)
        else:
            graph.add_edges_from(network_data)

    return graph

########################
####   .SUMMARISE   ####
########################
def summarise(graph, backend):
    """Get graph metrics and format into soutput string.

    Args:
        graph (Network object): The graph for which to obtain summary metrics
        backend (str): The tool used to build graph ("GT", "NX", or "CG"(TODO))

    Returns:
        summary_contents (formatted str): Graph summary metrics formatted to print to stderr
    """
    if backend == "GT":
        import graph_tool.all as gt
        component_assignments, component_frequencies = gt.label_components(graph)
        components = len(component_frequencies)
        density = len(list(graph.edges()))/(0.5 * len(list(graph.vertices())) * (len(list(graph.vertices())) - 1))
        transitivity = gt.global_clustering(graph)[0]

        mean_bt = 0
        weighted_mean_bt = 0
        betweenness = []
        sizes = []
        for component, size in enumerate(component_frequencies):
            if size > 3:
                vfilt = component_assignments.a == component
                subgraph = gt.GraphView(graph, vfilt=vfilt)
                betweenness.append(max(gt.betweenness(subgraph, norm = True)[0].a))
                sizes.append(size)

        if len(betweenness) > 1:
            mean_bt = np.mean(betweenness)
            weighted_mean_bt = np.average(betweenness, weights=sizes)
        elif len(betweenness) == 1:
            mean_bt = betweenness[0]
            weighted_mean_bt = betweenness[0]

    elif backend == "NX":
        import networkx as nx
        components = nx.number_connected_components(graph)
        density = nx.density(graph)
        transitivity = nx.transitivity(graph)

        betweenness = []
        sizes = []
        for c in nx.connected_components(graph):
            betweenness.append(max((nx.betweenness_centrality(graph.subgraph(c))).values()))
            sizes.append(len(graph.subgraph(c)))

        if len(betweenness) > 1:
            mean_bt = np.mean(betweenness)
            weighted_mean_bt = np.average(betweenness, weights=sizes)
        elif len(betweenness) == 1:
            mean_bt = betweenness[0]
            weighted_mean_bt = betweenness[0]

    metrics = [components, density, transitivity, mean_bt, weighted_mean_bt]
    base_score = transitivity * (1 - density)
    scores = [base_score, base_score * (1 - metrics[3]), base_score * (1 - metrics[4])]
    
    return metrics, scores

########################
####     .PRUNE     ####
########################
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
        try:
            refs.add(subgraph.get_vertices()[1])
        except IndexError as ie:
            pass
        ref_list = refs
    else:
        ref_list = get_gt_clique_refs(subgraph, refs)
    #print(ref_list)
    return ref_list 

def print_clusters(graph, vertex_labels, outPrefix=None, oldClusterFile=None,
                  externalClusterCSV=None, printRef=True, printCSV=True,
                  clustering_type='combined', write_unwords=True,
                  use_gpu = False):

    import operator
    import graph_tool.all as gt
    from scipy.stats import rankdata
    from pp_netlib.utils import gen_unword, read_isolate_type_from_csv, print_external_clusters

    if oldClusterFile == None and printRef == False:
        raise RuntimeError("Trying to print query clusters with no query sequences")
    if write_unwords and not printCSV:
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
        component_assignments, component_frequencies = gt.label_components(graph)
        component_frequency_ranks = len(component_frequencies) - rankdata(component_frequencies, method = 'ordinal').astype(int)
        # use components to determine new clusters
        newClusters = [set() for rank in range(len(component_frequency_ranks))]
        for isolate_index, isolate_name in enumerate(vertex_labels):
            component = component_assignments.a[isolate_index]
            component_rank = component_frequency_ranks[component]
            newClusters[component_rank].add(isolate_name)

    oldNames = set()

    if oldClusterFile != None:
        oldAllClusters = read_isolate_type_from_csv(oldClusterFile, mode = 'external', return_dict = False)
        oldClusters = oldAllClusters[list(oldAllClusters.keys())[0]]
        new_id = len(oldClusters.keys()) + 1 # 1-indexed
        while new_id in oldClusters:
            new_id += 1 # in case clusters have been merged

        # Samples in previous clustering
        for prev_cluster in oldClusters.values():
            for prev_sample in prev_cluster:
                oldNames.add(prev_sample)

    # Assign each cluster a name
    clustering = {}
    foundOldClusters = []
    cluster_unword = {}
    if write_unwords:
        unword_generator = gen_unword()

    for newClsIdx, newCluster in enumerate(newClusters):
        needs_unword = False
        # Ensure consistency with previous labelling
        if oldClusterFile != None:
            merge = False
            cls_id = None

            # Samples in this cluster that are not queries
            ref_only = oldNames.intersection(newCluster)

            # A cluster with no previous observations
            if len(ref_only) == 0:
                cls_id = str(new_id)    # harmonise data types; string flexibility helpful
                new_id += 1
                needs_unword = True
            else:
                # Search through old cluster IDs to find a match
                for oldClusterName, oldClusterMembers in oldClusters.items():
                    join = ref_only.intersection(oldClusterMembers)
                    if len(join) > 0:
                        # Check cluster is consistent with previous definitions
                        if oldClusterName in foundOldClusters:
                            sys.stderr.write("WARNING: Old cluster " + oldClusterName + " split"
                                             " across multiple new clusters\n")
                        else:
                            foundOldClusters.append(oldClusterName)

                        # Query has merged clusters
                        if len(join) < len(ref_only):
                            merge = True
                            needs_unword = True
                            if cls_id == None:
                                cls_id = oldClusterName
                            else:
                                cls_id += "_" + oldClusterName
                        # Exact match -> same name as before
                        elif len(join) == len(ref_only):
                            assert merge == False # should not have already been part of a merge
                            cls_id = oldClusterName
                            break

            # Report merges
            if merge:
                merged_ids = cls_id.split("_")
                sys.stderr.write("Clusters " + ",".join(merged_ids) + " have merged into " + cls_id + "\n")

        # Otherwise just number sequentially starting from 1
        else:
            cls_id = newClsIdx + 1
            needs_unword = True

        if write_unwords and needs_unword:
            unword = next(unword_generator)
        else:
            unword = None

        for cluster_member in newCluster:
            clustering[cluster_member] = cls_id
            if unword is not None:
                cluster_unword[cluster_member] = unword

    # print clustering to file
    if printCSV:
        outFileName = outPrefix + "_clusters.csv"
        with open(outFileName, 'w') as cluster_file:
            cluster_file.write("Taxon,Cluster\n")
            if write_unwords:
                unword_file = open(outPrefix + "_unword_clusters.csv", 'w')
                unword_file.write("Taxon,Cluster_name\n")

            # sort the clusters by frequency - define a list with a custom sort order
            # first line gives tuples e.g. (1, 28), (2, 17) - cluster 1 has 28 members, cluster 2 has 17 members
            # second line takes first element - the cluster IDs sorted by frequency
            freq_order = sorted(dict(Counter(clustering.values())).items(), key=operator.itemgetter(1), reverse=True)
            freq_order = [x[0] for x in freq_order]

            # iterate through cluster dictionary sorting by value using above custom sort order
            for cluster_member, cluster_name in sorted(clustering.items(), key=lambda i:freq_order.index(i[1])):
                if printRef or cluster_member not in oldNames:
                    cluster_file.write(",".join((cluster_member, str(cluster_name))) + "\n")
                if write_unwords and cluster_member in cluster_unword:
                    unword_file.write(",".join((cluster_member, cluster_unword[cluster_member])) + "\n")

            if write_unwords:
                unword_file.close()

        if externalClusterCSV is not None:
            print_external_clusters(newClusters, externalClusterCSV, outPrefix, oldNames, printRef)

    return(clustering)

def gt_get_ref_graph(graph, ref_indices, vertex_labels):

    import graph_tool.all as gt

    reference_vertex = graph.new_vertex_property('bool')
    for n, vertex in enumerate(graph.vertices()):
        if n in ref_indices:
            reference_vertex[vertex] = True
        else:
            reference_vertex[vertex] = False

    ref_graph = gt.GraphView(graph, vfilt = reference_vertex)
    ref_graph = gt.Graph(ref_graph, prune = True)

    ###
    clusters_in_full_graph = print_clusters(graph, vertex_labels, printCSV=False)
    reference_clusters_in_full_graph = defaultdict(set)
    for reference_index in ref_indices:
        try:
            reference_clusters_in_full_graph[clusters_in_full_graph[vertex_labels[reference_index]]].add(reference_index)
        except IndexError:
            pass

    # Calculate the component membership within the reference graph
    ref_order = [name for idx, name in enumerate(vertex_labels) if idx in frozenset(ref_indices)]
    clusters_in_reference_graph = print_clusters(ref_graph, ref_order, printCSV=False)
    # Record the components/clusters the references are in the reference graph
    # dict: name: ref_cluster
    reference_clusters_in_reference_graph = {}
    for reference_name in ref_order:
        reference_clusters_in_reference_graph[reference_name] = clusters_in_reference_graph[reference_name]

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
                        vertex_list, edge_list = gt.shortest_path(graph, check[i], check[j])
                        # update reference list
                        for vertex in vertex_list:
                            reference_vertex[vertex] = True
                            ref_indices.add(int(vertex))

    # update reference graph if vertices have been added
    if network_update_required:
        ref_graph = gt.GraphView(graph, vfilt = reference_vertex)
        ref_graph = gt.Graph(ref_graph, prune = True) # https://stackoverflow.com/questions/30839929/graph-tool-graphview-object

    # Order found references as in sketch files
    #reference_names = [vertex_labels[int(x)] for x in sorted(ref_indices)]
    ###

    return ref_graph

def get_nx_clique_refs(graph, reference_indices = set()):

    cliques = nx.find_cliques(graph)

    try:
        clique = frozenset(next(cliques))
        if clique.isdisjoint(reference_indices):
            reference_indices.add(list(clique)[0])
        
        subgraph = graph.subgraph([v not in clique for v in graph.nodes()])
        if subgraph.num_vertices() > 1:
            get_nx_clique_refs(subgraph, reference_indices)
        elif subgraph.num_vertices() == 1:
            reference_indices.add(subgraph.get_vertices()[0])
    except StopIteration:
        pass

    return reference_indices

def nx_prune_cliques(graph, reference_indices, component, components_list):

    import networkx as nx

    sys.setrecursionlimit(3000)
    subgraph = graph.subgraph(components_list == component)
    refs = reference_indices.copy()
    if len(subgraph.nodes()) <= 2:
        refs.add(list(subgraph.nodes())[0])
        ref_list = refs
    else:
        ref_list = get_gt_clique_refs(subgraph, refs)

    
    return(list(ref_list))