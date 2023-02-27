import sys
import operator
import pandas as pd
from scipy.stats import rankdata
import graph_tool.all as gt
from collections import defaultdict, Counter

RECURSION_LIMIT = 3000

def gen_unword(unique=True):
    import json
    import gzip
    import random
    import string
    import pkg_resources

    # Download from https://github.com/dwyl/english-words/raw/master/words_dictionary.json
    word_list = pkg_resources.resource_stream(__name__, "data/words_dictionary.json.gz")
    with gzip.open(word_list, "rb") as word_list:
        real_words = json.load(word_list)

    vowels = ["a", "e", "i", "o", "u"]
    trouble = ["q", "x", "y"]
    consonants = set(string.ascii_lowercase) - set(vowels) - set(trouble)

    vowel = lambda: random.sample(vowels, 1)
    consonant = lambda: random.sample(consonants, 1)
    cv = lambda: consonant() + vowel()
    cvc = lambda: cv() + consonant()
    syllable = lambda: random.sample([vowel, cv, cvc], 1)

    returned_words = set()
    # Iterator loop
    while True:
        # Retry loop
        while True:
            word = ""
            for i in range(random.randint(2, 3)):
                word += "".join(syllable()[0]())
            if word not in real_words and (not unique or word not in returned_words):
                returned_words.add(word)
                break
        yield word

def read_isolate_type_from_csv(clusters_csv, mode = "clusters", return_dict = False):
    """Read cluster definitions from CSV file.
    Args:
        clusters_csv (str)
            File name of CSV with isolate assignments
        return_type (str)
            If True, return a dict with sample->cluster instead
            of sets
    Returns:
        clusters (dict)
            Dictionary of cluster assignments (keys are cluster names, values are
            sets containing samples in the cluster). Or if return_dict is set keys
            are sample names, values are cluster assignments.
    """
    # data structures
    if return_dict:
        clusters = defaultdict(dict)
    else:
        clusters = {}

    # read CSV
    clusters_csv_df = pd.read_csv(clusters_csv, index_col = 0, quotechar='"')

    # select relevant columns according to mode
    if mode == "clusters":
        type_columns = [n for n,col in enumerate(clusters_csv_df.columns) if ("Cluster" in col)]
    elif mode == "lineages":
        type_columns = [n for n,col in enumerate(clusters_csv_df.columns) if ("Rank_" in col or "overall" in col)]
    elif mode == "external":
        if len(clusters_csv_df.columns) == 1:
            type_columns = [0]
        elif len(clusters_csv_df.columns) > 1:
            type_columns = range((len(clusters_csv_df.columns)-1))
    else:
        sys.stderr.write("Unknown CSV reading mode: " + mode + "\n")
        sys.exit(1)

    # read file
    for row in clusters_csv_df.itertuples():
        for cls_idx in type_columns:
            cluster_name = clusters_csv_df.columns[cls_idx]
            cluster_name = cluster_name.replace("__autocolour","")
            if return_dict:
                clusters[cluster_name][str(row.Index)] = str(row[cls_idx + 1])
            else:
                if cluster_name not in clusters.keys():
                    clusters[cluster_name] = defaultdict(set)
                clusters[cluster_name][str(row[cls_idx + 1])].add(row.Index)

    # return data structure
    return clusters

def print_external_clusters(newClusters, extClusterFile, outPrefix,
                          oldNames, printRef = True):
    # Object to store output csv datatable
    data_table = defaultdict(list)

    # Read in external clusters
    extClusters = read_isolate_type_from_csv(extClusterFile, mode = "external", return_dict = True)

    # Go through each cluster (as defined by poppunk) and find the external
    # clusters that had previously been assigned to any sample in the cluster
    for ppCluster in newClusters:
        # Store clusters as a set to avoid duplicates
        prevClusters = defaultdict(set)
        for sample in ppCluster:
            for extCluster in extClusters:
                if sample in extClusters[extCluster]:
                    prevClusters[extCluster].add(extClusters[extCluster][sample])

        # Go back through and print the samples that were found
        for sample in ppCluster:
            if printRef or sample not in oldNames:
                data_table["sample"].append(sample)
                for extCluster in extClusters:
                    if extCluster in prevClusters:
                        data_table[extCluster].append(";".join(prevClusters[extCluster]))
                    else:
                        data_table[extCluster].append("NA")

    if "sample" not in data_table:
        sys.stderr.write("WARNING: No new samples found, cannot write external clusters\n")
    else:
        pd.DataFrame(data=data_table).to_csv(outPrefix + "_external_clusters.csv",
                                    columns = ["sample"] + list(extClusters.keys()),
                                    index = False)

def get_gt_clique_refs(graph, reference_indices:set()):
    """Recursively prune a network of its cliques. Returns one vertex from
    a clique at each stage

    Args:
        graph (gt.Graph): The graph to get clique representatives from
        reference_indices (set): The unique list of vertices being kept, to add to
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
            get_gt_clique_refs(subgraph, reference_indices)
        elif subgraph.num_vertices() == 1:
            reference_indices.add(subgraph.get_vertices()[0])
    except StopIteration:
        pass

    return reference_indices

def gt_clique_prune(component, graph, reference_indices, components_list):
    """Wraps around get_gt_clique_refs

    Args:
        component (gt.Graph component): component of the graph to be pruned
        graph (gt.Graph): graph to be pruned
        reference_indices (set): set of reference indices
        components_list (list): list of components in the grpah to be pruned

    Returns:
        ref_list (set: set of references
    """
    if gt.openmp_enabled():
        gt.openmp_set_num_threads(1)
    sys.setrecursionlimit(RECURSION_LIMIT)
    subgraph = gt.GraphView(graph, vfilt=components_list == component)
    refs = reference_indices.copy()
    if subgraph.num_vertices() <= 2:
        refs.add(subgraph.get_vertices()[0])
        ref_list = refs
    else:
        ref_list = get_gt_clique_refs(subgraph, refs)

    return ref_list

def gt_print_clusters(graph, vertex_labels, out_prefix=None, old_cluster_file=None, external_cluster_csv=None, print_ref=True, print_csv=True, write_unwords=True):
    """Prints out a dict of cluster assignments for samples in the graph

    Args:
        graph (gt.Graph): graph for which clusters are to be assigned
        vertex_labels (lsit): list of sample names
        out_prefix (str, optional): prefix to use to name output file if writing clusters to csv. Defaults to None.
        old_cluster_file (str, optional): file name of exisitng csv with isolate assignments. Defaults to None.
        external_cluster_csv (str, optional): file name of exisitng csv with external cluster assignments. Defaults to None.
        print_ref (bool, optional): whether to print refs or not. Defaults to True.
        print_csv (bool, optional): whether to write cluster assignments to file or not. Defaults to True.
        write_unwords (bool, optional): whether to produce unique cluster names for each cluster. Defaults to True.

    Returns:
        clustering (dict): dict of cluster assignmentd
    """
    if old_cluster_file == None and print_ref == False:
        raise RuntimeError("Trying to print query clusters with no query sequences")
    if write_unwords and not print_csv:
        write_unwords = False

    # get a sorted list of component assignments
    component_assignments, component_frequencies = gt.label_components(graph)
    component_frequency_ranks = len(component_frequencies) - rankdata(component_frequencies, method = 'ordinal').astype(int)

    # use components to determine new clusters
    new_clusters = [set() for rank in range(len(component_frequency_ranks))]

    for isolate_index, isolate_name in enumerate(vertex_labels):
        component = component_assignments.a[isolate_index]
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

def gt_get_ref_graph(graph, ref_indices, vertex_labels, type_isolate):
    """Create pruned graph

    Args:
        graph (gt.Graph): graph to be pruned
        ref_indices (set): set of reference indices from which to make pruned graph
        vertex_labels (list): list of sample names
        type_isolate (str): the name of the type isolate as calculated by poppunk

    Returns:
        ref_graph: pruned graph
    """

    if type_isolate is not None:
        type_isolate_index = vertex_labels.index(type_isolate)
        if type_isolate_index not in ref_indices:
            ref_indices.add(type_isolate_index)
    else:
        type_isolate_index = None

    reference_vertex = graph.new_vertex_property('bool')
    for n, vertex in enumerate(graph.vertices()):
        if n in ref_indices:
            reference_vertex[vertex] = True
        else:
            reference_vertex[vertex] = False

    ref_graph = gt.GraphView(graph, vfilt = reference_vertex)
    ref_graph = gt.Graph(ref_graph, prune = True)
    
    clusters_in_full_graph = gt_print_clusters(graph, vertex_labels, print_csv=False)
    reference_clusters_in_full_graph = defaultdict(set)
    for reference_index in ref_indices:
        reference_clusters_in_full_graph[clusters_in_full_graph[vertex_labels[reference_index]]].add(reference_index)

    # Calculate the component membership within the reference graph
    ref_order = [name for idx, name in enumerate(vertex_labels) if idx in frozenset(ref_indices)]
    clusters_in_reference_graph = gt_print_clusters(ref_graph, ref_order, print_csv=False)
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
                        for vertex in vertex_list:
                            reference_vertex[vertex] = True
                            ref_indices.add(int(vertex))


    # update reference graph if vertices have been added
    if network_update_required:
        ref_graph = gt.GraphView(graph, vfilt = reference_vertex)
        ref_graph = gt.Graph(ref_graph, prune = True) # https://stackoverflow.com/questions/30839929/graph-tool-graphview-object


    return ref_graph