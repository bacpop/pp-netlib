import os, sys
import operator
from functools import partial
from multiprocessing import Pool
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import graph_tool.all as gt

import cugraph
import cudf

from .cliques import *
from .utils import *

def add_self_loop(edge_df, seq_num, weights = False, renumber = True):
    """Adds self-loop to cugraph graph to ensure all nodes are included in
    the graph, even if singletons.

    ## DEPENDS ON Fns: {none}

    Args:
        G_df (cudf)
            cudf data frame containing edge list
        seq_num (int)
            The expected number of nodes in the graph
        renumber (bool)
            Whether to renumber the vertices when added to the graph
    Returns:
        new_graph (graph)
            Dictionary of cluster assignments (keys are sequence names)
    """
    # use self-loop to ensure all nodes are present
    min_in_df = np.amin([edge_df["source"].min(), edge_df["destination"].min()])
    if min_in_df.item() > 0:
        G_self_loop = cudf.DataFrame()
        G_self_loop["source"] = [0]
        G_self_loop["destination"] = [0]
        if weights:
            G_self_loop["weights"] = 0.0
        edge_df = cudf.concat([edge_df,G_self_loop], ignore_index = True)
    max_in_df = np.amax([edge_df["source"].max(),edge_df["destination"].max()])
    if max_in_df.item() != seq_num:
        G_self_loop = cudf.DataFrame()
        G_self_loop["source"] = [seq_num]
        G_self_loop["destination"] = [seq_num]
        if weights:
            G_self_loop["weights"] = 0.0
        edge_df = cudf.concat([edge_df,G_self_loop], ignore_index = True)
    # Construct graph
    new_graph = cugraph.Graph()
    if weights:
        new_graph.from_cudf_edgelist(edge_df, edge_attr = "weights", renumber = renumber)
    else:
        new_graph.from_cudf_edgelist(edge_df, renumber = renumber)
    return new_graph

def translate_network_indices(graph_ref_df, reference_indices):
    """Function for ensuring an updated reference network retains
    numbering consistent with sample names

    ## DEPENDS ON Fns: {none}

       Args:
           graph_ref_df (cudf data frame)
               List of edges in reference network
           reference_indices (list)
               The ordered list of reference indices in the original network
       Returns:
           graph (cugraph network)
               Network of reference sequences
    """
    # Translate network indices to match name order
    graph_ref_df['source'] = [reference_indices.index(x) for x in graph_ref_df['old_source'].to_arrow().to_pylist()]
    graph_ref_df['destination'] = [reference_indices.index(x) for x in graph_ref_df['old_destination'].to_arrow().to_pylist()]
    graph = add_self_loop(graph_ref_df, len(reference_indices) - 1, renumber = True)
    return(graph)

def write_references(ref_list, out_prefix, out_suffix = ""):
    """Writes chosen references to file

    ## DEPENDS ON Fns: {None}

    Args:
        ref_list (list)
            Reference names to write
        out_prefix (str)
            Prefix for output file
        out_suffix (str)
            Suffix for output file (.refs will be appended)
    Returns:
        ref_filename (str)
            The name of the file references were written to
    """
    # write references to file
    ref_filename = out_prefix + "/" + os.path.basename(out_prefix) + out_suffix + ".refs"
    with open(ref_filename, "w") as r_file:
        for ref in ref_list:
            r_file.write(ref + "\n")
    return ref_filename

def print_external_clusters(new_clusters, ext_cluster_file, out_prefix, old_names, print_ref = True):
    """Prints cluster assignments with respect to previously defined
    clusters or labels.

    ## DEPENDS ON Fns: {poppunk.utils: [readIsolateTypeFromCsv]}

    Args:
        new_clusters (set iterable)
            The components from the graph G, defining the PopPUNK clusters
        ext_cluster_file (str)
            A CSV file containing definitions of the external clusters for
            each sample (does not need to contain all samples)
        out_prefix (str)
            Prefix for output CSV (_external_clusters.csv)
        old_names (list)
            A list of the reference sequences
        print_ref (bool)
            If false, print only query sequences in the output
            Default = True
    """
    # Object to store output csv datatable
    data_table = defaultdict(list)

    # Read in external clusters
    ext_clusters = \
        read_isolate_type_from_csv(ext_cluster_file,
                               mode = "external",
                               return_dict = True)

    # Go through each cluster (as defined by poppunk) and find the external
    # clusters that had previously been assigned to any sample in the cluster
    for pp_cluster in new_clusters:
        # Store clusters as a set to avoid duplicates
        prev_clusters = defaultdict(set)
        for sample in pp_cluster:
            for ext_cluster in ext_clusters:
                if sample in ext_clusters[ext_cluster]:
                    prev_clusters[ext_cluster].add(ext_clusters[ext_cluster][sample])

        # Go back through and print the samples that were found
        for sample in pp_cluster:
            if print_ref or sample not in old_names:
                data_table["sample"].append(sample)
                for ext_cluster in ext_clusters:
                    if ext_cluster in prev_clusters:
                        data_table[ext_cluster].append(";".join(prev_clusters[ext_cluster]))
                    else:
                        data_table[ext_cluster].append("NA")

    if "sample" not in data_table:
        sys.stderr.write("WARNING: No new samples found, cannot write external clusters\n")
    else:
        pd.DataFrame(data=data_table).to_csv(out_prefix + "_external_clusters.csv",
                                    columns = ["sample"] + list(ext_clusters.keys()),
                                    index = False)

def print_clusters(graph, ref_list, out_prefix=None, old_cluster_file=None, external_cluster_csv=None, print_ref=True, print_csv=True, clustering_type='combined', write_unwords=True, use_gpu = False):
    """Get cluster assignments
    Also writes assignments to a CSV file

    ## DEPENDS ON Fns: {poppunk.utils: [check_and_set_gpu, readIsolateTypeFromCsv], .: [print_external_clusters], poppunk.unword: [gen_unword]}

    Args:
        graph (graph)
            Network used to define clusters
        out_prefix (str)
            Prefix for output CSV
            Default = None
        old_cluster_file (str)
            CSV with previous cluster assignments.
            Pass to ensure consistency in cluster assignment name.
            Default = None
        external_cluster_csv (str)
            CSV with cluster assignments from any source. Will print a file
            relating these to new cluster assignments
            Default = None
        print_ref (bool)
            If false, print only query sequences in the output
            Default = True
        print_csv (bool)
            Print results to file
            Default = True
        clustering_type (str)
            Type of clustering network, used for comparison with old clusters
            Default = 'combined'
        write_unwords (bool)
            Write clusters with a pronouncable name rather than numerical index
            Default = True
        use_gpu (bool)
            Whether to use cugraph for network analysis
    Returns:
        clustering (dict)
            Dictionary of cluster assignments (keys are sequence names)
    """
    if old_cluster_file == None and print_ref == False:
        raise RuntimeError("Trying to print query clusters with no query sequences")
    if write_unwords and not print_csv:
        write_unwords = False

    # get a sorted list of component assignments
    if not use_gpu:
        component_assignments, component_frequencies = gt.label_components(graph)
        component_frequency_ranks = len(component_frequencies) - rankdata(component_frequencies, method = "ordinal").astype(int)
        # use components to determine new clusters
        new_clusters = [set() for rank in range(len(component_frequency_ranks))]
        for isolate_index, isolate_name in enumerate(ref_list):
            component = component_assignments.a[isolate_index]
            component_rank = component_frequency_ranks[component]
            new_clusters[component_rank].add(isolate_name)
    else:
        use_gpu = check_and_set_gpu(use_gpu, gpu_lib, quit_on_fail = True)
        component_assignments = cugraph.components.connectivity.connected_components(graph)
        component_frequencies = component_assignments["labels"].value_counts(sort = True, ascending = False)
        new_clusters = [set() for rank in range(component_frequencies.size)]
        for isolate_index, isolate_name in enumerate(ref_list): # assume sorted at the moment
            component = component_assignments["labels"].iloc[isolate_index].item()
            component_rank_bool = component_frequencies.index == component
            component_rank = np.argmax(component_rank_bool.to_array())
            new_clusters[component_rank].add(isolate_name)

    old_names = set()

    if old_cluster_file != None:
        all_old_clusters = read_isolate_type_from_csv(old_cluster_file, mode = 'external', return_dict = False)
        old_clusters = all_old_clusters[list(all_old_clusters.keys())[0]]
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

    for new_cluster_idx, new_cluster in enumerate(new_clusters):
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
            cls_id = new_cluster_idx + 1
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
        with open(out_filename, "w") as cluster_file:
            cluster_file.write("Taxon,Cluster\n")
            if write_unwords:
                unword_file = open(out_prefix + "_unword_clusters.csv", "w")
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

    return(clustering)

def extract_references(graph, db_order, out_prefix, out_suffix = '', type_isolate = None, existing_refs = None, threads = 1, use_gpu = False):
    """Extract references for each cluster based on cliques
       Writes chosen references to file by calling :func:`~writeReferences`

        ## DEPENDS ON Fns: {.network_io: [print_clusters, write_references], .: [translate_network_indices]}

       Args:
           graph (graph)
               A network used to define clusters
           db_order (list)
               The order of files in the sketches, so returned references are in the same order
           out_prefix (str)
               Prefix for output file
           out_suffix (str)
               Suffix for output file  (.refs will be appended)
           type_isolate (str)
               Isolate to be included in set of references
           existing_refs (list)
               References that should be used for each clique
           use_gpu (bool)
               Use cugraph for graph analysis (default = False)
       Returns:
           ref_filename (str)
               The name of the file references were written to
           references (list)
               An updated list of the reference names
    """
    if existing_refs == None:
        references = set()
        reference_indices = set()
    else:
        references = set(existing_refs)
        index_lookup = {v:k for k,v in enumerate(db_order)}
        reference_indices = set([index_lookup[r] for r in references])
    # Add type isolate, if necessary
    type_isolate_index = None
    if type_isolate is not None:
        if type_isolate in db_order:
            type_isolate_index = db_order.index(type_isolate)
        else:
            sys.stderr.write("Type isolate " + type_isolate + " not found\n")
            sys.exit(1)

    if not use_gpu:

        # Each component is independent, so can be multithreaded
        components = gt.label_components(graph)[0].a

        # Turn gt threading off and on again either side of the parallel loop
        if gt.openmp_enabled():
            gt.openmp_set_num_threads(1)

        # Cliques are pruned, taking one reference from each, until none remain
        sys.setrecursionlimit = 5000
        with Pool(processes=threads) as pool:
            ref_lists = pool.map(partial(prune_cliques,
                                            graph=graph,
                                            reference_indices=reference_indices,
                                            components_list=components),
                                 set(components))
        sys.setrecursionlimit = 1000
        # Returns nested lists, which need to be flattened
        reference_indices = set([entry for sublist in ref_lists for entry in sublist])

        # Add type isolate if necessary - before edges are added
        if type_isolate_index is not None and type_isolate_index not in reference_indices:
            reference_indices.add(type_isolate_index)

        if gt.openmp_enabled():
            gt.openmp_set_num_threads(threads)

        # Use a vertex filter to extract the subgraph of refences
        # as a graphview
        reference_vertex = graph.new_vertex_property("bool")
        for n, vertex in enumerate(graph.vertices()):
            if n in reference_indices:
                reference_vertex[vertex] = True
            else:
                reference_vertex[vertex] = False
        G_ref = gt.GraphView(graph, vfilt = reference_vertex)
        G_ref = gt.Graph(G_ref, prune = True) # https://stackoverflow.com/questions/30839929/graph-tool-graphview-object

        # Find any clusters which are represented by >1 references
        # This creates a dictionary: cluster_id: set(ref_idx in cluster)
        clusters_in_full_graph = print_clusters(graph, db_order, print_csv=False)
        reference_clusters_in_full_graph = defaultdict(set)
        for reference_index in reference_indices:
            reference_clusters_in_full_graph[clusters_in_full_graph[db_order[reference_index]]].add(reference_index)

        # Calculate the component membership within the reference graph
        ref_order = [name for idx, name in enumerate(db_order) if idx in frozenset(reference_indices)]
        clusters_in_reference_graph = print_clusters(G_ref, ref_order, print_csv=False)
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
                    component_i = reference_clusters_in_reference_graph[db_order[check[i]]]
                    for j in range(i + 1, len(check)):
                        # Add intermediate nodes
                        component_j = reference_clusters_in_reference_graph[db_order[check[j]]]
                        if component_i != component_j:
                            network_update_required = True
                            vertex_list, edge_list = gt.shortest_path(graph, check[i], check[j])
                            # update reference list
                            for vertex in vertex_list:
                                reference_vertex[vertex] = True
                                reference_indices.add(int(vertex))

        # update reference graph if vertices have been added
        if network_update_required:
            G_ref = gt.GraphView(graph, vfilt = reference_vertex)
            G_ref = gt.Graph(G_ref, prune = True) # https://stackoverflow.com/questions/30839929/graph-tool-graphview-object

    else:
        # For large network, use more approximate method for extracting references
        reference = {}
        # Record the original components to which sequences belonged
        component_assignments = cugraph.components.connectivity.connected_components(graph)
        # Leiden method has resolution parameter - higher values give greater precision
        partition_assignments, score = cugraph.leiden(graph, resolution = 0.1)
        # group by partition, which becomes the first column, so retrieve second column
        reference_index_df = partition_assignments.groupby("partition").nth(0)
        reference_indices = reference_index_df["vertex"].to_arrow().to_pylist()

        # Add type isolate if necessary - before edges are added
        if type_isolate_index is not None and type_isolate_index not in reference_indices:
            reference_indices.append(type_isolate_index)

        # Order found references as in sketchlib database
        reference_names = [db_order[int(x)] for x in sorted(reference_indices)]

        # Extract reference edges
        graph_edge_df = graph.view_edge_list()
        if "src" in graph_edge_df.columns:
            graph_edge_df.rename(columns={"src": "old_source","dst": "old_destination"}, inplace=True)
        else:
            graph_edge_df.rename(columns={"source": "old_source","destination": "old_destination"}, inplace=True)
        G_ref_df = graph_edge_df[graph_edge_df["old_source"].isin(reference_indices) & graph_edge_df["old_destination"].isin(reference_indices)]
        # Translate network indices to match name order
        G_ref = translate_network_indices(G_ref_df, reference_indices)

        # Check references in same component in overall graph are connected in the reference graph
        # First get components of original reference graph
        reference_component_assignments = cugraph.components.connectivity.connected_components(G_ref)
        reference_component_assignments.rename(columns={"labels": "ref_labels"}, inplace=True)
        # Merge with component assignments from overall graph
        combined_vertex_assignments = reference_component_assignments.merge(component_assignments,
                                                                            on = "vertex",
                                                                            how = "left")
        combined_vertex_assignments = combined_vertex_assignments[combined_vertex_assignments["vertex"].isin(reference_indices)]
        # Find the number of components in the reference graph associated with each component in the overall graph -
        # should be one if there is a one-to-one mapping of components - else links need to be added
        max_ref_comp_count = combined_vertex_assignments.groupby(["labels"], sort = False)["ref_labels"].nunique().max()
        if max_ref_comp_count > 1:
            # Iterate through components
            for component, component_df in combined_vertex_assignments.groupby(["labels"], sort = False):
                # Find components in the overall graph matching multiple components in the reference graph
                if component_df.groupby(["labels"], sort = False)["ref_labels"].nunique().iloc[0] > 1:
                    # Make a graph of the component from the overall graph
                    vertices_in_component = component_assignments[component_assignments["labels"]==component]["vertex"]
                    references_in_component = vertices_in_component[vertices_in_component.isin(reference_indices)].values
                    G_component_df = graph_edge_df[graph_edge_df["source"].isin(vertices_in_component) & graph_edge_df["destination"].isin(vertices_in_component)]
                    G_component = cugraph.Graph()
                    G_component.from_cudf_edgelist(G_component_df)
                    # Find single shortest path from a reference to all other nodes in the component
                    traversal = cugraph.traversal.sssp(G_component,source = references_in_component[0])
                    reference_index_set = set(reference_indices)
                    # Add predecessors to reference sequences on the SSSPs
                    predecessor_list = traversal[traversal["vertex"].isin(reference_indices)]["predecessor"].values
                    predecessors = set(predecessor_list[predecessor_list >= 0].flatten().tolist())
                    # Add predecessors to reference set and check whether this results in complete paths
                    # where complete paths are indicated by references' predecessors being within the set of
                    # references
                    while len(predecessors) > 0 and len(predecessors - reference_index_set) > 0:
                        reference_index_set = reference_index_set.union(predecessors)
                        predecessor_list = traversal[traversal["vertex"].isin(reference_indices)]["predecessor"].values
                        predecessors = set(predecessor_list[predecessor_list >= 0].flatten().tolist())
                    # Add expanded reference set to the overall list
                    reference_indices = list(reference_index_set)
            # Create new reference graph
            G_ref_df = graph_edge_df[graph_edge_df["old_source"].isin(reference_indices) & graph_edge_df["old_destination"].isin(reference_indices)]
            G_ref = translate_network_indices(G_ref_df, reference_indices)

    # Order found references as in sketch files
    reference_names = [db_order[int(x)] for x in sorted(reference_indices)]
    ref_filename = write_references(reference_names, out_prefix, out_suffix = out_suffix)
    return reference_indices, reference_names, ref_filename, G_ref

def generate_minimum_spanning_tree(graph, from_cugraph = False):
    """Generate a minimum spanning tree from a network

    ## DEPENDS ON Fns: {none}

    Args:
       graph (network)
           Graph tool network
       from_cugraph (bool)
            If a pre-calculated MST from cugraph
            [default = False]
    Returns:
       mst_network (str)
           Minimum spanning tree (as graph-tool graph)
    """
    #
    # Create MST
    #
    if from_cugraph:
        mst_network = graph
    else:
        sys.stderr.write("Starting calculation of minimum-spanning tree\n")

        # Test if weighted network and calculate minimum spanning tree
        if "weight" in graph.edge_properties:
            mst_edge_prop_map = gt.min_spanning_tree(graph, weights = graph.ep["weight"])
            mst_network = gt.GraphView(graph, efilt = mst_edge_prop_map)
            mst_network = gt.Graph(mst_network, prune = True)
        else:
            sys.stderr.write("generate_minimum_spanning_tree requires a weighted graph\n")
            raise RuntimeError("MST passed unweighted graph")

    # Find seed nodes as those with greatest outdegree in each component
    num_components = 1
    seed_vertices = set()
    if from_cugraph:
        mst_df = cugraph.components.connectivity.connected_components(mst_network)
        num_components_idx = mst_df["labels"].max()
        num_components = mst_df.iloc[num_components_idx]["labels"]
        if num_components > 1:
            mst_df["degree"] = mst_network.in_degree()["degree"]
            # idxmax only returns first occurrence of maximum so should maintain
            # MST - check cuDF implementation is the same
            max_indices = mst_df.groupby(["labels"])["degree"].idxmax()
            seed_vertices = mst_df.iloc[max_indices]["vertex"]
    else:
        component_assignments, component_frequencies = gt.label_components(mst_network)
        num_components = len(component_frequencies)
        if num_components > 1:
            for component_index in range(len(component_frequencies)):
                component_members = component_assignments.a == component_index
                component = gt.GraphView(mst_network, vfilt = component_members)
                component_vertices = component.get_vertices()
                out_degrees = component.get_out_degrees(component_vertices)
                seed_vertex = list(component_vertices[np.where(out_degrees == np.amax(out_degrees))])
                seed_vertices.add(seed_vertex[0]) # Can only add one otherwise not MST


    # If multiple components, add distances between seed nodes
    if num_components > 1:

        # Extract edges and maximum edge length - as DF for cugraph
        # list of tuples for graph-tool
        if from_cugraph:
            # With cugraph the MST is already calculated
            # so no extra edges can be retrieved from the graph
            G_df = graph.view_edge_list()
            max_weight = G_df["weights"].max()
            first_seed = seed_vertices.iloc[0]
            G_seed_link_df = cudf.DataFrame()
            G_seed_link_df["dst"] = seed_vertices.iloc[1:seed_vertices.size]
            G_seed_link_df["src"] = seed_vertices.iloc[0]
            G_seed_link_df["weights"] = seed_vertices.iloc[0]
            G_df = G_df.append(G_seed_link_df)
        else:
            # With graph-tool look to retrieve edges in larger graph
            connections = []
            max_weight = float(np.max(graph.edge_properties["weight"].a))

            # Identify edges between seeds to link components together
            for ref in seed_vertices:
                seed_edges = graph.get_all_edges(ref, [graph.ep["weight"]])
                found = False  # Not all edges may be in graph
                for seed_edge in seed_edges:
                    if seed_edge[1] in seed_vertices:
                        found = True
                        connections.append((seed_edge))
                # TODO: alternative would be to requery the DB (likely quick)
                if found == False:
                    for query in seed_vertices:
                        if query != ref:
                            connections.append((ref, query, max_weight))

        # Construct graph
        if from_cugraph:
            mst_network = cugraph.Graph()
            G_df.rename(columns={"src": "source","dst": "destination"}, inplace=True)
            mst_network.from_cudf_edgelist(G_df, edge_attr="weights", renumber=False)
        else:
            seed_G = gt.Graph(directed = False)
            seed_G.add_vertex(len(seed_vertex))
            eweight = seed_G.new_ep("float")
            seed_G.add_edge_list(connections, eprops = [eweight])
            seed_G.edge_properties["weight"] = eweight
            seed_mst_edge_prop_map = gt.min_spanning_tree(seed_G, weights = seed_G.ep["weight"])
            seed_mst_network = gt.GraphView(seed_G, efilt = seed_mst_edge_prop_map)
            # Insert seed MST into original MST - may be possible to use graph_union with include=True & intersection
            deep_edges = seed_mst_network.get_edges([seed_mst_network.ep["weight"]])
            mst_network.add_edge_list(deep_edges)

    sys.stderr.write("Completed calculation of minimum-spanning tree\n")

    return mst_network
