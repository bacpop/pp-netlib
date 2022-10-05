import cugraph
import cudf
import numpy as np

def add_self_loop(graph_df, seq_num, weights = False, renumber = True):
    """Adds self-loop to cugraph graph to ensure all nodes are included in
    the graph, even if singletons.
    Args:
        G_df (cudf)
            cudf data frame containing edge list
        seq_num (int)
            The expected number of nodes in the graph
        renumber (bool)
            Whether to renumber the vertices when added to the graph
    Returns:
        G_new (graph)
            Dictionary of cluster assignments (keys are sequence names)
    """
    # use self-loop to ensure all nodes are present
    min_in_df = np.amin([graph_df["source"].min(), graph_df["destination"].min()])
    if min_in_df.item() > 0:
        graph_self_loop = cudf.DataFrame()
        graph_self_loop["source"] = [0]
        graph_self_loop["destination"] = [0]
        if weights:
            graph_self_loop["weights"] = 0.0
        graph_df = cudf.concat([graph_df,graph_self_loop], ignore_index = True)
    max_in_df = np.amax([graph_df["source"].max(),graph_df["destination"].max()])
    if max_in_df.item() != seq_num:
        graph_self_loop = cudf.DataFrame()
        graph_self_loop["source"] = [seq_num]
        graph_self_loop["destination"] = [seq_num]
        if weights:
            graph_self_loop["weights"] = 0.0
        graph_df = cudf.concat([graph_df,graph_self_loop], ignore_index = True)
    # Construct graph
    new_graph = cugraph.Graph()
    if weights:
        new_graph.from_cudf_edgelist(graph_df, edge_attr = "weights", renumber = renumber)
    else:
        new_graph.from_cudf_edgelist(graph_df, renumber = renumber)
    return new_graph

def translate_network_indices(ref_graph_df, reference_indices):
    """Function for ensuring an updated reference network retains
    numbering consistent with sample names
       Args:
           G_ref_df (cudf data frame)
               List of edges in reference network
           reference_indices (list)
               The ordered list of reference indices in the original network
       Returns:
           G_ref (cugraph network)
               Network of reference sequences
    """
    # Translate network indices to match name order
    ref_graph_df["source"] = [reference_indices.index(x) for x in ref_graph_df["old_source"].to_arrow().to_pylist()]
    ref_graph_df["destination"] = [reference_indices.index(x) for x in ref_graph_df["old_destination"].to_arrow().to_pylist()]
    ref_graph = add_self_loop(ref_graph_df, len(reference_indices) - 1, renumber = True)
    return ref_graph

def extract_cu_refs(graph, vertex_labels, type_isolate = None):
    reference = {}
    # Record the original components to which sequences belonged
    component_assignments = cugraph.components.connectivity.connected_components(graph)
    # Leiden method has resolution parameter - higher values give greater precision
    partition_assignments, score = cugraph.leiden(graph, resolution = 0.1)
    # group by partition, which becomes the first column, so retrieve second column
    reference_index_df = partition_assignments.groupby("partition").nth(0)
    reference_indices = reference_index_df["vertex"].to_arrow().to_pylist()

    # Add type isolate if necessary - before edges are added
    if type_isolate is not None:
        type_isolate_index = vertex_labels.index(type_isolate)
        if type_isolate_index not in reference_indices:
            reference_indices.append(type_isolate_index)

    # Order found references as in sketchlib database
    reference_names = [vertex_labels[int(x)] for x in sorted(reference_indices)]

    # Extract reference edges
    graph_df = graph.view_edge_list()
    if 'src' in graph_df.columns:
        graph_df.rename(columns={"src": "old_source","dst": "old_destination"}, inplace=True)
    else:
        graph_df.rename(columns={"source": "old_source","destination": "old_destination"}, inplace=True)
    ref_graph_df = graph_df[graph_df['old_source'].isin(reference_indices) & graph_df['old_destination'].isin(reference_indices)]
    # Translate network indices to match name order
    ref_graph = translate_network_indices(ref_graph_df, reference_indices)

    # Check references in same component in overall graph are connected in the reference graph
    # First get components of original reference graph
    reference_component_assignments = cugraph.components.connectivity.connected_components(ref_graph)
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
                graph_component_df = graph_df[graph_df["source"].isin(vertices_in_component) & graph_df["destination"].isin(vertices_in_component)]
                component_graph = cugraph.Graph()
                component_graph.from_cudf_edgelist(graph_component_df)
                # Find single shortest path from a reference to all other nodes in the component
                traversal = cugraph.traversal.sssp(component_graph,source = references_in_component[0])
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
        ref_graph_df = graph_df[graph_df["old_source"].isin(reference_indices) & graph_df["old_destination"].isin(reference_indices)]
        ref_graph = translate_network_indices(ref_graph_df, reference_indices)

    return ref_graph