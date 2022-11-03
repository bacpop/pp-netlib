########################
####   .CONSTRUCT   ####
########################
from collections import defaultdict
import subprocess
import scipy
from scipy.stats import rankdata
import numpy as np
import pandas as pd
import os, sys

def get_edge_list(network_data, weights = None):
    """Get an edge list from input data.

    Args:
        network_data (dataframe OR edge list OR sparse coordinate matrix): Data describing graph edges
        weights (bool/list, optional): Whether edges are weighted. Defaults to None. If not None, it is expected to be a list of weights (floats).
            If list, weights are expected to be in the same order as their corresponding edges in network data.

    Returns:
        vertex_map (dict): Dictionary of sample names taken from network_data; sample names are keys, their index in a sorted list of names is the value
        edge_list (list of tuples): Edge list as (source, target) if unweighted data and (source, target, weight) if weighted data provided
    """
    ########################
    ####    DF INPUT    ####
    ########################
    if isinstance(network_data, pd.DataFrame):
        vertices = set(network_data["source"]).union(set(network_data["target"]))
        vertex_map = {}
        for idx, vertex in enumerate(sorted(vertices)):
            vertex_map[vertex] = idx

        sources = [vertex_map[vertex] for vertex in network_data["source"]]
        targets = [vertex_map[vertex] for vertex in network_data["target"]]
        
        ## add weights column if weights provided as list (add error catching?)
        if weights is not None:
            if isinstance(weights, list):
                edge_list = list(zip(sources, targets, weights))
            else:
                try:
                    weights = network_data["weights"]
                    edge_list = list(zip(sources, targets, weights))
                except KeyError as ke:
                    raise ke
    
        else:
            edge_list = list(zip(sources, targets))

    ##########################
    #### SPARSE MAT INPUT ####
    ##########################
    elif isinstance(network_data, scipy.sparse.coo_matrix):
        sources = list(network_data.row)
        targets = list(network_data.col)
        weights = list(network_data.data)

        vertices = set(sources).union(set(targets))
        vertex_map = {}
        for idx, vertex in enumerate(sorted(vertices)):
            vertex_map[vertex] = idx
        source_idx = [vertex_map[vertex] for vertex in sources]
        target_idx = [vertex_map[vertex] for vertex in targets]

        edge_list = list(zip(source_idx, target_idx, weights))
        

    ########################
    ####   LIST INPUT   ####
    ########################
    elif isinstance(network_data, list):

        if weights is None:
            sources, targets = zip(*network_data)
            vertices = set(sources).union(set(targets))
            vertex_map = {}
            for idx, vertex in enumerate(sorted(vertices)):
                vertex_map[vertex] = idx
            source_idx = [vertex_map[vertex] for vertex in sources]
            target_idx = [vertex_map[vertex] for vertex in targets]
            edge_list = list(zip(source_idx, target_idx))

        else:
            try:
                sources, targets, weights = zip(*network_data)
                vertices = set(sources).union(set(targets))
                vertex_map = {}
                for idx, vertex in enumerate(sorted(vertices)):
                    vertex_map[vertex] = idx
                source_idx = [vertex_map[vertex] for vertex in sources]
                target_idx = [vertex_map[vertex] for vertex in targets]

                edge_list = list(zip(source_idx, target_idx, weights))
            except ValueError as ve:
                if isinstance(weights, list):
                    sources, targets = zip(*network_data)
                    vertices = set(sources).union(set(targets))
                    vertex_map = {}
                    for idx, vertex in enumerate(sorted(vertices)):
                        vertex_map[vertex] = idx
                    source_idx = [vertex_map[vertex] for vertex in sources]
                    target_idx = [vertex_map[vertex] for vertex in targets]

                    edge_list = list(zip(source_idx, target_idx, weights))
                else:
                    raise ve
        
    return vertex_map, edge_list

def construct_graph(network_data, vertex_labels, backend, weights = None):
    """Construct graph from edge list produced by get_edge_list

    Args:
        network_data (dataframe OR edge list OR sparse coordinate matrix): Data describing graph edges
        vertex_labels (list): List of sample names
        backend (str): Graphing module to use ("GT"/"NX")
        weights (bool/list, optional): Whether edges are weighted. Defaults to None. If not None, it is expected to be a list of weights (floats).
            If list, weights are expected to be in the same order as their corresponding edges in network data.

    Returns:
        graph (gt.Graph or nx.Graph): Graph populated with nodes, edges; nodes have attribute "id"

        ***
        NB: Note that vertex_labels are sorted before being passed to this function
        
        If edges in network_data are of type (0,1), (0,2)...: and vertex_labels = ["s1", "s2", "s3"];
        Edge 1 is ("s1", "s2");, edge 2 is ("s1", "s3")
        
        If edges in network_data are of type ("s1", "s2"), ("s1", "s3")...: and vertex_labels = ["s1", "s2", "s3"];
        Edge 1 and edge 2 are as above. 
        ***
    """

    vertex_map, edge_list = get_edge_list(network_data=network_data, weights=weights)

    ## set weights to True is input data is a sparse matrix
    if isinstance(network_data, scipy.sparse.coo_matrix):
        weights = True

    if backend == "GT":
        import graph_tool.all as gt

        graph = gt.Graph(directed = False) ## initialise graph_tool graph object
        if weights is not None:
            eweight = graph.new_ep("float")
            graph.add_edge_list(edge_list, eprops = [eweight])
            graph.edge_properties["weight"] = eweight
        else:
            graph.add_edge_list(edge_list)

        v_name_prop = graph.new_vp("string")
        graph.vertex_properties["id"] = v_name_prop
        for idx in vertex_map.values():
            v_name_prop[idx] = vertex_labels[idx]

    elif backend == "NX":
        import networkx as nx
    
        ## initialise nx graph and add nodes
        graph = nx.Graph()
        nodes_list = [(i, dict(id=vertex_labels[i])) for i in range(len(vertex_labels))]
        graph.add_nodes_from(nodes_list)

        if weights is not None:
            graph.add_weighted_edges_from(edge_list)
        else:
            graph.add_edges_from(edge_list)

        # for idx in vertex_map.values():
        #     graph.nodes[idx]["id"] = vertex_labels[idx]
    
    elif backend == "CU":
            raise NotImplementedError("GPU graph not yet implemented")

    return graph, vertex_map

########################
####   .SUMMARISE   ####
########################
def summarise(graph, backend):
    """Get graph metrics and format into soutput string.

    Args:
        graph (Network object): The graph for which to obtain summary metrics
        backend (str): The tool used to build graph ("GT", "NX", or "CU"(TODO))

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
####      .SAVE     ####
########################
def prepare_graph(graph, backend, labels = None, clustering = None):
    """Prepare a graph, by checking whether the graph nodes have id and component membership/clusteing attributes, and graph edges have weights.
        If "id" attribute missing, labels are applied as "id"; if labels not provided, "id" is set as "node_(node)"
        If "comp_membership" attribute missing, clusters are calculated and stored as node attributes
        If "weight" attribute missing, arbitrary weights added (TODO: is this a good idea?)

    Args:
        graph (gt.Graph or nx.Graph): Graph to prepare
        labels (list): Vertex labels to apply as ids if nodes do not have id attributes
        backend (str): Whether the graph passed to the function is from "GT" or "NX" 
    """

    def prep_gt(gt_graph, labels, clustering):
        import graph_tool.all as gt
        ## check that nodes have labels -- required
        if "id" not in gt_graph.vertex_properties:
            ## if no list of labels is provided, improvise node ids such that for node "i", id=str(i+1)
            if labels is None:
                v_name_prop = gt_graph.new_vp("string")
                gt_graph.vertex_properties["id"] = v_name_prop
                for i, vertex in enumerate(gt_graph.vertices()):
                    v_name_prop[vertex] = f"node{i+1}"

            ## if a list of labels is provided, apply labels to nodes
            else:
                v_name_prop = gt_graph.new_vp("string")
                gt_graph.vertex_properties["id"] = v_name_prop
                for vertex, label in zip(gt_graph.vertices(), labels):
                    v_name_prop[vertex] = label

        ## check if comp_membership is assigned -- make this consistent with clustering by making sure that comp 1 is the largest. Here, calling get_gt_clusters
        if "comp_membership" not in gt_graph.vertex_properties:
            if clustering is None: ## if previous clustering not give, calculate clustering -- used primarily for load_network
                clustering = get_gt_clusters(gt_graph)

            vprop = gt_graph.new_vp("string")
            gt_graph.vp.comp_membership = vprop
            for vertex in gt_graph.iter_vertices():
                ## comp membership of vertex = the value corresponding to the id of that vertex in clustering
                gt_graph.vp.comp_membership[vertex] = str(clustering[gt_graph.vp.id[vertex]])

        elif "comp_membership" in gt_graph.vertex_properties:
            # sys.stderr.write("Checking if node component memberships need updating...")
            component_assignments, component_frequencies = gt.label_components(graph)
            for component_index in range(len(component_frequencies)):
                component_members = component_assignments.a == component_index
                component = gt.GraphView(graph, vfilt = component_members)
                component_vertices = component.get_vertices()
                old_comp_memberships = list(set(graph.vp.comp_membership[v] for v in component_vertices))
                if len(old_comp_memberships) > 1:
                    sys.stderr.write("Updating...\n")
                    new_comp = "_".join(str(i) for i in old_comp_memberships)
                    for v in component_vertices:
                        graph.vp.comp_membership[v] = new_comp

        ## check if edges have weights -- not required for most processes #TODO: is adding arbitrary weights a good idea? Weights are needed for graph viz.
        if "weight" not in gt_graph.edge_properties:
            sys.stderr.write("Graph edges are not weighted.\n")

        return gt_graph


    def prep_nx(nx_graph, labels, clustering):
        import networkx as nx
        node_attrs = list(nx_graph.nodes(data=True))[0][-1].keys() # get keys of attribute dictionary associated with the first node, ie node attributes
        edge_attrs = list(nx_graph.edges(data=True))[0][-1].keys() # get keys of attribute dictionary associated with the first edge, ie edge attributes
        ## check that nodes have labels -- required
        if "id" not in node_attrs:
            ## if no list of labels is provided, improvise node ids such that for node "i", id=str(i+1)
            if labels is None:
                for idx, v in enumerate(nx_graph.nodes()):
                    nx_graph.nodes[v]["id"] = "node_"+str(idx+1)

            ## if a list of labels is provided, apply labels to nodes
            else:
                for i, v in enumerate(nx_graph.nodes()):
                    nx_graph.nodes[v]["id"] = labels[i]

        ## check if comp_membership is assigned -- could make things easier?
        if "comp_membership" not in node_attrs:
            if clustering is None: ## if previous clustering not give, calculate clustering -- used primarily for load_network
                clustering = get_nx_clusters(nx_graph)

            for v in graph.nodes():
                graph.nodes[v]["comp_membership"] = clustering[v]
        ## check if comp_memberships are up to date
        elif "comp_membership" in node_attrs:
            # sys.stderr.write("Checking if node component memberships need updating...\n")
            for comp in sorted(nx.connected_components(graph), key=len, reverse=True):
                old_comp_memberships = list(set(graph.nodes.data("comp_membership")[v] for v in comp))
                if len(old_comp_memberships) > 1:
                    sys.stderr.write("Updating...\n")
                    new_comp = "_".join(str(i) for i in old_comp_memberships)
                    for v in comp:
                        graph.nodes[v]["comp_membership"] = new_comp

        ## check if edges have weights -- not required for most processes
        if "weight" not in edge_attrs:
            sys.stderr.write("Graph edges are not weighted.\n")

        return nx_graph

    if graph is None:
        raise RuntimeError("Graph not constructed or loaded")

    else:
        if backend == "GT":
            graph = prep_gt(graph, labels, clustering)
        elif backend == "NX":
            graph = prep_nx(graph, labels, clustering)

    return graph

def save_graph(graph, backend, outdir, file_name, file_format):
    if backend == "GT":
        if file_format is None:
            graph.save(os.path.join(outdir, file_name+".graphml"))
        elif file_format is not None:
            if file_format not in [".gt", ".graphml"]:
                raise NotImplementedError("Supported file formats to save a graph-tools graph are .gt or .graphml")
            else:
                graph.save(os.path.join(outdir, file_name+file_format))

    if backend == "NX":
        import networkx as nx
        nx.write_graphml(graph, os.path.join(outdir, file_name+".graphml"))

########################
####   .VISUALISE   ####
########################
def gt_generate_mst(graph):
    """Create a minimum-spanning tree with graph-tool

    Args:
        graph (gt.Graph): Graph from which to calculate mst

    Returns:
        gt.Graph: the mst generated from the graph
    """
    import graph_tool.all as gt

    mst_edge_prop_map = gt.min_spanning_tree(graph, weights = graph.ep["weight"])
    mst_network = gt.GraphView(graph, efilt = mst_edge_prop_map)
    mst_network = gt.Graph(mst_network, prune = True)
    seed_vertices = set()

    component_assignments, component_frequencies = gt.label_components(mst_network)
    num_components = len(component_frequencies)
    ## if more than one component, get the node from each component, that has the highest out_degree
    if num_components > 1:
        for component_index in range(len(component_frequencies)):
            component_members = component_assignments.a == component_index
            component = gt.GraphView(mst_network, vfilt = component_members)
            component_vertices = component.get_vertices()
            out_degrees = component.get_out_degrees(component_vertices)
            seed_vertex = list(component_vertices[np.where(out_degrees == np.amax(out_degrees))])
            seed_vertices.add(seed_vertex[0]) # Can only add one otherwise not MST

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
        seed_G = gt.Graph(directed = False)
        seed_G.add_vertex(len(seed_vertices))
        eweight = seed_G.new_ep("float")
        seed_G.add_edge_list(connections, eprops = [eweight])
        seed_G.edge_properties["weight"] = eweight
        seed_mst_edge_prop_map = gt.min_spanning_tree(seed_G, weights = seed_G.ep["weight"])
        seed_mst_network = gt.GraphView(seed_G, efilt = seed_mst_edge_prop_map)
        # Insert seed MST into original MST - may be possible to use graph_union with include=True & intersection
        deep_edges = seed_mst_network.get_edges([seed_mst_network.ep["weight"]])
        mst_network.add_edge_list(deep_edges)

    return mst_network

def nx_generate_mst(graph):
    """Create a minimum-spanning tree with networkx

    Args:
        graph (nx.Graph): Graph from which to calculate mst

    Returns:
        nx.Graph: the mst generated from the graph
    """
    import networkx as nx

    mst_network = nx.minimum_spanning_tree(graph)

    seed_vertices = set()

    num_components = nx.number_connected_components(mst_network)
    if num_components > 1:
        for component in nx.connected_components(mst_network):
            # comp_graphview = nx.subgraph(mst_network, component)
            comp_vertices = list(component)
            out_degrees = np.array(j for (i, j) in mst_network.out_degree(comp_vertices))
            seed_vertex = list(comp_vertices[np.where(out_degrees == np.amax(out_degrees))])
            seed_vertices.add(seed_vertex[0])

    if num_components >1:
        connections = []
        weights = list(k for (i, j, k) in graph.edges.data("weight"))
        max_weight = float(np.max(weights))

        for ref in seed_vertices:
            seed_edges = graph.edges(ref).data("weight")
            found = False
            for seed_edge in seed_edges:
                if seed_edge[1] in seed_vertices:
                    found = True
                    connections.append((seed_edge))
    
            if found == False:
                for query in seed_vertices:
                    if query != ref:
                        connections.append((ref, query, max_weight))

        seed_G = nx.Graph()
        seed_G.add_nodes_from(len(seed_vertex))
        seed_G.add_weighted_edges_from(connections)

        seed_mst_network = nx.minimum_spanning_tree(seed_G)
        deep_edges = list(seed_mst_network.edges(data = "weight"))
        mst_network.add_weighted_edges_from(deep_edges)

    return mst_network

def generate_mst_network(graph, backend):
    unweighted_error_msg = RuntimeError("MST passed unweighted graph, weighted tree required.")

    if backend == "GT":
        if "weight" not in graph.edge_properties:
            raise unweighted_error_msg
        else:
            mst_network = gt_generate_mst(graph)

    elif backend == "NX":
        if "weight" not in list(list(graph.edges(data=True))[0][-1].keys()): ## https://stackoverflow.com/questions/63610396/how-to-get-the-list-of-edge-attributes-of-a-networkx-graph
            raise unweighted_error_msg
        else:
            mst_network = nx_generate_mst(graph) ##TODO

    elif backend == "CU":
        raise NotImplementedError("GPU graph not yet implemented")
        # if not graph.is_weighted():
        #     raise unweighted_error_msg
        # else:
        #     mst_network = cu_generate_mst(graph)

    return mst_network

def get_cluster_dict(clusters):
    clustering = {}
    for new_cls_idx, new_cluster in enumerate(clusters):
        cls_id = new_cls_idx + 1
        for cluster_member in new_cluster:
            clustering[cluster_member] = cls_id
    return clustering

def get_gt_clusters(graph):
    """Calculate clusters from graph-tool graph

    Args:
        graph (gt.Graph): the graph from which to compute clusters

    Returns:
        dict: dictionary with node id as key and cluster number as value
    """
    import graph_tool.all as gt

    vertex_labels = list(str(graph.vp["id"][v]) for v in graph.vertices())

    # get a sorted list of component assignments
    component_assignments, component_frequencies = gt.label_components(graph)
    component_frequency_ranks = len(component_frequencies) - rankdata(component_frequencies, method = 'ordinal').astype(int)

    # use components to determine new clusters
    new_clusters = [set() for rank in range(len(component_frequency_ranks))]

    for isolate_index, isolate_name in enumerate(vertex_labels):
        component = component_assignments.a[isolate_index]
        component_rank = component_frequency_ranks[component]
        new_clusters[component_rank].add(isolate_name)

    clustering = get_cluster_dict(new_clusters)

    return clustering

def get_nx_clusters(graph):
    """Calculate clusters from networkx graph

    Args:
        graph (nx.Graph): the graph from which to compute clusters

    Returns:
        dict: dictionary with node id as key and cluster number as value
    """
    import networkx as nx

    new_clusters = sorted(nx.connected_components(graph), key=len, reverse=True)
    clustering = get_cluster_dict(new_clusters)

    return clustering

def draw_gt_mst(mst, out_prefix, isolate_clustering, overwrite):
    """Plot a layout of the minimum spanning tree with graph-tool
    Args:
        mst (gt.Graph): A minimum spanning tree
        out_prefix (str): Output prefix for save files
        isolate_clustering (dict): Dictionary of ID: cluster, used for colouring vertices
        overwrite (bool): Overwrite existing output files
    """
    import graph_tool.all as gt
    
    graph1_file_name = out_prefix + "_mst_stress_plot.png"
    graph2_file_name = out_prefix + "_mst_cluster_plot.png"
    if overwrite or not os.path.isfile(graph1_file_name) or not os.path.isfile(graph2_file_name):
        sys.stderr.write("Drawing MST\n")
        pos = gt.sfdp_layout(mst)
        if overwrite or not os.path.isfile(graph1_file_name):
            deg = mst.degree_property_map("total")
            deg.a = 4 * (np.sqrt(deg.a) * 0.5 + 0.4)
            ebet = gt.betweenness(mst)[1]
            #print(list(ebet))
            ebet.a /= ebet.a.max() / 50.
            eorder = ebet.copy()
            eorder.a *= -1
            gt.graph_draw(mst, pos=pos, vertex_size=gt.prop_to_size(deg, mi=20, ma=50),
                            vertex_fill_color=deg, vorder=deg,
                            edge_color=ebet, eorder=eorder, edge_pen_width=ebet,
                            output=graph1_file_name, output_size=(3000, 3000))
        if overwrite or not os.path.isfile(graph2_file_name):
            cluster_fill = {}
            for cluster in set(isolate_clustering.values()):
                cluster_fill[cluster] = list(np.random.rand(3)) + [0.9]
            plot_color = mst.new_vertex_property('vector<double>')
            mst.vertex_properties['plot_color'] = plot_color
            for v in mst.vertices():
                plot_color[v] = cluster_fill[isolate_clustering[mst.vp.id[v]]]

            gt.graph_draw(mst, pos=pos, vertex_fill_color=mst.vertex_properties['plot_color'],
                    output=graph2_file_name, output_size=(3000, 3000))

def draw_nx_mst(mst, out_prefix, isolate_clustering, overwrite):
    """Plot a layout of the minimum spanning tree with networkx
    Args:
        mst (nx.Graph): A minimum spanning tree
        out_prefix (str): Output prefix for save files
        isolate_clustering (dict): Dictionary of ID: cluster, used for colouring vertices
        overwrite (bool): Overwrite existing output files
    """
    import networkx as nx
    
    import matplotlib.pyplot as plt

    graph1_file_name = out_prefix + "_mst_stress_plot.png"
    graph2_file_name = out_prefix + "_mst_cluster_plot.png"

    if overwrite or not os.path.isfile(graph1_file_name) or not os.path.isfile(graph2_file_name):
        sys.stderr.write("Drawing MST\n")
        pos = nx.spring_layout(mst, seed=13)
        if overwrite or not os.path.isfile(graph1_file_name):
            deg = list(degree for (node, degree) in mst.degree(mst.nodes()))
            deg = 4 * (np.sqrt(deg) * 0.5 + 0.4)
            ebet = np.array(list((nx.edge_betweenness_centrality(mst)).values()))
            ebet /= ebet.max() / 10.
            plt.figure(figsize=(10,7))
            nx.draw_networkx_nodes(mst, pos=pos, node_size=10*deg, node_color=deg)
            nx.draw_networkx_edges(mst, pos=pos, edge_color=ebet, width=ebet)
            plt.axis("off")
            plt.savefig(graph1_file_name)
            plt.close()

        if overwrite or not os.path.isfile(graph2_file_name):
            cluster_fill = {}
            for cluster in set(isolate_clustering.values()):
                cluster_fill[cluster] = list(np.random.rand(3)) + [0.9]
            for v in mst.nodes():
                mst.nodes[v]["plot_color"] = cluster_fill[isolate_clustering[v]]

            plt.figure(figsize=(10,7))
            nx.draw_networkx_nodes(mst, pos=pos, node_size=100, node_color=[color for (node, color) in mst.nodes(data="plot_color")])
            nx.draw_networkx_edges(mst, pos=pos)
            plt.axis("off")
            plt.savefig(graph2_file_name)
            plt.close()

def gt_save_graph_components(graph, out_prefix, outdir):
    """Save individual components of a graph-tool graph

    Args:
        graph (gt.Graph): Graph to save
        out_prefix (str): prefix to be applied to output files
        outdir (path/str): path to directory to save outputs to 
    """
    import graph_tool.all as gt

    component_assignments, component_hist = gt.label_components(graph)
    for component_idx in range(len(component_hist)):
        remove_list = []
        for vidx, v_component in enumerate(component_assignments.a):
            if v_component != component_idx:
                remove_list.append(vidx)
        G_copy = graph.copy()
        G_copy.remove_vertex(remove_list)

        save_graph(G_copy, "GT", outdir, f"{out_prefix}_component_{str(component_idx + 1)}", ".graphml")
        del G_copy

def nx_save_graph_components(graph, out_prefix, outdir):
    """Save individual components of a networkx graph

    Args:
        graph (nx.Graph): Graph to save
        out_prefix (str): prefix to be applied to output files
        outdir (path/str): path to directory to save outputs to 
    """
    import networkx as nx

    for idx, c in enumerate(nx.connected_components(graph)):
        subgraph = graph.subgraph(c)
        save_graph(subgraph, "NX", outdir, f"{out_prefix}_component_{idx + 1}", ".graphml")
        del subgraph

########################
####   .METADATA    ####
########################
def gt_get_graph_data(graph):
    weighted = False
    if "weight" in graph.edge_properties:
        weighted = True

    edge_data = defaultdict(list)
    node_data = defaultdict(list)
    if weighted:
        edge_list = list(graph.ep["weight"].a)

    for idx, e in enumerate(graph.iter_edges()):
        source_node = graph.vp.id[e[0]]
        target_node = graph.vp.id[e[1]]
        if weighted:
            edge_weight = edge_list[idx]
            edge_data[idx] = [source_node, target_node, edge_weight]
        else:
            edge_data[idx] = [source_node, target_node]

    for idx, v in enumerate(graph.iter_vertices()):
        node_id = graph.vp.id[v]
        node_comp = graph.vp.comp_membership[v]
        node_data[idx] = [node_id, node_comp]

    return edge_data, node_data

def nx_get_graph_data(graph):
    weighted = False
    edge_attrs = list(graph.edges(data=True))[0][-1].keys()
    if "weight" in edge_attrs:
        weighted = True
    edge_data = defaultdict(list)
    node_data = defaultdict(list)

    if weighted:
        for idx, (s, t, w) in enumerate(graph.edges.data("weight")):
            edge_data[idx] = [graph.nodes()[s]["id"], graph.nodes()[t]["id"], w]
    else:
        for idx, (s, t) in enumerate(graph.edges(data=False)):
            edge_data[idx] = [graph.nodes()[s]["id"], graph.nodes()[t]["id"]]

    for idx, (v, v_data) in enumerate(graph.nodes(data=True)):
        node_data[idx] = [v_data["id"], v_data["comp_membership"]]
    
    return edge_data, node_data

def write_cytoscape_csv(outfile, node_names, clustering, epi_csv = None, suffix = '_Cluster'):
    colnames = []
    colnames = ['id']
    for cluster_type in clustering:
        col_name = cluster_type + suffix
        colnames.append(col_name)
    
    # process epidemiological data
    d = defaultdict(list)

    # process epidemiological data without duplicating names
    # used by PopPUNK
    if epi_csv is not None:
        columns_to_be_omitted = ['id', 'Id', 'ID', 'combined_Cluster__autocolour',
        'core_Cluster__autocolour', 'accessory_Cluster__autocolour',
        'overall_Lineage']
        epiData = pd.read_csv(epi_csv, index_col = False, quotechar='"')
        epiData.index = [name.split('/')[-1].replace('.','_').replace(':','').replace('(','_').replace(')','_') for name in (epiData.iloc[:,0])]
        for e in epiData.columns.values:
            if e not in columns_to_be_omitted:
                colnames.append(str(e))

    # get example clustering name for validation
    example_cluster_title = list(clustering.keys())[0]

    for name in node_names:
        d['id'].append(name)
        for cluster_type in clustering:
            col_name = cluster_type + suffix
            d[col_name].append(clustering[name])

    if epi_csv is not None:
        if name in epiData.index:
            for col, value in zip(epiData.columns.values, epiData.loc[name].values):
                if col not in columns_to_be_omitted:
                    d[col].append(str(value))

    # print CSV
    sys.stderr.write("Parsed data, now writing to CSV\n")
    try:
        pd.DataFrame(data=d).to_csv(outfile, columns = colnames, index = False)
    except subprocess.CalledProcessError as e:
        sys.stderr.write("Problem with epidemiological data CSV; returned code: " + str(e.returncode) + "\n")
        # check CSV
        prev_col_items = -1
        prev_col_name = "unknown"
        for col in d:
            this_col_items = len(d[col])
            if prev_col_items > -1 and prev_col_items != this_col_items:
                sys.stderr.write("Discrepant length between " + prev_col_name + \
                                 " (length of " + prev_col_items + ") and " + \
                                 col + "(length of " + this_col_items + ")\n")
        sys.exit(1)


