## internal imports
from .utils import *
from .cliques import *
from .vertices import *
from .load_network import *
from .indices_refs_clusters import *
from .construct_network import *

## placeholder vars
model = ""
assignments = ""

## parsing args - wrap in a function later

# parser = argparse.ArgumentParser(description="PopPUNK network utilities")

# parser.add_argument('--ref-db',type = str, help='Location of built reference database')
# parser.add_argument('--threads', default=1, type=int, help='Number of threads to use [default = 1]')
# graph_weights = parser.add_argument("--graph-weights", help="Save within-strain Euclidean distances into the graph", default=False, action="store_true")
# distances = parser.add_argument("--distances", help="Prefix of input pickle of pre-calculated distances")
# ranks = parser.add_argument("--ranks", help="Comma separated list of ranks used in lineage clustering [default = 1,2,3]", type = str, default = "1,2,3")
# betweenness_sample = parser.add_argument('--betweenness-sample', help='Number of sequences used to estimate betweeness with a GPU [default = 100]', type = int, default = betweenness_sample_default)
# gpu_graph = parser.add_argument("--gpu-graph", default=False, action="store_true", help="Use a GPU when calculating networks [default = False]")
# external_clustering = parser.add_argument("--external-clustering", help="File with cluster definitions or other labels generated with any other method.", default=None)
# indive_refine = parser.add_argument("--indiv-refine", help="Also run refinement for core and accessory individually", choices=["both", "core", "accessory"], default=None)
# output = parser.add_argument("--output", help="Prefix for output files")

# args = parser.parse_args()

args = get_args()
distances = args.distances
output = args.output

ref_list, query_symbol, self, dist_matrix = read_pickle(distances, enforce_self=True)
rank_list = sorted([int(x) for x in args.ranks.split(",")])

## network construction begins

if model.type != "lineage":
    if args.graph_weights:
        weights_type = "euclidean"
    else:
        weights_type = None
    genome_network = construct_network_from_assignments(ref_list, query_symbol, assignments, model.within_label, dist_matrix = dist_matrix, weights_type = weights_type, betweenness_sample = args.betweenness_sample, use_gpu = args.gpu_graph)
else:
    # Lineage fit requires some iteration
    individual_networks = {}
    lineage_clusters = defaultdict(dict)
    for rank in sorted(rank_list):
        sys.stderr.write("Network for rank " + str(rank) + "\n")
        if args.graph_weights:
            weights = model.edge_weights(rank)
        else:
            weights = None
        individual_networks[rank] = construct_network_from_edge_list(ref_list, ref_list, assignments[rank], weights = weights, betweenness_sample = args.betweenness_sample, use_gpu = args.gpu_graph, summarise = False )
        lineage_clusters[rank] = print_clusters(individual_networks[rank], ref_list, printCSV = False, use_gpu = args.gpu_graph)

    # print output of each rank as CSV
    overall_lineage = create_overall_lineage(rank_list, lineage_clusters)
    write_cluster_csv(output + "/" + os.path.basename(output) + "_lineages.csv", ref_list, ref_list, overall_lineage, output_format = "phandango", epiCsv = None, suffix = "_Lineage")
    genome_network = individual_networks[min(rank_list)]

# Ensure all in dists are in final network
check_network_vertex_count(ref_list, genome_network, use_gpu = args.gpu_graph)

fit_type = model.type
isolate_clustering = {fit_type: print_clusters(genome_network, ref_list, output + "/" + os.path.basename(output), external_cluster_csv = args.external_clustering, use_gpu = args.gpu_graph)}

# Save network
save_network(genome_network, prefix = output, suffix = "_graph", use_gpu = args.gpu_graph)

# Write core and accessory based clusters, if they worked
if model.indiv_fitted:
    individual_networks = {}
    for dist_type, slope in zip(["core", "accessory"], [0, 1]):
        if args.indiv_refine == "both" or args.indiv_refine == dist_type:
            individual_assignments = model.assign(dist_matrix, slope = slope)
            individual_networks[dist_type] = construct_network_from_assignments(ref_list, query_symbol, individual_assignments, model.within_label, betweenness_sample = args.betweenness_sample, use_gpu = args.gpu_graph)
            isolate_clustering[dist_type] = print_clusters(individual_networks[dist_type], ref_list, output + "/" + os.path.basename(output) + "_" + dist_type, external_cluster_csv = args.external_clustering, use_gpu = args.gpu_graph)
            save_network(individual_networks[dist_type],
                            prefix = output,
                            suffix = "_" + dist_type + "_graph",
                            use_gpu = args.gpu_graph)

if model.type != "lineage":
    dist_type_list = ["original"]
    dist_string_list = [""]
    if args.indiv_refine == "both" or args.indiv_refine == "core":
        dist_type_list.append("core")
        dist_string_list.append("_core")
    if args.indiv_refine == "both" or args.indiv_refine == "accessory":
        dist_type_list.append("accessory")
        dist_string_list.append("_accessory")
    # Iterate through different network types
    for dist_type, dist_string in zip(dist_type_list, dist_string_list):
        if dist_type == "original":
            network_for_refs = genome_network
        elif dist_type == "core":
            network_for_refs = individual_networks[dist_type]
        elif dist_type == "accessory":
            network_for_refs = individual_networks[dist_type]
        new_References_indices, new_references_names, new_references_file, genome_network = extract_references(network_for_refs, ref_list, output, outSuffix = dist_string, type_isolate = qc_dict["type_isolate"], threads = args.threads, use_gpu = args.gpu_graph)
        nodes_to_remove = set(range(len(ref_list))).difference(new_References_indices)
        names_to_remove = [ref_list[n] for n in nodes_to_remove]

    
        if (len(names_to_remove) > 0):
            # Save reference distances
            dists_suffix = dist_string + ".refs.dists"
            prune_distance_matrix(ref_list, names_to_remove, dist_matrix,
                                    output + "/" + os.path.basename(output) + dists_suffix)
            # Save reference network
            graphs_suffix = dist_string + ".refs_graph"
            save_network(genome_network,
                            prefix = output,
                            suffix = graphs_suffix,
                            use_gpu = args.gpu_graph)
            db_suffix = dist_string + ".refs.h5"
            remove_from_db(args.ref_db, output, names_to_remove)
            os.rename(output + "/" + os.path.basename(output) + ".tmp.h5",
                        output + "/" + os.path.basename(output) + db_suffix)

sys.stderr.write("\nDone\n")
