## will become the cli and associated logic for this package: mainly for testing, maybe for general use

## imports
#### internal imports
from .utils import *
from .cliques import *
from .vertices import *
from .load_network import *
from .indices_refs_clusters import *
from .construct_network import *

args = get_args()

## need to get the following as args OR info from poppunk
model = "" ##TODO get model type from poppunk (as string to simplify?)
ref_list = "something.refs"
query_list = "" ##?
assignments = "something.refs.dists.npy"

if model != "lineage":
    if args.graph_weights:
        weights_type = "euclidean"
    else:
        weights_type = None
    genome_network = construct_network_from_assignments(ref_list, 
                                                        query_list, 
                                                        assignments, 
                                                        model.within_label, 
                                                        dist_matrix = dist_matrix, 
                                                        weights_type = weights_type, 
                                                        betweenness_sample = args.betweenness_sample, 
                                                        use_gpu = args.gpu_graph)