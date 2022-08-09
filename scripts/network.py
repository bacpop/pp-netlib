## internal imports
# from .utils import *
# from .cliques import *
# from .vertices import *
# from .load_network import *
# from .indices_refs_clusters import *
# from .construct_network import *

class Graph:
    def __init__(self, rlist, assignments, outdir, model_type, use_gpu = False):
        self.rlist = rlist
        self.assignments = assignments
        self.outdir = outdir
        self.use_gpu = use_gpu
        self.model_type = model_type

        if use_gpu:
            try:
                import cupyx
                import cugraph
                import cudf
                import cupy as cp
                from numba import cuda
                import rmm
                use_gpu = True
            except ImportError as e:
                use_gpu = False

        print(rlist, assignments, outdir, use_gpu, model_type)

    def construct(self):
        # make network from df
        print("making network")
        return #genome_network 

    def prune(self):
        # call to functions that prune a network
        print("pruning network")
        return

    def summarize(self):
        # print network summary and stats
        print("network summary")
        return

    def visualize(self):
        # code for calling viz functions
        print("visualizing network")
        return #files associated with viz

    def convert(self, type1, type2):
        # interconversions between cugraph, graphtool, networkx?
        print(f"converting from {type1} to {type2}")
        return

    def load_network(self):
        # reading in dataframe
        # add input data conversion to dataframe here?
        print("loading data")
        return

    def add_to_network(self, datapoint):
        # calls functions which load a preexisting network, or work with a newly built one, and add data to it
        print(f"adding {datapoint} to network")
        return

    def save(self):
        # call to save_network
        print("saving network")
        return


a = Graph("refs_list", "assignments.npy", "/Home/Desktop/", "dbscan", True)

a.load_network()
a.construct()
a.prune()
a.summarize()
a.visualize()
a.convert("cugraph", "graphtool")
a.add_to_network("random_data_point")
a.save()