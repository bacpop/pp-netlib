# make an overarching class to make it easy to call things in "api" mode. eg:

from pandas import DataFrame


class Graph:
    def __init__(self, rlist, assignments):
        self.rlist = rlist
        self.assignments = assignments
        # make graph here
        pass

    def edge_list_to_df(edge_list):
        # code
        return #dataFrame

    ## other conversions to df

    def construct_dense_weighted_network():
        # code
        return network

    def save():
        # code to save graph
        return

network = Graph(rlist, assignments)
network.save(file)
network.prune()
