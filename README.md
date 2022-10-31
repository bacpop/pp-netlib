# Usage:

## Initialising Network object
```
from pp_netlib.network import Network

samples_list = ["sample1", "sample2", "sample3"]

#initialise a graph object, supplying three labels, outdir as Desktop,
#setting graphing backend as graph-tool for Network instance
example_graph = Network(ref_list = samples_list, outdir = "/Users/username/Desktop", backend = "GT")
```

Graphing backend can alternatively be set as an environment variable as follows:
```
os.environ["GRAPH_BACKEND"] = "GT" ## inside of your_script.py or in interactive python terminal

*OR*

export GRAPH_BACKEND=GT ## inside bash/shell terminal/environment
```
Note that if both of the above methods (setting backend as argument to Network instance and setting an environment variable), setting backend as the argument will take priority, i.e.:
```
os.environ["GRAPH_BACKEND"] = "NX" ## set backend to networkx as an environment variable
example_graph = Network(ref_list = samples_list, outdir = "/Users/username/Desktop", backend = "GT")
```
will result in the graph still using graph-tool as the backend.

## .construct

Called on a Network object and produces a graph populated with edges.

```
example_graph.construct(network_data, weights) ## call the construct method to populate graph object with your data
```

Weights may be None, or a list of weights, provided in the same order as the list of edges.

network_data can be a dataframe, sparse matrix, or a list of tuples where each tuple contains source and destination node indices.

Edge List
```
>> edge_list
[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
```

Dataframe 
```
>> edge_df
##column [0] is source nodes
##column [1] is destination nodes
    0   1
0   0   1
1   1   2
2   2   3
3   3   4
4   4   0
```

Sparse coordinate matrix
```
>> edge_matrix
##edge_matrix.row should return source nodes
##edge_matrix.col should return destination nodes
##edge_matrix.data should return weights
(0, 1)	0.1
(1, 2)	0.1
(2, 3)	0.5
(3, 4)	0.7
(4, 0)	0.2

##the above matrix as a numpy array for reference
>> edge_matrix.toarray()
[[0.0  0.1 0.0  0.0  0.0 ]
[0.0  0.0  0.1 0.0  0.0 ]
[0.0  0.0  0.0  0.5 0.0 ]
[0.0  0.0  0.0  0.0  0.7]
[0.2 0.0  0.0  0.0  0.0 ]]
```
All of the above inputs produce the same graph.

## .prune
Called on initialised and populated Network object. Recursively prunes nodes from the input graph to find representative "reference" nodes from each maximal clique.
```
example_graph.prune(type_isolate = "type_isolate_label", threads = 4)

## pruned graph can be accessed as follows:
example_graph.ref_graph
```
The `type_isolate` is used to specify the sample name corresponding to the type isolate as computed by PopPUNK. If not already present in the pruned graph, the type isolate will be added by this method. If `threads` is not specified, it defaults to 4.

## .get_summary
Called on initialised and populated Network object. Prints summary of network properties to stderr and optionally to plain text file.

```
example_graph.get_summary() #no text file produced
example_graph.get_summary("summary_file.txt") #summary written to stderr AND to summary_file.txt
```

## .visualize
Called on initialised and populated Network object. Creates visualizations from the graph.
```
example_graph.visualize(viz_type = "mst", out_prefix = "filename_prefix")
```
When `viz_type` is "mst", a minimum-spanning tree is calculated from the graph and saved as a .graphml file. Additionally, two visualizations - a stress plot and a cluster plot -  are saved as .png files. `viz_type` can also be "cytoscape", in which case the full graph is saved; each component of the full graph is saved in its own file; an mst graph is saved *if* one has already been generated; and a csv file containing metadata is also saved.

Note that `out_prefix` is not expected to be a filepath but a prefix string that will be applied to all output files generated in this method.
Outputs from `viz_type = "mst"` are written to outdir/mst, where `outdir` is the output directory specified when initializing the Network object. Similarly, when `viz_type = "cytoscape"`, outputs are written to outdir/cytoscape.

## .load_network
Called on empty initialized Network object. Loads a premade graph from a network file. Network file must be .gt, .graphml, or .xml if using graph_tools as the backend library; it must be .graphml, or .xml if using networkx.

```
#initialize empty Network object
example_loaded_network = Network([], outdir = "path/to/outdir", backend = "GT")
example_loaded_network.load_network("path/to/network_file.graphml")

# the graph loaded into the Network object can be accessed as
example_loaded_network.graph
```

## .add_to_network
Called on initialised and populated Network object. Adds the given edge data to the existing graph.
```
example_graph.add_to_network(new_data_df = new_data, new_vertex_labels = ["new_label_1", "new_label_2", "new_label_3"])
```
`new_data_df` is expected to be a pandas dataframe. If existing graph edges are weighted but no weights are provided in `new_data_df`, new edges will be added with a dummy weight value of 0. If weights are provided in `new_data_df` but the existing graph edges are not weighted, the weights in `new_data_df` are ignored.

## .write_metadata
Called on initialised and populated Network object. Writes graph edge and node data to a .csv file.
```
example_graph.write_metadata(out_prefix = "filename_prefix")
```
Output filenames have a prefix specified by `out_prefix`, and are saved to outdir, where outdir is the output directory specified when initializing the Network object. Another output directory can be specified by providing a filepath to the method with the argument `meta_outdir`. If there is metadata associated with the samples, i.e. the graph nodes, this data can be provided using the argument `external_data` as a path to a csv file, or a pandas dataframe. `external_data` is expected to have a column called "sample_id" which matches the node names in the graph; this is the column that will be used to merge `external_data` with data scraped from the graph.

## .save
Called on Network object, saves the graph to file. 
```
example_graph.save("example_graph", ".graphml")
```
If the second argument to the .save method (file_format) is not specified, the file will be saved as .graphml; other valid file_format options with the "GT" backend are ".graphml" and ".xml"

If the backend is "NX", the file will be saved as .graphml regardless of the file_format specified.
