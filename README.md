# Network utilities for PopPUNK  

## Install  

```
$ git clone https://github.com/bacpop/pp-netlib.git
$ cd pp-netlib
$ python setup.py install
```  

## Quick Start  

### Initialise Network Object  

```
from pp_netlib.network import Network
example_graph = Network(ref_list = samples_list, outdir = "path/to/output/dir", backend = "backend")
```  

### Construct or Load a Network  

```
## construct from a list of edges
example_graph.construct(network_data = edges_data_df, weights = weights)

## load from file (.gt, .graphml, .xml)
example_loaded_network = Network([], outdir = "path/to/output/dir", backend = "backend")
example_loaded_network.load_network("path/to/network_file.graphml")
```  

### Visualise a Network  

```
## produce an mst with cluster-plot and stress-plot images
example_graph.visualize(viz_type = "mst", out_prefix = "some_prefix")

## produce a bunch of output files to load into cytoscape, including merging metadata
example_graph.visualize(viz_type = "cytoscape", out_prefix = "some_prefix", external_data="path/to/metadata.csv")
```

### Prune a Network  

```
## prune graph with 4 threads, make sure "type_isolate_id" is retained in the pruned graph
example_graph.prune(type_isolate = "type_isolate_id", threads = 4)
```

### Add Data to Network  

```
## add new edges from new_data, with specified new vertex labels
example_graph.add_to_network(new_data_df = new_data, new_vertex_labels = ["new_label_1", "new_label_2", "new_label_3"])
```  

### Summarise and Save a Network  

```
## get summary stats
example_graph.get_summary() #summary written to stderr, no text file produced
example_graph.get_summary("summary_file.txt") #summary written to stderr AND to summary_file.txt

## save graph to file, also save pruned graph if it exists in the Network instance
example_graph.save(file_name="example_graph", file_format=".graphml", to_save="both")
```  

### Write Network Metadata to File  

```
## write node labels, clusters, and metadata to some_prefix_node_data.tsv, edge data to some_prefix_edge_data.tsv
example_graph.write_metadata(out_prefix = "some_prefix")
```
---

## List of Options and Default Behaviours  

### Initialising  

```
example_graph = Network(ref_list, outdir, backend)
```  
`ref_list (list)`, **required**: List (or set) of sample labels (str), one for each node. Any special characters (".", ":", ")", "(" ) in the labels are replaced with underscores.  
**NOTE**: When loading a graph, a `ref_list` value is still expected; an empty list will work.  

`outdir (pathlike/str)`, **optional**: Path (ideally absolute path) to where any outputs will be written. Defaults to "./", i.e. the working directory.  

`backend (str)`, **optional**: Graph analysis library to use. Accepted values are "GT" and "NX", for [graph-tool](https://graph-tool.skewed.de/static/doc/index.html) and [NetworkX](https://networkx.org/documentation/stable/). A backend can be set this way (i.e. specifying as an argument to the Network instance) or as an environment variable. **NOTE**: If you do both of the above, the value passed as an arg to the instance will take precedence.  

The latter can be achieved as follows:  
```
## set as env variable inside of a python script that contains Network instance
os.environ["GRAPH_BACKEND"] = "GT"

*OR*

## set as bash/shell env variable
export GRAPH_BACKEND=GT ## inside bash/shell terminal/environment
```  

### Construct  

```
## construct from a list of edges
example_graph.construct(network_data, weights)
```  
`network_data (pd.DataFrame *OR* list *OR* scipy.sparse.coo_matrix)`, **required**: Data containing a series of edges. This can be in one of the three data types noted.  

- If a dataframe is passed, it is expected to have a column called "source" and a column called "target". Optionally, it can have a column called "weights".  
- If a list is passed, it is expected to be a list of tuples where the first element of each tuple is the source node and the second is the target node. Optionally, each tuple can contain a third element, which will be used as the weight for that edge.  
```
## weighted list
edge_list = [(s1, t1, w1), (s2, t2, w2), (s3, t3, w3)...]

## unweighted list
edge_list = [(s1, t1), (s2, t2), (s3, t3)...]
```  
- If a sparse matrix (arbitrarily called "edge_matrix") is provided, edge_matrix.row should return source nodes, edge_matrix.col should return destination nodes, and edge_matrix.data should return weights  

`weights (list *OR* bool)`, **optional**: A list of weights to apply to the edges. These are expected to be in the same order as the edges in `network_data`. Defaults to None.  
- If `network_data` is a dataframe and contains weights (in a column called "weights") already, `weights` can be set to True. Note that in this case, if there is no column called "weights", an error is thrown. If weights exists in `network_data` but should not be used, it can be set to False. If `weights` is not specified, an unweighted network will be constructed regardless of whether `network_data` contains a "weights" column.  
- If `network_data` is an unweighted list, and `weights` is also a list, a weighted Network is constructed. If `weights` is set to False in this case, an unweighted Network is made. However, if an unweighted list and `weights=True` is passed, an error will be thrown. If a weighted list is passed, `weights` **needs** to be set to True.  
- If `network_data` is a sparse matrix, the `weights` argument is ignored and a weighted network will be created.  

### Load  

```
## load from file (.gt, .graphml, .xml)
## note that initialisation is different, an empty list if passed to `ref_list`
example_loaded_network = Network(ref_list = [], outdir, backend)
example_loaded_network.load_network(network_file, sample_metadata_csv)
```  
`network_file (pathlike/str)`, **required**: Path (ideally the absolute path) to the graph file you'd like to load into the Network instance.  
`sample_metadata_csv (pathlike/str)`, **optional**: Path (ideally the absolute path) to a file containing node metadata. This is useful if the graph you'd like to load in has no node attributes already saved in the `network_file`. The program looks for a "Taxon" or "sample_id" column, and a "Cluster" column in `sample_metadata_csv`, and uses these data to apply node labels and cluster assignments.  

### Visualise  

```
example_graph.visualize(viz_type, out_prefix, external_data)
```  

`viz_type (str)`, **required**: The type of visualisation to produce. Accepted values are "mst" and "cytoscape".  
- If "mst" is specified, a new "mst" directory, within the Network instance `outdir` is created, and the following outputs are written:  
    - `out_prefix`_mst_network_data.graphml (graphml file of the minimum spanning tree)  
    - `out_prefix`_mst_cluster_plot.png  
    - `out_prefix`_mst_stress_plot.png  

![cluster_plot](./tests/example/mlst_mst_mst_cluster_plot.png)
![stress_plot](./tests/example/mlst_mst_mst_stress_plot.png)
- If "cytoscape" is specified, a new "cytoscape" directory is created as above and the following outputs are generated:  
    - `out_prefix`_mst.graphml (this is produced ONLY if the Network instance contains an MST, i.e. is `viz_type="mst"` has been run first)  
    - `out_prefix`_cytoscape.graphml (graphml file of the full graph)  
    - `out_prefix`_component_1.graphml through `out_prefix`_component_N.graphml (each graph component is individually saved)  
    - `out_prefix`_node_data.tsv (sample metadata)  
    - `out_prefix`_edge_data.tsv  






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
