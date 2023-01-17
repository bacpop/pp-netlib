# Network utilities for PopPUNK  
{{< toc >}}
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

`backend (str)`, **optional**: Graph analysis library to use. Accepted values are "GT" and "NX", for [graph-tool](https://graph-tool.skewed.de/static/doc/index.html) and [NetworkX](https://networkx.org/documentation/stable/). A backend can be set this way (i.e. specifying as an argument to the Network instance) or as an environment variable.  
**NOTE**: If you do both of the above, the value passed as an arg to the instance will take precedence.  

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
## note that initialisation is different, an empty list is passed to `ref_list`
example_loaded_network = Network(ref_list = [], outdir, backend)
example_loaded_network.load_network(network_file, sample_metadata_csv)
```  
`network_file (pathlike/str)`, **required**: Path (ideally the absolute path) to the graph file you'd like to load into the Network instance.  
`sample_metadata_csv (pathlike/str)`, **optional**: Path (ideally the absolute path) to a file containing node metadata. This is useful if the graph you'd like to load in has no node attributes already saved in the `network_file`. The program looks for a "Taxon" or "sample_id" column, and a "Cluster" column in `sample_metadata_csv`, and uses these data to apply node labels and cluster assignments.  

**NOTE**: Loading in ".gt" files will only work with `backend = "GT"`. ".xml" and ".graphml" files can be used by either backend.

### Visualise  

```
example_graph.visualize(viz_type, out_prefix, external_data)
```  

`viz_type (str)`, **required**: The type of visualisation to produce. Accepted values are "mst" and "cytoscape".  
- If "mst" is specified, a new "mst" directory, within the Network instance `outdir` is created, and the following outputs are written:  
    - `out_prefix`_mst_network_data.graphml (graphml file of the minimum spanning tree)  
    - `out_prefix`_mst_cluster_plot.png (left)  
    - `out_prefix`_mst_stress_plot.png (right)  

![cluster_plot](./tests/example/mlst_mst_mst_cluster_plot.png) ![stress_plot](./tests/example/mlst_mst_mst_stress_plot.png)

- If "cytoscape" is specified, a new "cytoscape" directory is created as above and the following outputs are generated:  
    - `out_prefix`_mst.graphml (this is produced ONLY if the Network instance contains an MST, i.e. is `viz_type="mst"` has been run first)  
    - `out_prefix`_cytoscape.graphml (graphml file of the full graph)  
    - `out_prefix`_component_1.graphml through `out_prefix`_component_N.graphml (each graph component is individually saved)  
    - `out_prefix`_node_data.tsv (sample metadata)  
    - `out_prefix`_edge_data.tsv  

### Prune  

```
## prune graph  
example_graph.prune(type_isolate, threads)
```
`type_isolate (str)`, **optional**: The sample label associated with the type isolate in the data. Once pruning is done, if `type_isolate` was not retained, it is added back into the graph. A type isolate is chosen by PopPUNK. Defaults to None.  
`threads (int)`, **optional**: The number of threads to use: pruning can be time-consuming for very big graphs. Defaults to 4.  

### Add to Network  

```
## add new edges from new_data, with specified new vertex labels
example_graph.add_to_network(new_data_df, new_vertex_labels)
```  

`new_data_df (pd.DataFrame)`, **required**: A dataframe containing the new edges to add. It is expected to contain a "source" column and a "target" column.  
- If you are adding data to a weighted graph, the program looks for a "weights" column in `new_data_df`, and if no such column is found, arbitrarily weights all new edges as 0.  
- If you are adding to an unweighted graph, then the new edges will also be added unweighted.  

`new_vertex_labels (list)`, **required**: A list of all unique node labels in `new_data_df`.  

### Summarise  

```
## get summary stats
example_graph.get_summary(print_to_std, summary_file_prefix)
```

`print_to_std (bool)` **optional**: Whether to print network summary statistics to stderr. Defaults to True.  
`summary_file_prefix (str)`, **optional**: Whether to write network summary statistics to a text file. Defaults to None.

### Save  

```
## save graph to file, also save pruned graph if it exists in the Network instance
example_graph.save(file_name, file_format, to_save)
```  

`file_name (str)`, **required**: Prefix to be applied to the saved file.  
`file_format (str)`, **required**: The file format to use. Accepted values are ".gt" (only when `backend = "GT"`), ".graphml", and ".xml".  
`to_save (str)`, **optional**: Whether to save only the full graph (`to_save="full_graph"`), only the pruned graph (`to_save="ref_graph"`), or to save both (`to_save="both"`). If "both" is given but the graph was not pruned, only the full graph is saved.  

**NOTE**: Visualising with `viz_type = "cytoscape"` saves the full graph to file as noted above.  

### Write Metadata  

```
## write node labels, clusters, and metadata to some_prefix_node_data.tsv, edge data to some_prefix_edge_data.tsv
example_graph.write_metadata(out_prefix, meta_outdir, external_data)
```

`out_prefix (str)`, **required**: Prefix to be applied to all metadata files.  
`meta_outdir (pathlike/str)`, **optional**: A separate directory to save metadata files to. Defaults to the `outdir` of the Network instance.  
`external_data (pathlike/str *OR* pd.DataFrame)`, **optional**: A file containing other data associated with each sample.  

Writing metadata produces:
    - `out_prefix`_node_data.tsv (with `external_data` merged by sample_id)  
    - `out_prefix`_edge_data.tsv`  

