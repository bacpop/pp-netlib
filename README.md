# Usage:

## Initialising Network object
```
from pp_netlib.network import Network

samples_list = ["sample1", "sample2", "sample3"]

#initialise a graph object, supplying three labels, outdir as Desktop,
#graphing module as graph-tool, and not using gpu and related modules
example_graph = Network(ref_list = samples_list, outdir = "/Users/user/Desktop", backend = "GT", use_gpu = False)
```

Graphing backend can alternatively be set as follows:
```
os.environ["GRAPH_BACKEND"] = "GT" ## inside of your_script.py or in interactive python terminal

*OR*

export GRAPH_BACKEND=GT ## inside bash/shell terminal/environment
```
## .construct

Called on a Network object and produces a graph populated with edges.

```
example_graph.construct(network_data, weights) ## call the construct method to populate graph object with your data
```

The number of ref_list elements (used in initialising the Network object) is assumed to be equal to the number of edges in any of
the following data types. Additionally, the orders of elements in the ref_list and network_data are also assumed to correspond exactly.

Weights may be None, or a list of weights, provided in the same order as the list of edges.

network_data can be a dataframe, sparse matrix, or a list of tuples where each tuple contains source and destination node indices.
The following data generate identical graphs: 
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

## .get_summary
Called on initialised and populated Network object. Prints summary of network properties to stderr and optionally to plain text file.

```
example_graph.get_summary() #no text file produced
example_graph.get_summary("summary_file.txt") #summary written to stderr AND to summary_file.txt
```

## .load_network
Called on empty initialized Network object. Loads a premade graph from a network file. Network file must be .gt, .graphml, or .xml if using graph_tools as the backend library; it must be .graphml, or .xml if using networkx.

```
#initialize empty Network object
example_loaded_network = Network([])
example_loaded_network.load_network("path/to/network_file.graphml")

# access loaded data using
example_loaded_network.graph

```

## .save
Called on Network object, saves the graph to file. 
```
example_graph.save("example_graph", ".graphml")
```
If the second argument to the .save method (file_format) is not specified: if the backend is "GT", the file will be saved as .gt; other valid file_format options with the "GT" backend are ".graphml" and ".xml"

If the backend is "NX", the file will be saved as .graphml regardless of the file_format specified.
