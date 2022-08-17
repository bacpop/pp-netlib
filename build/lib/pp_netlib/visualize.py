## TODO populate with fns for min spanning tree, cytoscape (poppunk_visualize_network/visualize_network)

from .utils import *
# Check on parallelisation of graph-tools
def visualize_network(threads, output):
    setGtThreads(threads)

    sys.stderr.write("PopPUNK: visualise\n")
    if not (microreact or phandango or grapetree or cytoscape):
        sys.stderr.write("Must specify at least one type of visualisation to output\n")
        sys.exit(1)

    # make directory for new output files
    if not os.path.isdir(output):
        try:
            os.makedirs(output)
        except OSError:
            sys.stderr.write("Cannot create output directory\n")
            sys.exit(1)