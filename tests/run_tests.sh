#!/bin/bash

set -e
set -u
set -o pipefail
set -x

## run setup

## test graph-tool construct functions
python tests/test_gt_construct.py

## test networkx construct functions
python tests/test_nx_construct.py

## test backend selector
python tests/test_backend_selector.py --set_with_python

export GRAPH_BACKEND=GT
python tests/test_backend_selector.py

export GRAPH_BACKEND=NX
python tests/test_backend_selector.py

## test summarise method
python tests/test_summarise.py

## test saving and loading
python tests/test_save_load.py

## test add_to_network method
python tests/test_add.py

## test prune method
python tests/test_prune.py

## test visualization; currently mst only
python tests/test_viz.py