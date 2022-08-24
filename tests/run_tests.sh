#!/bin/bash

set -e
set -u 
set -o pipefail

## run setup
python setup.py install

## test graph-tool construct functions
python tests/test_gt_construct.py