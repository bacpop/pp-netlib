#!/bin/bash

set -e
set -u 
set -o pipefail

## run setup

## test graph-tool construct functions
python tests/test_gt_construct.py