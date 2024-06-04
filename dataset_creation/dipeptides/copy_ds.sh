#!/bin/bash

# stop upon error
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # dir in which this script lies

source_path="$SCRIPT_DIR/../../../software/grappa_data_creation/grappa_datasets/dipeptides_300K"

target_path="$SCRIPT_DIR/../../data/grappa_datasets/dipeptides_300K_amber99sbildn"

cp -r $source_path $target_path