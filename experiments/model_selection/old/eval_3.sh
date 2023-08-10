#!/bin/bash

BASEPATH="/hits/fast/mbm/seutelf/grappa/mains/runs/compare_on_both4/versions"

# Change to the base directory
cd $BASEPATH

# Iterate over all directories
for dir in */ ; do
    # Absolute path for the directory
    abs_path="$BASEPATH/$dir"
    
    # Run your sbatch command here
    echo $abs_path
    sbatch /hits/fast/mbm/seutelf/grappa/experiments/model_selection/both/eval.sh $abs_path --all_loaders

    # No need to change back to the base directory if you're not changing directories in the loop
done
