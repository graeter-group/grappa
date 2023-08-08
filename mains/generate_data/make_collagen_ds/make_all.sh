#!/bin/bash

N_PER_BATCH=20

set -e

#conda path:
CONDA_PREFIX="/hits/basement/mbm/seutelf/software/conda"

eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"

conda activate grappa_haswell

# get an integer named batch_idx as argument:
BATCH_IDX=${1:-0}
START_IDX=$(($N_PER_BATCH * $BATCH_IDX))
END_IDX=$(($N_PER_BATCH * ($BATCH_IDX + 1)-1))

# Assuming Python function is in a script named 'sequences.py'
# and it prints the list of strings one per line.
strings=($(python sequences.py))

# Print the total number of strings
echo "Total strings: ${#strings[@]}"
echo "Calculating strings from $START_IDX to $END_IDX"

# Iterate over the desired subset
counter=0
for idx in $(seq $START_IDX $END_IDX); do
    if [[ $idx -lt ${#strings[@]} ]]; then
        counter=$((counter + 1))
        echo "Calculating string ${strings[$idx]}"
        sbatch run.sh "${strings[$idx]}"
    fi
done

echo "Submitted $counter jobs"