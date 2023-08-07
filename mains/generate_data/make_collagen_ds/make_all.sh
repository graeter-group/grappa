#!/bin/bash

set -e

#conda path:
CONDA_PREFIX="/hits/fast/mbm/seutelf/software/conda"

eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"

conda activate grappa

NUM_CONFS=50

# get an integer named batch_idx as argument:
BATCH_IDX=${1:-0}
END_IDX=$((BATCH_IDX + 20))

# Assuming your Python function is in a script named 'your_python_script.py'
# and it prints the list of strings one per line. If this isn't the case, adjust accordingly.
IFS=$'\n' read -d '' -r -a strings < <(python sequences.py && printf '\0')

# Print the total number of strings
echo "Total strings: ${#strings[@]}"
echo "Calculating strings from $BATCH_IDX to $((END_IDX-1))"

# Iterate over the desired subset
for idx in $(seq $BATCH_IDX $((END_IDX-1))); do
    if [[ $idx -lt ${#strings[@]} ]]; then
        sbatch run.sh "${strings[$idx]}" $NUM_CONFS
    fi
done
