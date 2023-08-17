#!/bin/bash

#SBATCH -o /hits/fast/mbm/seutelf/sbatch/outfiles/info.out-%j
#SBATCH -t 24:00:00
#SBATCH --mem=32000
#SBATCH -n 20


#conda path to fast:
CONDA_PREFIX="/hits/fast/mbm/seutelf/software/conda"

eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"

export DGLBACKEND=pytorch

conda activate grappa_cascade

cd "/hits/fast/mbm/seutelf/grappa/mains/create_data"

$*