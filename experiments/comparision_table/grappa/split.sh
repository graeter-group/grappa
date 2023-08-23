#!/bin/bash

#SBATCH -o /hits/fast/mbm/seutelf/sbatch/outfiles/info.out-%j
#SBATCH -t 1:00:00
#SBATCH --mem=8000
#SBATCH -n 4


splitpath=${1}
ds_short=${2}


#conda path to fast:
CONDA_PREFIX="/hits/fast/mbm/seutelf/software/conda"

export CONDA_PREFIX VPATH OUTDIR
eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"

export DGLBACKEND=pytorch

conda activate grappa_cascade


grappa_fold $splitpath --ds_short $ds_short --k 10
