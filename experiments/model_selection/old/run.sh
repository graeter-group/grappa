#!/bin/bash

#SBATCH -o /hits/fast/mbm/seutelf/sbatch/outfiles/info.out-%j
#SBATCH -t 5:00:00
#SBATCH --mem=8000
#SBATCH -n 4
#SBATCH -G 1
#SBATCH --gres=gpu:1

VPATH="/hits/fast/mbm/seutelf/grappa/mains/runs/compare_small_scaled_samesplit"

# Create the output directory if it doesn't exist yet
mkdir -p $VPATH

#conda path to fast:
CONDA_PREFIX="/hits/fast/mbm/seutelf/software/conda"

export CONDA_PREFIX VPATH OUTDIR
eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"

export DGLBACKEND=pytorch

conda activate grappa_cascade

cd $VPATH

grappa_full_run $*

# simply run 'sbatch run.sh <your options>'
