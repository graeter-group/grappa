#!/bin/bash

#SBATCH -t 24:00:00
#SBATCH --mem=16000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16


#conda path to fast:
CONDA_PREFIX="/hits/fast/mbm/seutelf/software/conda"

eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"

conda activate grappa_cascade

cd "/hits/fast/mbm/seutelf/grappa/tests"

$*