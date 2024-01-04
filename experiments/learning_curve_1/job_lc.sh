#!/bin/bash

#SBATCH -t 24:00:00
#SBATCH --mem=32000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/hits/fast/mbm/seutelf/grappa/experiments/learning_curve_1/logs/%j.out
#SBATCH --error=/hits/fast/mbm/seutelf/grappa/experiments/learning_curve_1/logs/%j.err
#SBATCH --job-name=lc_grappa

#conda path to fast:
CONDA_PREFIX="/hits/fast/mbm/seutelf/software/conda"

eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"

conda activate grappa_cascade

cd "/hits/fast/mbm/seutelf/grappa/experiments/learning_curve_1"

$*