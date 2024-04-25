#!/bin/bash

#SBATCH -t 24:00:00
#SBATCH --mem=32000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/hits/fast/mbm/seutelf/logs/%j
#SBATCH --error=/hits/fast/mbm/seutelf/logs/%j.err
#SBATCH --job-name=grappa

# first arg is the directory to run in
DIR=$1

#conda path to fast:
CONDA_PREFIX="/hits/fast/mbm/seutelf/software/conda"

eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"

conda activate grappa_cascade

cd $DIR

# remove the first argument from the list
shift

# forward all remaining args to the script
"$@"