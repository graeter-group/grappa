#!/bin/bash

#SBATCH -t 24:00:00
#SBATCH --mem=32000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/hits/fast/mbm/seutelf/logs/%j
#SBATCH --error=/hits/fast/mbm/seutelf/logs/%j.err
#SBATCH --job-name=grappa

#conda path to fast:
CONDA_PREFIX="/hits/fast/mbm/seutelf/software/conda"

eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"

conda activate grappa_cascade

# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # dir in which this script lies
SCRIPT_DIR="/hits/fast/mbm/seutelf/grappa/experiments/focus_on_opt"

cd $SCRIPT_DIR

$*