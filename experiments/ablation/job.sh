#!/bin/bash

#SBATCH -t 48:00:00
#SBATCH --partition=genoa-deep.p
#SBATCH --mem=32000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --output=/hits/fast/mbm/seutelf/logs/%j
#SBATCH --error=/hits/fast/mbm/seutelf/logs/%j.err
#SBATCH --job-name=grappa-abl

# first arg is the directory to run in
DIR=$1

source ~/.bashrc
genoa-conda

set -e

conda activate grappa

cd $DIR

# remove the first argument from the list
shift

# forward all remaining args to the script
"$@"