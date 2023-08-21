#!/bin/bash

SEED=${1:-1}

sbatch run.sh --ds_short pep --seed $SEED --default_tag med --mols 10 --name 10
sbatch run.sh --ds_short pep --seed $SEED --default_tag med --mols 50 --name 50
sbatch run.sh --ds_short pep --seed $SEED --default_tag med --mols 100 --name 100
sbatch run.sh --ds_short pep --seed $SEED --default_tag med --mols 200 --name 200
sbatch run.sh --ds_short pep --seed $SEED --default_tag med --mols 400 --name 400
sbatch run.sh --ds_short pep --seed $SEED --default_tag med --mols 600 --name 600
sbatch run.sh --ds_short pep --seed $SEED --default_tag med --mols 800 --name 800