#!/bin/bash

SEED=${1:-1}

sbatch run.sh --ds_short pep --seed $SEED --default_tag med --mols 10 --name 10
sbatch run.sh --ds_short pep --seed $SEED --default_tag med --mols 20 --name 20
sbatch run.sh --ds_short pep --seed $SEED --default_tag med --mols 50 --name 50
sbatch run.sh --ds_short pep --seed $SEED --default_tag med --mols 100 --name 100
sbatch run.sh --ds_short pep --seed $SEED --default_tag med --mols 200 --name 200
sbatch run.sh --ds_short pep --seed $SEED --default_tag med --mols 300 --name 300
sbatch run.sh --ds_short pep --seed $SEED --default_tag med --mols 400 --name 400
sbatch run.sh --ds_short pep --seed $SEED --default_tag med --mols 500 --name 500
sbatch run.sh --ds_short pep --seed $SEED --default_tag med --name all