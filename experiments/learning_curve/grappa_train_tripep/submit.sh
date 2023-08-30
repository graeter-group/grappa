#!/bin/bash

SEED=${1:-1}


sbatch run.sh --ds_short pep tripeptides --seed $SEED --default_tag med --mols 50 --name 50
sbatch run.sh --ds_short pep tripeptides --seed $SEED --default_tag med --mols 100 --name 100
sbatch run.sh --ds_short pep tripeptides --seed $SEED --default_tag med --mols 300 --name 300
sbatch run.sh --ds_short pep tripeptides --seed $SEED --default_tag med --mols 500 --name 500
sbatch run.sh --ds_short pep tripeptides --seed $SEED --default_tag med --name all