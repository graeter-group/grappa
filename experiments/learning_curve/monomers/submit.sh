#!/bin/bash

SEED=${1:-1}


sbatch run.sh --ds_short spice_monomers --seed $SEED --default_tag med --mols 50 --name 50
sbatch run.sh --ds_short spice_monomers --seed $SEED --default_tag med --mols 100 --name 100
sbatch run.sh --ds_short spice_monomers --seed $SEED --default_tag med --mols 200 --name 200
sbatch run.sh --ds_short spice_monomers --seed $SEED --default_tag med --mols 400 --name 400
