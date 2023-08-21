#!/bin/bash

SEED=${1:-1}

sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --mols 50 --name 50
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --mols 100 --name 100
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --mols 200 --name 200
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --mols 400 --name 400
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --mols 600 --name 600
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --mols 800 --name 800
