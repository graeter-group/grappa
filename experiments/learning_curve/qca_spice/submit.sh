#!/bin/bash

SEED=${1:-1}



sbatch run.sh --ds_short spice_qca --seed $SEED --default_tag med --mols 20 --name 20

sbatch run.sh --ds_short spice_qca --seed $SEED --default_tag med --mols 50 --name 50

sbatch run.sh --ds_short spice_qca --seed $SEED --default_tag med --mols 100 --name 100

sbatch run.sh --ds_short spice_qca --seed $SEED --default_tag med --mols 200 --name 200

sbatch run.sh --ds_short spice_qca --seed $SEED --default_tag med --mols 350 --name 350

sbatch run.sh --ds_short spice_qca --seed $SEED --default_tag med --name all
