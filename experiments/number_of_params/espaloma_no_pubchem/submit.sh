#!/bin/bash

SEED=${1:-1}

sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_scale 0.2 --name 0.2
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_scale 0.5 --name 0.5
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_scale 0.8 --name 0.8
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_scale 1 --name 1
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_scale 1.5 --name 1.5
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_scale 2 --name 2
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_scale 5 --name 5

sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_scale 2 --n_conv 0 --n_att 0 --name nograph -a additional_features