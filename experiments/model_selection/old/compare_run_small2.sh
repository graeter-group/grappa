#!/bin/bash

SEED=${1:-1}

sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag small --name small --lr 1e-4 --time_limit 2
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag small_att --name small_att --n_att 4 --n_conv 0 --lr 1e-4 --time_limit 2
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag deep --name deep --lr 1e-4 --time_limit 2
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag med --name med --lr 1e-4 --time_limit 2

sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag small --name small_add -a additional_features --lr 1e-4 --time_limit 2
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag deep --name deep_add -a additional_features --lr 1e-4 --time_limit 2
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag med --name med_add -a additional_features --lr 1e-4 --time_limit 2