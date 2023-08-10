#!/bin/bash

SEED=${1:-1}

sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag best --name best
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag small --name small
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag new_small --name new_small

sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag best --name best --n_att 5 --n_conv 0
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag small --name small --n_att 5 --n_conv 0
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag new_small --name new_small --n_att 4 --n_conv 0