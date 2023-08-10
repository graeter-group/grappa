#!/bin/bash

SEED=${1:-1}

sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag small --name small

sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag small --name small_att --n_att 5 --n_conv 0

sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag best --name best_att_drop --n_att 5 --n_conv 0 --rep_dropout 0.2 --final_dropout
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag new_small --name new_small_att_drop --n_att 4 --n_conv 0 --rep_dropout 0.2 --final_dropout

sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag best --name best_drop --rep_dropout 0.2 --final_dropout
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag new_small --name new_small_drop --rep_dropout 0.2 --final_dropout