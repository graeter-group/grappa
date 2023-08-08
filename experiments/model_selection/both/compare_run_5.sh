#!/bin/bash

SEED=${1:-1}

sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag best --name best
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag large --name large --dropout 0 --rep_dropout 0
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag small --name small
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag new_small --name new_small


# now all with dropout of 0.2:
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag best --name best_dropout --rep_dropout 0.2 --final_dropout
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag large --name large_dropout --rep_dropout 0.2 --final_dropout
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag small --name small_dropout --rep_dropout 0.2 --final_dropout
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag new_small --name new_small_dropout --rep_dropout 0.2 --final_dropout