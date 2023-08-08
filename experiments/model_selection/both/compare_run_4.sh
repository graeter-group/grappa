#!/bin/bash

SEED=${1:-1}

sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag best --name best
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag large --name large --dropout 0 --rep_dropout 0
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag small --name small
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag med --name med

# now all with dropout of 0.3:
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag best --name best_dropout --dropout 0.3 --rep_dropout 0.3 --final_dropout
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag large --name large_dropout --dropout 0.3 --rep_dropout 0.3 --final_dropout
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag small --name small_dropout --dropout 0.3 --rep_dropout 0.3 --final_dropout
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag med --name med_dropout --dropout 0.3 --rep_dropout 0.3 --final_dropout