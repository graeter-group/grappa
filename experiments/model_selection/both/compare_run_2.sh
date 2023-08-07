#!/bin/bash

SEED=${1:-1}

sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag deep --n_att_readout 5 --name deep_read --final_dropout --rep_dropout 0.3
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag deep --name deep_conv --n_att 5 --final_dropout --rep_dropout 0.3
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag deep --name deep --final_dropout --rep_dropout 0.3
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag med --n_att 5 --name med_deep --final_dropout --rep_dropout 0.3
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag small --name small --final_dropout --rep_dropout 0.3
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag large --name large --final_dropout --rep_dropout 0.3
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag best --name best --final_dropout --rep_dropout 0.3
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag best --name best_larger --final_dropout --rep_dropout 0.3 --attention_hidden_feats 2048 --readout_width 512