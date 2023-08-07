#!/bin/bash

mkdir -p /hits/fast/mbm/seutelf/sbatch/outfiles

SEED=${1:-1}

sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag deep --n_att_readout 5 --rep_dropout 0.3 --name very_deep
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag deep --n_att_readout 5 --rep_dropout 0 --name very_deep_nodrop
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag deep --rep_dropout 0.3 --name deep
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag med --rep_dropout 0.3 --name med
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag med --rep_dropout 0 --name med_nodrop
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag large --rep_dropout 0.3 --name large
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag small --rep_dropout 0.3 --name small