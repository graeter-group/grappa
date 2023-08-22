#!/bin/bash

SEED=${1:-1}

sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --mols 50 --name 50 -a additional_features
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --mols 100 --name 100 -a additional_features
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --mols 200 --name 200 -a additional_features
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --mols 400 --name 400 -a additional_features
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --mols 600 --name 600 -a additional_features
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --mols 800 --name 800 -a additional_features

sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --name all -a additional_features

sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --name all_nograph -a additional_features --n_conv 0 --n_att 0 --n_att_readout 4