#!/bin/bash

SEED=${1:-1}

sbatch run.sh --ds_short spice_qca spice_monomers spice_pubchem --seed $SEED --mols 50 --name 50
sbatch run.sh --ds_short spice_qca spice_monomers spice_pubchem --seed $SEED --mols 100 --name 100
sbatch run.sh --ds_short spice_qca spice_monomers spice_pubchem --seed $SEED --mols 200 --name 200
sbatch run.sh --ds_short spice_qca spice_monomers spice_pubchem --seed $SEED --mols 400 --name 400
sbatch run.sh --ds_short spice_qca spice_monomers spice_pubchem --seed $SEED --mols 800 --name 800
sbatch run.sh --ds_short spice_qca spice_monomers spice_pubchem --seed $SEED --mols 1200 --name 1200
#sbatch run.sh --ds_short spice_qca spice_monomers spice_pubchem --seed $SEED --mols 2000 --name 2000
sbatch run.sh --ds_short spice_qca spice_monomers spice_pubchem --seed $SEED --name all

sbatch run.sh --ds_short spice_qca spice_monomers spice_pubchem --seed $SEED --name all_nograph -a additional_features --n_conv 0 --n_att 0 --n_att_readout 4 --readout_width 1024