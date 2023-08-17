#!/bin/bash

SEED=${1:-1}


sbatch run.sh --ds_short spice_qca spice_monomers spice_pubchem --seed $SEED --default_tag small --default_scale 1 --name small_scaled_1 
sbatch run.sh --ds_short spice_qca spice_monomers spice_pubchem --seed $SEED --default_tag small --default_scale 2 --name small_scaled_2 

sbatch run.sh --ds_short spice_qca spice_monomers spice_pubchem --seed $SEED --default_tag deep --default_scale 1 --name deep_scaled_1 
sbatch run.sh --ds_short spice_qca spice_monomers spice_pubchem --seed $SEED --default_tag deep --default_scale 2 --name deep_scaled_2 

sbatch run.sh --ds_short spice_qca spice_monomers spice_pubchem --seed $SEED --default_tag med --default_scale 1 --name med_scaled_1 
sbatch run.sh --ds_short spice_qca spice_monomers spice_pubchem --seed $SEED --default_tag med --default_scale 2 --name med_scaled_2 