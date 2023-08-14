#!/bin/bash

SEED=${1:-1}

sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag small --default_scale 0.2 --name small_scaled_0.2 
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag deep --default_scale 0.2 --name deep_scaled_0.2 
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag med --name med --default_scale 0.2 --name med_scaled_0.2 

# now with scale 0.5, 1 and 2:
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag small --default_scale 0.5 --name small_scaled_0.5
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag small --default_scale 1 --name small_scaled_1 
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag small --default_scale 2 --name small_scaled_2 

sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag deep --default_scale 0.5 --name deep_scaled_0.5 
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag deep --default_scale 1 --name deep_scaled_1 
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag deep --default_scale 2 --name deep_scaled_2 

sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag med --default_scale 0.5 --name med_scaled_0.5 
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag med --default_scale 1 --name med_scaled_1 
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag med --default_scale 2 --name med_scaled_2 