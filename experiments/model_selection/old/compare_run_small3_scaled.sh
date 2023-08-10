#!/bin/bash

SEED=${1:-1}

sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag small --name small --default_scale 0.1 --name small_scaled_0.1
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag small --name small_att --n_att 4 --n_conv 0 --default_scale 0.1 --name small_att_scaled_0.1
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag deep --name deep --default_scale 0.1 --name deep_scaled_0.1
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag med --name med --default_scale 0.1 --name med_scaled_0.1

# now all for 0.5 and 2 and 10:
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag small --name small --default_scale 0.5 --name small_scaled_0.5
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag small --name small --default_scale 2 --name small_scaled_2
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag small --name small --default_scale 10 --name small_scaled_10

sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag small --name small_att --n_att 4 --n_conv 0 --default_scale 0.5 --name small_att_scaled_0.5
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag small --name small_att --n_att 4 --n_conv 0 --default_scale 2 --name small_att_scaled_2
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag small --name small_att --n_att 4 --n_conv 0 --default_scale 10 --name small_att_scaled_10

sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag deep --name deep --default_scale 0.5 --name deep_scaled_0.5
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag deep --name deep --default_scale 2 --name deep_scaled_2
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag deep --name deep --default_scale 10 --name deep_scaled_10

sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag med --name med --default_scale 0.5 --name med_scaled_0.5
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag med --name med --default_scale 2 --name med_scaled_2
sbatch run.sh --ds_short spice_qca spice_monomers --seed $SEED --default_tag med --name med --default_scale 10 --name med_scaled_10