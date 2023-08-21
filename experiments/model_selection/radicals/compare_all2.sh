#!/bin/bash

SEED=${1:-1}


sbatch run.sh --ds_short pep --seed $SEED --default_tag med --default_scale 1 --name med_1
sbatch run.sh --ds_short pep --seed $SEED --default_tag med --default_scale 2 --name med_2

sbatch run.sh --ds_short pep --seed $SEED --default_tag large --default_scale 1 --name large_1

sbatch run.sh --ds_short pep --seed $SEED --default_tag small --default_scale 1 --name small_1

# sbatch run.sh --ds_short spice collagen radical_AAs radical_dipeptides --seed $SEED --default_tag med --default_scale 1 --name med_1_noscans