#!/bin/bash

SEED=${1:-1}


sbatch run.sh --ds_short pep --seed $SEED --default_tag med --default_scale 1 --name med_1
sbatch run.sh --ds_short pep --seed $SEED --default_tag med --default_scale 2 --name med_2
sbatch run.sh --ds_short pep --seed $SEED --default_tag med --default_scale 5 --name med_5

sbatch run.sh --ds_short pep --seed $SEED --default_tag med --default_scale 1 --name med_1_drop --final_dropout --rep_dropout 0.3
sbatch run.sh --ds_short pep --seed $SEED --default_tag med --default_scale 2 --name med_2_drop --final_dropout --rep_dropout 0.3
sbatch run.sh --ds_short pep --seed $SEED --default_tag med --default_scale 5 --name med_5_drop --final_dropout --rep_dropout 0.3