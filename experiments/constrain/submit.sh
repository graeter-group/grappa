#!/bin/bash

SEED=${1:-1}


sbatch run.sh --ds_short pep --seed $SEED --default_tag med --name 1 --param_weight 1
sbatch run.sh --ds_short pep --seed $SEED --default_tag med --name 10 --param_weight 10
sbatch run.sh --ds_short pep --seed $SEED --default_tag med --name 100 --param_weight 100
sbatch run.sh --ds_short pep --seed $SEED --default_tag med --name 1000 --param_weight 1000