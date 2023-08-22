#!/bin/bash

SEED=${1:-1}


sbatch run.sh --ds_short pep --seed $SEED --default_tag med --default_scale 1 --name med_1_corrected