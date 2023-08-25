#!/bin/bash

SEED=${1:-1}


sbatch run.sh --ds_short radical_dipeptides radical_AAs --seed $SEED --default_tag med --mols 10 --name 10
sbatch run.sh --ds_short radical_dipeptides radical_AAs --seed $SEED --default_tag med --mols 20 --name 20
sbatch run.sh --ds_short radical_dipeptides radical_AAs --seed $SEED --default_tag med --mols 50 --name 50
sbatch run.sh --ds_short radical_dipeptides radical_AAs --seed $SEED --default_tag med --mols 100 --name 100
sbatch run.sh --ds_short radical_dipeptides radical_AAs --seed $SEED --default_tag med --name all
