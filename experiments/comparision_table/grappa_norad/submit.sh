#!/bin/bash

SEED=${1:-1}

splitpath="/hits/fast/mbm/seutelf/grappa/mains/split_names/grappa_pep"

sbatch split.sh $splitpath spice collagen

sbatch run.sh --ds_short spice collagen --seed $SEED --ds_split_names "$splitpath/fold_0.json"
sbatch run.sh --ds_short spice collagen --seed $SEED --ds_split_names "$splitpath/fold_1.json"
sbatch run.sh --ds_short spice collagen --seed $SEED --ds_split_names "$splitpath/fold_2.json"
sbatch run.sh --ds_short spice collagen --seed $SEED --ds_split_names "$splitpath/fold_3.json"
sbatch run.sh --ds_short spice collagen --seed $SEED --ds_split_names "$splitpath/fold_4.json"
sbatch run.sh --ds_short spice collagen --seed $SEED --ds_split_names "$splitpath/fold_5.json"
sbatch run.sh --ds_short spice collagen --seed $SEED --ds_split_names "$splitpath/fold_6.json"
sbatch run.sh --ds_short spice collagen --seed $SEED --ds_split_names "$splitpath/fold_7.json"
sbatch run.sh --ds_short spice collagen --seed $SEED --ds_split_names "$splitpath/fold_8.json"
sbatch run.sh --ds_short spice collagen --seed $SEED --ds_split_names "$splitpath/fold_9.json"