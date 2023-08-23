#!/bin/bash

SEED=${1:-1}

splitpath="/hits/fast/mbm/seutelf/grappa/mains/split_names/grappa_pep"

sbatch split.sh $splitpath spice collagen radical_AAs radical_dipeptides

sbatch run.sh --ds_short spice collagen radical_AAs radical_dipeptides --seed $SEED --ds_split_names "$splitpath/fold_0.json" --name fold_model
sbatch run.sh --ds_short spice collagen radical_AAs radical_dipeptides --seed $SEED --ds_split_names "$splitpath/fold_1.json" --name fold_model
sbatch run.sh --ds_short spice collagen radical_AAs radical_dipeptides --seed $SEED --ds_split_names "$splitpath/fold_2.json" --name fold_model
sbatch run.sh --ds_short spice collagen radical_AAs radical_dipeptides --seed $SEED --ds_split_names "$splitpath/fold_3.json" --name fold_model
sbatch run.sh --ds_short spice collagen radical_AAs radical_dipeptides --seed $SEED --ds_split_names "$splitpath/fold_4.json" --name fold_model
sbatch run.sh --ds_short spice collagen radical_AAs radical_dipeptides --seed $SEED --ds_split_names "$splitpath/fold_5.json" --name fold_model
sbatch run.sh --ds_short spice collagen radical_AAs radical_dipeptides --seed $SEED --ds_split_names "$splitpath/fold_6.json" --name fold_model
sbatch run.sh --ds_short spice collagen radical_AAs radical_dipeptides --seed $SEED --ds_split_names "$splitpath/fold_7.json" --name fold_model
sbatch run.sh --ds_short spice collagen radical_AAs radical_dipeptides --seed $SEED --ds_split_names "$splitpath/fold_8.json" --name fold_model
sbatch run.sh --ds_short spice collagen radical_AAs radical_dipeptides --seed $SEED --ds_split_names "$splitpath/fold_9.json" --name fold_model