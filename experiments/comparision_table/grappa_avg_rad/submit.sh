#!/bin/bash

SEED=${1:-1}

splitpath="/hits/fast/mbm/seutelf/grappa/mains/split_names/grappa_pep"

sbatch split.sh $splitpath spice collagen radical_AAs_avg radical_dipeptides_avg

sbatch run.sh --ds_short spice collagen radical_AAs_avg radical_dipeptides_avg --seed $SEED --ds_split_names "$splitpath/fold_0.json" --name fold_0
sbatch run.sh --ds_short spice collagen radical_AAs_avg radical_dipeptides_avg --seed $SEED --ds_split_names "$splitpath/fold_1.json" --name fold_1
sbatch run.sh --ds_short spice collagen radical_AAs_avg radical_dipeptides_avg --seed $SEED --ds_split_names "$splitpath/fold_2.json" --name fold_2
sbatch run.sh --ds_short spice collagen radical_AAs_avg radical_dipeptides_avg --seed $SEED --ds_split_names "$splitpath/fold_3.json" --name fold_3
sbatch run.sh --ds_short spice collagen radical_AAs_avg radical_dipeptides_avg --seed $SEED --ds_split_names "$splitpath/fold_4.json" --name fold_4
sbatch run.sh --ds_short spice collagen radical_AAs_avg radical_dipeptides_avg --seed $SEED --ds_split_names "$splitpath/fold_5.json" --name fold_5
sbatch run.sh --ds_short spice collagen radical_AAs_avg radical_dipeptides_avg --seed $SEED --ds_split_names "$splitpath/fold_6.json" --name fold_6
sbatch run.sh --ds_short spice collagen radical_AAs_avg radical_dipeptides_avg --seed $SEED --ds_split_names "$splitpath/fold_7.json" --name fold_7
sbatch run.sh --ds_short spice collagen radical_AAs_avg radical_dipeptides_avg --seed $SEED --ds_split_names "$splitpath/fold_8.json" --name fold_8
sbatch run.sh --ds_short spice collagen radical_AAs_avg radical_dipeptides_avg --seed $SEED --ds_split_names "$splitpath/fold_9.json" --name fold_9