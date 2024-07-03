#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # dir in which this script lies



espaloma_ds_path="$SCRIPT_DIR/../../data/esp_data"

# stop upon error:
set -e

pushd $SCRIPT_DIR


tar -xvzf "$espaloma_ds_path/duplicated-isomeric-smiles-merge.tar.gz" -C $espaloma_ds_path
tar -xvzf "$espaloma_ds_path/spice-des-monomers.tar.gz" -C $espaloma_ds_path
tar -xvzf "$espaloma_ds_path/gen2-opt.tar.gz" -C $espaloma_ds_path
tar -xvzf "$espaloma_ds_path/gen2-torsion.tar.gz" -C $espaloma_ds_path
tar -xvzf "$espaloma_ds_path/pepconf-opt.tar.gz" -C $espaloma_ds_path
tar -xvzf "$espaloma_ds_path/protein-torsion.tar.gz" -C $espaloma_ds_path
tar -xvzf "$espaloma_ds_path/rna-diverse.tar.gz" -C $espaloma_ds_path
tar -xvzf "$espaloma_ds_path/rna-nucleoside.tar.gz" -C $espaloma_ds_path
tar -xvzf "$espaloma_ds_path/rna-trinucleotide.tar.gz" -C $espaloma_ds_path
tar -xvzf "$espaloma_ds_path/spice-dipeptide.tar.gz" -C $espaloma_ds_path
tar -xvzf "$espaloma_ds_path/spice-pubchem.tar.gz" -C $espaloma_ds_path

# remove tar files:
pushd $espaloma_ds_path
rm *.tar.gz
popd


popd