#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # dir in which this script lies



espaloma_ds_path="$SCRIPT_DIR/../data/esp_data"

# stop upon error:
set -e

tar -xvzf duplicated-isomeric-smiles-merge.tar.gz -C $espaloma_ds_path
tar -xvzf gen2-opt.tar.gz -C $espaloma_ds_path
tar -xvzf gen2-torsion.tar.gz -C $espaloma_ds_path
tar -xvzf pepconf-opt.tar.gz -C $espaloma_ds_path
tar -xvzf protein-torsion.tar.gz -C $espaloma_ds_path
tar -xvzf rna-diverse.tar.gz -C $espaloma_ds_path
tar -xvzf rna-nucleoside.tar.gz -C $espaloma_ds_path
tar -xvzf rna-trinucleotide.tar.gz -C $espaloma_ds_path
tar -xvzf spice-des-monomers.tar.gz -C $espaloma_ds_path
tar -xvzf spice-dipeptide.tar.gz -C $espaloma_ds_path
tar -xvzf spice-pubchem.tar.gz -C $espaloma_ds_path

cwd=$(pwd)
cd $SCRIPT_DIR

# remove tar files:
rm *.tar.gz

cd $cwd