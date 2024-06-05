#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # dir in which this script lies

espaloma_ds_path="$SCRIPT_DIR/../../data/esp_data"

mkdir -p $espaloma_ds_path

# Download files from Zenodo
wget "https://zenodo.org/records/8150601/files/duplicated-isomeric-smiles-merge.tar.gz" -P $espaloma_ds_path
wget "https://zenodo.org/records/8150601/files/spice-des-monomers.tar.gz" -P $espaloma_ds_path
# wget "https://zenodo.org/records/8150601/files/gen2-opt.tar.gz" -P $espaloma_ds_path
# wget "https://zenodo.org/records/8150601/files/gen2-torsion.tar.gz" -P $espaloma_ds_path
# wget "https://zenodo.org/records/8150601/files/pepconf-opt.tar.gz" -P $espaloma_ds_path
# wget "https://zenodo.org/records/8150601/files/protein-torsion.tar.gz" -P $espaloma_ds_path
# wget "https://zenodo.org/records/8150601/files/rna-diverse.tar.gz" -P $espaloma_ds_path
# wget "https://zenodo.org/records/8150601/files/rna-nucleoside.tar.gz" -P $espaloma_ds_path
# wget "https://zenodo.org/records/8150601/files/rna-trinucleotide.tar.gz" -P $espaloma_ds_path
# wget "https://zenodo.org/records/8150601/files/spice-dipeptide.tar.gz" -P $espaloma_ds_path
# wget "https://zenodo.org/records/8150601/files/spice-pubchem.tar.gz" -P $espaloma_ds_path