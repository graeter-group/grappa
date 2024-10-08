DATASETS=("dipeptides-300K-amber99" "dipeptides-1000K-amber99" "uncapped-300K-amber99" "dipeptides-hyp-dop-300K-amber99" "dipeptides-radical-300K" "bondbreak-radical-peptides-300K")
DATASETS+=("dipeptides-300K-openff-1.2.0" "dipeptides-1000K-openff-1.2.0" "uncapped-300K-openff-1.2.0")
DATASETS+=("dipeptides-300K-charmm36_nonb" "dipeptides-1000K-charmm36_nonb")
DATASETS+=("espaloma_split")
DATASETS+=("spice-pubchem" "rna-nucleoside" "gen2" "spice-des-monomers" "spice-dipeptide" "rna-diverse" "gen2-torsion" "pepconf-dlc" "protein-torsion" "rna-trinucleotide")

TAG=v.1.3.0

python upload_datasets.py -t $TAG -d ${DATASETS[@]}