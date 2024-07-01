DATASETS=("spice-pubchem" "rna-nucleoside" "gen2" "spice-des-monomers" "spice-dipeptide" "rna-diverse" "gen2-torsion" "pepconf-dlc" "protein-torsion" "rna-trinucleotide")

DATASETS=$DATASETS + ("dipeptides-300K-amber99" "dipeptides-1000K-amber99" "uncapped-300K-amber99" "dipeptides-hyp-dop-300K-amber99" "dipeptides-radical-300K" "bondbreak-radical-peptides-300K")
DATASETS=$DATASETS + ("dipeptides-300K-openff-1.2.0" "dipeptides-1000K-openff-1.2.0" "uncapped-300K-openff-1.2.0")
DATASETS=$DATASETS + ("dipeptides-300K-charmm_nonb" "dipeptides-1000K-charmm_nonb")
DATASETS=$DATASETS + ("espaloma_split")

python -c "from grappa.training.export_model import release_model; release_model() "