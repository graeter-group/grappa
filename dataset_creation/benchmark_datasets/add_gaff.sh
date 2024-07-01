DATASETS=("spice-pubchem" "rna-nucleoside" "gen2" "spice-des-monomers" "spice-dipeptide" "rna-diverse" "gen2-torsion" "pepconf-dlc" "protein-torsion" "rna-trinucleotide")

for dataset in ${DATASETS[@]}; do
    echo "Processing $dataset"
    python ../peptides/calc_gaff.py $dataset
done