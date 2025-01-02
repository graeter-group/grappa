DATASETS=("peptide-radical-MD" "peptide-radical-scan" "peptide-radical-opt")

TAG=v.1.4.1

python upload_datasets.py -t $TAG -d ${DATASETS[@]}