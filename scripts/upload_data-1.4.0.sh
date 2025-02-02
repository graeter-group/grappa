DATASETS=("spice-dipeptide-amber99" "spice-dipeptide-charmm36" "protein-torsion-amber99" "protein-torsion-charmm36")

TAG=v.1.4.0

python upload_datasets.py -t $TAG -d ${DATASETS[@]}