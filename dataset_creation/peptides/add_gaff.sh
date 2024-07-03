DATASETS=(dipeptides-300K-openff-1.2.0 dipeptides-1000K-openff-1.2.0 uncapped-300K-openff-1.2.0)

for dataset in ${DATASETS[@]}; do
    echo "Processing $dataset"
    python calc_gaff.py $dataset
done