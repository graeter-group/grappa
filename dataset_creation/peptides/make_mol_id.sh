DATASETS=(dipeptides-300K-amber99 dipeptides-1000K-amber99)

for dataset in ${DATASETS[@]}; do
    echo "Processing $dataset"
    python find_spice_correspondent.py $dataset
done