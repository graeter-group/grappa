FF=openff-1.2.0
FF_ORIG=amber99

DATASETS=(dipeptides-300K-amber99 dipeptides-1000K-amber99 uncapped-300K-amber99)

# replace amber99 with charmm36:
OUT_DATASETS=(${DATASETS[@]/$FF_ORIG/$FF})

echo "DATASETS: ${DATASETS[@]}"
echo "OUT_DATASETS: ${OUT_DATASETS[@]}"

for i in ${!DATASETS[@]}; do
    dataset=${DATASETS[$i]}
    out_dataset=${OUT_DATASETS[$i]}
    echo "Processing $dataset"
    python reparametrize.py $dataset $out_dataset openff_unconstrained-1.2.0.offxml --ff_type openff
done