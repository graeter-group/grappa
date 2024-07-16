NOISE=0.5
FF_NAME=noise-${NOISE}
FF_ORIG=amber99
FF=amber99sbildn.xml

DATASETS=(dipeptides-300K-amber99 dipeptides-1000K-amber99)

# replace amber99:
OUT_DATASETS=(${DATASETS[@]/$FF_ORIG/$FF_NAME})

echo "DATASETS: ${DATASETS[@]}"
echo "OUT_DATASETS: ${OUT_DATASETS[@]}"

for i in ${!DATASETS[@]}; do
    dataset=${DATASETS[$i]}
    out_dataset=${OUT_DATASETS[$i]}
    echo "Processing $dataset"
    python reparametrize.py $dataset $out_dataset $FF --charge_noise $NOISE
done


NOISE=0.2
FF_NAME=noise-${NOISE}
FF_ORIG=amber99
FF=amber99sbildn.xml

DATASETS=(dipeptides-300K-amber99 dipeptides-1000K-amber99)

# replace amber99:
OUT_DATASETS=(${DATASETS[@]/$FF_ORIG/$FF_NAME})

echo "DATASETS: ${DATASETS[@]}"
echo "OUT_DATASETS: ${OUT_DATASETS[@]}"

for i in ${!DATASETS[@]}; do
    dataset=${DATASETS[$i]}
    out_dataset=${OUT_DATASETS[$i]}
    echo "Processing $dataset"
    python reparametrize.py $dataset $out_dataset $FF --charge_noise $NOISE
done