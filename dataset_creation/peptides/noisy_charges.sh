NOISE=0.1
FF_NAME=noise-0-1
FF_ORIG=amber99
FF=amber99sbildn.xml

DATASETS=(dipeptides-300K-amber99)

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


NOISE=0.05
FF_NAME=noise-0-05
FF_ORIG=amber99
FF=amber99sbildn.xml

DATASETS=(dipeptides-300K-amber99)

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


NOISE=0.3
FF_NAME=noise-0-3
FF_ORIG=amber99
FF=amber99sbildn.xml

DATASETS=(dipeptides-300K-amber99)

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
