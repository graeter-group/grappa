DATASETS=("spice-pubchem-filtered")

TAG=v.1.3.1

python upload_datasets.py -t $TAG -d ${DATASETS[@]}