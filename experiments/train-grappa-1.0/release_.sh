# creates the grappa 1.0 release and uploads model and datasets to github using github cli

# get an abspath to this script
THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# throw upon error
set -e

# cd to /grappa:
pushd $THISDIR/../..

RUNDIR=$THISDIR/wandb/run-20240209_112055-atbogjqt

MODELNAME=grappa-1.0-20240209

MODELPATH=$RUNDIR/files/checkpoints/best-model.ckpt

TAG='v.1.0.0'

# create release
# gh release create $TAG

# upload the model:
pushd 'models'
python export_model.py $MODELPATH $MODELNAME --release_tag $TAG
popd

# upload files
DATASETS=(
  spice-des-monomers
  spice-pubchem
  gen2
  gen2-torsion
  rna-diverse
  rna-trinucleotide
  rna-nucleoside
  spice-dipeptide
  protein-torsion
  pepconf-dlc
  spice-dipeptide_amber99sbildn
  protein-torsion_amber99sbildn
  pepconf-dlc_amber99sbildn
  tripeptides_amber99sbildn
  dipeptide_rad
  hyp-dop_amber99sbildn
)

DATADIR=data/dgl_datasets


# for each dir, upload a zipped version of it:
for dir in "${DATASETS[@]}"; do
    echo "Uploading $dir. zipping..."

    # Navigate to the directory just above the target directory
    dir="$DATADIR/$dir"
    parent_dir=$(dirname "$dir")
    dir_name=$(basename "$dir")

    # Change to the parent directory
    pushd "$parent_dir"

    # Create zip file of the directory
    zip -r "$dir_name.zip" "$dir_name"

    # Go back to the original directory
    popd

    # Upload the zip file
    gh release upload $TAG "$parent_dir/$dir_name.zip"

    # Remove the zip file
    rm "$parent_dir/$dir_name.zip"

done

popd