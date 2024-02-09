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
