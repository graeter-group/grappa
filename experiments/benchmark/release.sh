# creates the grappa 1.1 release and uploads model and datasets to github using github cli.

# throw upon error
set -e

# get an abspath to this script
THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

RUN_ID=leif-seute/benchmark-grappa-1.0/te5k7pbo

MODELNAME=grappa-1.1-benchmark

TAG='v.1.1.0'

# create release
# gh release create $TAG

# cd to this directory:
pushd $THISDIR

# # export model to local models directory
bash prepare_release.sh # NOTE: UNCOMMENT THIS LINE TO EXPORT AND EVALUATE THE MODEL (will take some time)
grappa_release -t $TAG -m $MODELNAME
