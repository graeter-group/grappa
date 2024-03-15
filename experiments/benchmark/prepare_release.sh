# creates the grappa 1.1 release and uploads model and datasets to github using github cli.

# throw upon error
set -e

# get an abspath to this script
THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

RUN_ID=leif-seute/benchmark-grappa-1.0/te5k7pbo

MODELNAME=grappa-1.1-benchmark


# cd to this directory:
pushd $THISDIR

# export model to local models directory
grappa_export --id $RUN_ID --modelname $MODELNAME
grappa_eval --modeltag $MODELNAME --with_train