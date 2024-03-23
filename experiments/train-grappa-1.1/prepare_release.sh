# creates the grappa 1.1 release and uploads model and datasets to github using github cli.

# throw upon error
set -e

# get an abspath to this script
THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

RUN_ID=leif-seute/grappa-1.1/4te386b4

MODELNAME=grappa-1.1.1


# cd to this directory:
pushd $THISDIR

# export model to local models directory
grappa_export --id $RUN_ID --modelname $MODELNAME
grappa_eval --modeltag $MODELNAME