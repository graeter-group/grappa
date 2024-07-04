SCRIPTDIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# cd to root:
popd '$SCRIPTDIR/../../' > /dev/null

# find all best checkpoint paths:
CKPT_PARENT...