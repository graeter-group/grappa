TAG=v.1.4.0

CKPT=../models/grappa-1.4.0/checkpoint.ckpt
NAME=grappa-1.4.0
DESCR="Covers peptides, small molecules, rna"

grappa_export -n $NAME -c $CKPT -m "$DESCR"

python release_model.py -n $NAME -t $TAG