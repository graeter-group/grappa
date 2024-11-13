TAG=v.1.3.1

CKPT=../models/grappa-1.3.2/grappa.ckpt
NAME=grappa-1.3.2
DESCR="Covers peptides, small molecules, rna"

grappa_export -n $NAME -c $CKPT -m "$DESCR"

python release_model.py -n $NAME -t $TAG