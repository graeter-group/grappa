TAG=v.1.3.0

CKPT=../ckpt/grappa-1.3/published/2024-06-26_01-30-36/epoch:789-early_stop_loss:19.65.ckpt
NAME=grappa-1.3.0
DESCR="Covers peptides, small molecules, rna and radical peptides"

grappa_export -n $NAME -c $CKPT -m "$DESCR"

python release_model.py -n $NAME -t $TAG