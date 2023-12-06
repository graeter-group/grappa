set -e # stop upon error

# check whether the base environment is active. if yes, abort:
# get name of current conda environment
ENV_NAME=$(basename "$CONDA_PREFIX")
# if name is conda, abort since this is the base environment
if [ "$ENV_NAME" = "conda" ]; then
    echo "Please activate a conda environment other than base before running this script."
    exit 1
fi

# old openmm needs python 3.9 (current openmm does not work with cuda-11.7-torch)
conda install python=3.9 openmm=7.7.0=py39hb10b54c_0 cuda-toolkit -c conda-forge -c "nvidia/label/cuda-11.7.1" -y
pip install numpy matplotlib rdkit torch==2.0.0 torchvision==0.15.1 dgl -f https://data.dgl.ai/wheels/cu117/repo.html dglgo -f https://data.dgl.ai/wheels-test/repo.html pytorch-lightning wandb pytorch_warmup

# install grappa:
THIS_FILE=$(readlink -f "${BASH_SOURCE[0]}")
THIS_DIR=$(dirname "$THIS_FILE")
pip install -e "$THIS_DIR/../"