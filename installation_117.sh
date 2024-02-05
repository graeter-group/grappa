#!/bin/bash

# DOESNT WORK WITH PYTHON 3.10 due to dgl cu-sparse

# This script installs grappa and its dependencies in a Conda environment.

# throw upon error:
set -e

# go to the directory of this script:
cd "$(dirname "$0")"


# Check if the current Conda environment is 'base'. If so ask the user if they want to continue.
current_env=$(conda info --json | grep '"default_prefix":' | cut -d'"' -f4 | xargs basename)
echo "Installing grappa to conda environment: $current_env"
if [ "$current_env" == "conda" ]; then
    read -p "You are in the base Conda environment. It is recommended to use a separate environment. Continue? (y/n) " choice
    case "$choice" in 
        y|Y ) echo "Continuing installation...";;
        n|N ) echo "Installation aborted."; exit 1;;
        * ) echo "Invalid response. Installation aborted."; exit 1;;
    esac
fi


echo "Installing openmm for cuda 11.7..."
conda install python=3.9 openmm=7.7.0=py39hb10b54c_0 cuda-toolkit -c conda-forge -c "nvidia/label/cuda-11.7.1" -y

# for the rest use pip since it is much faster (and for some cuda versions, conda throws with unsolvable conflicts)
echo 'installing torch...'
pip install torch==2.0.1

echo "Installing dgl for cuda 11.7..."
pip install dgl -f "https://data.dgl.ai/wheels/cu117/repo.html"
pip install dglgo -f "https://data.dgl.ai/wheels-test/repo.html"

echo "Installing other requirements..."
pip install -r requirements.txt

echo "installing grappa..."
pip install -e .

# go back to the original directory:
cd - > /dev/null

echo "Installation complete."
