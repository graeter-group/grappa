#!/bin/bash

# This script installs grappa and its dependencies in a Conda environment.

# Usage: ./installation.sh [cuda_version] (default: 11.7)

# throw upon error:
set -e

# go to the directory of this script:
cd "$(dirname "$0")"

cuda_version="${1:-11.7}"

# Check if the provided CUDA version is supported
if [[ ! $cuda_version =~ ^(11.7|11.8|12.1)$ ]]; then
    echo "Unsupported CUDA version. Supported versions are 11.7, 11.8, and 12.1."
    exit 1
fi


# if 12.1, warn user that it is not recommended:
if [[ ! $cuda_version =~ ^(11.8|12.1)$ ]]; then
    echo "Installing with CUDA version 12.1/11.8 is not recommended.\n\tIt could not be verified that the dgl installation works due to problems finding the libcusparse.so.12 / libcusparse.so.11 shared object file."
    read -p "Continue? (y/n) " choice
    case "$choice" in 
        y|Y ) echo "Continuing installation...";;
        n|N ) echo "Installation aborted."; exit 1;;
        * ) echo "Invalid response. Installation aborted."; exit 1;;
    esac
fi

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

cuda_version="${cuda_version//.}" # e.g. from 11.7 to 117

echo "Installing grappa with CUDA version $cuda_version"

echo "Installing openmm from conda-forge..."
conda install python=3.9 openmm=7.7.0=py39hb10b54c_0 -c conda-forge -y

# for the rest use pip since it is much faster (and for some cuda versions, conda throws with unsolvable conflicts)

# Install specific versions of torch and pytorch-cuda
echo "Installing torch for CUDA version $cuda_version..."
if [ "$cuda_version" == "117" ]; then
    pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu$cuda_version
else
    pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu$cuda_version
fi

echo "Installing other requirements..."
pip install -r requirements.txt

echo "Installing dgl for CUDA version $cuda_version..."
pip install dgl -f "https://data.dgl.ai/wheels/cu$cuda_version/repo.html"
pip install dglgo -f "https://data.dgl.ai/wheels-test/repo.html"

echo "installing grappa..."
pip install -e .

# go back to the original directory:
cd - > /dev/null

echo "Installation complete."
