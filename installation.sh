#!/bin/bash

# This script installs grappa and its dependencies in the current Conda environment based on the CUDA version specified.
# call: ./installation.sh [cuda_version]
# with cuda_version being one of the following: 11.7, 11.8, 12.1, cpu

# throw upon error:
set -e

# Check for the provided argument (CUDA version or CPU mode)
if [ $# -eq 0 ]; then
    echo "No arguments provided. Defaulting to CPU mode."
    cuda_version="cpu"
else
    cuda_version=$1
fi

# Function definitions for each installation procedure
install_11_7() {
    echo "Installing for CUDA version 11.7..."
    conda install python=3.9 openmm=7.7.0=py39hb10b54c_0 cuda-toolkit -c conda-forge -c "nvidia/label/cuda-11.7.1" -y
    pip install torch==2.0.1
    pip install dgl -f "https://data.dgl.ai/wheels/cu117/repo.html"
    pip install dglgo -f "https://data.dgl.ai/wheels-test/repo.html"
}

install_11_8() {
    echo "Installing for CUDA version 11.8..."
    conda install python=3.10 openmm=8.1.1 pytorch=2.2.0 pytorch-cuda=11.8 cudatoolkit=11.8 -c nvidia -c pytorch -c conda-forge -y
    pip install dgl -f "https://data.dgl.ai/wheels/cu118/repo.html"
    pip install dglgo -f "https://data.dgl.ai/wheels-test/repo.html"
}

install_12_1() {
    echo "Installing for CUDA version 12.1..."
    conda install python=3.10 openmm=8.1.1 pytorch=2.2.0 pytorch-cuda=12.1 -c nvidia -c pytorch -c conda-forge -y
    pip install dgl -f "https://data.dgl.ai/wheels/cu121/repo.html"
    pip install dglgo -f "https://data.dgl.ai/wheels-test/repo.html"
}

install_cpu() {
    echo "Installing for CPU mode..."
    conda install python=3.10 openmm=8.1.1 pytorch=2.2.0 cpuonly -c pytorch -c conda-forge -y
    pip install dgl -f https://data.dgl.ai/wheels/repo.html
    pip install dglgo -f "https://data.dgl.ai/wheels-test/repo.html"
}

# Main installation process
# Going to the directory of this script
cd "$(dirname "$0")"

# Environment check
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

# Installation based on CUDA version or CPU mode
case $cuda_version in
    11.7)
        install_11_7
        ;;
    11.8)
        install_11_8
        ;;
    12.1)
        install_12_1
        ;;
    cpu)
        install_cpu
        ;;
    *)
        echo "Invalid argument. Please provide one of the following: 11.7, 11.8, 12.1, cpu. Defaulting to CPU mode."
        install_cpu
        ;;
esac

# Common steps for all installations
echo "Installing other requirements..."
pip install -r requirements.txt

echo "Installing grappa..."
pip install -e .

# Go back to the original directory
cd - > /dev/null

echo "Installation complete."
