#!/bin/bash

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


echo "Installing openmm and pytorch in cpu mode..."
conda install python=3.10 openmm=8.1.1 pytorch=2.2.0 cpuonly -c pytorch -c conda-forge -y

# for the rest use pip since it is much faster (and for some cuda versions, conda throws with unsolvable conflicts)

echo "Installing dgl in cpu mode..."
pip install dgl -f https://data.dgl.ai/wheels/repo.html
pip install dglgo -f "https://data.dgl.ai/wheels-test/repo.html"

echo "Installing other requirements..."
pip install -r requirements.txt

echo "installing grappa..."
pip install -e .

# go back to the original directory:
cd - > /dev/null

echo "Installation complete."
