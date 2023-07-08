# Graph Attentional Protein Parametrization (GrAPPa)

Collection of models and utilities for protein force fields with parameters learned by deep graph neural networks. Usage: See grappa.ff.Forcefield class and mains/tutorials.


## Installation
Openmm and openff-toolkit are requirements that will be made optional in the future. Currently they are necessary and cannot be installed succesfull using pip or from source. Therefore one has to use conda to install the package. It is advised to use mamba since conda can be very slow in resolving dependencies.

- If not installed already, install mamba using conda (has to be in base environment, see https://mamba.readthedocs.io/en/latest/installation.html):
```
conda activate base
conda install -n base --override-channels -c conda-forge mamba 'python_abi=*=*cp*'
```
- Create a new conda environment, install grappas dependecies using mamba, clone the repository and install grappa from source:
```
conda env create -n grappa && conda activate grappa
mamba env update --file environment.yml

git clone git@github.com:hits-mbm-dev/grappa.git; cd grappa;
pip install -e .
```
