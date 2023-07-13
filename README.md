# Graph Attentional Protein Parametrization (GrAPPa)

Collection of models and utilities for protein force fields with parameters learned by deep graph neural networks. Usage: See grappa.ff.Forcefield class and mains/tutorials.


## Installation
Openmm is a requirement that will be made optional in the future. Currently they are necessary and cannot be installed using pip.
Therefore one has to use conda to install the package.

- Create a new conda environment, **install grappas dependecies using conda**, clone the repository and install grappa from source:
```
conda activate base && mamba create -n grappa && conda activate grappa
conda env update --file environment.yml

git clone git@github.com:hits-mbm-dev/grappa.git; cd grappa;
pip install -e .
```

- For the **grappa.PDBData.matching submodule**, providing the creation of pdbfiles from positions and element alone, one also needs to install ase. In this case use opt_environment.yml or install it via ```conda install -c conda-forge ase```.
- If you wish to train models yourself, the additional package pytorch-warmup is required. Install it via ```pip install pytorch-warmup``` or use opt_environment.yml.