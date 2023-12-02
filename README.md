# Graph Attentional Protein Parametrization (GrAPPa)

Collection of models and utilities for protein force fields with parameters learned by deep graph neural networks. Usage: See `examples/usage`.


## Installation
```
conda create -n grappa python=3.9
conda activate grappa
bash envs/cuda_117.sh
```


## Pretrained Models
Pretrained models can be obtained by using `grappa.utils.run_utils.load_model(url)` with an url that points to a release file. An example can be found at `examples/usage/openmm_wrapper.py`. There will be release-dependent tags linking to the proper urls in the future.

## Datasets
Datasets of dgl graphs representing molecules can be obtained by using the `grappa.data.Dataset.from_tag(tag)` constructor. An example can be found at `examples/usage/evaluation.py`. Available tags are listed in the documentation of the Dataset class.