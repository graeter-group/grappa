# Graph Attentional Protein Parametrization (GrAPPa)

Collection of models and utilities for protein force fields with parameters learned by deep graph neural networks. Usage: See `examples/usage`.

## Installation

### Install Script

Simply activate the target conda environment and execute `./installation.sh 11.7` to install grappa for cuda version 11.7. If you want to install grappa for a different cuda version, replace 11.7 with the desired version. As of now, only `11.7` and `11.8` have been tested.

### Manual Installation

Alternatively, you can install grappa manually by the following steps:

Unfortunately, openmm is not available on pip and has to be install via conda. It is recommended to use the openmm=7.7.0=py39hb10b54c_0 version.

DGL has to be installed separately since index files are needed ([dgl installation](https://www.dgl.ai/pages/start.html)). Modify the cuda version in the script below to your needs.

```{bash}
git clone git@github.com:hits-mbm-dev/grappa.git
cd grappa

conda create -n grappa python=3.9 openmm=7.7.0=py39hb10b54c_0 -c conda-forge -y
conda activate grappa

pip install torch==2.1.0 pytorch-cuda=11.7

pip install -r requirements.txt

pip install dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html

pip install -e .
```

## Pretrained Models

Pretrained models can be obtained by using `grappa.utils.run_utils.model_from_tag` with a tag (e.g.`latest`) that will point to a url that points to a version-dependent release file, from which model weights are downloaded. An example can be found at `examples/usage/openmm_wrapper.py`.

## Datasets

Datasets of dgl graphs representing molecules can be obtained by using the `grappa.data.Dataset.from_tag` constructor. An example can be found at `examples/usage/evaluation.py`. Available tags are listed in the documentation of the Dataset class.

To re-create the benchmark experiment, also the splitting into train/val/test sets is needed. This can be done by running `dataset_creation/get_espaloma_split/save_split.py` has to be run. This will create a file `espaloma_split.json` that contains lists of smilestrings for each of the sub-datasets. These are used to classify molecules as being train/val/test molecules upon loading the dataset in the train scripts from `experiments/benchmark`.