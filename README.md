# Graph Attentional Protein Parametrization (GrAPPa)

Collection of models and utilities for protein force fields with parameters learned by deep graph neural networks. Usage: See `examples/usage`.

## Installation

Unfortunately, openmm is not available on pip and has to be install via conda. It is recommended to use the openmm=7.7.0=py39hb10b54c_0 version.

DGL has to be installed separately since index files are needed ([dgl installation](https://www.dgl.ai/pages/start.html)). Modify the cu117 in the index file below to your needs as explained in the dgl installation guide.

```{bash}
git clone git@github.com:hits-mbm-dev/grappa.git
cd grappa

conda create -n grappa python=3.9 openmm=7.7.0=py39hb10b54c_0 -c conda-forge -y
conda activate grappa

pip install -r requirements.txt

pip install dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html

pip install -e .
```

## Pretrained Models

Pretrained models can be obtained by using `grappa.utils.run_utils.load_model(url)` with an url that points to a release file. An example can be found at `examples/usage/openmm_wrapper.py`. There will be release-dependent tags linking to the proper urls in the future.

## Datasets

Datasets of dgl graphs representing molecules can be obtained by using the `grappa.data.Dataset.from_tag(tag)` constructor. An example can be found at `examples/usage/evaluation.py`. Available tags are listed in the documentation of the Dataset class.
