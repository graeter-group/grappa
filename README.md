# Graph Attentional Protein Parametrization (GrAPPa)

_A machine-learned molecular mechanics force field using a deep graph attentional network <br>(code supporting [https://arxiv.org/abs/2404.00050](https://arxiv.org/abs/2404.00050))_

## Abstract

Simulating large molecular systems over long timescales requires force fields that are both accurate and efficient. In recent years, E(3) equivariant neural networks have lifted the tension between computational efficiency and accuracy of force fields, but they are still several orders of magnitude more expensive than classical molecular mechanics (MM) force fields.

Here, we propose a novel machine learning architecture to predict MM parameters from the molecular graph, employing a graph attentional neural network and a transformer with symmetry-preserving positional encoding. The resulting force field, Grappa, outperforms established and other machine-learned MM force fields in terms of accuracy at the same computational efficiency and can be used in existing Molecular Dynamics (MD) engines like GROMACS and OpenMM. It predicts energies and forces of small molecules, peptides, RNA and - showcasing its extensibility to uncharted regions of chemical space - radicals at state-of-the-art MM accuracy. We demonstrate Grappa's transferability to macromolecules in MD simulations, during which large protein are kept stable and small proteins can fold. Our force field sets the stage for biomolecular simulations close to chemical accuracy, but with the same computational cost as established protein force fields.

<details open>
  <summary>Grappa Overview</summary>
  <p align="center">
    <img src="docs/figures/grappa_overview.png" width="50%" style="max-width: 200px; display: block; margin: auto;">
  </p>
  <p><i>Grappa first predicts node embeddings from the molecular graph. In a second step, it predicts MM parameters for each n-body interaction from the embeddings of the contributing nodes, respecting the necessary permutation symmetry.</i></p>
</details>

<details>
  <summary><b>Performance on MM Benchmark Datasets</b></summary>
  <p align="center">
    <img src="docs/figures/table_benchmark.png" width="100%" style="max-width: 200px; display: block; margin: auto;">
  </p>
  <p><i>Grappa's energy and force-component RMSE in kcal/mol and kcal/mol/Å on the test dataset (trained with the same train-val-test partition) from Espaloma [<a href="https://arxiv.org/abs/2307.07085v4">Takaba et al. 2023</a>], compared with classical forcefields [<a href="https://pubs.aip.org/aip/jcp/article/153/11/114502/199591/A-fast-and-high-quality-charge-model-for-the-next">He et al.</a>], [<a href="https://doi.org/10.1021/acs.jctc.5b00255">Maier et al.</a>, <a href="https://pubs.acs.org/doi/10.1021/ct200162x">Zgarbova et al.</a>]</i></p>
</details>



<details open><summary><b>Table of contents</b></summary>
  
- [Usage](#usage)
- [Installation](#installation)
- [Pretrained Models](#pretrained-models)
- [Datasets](#datasets)
- [Training](#training)
- [Reproducibility](#reproducibility)
</details>


## Usage

The current version of Grappa only predicts bonded parameters; the nonbonded parameters like partial charges and Lennard Jones parameters are predicted with a traditional force field of choice.
The input to Grappa is therefore a representation of the system of interest that already contains information on the nonbonded parameters.
Currently, Grappa is compatible with GROMACS and OpenMM.

For complete example scripts, see `examples/usage`.

### GROMACS

In GROMACS, Grappa can be used as command line application that receives the path to a topology file and writes the bonded parameters in a new topology file.

```{bash}
# parametrize the system with a traditional forcefield:
gmx pdb2gmx -f your_protein.pdb -o your_protein.gro -p topology.top -ignh

# create a new topology file with the bonded parameters from Grappa, specifying the tag of the grappa model:
grappa_gmx -f topology.top -o topology_grappa.top -t grappa-1.3 -p

# (you can create a plot of the parameters for inspection using the -p flag)

# continue with ususal gromacs workflow (solvation etc.)
```

### OpenMM

To use Grappa in OpenMM, parametrize your system with a traditional forcefield, from which the nonbonded parameters are taken, and then pass it to Grappas OpenMM wrapper class:

```{python}
from openmm.app import ForceField, Topology
from grappa import OpenmmGrappa

topology = ... # load your system as openmm.Topology

classical_ff = ForceField('amber99sbildn.xml', 'tip3p.xml')
system = classical_ff.createSystem(topology)

# load the pretrained ML model from a tag. Currently, possible tags are 'grappa-1.3' and 'latest'
grappa_ff = OpenmmGrappa.from_tag('grappa-1.3')

# parametrize the system using grappa.
system = grappa_ff.parametrize_system(system, topology, charge_model='amber99')
```

There is also the option to obtain an openmm.app.ForceField that calls Grappa for bonded parameter prediction behind the scenes:

```{python}
from openmm.app import ForceField, Topology
from grappa import as_openmm

topology = ... # load your system as openmm.Topology

grappa_ff = as_openmm('grappa-1.3', base_forcefield=['amber99sbildn.xml', 'tip3p.xml'])

system = grappa_ff.createSystem(topology)
```

## Installation

For using Grappa in GROMACS or OPENMM, cpu mode is usually sufficient since the inference runtime of Grappa is usually small compared to the simulation runtime.

To install Grappa, simply clone the repository, install additional requirements and the package itself with pip:

```{bash}
git clone https://github.com/hits-mbm-dev/grappa.git
cd grappa

conda create -n grappa python=3.10 -y
conda activate grappa

pip install -r installation/cpu_requirements.txt
pip install -e .
```

Verify the installation by running
```
python tests/test_installation.py
```

### GROMACS

The creation of custom GROMACS topology files is handled by [Kimmdy](https://github.com/hits-mbm-dev/kimmdy), which can be installed in the same environment as grappa via pip,

```{bash}
pip install kimmdy==6.8.3
```

### OpenMM

Unfortunately, OpenMM is not available on pip and has to be installed via conda in the same environment as grappa,

```{bash}
conda install -c conda-forge openmm
```

Since the resolution of packages can be slow in conda, it is recommended to install OpenMM first and then install Grappa.


### Development

To facilitate the interface to OpenMM and GROMACS, Grappa has an optional dependency on [OpenMM](https://github.com/openmm/openmm) and [Kimmdy](https://github.com/hits-mbm-dev/kimmdy), which is used to create custom GROMACS topology files. To train and evaluate Grappa on existing datasets, neither of these packages are needed.

In this case, only an environment with a working installation of [PyTorch](https://pytorch.org/) and [DGL](https://www.dgl.ai/) for the cuda version of choice is needed. In this environment, Grappa can be installed by

```{bash}
pip install -r installation/requirements.txt
pip install -e .
```

Verify the installation by running
```
python tests/test_installation.py
```

The installation of dgl with potentially necessary bugfixes is explained in `installation/README.md`.

## Pretrained Models

Pretrained models can be obtained by using `grappa.utils.run_utils.model_from_tag` with a tag (e.g. `latest`) that will point to a version-dependent url, from which model weights are downloaded.
Available models are listed in `models/published_models.csv`.
An example can be found at `examples/usage/openmm_wrapper.py`, available tags are listed in `models/published_models.csv`.

For full reproducibility, also the respective partition of the dataset and the configuration file used for training is included in the released checkpoints and can be found at `models/tag/config.yaml` and `models/tag/split.json` after downloading the respective model (see `examples/reproducibility`). For the case of `grappa-1.3`, this is equivalent to running


## Datasets

Datasets of dgl graphs representing molecules can be obtained by using the `grappa.data.Dataset.from_tag` constructor.
An example can be found at `examples/usage/dataset.py`, available tags are listed in `data/published_datasets.csv`.

To re-create the benchmark experiment, also the splitting into train/val/test sets from Espaloma is needed. This can be done by running `dataset_creation/get_espaloma_split/save_split.py`, which will create a file `espaloma_split.json` that contains lists of smilestrings for each of the sub-datasets. These are used to classify molecules as being train/val/test molecules upon loading the dataset in the train scripts from `experiments/benchmark`.

The datasets 'dipeptides-300K-...', 'dipeptides-1000K-...', 'uncapped_...', 'hyp-dop_...' and 'dipeptides_radical-300K' were generated using scripts at [grappa-data-creation](https://github.com/LeifSeute/grappa-data-creation).

For the creation of custom datasets, take a look at the tutorials `examples/dataset_creation/create_dataset.py` and `examples/dataset_creation/uncommon_molecule_dataset.py`.

## Training

Grappa models can be trained with a given configuration specified using hydra by running

```{bash}
python experiments/train.py
```

With hydra, configuration files can be defined in a modular way. For Grappa, we have configuration types `model`, `data` and `experiment`, for each of which default values can be overwritten in the command line or in a separate configuration file. For example, to train a model with less node features, one can run
```{bash}
python experiments/train.py model.graph_node_features=32
```

and for training on the datasets of grappa-1.3 (defined in `configs/data/grappa-1.3`), one can run
```{bash}
python experiments/train.py data=grappa-1.3 model=default experiment=default
```

For starting training with pretrained model weights, call e.g.
```{bash}
python experiments/train.py experiment.ckpt_path=models/grappa-1.3.0/checkpoint.ckpt
```

Training is logged in [wandb](https://docs.wandb.ai/quickstart) and can be safely interrupted by pressing `ctrl+c` at any time. Checkpoints with the best validation loss will be saved in the `ckpt/<project>/<name>/<data>` directory.

For evaluation, modify `configs/evaluate.yaml` and run  
```{bash}
python experiments/evaluate.py ckpt_path=<path_to_checkpoint>
```