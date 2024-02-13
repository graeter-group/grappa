# Graph Attentional Protein Parametrization (GrAPPa)

_A machine-learned molecular mechanics force field using deep graph attention networks_


## Abstract

Simulating large molecular systems over long timescales requires force fields that are both accurate and efficient.
While E(3) equivariant neural networks are providing a speedup over computational Quantum Mechanics (QM) at high accuracy, they are several orders of magnitude slower than Molecular Mechanics (MM) force fields.

Here, we present a state of the art machine-learned MM force field that outperforms traditional and other machine-learned MM forcefields [Wang et al. ()] significantly in terms of accuracy, at the same computational cost.
Our forcefield, Grappa, covers a broad range of chemical space: The same forcefield can parametrize small molecules, proteins, RNA and even uncommon molecules like radical peptides.
Besides predicting energies and forces at greatly improved accuracy, Grappa is transferable to large molecules. We show that it keeps Ubiquitin stable and can fold small proteins in molecular dynamics simulations.

Grappa uses a deep graph attention network and a transformer with symmetry-preserving positional encoding to predict MM paramaters from molecular graphs. The current model is trained on QM energies and forces of over 14,000 molecules and over 800,000 states, and is available for use with GROMACS and OpenMM.


<p align="center">
  <img src="docs/grappa_overview.png" width="50%" style="max-width: 200px;">
    <i>Grappa Overview</i>
</p>

<p align="center">
  <img src="docs/table.png" width="50%" style="max-width: 200px;">
</p>

<details open><summary><b>Table of contents</b></summary>
- [Usage](#usage)
- [Installation](#installation)
- [Results](#results)
  - [Grappa is state-of-the-art](#grappa-is-state-of-the-art)
  - [Grappa keeps large proteins stable](#grappa-keeps-ubiquitin-stable)
  - [Grappa can fold small proteins](#grappa-can-fold-small-proteins)
  - [Grappa can parametrize radicals](#grappa-can-parametrize-radicals)
- [Method](#method)
  - [Framework](#framework)
  - [Permutation Symmetry](#permutation-symmetry)
  - [Architecture](#architecture)
- [Training](#training)
- [Datasets](#datasets)
- [Pretrained Models](#pretrained-models)
</details>


## Usage

Currently, Grappa is compatible with GROMACS and OpenMM. To use Grappa in openmm, parametrize your system with a classical forcefield, from which the nnbonded parameters are taken, and then pass it to Grappas Openmm wrapper class:

```{python}
from openmm.app import ForceField, Topology
from grappa.utils.loading_utils import model_from_tag
from grappa.wrappers.openmm_wrapper import openmm_Grappa

model = model_from_tag('grappa-1.0')

topology = ... # load your system as openmm.Topology

classical_ff = ForceField('amber99sbildn.xml', 'tip3p.xml')
system = classical_ff.createSystem(topology)

grappa_ff = openmm_Grappa(model)

system = grappa.parametrize_system(system, topology)
```

Note that the current version of the OpenMM wrapper will parametrize the whole topology with Grappa, including the solvent. Grappa is not trained to parametrize water, the solvent should thus be removed from the topology before parametrization. In future versions, there will be the option to parametrize only a subset of the topology.

More: See `examples/usage`.

## Installation

Unfortunately, openmm is not available on pip and has to be installed via conda. Since openmm, torch and dgl use cuda, the choice of package-versions is not trivial and is thus handled by installscripts. The installation scripts are tested on Ubuntu 22.04 and install the following versions:

| CUDA | Python | Torch | OpenMM |
|------|--------|-------|---------|
| 11.7 | 3.9    | 2.0.1 | 7.7.0   |
| 11.8 | 3.10   | 2.2.0 | 8.1.1   |
| 12.1 | 3.10   | 2.2.0 | 8.1.1   |
| cpu  | 3.10   | 2.2.0 | 8.1.1   |

Simply activate the target conda environment and run the install script for the cuda version of choice, e.g. for 12.1:
```{bash}
conda create -n grappa -y
conda activate grappa
./installation.sh 12.1
```

## Results

### Grappa is state-of-the-art
table




## Pretrained Models

Pretrained models can be obtained by using `grappa.utils.run_utils.model_from_tag` with a tag (e.g.`latest`) that will point to a url that points to a version-dependent release file, from which model weights are downloaded. An example can be found at `examples/usage/openmm_wrapper.py`.



## Datasets

Datasets of dgl graphs representing molecules can be obtained by using the `grappa.data.Dataset.from_tag` constructor. An example can be found at `examples/usage/evaluation.py`. Available tags are listed in the documentation of the Dataset class.

To re-create the benchmark experiment, also the splitting into train/val/test sets is needed. This can be done by running `dataset_creation/get_espaloma_split/save_split.py` has to be run. This will create a file `espaloma_split.json` that contains lists of smilestrings for each of the sub-datasets. These are used to classify molecules as being train/val/test molecules upon loading the dataset in the train scripts from `experiments/benchmark`.
