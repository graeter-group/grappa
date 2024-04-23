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
- [Datasets](#datasets)
- [Pretrained Models](#pretrained-models)
- [Reproducibility](#reproducibility)
</details>


## Usage

Grapa predicts bonded parameters, the nonbonded parameters like partial charges are predicted with a traditional force field. The input to Grappa is therefore a representation of the system of interest that already contains information on the nonbonded parameters. Currently, Grappa is compatible with GROMACS and OpenMM.

For complete example scripts, see `examples/usage`.

### GROMACS

In GROMACS, Grappa can be used as command line application that receives the path to a topology file and writes the bonded parameters in there.

```{bash}
# parametrize the system with a traditional forcefield:
gmx pdb2gmx -f your_protein.pdb -o your_protein.gro -p topology.top -ignh

# create a new topology file with the bonded parameters from Grappa, specifying the tag of the grappa model:
grappa_gmx -f topology.top -o topology_grappa.top -t grappa-1.2

# (you can also create a plot of the parameters for inspection using the -p flag)

# continue with ususal gromacs workflow
```

### OpenMM

To use Grappa in OpenMM, parametrize your system with a traditional forcefield, from which the nonbonded parameters are taken, and then pass it to Grappas Openmm wrapper class:

```{python}
from openmm.app import ForceField, Topology
from grappa import OpenmmGrappa

topology = ... # load your system as openmm.Topology

classical_ff = ForceField('amber99sbildn.xml', 'tip3p.xml')
system = classical_ff.createSystem(topology)

# load the pretrained ML model from a tag. Currently, possible tags are 'grappa-1.1', grappa-1.2' and 'latest'
grappa_ff = OpenmmGrappa.from_tag('grappa-1.2')

# parametrize the system using grappa.
# The charge_model tag tells grappa how the charges were obtained, in this case from the classical forcefield amberff99sbildn. possible tags are 'amber99' and 'am1BCC'.
system = grappa_ff.parametrize_system(system, topology, charge_model='amber99')
```


## Installation


### GROMACS

The creation of custom GROMACS topology files is handled by [Kimmdy](https://github.com/hits-mbm-dev/kimmdy), which can be installed via pip.

For simplicity, we recommend to use Grappa for GROMACS in cpu mode since the inference runtime of Grappa is usually small compared to the simulation runtime, even without a GPU. (Simply create another environment if you also intend to train Grappa.)

```{bash}
git clone git@github.com:hits-mbm-dev/kimmdy.git && cd kimmdy && pip install -e . && cd ..
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
pip install dgl -f https://data.dgl.ai/wheels/repo.html dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

To install Grappa, simply clone the repository, install additional requirements and the package itself with pip:

```{bash}
git clone https://github.com/hits-mbm-dev/grappa.git
cd grappa
conda create -n grappa -y
conda activate grappa
pip install -r requirements.txt
pip install .
```

Verify the installation by running
```
python tests/test_installation.py
```

### OpenMM

Unfortunately, OpenMM is not available on pip and has to be installed via conda. Since OpenMM, torch and dgl use cuda, the choice of package-versions is not trivial and is thus handled by installscripts. The installation scripts are tested on Ubuntu 22.04 and cuda 12.1 and install the following versions:

| CUDA | Python | Torch | OpenMM  |
|------|--------|-------|---------|
| 11.7 | 3.9    | 2.0.1 | 7.7.0   |
| 11.8 | 3.10   | 2.2.0 | 8.1.1   |
| 12.1 | 3.10   | 2.2.0 | 8.1.1   |
| cpu  | 3.10   | 2.2.0 | 8.1.1   |

Simply activate the target conda environment and run the install script for the cuda version of choice, for pure inference usualluy the cpu version is sufficient.

```{bash}
git clone https://github.com/hits-mbm-dev/grappa.git
cd grappa
conda create -n grappa -y
conda activate grappa
bash installation_openmm.sh cpu
# bash installation_openmm.sh 11.8 # with cuda
```

Verify the installation by running
```
python tests/test_installation.py
```

### Development

To facilitate the interface to OpenMM and GROMACS, Grappa has an optional dependency on [OpenMM](https://github.com/openmm/openmm) and [Kimmdy](https://github.com/hits-mbm-dev/kimmdy), which is used to create custom GROMACS topology files. To train and evaluate Grappa on existing datasets, neither of these packages are needed.

In this case, Grappa only needs a working installation of [PyTorch](https://pytorch.org/) and [DGL](https://www.dgl.ai/) for the cuda version of choice, e.g. with conda by

```{bash}
conda install pytorch=2.1.0 pytorch-cuda=11.8 dgl -c pytorch -c nvidia -c dglteam/label/cu118
```

or with pip,

```{bash}
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html dglgo -f https://data.dgl.ai/wheels-test/repo.html
conda install cuda-toolkit -c "nvidia/label/cuda-11.7.1" -y
```

To then install Grappa, simply clone the repository, install additional requirements and the package itself:

```{bash}
git clone git@github.com:hits-mbm-dev/grappa.git
cd grappa
pip install -r requirements.txt
pip install .
```

Alternatively, install grappa with openmm by running the installation script provided. This will, however, take a bit longer since it will install openmm via conda.

Verify the installation by running
```
python tests/test_installation.py
```


## Pretrained Models

Pretrained models can be obtained by using `grappa.utils.run_utils.model_from_tag` with a tag (e.g. `latest`) that will point to a url that points to a version-dependent release file, from which model weights are downloaded. An example can be found at `examples/usage/openmm_wrapper.py`. For full reproducibility, one can also obtain the model weights toghether with the respective partition of the dataset and the configuration file used for training by `grappa.utils.run_utils.model_dict_from_tag`, which returns a dictionary with the keys `{'state_dict', 'config', 'split_names', 'results','ds_size'}`.


## Datasets

Datasets of dgl graphs representing molecules can be obtained by using the `grappa.data.Dataset.from_tag` constructor. An example can be found at `examples/usage/dataset.py`. Available tags are listed in the documentation of the Dataset class.

To re-create the benchmark experiment, also the splitting into train/val/test sets is needed. This can be done by running `dataset_creation/get_espaloma_split/save_split.py`, which will create a file `espaloma_split.json` that contains lists of smilestrings for each of the sub-datasets. These are used to classify molecules as being train/val/test molecules upon loading the dataset in the train scripts from `experiments/benchmark`.

The datasets 'uncapped_amber99sbildn', 'tripeptides_amber99sbildn', 'hyp-dop_amber99sbildn' and 'dipeptide_rad' from [grappa-1.1](https://github.com/hits-mbm-dev/grappa/releases/tag/v.1.1.0) were generated using scripts at [grappa-data-creation](https://github.com/LeifSeute/grappa-data-creation).

## Reproducibility

Every Grappa model that is published does not only contain the model weights but also a config dict that describes hyperparameters of the model and training and the split of the dataset into train/val/test molecules. This allows seamless reproducibility in a few lines of code. For example, to reproduce the training of Grappa 1.1, one can simply run:

```{python}
from grappa.utils.run_utils import model_dict_from_tag
from grappa.utils.loading_utils import model_dict_from_tag
from grappa.training.trainrun import do_trainrun
import json

model_dict = model_dict_from_tag('grappa-1.2') # change tag to grappa-1.1-benchmark to reproduce the benchmark table

split_ids = model_dict['split_names']
with open('split_ids.json', 'w') as f:
    json.dump(split_ids, f)

config['data_config']['splitpath'] = splitpath

do_trainrun(config=config, project='reproduce-grappa')
```
