# Graph Attentional Protein Parametrization (GrAPPa)

_A machine-learned molecular mechanics force field using deep graph attention networks_


## Abstract

Simulating large molecular systems over long timescales requires force fields that are both accurate and efficient.
While E(3) equivariant neural networks are providing a speedup over computational Quantum Mechanics (QM) at high accuracy, they are several orders of magnitude slower than Molecular Mechanics (MM) force fields.

Here, we present a state of the art machine-learned MM force field that outperforms traditional and other machine-learned MM force fields [[Takaba et al. 2023](https://arxiv.org/abs/2307.07085v4)] significantly in terms of accuracy, at the same computational cost.
Our forcefield, Grappa, covers a broad range of chemical space: The same force field can parametrize small molecules, proteins, RNA and even uncommon molecules like radical peptides.
Besides predicting energies and forces at greatly improved accuracy, Grappa is transferable to large molecules. We show that it keeps Ubiquitin stable and can fold small proteins in molecular dynamics simulations.

Grappa uses a deep graph attention network and a transformer with symmetry-preserving positional encoding to predict MM parameters from molecular graphs. The current model is trained on QM energies and forces of over 14,000 molecules and over 800,000 states, and is available for use with GROMACS and OpenMM.

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
- [Results](#results)
  - [Grappa is state-of-the-art](#grappa-is-state-of-the-art)
  - [Grappa keeps large proteins stable](#grappa-keeps-large-proteins-stable)
  - [Grappa can fold small proteins](#grappa-can-fold-small-proteins)
  - [Grappa can parametrize radicals](#grappa-can-parametrize-radicals)
- [Method](#method)
  - [Framework](#framework)
  - [Permutation Symmetry](#permutation-symmetry)
  - [Architecture](#architecture)
- [Training](#training)
- [Datasets](#datasets)
- [Pretrained Models](#pretrained-models)
- 
</details>


## Usage

Currently, Grappa is compatible with GROMACS and OpenMM. To use Grappa in OpenMM, parametrize your system with a classical forcefield, from which the nonbonded parameters are taken, and then pass it to Grappas Openmm wrapper class:

```{python}
from openmm.app import ForceField, Topology
from grappa import OpenmmGrappa

topology = ... # load your system as openmm.Topology

classical_ff = ForceField('amber99sbildn.xml', 'tip3p.xml')
system = classical_ff.createSystem(topology)

# load the pretrained ML model from a tag. Currently, possible tags are 'grappa-1.0', grappa-1.1' and 'latest'
grappa_ff = OpenmmGrappa.from_tag('grappa-1.1')

# parametrize the system using grappa. The charge_model tag tells grappa how the charges were obtained, in this case from the classical forcefield amberff99sbildn. possible tags are 'classical' and 'am1BCC'.
system = grappa_ff.parametrize_system(system, topology, charge_model='classical')
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

In this section, we show that Grappa outperforms established MM force fields and the recent machine-learned Espaloma [<a href="https://arxiv.org/abs/2307.07085v4">Takaba et al. 2023</a>] force field in terms of accuracy and that it can be transfered to large molecules.

### Grappa is State-of-the-Art

We trained Grappa on the dataset (and train-validation-test partition) from Espaloma [<a href="https://arxiv.org/abs/2307.07085v4">Takaba et al. 2023</a>] and compared it with established MM force fields [<a href="https://pubs.aip.org/aip/jcp/article/153/11/114502/199591/A-fast-and-high-quality-charge-model-for-the-next">He et al.</a>], [<a href="https://doi.org/10.1021/acs.jctc.5b00255">Maier et al.</a>], [<a href="https://pubs.acs.org/doi/10.1021/ct200162x">Zgarbova et al.</a>].

The Espaloma dataset covers small molecules, peptides and RNA with states sampled from the Boltzmann distribution at 300K and 500K, from optimization trajectories and from torsion scans. For all types of molecules, Grappa outperforms established MM force fields and Espaloma in terms of force accuracy, and for Boltzmann-sampled states also in terms of energy accuracy. To the best of our knowledge, this makes it the most accurate MM force field currently available (as of February 2024).

<p align="center">
    <img src="docs/figures/table_benchmark.png" width="100%" style="max-width: 200px; display: block; margin: auto;">
  </p>
  <p><i>Energy and force-component RMSE on test molecules in kcal/mol and kcal/mol/Å. The dataset is split into 80% train, 10% validation and 10% test molecules, demonstrating not transferability not only in conformational but also in chemical space.</i></p>
  

### Grappa keeps large Proteins stable

Grappa can not only accurately predict QM energies and forces, but also reproduces well-known behaviour of established protein force fields. Ubiquitin [<a href="https://www.rcsb.org/structure/1UBQ">1UBQ</a>] shows a similar magnitude of fluctuation when simulated with Grappa and <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2970904/">Amberff99sbildn</a>.

<p align="center">
    <img src="docs/figures/rmsd.png" width="70%" style="max-width: 200px; display: block; margin: auto;">
  </p>
  <p><i>RMSD Distribution between states with a given time difference during 40 ns of MD simulation of Ubiquitin in solution at 300K. The shaded area corresponds to the range between the 25th and 75th percentile.</i></p>

### Grappa can fold small Proteins

We have simulated the small protein Chignolin in solution starting from an unfolded configuration and observed that it folds into the experimentally measured state [1UAO](https://www.rcsb.org/structure/1UAO) on a timescale of microseconds. We identified a cluster of folded states whose center has an C-alpha RMSD of 1.1 Å compared to 1.0 Å obtained in the same setting with <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2970904/">Amberff99sbildn</a> in [Lindorff-Larsen et al.](https://www-science-org.ubproxy.ub.uni-heidelberg.de/doi/epdf/10.1126/science.1208351).

<p align="center">
    <img src="docs/figures/structure_grappa.png" width="70%" style="max-width: 200px; display: block; margin: auto;">
  </p>
  <p><i>The cluster center of Chignolin during an MD simulation using Grappa (blue) and the experimentally measured structure.</i></p>


### Grappa 1.0

The published model grappa-1.0 has been trained on an extension of the Espaloma dataset that contains Boltzmann-sampled states of tripeptides and radical dipeptides that can be formed by hydrogen atom transfer. For the peptide datasets in Espaloma, we also calculate nonbonded contributions with <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2970904/">Amberff99sbildn</a> instead of am1BCC (as is done in Espaloma). We split the dataset into 80% train, 10% validation and 10% test molecules using the same partition as Espaloma.

<p align="center">
    <img src="docs/figures/table-grappa-10.png" width="100%" style="max-width: 200px; display: block; margin: auto;">
  </p>
  <p><i>Energy and force-component RMSE on test molecules in kcal/mol and kcal/mol/Å. Grappa can differentiate between the optimal bonded parameters for molecules whose nonbonded interaction is modeled with am1BCC-charges and amber99sbildn-charges.</i></p>
  

### Grappa can parametrize Radicals

Unlike many other machine-learned force fields, Grappa does not rely on hand-crafted input features from Cheminformatics-tools but only on the molecular graph and partial charges as input. This makes it applicable beyond the coverage of existing Cheminformatics-tools, for example to radicals.

Grappa 1.0 has been trained on radical peptides that can be formed by hydrogen atom transfer, i.e. that 'miss' a hydrogen (as opposed to being protonated). Grappa is the first MM force field capable of accurately simulating radical peptides. To demonstrate this, we simulate a small radical peptide that undergoes a hydrogen atom transfer in [KIMMDY](https://github.com/hits-mbm-dev/kimmdy), a GROMACS extension for reactive MD via kinetic Monte Carlo methods.

<p align="center">
    <img src="docs/figures/kimmdy-grappa.gif" width="100%" style="max-width: 200px; display: block; margin: auto;">
  </p>
  <p><i>Example simulation of a hydrogen atom transfer in a small radical peptide to demonstrate that the effect of the radical carbon on the geometry is captured with Grappa.</i></p>


## Method

### Architecture

<p align="center">
    <img src="docs/figures/gnn.png" width="50%" style="max-width: 200px; display: block; margin: auto;">
  </p>
  <p><i>The architecture of Grappas Graph Neural Network</i></p>

<p align="center">
    <img src="docs/figures/symmetric_transformer.png" width="70%" style="max-width: 200px; display: block; margin: auto;">
  </p>
  <p><i>The architecture of Grappas Symmetric Transformer</i></p>


### Permutation Symmetry


## Pretrained Models

Pretrained models can be obtained by using `grappa.utils.run_utils.model_from_tag` with a tag (e.g.`latest`) that will point to a url that points to a version-dependent release file, from which model weights are downloaded. An example can be found at `examples/usage/openmm_wrapper.py`.



## Datasets

Datasets of dgl graphs representing molecules can be obtained by using the `grappa.data.Dataset.from_tag` constructor. An example can be found at `examples/usage/evaluation.py`. Available tags are listed in the documentation of the Dataset class.

To re-create the benchmark experiment, also the splitting into train/val/test sets is needed. This can be done by running `dataset_creation/get_espaloma_split/save_split.py` has to be run. This will create a file `espaloma_split.json` that contains lists of smilestrings for each of the sub-datasets. These are used to classify molecules as being train/val/test molecules upon loading the dataset in the train scripts from `experiments/benchmark`.
