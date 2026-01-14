# Code Overview

Grappa is a machine-learned bonded force-field parameterizer. It predicts bonded
terms (bonds, angles, torsions) from molecular graphs, while nonbonded terms are
kept from a classical force field. The repo contains the training stack, data
pipeline, model components, and deployment wrappers.

## Training and evaluation stack

The training flow is centered on the `Experiment` class in
`src/grappa/training/experiment.py`, which wires together data, model, and the
Lightning trainer.

- `Experiment` builds the data module (`GrappaData`), constructs the model stack
  (`GrappaModel` + `Energy`), and creates the `GrappaLightningModel` wrapper.
- `GrappaLightningModel` in `src/grappa/training/lightning_model.py` owns the
  training schedule, warmup, staged loss weights, early stopping metric, and
  evaluation hooks.
- Losses come from `MolwiseLoss` in `src/grappa/training/loss.py`, which combines
  energy/gradient matching with parameter regularization and optional per-dataset
  weighting.
- Metrics are computed by `FastEvaluator` and `Evaluator` in
  `src/grappa/training/evaluator.py`, which aggregate RMSE values and per-term
  contributions during validation and testing.

## Data and graph stack

Data moves from chemistry objects to DGL graphs, then to batched loaders.

- `Molecule` in `src/grappa/data/molecule.py` defines atoms, bonds, angles,
  torsions, and atom-level features. Its `to_dgl()` method constructs a DGL
  heterograph with node types `g`, `n1`, `n2`, `n3`, `n4`, `n4_improper`.
- `MolData` in `src/grappa/data/mol_data.py` extends `Molecule` with conformations,
  QM energies, gradients, and classical FF contributions. Its `to_dgl()` writes
  `xyz`, `energy_*`, and `gradient_*` into graph features.
- `Dataset` in `src/grappa/data/dataset.py` stores graphs + `mol_id` and handles
  mol-wise train/val/test splits, subsampling, and conformation caps.
- `GrappaData` in `src/grappa/data/grappa_data.py` is the LightningDataModule
  that loads datasets by tag, applies split files, computes reference energies
  (`create_reference`), and provides loaders.
- `GraphDataLoader` in `src/grappa/data/graph_data_loader.py` batches graphs and
  normalizes the number of conformations per batch via `conf_strategy`.

## Model and parameter stack

The core model predicts parameters, then evaluates them through a differentiable
MM energy layer.

- `GrappaModel` in `src/grappa/models/grappa.py` is the parameter predictor.
  It combines `GrappaGNN` (`src/grappa/models/graph_attention.py`) with
  `WriteParameters` (`src/grappa/models/interaction_parameters.py`), which uses
  symmetry-aware transformers to output bonded parameters.
- `Energy` in `src/grappa/models/energy.py` computes bonded energies and
  gradients from predicted parameters and writes them back into the graph.
  This enables training directly on QM energies/forces.

## Inference and wrappers

Deployment uses a thin wrapper around the trained model.

- `Grappa` in `src/grappa/grappa.py` loads a model from a tag or checkpoint and
  predicts parameters for a `Molecule`.
- `OpenmmGrappa` and `as_openmm` in `src/grappa/wrappers/openmm_wrapper.py`
  insert predicted bonded terms into an OpenMM `System`.
- `GromacsGrappa` in `src/grappa/wrappers/gromacs_wrapper.py` writes a new
  GROMACS `.top` with Grappa parameters via gmxtop.

## Registries and tag resolution

Published model and dataset tags are tracked in CSV registries and resolved at
runtime.

- `models/published_models.csv` lists released model tags.
- `data/published_datasets.csv` lists dataset tags and download URLs.
- Tag resolution, downloads, and split-file lookup live in
  `src/grappa/utils/data_utils.py`.

## Entry points

- Training: `experiments/train.py`
- Evaluation: `experiments/evaluate.py`
- Usage examples: `examples/usage/`
