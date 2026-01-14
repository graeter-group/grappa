# Config argument notes (configs/train.yaml + configs/*/default.yaml)

```yaml
data:
  data_module:
    datasets: []  # List of dataset tags/paths that are split by mol_id.
    pure_test_datasets: []  # Appended to test set without splitting.
    pure_val_datasets: []  #
    pure_train_datasets: []  #
    splitpath: espaloma_split  # Split file path or tag; see load_splitfile.
    partition: [0.8, 0.1, 0.1]  # Used only if splitpath is not provided.
    train_batch_size: 32  # Molecules per batch.
    val_batch_size: 32  # Molecules per batch.
    ref_terms:
      - nonbonded  # Reference FF terms subtracted from QM energies/gradients.
    train_loader_workers: 4  #
    val_loader_workers: 4  #
    test_loader_workers: 4  #
    conf_strategy: 32  # Confs per molecule per batch (int or "min"/"max"/"mean"/"all").
    ff_lookup: {}  # Optional dataset -> force field mapping for ref_terms lookup.
    seed: 0  # Used for splits and subsampling.
    pin_memory: true  #
    tr_subsampling_factor: null  # Fraction of molecules kept for training.
    tr_max_confs: null  # Cap conformations per molecule for training.
    weights:
      rna-diverse: 3  # Sampling multipliers per subdataset (WeightedRandomSampler).
      rna-nucleoside: 5  # Use to rebalance large vs small subsets or fine-tune.
      spice-pubchem: 0.8
    balance_factor: 0.0  # 0 = no balancing, 1 = equalize dataset sampling rates.
    val_conf_strategy: 200  # Max confs per molecule for val/test.
    max_energy: null  # Drop conformations above this energy delta.
    # max_force: null  # Drop conformations above this force norm.

  energy:
    terms:
      - bond
      - angle
      - proper
      - improper  # Predicted by Grappa; must not overlap with ref_terms.
    gradients: true  # Enables gradient computation in Energy module.

experiment:
  ckpt_path: null  # Checkpoint path or tag to warm-start training.
  ckpt_cfg_override: false  # If true, replace current model config with ckpt model config.
  use_wandb: True  # Toggle WandB logging.
  wandb:
    name: baseline  #
    project: grappa  #
    save_code: True  #
    tags: []  #
  checkpointer:
    dirpath: ckpt/${experiment.wandb.project}/${experiment.wandb.name}/${now:%Y-%m-%d}_${now:%H-%M-%S-%f}  # Run directory (ckpts, config.yaml, split.json).
    monitor: early_stopping_loss  # Logged in GrappaLightningModel.on_validation_epoch_end.
    mode: min  #
    save_top_k: 1  #
    save_last: True  #
    every_n_epochs: 5  #
    auto_insert_metric_name: False  #
    filename: 'epoch:{epoch}-early_stop_loss:{early_stopping_loss:.2f}'  # Requires early_stopping_loss.
  progress_bar: true  #
  trainer:
    overfit_batches: 0  #
    min_epochs: 5  #
    max_epochs: 1000  #
    accelerator: gpu  #
    log_every_n_steps: 10  #
    deterministic: False  #
    check_val_every_n_epoch: 1  #
  evaluation:
    n_bootstrap: 1  # Passed to Experiment.test after training.
  train:
    lr: 1e-05  # Base LR for Adam.
    energy_weight: 1.0  # Loss weight for energy RMSE.
    gradient_weight: 0.8  # Loss weight for gradient RMSE.
    start_qm_epochs: 1  # Start QM energy/gradient training after this epoch.
    param_loss_epochs: 10  # After this epoch, parameter loss weights are set to zero.
    warmup_steps: 500  # Linear warmup steps for LR.
    early_stopping_energy_weight: 3.0  # Energy RMSE weight in early_stopping_loss.
    param_weight: 0.001  # Parameter loss weight before param_loss_epochs.
    proper_regularisation: 1.  # L2 penalty on proper torsion parameters.
    improper_regularisation: 1.  # L2 penalty on improper torsion parameters.
    weight_decay: 0  #
    patience: -1  # If >0, decays LR after this many epochs without improvement.
    lr_decay: 1.  # LR multiplier when patience is exceeded.
    tuplewise_weight: 0.  # Tuplewise loss is not implemented (must stay 0).
    time_limit: 47.5  # Max training time in hours before early stop.
    reset_optimizer_on_load: False  # If true, reinitialize optimizer when loading a ckpt.
    add_noise_on_load: 0.  # Stddev of parameter noise when loading a ckpt.

model:
  graph_node_features: 256  # Output size of GrappaGNN and input size of parameter writers.
  in_feats: null  # If set, overrides inferred input feature dimension.
  in_feat_name:
    - atomic_number
    - ring_encoding
    - partial_charge
    - degree  # Node feature names read from the DGL graph.
  in_feat_dims: {}  # Explicit per-feature dims if not inferable.
  gnn_width: 512  #
  gnn_attentional_layers: 4  #
  gnn_convolutions: 0  # Extra convolution layers after attention blocks.
  gnn_attention_heads: 16  #
  gnn_dropout_attention: 0.3  #
  gnn_dropout_initial: 0.0  #
  gnn_dropout_conv: 0.0  #
  gnn_dropout_final: 0.1  #
  symmetric_transformer_dropout: 0.5  # Shared dropout for parameter writers.
  symmetric_transformer_depth: 1  #
  symmetric_transformer_n_heads: 8  #
  symmetric_transformer_width: 512  #
  symmetriser_depth: 4  #
  symmetriser_width: 256  #
  n_periodicity_proper: 3  # Fourier terms per proper torsion.
  n_periodicity_improper: 2  # Fourier terms per improper torsion.
  gated_torsion: true  # Enables gating in torsion writers.
  positional_encoding: true  # Adds positional encodings in parameter writers.
  layer_norm: true  #
  self_interaction: true  # Enable self-interaction in the GNN.
  learnable_statistics: false  # Make parameter mean/std learnable.
  torsion_cutoff: 1.e-4  # Cutoff for small torsion force constants.
  harmonic_gate: false  # Gate bond/angle force constants (scaled sigmoid).
  only_n2_improper: true  # Restrict improper torsions to n2 interactions.
  stat_scaling: true  # Scale outputs using parameter statistics.
  shifted_elu: true  # Use shifted ELU for positive outputs in writers.
```

# Evaluation config (configs/evaluate.yaml + configs/evaluate/default.yaml)

```yaml
evaluate:
  ckpt_path: grappa-1.4.0  # Tag or path to checkpoint (.ckpt).
  n_bootstrap: 1  # Number of bootstrap samples for metrics.
  store_test_data: false  # Save evaluated data to .npz.
  test_data_path: null  # If set, path to write test data .npz.
  accelerator: 'gpu'  # Overridden to 'cpu' if CUDA is unavailable.
  plot: false  # Generate plots via Evaluator.
  classical_force_fields: []  # Evaluate classical FFs alongside Grappa.
  datasets: []  # Override datasets from checkpoint config.
  pure_test_datasets: null  # Override test-only datasets.
  ff_lookup: {}  # Overrides data_module.ff_lookup.
  splitpath: null  # Split file path or tag (overrides checkpoint split).
  gradient_contributions: []  # Energy terms to compute gradient contributions for.
  compare_forcefields: []  # Compare multiple FFs in eval_classical.
  eval_model: true  # Run model evaluation; if false, only classical comparisons.
  grappa_contributions: null  # Restrict Grappa to these terms; others become ref_terms.
```


Further details:

Split file format (`split.json`):
- JSON with keys `train`, `val`, `test`.
- Values are lists of `mol_id` strings (from `MolData.mol_id`).
- `splitpath` accepts a path or a tag; if the path doesn't exist, it is treated as a tag
  and resolved to `data/datasets/<tag>/split.json`.

Batching:
- `train_batch_size` / `val_batch_size` are counts of molecules (graphs).
- Conformations per molecule are controlled by `conf_strategy` (train) and
  `val_conf_strategy` (val/test).