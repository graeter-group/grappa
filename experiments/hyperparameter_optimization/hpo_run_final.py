from grappa.utils.dataset_utils import get_data_path
from grappa.training.trainrun import do_trainrun
import yaml
import math
from collections import defaultdict
import wandb
import argparse

MANUAL = False

NAME = None
PROJECT = 'hpo_grappa_final'


def config_from_sweep(sweep_config):
    config = defaultdict(dict)

    config['lit_model_config']['lr'] = 10**sweep_config.lr

    config['data_config']['train_batch_size'] = sweep_config.batch_size
    
    config['model_config']['graph_node_features'] = 2**sweep_config.atom_typing_features
    config['model_config']['gnn_width'] = 2**sweep_config.gnn_width

    config['model_config']['gnn_attentional_layers'] = sweep_config.attentional_layers

    config['model_config']['gnn_convolutions'] = sweep_config.convolutions

    config['model_config']['gnn_attention_heads'] = sweep_config.gnn_attention_heads

    parameter_width = 2**sweep_config.parameter_width

    config['model_config']['bond_symmetriser_width'] = parameter_width
    config['model_config']['angle_symmetriser_width'] = parameter_width
    config['model_config']['proper_symmetriser_width'] = parameter_width
    config['model_config']['improper_symmetriser_width'] = parameter_width

    config['model_config']['bond_transformer_width'] = sweep_config.parameter_trafo_factor*parameter_width
    config['model_config']['angle_transformer_width'] = sweep_config.parameter_trafo_factor*parameter_width
    config['model_config']['proper_transformer_width'] = sweep_config.parameter_trafo_factor*parameter_width
    config['model_config']['improper_transformer_width'] = sweep_config.parameter_trafo_factor*parameter_width

    config['model_config']['bond_symmetriser_depth'] = sweep_config.symmetriser_depth
    config['model_config']['angle_symmetriser_depth'] = sweep_config.symmetriser_depth
    config['model_config']['proper_symmetriser_depth'] = sweep_config.symmetriser_depth
    config['model_config']['improper_symmetriser_depth'] = sweep_config.symmetriser_depth

    config['model_config']['bond_transformer_depth'] = sweep_config.transformer_depth
    config['model_config']['angle_transformer_depth'] = sweep_config.transformer_depth
    config['model_config']['proper_transformer_depth'] = sweep_config.transformer_depth
    config['model_config']['improper_transformer_depth'] = sweep_config.transformer_depth

    config['model_config']['bond_n_heads'] = sweep_config.interaction_heads
    config['model_config']['angle_n_heads'] = sweep_config.interaction_heads
    config['model_config']['proper_n_heads'] = sweep_config.interaction_heads
    config['model_config']['improper_n_heads'] = sweep_config.interaction_heads

    config['model_config']['gnn_dropout_conv'] = sweep_config.gnn_dropout
    config['model_config']['gnn_dropout_final'] = sweep_config.gnn_dropout
    config['model_config']['gnn_dropout_attention'] = sweep_config.gnn_dropout
    config['model_config']['parameter_dropout'] = sweep_config.param_dropout

    config['lit_model_config']['gradient_weight'] = 10**sweep_config.gradient_weight

    config['lit_model_config']['weight_decay'] = sweep_config.weight_decay

    config['data_config']['balance_factor'] = sweep_config.balance_factor
    
    return config


model_config = {
    "graph_node_features": 256,
    "in_feats": None,
    "in_feat_name": ["atomic_number", "partial_charge", "ring_encoding"],
    "in_feat_dims": {},
    "gnn_width": 128,
    "gnn_attentional_layers": 2,
    "gnn_convolutions": 2,
    "gnn_attention_heads": 8,
    "gnn_dropout_attention": 0.,
    "gnn_dropout_initial": 0.,
    "gnn_dropout_conv": 0.,
    "gnn_dropout_final": 0.,
    "parameter_dropout": 0.,
    "bond_transformer_depth": 2,
    "bond_n_heads": 8,
    "bond_transformer_width": 512,
    "bond_symmetriser_depth": 2,
    "bond_symmetriser_width": 256,
    "angle_transformer_depth": 2,
    "angle_n_heads": 8,
    "angle_transformer_width": 512,
    "angle_symmetriser_depth": 2,
    "angle_symmetriser_width": 256,
    "proper_transformer_depth": 2,
    "proper_n_heads": 8,
    "proper_transformer_width": 512,
    "proper_symmetriser_depth": 2,
    "proper_symmetriser_width": 256,
    "improper_transformer_depth": 2,
    "improper_n_heads": 8,
    "improper_transformer_width": 512,
    "improper_symmetriser_depth": 2,
    "improper_symmetriser_width": 256,
    "n_periodicity_proper": 6,
    "n_periodicity_improper": 3,
    "gated_torsion": True,
    "wrong_symmetry": False,
    "positional_encoding": True,
    "layer_norm": True,
    "self_interaction": True,
    "learnable_statistics": False,
}

data_config = {
    "datasets": [
        str(get_data_path()/"dgl_datasets"/dsname) for dsname in
        [
            "spice-dipeptide",
            "spice-des-monomers",
            "spice-pubchem",
            "gen2",
            "gen2-torsion",
            "pepconf-dlc",
            "protein-torsion",
            "rna-diverse",
        ]
    ],
    "conf_strategy": 100,
    "train_batch_size": 32,
    "val_batch_size": 64,
    "test_batch_size": 1,
    "train_loader_workers": 2,
    "val_loader_workers": 2,
    "test_loader_workers": 1,
    "pin_memory": True,
    "splitpath": None,
    "partition": [ # may either be a partition list or a list of a default partition list and a dictionary mapping dsname to partition
        [0.7,0.3,0.0],
        ],
    "pure_train_datasets": ['rna-nucleoside'],
    "pure_val_datasets": [],
    "pure_test_datasets": [str(get_data_path()/"dgl_datasets"/'rna-trinucleotide')], # this can be used to be independent of splitting. in the case of the espaloma benchmark, this is used to have the same molecules in the test and train set (where training is on rna-diverse-conformations and testing on rna-trinucleotide-conformations)
    "weights": {
        'gen2': 0.5, # empirical, makes training a bit more stable, reduces overfitting
        'gen2-torsion': 0.5, # empirical, makes training a bit more stable, reduces overfitting
        'pepconf': 4, # empirical, harder to fit apparently
        'spice-pubchem': 0.8,
        'rna-diverse': 2,
    },
    "balance_factor": 0.3,
}

lit_model_config = {
    "lr": 1e-5,
    "start_qm_epochs": 2,
    "add_restarts": [],
    "warmup_steps": int(2e2),
    "energy_weight": 1.,
    "gradient_weight": 0.5,
    "tuplewise_weight": 1e-5,
    "param_weight": 1e-4,
    "proper_regularisation": 1e-5,
    "improper_regularisation": 1e-5,
    "log_train_interval": 5,
    "log_classical": False,
    "log_params": False,
    "weight_decay": 0.,
    "early_stopping_energy_weight": 3., # weight of the energy rmse in the early stopping criterion
    "log_metrics":True,
    "patience": 30,
    "lr_decay": 0.8,
    "time_limit": 30,
    "finish_criterion": {1:40, 2:30, 4:25, 10:20, 15:16, 24:15}, # {hours: max_early_stopping_val_loss} finishes sweep runs that are not promising
    "param_loss_epochs": 100,
}

trainer_config = {
    "max_epochs": 10000, # time limit applies anyways
    "gradient_clip_val": 1e1,
    "profiler": "simple",
    'early_stopping_criterion': 'early_stopping_loss',
    'name': NAME,
    'notes': None,
}

config = {
    "model_config": model_config,
    "data_config": data_config,
    "lit_model_config": lit_model_config,
    "trainer_config": trainer_config,
    'test_model': False,
}

def default_sweep_config():
    wandb.config.update({
        "lr": -5,
        "batch_size": 16,
        "atom_typing_features": 8,
        "gnn_width": 7,
        "attentional_layers": 3,
        "convolutions": 2,
        "gnn_attention_heads": 8,
        "param_dropout": 0.6,
        "gnn_dropout": 0.2,
        "parameter_width": 7,
        "parameter_trafo_factor": 2,
        "symmetriser_depth": 2,
        "transformer_depth": 3,
        "interaction_heads": 8,
        "gradient_weight": -0.5,
        "weight_decay": 1e-4,
        "balance_factor": 0.2,
    })


if MANUAL:
    do_trainrun(config=config, project=PROJECT, config_from_sweep=config_from_sweep, manual_sweep_config=default_sweep_config)
else:
    do_trainrun(config=config, project=PROJECT, config_from_sweep=config_from_sweep)

