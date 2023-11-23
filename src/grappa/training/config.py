from typing import Dict
from pathlib import Path
import yaml
from grappa.utils.run_utils import load_yaml, write_yaml, get_data_path


def overwrite_config(kwargs, config):
    """
    Overwrite the configuration with the kwargs.
    """
    for k,v in kwargs.items():
        if k in config:
            config[k] = v
        else:
            raise KeyError(f'Key {k} not in config.')
    
    return config


def write_default_config(path:Path=Path.cwd(), model_tag='med'):
    """
    Write the default configuration to a file.
    """
    config = default_config(model_tag=model_tag)
    write_yaml(config, path)


def default_config(model_tag:str='small')->Dict:
    """
    Returns the default configuration.
    """
    from grappa.models.deploy import get_default_model_config

    model_config = get_default_model_config(tag=model_tag)

    model_config['in_feat_name'] = ['atomic_number', 'partial_charge', 'ring_encoding']

    data_config = {
        "datasets": [
            str(get_data_path()/"dgl_datasets"/dsname) for dsname in
            [
                "spice-des-monomers",
                "spice-dipeptide",
                "spice-pubchem", # NOTE: ALSO ADD RNA
                "gen2",
                "gen2-torsion",
                "pepconf-dlc",
                "protein-torsion",
            ] 
        ],
        "conf_strategy": "mean",
        "train_batch_size": 20,
        "val_batch_size": 1,
        "test_batch_size": 1,
        "train_loader_workers": 1,
        "val_loader_workers": 2,
        "test_loader_workers": 2,
        "pin_memory": True,
        "splitpath": None,
        "partition": [0.8,0.1,0.1]
    }

    lit_model_config = {
        "lrs": {0: 1e-4, 3: 1e-5, 200: 1e-6, 400: 1e-7},
        "start_qm_epochs": 5,
        "add_restarts": [200, 400],
        "warmup_steps": int(1e3),
        "classical_epochs": 40,
        "energy_weight": 1e-5,
        "gradient_weight": 10.0,
        "tuplewise_weight": 0.,
        "log_train_interval": 5,
        "log_classical": False,
        "log_params": False,
    }

    trainer_config = {
        "max_epochs": 500,
        "gradient_clip_val": 1e1,
        "profiler": "simple"
    }

    config = {
        "model_config": model_config,
        "data_config": data_config,
        "lit_model_config": lit_model_config,
        "trainer_config": trainer_config
    }

    return config