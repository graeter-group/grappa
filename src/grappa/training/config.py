from typing import Dict
from pathlib import Path
import yaml
from grappa.utils.run_utils import write_yaml
from grappa.utils.dataset_utils import get_data_path, benchmark_data_config


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


def default_config(model_tag:str='small', benchmark:bool=False)->Dict:
    """
    Returns the default configuration.
    """
    from grappa.models.deploy import get_default_model_config

    model_config = get_default_model_config(tag=model_tag)

    data_config = {
        "datasets": [
            str(get_data_path()/"dgl_datasets"/dsname) for dsname in
            [
                "spice-des-monomers",
                "spice-dipeptide",
                "spice-pubchem",
                "gen2",
                "gen2-torsion",
                "pepconf-dlc",
                "protein-torsion",
                "rna-nucleoside",
                "rna-diverse",
                "rna-trinucleotide"
            ]
        ],
        "conf_strategy": "mean",
        "train_batch_size": 20,
        "val_batch_size": 1,
        "test_batch_size": 1,
        "train_loader_workers": 1,
        "val_loader_workers": 2,
        "test_loader_workers": 2,
        "seed": 0,
        "pin_memory": True,
        "splitpath": None,
        "partition": [0.8,0.1,0.1], # may either be a partition list or a list of a default partition list and a dictionary mapping dsname to partition
        "pure_train_datasets": [],
        "pure_val_datasets": [],
        "pure_test_datasets": [], # paths to datasets that are only for one specific set type, independent on which mol_ids occur. this can be used to be independent of the splitting by mol_ids. in the case of the espaloma benchmark, this is used to have the same molecules in the test and train set (where training is on rna-diverse-conformations and testing on rna-trinucleotide-conformations)
        "subsample_train": {}, # dictionary of dsname and a float between 0 and 1 specifying the subsampling factor (that is applied after splitting).
        "subsample_val": {},
        "subsample_test": {},
    }

    if benchmark:
        data_config = benchmark_data_config()

    lit_model_config = {
        "lrs": {0: 1e-4, 3: 1e-5, 200: 1e-6, 400: 1e-7},
        "start_qm_epochs": 1,
        "add_restarts": [200, 400],
        "warmup_steps": int(2e2),
        "energy_weight": 1.,
        "gradient_weight": 1e-1,
        "tuplewise_weight": 0.,
        "param_weight": 1e-4,
        "proper_regularisation": 1e-5,
        "improper_regularisation": 1e-5,
        "log_train_interval": 5,
        "log_classical": False,
        "log_params": False,
        "weight_decay": 0.,
        "early_stopping_energy_weight": 2., # weight of the energy rmse in the early stopping criterion
    }

    trainer_config = {
        "max_epochs": 500,
        "gradient_clip_val": 1e1,
        "profiler": "simple",
        'early_stopping_criterion': 'early_stopping_loss',
    }

    config = {
        "model_config": model_config,
        "data_config": data_config,
        "lit_model_config": lit_model_config,
        "trainer_config": trainer_config,
        'test_model': False,
    }

    return config