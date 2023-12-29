from grappa.training.trainrun import do_trainrun
from grappa.utils.dataset_utils import get_data_path
from pathlib import Path
import yaml

PROJECT = 'generalize_on_radicals'
PRETRAIN_PATH = '../hyperparameter_optimization/wandb/run-20231228_192748-80s5jy2a/files/checkpoints/best-model.ckpt'


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
            "rna-nucleoside",
            "rna-diverse",
            'AA_radical',
            'dipeptide_rad',
            'spice_dipeptide_amber99sbildn',
        ]
    ],
    "conf_strategy": 100,
    "splitpath": None,
    "partition": [ # may either be a partition list or a list of a default partition list and a dictionary mapping dsname to partition
        [0.8,0.1,0.1],
        {
            'rna-nucleoside': [1.,0.,0.],
        }
        ],
    "pure_train_datasets": [],
    "pure_val_datasets": [str(get_data_path()/"dgl_datasets"/'tripeptides_amber99sbildn')],
    "pure_test_datasets": [str(get_data_path()/"dgl_datasets"/'rna-trinucleotide')], # this can be used to be independent of splitting. in the case of the espaloma benchmark, this is used to have the same molecules in the test and train set (where training is on rna-diverse-conformations and testing on rna-trinucleotide-conformations)
    "weights": {
        'gen2': 0.5, # empirical, makes training more stable
        'pepconf': 2, # empirical, harder to fit apparently
        'AA_radical': 3,
        'dipeptide_rad': 3,
        'spice_dipeptide_amber99sbildn': 3,
    },
    "balance_factor": 0.3,
}

lit_model_config = {
    "start_qm_epochs": 0,
    "add_restarts": [0],
    "warmup_steps": int(1e3),
    "time_limit": 10,
    "finish_criterion": {},
    "param_loss_epochs": 0,
}

trainer_config = {
    'name': None,
    'notes': None,
}

overwrite_config = {
    "data_config": data_config,
    "lit_model_config": lit_model_config,
    "trainer_config": trainer_config,
    'test_model': False,
}

pretrain_config_path = Path(PRETRAIN_PATH).parent.parent/'grappa_config.yaml'
with open(pretrain_config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# overwrite_config:
for k in overwrite_config.keys():
    assert k in config.keys(), f"Key {k} not in config."
    if isinstance(overwrite_config[k], dict):
        for kk in overwrite_config[k].keys():
            assert kk in config[k].keys(), f"Key {kk} not in config[{k}]."
            print(f'Overwriting config[{k}][{kk}] from {config[k][kk]} to {overwrite_config[k][kk]}')
            config[k][kk] = overwrite_config[k][kk]
    else:
        config[k] = overwrite_config[k]


do_trainrun(config=config, project=PROJECT, pretrain_path=PRETRAIN_PATH)
