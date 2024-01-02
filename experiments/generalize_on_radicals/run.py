from grappa.training.trainrun import do_trainrun
from grappa.utils.dataset_utils import get_data_path
from pathlib import Path
import yaml

PROJECT = 'generalize_on_radicals'


data_config = {
    "datasets": [
        str(get_data_path()/"dgl_datasets"/dsname) for dsname in
        [
            # "spice-dipeptide",
            # "spice-des-monomers",
            # "spice-pubchem",
            # "gen2",
            # "gen2-torsion",
            # "pepconf-dlc",
            # "protein-torsion",
            # "rna-nucleoside",
            # "rna-diverse",
            'dipeptide_rad',
            'tripeptides_amber99sbildn',
            'spice_dipeptide_amber99sbildn',
            'AA_radical',
        ]
    ],
    "conf_strategy": 100,
    "splitpath": None,
    "partition": [ # may either be a partition list or a list of a default partition list and a dictionary mapping dsname to partition
        [0.8,0.1,0.1],
        {
            # 'rna-nucleoside': [1.,0.,0.],
        }
        ],
    "pure_train_datasets": [],
    "pure_val_datasets": [
        # str(get_data_path()/"dgl_datasets"/'tripeptides_amber99sbildn'),
        str(get_data_path()/"dgl_datasets"/'AA_natural'),
        # str(get_data_path()/"dgl_datasets"/'spice_dipeptide_amber99sbildn'),
    ],
    "pure_test_datasets": [
        # str(get_data_path()/"dgl_datasets"/'rna-trinucleotide')
    ],
    "weights": {
        # 'gen2': 0.5, # empirical, makes training more stable
        # 'pepconf': 2, # empirical, harder to fit apparently
        # 'AA_radical': 2,
        # 'dipeptide_rad': 5,
        # 'spice_dipeptide_amber99sbildn': 5,
    },
    "balance_factor": 0.,
}

lit_model_config = {
    "time_limit": 15,
    "finish_criterion": {},
}

trainer_config = {
    'name': 'only_peptide_amber99sbildn',
    'notes': None,
}

model_config = {
    'in_feat_name':['atomic_number', 'partial_charge', 'ring_encoding', 'is_radical'],
}

overwrite_config = {
    "model_config": model_config,
    "data_config": data_config,
    "lit_model_config": lit_model_config,
    "trainer_config": trainer_config,
    'test_model': False,
}


default_config_path = Path(__file__).parent/'default_grappa_config.yaml'
with open(default_config_path, 'r') as f:
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


do_trainrun(config=config, project=PROJECT)
