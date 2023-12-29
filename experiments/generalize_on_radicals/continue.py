from grappa.training.resume_trainrun import resume_trainrun
from grappa.utils.dataset_utils import get_data_path
from pathlib import Path
import yaml

PROJECT = 'generalize_on_radicals'

RUN_ID = 'ogyoai7r'

data_config = {
    "weights": {
        'gen2': 0.5, # empirical, makes training more stable
        'pepconf': 2, # empirical, harder to fit apparently
        'dipeptide_rad': 10,
        'AA_radical': 2,
        'spice_dipeptide_amber99sbildn': 10,
    },
    "balance_factor": 0.8,
}

lit_model_config = {
}

trainer_config = {
}

model_config = {
}

overwrite_config = {
    "model_config": model_config,
    "data_config": data_config,
    "lit_model_config": lit_model_config,
    "trainer_config": trainer_config,
    'test_model': False,
}


resume_trainrun(project=PROJECT, run_id=RUN_ID, overwrite_config=overwrite_config)