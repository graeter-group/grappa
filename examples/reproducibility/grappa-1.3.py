#%%

"""
How to obtain a training config file from an exported model
"""

from grappa.utils.model_loading_utils import get_path_from_tag
from pathlib import Path
import shutil
import yaml

# Load the model
modelpath = get_path_from_tag('grappa-1.3')

configpath = Path(modelpath).parent / 'config.yaml'
splitpath = Path(modelpath).parent / 'split.json'

shutil.copy(configpath, Path(__file__).parent/'config.yaml')

# replace the splitpath in the config file:
with open(Path(__file__).parent/'config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config['data']['data_module']['split_path'] = str(splitpath)

with open(Path(__file__).parent/'config.yaml', 'w') as f:
    yaml.dump(config, f)

print("Copied model config and specified split file.")
print("Train by running\n`python experiments/train.py --config-name train --config-path examples/reproducibility`\nfrom the root directory of the project.")
# %%
