#%%

"""
How to obtain a training config file from an exported model
"""

from grappa.utils.model_loading_utils import get_path_from_tag
from pathlib import Path
import shutil

# Load the model
modelpath = get_path_from_tag('grappa-1.3')

configpath = Path(modelpath).parent / 'config.yaml'

shutil.copy(configpath, Path(__file__).parent/'config.yaml')
# %%
