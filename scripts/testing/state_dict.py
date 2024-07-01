#%%
import torch
from grappa.models import GrappaModel
import yaml
from pathlib import Path

ckpt_path = '/local/user/seutelf/grappa/ckpt/grappa-1.3/baseline/2024-06-19_04-54-30/last.ckpt'

#%%

def get_model_from_checkpoint(ckpt: dict, model_config:dict, device: torch.device='cpu') -> GrappaModel:
    model = GrappaModel(**model_config).to(device)
    state_dict = ckpt['state_dict']
    # grappa model wieghts are stored as model.0 (zeroth entry in torch.sequential and model entry of ckpt state dict):
    state_dict = {k.replace('model.0.', ''): v for k, v in state_dict.items() if 'model.0.' in k}
    model.load_state_dict(state_dict=state_dict)
    return model

def get_model_from_path(ckpt_path: str, device: torch.device='cpu') -> GrappaModel:
    ckpt = torch.load(ckpt_path, map_location=device)
    with open(Path(ckpt_path).parent / 'config.yaml', 'r') as f:
        model_config = yaml.safe_load(f)['model']
    return get_model_from_checkpoint(ckpt, model_config, device)


model = get_model_from_path(ckpt_path)
# %%
import sys
import numpy as np

def get_size(obj, unit='MB'):
    size_bytes = sys.getsizeof(obj)
    if isinstance(obj, (list, tuple, set, dict)):
        size_bytes += sum(get_size(item, 'B') for item in obj)
    elif hasattr(obj, '__dict__'):
        size_bytes += get_size(obj.__dict__, 'B')
    elif isinstance(obj, np.ndarray):
        size_bytes += obj.nbytes
    size_mb = size_bytes / (1024 ** 2)
    return size_mb if unit == 'MB' else size_bytes


# %%
from grappa.utils.model_loading_utils import model_from_tag

model = model_from_tag('test')
# %%
