import torch
from pathlib import Path
from typing import Union
import yaml

def load_weights_torchhub(url:str, filename:str) -> dict:
    models_path = Path(__file__).parents[3] / "models"
    models_path.mkdir(parents=True, exist_ok=True)
    #torch.hub.set_dir('models_path')   # probably not necessary
    state_dict = torch.hub.load_state_dict_from_url(url, model_dir=str(models_path),file_name=filename)
    return state_dict


def load_yaml(path:Union[str,Path]):
    with open(str(path), 'r') as f:
        d = yaml.safe_load(f)
    return d

def store_yaml(d:dict, path:Union[str,Path]):
    with open(str(path), 'w') as f:
        yaml.dump(d, f)
