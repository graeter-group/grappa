import torch
from pathlib import Path
from typing import Union, Dict
import yaml

def get_rundir()->Path:
    """
    Returns the path to the directory in which runs are stored
    """
    rundir = Path(__file__).parents[3] / "runs"
    print(rundir)
    rundir.mkdir(parents=True, exist_ok=True)
    return rundir

def load_weights_torchhub(url:str, filename:str) -> dict:
    models_path = Path(__file__).parents[3] / "models"
    models_path.mkdir(parents=True, exist_ok=True)
    #torch.hub.set_dir('models_path')   # probably not necessary
    state_dict = torch.hub.load_state_dict_from_url(url, model_dir=str(models_path),file_name=filename)
    return state_dict


def load_yaml(path:Union[str,Path])->Dict:    
    with open(str(path), 'r') as f:
        d = yaml.safe_load(f)
    return d

def write_yaml(d:dict, path:Union[str,Path])->None:
    # recursively redefine all paths to strings:
    d = d.copy()
    def path_to_str(d):
        for k,v in d.items():
            if isinstance(v, dict):
                d[k] = path_to_str(v)
            elif isinstance(v, Path):
                d[k] = str(v)
        return d

    d = path_to_str(d)

    with open(str(path), 'w') as f:
        yaml.dump(d, f)


def get_data_path()->Path:
    '''
    Returns the default path where to look for datasets.
    '''
    return Path(__file__).parents[3] / "data"