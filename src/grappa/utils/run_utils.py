import torch
from pathlib import Path
from typing import Union, Dict
import yaml

def get_rundir()->Path:
    """
    Returns the path to the directory in which runs are stored
    """
    rundir = Path(__file__).parents[3] / "runs"
    rundir.mkdir(parents=True, exist_ok=True)
    return rundir


def load_yaml(path:Union[str,Path])->Dict:    
    with open(str(path), 'r') as f:
        d = yaml.safe_load(f)
    return d

def write_yaml(d:dict, path:Union[str,Path])->None:
    """
    Stores a dictionary as a yaml file. All pathlib.Path are converted to strings.
    """
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


