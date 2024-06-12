import torch
from pathlib import Path
from typing import Union, Dict
import yaml

def flatten_dict(d, parent_key='', sep=':'):
    """
    Flatten a nested dictionary.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key string for recursion.
        sep (str): Separator between keys.

    Returns:
        dict: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        if sep in k:
            raise ValueError(f"Separator '{sep}' is not allowed in dict keys. Found in key '{k}'")
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)



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


