import torch
from pathlib import Path
from typing import Union, Dict
import yaml
from grappa.models.deploy import model_from_config


def load_weights_torchhub(url:str, filename:str=None) -> dict:
    """
    Loads a model from a url. If filename is None, the filename is inferred from the url.
    """
    models_path = Path(__file__).parents[3] / "models"
    models_path.mkdir(parents=True, exist_ok=True)
    model_dict = torch.hub.load_state_dict_from_url(url, model_dir=str(models_path),file_name=filename)
    return model_dict

def model_from_dict(model_dict:dict):
    """
    Loads a model from a dictionary that contains a state_dict and a config.
    """
    state_dict = model_dict['state_dict']
    config = model_dict['config']
    model = model_from_config(model_config=config['model_config'])
    model.load_state_dict(state_dict)
    model.eval()
    return model


def model_from_url(url:str, filename:str=None):
    '''
    Loads a model from a url. If filename is None, the filename is inferred from the url.
    '''
    model_dict = load_weights_torchhub(url, filename)
    return model_from_dict(model_dict)


def model_from_tag(tag:str):
    """
    Loads a model from a tag. With each release, the mapping tag to url of model weights is updated such that models returned by this function are always at a version that works in the respective release.
    Possible tags:
    - latest
    """
    MODEL_NAMES = {
        'grappa-1.0-01-26-2024.pth': ['grappa-1.0', 'latest'],
    }

    BASE_URL = 'https://github.com/LeifSeute/test_torchhub/releases/download/model_release/'

    # first, go through the hard-coded dictionary above:
    filename = None
    for name in MODEL_NAMES.keys():
        if tag == name or tag in MODEL_NAMES[name]:
            filename = name
            break

    # if not found, go through all models, a user could have exported:
    if filename is None:
        for name in Path(__file__).parent.parent.parent.parent.glob('models/*.pth'):
            if tag == name.stem:
                filename = name.name
                break

    if filename is None:
        raise ValueError(f"Tag {tag} not found in model names {MODEL_NAMES} and not found in the grappa/models directory.")
            
    url = BASE_URL + filename

    return model_from_url(url)