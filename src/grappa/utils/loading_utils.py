import torch
from pathlib import Path
from typing import Union, Dict
import yaml
from grappa.models.deploy import model_from_config

def model_from_tag(tag:str, filename=None):
    """
    Loads a model from a tag. With each release, the mapping tag to url of model weights is updated such that models returned by this function are always at a version that works in the respective release.
    Possible tags:
    - latest
    - grappa-1.0
    """
    model_dict = model_dict_from_tag(tag, filename=filename)
    model = model_from_dict(model_dict)
    return model

def url_from_tag(tag:str):
    """
    Loads a model from a tag. With each release, the mapping tag to url of model weights is updated such that models returned by this function are always at a version that works in the respective release.
    Possible tags:
    - latest
    - grappa-1.0
    """
    MODEL_NAMES = {
        'https://github.com/hits-mbm-dev/grappa/releases/download/v.1.0.0/grappa-1.0-20240209.pth': ['grappa-1.0', 'latest'],
    }

    # first, go through the hard-coded dictionary above:
    filename = None
    for name in MODEL_NAMES.keys():
        if tag == Path(name).stem or tag in MODEL_NAMES[name]:
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


    return filename


def load_from_torchhub(url:str, filename:str=None) -> dict:
    """
    Loads a model from a url. If filename is None, the filename is inferred from the url.
    """
    models_path = Path(__file__).parents[3] / "models"
    models_path.mkdir(parents=True, exist_ok=True)
    model_dict = torch.hub.load_state_dict_from_url(url, model_dir=str(models_path),file_name=filename)
    return model_dict

def model_from_dict(model_dict:dict):
    """
    Loads a model from a dictionary that contains (at least) a state_dict and a config.
    """
    state_dict = model_dict['state_dict']
    config = model_dict['config']
    model = model_from_config(model_config=config['model_config'])
    model.load_state_dict(state_dict)
    model.eval()
    return model


def model_dict_from_tag(tag:str, filename:str=None):
    '''
    Loads a model_dict, that is a dictionary
    {'state_dict': state_dict, 'config': config, 'split_names': split_names}
    from a tag. If filename is None, the filename is inferred from the url.
    '''
    url = url_from_tag(tag)

    model_dict = load_from_torchhub(url, filename)
    return model_dict
