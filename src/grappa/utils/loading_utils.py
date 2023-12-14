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
    state_dict = torch.hub.load_state_dict_from_url(url, model_dir=str(models_path),file_name=filename)
    return state_dict


def load_model(url:str, filename:str=None):
    '''
    Loads a model from a url. If filename is None, the filename is inferred from the url.
    '''
    model_dict = load_weights_torchhub(url, filename)
    state_dict = model_dict['state_dict']
    config = model_dict['config']
    model = model_from_config(model_config=config['model_config'])
    model.load_state_dict(state_dict)
    return model


