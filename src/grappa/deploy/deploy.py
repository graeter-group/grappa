
from typing import Union
from grappa.models import get_models
from grappa.run import run_utils
from pathlib import Path
import torch





def model_from_config(config_path:Union[Path,str]=None, config:dict=None):
    """
    Initialize an untrained model from either a path to a config file or a config dict.
    """
    assert not (config_path is None and config is None), "Either config_path or config must be given."
    assert not (config_path is not None and config is not None), "Either config_path or config must be given, not both."

    if config is None:
        args = run_utils.load_yaml(config_path)
    else:
        args = config

    width = args["width"]
    n_res = args["n_res"]
    in_feat_name = args["in_feat_name"]
    partial_charges = args["partial_charges"]
    old_model = args["old_model"]
    n_heads = args["n_heads"]
    statistics = None


    REP_FEATS = width
    BETWEEN_FEATS = width*2

    bonus_feats = []
    bonus_dims = []
    if partial_charges:
        bonus_feats = ["q_ref"]
        bonus_dims = [1]

    model = get_models.get_full_model(statistics=statistics, n_res=n_res, rep_feats=REP_FEATS, between_feats=BETWEEN_FEATS, in_feat_name=in_feat_name, bonus_features=bonus_feats, bonus_dims=bonus_dims, old=old_model, n_heads=n_heads)
    return model


def model_from_version(version:Union[Path,str], device:str="cpu", model_name:str="best_model.pt"):
    """
    Loads a trained model from a version folder.
    """
    config_path = Path(version)/Path("config.yaml")
    model = model_from_config(config_path=config_path)
    model = model.to(device)

    model.load_state_dict(torch.load(Path(version)/Path(model_name), map_location=device))

    return model
    

def model_from_path(model_path:Union[Path,str], device:str="cpu"):
    """
    Loads a trained model from a path to a state_dict. In the parent folder of the state_dict file, there must be a config.yaml file.
    """
    model_path = Path(model_path)
    config_path = model_path.parent/"config.yaml"
    model = model_from_config(config_path=config_path)
    model = model.to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    return model