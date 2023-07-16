
from typing import Union
from grappa.models import get_models
from grappa.run import run_utils
from pathlib import Path
import torch





def model_from_config(config_path:Union[Path,str]=None, config:dict=None, stat_dict:dict=None):
    """
    Initialize an untrained model from either a path to a config file or a config dict.
    If you intend to train the model, it is recommended to provide a stat_dict, which is a dictionary containing the mean and std of classical ff parameters in a training set.
    """
    assert not (config_path is None and config is None), "Either config_path or config must be given."
    assert not (config_path is not None and config is not None), "Either config_path or config must be given, not both."

    if config is None:
        args = run_utils.load_yaml(config_path)
    else:
        args = config

    width = args["width"]
    rep_feats = args["rep_feats"]
    n_conv = args["n_conv"]
    n_att = args["n_att"]
    in_feat_name = args["in_feat_name"]
    old_model = args["old_model"]
    n_heads = args["n_heads"]
    readout_width = args["readout_width"]


    REP_FEATS = rep_feats
    BETWEEN_FEATS = width


    model = get_models.get_full_model(statistics=stat_dict, rep_feats=REP_FEATS, between_feats=BETWEEN_FEATS, n_conv=n_conv, n_att=n_att, in_feat_name=in_feat_name, old=old_model, n_heads=n_heads, readout_feats=readout_width)

    return model


def model_from_version(version:Union[Path,str], device:str="cpu", model_name:str="best_model.pt"):
    """
    Loads a trained model from a version folder.
    """
    config_path = Path(version)/Path("model_config.yml")
    model = model_from_config(config_path=config_path)
    model = model.to(device)

    model.load_state_dict(torch.load(Path(version)/Path(model_name), map_location=device))

    return model
    

def model_from_path(model_path:Union[Path,str], device:str="cpu"):
    """
    Loads a trained model from a path to a state_dict. In the parent folder of the state_dict file, there must be a model_config.yml file.
    """
    model_path = Path(model_path)
    config_path = model_path.parent/"model_config.yml"
    model = model_from_config(config_path=config_path)
    model = model.to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def get_default_model_config():
    args = {
        "width":512,
        "rep_feats":512,
        "n_conv":3,
        "n_att":3,
        "n_heads":6,
        "readout_width":512,
        "old_model":False,
        "in_feat_name":["atomic_number", "residue", "in_ring", "formal_charge", "is_radical"],
    }

    return args
