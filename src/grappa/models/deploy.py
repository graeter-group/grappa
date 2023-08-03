
from typing import Union
from . import get_models
from ..run import run_utils
from pathlib import Path
import torch
from typing import Union, List, Tuple, Dict


def model_from_tag(tag:str, device:str="cpu")->torch.nn.Module:
    """
    Load a trained model from a tag. Available tags are:
    
    - example: An example model, not fine-tuned for good results.

    """

    if tag == "example":
        path = "/hits/fast/mbm/seutelf/grappa/mains/runs/stored_models/example/best_model.pt"
        config = None

    elif tag == "kimmdy_example":
        raise NotImplementedError("This model is not yet available.")
    else:
        raise ValueError(f"Unknown tag {tag}")
    

    model = model_from_path(model_path=path, config_path=config, device=device)

    return model



def model_from_config(config_path: Union[Path, str] = None, config: Dict = None, stat_dict: Dict = None):
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

    # Extract the required arguments from the config
    model_args = {
        "statistics": stat_dict,
        "rep_feats": args["rep_feats"],
        "between_feats": args["width"],
        "n_conv": args["n_conv"],
        "n_att": args["n_att"],
        "in_feat_name": args["in_feat_name"],
        "old": args["old_model"],
        "n_heads": args["n_heads"],
        "readout_feats": args["readout_width"],
        "use_improper": args["use_improper"],
        "dense_layers": args["dense_layers_readout"],
        "n_att_readout": args["n_att_readout"],
        "n_heads_readout": args["n_heads_readout"],
        "reducer_feats": args["reducer_feats"],
        "attention_hidden_feats": args["attention_hidden_feats"],
        "positional_encoding": args["positional_encoding"],
        "layer_norm": args["layer_norm"],
        "dropout": args["dropout"],
        "rep_dropout": args["rep_dropout"],
        "attentional": args["attentional"],
        "n_periodicity_proper": args["n_periodicity_proper"],
        "n_periodicity_improper": args["n_periodicity_improper"],
    }

    model = get_models.get_full_model(**model_args)

    return model


def model_from_version(version:Union[Path,str], device:str="cpu", model_name:str="best_model.pt"):
    """
    Loads a trained model from a version folder.
    """
    config_path = Path(version)/Path("model_config.yml")
    model_path = Path(version)/Path(model_name)
    return model_from_path(model_path=model_path, device=device, config_path=config_path)
    

def model_from_path(model_path:Union[Path,str], device:str="cpu", config_path:Union[Path,str]=None):
    """
    Loads a trained model from a path to a state_dict. In the parent folder of the state_dict file, there must be a model_config.yml file.
    """
    model_path = Path(model_path)
    if config_path is None:
        config_path = model_path.parent/"model_config.yml"
    model = model_from_config(config_path=config_path)

    model = model.to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))


    return model


def get_default_model_config(tag:str, scale:float=1.0):

    if tag == "small":
        args = get_small_model_config()
    elif tag == "med":
        args = get_med_model_config()
    elif tag == "large":
        args = get_large_model_config()

    return scale_model_config(args, scale=scale)



def get_med_model_config():
    args = {
        "width":128,
        "rep_feats":1024,
        "readout_width":512,
        "n_conv":3,
        "n_att":2,
        "n_heads":8,
        "old_model":False,
        "use_improper":True,
        "in_feat_name":["atomic_number", "in_ring", "q_ref", "is_radical"],
        "layer_norm":True,
        "dropout":0,
        "rep_dropout":0.,
        "n_att_readout":3,
        "dense_layers_readout":2,
        "n_heads_readout":32,
        "reducer_feats":512,
        "attention_hidden_feats":1024,
        "positional_encoding":True,
        "attentional":True,
        "n_periodicity_proper":6,
        "n_periodicity_improper":3,
    }

    return args


def get_small_model_config():
    args = {
        "width":64,
        "rep_feats":512,
        "readout_width":256,
        "n_conv":2,
        "n_att":2,
        "n_heads":8,
        "old_model":False,
        "use_improper":True,
        "in_feat_name":["atomic_number", "in_ring", "q_ref", "is_radical"],
        "layer_norm":True,
        "dropout":0,
        "rep_dropout":0.,
        "n_att_readout":2,
        "dense_layers_readout":2,
        "n_heads_readout":16,
        "reducer_feats":256,
        "attention_hidden_feats":512,
        "positional_encoding":True,
        "attentional":True,
        "n_periodicity_proper":6,
        "n_periodicity_improper":3,
    }

    return args


def get_large_model_config():
    args = {
        "width":128,
        "rep_feats":2048,
        "readout_width":512,
        "n_conv":3,
        "n_att":2,
        "n_heads":16,
        "old_model":False,
        "use_improper":True,
        "in_feat_name":["atomic_number", "in_ring", "q_ref", "is_radical"],
        "layer_norm":True,
        "dropout":0,
        "rep_dropout":0.,
        "n_att_readout":3,
        "dense_layers_readout":2,
        "n_heads_readout":64,
        "reducer_feats":512,
        "attention_hidden_feats":2048,
        "positional_encoding":True,
        "attentional":True,
        "n_periodicity_proper":6,
        "n_periodicity_improper":3,
    }

    return args


def scale_model_config(args, scale=1):
    """
    Scales a model by a factor. Only affects the widths.
    """
    for key in ["width", "rep_feats", "reducer_feats", "attention_hidden_feats", "n_heads"]:
        args[key] = int(args[key]*scale)
    
    # if scale is integer, also scale multihead attention parameters:
    if scale == float(int(scale)):
        args["readout_width"] = int(args["readout_width"]*scale)
        args["n_heads_readout"] = int(args["n_heads_readout"]*scale)


    return args