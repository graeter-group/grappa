"""
Contains the model wrapper class 'Grappa' that is used for parameter prediction.
"""

from typing import Union
from pathlib import Path

from grappa.io import Molecule, Parameters
from grappa.run.run_utils import load_yaml
from grappa.models.get_models import get_full_model


class Grappa():
    """Model wrapper class
    """
    def __init__(self, model) -> None:
        self.model = model

    @classmethod
    def from_config(cls, config_path:Union[Path,str]):
        config = load_yaml(config_path)

        stat_dict = None # is it necessary to give access to this or is it always the default statistics in get_full_model?
        # could just enforce proper naming in config 
        model_args = {
            "statistics": stat_dict,
            "rep_feats": config["rep_feats"],
            "between_feats": config["width"],
            "n_conv": config["n_conv"],
            "n_att": config["n_att"],
            "in_feat_name": config["in_feat_name"],
            "old": config["old_model"],
            "n_heads": config["n_heads"],
            "readout_feats": config["readout_width"],
            "use_improper": config["use_improper"],
            "dense_layers": config["dense_layers_readout"],
            "n_att_readout": config["n_att_readout"],
            "n_heads_readout": config["n_heads_readout"],
            "reducer_feats": config["reducer_feats"],
            "attention_hidden_feats": config["attention_hidden_feats"],
            "positional_encoding": config["positional_encoding"],
            "layer_norm": config["layer_norm"],
            "dropout": config["dropout"],
            "final_dropout": config["final_dropout"],
            "rep_dropout": config["rep_dropout"],
            "attentional": config["attentional"],
            "n_periodicity_proper": config["n_periodicity_proper"],
            "n_periodicity_improper": config["n_periodicity_improper"],
        }

        model = get_full_model(**model_args)
        return Grappa(model)

    def predict(self, input: Molecule) -> Parameters:
        self.model.eval()
        # try if this give good error messages if input is wrong size
        parameters = self.model(input)
        return parameters



