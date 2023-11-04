"""
Contains the model wrapper class 'Grappa' that is used for parameter prediction.
"""

from typing import Union
from pathlib import Path

from grappa.data import Molecule, Parameters
from grappa.utils.run_utils import load_yaml, load_weights_torchhub
# from grappa.models.get_models import get_full_model


class Grappa():
    """Model wrapper class
    """
    def __init__(self, model) -> None:
        self.model = model

    @classmethod
    def from_config(cls, config_path:Union[Path,str]):
        config = load_yaml(config_path)

        # load model 
        architecture = config["architecture"]
        stat_dict = None # is it necessary to give access to this or is it always the default statistics in get_full_model?
        #TODO: could just enforce proper naming in config 
        model_args = {
            "statistics": stat_dict,
            "rep_feats": architecture["rep_feats"],
            "between_feats": architecture["width"],
            "n_conv": architecture["n_conv"],
            "n_att": architecture["n_att"],
            "in_feat_name": architecture["in_feat_name"],
            "old": architecture["old_model"],
            "n_heads": architecture["n_heads"],
            "readout_feats": architecture["readout_width"],
            "use_improper": architecture["use_improper"],
            "dense_layers": architecture["dense_layers_readout"],
            "n_att_readout": architecture["n_att_readout"],
            "n_heads_readout": architecture["n_heads_readout"],
            "reducer_feats": architecture["reducer_feats"],
            "attention_hidden_feats": architecture["attention_hidden_feats"],
            "positional_encoding": architecture["positional_encoding"],
            "layer_norm": architecture["layer_norm"],
            "dropout": architecture["dropout"],
            "final_dropout": architecture["final_dropout"],
            "rep_dropout": architecture["rep_dropout"],
            "attentional": architecture["attentional"],
            "n_periodicity_proper": architecture["n_periodicity_proper"],
            "n_periodicity_improper": architecture["n_periodicity_improper"],
        }
        model = None
        # model = get_full_model(**model_args)

        #load parameters
        weights = config["weights"]
        state_dict = load_weights_torchhub(weights["url"],weights["filename"])
        model.load_state_dict(state_dict)

        return Grappa(model)

    def predict(self, input: Molecule) -> Parameters:
        self.model.eval()
        # try if this give good error messages if input is wrong size
        # what happens if no weights are specified?
        parameters = self.model(input)
        return parameters


