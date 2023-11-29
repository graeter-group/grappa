"""
Contains the model wrapper class 'Grappa' that is used for parameter prediction.
"""

from typing import Union, Dict
from pathlib import Path

from grappa import constants
from grappa.data import Molecule, Parameters
from grappa.utils.run_utils import load_yaml, load_weights_torchhub
from grappa.models.deploy import model_from_config


class Grappa:
    """
    Model wrapper class. Wraps a trained model and provides the interface to predict bonded parameters for a certain molecule.
    """
    def __init__(self, model, max_element=constants.MAX_ELEMENT) -> None:
        self.model = model

    @classmethod
    def from_config(cls, config:Union[Dict,Path,str]):
        """
        Loads the model from a config file. Then assigns weights by loading them from a url using torchhub.
        config must be a dictionary defining a model architecture as in grappa.deploy
        """
        if isinstance(config, str):
            config = Path(config)
        if isinstance(config, Path):
            config = load_yaml(config)
        elif not isinstance(config, dict):
            raise TypeError("config must be a dict, Path or str")
        
        model = model_from_config(config=config)

        max_element = config.get("max_element", constants.MAX_ELEMENT)

        #load parameters
        state_dict = load_weights_torchhub(config["url"], config["filename"])
        model.load_state_dict(state_dict)

        return Grappa(model, max_element=max_element)


    def predict(self, input: Molecule) -> Parameters:
        self.model.eval()
        # try if this give good error messages if input is wrong size
        # what happens if no weights are specified?

        # transform the input to a dgl graph
        g = input.to_dgl(max_element=self.max_element, exclude_feats=[])

        # write parameters in the graph
        g = self.model(g)

        # extract parameters from the graph
        parameters = Parameters.from_dgl(g)

        return parameters