"""
Contains the model wrapper class 'Grappa' that is used for parameter prediction.
"""

from typing import Union, Dict
from pathlib import Path

from grappa import constants
from grappa.data import Molecule, Parameters
import torch


class Grappa:
    """
    Model wrapper class. Wraps a trained model and provides the interface to predict bonded parameters for a certain molecule.
    """
    def __init__(self, model, max_element=constants.MAX_ELEMENT, device='cpu') -> None:
        self.model = model.to(device)
        self.model.eval()
        self.max_element = max_element


    def predict(self, molecule: Molecule) -> Parameters:
        self.model.eval()
        # try if this give good error messages if input is wrong size
        # what happens if no weights are specified?

        # transform the input to a dgl graph
        g = molecule.to_dgl(max_element=self.max_element, exclude_feats=[])

        # write parameters in the graph
        with torch.no_grad():
            g = self.model(g)

        # extract parameters from the graph
        parameters = Parameters.from_dgl(g)

        return parameters