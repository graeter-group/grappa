"""
Contains the model wrapper class 'Grappa' that is used for parameter prediction.
"""

import torch

from grappa import constants
from grappa.data import Molecule, Parameters
from grappa.utils.loading_utils import model_from_tag
from grappa.models.grappa import GrappaModel



class Grappa:
    """
    Model wrapper class. Wraps a trained model and provides the interface to predict bonded parameters for a certain molecule.
    """
    def __init__(self, model:GrappaModel, max_element:int=constants.MAX_ELEMENT, device:str='cpu') -> None:
        self.model = model.to(device)
        self.model.eval()
        self.max_element = max_element
        self.device = device

    @classmethod
    def from_tag(cls, tag:str='latest', max_element=constants.MAX_ELEMENT, device:str='cpu') -> 'Grappa':
        """
        Load a model from a tag. Available tags are:
            - 'latest'
            - 'grappa-1.0'
            - 'grappa-1.1'
        """
        model = model_from_tag(tag)
        return cls(model, max_element, device)

    def predict(self, molecule: Molecule) -> Parameters:
        self.model.eval()
        # try if this give good error messages if input is wrong size
        # what happens if no weights are specified?

        # transform the input to a dgl graph
        g = molecule.to_dgl(max_element=self.max_element, exclude_feats=[])

        g = g.to(self.device)

        # write parameters in the graph
        with torch.no_grad():
            g = self.model(g)

        g = g.to('cpu')

        # extract parameters from the graph
        parameters = Parameters.from_dgl(g)

        return parameters