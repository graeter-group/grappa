"""
Contains the model wrapper class 'Grappa' that is used for parameter prediction.
"""

import torch

from grappa import constants
from grappa.data import Molecule, Parameters
from grappa.utils.model_loading_utils import model_from_tag, model_from_path
from grappa.models.grappa_model import GrappaModel
from grappa.utils.dgl_utils import check_disconnected_graphs
import logging
from pathlib import Path
import warnings
logging.basicConfig(level=logging.INFO)

class Grappa:
    """
    Model wrapper class. Wraps a trained model and provides the interface to predict bonded parameters for a certain molecule.
    """
    def __init__(self, model:GrappaModel, max_element:int=constants.MAX_ELEMENT, device:str='cpu') -> None:
        self.model = model.to(device)
        self.model.eval()
        self.max_element = max_element
        self.device = device
        if not hasattr(model, 'field_of_view'):
            warnings.warn("Input model does not have a field_of_view attribute. Set it to msg_passing_steps + 3. Assumes 10 message passing steps by default, the actual number should be much smaller.")
            self.field_of_view = 10 + 3
        else:
            self.field_of_view = model.field_of_view

    @classmethod
    def from_tag(cls, tag:str='latest', max_element=constants.MAX_ELEMENT, device:str='cpu') -> 'Grappa':
        """
        Load a model from a tag. Available tags are:
            - 'latest'
            - 'grappa-1.4.0'
            - 'grappa-1.4.1-radical'
        """
        logging.info(f"Initializing model with tag {tag}...")
        model = model_from_tag(tag)
        return cls(model, max_element, device)

    @classmethod
    def from_ckpt(cls, ckpt_path:Path, max_element=constants.MAX_ELEMENT, device:str='cpu') -> 'Grappa':
        """
        Load a model from a .ckpt path.
        """
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Model checkpoint {ckpt_path} not found.")

        logging.info(f"Initializing model from checkpoint {ckpt_path}...")
        model = model_from_path(ckpt_path)
        return cls(model, max_element, device)

    def predict(self, molecule: Molecule) -> Parameters:
        self.model.eval()
        # try if this give good error messages if input is wrong size
        # what happens if no weights are specified?

        # transform the input to a dgl graph
        g = molecule.to_dgl(max_element=self.max_element, exclude_feats=[])

        # check if water is contained, throw an error if so
        check_disconnected_graphs(g)

        g = g.to(self.device)

        # write parameters in the graph
        with torch.no_grad():
            g = self.model(g)

        g = g.to('cpu')

        # extract parameters from the graph
        parameters = Parameters.from_dgl(g)

        return parameters