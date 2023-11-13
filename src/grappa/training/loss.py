"""

"""

import dgl
import torch
from typing import Dict, List
from grappa.utils import graph_utils


class ParameterLoss(torch.nn.Module):
    def __init__(self, weights:Dict[str,float]={"n2_k":1e-3, "n3_k":1e-2}, param_names:List[str]=["n2_k", "n2_eq", "n3_k", "n3_eq", "n4_k"]):
        super().__init__()
        self.weights = weights
        self.mse_loss = torch.nn.MSELoss()
        self.param_names = param_names


    def forward(self, g):
        loss = 0
        params = graph_utils.get_parameters(g)
        params_ref = graph_utils.get_parameters(g, suffix="_ref")
        for param in self.param_names:
            loss_term = self.mse_loss(params[param], params_ref[param])
            if param in self.weights.keys():
                loss_term = loss_term * self.weights[param]

            loss = loss + loss_term
        
        return loss
    

