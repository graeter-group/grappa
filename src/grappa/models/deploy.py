
from typing import Union, List, Tuple, Dict
from .grappa import GrappaModel
from grappa.utils.graph_utils import get_default_statistics



def model_from_config(model_config:Dict, param_statistics:Dict=get_default_statistics()):
    """
    Initialize an untrained model from either a path to a config file or a config dict.
    If you intend to train the model, it is recommended to provide a param_statistics, which is a dictionary containing the mean and std of classical ff parameters in a training set. This is not necessary if you intend to load model weights from a pretrained model.
    """

    model = GrappaModel(param_statistics=param_statistics, **model_config)

    return model

def get_default_model_config():
    args = {
        "graph_node_features": 256,
        "in_feats": None,
        "in_feat_name": ["atomic_number", "partial_charge", "ring_encoding", "degree", "charge_model"],
        "in_feat_dims": {},
        "gnn_width": 512,
        "gnn_attentional_layers": 7,
        "gnn_convolutions": 0,
        "gnn_attention_heads": 16,
        "gnn_dropout_attention": 0.3,
        "gnn_dropout_initial": 0.0,
        "gnn_dropout_conv": 0.1,
        "gnn_dropout_final": 0.1,
        "parameter_dropout": 0.5,
        "bond_transformer_depth": 3,
        "bond_n_heads": 8,
        "bond_transformer_width": 512,
        "bond_symmetriser_depth": 3,
        "bond_symmetriser_width": 256,
        "angle_transformer_depth": 3,
        "angle_n_heads": 8,
        "angle_transformer_width": 512,
        "angle_symmetriser_depth": 3,
        "angle_symmetriser_width": 256,
        "proper_transformer_depth": 3,
        "proper_n_heads": 8,
        "proper_transformer_width": 512,
        "proper_symmetriser_depth": 3,
        "proper_symmetriser_width": 256,
        "improper_transformer_depth": 3,
        "improper_n_heads": 8,
        "improper_transformer_width": 512,
        "improper_symmetriser_depth": 3,
        "improper_symmetriser_width": 256,
        "n_periodicity_proper": 6,
        "n_periodicity_improper": 3,
        "gated_torsion": True,
        "wrong_symmetry": False,
        "positional_encoding": True,
        "layer_norm": True,
        "self_interaction": True,
        "learnable_statistics": False,
        "torsion_cutoff": 1e-4,
    }

    return args
