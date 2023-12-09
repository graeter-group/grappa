
from typing import Union, List, Tuple, Dict
from .grappa import GrappaModel



def model_from_config(model_config:Dict, stat_dict:Dict=None):
    """
    Initialize an untrained model from either a path to a config file or a config dict.
    If you intend to train the model, it is recommended to provide a stat_dict, which is a dictionary containing the mean and std of classical ff parameters in a training set. This is not necessary if you intend to load model weights from a pretrained model.
    """

    model = GrappaModel(stat_dict=stat_dict, **model_config)

    return model


def get_default_model_config(tag:str="med"):

    if tag == "small":
        args = get_small_model_config()
    # elif tag == "med":
    #     args = get_med_model_config()
    # elif tag == "large":
    #     args = get_large_model_config()
    # elif tag == "deep":
    #     args = get_deep_model_config()
    else:
        raise ValueError(f"Unknown tag {tag}")

    return args

def get_small_model_config():
    args = {
        "graph_node_features": 256,  # from "rep_feats"
        "in_feats": None,  # No direct counterpart, keeping default
        "in_feat_name": ["atomic_number", "partial_charge", "ring_encoding"],  # from "in_feat_name"
        "in_feat_dims": {},  # No direct counterpart, keeping default
        "gnn_width": 128,  # from "gnn_width"
        "gnn_attentional_layers": 2,  # from "n_att"
        "gnn_convolutions": 2,  # from "n_conv"
        "gnn_attention_heads": 8,  # from "n_heads"
        "gnn_dropout_attention": 0.,  # from "attention_dropout"
        "gnn_dropout_initial": 0.,  # Assuming similar to "dropout"
        "gnn_dropout_conv": 0.,  # from "conv_dropout"
        "gnn_dropout_final": 0.,  # from "final_gnn_dropout"
        "parameter_dropout": 0.,  # from "dropout" (assuming it's used similarly)
        "bond_transformer_depth": 2,  # from "n_att_readout" (assuming similar usage)
        "bond_n_heads": 8,  # from "n_heads_readout"
        "bond_transformer_width": 512,  # from "attention_hidden_feats"
        "bond_symmetriser_depth": 2,  # from "dense_layers_readout"
        "bond_symmetriser_width": 256,  # Assuming similar to "reducer_feats"
        # Repeat similarly for angle, proper, and improper parameters
        "angle_transformer_depth": 2,
        "angle_n_heads": 8,
        "angle_transformer_width": 512,
        "angle_symmetriser_depth": 2,
        "angle_symmetriser_width": 256,
        "proper_transformer_depth": 2,
        "proper_n_heads": 8,
        "proper_transformer_width": 512,
        "proper_symmetriser_depth": 2,
        "proper_symmetriser_width": 256,
        "improper_transformer_depth": 2,
        "improper_n_heads": 8,
        "improper_transformer_width": 512,
        "improper_symmetriser_depth": 2,
        "improper_symmetriser_width": 256,
        "n_periodicity_proper": 6,  # Directly from "n_periodicity_proper"
        "n_periodicity_improper": 3,  # Directly from "n_periodicity_improper"
        "gated_torsion": False,
        "wrong_symmetry": False,  # No direct counterpart, keeping default
        "positional_encoding": True,  # Directly from "positional_encoding"
        "layer_norm": True,  # Directly from "layer_norm"
        "self_interaction": True,  # No direct counterpart, assuming True
    }

    return args



# def get_med_model_config():
#     args = {
#         # "width":64,
#         "rep_feats":256,
#         "gnn_width":256,
#         "n_conv":2,
#         "n_att":2,
#         "n_heads":8,
#         "old_model":False,
#         "use_improper":True,
#         "in_feat_name":["atomic_number", "in_ring", "q_ref", "is_radical"],
#         "layer_norm":True,
#         "final_gnn_dropout":0.0,
#         "dropout":0,
#         "conv_dropout":0.,
#         "attention_dropout":0.,
#         "n_att_readout":2,
#         "dense_layers_readout":2,
#         "n_heads_readout":16,
#         "reducer_feats":256,
#         "attention_hidden_feats":512,
#         "positional_encoding":True,
#         "attentional":True,
#         "n_periodicity_proper":6,
#         "n_periodicity_improper":3,
#     }

#     return args


# def get_deep_model_config():
#     args = {
#         # "width":128,
#         "rep_feats":256,
#         "gnn_width":128,
#         "n_conv":0,
#         "n_att":6,
#         "n_heads":8,
#         "old_model":False,
#         "use_improper":True,
#         "in_feat_name":["atomic_number", "in_ring", "q_ref", "is_radical"],
#         "layer_norm":True,
#         "final_gnn_dropout":0.0,
#         "dropout":0,
#         "conv_dropout":0.,
#         "attention_dropout":0.,
#         "n_att_readout":3,
#         "dense_layers_readout":2,
#         "n_heads_readout":8,
#         "reducer_feats":128,
#         "attention_hidden_feats":512,
#         "positional_encoding":True,
#         "attentional":True,
#         "n_periodicity_proper":6,
#         "n_periodicity_improper":3,
#     }

#     return args


# def get_small_model_config():
#     args = {
#         # "width":128,
#         "rep_feats":256,
#         "gnn_width":128,
#         "n_conv":2,
#         "n_att":2,
#         "n_heads":8,
#         "old_model":False,
#         "use_improper":True,
#         "in_feat_name":["atomic_number", "in_ring", "q_ref", "is_radical"],
#         "layer_norm":True,
#         "final_gnn_dropout":0.0,
#         "dropout":0,
#         "conv_dropout":0.,
#         "attention_dropout":0.,
#         "n_att_readout":2,
#         "dense_layers_readout":2,
#         "n_heads_readout":8,
#         "reducer_feats":128,
#         "attention_hidden_feats":512,
#         "positional_encoding":True,
#         "attentional":True,
#         "n_periodicity_proper":6,
#         "n_periodicity_improper":3,
#     }

#     return args




# def get_large_model_config():
#     args = {
#         # "width":256,
#         "rep_feats":2048,
#         "gnn_width":512,
#         "n_conv":2,
#         "n_att":2,
#         "n_heads":16,
#         "old_model":False,
#         "use_improper":True,
#         "in_feat_name":["atomic_number", "in_ring", "q_ref", "is_radical"],
#         "layer_norm":True,
#         "final_gnn_dropout":0.0,
#         "dropout":0.,
#         "conv_dropout":0.,
#         "attention_dropout":0.,
#         "n_att_readout":4,
#         "dense_layers_readout":1,
#         "n_heads_readout":32,
#         "reducer_feats":512,
#         "attention_hidden_feats":2048,
#         "positional_encoding":True,
#         "attentional":True,
#         "n_periodicity_proper":6,
#         "n_periodicity_improper":3,
#     }

#     return args