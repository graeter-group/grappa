
from grappa.models import readout
from grappa.models import gated_torsion
from grappa.models import old_gated_torsion

from grappa.models import attention_readout

from grappa.models.readout import get_default_statistics
from grappa.models.graph_attention_model import Representation
from grappa.models.old_graph_model import Representation as old_Representation
from grappa.models import old_readout
import torch
from typing import Union, List, Tuple


# decide for each interaction type its own readout module in the future
def get_readout(statistics, rep_feats=512, between_feats=512, old=False, use_improper=True, attentional=True, n_att=2, n_heads=8, dense_layers=2, dropout=0.2, layer_norm=True, reducer_feats=None, attention_hidden_feats=None, positional_encoding=True, n_periodicity_proper=6, n_periodicity_improper=3):
    """

    Attention readout hyperparameters (only have an effect if attentional=True):
        n_att: number of attention layers
        n_heads: number of heads for each attention layer
        dense_layers: number of dense layers in the reducer
        dropout: dropout probability for the attention layers
        layer_norm: whether to use layer normalization in the attention layers
        reducer_feats: number of features in the dense layers of the reducer
        attention_hidden_feats: number of features in the hidden layers of the attention layers

    """

    assert "n2_k" in statistics["mean"].keys(), "statistics must contain at least the parameters for the bond term, keys are: " + str(list(statistics["mean"].keys()))

    if attentional:
        bond = attention_readout.WriteBondParameters(rep_feats=rep_feats, between_feats=between_feats, stat_dict=statistics, n_att=n_att, n_heads=n_heads, dense_layers=dense_layers, dropout=dropout, layer_norm=layer_norm, reducer_feats=reducer_feats, attention_hidden_feats=attention_hidden_feats, positional_encoding=positional_encoding)

        # assume that the poitional encoding is of shape n_atoms_in_interaction x 1
        if positional_encoding:
            between_feats -= 1

            angle = attention_readout.WriteAngleParameters(rep_feats=rep_feats, between_feats=between_feats, stat_dict=statistics, n_att=n_att, n_heads=n_heads, dense_layers=dense_layers, dropout=dropout, layer_norm=layer_norm, reducer_feats=reducer_feats, attention_hidden_feats=attention_hidden_feats, positional_encoding=positional_encoding)

            torsion = attention_readout.WriteTorsionParameters(rep_feats=rep_feats, between_feats=between_feats, stat_dict=statistics, improper=False, n_att=n_att, n_heads=n_heads, dense_layers=dense_layers, dropout=dropout, layer_norm=layer_norm, reducer_feats=reducer_feats, attention_hidden_feats=attention_hidden_feats, positional_encoding=positional_encoding, n_periodicity=n_periodicity_proper)


        if use_improper:
            improper = attention_readout.WriteTorsionParameters(rep_feats=rep_feats, between_feats=between_feats, stat_dict=statistics, improper=True, n_att=n_att, n_heads=n_heads, dense_layers=dense_layers, dropout=dropout, layer_norm=layer_norm, reducer_feats=reducer_feats, attention_hidden_feats=attention_hidden_feats, positional_encoding=positional_encoding, n_periodicity=n_periodicity_improper)
        else:
            improper = attention_readout.Identity()

    else:

        if old:
            readout_module = old_readout
            torsion_module = old_gated_torsion
        else:
            readout_module = readout
            torsion_module = gated_torsion


        bond = readout_module.WriteBondParameters(rep_feats=rep_feats, between_feats=between_feats, stat_dict=statistics)
        angle = readout_module.WriteAngleParameters(rep_feats=rep_feats, between_feats=between_feats, stat_dict=statistics)

        torsion = torsion_module.GatedTorsion(rep_feats=rep_feats, between_feats=between_feats, improper=False)

        if use_improper:
            improper = torsion_module.GatedTorsion(rep_feats=rep_feats, between_feats=between_feats, improper=True)
        

    model = torch.nn.Sequential(bond)
    model.add_module("angle", angle)
    model.add_module("torsion", torsion)

    if use_improper:
        model.add_module("improper", improper)

    return model



def get_full_model(statistics=None, rep_feats=512, between_feats=512, readout_feats=512, n_conv=3, n_att=3, in_feat_name:Union[str,List[str]]=["atomic_number", "residue", "in_ring", "formal_charge", "is_radical"], bonus_features=[], bonus_dims=[], old=False, n_heads=6, use_improper=True, attentional=True, n_att_readout=2, n_heads_readout=8, dense_layers=2, dropout=0., layer_norm=True, reducer_feats=None, attention_hidden_feats=None, positional_encoding=True, rep_dropout=0, n_periodicity_proper=6, n_periodicity_improper=3):
    
    if statistics is None:
        statistics = get_default_statistics()

    if old:
        assert n_att == 0, "old model does not support attention"
        representation = old_Representation(h_feats=between_feats, out_feats=rep_feats, n_conv=n_conv, in_feat_name=in_feat_name, bonus_features=bonus_features, bonus_dims=bonus_dims)
    else:
        representation = Representation(h_feats=between_feats, out_feats=rep_feats, n_conv=n_conv, n_att=n_att, in_feat_name=in_feat_name, bonus_features=bonus_features, bonus_dims=bonus_dims, n_heads=n_heads, dropout=rep_dropout)


    readout = get_readout(statistics=statistics, rep_feats=rep_feats, between_feats=readout_feats, old=old, use_improper=use_improper, attentional=attentional, n_att=n_att_readout, n_heads=n_heads_readout, dense_layers=dense_layers, dropout=dropout, layer_norm=layer_norm, reducer_feats=reducer_feats, attention_hidden_feats=attention_hidden_feats, positional_encoding=positional_encoding)

    model = torch.nn.Sequential(
        representation,
        readout
    )

    return model