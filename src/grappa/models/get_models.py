
from grappa.models import readout
from grappa.models import gated_torsion
from grappa.models import old_gated_torsion

from grappa.models.readout import get_default_statistics
from grappa.models.graph_attention_model import Representation
from grappa.models.old_graph_model import Representation as old_Representation
from grappa.models import old_readout
import torch
from typing import Union, List, Tuple


def get_readout(statistics, rep_feats=512, between_feats=512, old=False, use_improper=True):

    if old:
        readout_module = old_readout
        torsion_module = old_gated_torsion
    else:
        readout_module = readout
        torsion_module = gated_torsion


    bond_angle = torch.nn.Sequential(
        readout_module.WriteBondParameters(rep_feats=rep_feats, between_feats=between_feats, stat_dict=statistics),
        readout_module.WriteAngleParameters(rep_feats=rep_feats, between_feats=between_feats, stat_dict=statistics)
    )

    torsion = torsion_module.GatedTorsion(rep_feats=rep_feats, between_feats=between_feats, improper=False)

    model = bond_angle
    model.add_module("torsion", torsion)

    if use_improper:
        improper = torsion_module.GatedTorsion(rep_feats=rep_feats, between_feats=between_feats, improper=True)
        model.add_module("improper", improper)

    return model



def get_full_model(statistics=None, rep_feats=512, between_feats=512, readout_feats=512, n_conv=3, n_att=3, in_feat_name:Union[str,List[str]]=["atomic_number", "residue", "in_ring", "formal_charge", "is_radical"], bonus_features=[], bonus_dims=[], old=False, n_heads=6, use_improper=True):
    
    if statistics is None:
        statistics = get_default_statistics()

    if old:
        assert n_att == 0, "old model does not support attention"
        representation = old_Representation(h_feats=between_feats, out_feats=rep_feats, n_conv=n_conv, in_feat_name=in_feat_name, bonus_features=bonus_features, bonus_dims=bonus_dims)
    else:
        representation = Representation(h_feats=between_feats, out_feats=rep_feats, n_conv=n_conv, n_att=n_att, in_feat_name=in_feat_name, bonus_features=bonus_features, bonus_dims=bonus_dims, n_heads=n_heads)


    readout = get_readout(statistics=statistics, rep_feats=rep_feats, between_feats=readout_feats, old=old, use_improper=use_improper)

    model = torch.nn.Sequential(
        representation,
        readout
    )

    return model