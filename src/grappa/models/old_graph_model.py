# NOTE: include batchnorm

import torch
import dgl
from typing import List, Tuple, Dict, Union, Callable

"""
Implements one residual layer consisting of 2 message passing steps with optional self interaction (linear, activation, linear on the node features with shared weights), and a skip connection, also with optional self interaction. No nonlinearity is put on beginning and end of the block.
Can only be used for homogeneous graphs.
"""
class ResidualGraphBlock(torch.nn.Module):
    def __init__(self, n_feat, *message_args, activation=torch.nn.ELU(), message_class=dgl.nn.pytorch.conv.SAGEConv, self_interaction=True):
        super().__init__()
        self.module_skip = torch.nn.Sequential(
            torch.nn.Linear(n_feat,n_feat),
            activation,
            torch.nn.Linear(n_feat,n_feat),
        )

        self.module1 = message_class(*message_args)
        self.module2 = message_class(*message_args)
        self.activation = activation

        if self_interaction:
            self.self_interaction = torch.nn.Sequential(
                torch.nn.Linear(n_feat,n_feat),
                self.activation,
                torch.nn.Linear(n_feat,n_feat),
            )
        else:
            self.self_interaction = torch.nn.ELU()

    def forward(self, g, h):
        h_skip = self.module_skip(h)
        h = self.activation(self.module1(g,h))
        h = self.activation(self.self_interaction(h))
        h = self.module2(g,h)
        return h_skip+h


"""
Implementing a stack of ResidualGraphBlocks followed by message passing layers without skip connection and one large skip connection skipping all message passing steps. Also implements linear layers with node-level-shared weights as first and final layer.
"""
class Representation(torch.nn.Module):
    def __init__(self, h_feats:int, out_feats:int, in_feats:int=None, n_residuals=3, n_conv=2, large_skip:bool=True, in_feat_name:Union[str,List[str]]=["atomic_number", "residue", "in_ring", "mass", "degree", "formal_charge", "q_ref", "is_radical"], in_feat_dims:dict={}, bonus_features:List[str]=[], bonus_dims:List[int]=[]):
        super().__init__()

        if not isinstance(in_feat_name, list):
            in_feat_name = [in_feat_name]

        if in_feats is None:
            # infer the input features from the in_feat_name
            default_dims = {
                "atomic_number": 26,
                "residue": 27,
                "in_ring": 7,
                "mass": 1,
                "degree": 7,
                "formal_charge": 1,
                "q_ref":1,
                "is_radical": 1,
            }
            # overwrite/append to these default values:
            for key in in_feat_dims.keys():
                default_dims[key] = in_feat_dims[key]
            in_feat_dims = [default_dims[feat] for feat in in_feat_name]

        for i, feat in enumerate(bonus_features):
            if feat in in_feat_name:
                continue
            in_feat_name.append(feat)
            if bonus_dims == []:
                in_feat_dims.append(1)
            else:
                in_feat_dims.append(bonus_dims[i])
            
        if in_feats is None:
            in_feats = sum(in_feat_dims)

        self.in_feats = in_feats

        self.in_feat_name = in_feat_name

        self.large_skip = large_skip

        self.pre_dense = torch.nn.Sequential(
            torch.nn.Linear(in_feats, h_feats),
        )

        self.post_dense = torch.nn.Sequential(
            torch.nn.Linear(h_feats, out_feats),
        )
        if self.large_skip:
            self.post_dense = torch.nn.Sequential(
                torch.nn.Linear(2*h_feats, out_feats),
            )

        self.blocks = torch.nn.ModuleList([
                ResidualGraphBlock(h_feats, h_feats, h_feats, "mean", message_class=dgl.nn.pytorch.conv.SAGEConv)
                for i in range(n_residuals)
            ])

        self.blocks.extend(
            [
            dgl.nn.pytorch.conv.SAGEConv(h_feats, h_feats, aggregator_type="mean")
                for i in range(n_conv)
            ]
        )


    def forward(self, g, in_feature=None):
        if in_feature is None:
            in_feature = torch.cat([g.nodes["n1"].data[feat]*1. for feat in self.in_feat_name], dim=-1)

        h = torch.nn.functional.elu(self.pre_dense(in_feature))
        if self.large_skip:
            h_skip = torch.nn.functional.elu(h)

        g_ = dgl.to_homogeneous(g.node_type_subgraph(["n1"]))

        for block in self.blocks:
            h = torch.nn.functional.elu(block(g_,h))

        if self.large_skip:
            h = torch.cat((h,h_skip), dim=-1)
            
        h = self.post_dense(h)
        g.nodes["n1"].data["h"] = h
        return g
# %%
# model = Representation(256, out_feats=1, in_feats=1)