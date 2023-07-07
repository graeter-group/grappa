#%%

import torch
import dgl
from typing import List, Tuple, Dict, Union, Callable
import math

class ResidualGraphBlock(torch.nn.Module):
    """
    Implements one residual layer consisting of 1 multi-head-attention message passing step (linear, activation on the node features with shared weights), and a skip connection. Block has a nonlinearity at the end but not in the beginning.
    Can only be used for homogeneous graphs.
    The individual heads map to ceil(out_feats/num_heads) features. This is being followed by a linear layer that maps from ceil(out_feats/num_heads)*num_heads to out_feats features.

    With self interaction we mean a linear layer that is put behind the multi head attention as in the attention is all you need paper.
    Layer norm is performed at the beginning of the block, over the feature dimension, not the node dimension.
    The gated_attention parameter determines whether the procedure from https://arxiv.org/pdf/1803.07294.pdf is used, assigning each attention head a gate that determines how important the head is. This is applied before the fully connected layer that mixes the different heads.
    """
    def __init__(self, in_feats:int, out_feats:int, num_heads:int=10, activation=torch.nn.ELU(), self_interaction=True, layer_norm=True, attention_layer=dgl.nn.pytorch.conv.DotGatConv, gated_attention:bool=False):
        super().__init__()

        assert attention_layer in [dgl.nn.pytorch.conv.DotGatConv, dgl.nn.pytorch.conv.GATConv, dgl.nn.pytorch.conv.GATv2Conv], "Attention layer must be one of the dgl attention layers"

        self.do_layer_norm = layer_norm

        outfeat_per_head = math.ceil(out_feats/num_heads)

        self.module = dgl.nn.pytorch.conv.DotGatConv(in_feats=in_feats, out_feats=outfeat_per_head, num_heads=num_heads)

        self.gated_attention = gated_attention
        if gated_attention:
            # two layer: one convolutional mapping to gate_features, one linear layer mapping to num_heads followed by a sigmoid.
            gate_features = 64 # hyperparameter
            self.gate_module = dgl.nn.pytorch.SAGEConv(in_feats=in_feats, out_feats=gate_features, aggregator_type="pool")
            self.gate_reducer = torch.nn.Sequential(
                torch.nn.Linear(gate_features, num_heads),
                torch.nn.Sigmoid(),
            )

        if layer_norm:
            self.layer_norm = torch.nn.LayerNorm(normalized_shape=(in_feats,)) # normalize over the feature dimension, not the node dimension (since this is not of constant length)

        self.activation = activation
        self.head_reducer = torch.nn.Linear(num_heads*outfeat_per_head, out_feats)

        if self_interaction:
            
            if layer_norm:
                self.interaction_norm = torch.nn.LayerNorm(normalized_shape=(out_feats,))

            self.self_interaction = torch.nn.Sequential(
                torch.nn.Linear(out_feats,out_feats),
                self.activation,
            )
        else:
            self.self_interaction = None

    def forward(self, g, h):

        if self.do_layer_norm:
            h = self.layer_norm(h)
 
        h_skip = h
        h = self.activation(self.module(g,h))
        # h now has the shape (num_nodes, num_heads, out_feats)

        if self.gated_attention:
            gate = self.gate_module(g,h_skip)
            gate = self.gate_reducer(gate)
            gate = gate.unsqueeze(dim=-1) # add a dimension at the out_feats position
            # gate now has shape (num_nodes, num_heads, 1)
            h = h*gate

        h = h.flatten(start_dim=-2, end_dim=-1) # flatten the head dimension to shape (num_nodes, out_feat*num_heads)
        h = self.head_reducer(h) # now has the shape num_nodes, out_feats

        h += h_skip

        if not self.self_interaction is None:

            if self.do_layer_norm:
                h = self.interaction_norm(h)
            
            h_skip = h
            h = self.self_interaction(h)
            h += h_skip

        return h



"""
Implementing a stack of ResidualGraphBlocks followed by message passing layers without skip connection and one large skip connection skipping all message passing steps. Also implements linear layers with node-level-shared weights as first and final layer.
"""
class Representation(torch.nn.Module):
    def __init__(self, h_feats:int=256, out_feats:int=512, in_feats:int=None, n_conv=5, n_heads=10, in_feat_name:Union[str,List[str]]=["atomic_number", "residue", "in_ring", "mass", "degree", "formal_charge", "q_ref", "is_radical"], in_feat_dims:dict={}, bonus_features:List[str]=[], bonus_dims:List[int]=[], many_skips:bool=False):
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


        self.pre_dense = torch.nn.Sequential(
            torch.nn.Linear(in_feats, h_feats),
            torch.nn.ELU(),
        )

        self.post_dense = torch.nn.Sequential(
            torch.nn.Linear(h_feats, out_feats),
        )

        self.blocks = torch.nn.ModuleList([
                ResidualGraphBlock(in_feats=h_feats, out_feats=h_feats, num_heads=n_heads, activation=torch.nn.ELU(), self_interaction=True)
                for i in range(n_conv)
            ])



    def forward(self, g, in_feature=None):
        if in_feature is None:
            in_feature = torch.cat([g.nodes["n1"].data[feat]*1. for feat in self.in_feat_name], dim=-1)

        h = self.pre_dense(in_feature)

        g_ = dgl.to_homogeneous(g.node_type_subgraph(["n1"]))

        for block in self.blocks:
            h = block(g_,h)

        h = self.post_dense(h)
        g.nodes["n1"].data["h"] = h
        return g
# %%
# model = Representation(256, out_feats=1, in_feats=1)


if __name__ == '__main__':
    from grappa.run import run_utils
    [ds], _ = run_utils.get_data(["/hits/fast/mbm/seutelf/data/datasets/PDBDatasets/spice/amber99sbildn_60_dgl.bin"], n_graphs=1)
    g = ds[0]
    # transform to homograph:

    model = Representation(h_feats=256, out_feats=1, in_feats=None, in_feat_name=["atomic_number", "in_ring"])
    #%%
    g = model(g)

# %%
