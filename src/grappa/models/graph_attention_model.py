#%%

import torch
import dgl
from typing import List, Tuple, Dict, Union, Callable
import math
from grappa.constants import MAX_ELEMENT, RESIDUES

class ResidualAttentionBlock(torch.nn.Module):
    """
    Implements one residual layer consisting of 1 multi-head-attention message passing step (linear, activation on the node features with shared weights), and a skip connection. Block has a nonlinearity at the end but not in the beginning.
    Can only be used for homogeneous graphs.
    The individual heads map to ceil(out_feats/num_heads) features. This is being followed by a linear layer that maps from ceil(out_feats/num_heads)*num_heads to out_feats features.

    With self interaction we mean a linear layer that is put behind the multi head attention as in the attention is all you need paper.
    Layer norm is performed at the beginning of the block, over the feature dimension, not the node dimension.
    The gated_attention parameter determines whether the procedure from https://arxiv.org/pdf/1803.07294.pdf is used, assigning each attention head a gate that determines how important the head is. This is applied before the fully connected layer that mixes the different heads.
    """
    def __init__(self, in_feats:int, out_feats:int=None, num_heads:int=10, activation=torch.nn.ELU(), self_interaction=True, layer_norm=True, attention_layer=dgl.nn.pytorch.conv.DotGatConv, gated_attention:bool=False, dropout:float=0.2):
        super().__init__()

        if out_feats is None:
            out_feats = in_feats
            
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.out_feat_factor = int(out_feats/in_feats)

        # only do a skip connection if the output features is a multiple of the input features
        self.do_skip = (out_feats/in_feats).is_integer()


        assert attention_layer in [dgl.nn.pytorch.conv.DotGatConv, dgl.nn.pytorch.conv.GATConv, dgl.nn.pytorch.conv.GATv2Conv], "Attention layer must be one of the dgl attention layers"

        self.do_layer_norm = layer_norm

        outfeat_per_head = math.ceil(out_feats/num_heads)

        self.module = dgl.nn.pytorch.conv.DotGatConv(in_feats=in_feats, out_feats=outfeat_per_head, num_heads=num_heads)

        ############################
        self.gated_attention = gated_attention
        if gated_attention:
            # two layer: one convolutional mapping to gate_features, one linear layer mapping to num_heads followed by a sigmoid.
            gate_features = 64 # hyperparameter
            self.gate_module = dgl.nn.pytorch.SAGEConv(in_feats=in_feats, out_feats=gate_features, aggregator_type="pool")
            self.gate_reducer = torch.nn.Sequential(
                torch.nn.Linear(gate_features, num_heads),
                torch.nn.Sigmoid(),
            )
        ############################
        
        # use no attention dropout
        self.attention_dropout = torch.nn.Dropout(p=0)
        self.ff_dropout = torch.nn.Dropout(p=dropout)

        if layer_norm:
            self.layer_norm = torch.nn.LayerNorm(normalized_shape=(in_feats,)) # normalize over the feature dimension, not the node dimension (since this is not of constant length)

        self.activation = activation
        self.head_reducer = torch.nn.Linear(num_heads*outfeat_per_head, out_feats)

        if self_interaction:
            
            if layer_norm:
                self.interaction_norm = torch.nn.LayerNorm(normalized_shape=(out_feats,))

            self.self_interaction = torch.nn.Sequential(
                torch.nn.Linear(out_feats,4*out_feats),
                self.activation,
                torch.nn.Linear(4*out_feats,out_feats),
                self.activation,
            )
        else:
            self.self_interaction = None

    def forward(self, g, h):

        # do the skip only after the layer norm to keep normalization
        if self.do_layer_norm:
            h = self.layer_norm(h)
 
        h_skip = h
        h = self.activation(self.module(g,h))
        h = self.attention_dropout(h)
        # h now has the shape (num_nodes, num_heads, out_feats)

        if self.gated_attention:
            gate = self.gate_module(g,h_skip)
            gate = self.gate_reducer(gate)
            gate = gate.unsqueeze(dim=-1) # add a dimension at the out_feats position
            # gate now has shape (num_nodes, num_heads, 1)
            h = h*gate

        h = h.flatten(start_dim=-2, end_dim=-1) # flatten the head dimension to shape (num_nodes, out_feat*num_heads)
        h = self.head_reducer(h) # now has the shape num_nodes, out_feats

        if self.do_skip:
            # repeat h_skip self.out_feat_factor times and add it to h
            h_skip = h_skip.repeat_interleave(repeats=self.out_feat_factor, dim=-1)
            h = h + h_skip

        if not self.self_interaction is None:

            if self.do_layer_norm:
                h = self.interaction_norm(h)
            
            h_skip = h
            h = self.self_interaction(h)

            h = self.ff_dropout(h)

            # the interaction layer does not change the number of features, so we can always do the skip connection
            h = h + h_skip


        return h



class ResidualConvBlock(torch.nn.Module):
    """
    Implements one residual layer consisting of 1 graph convolutional step, and a skip connection. Block has a nonlinearity at the end but not in the beginning.
    Can only be used for homogeneous graphs. If self_interaction is True, a skipped linear layer is put behind the convolution.
    """
    def __init__(self, in_feats, out_feats=None, *message_args, activation=torch.nn.ELU(), message_class=dgl.nn.pytorch.conv.SAGEConv, self_interaction=True, layer_norm=True, dropout=0.2):
        super().__init__()


        if out_feats is None:
            out_feats = in_feats
            
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.out_feat_factor = int(out_feats/in_feats)

        self.dropout = torch.nn.Dropout(p=dropout)

        # only do a skip connection if the output features is a multiple of the input features
        self.do_skip = (out_feats/in_feats).is_integer()


        if len(message_args) == 0 and message_class == dgl.nn.pytorch.conv.SAGEConv:
            message_args = (in_feats, out_feats, "mean")

        self.module = message_class(*message_args)
        self.activation = activation

        if layer_norm:
            self.layer_norm = torch.nn.LayerNorm(normalized_shape=(in_feats,)) # normalize over the feature dimension, not the node dimension (since this is not of constant length)

        if self_interaction:
            self.self_interaction = torch.nn.Sequential(
                torch.nn.Linear(out_feats,out_feats),
                self.activation,
            )
            if layer_norm:
                self.interaction_norm = torch.nn.LayerNorm(normalized_shape=(out_feats,))
        else:
            self.self_interaction = None


        self.do_layer_norm = layer_norm



    def forward(self, g, h):

        # do the skip only after the layer norm to keep normalization
        if self.do_layer_norm:
            h = self.layer_norm(h)

        h_skip = h

        h = self.module(g,h)

        h = self.dropout(h)
        
        if self.do_skip:
            # repeat h_skip self.out_feat_factor times and add it to h
            h_skip = h_skip.repeat_interleave(repeats=self.out_feat_factor, dim=-1)
            h = h + h_skip

        if self.self_interaction is not None:
            if self.do_layer_norm:
                h = self.interaction_norm(h)
            
            h_skip = h
            h = self.self_interaction(h)

            # the self_interaction can always be skipped since it maps from out_feats to out_feats
            h = h + h_skip

        return h



class Representation(torch.nn.Module):
    """
    Implementing:
        - a single linear layer with node-level-shared weights mapping from in_feats to h_feats
        - a stack of n_conv ResidualConvBlocks with self interaction and width h_feats
        - a stack of n_att ResidualAttentionBlocks with self interaction, output width h_feats and num_heads attention heads
        - a single linear layer with node-level-shared weights mapping from h_feats to out_feats
        if increase_width is true, doubles the width of the convolutional layers every step until one is larger than out_feats
        If attention_last is true, the attention blocks are put after the convolutional blocks, otherwise before.
    """
    def __init__(self, h_feats:int=256, out_feats:int=512, in_feats:int=None, n_conv=3, n_att=3, n_heads=10, increase_width=True, attention_last=True, in_feat_name:Union[str,List[str]]=["atomic_number", "residue", "in_ring", "formal_charge", "is_radical"], in_feat_dims:dict={}, bonus_features:List[str]=[], bonus_dims:List[int]=[], dropout=0.2):
        """
        Implementing:
            - a single linear layer with node-level-shared weights mapping from in_feats to h_feats
            - a stack of n_conv ResidualConvBlocks with self interaction and width h_feats
            - a stack of n_att ResidualAttentionBlocks with self interaction, output width h_feats and num_heads attention heads
            - a single linear layer with node-level-shared weights mapping from h_feats to out_feats

        If in_feats is None, the number of input features are inferred from the in_feat_name. The input graph of the forward call must have an entry at g.nodes["n1"].data[feat] for each feat in in_feat_name.
        The in_feat_dims dictionary can be used to overwrite the default dimensions of the in-features.
        """
        super().__init__()

        if not isinstance(in_feat_name, list):
            in_feat_name = [in_feat_name]

        if in_feats is None:
            # infer the input features from the in_feat_name
            default_dims = {
                "atomic_number": MAX_ELEMENT,
                "residue": len(RESIDUES),
                "in_ring": 7,
                "mass": 1,
                "degree": 7,
                "formal_charge": 1,
                "q_ref":1,
                "is_radical": 1,
                "additional_features": 5,
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

        # self.layer_norm = torch.nn.LayerNorm(normalized_shape=(out_feats,)) # normalize over the feature dimension, not the node dimension (since this is not of constant length)

        self.pre_dense = torch.nn.Sequential(
            torch.nn.Linear(in_feats, h_feats),
            torch.nn.ELU(),
        )

        n_in_feats = [h_feats] * (n_conv + n_att)
        n_out_feats = [h_feats] * (n_conv + n_att)

        if increase_width:
            for i in range(n_conv + n_att-1):
                if n_out_feats[i] < out_feats:
                    n_out_feats[i] *= 2
                    n_in_feats[i+1] *= 2
                else:
                    n_out_feats[i] = n_out_feats[i-1]
            
            # now outfeats is increasing by factor 2 or constant until the last entry
            if len(n_out_feats) > 1:
                n_out_feats[-1] = n_out_feats[-2]
            elif len(n_out_feats) == 1:
                n_out_feats[0] = out_feats

        # use dropout every second layer:

        dropout_layers = [dropout if i%2==0 else 0. for i in range(n_conv+n_att)]
        
        if len(n_out_feats) > 0:

            self.no_convs = False

            self.conv_blocks = torch.nn.ModuleList([
                    ResidualConvBlock(in_feats=n_in_feats[i], out_feats=n_out_feats[i], activation=torch.nn.ELU(), self_interaction=True, dropout=dropout_layers[i])
                    for i in range(n_conv)
                ])

            
            self.att_blocks = torch.nn.ModuleList([
                    ResidualAttentionBlock(in_feats=n_in_feats[i], out_feats=n_out_feats[i], num_heads=n_heads, activation=torch.nn.ELU(), self_interaction=True, dropout=dropout_layers[i])
                    for i in range(n_conv, n_conv+n_att)
                ])

            
            self.post_dense = torch.nn.Sequential(
                torch.nn.Linear(n_out_feats[-1], out_feats),
            )


            if attention_last:
                self.blocks = self.conv_blocks + self.att_blocks
            else:
                self.blocks = self.att_blocks + self.conv_blocks

        # no convolutional blocks:
        else:

            self.no_convs = True

            self.post_dense = torch.nn.Sequential(
                torch.nn.Linear(h_feats, out_feats),
            )



    def forward(self, g, in_feature=None):
        """
        Input: graph with node features of shape (n_nodes, in_feature_dim)
        Output: graph with node features of shape (n_nodes, out_feature_dim)
        """
        if in_feature is None:
            try:
                # concatenate all the input features, allow the shape (n_nodes) and (n_nodes,n_feat)
                in_feature = torch.cat([g.nodes["n1"].data[feat]*1.
                                        if len(g.nodes["n1"].data[feat].shape) >=2 else g.nodes["n1"].data[feat].unsqueeze(dim=-1)
                                        for feat in self.in_feat_name], dim=-1)
                assert len(in_feature.shape) == 2, f"the input features must be of shape (n_nodes, n_features), but got {in_feature.shape}"
            except:
                print(*[(feat, g.nodes["n1"].data[feat].shape) for feat in self.in_feat_name], sep="\n")
                raise

        h = self.pre_dense(in_feature)

        g_ = dgl.to_homogeneous(g.node_type_subgraph(["n1"]))

        if not self.no_convs:
            # do message passing:
            for block in self.blocks:
                h = block(g_,h)

        h = self.post_dense(h)

        # h = self.layer_norm(h)

        g.nodes["n1"].data["h"] = h
        return g
    
# %%
