#%%

import torch
import dgl
from typing import List, Tuple, Dict, Union, Callable
import math
from grappa.constants import MAX_ELEMENT


class GrappaGNN(torch.nn.Module):
    """
    Implements a Graph Neural Network model, consisting of a sequence of graph convolutional and attention layers.

    This model sequentially applies:
    - A linear layer mapping input features to hidden features.
    - A stack of graph convolutional blocks (ResidualConvBlocks) with self-interaction.
    - A stack of graph attention blocks (ResidualAttentionBlocks) with self-interaction.
    - A final linear layer mapping hidden features to output features.

    ----------
    Parameters:
    ----------
        out_feats (int): Number of output features.
        in_feats (int): Number of input features. If None, inferred from `in_feat_name` and `in_feat_dims`.
        node_feats (int): Number of hidden node features. If None, defaults to `out_feats`.
        n_conv (int): Number of convolutional blocks.
        n_att (int): Number of attention blocks.
        n_heads (int): Number of attention heads in each attention block.
        in_feat_name (Union[str, List[str]]): Names of input features. Defaults to a standard list of graph features.
        in_feat_dims (Dict[str, int]): Dictionary mapping feature names to their dimensions. Used to override default feature dimensions.
        conv_dropout (float): Dropout rate in convolutional layers. Defaults to 0.
        attention_dropout (float): Dropout rate in attention layers. Defaults to 0.
        final_dropout (float): Dropout rate in the final output layer. Defaults to 0.
        initial_dropout (float): Dropout rate after the initial linear layer. Defaults to 0.

    ----------
    Returns:
    ----------
        dgl.DGLGraph: The input graph with node features stored in `g.nodes["n1"].data["h"]`.

    ----------
    Notes:
    ----------
    The input graph for the forward call must contain node features named as specified in `in_feat_name`.
    """
    def __init__(self, out_feats:int=512, in_feats:int=None, node_feats:int=None, n_conv:int=3, n_att:int=3, n_heads:int=8, in_feat_name:Union[str,List[str]]=["atomic_number", "ring_encoding", "partial_charge"], in_feat_dims:Dict[str,int]={}, conv_dropout:float=0., attention_dropout:float=0., final_dropout:float=0., initial_dropout:float=0., layer_norm:bool=True, self_interaction:bool=True):

        super().__init__()

        if not isinstance(in_feat_name, list):
            in_feat_name = [in_feat_name]

        if in_feats is None:
            # infer the input features from the in_feat_name
            default_dims = {
                "atomic_number": MAX_ELEMENT,
                "ring_encoding": 7,
                "partial_charge":1,
                "sp_hybridization": 6,
                "mass": 1,
                "degree": 7,
                "is_radical": 1,
            }
            # overwrite/append to these default values:
            for key in in_feat_dims.keys():
                default_dims[key] = in_feat_dims[key]
            in_feat_dims = [default_dims[feat] for feat in in_feat_name]

            
        if in_feats is None:
            in_feats = sum(in_feat_dims)

        if node_feats is None:
            node_feats = out_feats

        self.in_feats = in_feats

        self.in_feat_name = in_feat_name

        # self.layer_norm = torch.nn.LayerNorm(normalized_shape=(out_feats,)) # normalize over the feature dimension, not the node dimension (since this is not of constant length) # not necessary, the blocks have normalization

        self.initial_dropout = torch.nn.Dropout(initial_dropout)
        self.final_dropout = torch.nn.Dropout(final_dropout)

        self.pre_dense = torch.nn.Sequential(
            torch.nn.Linear(in_feats, node_feats),
            torch.nn.ELU(),
        )
        
        if n_conv + n_att > 0:

            self.no_convs = False

            self.conv_blocks = torch.nn.ModuleList([
                    ResidualConvBlock(in_feats=node_feats, out_feats=node_feats, activation=torch.nn.ELU(), self_interaction=self_interaction, dropout=conv_dropout, layer_norm=layer_norm)
                    for i in range(n_conv)
                ])

            
            self.att_blocks = torch.nn.ModuleList([
                    ResidualAttentionBlock(in_feats=node_feats,
                                           out_feats=node_feats,
                                           num_heads=n_heads,
                                           self_interaction=self_interaction,
                                           dropout_behind_mha=attention_dropout,
                                           dropout_behind_self_interaction=attention_dropout,
                                           skip_connection=True, layer_norm=layer_norm)
                    for i in range(n_conv, n_conv+n_att)
                ])

            
            self.post_dense = torch.nn.Sequential(
                torch.nn.Linear(node_feats, out_feats),
            )

            self.blocks = self.conv_blocks + self.att_blocks

        # no convolutional blocks:
        else:

            self.no_convs = True

            self.post_dense = torch.nn.Sequential(
                torch.nn.Linear(node_feats, out_feats),
            )



    def forward(self, g, in_feature=None):
        """
        Processes the input graph through the GrappaGNN model.

        Parameters:
            g (dgl.DGLGraph): The input graph with node features of shape (n_nodes, out_feats).
            in_feature (torch.Tensor, optional): Tensor of input features. If None, features are extracted from the graph `g`.

        Returns:
            dgl.DGLGraph: The graph `g` with output node features stored in `g.nodes["n1"].data["h"]`.

        The function first concatenates the input features (handling different shapes), then applies the sequence of GNN layers, and finally updates the graph with the resulting node features.
        """
        if in_feature is None:
            try:
                # concatenate all the input features, allow the shape (n_nodes) and (n_nodes,n_feat)
                in_feature = torch.cat([g.nodes["n1"].data[feat].float()
                                        if len(g.nodes["n1"].data[feat].shape) >=2 else g.nodes["n1"].data[feat].unsqueeze(dim=-1).float()
                                        for feat in self.in_feat_name], dim=-1)
                assert len(in_feature.shape) == 2, f"the input features must be of shape (n_nodes, n_features), but got {in_feature.shape}"
            except:
                raise

        h = self.pre_dense(in_feature)

        h = self.initial_dropout(h)

        g_ = dgl.to_homogeneous(g.node_type_subgraph(["n1"]))

        if not self.no_convs:
            # do message passing:
            for block in self.blocks:
                h = block(g_,h)

        h = self.post_dense(h)

        h = self.final_dropout(h)

        g.nodes["n1"].data["h"] = h
        
        return g
    



class ResidualAttentionBlock(torch.nn.Module):
    """
    Implements one residual layer consisting of 1 multi-head-attention message passing step (mlp on the node features with shared weights), and a skip connection. The block has a nonlinearity at the end but not in the beginning.
    The individual heads map to int(out_feats/n_heads) features per node and attention head. This is being followed by a linear layer that maps to out_feats features.

    With self interaction we mean a linear layer that is put behind the multi head attention as in the attention is all you need paper.
    Layer norm is performed at the beginning of the block, over the feature dimension, not the node dimension.


    Implements a residual layer with multi-head attention followed by a nodewise MLP for dgl graphs.
    The attention is calculated as dot product between linear projections of the node features on feature vectors of size out_feats//n_heads. The feed forward MLP that follows has hidden dimension = 4 * out_feats.

    Parameters:
        in_feats (int): Number of input features (D_in).
        out_feats (int, optional): Number of output features per attention head (D_out). Defaults to `in_feats`.
        num_heads (int): Number of attention heads (H).
        activation (torch.nn.Module): Activation function, applied post-attention and self-interaction.
        self_interaction (bool): If True, adds a self-interaction layer post-attention. Defaults to True.
        layer_norm (bool): If True, applies layer normalization at start of block. Defaults to True.
        attention_layer (torch.nn.Module): Type of DGL attention layer. Must be one of `DotGatConv`, `GATConv`, or `GATv2Conv`.
        dropout_behind_mha (float): Dropout rate after multi-head attention layer. Defaults to 0.1.
        dropout_behind_self_interaction (float): Dropout rate after self-interaction layer. Defaults to 0.1.
        skip_connection (bool): If True, adds skip connection, conditional on (D_out * H / D_in) being an integer. Defaults to True.

    The block structure is as follows:
        1. LayerNorm: LayerNorm(h) where h ∈ ℝ^(N x D_in)
        2. Multi-Head Attention: MultiHeadAttention(h) -> h' where h' ∈ ℝ^(N x H x D_out//H)
        3. Concatenate and Linear transform on h' -> h'' where h'' ∈ ℝ^(N x D_out)
        4. Dropout on h''
        5. Add h to h''
        6. h'' = LayerNorm(h'')
        7. h''' = Self-Interaction Layer (h''): Similar to Transformer's Feed Forward NN, a hidden layer with 4*D_out followed by an activation function, a linear layer mapping to D_out and another activation function.
        8. Dropout on h'''
        9. h''' = h''' + h''

    N: Number of nodes in the graph.

    """
    def __init__(self, in_feats:int, out_feats:int=None, num_heads:int=8, activation:torch.nn.Module=torch.nn.ELU(), self_interaction:bool=True, layer_norm:bool=True, attention_layer:torch.nn.Module=dgl.nn.pytorch.conv.DotGatConv, dropout_behind_mha:float=0.1, dropout_behind_self_interaction:float=0.1, skip_connection:bool=True):
        super().__init__()

        # NOTE: USE SPARSE ATTENTION!!

        if out_feats is None:
            out_feats = in_feats
            
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.feats_per_head = out_feats//num_heads

        self.num_heads = num_heads

        self.skip_connection = skip_connection
        if skip_connection:
        # only do a skip connection if the output features is a multiple of the input features
            assert (out_feats/in_feats).is_integer(), f"out_feats*num_heads/in_feats must be an integer, but got out_feats={out_feats}, in_feats={in_feats}, num_heads={num_heads}"

        assert attention_layer in [dgl.nn.pytorch.conv.DotGatConv, dgl.nn.pytorch.conv.GATConv, dgl.nn.pytorch.conv.GATv2Conv], "Attention layer must be one of the dgl attention layers 'DotGatConv', 'GATConv' or 'GATv2Conv'"


        self.graph_module = attention_layer(in_feats=in_feats, out_feats=int(out_feats/num_heads), num_heads=num_heads)
        
        # use no attention dropout
        self.dropout1 = torch.nn.Dropout(p=dropout_behind_mha) # applied to output of the attention layer before add and layer norm as in transformer
        self.dropout2 = torch.nn.Dropout(p=dropout_behind_self_interaction) # applied to output of the self interaction layer

        self.do_layer_norm = layer_norm

        if layer_norm:
            self.layer_norm = torch.nn.LayerNorm(normalized_shape=(in_feats,)) # normalize over the feature dimension, not the node dimension (since this is not of constant length)

        self.activation = activation
        self.head_reducer = torch.nn.Linear(num_heads*self.feats_per_head, out_feats) # the linear layer behind concatenating the heads

        if self_interaction:
            if layer_norm:
                self.interaction_norm = torch.nn.LayerNorm(normalized_shape=(out_feats,))

            self.self_interaction = torch.nn.Sequential(
                torch.nn.Linear(out_feats,4*out_feats), # this factor 4 is the same as in the transformer paper
                self.activation,
                torch.nn.Linear(4*out_feats,out_feats),
                self.activation,
            )
        else:
            self.self_interaction = None

    def forward(self, g, h):

        # do the skip behind the layer norm to keep normalization effect
        if self.do_layer_norm:
            h = self.layer_norm(h)
 
        h_skip = h
        h = self.graph_module(g,h)
        # h now has the shape (num_nodes, num_heads, out_feats)

        h = h.flatten(start_dim=-2, end_dim=-1) # flatten the head dimension to shape (num_nodes, out_feat*num_heads)
        h = self.head_reducer(h) # now has the shape num_nodes, out_feats

        h = self.dropout1(h)

        if self.skip_connection:
            # repeat h_skip self.out_feat_factor times and add it to h
            h_skip = h_skip.repeat_interleave(repeats=int(self.out_feats/self.in_feats), dim=-1)
            h = h + h_skip

        if not self.self_interaction is None:

            if self.do_layer_norm:
                h = self.interaction_norm(h)
            
            h_skip = h
            h = self.self_interaction(h)

            h = self.dropout2(h)

            if self.skip_connection:
                # the self_interaction can always be skipped since it maps from out_feats to out_feats
                h = h + h_skip

        return h



class ResidualConvBlock(torch.nn.Module):
    """
    Implements a residual layer with a graph convolutional step followed by a self-interaction layer and a skip connection.

    Parameters:
        in_feats (int): Number of input features (D_in).
        out_feats (int, optional): Number of output features (D_out). Defaults to `in_feats`.
        message_args (*args): Arguments for the message passing class constructor.
        activation (torch.nn.Module): Activation function applied post convolution and self-interaction.
        message_class (dgl.nn.Module): DGL convolutional layer class.
        self_interaction (bool): If True, adds a self-interaction (linear layer) after the convolution. Defaults to True.
        layer_norm (bool): If True, applies layer normalization at the start of the block. Defaults to True.
        dropout (float): Dropout rate applied after convolution and activation. Defaults to 0.0.
        skip_connection (bool): If True, adds a skip connection, conditional on (D_out / D_in) being an integer. Defaults to True.

    The block structure is as follows:
        1. LayerNorm: LayerNorm(h) where h ∈ ℝ^(N x D_in)
        2. Graph Convolution: MessagePassing(h) -> h' where h' ∈ ℝ^(N x D_out)
        3. Activation: Activation(h')
        4. Dropout on h'
        5. Add h to h'
        6. h' = LayerNorm(h')
        7. h'' = Self-Interaction Layer (h'): Linear layer and activation maintaining feature dimension D_out.
        8. Dropout on h''
        9. h'' = h'' + h'

    N: Number of nodes in the graph.
    """

    def __init__(self, in_feats:int, out_feats:int=None, *message_args, activation:torch.nn.Module=torch.nn.ELU(), message_class:torch.nn.Module=dgl.nn.pytorch.conv.SAGEConv, self_interaction:bool=True, layer_norm:bool=True, dropout:float=0.0, skip_connection:bool=True):
        super().__init__()

        if out_feats is None:
            out_feats = in_feats
            
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.dropout2 = torch.nn.Dropout(p=dropout)

        self.skip_connection = skip_connection
        if skip_connection:
            # only do a skip connection if the output features is a multiple of the input features
            assert (out_feats/in_feats).is_integer()

        if len(message_args) == 0 and message_class == dgl.nn.pytorch.conv.SAGEConv:
            message_args = (in_feats, out_feats, "mean")

        self.graph_module = message_class(*message_args)
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

        # do the skip after the layer norm to keep normalization
        if self.do_layer_norm:
            h = self.layer_norm(h)

        h_skip = h

        h = self.graph_module(g,h)

        h = self.activation(h)

        h = self.dropout1(h)
        
        if self.skip_connection:
            # repeat h_skip self.out_feat_factor times and add it to h
            h_skip = h_skip.repeat_interleave(repeats=int(self.out_feats/self.in_feats), dim=-1)
            h = h + h_skip


        if self.self_interaction is not None:
            if self.do_layer_norm:
                h = self.interaction_norm(h)
            
            h_skip = h
            h = self.self_interaction(h)

            h = self.dropout2(h)

            # the self_interaction can always be skipped since it maps from out_feats to out_feats
            h = h + h_skip
        
        return h

