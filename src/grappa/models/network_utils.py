import torch
import torch.nn as nn


class FeedForwardLayer(nn.Module):
    """
    Simple MLP with one hidden layer and, optionally, a skip connection, dropout at the end and layer normalization.
    
    ----------
    Parameters:
    ----------
        in_feats (int): The number of features in the input and output.
        hidden_feats (int, optional): The number of features in the hidden layer of the feed-forward network.
                                     Defaults to `in_feats`, following the original transformer paper.
        activation (torch.nn.Module): The activation function to be used in the feed-forward network. Defaults to nn.ELU().
        dropout (float): Dropout rate applied in the feed-forward network. Defaults to 0.
        skip (bool): If True, adds a skip connection between the input and output of the feed-forward network. Defaults to False.
        layer_norm (bool): If True, applies layer normalization after the input and before the output of the feed-forward network. Defaults to True.
    """
    def __init__(self, in_feats:int, hidden_feats:int=None, out_feats:int=None, activation:torch.nn.Module=nn.ELU(), dropout:float=0., skip:bool=False, layer_norm:bool=True):
        super().__init__()
        if hidden_feats is None:
            hidden_feats = in_feats
        if out_feats is None:
            out_feats = in_feats

        self.in_feats = in_feats
        self.hidden_feats = hidden_feats
        self.out_feats = out_feats

        self.linear1 = nn.Linear(in_feats, hidden_feats)
        self.linear2 = nn.Linear(hidden_feats, out_feats)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.skip = skip
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.norm1 = nn.LayerNorm(in_feats)

        skip_is_possible = (out_feats/in_feats).is_integer()
        assert skip_is_possible or not skip, f"Skip connection is not possible with {in_feats} input features and {out_feats} output features."


    def forward(self, x):
        if self.layer_norm:
            x = self.norm1(x)
        x_out = self.linear1(x)
        x_out = self.activation(x_out)
        x_out = self.linear2(x_out)
        x_out = self.dropout(x_out)
        if self.skip:
            x = torch.repeat_interleave(x, int(self.out_feats/self.in_feats), dim=-1)
            x_out = x_out + x
        return x_out


class DottedAttWithMLP(nn.Module):
    """
    Implements a transformer block with a multihead self-attention layer followed by a feed-forward layer. 
    
    ----------
    Parameters:
    ----------
        n_feats (int): The number of features in the input and output.
        num_heads (int): The number of attention heads in the multihead attention layer.
        hidden_feats (int, optional): The number of features in the hidden layer of the feed-forward network.
                                     Defaults to 4 times `n_feats`, following the original transformer paper.
        layer_norm (bool): If True, applies layer normalization after each major step. Defaults to True.
        dropout (float): Dropout rate applied after the attention layer and in the feed-forward network. Defaults to 0.

    ----------
    The process involves the following sequential operations:
        - Calculate attention weights for each node using a subset of features (n_feats // n_heads) per head.
        - Update node features using the calculated attention weights.
        - Apply dropout to the updated features.
        - Add the original input features (residual connection).
        - Apply layer normalization (optional).
        - Pass the result through a feed-forward network with one hidden layer.
        - Apply dropout to the output of the feed-forward network.
        - Add the output of the attention layer (residual connection).
    ----------
    Note:
    ----------
        The input and output feature dimension (n_feats) is the same.

        In torch 2.1.0 the multihead attention layer seems to have bugs with eval mode:
        In some cases, when we set need_weights=False, the output is nan in eval mode but not in train mode.
        If we set need_weights=True, the weights are all the same in eval mode (but not in train mode).
        This happens for trained models, not randomly initialized ones.
        Thus, we do not use dropout in nn.MultiheadAttention but apply it manually.
    """
    def __init__(self, n_feats, num_heads, hidden_feats=None, layer_norm=True, dropout=0.):
        super().__init__()

        if hidden_feats is None:
            hidden_feats = n_feats * 4 # as in the original transformer paper

        self.layer_norm = layer_norm
        if self.layer_norm:
            self.norm1 = nn.LayerNorm(n_feats)

        # multihead attention layer with n_heads, each taking n_feats//n_heads features as input.
        # the updated values are returned, not the attention weights
        assert n_feats % num_heads == 0, f"Number of features ({n_feats}) must be divisible by the number of heads ({num_heads})."
        self.attn = nn.MultiheadAttention(n_feats, num_heads, dropout=0)

        self.dropout = nn.Dropout(dropout)

        self.ff = FeedForwardLayer(n_feats, hidden_feats, out_feats=n_feats, dropout=dropout, skip=True, layer_norm=layer_norm)


    def forward(self, x):
        if self.layer_norm:
            x = self.norm1(x)

        # In torch 2.1.0 the multihead attention layer seems to have bugs with eval mode:
        # In some cases, when we set need_weights=False, the output is nan in eval mode but not in train mode.
        # If we set need_weights=True, the weights are all the same in eval mode (but not in train mode).
        # This happens for trained models, not randomly initialized ones.
        # Thus, we do not use dropout in nn.MultiheadAttention but apply it manually.
            
        attn_output, _ = self.attn(x, x, x, need_weights=False)
        
        attn_output = self.dropout(attn_output)

        x = attn_output + x

        if x.isnan().any():
            raise ValueError('nan in attn output')
        x = self.ff(x)

        return x