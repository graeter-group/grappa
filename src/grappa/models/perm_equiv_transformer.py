"""
Module defining the PermutationEquivariantTransformer class and its helper classes such as a class for skip-connected feed-forward layers, for multihead attention, and for symmetrisation layers.
"""

import torch
from torch import nn
import numpy as np
from typing import Union, List, Tuple
import copy

class FeedForwardLayer(nn.Module):
    """
    Simple MLP with one hidden layer and, optionally, a skip connection, dropout and layer normalization.
    Parameters:
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

        self.linear1 = nn.Linear(in_feats, hidden_feats)
        self.linear2 = nn.Linear(hidden_feats, out_feats)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.skip = skip
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.norm1 = nn.LayerNorm(in_feats)

    def forward(self, x):
        if self.layer_norm:
            x = self.norm1(x)
        x_out = self.linear1(x)
        x_out = self.activation(x_out)
        x_out = self.linear2(x_out)
        x_out = self.dropout(x_out)
        if self.skip:
            x_out = x_out + x
        return x_out


class DottedAttWithMLP(nn.Module):
    """
    Implements a transformer block with a multihead self-attention layer followed by a feed-forward layer. 
    Parameters:
        n_feats (int): The number of features in the input and output.
        num_heads (int): The number of attention heads in the multihead attention layer.
        hidden_feats (int, optional): The number of features in the hidden layer of the feed-forward network.
                                     Defaults to 4 times `n_feats`, following the original transformer paper.
        layer_norm (bool): If True, applies layer normalization after each major step. Defaults to True.
        dropout (float): Dropout rate applied after the attention layer and in the feed-forward network. Defaults to 0.

        
    The process involves the following sequential operations:
        - Calculate attention weights for each node using a subset of features (n_feats // n_heads) per head.
        - Update node features using the calculated attention weights.
        - Apply dropout to the updated features.
        - Add the original input features (residual connection).
        - Apply layer normalization (optional).
        - Pass the result through a feed-forward network with one hidden layer.
        - Apply dropout to the output of the feed-forward network.
        - Add the output of the attention layer (residual connection).

    Note:
        The input and output feature dimension (n_feats) is the same.
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
        self.attn = nn.MultiheadAttention(n_feats, num_heads, dropout=dropout)

        self.ff = FeedForwardLayer(n_feats, hidden_feats, out_feats=n_feats, dropout=dropout, skip=True, layer_norm=layer_norm)


    def forward(self, x):
        if self.layer_norm:
            x = self.norm1(x)
        
        attn_output, _ = self.attn(x, x, x, need_weights=False)
        x = attn_output + x
        
        x = self.ff(x)

        return x
    

class Symmetriser(torch.nn.Module):
    """
    Implements a symmetrisation layer that takes as input a tensor of shape (n_seq, n_batch, n_feats) and returns a tensor of shape (n_batch, n_out_feats) that is symmetric under the given permutations along the n_seq dimension.
    Note: Dropout is not available in this layer since it would destroy the permutation equivariance.
    """
    def __init__(self, in_feats:int, out_feats:int, permutations:torch.tensor, permutation_prefactors:torch.tensor, hidden_feats:int=None, n_layers:int=1, skip:bool=True, layer_norm:bool=True):
        super().__init__()
        if hidden_feats is None:
            hidden_feats = in_feats

        self.pre_linear = nn.Linear(in_feats, hidden_feats)
        self.post_linear = nn.Linear(hidden_feats, out_feats)

        self.init_permutations(permutations, permutation_prefactors)

        self.n_seq = permutations.shape[1]

        self.mlp = nn.Sequential(*[
            FeedForwardLayer(in_feats=hidden_feats, out_feats=hidden_feats, hidden_feats=hidden_feats, skip=skip, layer_norm=layer_norm, dropout=0.)
            for _ in range(n_layers)
        ])

    def forward(self, x):

        # symmetriser: Now enforce permutation invariance by summing over the n_seq dimension
        # First create a vector with all permuted versions of x:
        x_permuted = torch.stack([x[p] for p in self.permutations], dim=0)
        # x_permuted has shape (n_perm, n_seq, n_batch, n_feats)
        
        n_batch = x_permuted.shape[2]

        # now we want to bring this into the shape (n_batch*n_perm, n_seq*n_feats) to apply the symmetriser for all permuted versions in a vectorized way
        x_permuted = x_permuted.transpose(1,2)

        # now, x_permuted has the shape (n_perm, n_batch, n_seq, n_feats)
        x_permuted = x_permuted.contiguous().view(self.n_perm, n_batch, self.n_seq * self.n_feats)
        x_permuted = x_permuted.view(self.n_perm * n_batch, self.n_seq * self.n_feats)


        # and then apply the symmetriser:
        x_permuted = self.symmetriser(x_permuted)

        # now, x_permuted has the shape (n_batch*n_perm, n_feats_out)
        # we bring x_permuted back in shape (n_perm, n_batch, n_feats_out)
        x_permuted = x_permuted.view(self.n_perm, n_batch, self.out_feats)

        if not self.permutation_prefactors is None:
            x_permuted = x_permuted * self.permutation_prefactors

        # then we sum over the n_perm dimension to get the final output:
        x_permuted = x_permuted.sum(dim=0)

        return x_permuted



    def init_permutations(self, permutations:Union[np.ndarray, torch.Tensor], permutation_prefactors:Union[np.ndarray, torch.Tensor]):
        """
        Helper function.
        """

        ### Permutations:

        assert len(permutations.shape) == 2, "permutations must be a 2d array"
        self.n_seq = permutations.shape[1]

        self.n_perm = permutations.shape[0]
        assert self.n_perm > 0, "permutations must have at least one row"

        if not isinstance(permutations, torch.Tensor):
            permutations_ = torch.tensor(permutations, dtype=torch.int32)
        else:
            permutations_ = permutations.int()

        # identity permutation must be included:
        assert torch.all(permutations_[0].int() == torch.arange(self.n_seq).int()), "permutations must include the identity permutation at the zeroth entry."


        ### Permutation prefactors:
        permutation_prefactors_ = permutation_prefactors

        if permutation_prefactors is None:
            # if no prefactors are given, we assume that every permutation has sign 1:
            permutation_prefactors_ = torch.ones(self.n_perm, dtype=torch.float32)
    
        if not isinstance(permutation_prefactors, torch.Tensor):
            permutation_prefactors_ = torch.tensor(permutation_prefactors, dtype=torch.float32)
        
        assert len(permutation_prefactors) == self.n_perm, f"permutation_prefactors must have length {self.n_perm} but has length {len(permutation_prefactors)}"
        assert len(permutation_prefactors_.shape) == 1, "permutation_prefactors must be a 1d array"
        
        permutation_prefactors_ = permutation_prefactors_.view(self.n_perm, 1, 1) # to be multiplied with the output of the symmetriser which has shape (n_perm, n_batch, n_feats_out)


        self.register_buffer("permutation_prefactors", permutation_prefactors_)
        self.register_buffer("permutations", permutations_)




class PermutationEquivariantTransformer(torch.nn.Module):
    """
    Class that implements a mapping from input vectors of the form
    (n_seq, n_batch, n_feats) to output vectors of the shape (n_seq', n_batch, n_feats') where the output is permutation equivariant with respect to given permutations. (In a usual attention layer without positional encoding, the output is permutation equivariant wrt all permutations). Currently n_seq' can be 1 (invariance) or n_seq (equivariance in the same perm. group representation as the input). This is controlled by the make_invariant flag.

    Having an invariant output can be achieved by:

        - Keeping full permutation equivariance until the last layer and then averaging psi(x0, ..., xn) over permuted versions to an output x_out = 1/n * sum_perm psi(perm(x))

        - Or using the information that the output is not invariant under all but only the given permutations:
        Giving the n_seq feature vectors a positional encoding that is invariant (only) under the given permutation, e.g. if the symmetry is (x0, x1, x2) -> (x2, x1, x0), we can use the positional encoding x0 = [x0, 0], x2 = [x2, 0] and x1 = [x1, 1]. Then apply the strategy above.

    We can generalize this by enforcing certain transformation behaviour out(p(x)) = f(out(x), p) instead of invariance.
    This can be e.g. antisymmetry under certain permutations. For this, only symmetry or antisymmetry is implemented as of now.

    The class implements this in the following way:
    - The input is a tensor of shape (n_seq, n_batch, n_feats)
    - Then we perform attentional layers as in the transformer architecture.
    - The output of this is permutation equivariant
    - Then we apply a symmetriser, which is dense NN that maps from (n_batch, n_seq*n_feats) to (n_batch, n_feats_out)
    - The output of this is permutation invariant wrt given permutations because we sum over the n_seq dimension: x_out = 1/n * sum_perm psi(perm(x))
    """
    def __init__(self, n_feats, n_heads, hidden_feats, n_layers, out_feats, permutations:Union[np.ndarray, torch.Tensor], layer_norm=True, dropout=0.0, symmetriser_layers=1, symmetriser_hidden_feats=None, permutation_prefactors=None, positional_encoding:Union[np.ndarray, torch.Tensor, bool]=None, make_invariant:bool=True):
        """
        Let n_seq = len(permutations)
        Positional encoding:
            torch tensor of shape (n_seq, n_encode) that is invariant under the subset of permutations which we want our output to be invariant (and our layers to be equivariant) to.
            can be generated automatically if the permutations are
                {+(0,1,2), +(2,1,0)} ["angle symmetry"] -> encoding = [[0],[1],[0]]
            or  {+(0,1,2,3), +(3,2,1,0),} ["torsion symmetry"] -> encoding = [[0],[1],[1],[0]]
        """
        super().__init__()
        self.n_layers = n_layers

        self.n_feats = n_feats
        self.out_feats = out_feats

        self.make_invariant = make_invariant

        if symmetriser_hidden_feats is None:
            symmetriser_hidden_feats = self.n_feats
        
        assert symmetriser_layers >= 1, "symmetriser_layers must be >= 1"

        self.symmetriser = Symmetriser(in_feats=self.n_feats, out_feats=self.out_feats, permutations=permutations, permutation_prefactors=permutation_prefactors, hidden_feats=symmetriser_hidden_feats, n_layers=symmetriser_layers, layer_norm=layer_norm)

        self.init_positional_encoding(positional_encoding, permutations=self.symmetriser.permutations)

        # the transformer is a sequence of n_layers attentional and mlp layers
        self.transformer = nn.Sequential(*[
            DottedAttWithMLP(self.n_feats, n_heads, hidden_feats, layer_norm=layer_norm, dropout=dropout)
            for i in range(n_layers)
        ])
        

    def forward(self, x):
        """
        The input is a tensor of shape (n_seq, n_batch, n_feats). This is fed to a permutation equivariant transformer and then to a permutation invariant symmetriser.
        """
        if not len(x.shape) == 3:
            raise ValueError(f"x must have shape (n_seq, n_batch, n_feats) but has shape {x.shape}.")
    
        if not self.n_seq == x.shape[0]:
            raise ValueError(f"x.shape[0] must be {self.n_seq} but the shape of x is {x.shape}.")
        
        if not self.positional_encoding is None:
            # concat the positional encoding (shape: (n_seq, n_encode_feat)) to the feature dim of the input (shape: (n_seq, n_batch, n_feats))).

            # first, repeat along the batch dimension to obtain the shape (n_seq, n_batch, n_encode_feat)
            pos_encoding_repeated = self.positional_encoding.unsqueeze(1).repeat(1, x.shape[1], 1)

            # then concat along the feature dimension:
            x = torch.cat([x, pos_encoding_repeated], dim=-1)


        assert self.n_feats == x.shape[-1], f"n_feats must be {self.n_feats} but is {x.shape[-1]}"

        # send x through the transformer:
        x = self.transformer(x)

        assert self.n_feats == x.shape[-1], f"n_feats must be {self.n_feats} but is {x.shape[-1]}"

        if not self.make_invariant:
            return x

        # send x through the symmetriser:
        x = self.symmetriser(x)

        return x



    def init_positional_encoding(self, positional_encoding:Union[np.ndarray, torch.Tensor, bool], permutations:Union[np.ndarray, torch.Tensor]):
        """
        Helper function.
        """

        ### Positional encoding:

        positional_encoding_ = None
        if not positional_encoding is None:
            if isinstance(positional_encoding, torch.Tensor):
                positional_encoding_ = copy.deepcopy(positional_encoding.float())
            elif isinstance(positional_encoding, np.ndarray):
                positional_encoding_ = copy.deepcopy(torch.tensor(positional_encoding, dtype=torch.float32))

            elif isinstance(positional_encoding, bool):
                if positional_encoding:
                    # bond case:
                    if self.permutations.shape[1] == 2:
                        positional_encoding_ = None # no positional encoding is needed, we have total symmetry (our subset of perms is all perms)
                    
                    # angle case:
                    elif permutations.shape[1] == 3 and all([p.tolist() in [[0,1,2], [2,1,0]] for p in permutations]):
                        positional_encoding_ = torch.tensor([[0],[1],[0]], dtype=torch.float32)
                    
                    # torsion case:
                    elif permutations.shape[1] == 4 and all([p.tolist() in [[0,1,2,3], [3,2,1,0], [0,2,1,3], [3,1,2,0]] for p in permutations]):
                        positional_encoding_ = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

                    else:
                        positional_encoding_ = None

        if not positional_encoding_ is None:
            self.register_buffer("positional_encoding", positional_encoding_)
            self.n_feats = self.n_feats + self.positional_encoding.shape[1]
        else:
            self.positional_encoding = None
