"""
Module defining the PermutationEquivariantTransformer class and its helper classes such as a class for skip-connected feed-forward layers, for multihead attention, and for symmetrisation layers.
"""

import torch
from torch import nn
import numpy as np
from typing import Union, List, Tuple
import copy
from .network_utils import FeedForwardLayer, DottedAttWithMLP


class SymmetrisedTransformer(nn.Module):
    """
    Chains a GrappaTransformer with a Symmetriser.

    ----------

    Class that implements a mapping from input vectors of the form
    (n_seq, n_batch, n_feats) to output vectors of the shape (n_seq', n_batch, n_feats') where the output is permutation invariant with respect to given permutations. This is done using a GrappaTransformer, which is equivariant wrt the given permutations, followed by a Symmetriser, which is invariant under these permutations.
    If positional encodings are given, the output of the GrappaTransformer is only equivariant wrt the given permutations, as opposed to a vanilla transformer, which is equivariant wrt all permutations. This enables the model to use the information on which permutations the output is invariant under already in the early layers of the transformer, which cannot be achieved by symmetric pooling at the very end of the network.

    ----------
    Parameters:
    ----------
        n_feats (int): Number of features in the input.
        n_heads (int): Number of attention heads in the transformer.
        hidden_feats (int): Number of hidden features in the transformer.
        n_layers (int): Number of layers in the transformer.
        out_feats (int): Number of output features.
        permutations (Union[np.ndarray, torch.Tensor]): Array or tensor of permutations for equivariance.
        layer_norm (bool): Flag to apply layer normalization. Defaults to True.
        dropout (float): Dropout rate in the transformer. Defaults to 0.0.
        symmetriser_layers (int): Number of layers in the symmetriser. Defaults to 1.
        symmetriser_hidden_feats (int): Number of hidden features in the symmetriser. Defaults to None, in which case it is set to `n_feats`.
        permutation_prefactors (Union[np.ndarray, torch.Tensor]): Array or tensor of prefactors for the permutations. Defaults to None, in which case all prefactors are set to 1.
        positional_encoding (Union[np.ndarray, torch.Tensor, bool]): Positional encoding tensor of shape (n_seq, n_encode_feats), or a flag to automatically generate it. This must be invariant under the permutations given in `permutations`. Defaults to None.

    """
    def __init__(self, n_feats, n_heads, hidden_feats, n_layers, out_feats, permutations:Union[np.ndarray, torch.Tensor], layer_norm=True, dropout=0.0, symmetriser_layers=1, symmetriser_hidden_feats=None, permutation_prefactors=None, positional_encoding:Union[np.ndarray, torch.Tensor, bool]=None):
        super().__init__()

        self.n_feats = n_feats

        self.layer_norm = layer_norm


        if n_layers > 0:
            self.grappa_transformer = GrappaTransformer(n_feats=n_feats, n_heads=n_heads, hidden_feats=hidden_feats, n_layers=n_layers, out_feats=n_feats, permutations=permutations, layer_norm=layer_norm, dropout=dropout, positional_encoding=positional_encoding)

            # the transformer 
            self.trafo_out_feats = n_feats + self.grappa_transformer.positional_encoding.shape[1] if not self.grappa_transformer.positional_encoding is None else n_feats

        else:
            self.grappa_transformer = None
            self.trafo_out_feats = n_feats

        assert symmetriser_layers >= 1, "symmetriser_layers must be >= 1"

        if layer_norm:
            self.norm = nn.LayerNorm(self.trafo_out_feats)

        self.symmetriser = Symmetriser(in_feats=self.trafo_out_feats, out_feats=out_feats, permutations=permutations, permutation_prefactors=permutation_prefactors, hidden_feats=symmetriser_hidden_feats, n_layers=symmetriser_layers, layer_norm=layer_norm)



    def forward(self, x):
        """
        The input is a tensor of shape (n_seq, n_batch, n_feats). This is fed to a permutation equivariant GrappaTransformer and then to a permutation invariant Symmetriser.
        """

        if not self.grappa_transformer is None:
            add_feats = self.trafo_out_feats - self.n_feats
            if add_feats > 0:
                x = self.grappa_transformer(x) + torch.cat([x, torch.zeros(x.size(0), x.size(1), add_feats, device=x.device)], dim=-1)
            else:                
                x = self.grappa_transformer(x) + x

        if self.layer_norm:
            x = self.norm(x)

        # send x through the symmetriser:
        x = self.symmetriser(x)

        return x




class GrappaTransformer(nn.Module):
    """
    Class that implements a mapping from n_seq input vectors, i.e. an input of the form (n_seq, n_batch, n_feats) to n_seq output vectors, i.e. the output shape is (n_seq, n_batch, out_feats) where the output is permutation equivariant.
    If positional encodings are given, the output of the GrappaTransformer is equivariant only with respect to the given permutations, as opposed to a vanilla transformer, which is equivariant wrt all permutations. This enables the model to use the information on which permutations the output is invariant under already in the early layers of the transformer, which cannot be achieved by symmetric pooling at the very end of the network.

    ----------
    Parameters:
    ----------
        n_feats (int): Number of features in the input, which must have shape (n_seq, n_batch, n_feats). The number of input features for the attention layers is the sum of n_feats and the features of the positional encoding. Thus, n_feats must be chosen such that this sum is a multiple of the number of heads
        n_heads (int): Number of attention heads in the transformer.
        hidden_feats (int): Number of hidden features in the transformer.
        n_layers (int): Number of layers in the transformer.
        out_feats (int): Number of output features without positional encoding. With positional encoding, the output has shape (n_seq, n_batch, out_feats + n_encode_feats).
        permutations (Union[np.ndarray, torch.Tensor]): Array or tensor of permutations for equivariance of shape (n_perm, n_seq).
        layer_norm (bool): Flag to apply layer normalization. Defaults to True.
        dropout (float): Dropout rate in the transformer. Defaults to 0.0.
        positional_encoding (Union[np.ndarray, torch.Tensor, bool]): Positional encoding tensor, or a flag to automatically generate it. This must be invariant under the permutations given in `permutations`. Defaults to None. Shape must be (n_seq, n_encode_feats).

    Positional encoding can be generated automatically if the permutations are
    {+(0,1,2), +(2,1,0)} ["angle symmetry"] -> encoding = [[0],[1],[0]]
    or {+(0,1,2,3), +(3,2,1,0),} ["torsion symmetry"] -> encoding = [[0],[1],[1],[0]]

    ----------
    Returns:
    ----------
        torch.Tensor: Output tensor of shape (n_seq', n_batch, n_feats') where n_seq' is either 1 (invariant output) or n_seq (equivariant output).
    """
    def __init__(self, n_feats, n_heads, hidden_feats, n_layers, out_feats, permutations:Union[np.ndarray, torch.Tensor], layer_norm=True, dropout=0.0, positional_encoding:Union[np.ndarray, torch.Tensor, bool]=None):
        super().__init__()

        assert out_feats == n_feats, f"out_feats != n_feats not implemented. out_feats: {out_feats}, n_feats: {n_feats}"

        self.n_layers = n_layers

        self.n_feats = n_feats

        self.norm = nn.LayerNorm(n_feats)

        self.init_positional_encoding(positional_encoding, permutations=permutations)

        self.n_seq = permutations.shape[1]

        if not self.positional_encoding is None:
            self.n_feats = n_feats + self.positional_encoding.shape[1]

        if not (self.n_feats / n_heads).is_integer():
            raise ValueError(f'The number of input features cannot be divided by the number of heads: Number of input features: {self.n_feats} = {n_feats} + {self.positional_encoding.shape[1]} (from positional encoding). Number of heads: {n_heads}')

        # the transformer is a sequence of n_layers with attentional and mlp layers
        self.transformer = nn.Sequential(*[
            DottedAttWithMLP(n_feats=self.n_feats, num_heads=n_heads, hidden_feats=hidden_feats, layer_norm=layer_norm, dropout=dropout)
            for i in range(n_layers)
        ])
        

    def forward(self, x):
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
        x = self.transformer(x) + x

        assert self.n_feats == x.shape[-1], f"n_feats must be {self.n_feats} but is {x.shape[-1]}"

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
                    if permutations.shape[1] == 2:
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
            self.register_buffer("positional_encoding", positional_encoding_.float())
        else:
            self.positional_encoding = None


    

class Symmetriser(torch.nn.Module):
    """
    Implements a symmetrisation layer that takes as input a tensor of shape (n_seq, n_batch, n_feats) and returns a tensor of shape (n_batch, n_out_feats) that is symmetric under the given permutations along the n_seq dimension. This is achieved by applying a dense NN to all permuted versions of the input and then summing over the n_seq dimension.

    ----------
    Parameters:
    ----------
        in_feats (int): Number of features in the input, which must have shape (n_seq, n_batch, in_feats).
        out_feats (int): The number of features in the output.
        permutations (torch.Tensor): A tensor of shape (n_perm, n_seq) containing the permutations to be applied to the input.
        permutation_prefactors (torch.Tensor): A tensor of shape (n_perm) containing the prefactors of the permutations.
        hidden_feats (int, optional): The number of features in the hidden layer of the feed-forward network. Defaults to `in_feats`.
        n_layers (int, optional): The number of hidden layers in the feed-forward network. Defaults to 1.
        skip (bool): If True, adds a skip connection between the input and output of the feed-forward network. Defaults to False.
        layer_norm (bool): If True, applies layer normalization after each major step. Defaults to True.

    Note: Dropout is not available in this layer since it would destroy the permutation equivariance.
    """
    def __init__(self, in_feats:int, out_feats:int, permutations:torch.tensor, permutation_prefactors:torch.tensor=None, hidden_feats:int=None, n_layers:int=1, skip:bool=True, layer_norm:bool=True):
        super().__init__()

        assert n_layers >= 1, "n_layers must be >= 1"

        if hidden_feats is None:
            hidden_feats = in_feats

        if permutation_prefactors is None:
            # if no prefactors are given, we assume that every permutation has prefactor 1:
            permutation_prefactors = torch.ones(permutations.shape[0], dtype=torch.float32)

        self.init_permutations(permutations, permutation_prefactors)

        self.n_seq = permutations.shape[1]

        self.n_feats = in_feats
        self.out_feats = out_feats

        self.mlp = nn.Sequential(
            FeedForwardLayer(in_feats=self.n_feats*self.n_seq, out_feats=hidden_feats if n_layers > 1 else out_feats, hidden_feats=hidden_feats, skip=False, layer_norm=layer_norm, dropout=0.),
            *[
            FeedForwardLayer(in_feats=hidden_feats, out_feats=hidden_feats if i!=n_layers-1 else out_feats, hidden_feats=hidden_feats, skip=skip if i!=n_layers-1 else False, layer_norm=layer_norm, dropout=0.)
            for i in range(1, n_layers)
            ]
        )

    def forward(self, x):

        assert len(x.shape) == 3, "x must have shape (n_seq, n_batch, n_feats)"

        assert x.shape[0] == self.n_seq, f"x.shape[0] must be {self.n_seq} but is {x.shape[0]}"
        assert x.shape[2] == self.n_feats, f"x.shape[2] must be {self.n_feats} but is {x.shape[2]}"

        # symmetriser: Now enforce permutation invariance by summing over the n_seq dimension
        # First create a vector with all permuted versions of x:
        x_permuted = torch.stack([x[p] for p in self.permutations], dim=0)

        # x_permuted has shape (n_perm, n_seq, n_batch, n_feats)
        
        n_batch = x_permuted.shape[2]

        # now we want to bring this into the shape (n_batch*n_perm, n_seq*n_feats) to apply the symmetriser for all permuted versions in a vectorized way
        x_permuted = x_permuted.transpose(1,2)

        # now, x_permuted has the shape (n_perm, n_batch, n_seq, n_feats)
        x_permuted = x_permuted.contiguous().view(self.n_perm, n_batch, self.n_seq * self.n_feats)

        # we can treat the permuted versions like a batch of size n_perm*n_batch and remember which one belong toghether (x_permuted.view(self.n_perm, n_batch, self.out_feats) will revert this)
        x_permuted = x_permuted.view(self.n_perm * n_batch, self.n_seq * self.n_feats)

        # and then apply the neural network:
        x_permuted = self.mlp(x_permuted)

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




