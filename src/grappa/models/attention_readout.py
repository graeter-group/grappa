import torch
from torch import nn
from .readout import skippedLinear
import numpy as np
from grappa.models.final_layer import ToPositive, ToRange
from typing import Union, List, Tuple
import copy

class FeedForwardLayer(nn.Module):
    def __init__(self, in_feats, hidden_feats, activation=nn.ELU()):
        super().__init__()
        self.linear1 = nn.Linear(in_feats, hidden_feats)
        self.linear2 = nn.Linear(hidden_feats, in_feats)
        self.activation = activation

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))


class DottedAttentionWithLayerNorm(nn.Module):
    def __init__(self, n_feats, num_heads, hidden_feats, layer_norm=True, dropout=0.0):
        super().__init__()

        self.layer_norm = layer_norm
        if self.layer_norm:
            self.norm1 = nn.LayerNorm(n_feats)
            self.norm2 = nn.LayerNorm(n_feats)

        self.attn = nn.MultiheadAttention(n_feats, num_heads, dropout=dropout)

        self.ff = FeedForwardLayer(n_feats, hidden_feats)

        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.layer_norm:
            x = self.norm1(x)
        
        attn_output, _ = self.attn(x, x, x, need_weights=False)
        attn_output += x
        
        if self.layer_norm:
            attn_output = self.norm2(attn_output)
        
        x = self.ff(attn_output)

        x = self.ff_dropout(x)

        x += attn_output

        return x


class PermutationEquivariantTransformer(torch.nn.Module):
    """
    Class that implements a mapping from input vectors of the form
    (n_seq, n_batch, n_feats) to output vectors of the shape (n_seq', n_batch, n_feats') where the output is permutation equivariant with respect to given permutations. (In a usual attention layer without positional encoding, the output is permutation equivariant wrt all permutations). Currently n_seq' can be 1 (invariance) or n_seq (equivariance in the same perm. group representation as the input).
    Having an invariant output can be achieved by:

        - Keeping full permutation equivariance until the last layer and then averaging psi(x0, ..., xn) over permuted versions to an output x_out = 1/n * sum_perm psi(perm(x))

        - Optionally: Giving the n_seq feature vectors a positional encoding that is invariant (only) under the given permutation, e.g. if the symmetry is (x0, x1, x2) -> (x2, x1, x0), we can use the positional encoding x0 = [x0, 0] and x1 = [x1, 1]. Then apply the strategy above.

    We can generalize this by enforcing certain transformation behaviour out(p(x)) = f(out(x), p) instead of invariance.
    This can be e.g. antisymmetry under certain permutations. For this, only symmetry or antisymmetry is implemented as of now.

    The class implements this in the following way:
    - The input is a tensor of shape (n_seq, n_batch, n_feats)
    - Then we perform attentional layers as in the transformer architecture.
    - The output of this is permutation equivariant
    - Then we apply a reducer, which is dense NN that maps from (n_batch, n_seq*n_feats) to (n_batch, n_feats_out)
    - The output of this is permutation invariant wrt given permutations because we sum over the n_seq dimension: x_out = 1/n * sum_perm psi(perm(x))
    """
    def __init__(self, n_feats, n_heads, hidden_feats, n_layers, out_feats, permutations:Union[np.ndarray, torch.Tensor], n_seq_out=1, layer_norm=True, dropout=0.0, reducer_layers=1, reducer_hidden_feats=None, permutation_prefactors=None, positional_encoding:Union[np.ndarray, torch.Tensor, bool]=None):
        """
        Let n_seq = len(permutations)
        Positional encoding:
            torch tensor of shape (n_seq, n_encode) that is invariant under the subset of permutations which we want our output to be invariant (and our layers to be equivariant) to.
            can be generated automatically if the permutations are
                {+(0,1,2), +(2,1,0)} ["angle symmetry"] -> encoding = [[0],[1],[0]]
            or  {+(0,1,2,3), +(3,2,1,0), -(0,2,1,3), -(3,1,2,0)} ["torsion symmetry"] -> encoding = [[0],[1],[1],[0]]
        """
        super().__init__()
        self.n_layers = n_layers

        self.n_feats = n_feats
        self.out_feats = out_feats
        
        self.n_seq_out = n_seq_out
        assert self.n_seq_out==1, "Currently only n_seq_out=1 is supported."


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
                        #print("bond_encoding")
                    
                    # angle case:
                    elif permutations.shape[1] == 3 and all([p.tolist() in [[0,1,2], [2,1,0]] for p in permutations]):
                        positional_encoding_ = torch.tensor([[0],[1],[0]], dtype=torch.float32)
                        #print("angle_encoding")
                    
                    # torsion case:
                    elif permutations.shape[1] == 4 and all([p.tolist() in [[0,1,2,3], [3,2,1,0], [0,2,1,3], [3,1,2,0]] for p in permutations]):
                        positional_encoding_ = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)
                        #print("torsion_encoding")

                    else:
                        positional_encoding_ = None

        if not positional_encoding_ is None:
            self.register_buffer("positional_encoding", positional_encoding_)
            self.n_feats = self.n_feats + self.positional_encoding.shape[1]
        else:
            self.positional_encoding = None


        self.transformer = nn.Sequential(*[
            DottedAttentionWithLayerNorm(self.n_feats, n_heads, hidden_feats, layer_norm=layer_norm, dropout=dropout)
            for _ in range(n_layers)
        ])

        if reducer_hidden_feats is None:
            reducer_hidden_feats = self.n_feats
        
        assert reducer_layers >= 1, "reducer_layers must be >= 1"

        # use no dropout in the reducer since this will destroy the permutation equivariance
        if reducer_layers == 1:
            self.reducer = nn.Linear(self.n_feats * self.n_seq, out_feats)
        else:
            self.reducer = nn.Sequential(*[
                skippedLinear(self.n_feats*self.n_seq, reducer_hidden_feats, activation=nn.ELU()),
                *[skippedLinear(reducer_hidden_feats, reducer_hidden_feats, activation=nn.ELU()) for _ in range(reducer_layers-2)],
                skippedLinear(reducer_hidden_feats, out_feats)
            ])


        ### Permutation prefactors:
        permutation_prefactors_ = permutation_prefactors
        
        if not permutation_prefactors is None:
            if not isinstance(permutation_prefactors, torch.Tensor):
                permutation_prefactors_ = torch.tensor(permutation_prefactors, dtype=torch.float32)
            
            assert len(permutation_prefactors) == self.n_perm, f"permutation_prefactors must have length {self.n_perm} but has length {len(permutation_prefactors)}"
            assert len(permutation_prefactors_.shape) == 1, "permutation_prefactors must be a 1d array"
            
            permutation_prefactors_ = permutation_prefactors_.view(self.n_perm, 1, 1) # to be multiplied with the output of the reducer which has shape (n_perm, n_batch, n_feats_out)


        self.register_buffer("permutation_prefactors", permutation_prefactors_)
        self.register_buffer("permutations", permutations_)
        

    def forward(self, x):
        """
        The input is a tensor of shape (n_seq, n_batch, n_feats). This is fed to a permutation equivariant transformer and then to a permutation invariant reducer.
        """
        assert self.n_seq == x.shape[0], f"x.shape[0] must be {self.n_seq} but is {x.shape[0]}"
        
        if not self.positional_encoding is None:
            # concat the positional encoding (shape: (n_seq, n_encode_feat)) to the feature dim of the input (shape: (n_seq, n_batch, n_feats))).

            # first, repeat along the batch dimension to obtain the shape (n_seq, n_batch, n_encode_feat)
            pos_encoding_repeated = self.positional_encoding.unsqueeze(1).repeat(1, x.shape[1], 1)

            # then concat along the feature dimension:
            x = torch.cat([x, pos_encoding_repeated], dim=-1)


        assert self.n_feats == x.shape[-1], f"n_feats must be {self.n_feats} but is {x.shape[-1]}"

        x = self.transformer(x)

        assert self.n_feats == x.shape[-1], f"n_feats must be {self.n_feats} but is {x.shape[-1]}"

        # Reducer:
        x_permuted = torch.stack([x[p] for p in self.permutations], dim=0)
        # x_permuted has shape (n_perm, n_seq, n_batch, n_feats)
        
        n_batch = x_permuted.shape[2]

        # now we want to bring this into the shape (n_batch*n_perm, n_seq*n_feats) to apply the reducer for all permuted versions in a vectorized way
        x_permuted = x_permuted.transpose(1,2)

        # now, x_permuted has the shape (n_perm, n_batch, n_seq, n_feats)
        x_permuted = x_permuted.contiguous().view(self.n_perm, n_batch, self.n_seq * self.n_feats)
        x_permuted = x_permuted.view(self.n_perm * n_batch, self.n_seq * self.n_feats)


        # and then apply the reducer:
        x_permuted = self.reducer(x_permuted)

        # now, x_permuted has the shape (n_batch*n_perm, n_feats_out)
        # we bring x_permuted back in shape (n_perm, n_batch, n_feats_out)
        x_permuted = x_permuted.view(self.n_perm, n_batch, self.out_feats)

        if not self.permutation_prefactors is None:
            x_permuted = x_permuted * self.permutation_prefactors

        # then we sum over the n_perm dimension to get the final output:
        x_permuted = x_permuted.sum(dim=0)

        return x_permuted


class RepProjector(torch.nn.Module):
    """
    This Layer takes a graph with node representation (num_nodes, feat_dim), passes it through one MLP layer and returns a stack of dim_tupel node feature vectors. The output thus has shape (dim_tupel, num_tuples, out_feat_dim).
    The graph must have node featires stored at g.nodes["n1"].data["h"] and tuple indices at g.nodes[f"n{dim_tupel}"].data["idxs"].
    """
    def __init__(self, dim_tupel, in_feats, out_feats, dropout, improper:bool=False) -> None:
        super().__init__()
        self.dim_tupel = dim_tupel
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_feats, out_feats),
            torch.nn.ELU(),
            torch.nn.Dropout(dropout)
        )

        self.improper = improper

    def forward(self, g):
        """
        This Layer takes a graph with node representation (num_nodes, feat_dim), passes it through one MLP layer and returns a stack of dim_tupel node feature vectors. The output thus has shape (dim_tupel, num_tuples, out_feat_dim).
        The graph must have node featires stored at g.nodes["n1"].data["h"] and tuple indices at g.nodes[f"n{dim_tupel}"].data["idxs"].
        """
        atom_feats = g.nodes["n1"].data["h"]
        atom_feats = self.mlp(atom_feats)

        if not self.improper:
            pairs = g.nodes[f"n{self.dim_tupel}"].data["idxs"]
        else:
            pairs = g.nodes[f"n{self.dim_tupel}_improper"].data["idxs"]


        try:
            # this has the shape num_pairs, 2, rep_dim
            tuples = atom_feats[pairs]
        except IndexError as err:
            err.message += f"\nIt might be that g.nodes['n{self.dim_tupel}'].data['idxs'] has the wrong datatype. It should be a long, byte or bool but is {pairs.dtype}"

        # transform the input to the shape 2, num_pairs, rep_dim
        tuples = tuples.transpose(0,1).contiguous()
        
        return tuples


class WriteBondParameters(torch.nn.Module):
    """
    Layer that takes as input the output of the representation layer and writes the bond parameters into the graph enforcing the permutation symmetry.
    """
    def __init__(self, rep_feats, between_feats, suffix="", stat_dict=None, n_att=2, n_heads=8, dense_layers=2, dropout=0., layer_norm=True, reducer_feats=None, attention_hidden_feats=None, positional_encoding=True):
        super().__init__()

        assert not stat_dict is None
        k_mean=stat_dict["mean"]["n2_k"].item()
        k_std=stat_dict["std"]["n2_k"].item()
        eq_mean=stat_dict["mean"]["n2_eq"].item()
        eq_std=stat_dict["std"]["n2_eq"].item()

        self.suffix = suffix

        # single layer dense nn to project the representation to the between_feats dimension
        # each interaction type has its own projector
        self.rep_projector = RepProjector(dim_tupel=2, in_feats=rep_feats, out_feats=between_feats, dropout=dropout)

        if reducer_feats is None:
            reducer_feats = between_feats
        if attention_hidden_feats is None:
            attention_hidden_feats = 4*between_feats


        self.bond_model = PermutationEquivariantTransformer(n_feats=between_feats, n_heads=n_heads, hidden_feats=attention_hidden_feats, n_layers=n_att, out_feats=2, permutations=torch.tensor([[0,1],[1,0]], dtype=torch.int32), layer_norm=layer_norm, dropout=dropout, reducer_layers=dense_layers, reducer_hidden_feats=reducer_feats, positional_encoding=copy.deepcopy(positional_encoding))
        
        self.to_k = ToPositive(mean=k_mean, std=k_std, min_=0)
        self.to_eq = ToPositive(mean=eq_mean, std=eq_std)


    def forward(self, g):

        # build a tuple of feature vectors from the representation.
        # inputs will have shape 2, num_pairs, rep_dim
        inputs = self.rep_projector(g)
        
        coeffs = self.bond_model(inputs)

        coeffs[:,0] = self.to_eq(coeffs[:,0])
        coeffs[:,1] = self.to_k(coeffs[:,1])

        g.nodes["n2"].data["eq"+self.suffix] = coeffs[:,0].unsqueeze(dim=-1)
        g.nodes["n2"].data["k"+self.suffix] = coeffs[:,1].unsqueeze(dim=-1)

        return g



class WriteAngleParameters(torch.nn.Module):
    """
    Layer that takes as input the output of the representation layer and writes the torsion parameters into the graph enforcing the permutation symmetry by a symmetrizer network \psi:
    symmetric_feature = \psi(xi,xj,xk) + \psi(xk,xj,xi)
    out = \phi(symmetric_feature)

    The default mean and std deviation of the dataset can be overwritten by handing over a stat_dict with stat_dict['mean'/'std'][level_name] = value
    """

    def __init__(self, rep_feats, between_feats, suffix="", stat_dict=None, n_att=2, n_heads=8, dense_layers=2, dropout=0., layer_norm=True, reducer_feats=None, attention_hidden_feats=None, positional_encoding=True):
        super().__init__()

        assert not stat_dict is None
        k_mean=stat_dict["mean"]["n3_k"].item()
        k_std=stat_dict["std"]["n3_k"].item()
        eq_mean=stat_dict["mean"]["n3_eq"].item()
        eq_std=stat_dict["std"]["n3_eq"].item()

        self.suffix = suffix

        # single layer dense nn to project the representation to the between_feats dimension
        # each interaction type has its own projector
        self.rep_projector = RepProjector(dim_tupel=3, in_feats=rep_feats, out_feats=between_feats, dropout=dropout)


        if reducer_feats is None:
            reducer_feats = between_feats
        if attention_hidden_feats is None:
            attention_hidden_feats = 4*between_feats


        self.angle_model = PermutationEquivariantTransformer(n_feats=between_feats, n_heads=n_heads, hidden_feats=attention_hidden_feats, n_layers=n_att, out_feats=2, permutations=torch.tensor([[0,1,2],[2,1,0]], dtype=torch.int32), layer_norm=layer_norm, dropout=dropout, reducer_layers=dense_layers, reducer_hidden_feats=reducer_feats, positional_encoding=copy.deepcopy(positional_encoding))
        

        self.to_k = ToPositive(mean=k_mean, std=k_std, min_=0)
        self.to_eq = ToRange(max_=torch.pi, std=eq_std)


    def forward(self, g):

        if not "n3" in g.ntypes:
            return g

        # transform the input to the shape 3, num_pairs, rep_dim
        inputs = self.rep_projector(g)
        
        coeffs = self.angle_model(inputs)

        coeffs[:,0] = self.to_eq(coeffs[:,0])
        coeffs[:,1] = self.to_k(coeffs[:,1])

        g.nodes["n3"].data["eq"+self.suffix] = coeffs[:,0].unsqueeze(dim=-1)
        g.nodes["n3"].data["k"+self.suffix] = coeffs[:,1].unsqueeze(dim=-1)

        return g





class WriteTorsionParameters(torch.nn.Module):
    """
    Multiply with a final binary softmax (ie sigmoid) layer, allowing the network to be more accurate around zero.
    This is still experimental. 

    GatedTorsion layer that takes as input the output of the representation layer and writes the torsion parameters into the graph enforcing the permutation symmetry by a symmetrizer network \psi:
    symmetric_feature = \sum_symmetric_permutations \psi(xi,xj,xk,xl)
    out = \phi(symmetric_feature) * sigmoid(\chi(symmetric_feature))

    \phi is a dense neural network and \chi is a classifier network, predicting a gate score of "how nonzero" the torsion parameter should be.

    For Impropers:
        Enforce antisymmetry under the permutation. Note that each improper torsion occurs thrice in the graph to ensure symmetry of the energy under permutation of the outer atoms. (To reduce the number of parameters that are to be learned, we can use the antisymmetry of the dihedral under the permutation of the outer atoms. If also k is antisymmetric, we have found an energy function that is symmetric under the permutation of the outer atoms.)

    For Propers:
        Enforce symmetry of the energy function by enforcing symmetry of k under [3,2,1,0] and using symmetry of the dihedral angle under this permutation.
    """
    def __init__(self, rep_feats, between_feats, suffix="", n_periodicity=None, magnitude=0.001, improper=False, n_att=2, n_heads=8, dense_layers=2, dropout=0., layer_norm=True, reducer_feats=None, attention_hidden_feats=None, stat_dict=None, positional_encoding=True):
        super().__init__()

        if n_periodicity is None:
            n_periodicity = 6
            if improper:
                n_periodicity = 3

        if not improper:
            k_mean=stat_dict["mean"]["n4_k"]
            k_std=stat_dict["std"]["n4_k"]
        else:
            if not "n4_improper_k" in stat_dict["mean"]:
                k_mean = torch.zeros(n_periodicity)
                k_std = torch.ones(n_periodicity)
            else:
                k_mean = stat_dict["mean"]["n4_improper_k"]
                k_std = stat_dict["std"]["n4_improper_k"]
                
                if len(k_mean) != n_periodicity:
                    raise ValueError(f"n_periodicity is {n_periodicity} but the stat_dict contains {len(k_mean)} values for the mean of the improper torsion parameters.")
                
                if len(k_std) != n_periodicity:
                    raise ValueError(f"n_periodicity is {n_periodicity} but the stat_dict contains {len(k_std)} values for the std of the improper torsion parameters.")

        k_mean = k_mean.unsqueeze(dim=0)
        k_std = k_std.unsqueeze(dim=0)
        self.register_buffer("k_mean", k_mean)
        self.register_buffer("k_std", k_std)

        self.suffix = suffix

        self.wrong_symmetry = False

        self.magnitude = magnitude
        self.improper = improper


        # single layer dense nn to project the representation to the between_feats dimension
        # each interaction type has its own projector
        self.rep_projector = RepProjector(dim_tupel=4, in_feats=rep_feats, out_feats=between_feats, dropout=dropout, improper=improper)


        if reducer_feats is None:
            reducer_feats = between_feats

        if attention_hidden_feats is None:
            attention_hidden_feats = 4*between_feats

        
        if not improper:
            perms = torch.tensor([[0,1,2,3],[3,2,1,0]], dtype=torch.int32)
            # enforce symmetry under the permutation
            prefactors = torch.tensor([1,1], dtype=torch.float32)

        else:
            # enforce antisymmetry under the permutation. Note that each improper torsion occurs thrice in the graph to ensure symmetry of the energy under permutation of the outer atoms. (To reduce the number of parameters that are to be learned, we can use the antisymmetry of the dihedral under the permutation of the outer atoms. If also k is antisymmetric, we have found an energy function that is symmetric under the permutation of the outer atoms.)
            perms = torch.tensor([[0,1,2,3],[3,1,2,0]], dtype=torch.int32)
            prefactors = torch.tensor([1,-1], dtype=torch.float32)

        self.torsion_model = PermutationEquivariantTransformer(n_feats=between_feats, n_heads=n_heads, hidden_feats=attention_hidden_feats, n_layers=n_att, out_feats=n_periodicity, permutations=perms, layer_norm=layer_norm, dropout=dropout, reducer_layers=dense_layers, reducer_hidden_feats=reducer_feats, permutation_prefactors=prefactors, positional_encoding=copy.deepcopy(positional_encoding))


    def forward(self, g):
        # bonds:
        # every index pair appears twice, with permuted end points therefore 0.5 factor in energy calculation and sum is done automatically
        level = "n4"
        if self.improper:
            level += "_improper"
        
        if not level in g.ntypes:
            return g

        
        # transform the input to the shape 4, num_pairs, rep_dim
        inputs = self.rep_projector(g)

        coeffs = self.torsion_model(inputs)

        coeffs = coeffs*self.k_std + self.k_mean

        g.nodes[level].data["k"+self.suffix] = coeffs

        return g


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x