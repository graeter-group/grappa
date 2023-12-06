import torch
import dgl
from grappa.models.final_layer import ToPositive, ToRange
from grappa import constants
from typing import Union, List, Tuple
from grappa.models.perm_equiv_transformer import PermutationEquivariantTransformer
import copy

# NOTE: CONTINUE HERE
# then, write it all in the get_models or make a new get_models function. For torsion, make the gated version optional again, this would just require 6/3 more output values whose sigmoid is multiplied with the output of the vanilla model.

class RepProjector(torch.nn.Module):
    """
    This Layer takes a graph with node representation (num_nodes, feat_dim), passes it through one MLP layer and returns a stack of dim_tupel node feature vectors. The output thus has shape (dim_tupel, num_tuples, out_feat_dim).
    The graph must have node features stored at g.nodes["n1"].data["h"] and tuple indices at g.nodes[f"n{dim_tupel}"].data["idxs"].
    """
    def __init__(self, dim_tupel, in_feats, out_feats, improper:bool=False) -> None:
        super().__init__()
        self.dim_tupel = dim_tupel
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_feats, out_feats),
            torch.nn.ELU(),
        )

        self.improper = improper

    def forward(self, g):
        """
        This Layer takes a graph with node representation (num_nodes, feat_dim), passes it through one MLP layer and returns a stack of dim_tupel node feature vectors. The output thus has shape (dim_tupel, num_tuples, out_feat_dim).
        The graph must have node features stored at g.nodes["n1"].data["h"] and tuple indices at g.nodes[f"n{dim_tupel}"].data["idxs"].
        """
        atom_feats = g.nodes["n1"].data["h"]
        atom_feats = self.mlp(atom_feats)

        if not self.improper:
            pairs = g.nodes[f"n{self.dim_tupel}"].data["idxs"]
        else:
            pairs = g.nodes[f"n{self.dim_tupel}_improper"].data["idxs"]

        if len(pairs) == 0:
            return torch.zeros((self.dim_tupel, 0, atom_feats.shape[-1]), dtype=atom_feats.dtype, device=atom_feats.device)

        try:
            # this has the shape num_pairs, dim_tuple, rep_dim
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
    def __init__(self, rep_feats, between_feats, suffix="", stat_dict=None, n_att=2, n_heads=8, dense_layers=2, dropout=0., layer_norm=True, symmetriser_feats=None, attention_hidden_feats=None, positional_encoding=True):
        super().__init__()

        assert not stat_dict is None
        k_mean=stat_dict["mean"]["n2_k"].item()
        k_std=stat_dict["std"]["n2_k"].item()
        eq_mean=stat_dict["mean"]["n2_eq"].item()
        eq_std=stat_dict["std"]["n2_eq"].item()

        self.suffix = suffix

        # single layer dense nn to project the representation to the between_feats dimension
        # each interaction type has its own projector
        self.rep_projector = RepProjector(dim_tupel=2, in_feats=rep_feats, out_feats=between_feats)

        if symmetriser_feats is None:
            symmetriser_feats = between_feats
        if attention_hidden_feats is None:
            attention_hidden_feats = 4*between_feats


        self.bond_model = PermutationEquivariantTransformer(n_feats=between_feats, n_heads=n_heads, hidden_feats=attention_hidden_feats, n_layers=n_att, out_feats=2, permutations=torch.tensor([[0,1],[1,0]], dtype=torch.int32), layer_norm=layer_norm, dropout=dropout, symmetriser_layers=dense_layers, symmetriser_hidden_feats=symmetriser_feats, positional_encoding=copy.deepcopy(positional_encoding))
        
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

    def __init__(self, rep_feats, between_feats, suffix="", stat_dict=None, n_att=2, n_heads=8, dense_layers=2, dropout=0., layer_norm=True, symmetriser_feats=None, attention_hidden_feats=None, positional_encoding=True):
        super().__init__()

        assert not stat_dict is None
        k_mean=stat_dict["mean"]["n3_k"].item()
        k_std=stat_dict["std"]["n3_k"].item()
        # eq_mean=stat_dict["mean"]["n3_eq"].item()
        eq_std=stat_dict["std"]["n3_eq"].item()

        self.suffix = suffix

        # single layer dense nn to project the representation to the between_feats dimension
        # each interaction type has its own projector
        self.rep_projector = RepProjector(dim_tupel=3, in_feats=rep_feats, out_feats=between_feats)


        if symmetriser_feats is None:
            symmetriser_feats = between_feats
        if attention_hidden_feats is None:
            attention_hidden_feats = 4*between_feats


        self.angle_model = PermutationEquivariantTransformer(n_feats=between_feats, n_heads=n_heads, hidden_feats=attention_hidden_feats, n_layers=n_att, out_feats=2, permutations=torch.tensor([[0,1,2],[2,1,0]], dtype=torch.int32), layer_norm=layer_norm, dropout=dropout, symmetriser_layers=dense_layers, symmetriser_hidden_feats=symmetriser_feats, positional_encoding=copy.deepcopy(positional_encoding))
        

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
    This is still experimental. NOTE: currently not done. why?

    GatedTorsion layer that takes as input the output of the representation layer and writes the torsion parameters into the graph enforcing the permutation symmetry by a symmetrizer network \psi:
    symmetric_feature = \sum_symmetric_permutations \psi(xi,xj,xk,xl)
    out = \phi(symmetric_feature) * sigmoid(\chi(symmetric_feature))

    \phi is a dense neural network and \chi is a classifier network, predicting a gate score of "how nonzero" the torsion parameter should be.

    For Impropers:
        Enforce symmetry under the permutation. Note that each improper torsion occurs thrice in the graph to ensure symmetry of the energy under permutation of the outer atoms. (To reduce the number of parameters that are to be learned, we can use the antisymmetry of the dihedral, or the symmetry of the cosine of the dihedral, under the permutation of the outer atoms. If k is symmetric, we have found an energy function that is symmetric under the permutation of the outer atoms.)

    For Propers:
        Enforce symmetry of the energy function by enforcing symmetry of k under [3,2,1,0] and using symmetry of the dihedral angle under this permutation.
    """
    def __init__(self, rep_feats, between_feats, suffix="", n_periodicity=None, magnitude=0.001, improper=False, n_att=2, n_heads=8, dense_layers=2, dropout=0., layer_norm=True, symmetriser_feats=None, attention_hidden_feats=None, stat_dict=None, positional_encoding=True):
        super().__init__()

        if n_periodicity is None:
            n_periodicity = constants.N_PERIODICITY_PROPER
            if improper:
                n_periodicity = constants.N_PERIODICITY_IMPROPER

        self.n_periodicity = n_periodicity

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
                
                if len(k_mean) < n_periodicity:
                    raise ValueError(f"n_periodicity is {n_periodicity} but the stat_dict contains {len(k_mean)} values for the mean of the improper torsion parameters.")
                
                if len(k_std) < n_periodicity:
                    raise ValueError(f"n_periodicity is {n_periodicity} but the stat_dict contains {len(k_std)} values for the std of the improper torsion parameters.")
                
                k_mean = k_mean[:n_periodicity]
                k_std = k_std[:n_periodicity]

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
        self.rep_projector = RepProjector(dim_tupel=4, in_feats=rep_feats, out_feats=between_feats, improper=improper)


        if symmetriser_feats is None:
            symmetriser_feats = between_feats

        if attention_hidden_feats is None:
            attention_hidden_feats = 4*between_feats

        
        if not improper:
            perms = torch.tensor([[0,1,2,3],[3,2,1,0]], dtype=torch.int32)
            # enforce symmetry under the permutation

        else:
            # enforce symmetry under the permutation. Note that each improper torsion occurs thrice in the graph to ensure symmetry of the energy under permutation of the outer atoms. (To reduce the number of parameters that are to be learned, we can use the antisymmetry of the dihedral (and thus the symmetry of cos(dihedral) ) under the permutation of the outer atoms. If also k is symmetric under this permutation, we have found an energy function that is symmetric under the permutation of the outer atoms.)
            # Note that k may not be invariant under all permutations that leave the central atom fixed since cos(dihedral) is not invariant under these permutations and we want the energy contribution to be invariant.
            perms = torch.tensor([[0,1,2,3],[3,1,2,0]], dtype=torch.int32)


        self.torsion_model = PermutationEquivariantTransformer(n_feats=between_feats, n_heads=n_heads, hidden_feats=attention_hidden_feats, n_layers=n_att, out_feats=n_periodicity, permutations=perms, layer_norm=layer_norm, dropout=dropout, symmetriser_layers=dense_layers, symmetriser_hidden_feats=symmetriser_feats, positional_encoding=copy.deepcopy(positional_encoding))


    def forward(self, g):
        level = "n4"
        if self.improper:
            level += "_improper"
        
        if not level in g.ntypes:
            return g

        
        # transform the input to the shape 4, num_pairs, rep_dim
        inputs = self.rep_projector(g)

        if inputs.shape[1] == 0: # no torsions in the graph
            coeffs = torch.zeros((0,self.n_periodicity), dtype=inputs.dtype, device=inputs.device)

        else:
            # shape: n_torsions, n_periodicity
            coeffs = self.torsion_model(inputs)

            coeffs = coeffs*self.k_std + self.k_mean

        g.nodes[level].data["k"+self.suffix] = coeffs

        return g