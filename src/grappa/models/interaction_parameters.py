import torch
import dgl
from grappa.models.final_layer import ToPositive, ToRange
from grappa import constants
from typing import Union, List, Tuple
from grappa.models.perm_equiv_transformer import SymmetrisedTransformer
import copy


class WriteParameters(torch.nn.Module):
    """
    A class that consolidates parameter writers for bonds, angles, proper torsions, and improper torsions. 
    Each writer is specialized for its respective parameter type and shares some common configuration options.

    Initialization Arguments:
        graph_node_features (int, optional): Number of features in the input graph nodes. Defaults to 256.
        parameter_dropout (float, optional): Dropout rate applied in all writers. Defaults to 0.
        layer_norm (bool, optional): Flag to apply layer normalization in all writers. Defaults to True.
        positional_encoding (bool, optional): Flag to apply positional encoding in all writers. Defaults to True.
        
        For each parameter type (bond, angle, proper, improper):
            {...}_transformer_depth (int, optional): Depth (number of layers) of the transformer. Defaults to 2.
            {...}_n_heads (int, optional): Number of attention heads in the transformer. Defaults to 8.
            {...}_transformer_width (int, optional): Width (hidden feature dimension) of the transformer. Defaults to 512.
            {...}_symmetriser_depth (int, optional): Number of layers in the symmetriser. Defaults to 2.
            {...}_symmetriser_width (int, optional): Width (feature dimension) of the symmetriser. Defaults to 256.

        n_periodicity_proper (int, optional): Number of periodicity terms for proper torsions. Defaults to 6.
        n_periodicity_improper (int, optional): Number of periodicity terms for improper torsions. Defaults to 3.
        gated_torsion (bool, optional): Flag indicating whether to a gate for torsions. Defaults to False.
        wrong_symmetry (bool, optional): Flag indicating whether to implement the wrong symmetry constraint for improper torsions. Defaults to False.
        stat_dict (dict, optional): Dictionary containing statistics for parameter initialization.

    The class initializes and holds four separate writers, each configured for their respective parameter type.
    """
    def __init__(self, graph_node_features=256, parameter_dropout=0, layer_norm=True, positional_encoding=True, bond_transformer_depth=2, bond_n_heads=8, bond_transformer_width=512, bond_symmetriser_depth=2, bond_symmetriser_width=256, angle_transformer_depth=2, angle_n_heads=8, angle_transformer_width=512, angle_symmetriser_depth=2, angle_symmetriser_width=256, proper_transformer_depth=2, proper_n_heads=8, proper_transformer_width=512, proper_symmetriser_depth=2, proper_symmetriser_width=256, improper_transformer_depth=2, improper_n_heads=8, improper_transformer_width=512, improper_symmetriser_depth=2, improper_symmetriser_width=256, n_periodicity_proper=6, n_periodicity_improper=3, gated_torsion:bool=False, suffix="", wrong_symmetry=False, stat_dict=None):
        super().__init__()


        # Initialize Bond Writer
        self.bond_writer = WriteBondParameters(
            rep_feats=graph_node_features,
            between_feats=bond_transformer_width,
            suffix=suffix,
            stat_dict=stat_dict,
            n_att=bond_transformer_depth,
            n_heads=bond_n_heads,
            dense_layers=bond_symmetriser_depth,
            dropout=parameter_dropout,
            layer_norm=layer_norm,
            symmetriser_feats=bond_symmetriser_width,
            attention_hidden_feats=bond_transformer_width,
        )

        # Initialize Angle Writer
        self.angle_writer = WriteAngleParameters(
            rep_feats=graph_node_features,
            between_feats=angle_transformer_width,
            suffix=suffix,
            stat_dict=stat_dict,
            n_att=angle_transformer_depth,
            n_heads=angle_n_heads,
            dense_layers=angle_symmetriser_depth,
            dropout=parameter_dropout,
            layer_norm=layer_norm,
            symmetriser_feats=angle_symmetriser_width,
            attention_hidden_feats=angle_transformer_width,
            positional_encoding=positional_encoding
        )

        # Initialize Proper Torsion Writer
        self.proper_writer = WriteTorsionParameters(
            rep_feats=graph_node_features,
            between_feats=proper_transformer_width,
            suffix=suffix,
            n_periodicity=n_periodicity_proper,
            improper=False,
            n_att=proper_transformer_depth,
            n_heads=proper_n_heads,
            dense_layers=proper_symmetriser_depth,
            dropout=parameter_dropout,
            layer_norm=layer_norm,
            symmetriser_feats=proper_symmetriser_width,
            attention_hidden_feats=proper_transformer_width,
            stat_dict=stat_dict,
            positional_encoding=positional_encoding,
            gated=gated_torsion,
        )

        # Initialize Improper Torsion Writer
        self.improper_writer = WriteTorsionParameters(
            rep_feats=graph_node_features,
            between_feats=improper_transformer_width,
            suffix=suffix,
            n_periodicity=n_periodicity_improper,
            improper=True,
            n_att=improper_transformer_depth,
            n_heads=improper_n_heads,
            dense_layers=improper_symmetriser_depth,
            dropout=parameter_dropout,
            layer_norm=layer_norm,
            symmetriser_feats=improper_symmetriser_width,
            attention_hidden_feats=improper_transformer_width,
            stat_dict=stat_dict,
            positional_encoding=positional_encoding,
            gated=gated_torsion,
            wrong_symmetry=wrong_symmetry
        )

    def forward(self, g):
        # NOTE:
        # Since these operations are all independent, they might be parallelized if they do not output the graph but parameter dicts that are then written to the graph.
        # also, it might make sense to extract the inputs for the writer first such that they do not all access the whole graph at the same time.

        g = self.bond_writer(g)
        g = self.angle_writer(g)
        g = self.proper_writer(g)
        g = self.improper_writer(g)

        return g




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
        This Layer takes a graph with node representation (num_nodes, feat_dim), passes it through one MLP layer and returns a stack of dim_tupel node feature vectors that contain the features of the nodes involved in the respective interaction given by self.dim_tupel. The output thus has shape (dim_tupel, num_tuples, out_feat_dim).
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

    ----------
    Parameters:
    ----------
        rep_feats (int): Number of features in the input representation.
        between_feats (int): Intermediate feature dimension used in the transformer model.
        suffix (str): Suffix for the parameter names in the graph, i.e. they will be written to g.nodes[...].data["k"+suffix].
        stat_dict (dict, optional): Dictionary containing statistics for parameter initialization.
        n_att (int): Number of attention layers in the transformer model.
        n_heads (int): Number of attention heads in each attention layer.
        dense_layers (int): Number of dense layers in the symmetrizer network.
        dropout (float): Dropout rate used in the transformer model.
        layer_norm (bool): Flag to apply layer normalization in the transformer model.
        symmetriser_feats (int, optional): Feature dimension for the symmetrizer network.
        attention_hidden_feats (int, optional): Hidden feature dimension in the transformer model.

    ----------

    Consists of:
        - a single layer dense nn to project the two node features to dimension between_feats
        - a SymmetrisedTransformer that takes the output of the rep_projector and outputs the bond parameter scores
        - a ToPositive layer that transforms the output of the SymmetrisedTransformer to positive values such that a normal distribution of symmetriser outputs would lead to a Gaussian distribution with mean stat_dict["mean"]["..."] and std stat_dict["std"]["..."]
    For initialization of the model weights, a stat_dict containing the expected mean and std deviation of the parameters can be given. Then the output of the symmetriser will be approximately a normal distribution with mean 0 and unit std deviation. This is optional, but recommended for achieving faster training convergence.
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


        self.bond_model = SymmetrisedTransformer(n_feats=between_feats, n_heads=n_heads, hidden_feats=attention_hidden_feats, n_layers=n_att, out_feats=2, permutations=torch.tensor([[0,1],[1,0]], dtype=torch.int32), layer_norm=layer_norm, dropout=dropout, symmetriser_layers=dense_layers, symmetriser_hidden_feats=symmetriser_feats, positional_encoding=False)
        
        self.to_k = ToPositive(mean=k_mean, std=k_std, min_=0)
        self.to_eq = ToPositive(mean=eq_mean, std=eq_std)


    def forward(self, g):

        # build a tuple of feature vectors from the representation.
        # inputs will have shape 2, num_pairs, rep_dim
        inputs = self.rep_projector(g)
        
        coeffs = self.bond_model(inputs)

        coeffs[:,0] = self.to_eq(coeffs[:,0])
        coeffs[:,1] = self.to_k(coeffs[:,1])

        g.nodes["n2"].data["eq"+self.suffix] = coeffs[:,0]
        g.nodes["n2"].data["k"+self.suffix] = coeffs[:,1]

        return g



class WriteAngleParameters(torch.nn.Module):
    """
    Layer that takes as input the output of the representation layer and writes the torsion parameters into the graph enforcing the permutation symmetry by a symmetrizer network \psi:
    symmetric_feature = \psi(xi,xj,xk) + \psi(xk,xj,xi)
    out = \phi(symmetric_feature)

    ----------
    Parameters:
    ----------
        rep_feats (int): Number of features in the input representation.
        between_feats (int): Intermediate feature dimension used in the transformer model.
        suffix (str): Suffix for the parameter names in the graph, i.e. they will be written to g.nodes[...].data["k"+suffix].
        stat_dict (dict, optional): Dictionary containing statistics for parameter initialization.
        n_att (int): Number of attention layers in the transformer model.
        n_heads (int): Number of attention heads in each attention layer.
        dense_layers (int): Number of dense layers in the symmetrizer network.
        dropout (float): Dropout rate used in the transformer model.
        layer_norm (bool): Flag to apply layer normalization in the transformer model.
        symmetriser_feats (int, optional): Feature dimension for the symmetrizer network.
        attention_hidden_feats (int, optional): Hidden feature dimension in the transformer model.
        positional_encoding (bool): Flag to apply positional encoding. Defaults to True.


    Consists of:
        - a single layer dense nn to project the two node features to dimension between_feats
        - a SymmetrisedTransformer that takes the output of the rep_projector and outputs the bond parameter scores
        - a ToPositive layer for the force constant that transforms the output of the SymmetrisedTransformer to positive values such that a normal distribution of symmetriser outputs would lead to a Gaussian distribution with mean stat_dict["mean"]["..."] and std stat_dict["std"]["..."]
        - a ToRange layer for the equilibrium angle that transforms the output of the SymmetrisedTransformer to values between 0 and pi such that a normal distribution of symmetriser outputs would lead to a Gaussian distribution with mean stat_dict["mean"]["..."] and std stat_dict["std"]["..."]
    For initialization of the model weights, a stat_dict containing the expected mean and std deviation of the parameters can be given. Then the output of the symmetriser will be approximately a normal distribution with mean 0 and unit std deviation. This is optional, but recommended for achieving faster training convergence.
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
        rep_projected_feats = between_feats if not positional_encoding else between_feats-1
        self.rep_projector = RepProjector(dim_tupel=3, in_feats=rep_feats, out_feats=rep_projected_feats)


        if symmetriser_feats is None:
            symmetriser_feats = between_feats
        if attention_hidden_feats is None:
            attention_hidden_feats = 4*between_feats


        self.angle_model = SymmetrisedTransformer(n_feats=rep_projected_feats, n_heads=n_heads, hidden_feats=attention_hidden_feats, n_layers=n_att, out_feats=2, permutations=torch.tensor([[0,1,2],[2,1,0]], dtype=torch.int32), layer_norm=layer_norm, dropout=dropout, symmetriser_layers=dense_layers, symmetriser_hidden_feats=symmetriser_feats, positional_encoding=copy.deepcopy(positional_encoding))
        

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

        g.nodes["n3"].data["eq"+self.suffix] = coeffs[:,0]
        g.nodes["n3"].data["k"+self.suffix] = coeffs[:,1]

        return g





class WriteTorsionParameters(torch.nn.Module):
    """
    Module for writing torsion parameters into a graph, accounting for permutation symmetry.
    The model uses a SymmetrisedTransformer to predict n_periodicity torsion Fourier amplitudes. It does not predict phase shifts but allows for negative values of the amplitude instaed. If the gated flag is True, the output of the model are 2*n_periodicity values, the first of which are fed through a sigmoid function to obtain a gate value between 0 and 1, which is mutliplied to the output of the model to allow for more accurate prediction of near-zero values.

    Improper torsions are required to be stored at tuple index 1 or 2 (as is done by default in grappa). Then, the symmetry constraints implemented are:
        - For improper torsions, symmetry under permutation of atom 0 and atom 3. 
        - For proper torsions, symmetry under order reversal.
        - If the wrong_symmetry flag is set, the symmetry constraints for improper torsion is:
            - Symmetry of the parameters under all permutations that leave the central atom fixed.
    
    For a more detailed explanation of the symmetry constraints, see the notes below.

    ----------
    Parameters:
    ----------
        rep_feats (int): Number of features in the input representation.
        between_feats (int): Intermediate feature dimension used in the transformer model.
        suffix (str): Suffix for the parameter names in the graph, i.e. they will be written to g.nodes[...].data["k"+suffix].
        n_periodicity (int, optional): Number of periodicity terms for torsion. Defaults based on torsion type.
        improper (bool): Flag to indicate if the torsion is improper. Defaults to False.
        n_att (int): Number of attention layers in the transformer model.
        n_heads (int): Number of attention heads in each attention layer.
        dense_layers (int): Number of dense layers in the symmetrizer network.
        dropout (float): Dropout rate used in the transformer model.
        layer_norm (bool): Flag to apply layer normalization in the transformer model.
        symmetriser_feats (int, optional): Feature dimension for the symmetrizer network.
        attention_hidden_feats (int, optional): Hidden feature dimension in the transformer model.
        stat_dict (dict, optional): Dictionary containing statistics for parameter initialization.
        positional_encoding (bool): Flag to apply positional encoding. Defaults to True.
        gated (bool): Flag to apply a gating mechanism using sigmoid activation. Defaults to False.
        wrong_symmetry (bool): Flag to indicate whether to implement the symmetry of the energy function the wrong way (as is done in espaloma. this is for ablation studies). Defaults to False.

    ----------
    Returns:
    ----------
        dgl.DGLGraph: The updated graph with torsion parameters 'k{suffix}' written into it at the nodes of type "n4" or "n4_improper".

    ----------
    Notes:
    ----------
    The reason for the (0,1,2,3) -> (3,1,2,0) symmetry for improper torsion is that the dihedral is antisymmetric under this permutation, making cos(dihedral) symmetric. Thus, if we enforce k to be invariant, the energy contribution will also be invariant under the permutation. Since we always store three versions of independent improper torsions, we can obtain a total energy that is invariant under all permutations that leave the central atom fixed, by summing over the three independent terms. Assuming that the central index for improper torsions is 2, this means:
        E = E(k_{0,1,2,3}) + E(k_{3,1,2,0}) + E(k_{1,3,2,0}) + E(k_{0,3,2,1}) + E(k_{3,0,2,1}) + E(k_{1,0,2,3})
          = 2 E(k_{0,1,2,3}) + 2 E(k_{1,3,2,0}) + 2 E(k_{3,0,2,1})
        is invariant under all permutations that leave the central atom fixed. (If we would enforce k to be invariant under all of these permutations, as is done if the wrong_symmetry flag is set, the energy contribution would not be invariant since the dihedral angle is not invariant under all of these permutations.)
    The gating mechanism (if enabled) allows the network to predict zero-valued parameters more accurately.
    For proper torsions, symmetry of the energy function is ensured by symmetrizing the torsion constants.
    For improper torsions, symmetry under permutation of outer atoms is enforced.

    The parameter statistics are used to initialize the model weights such that a normally distributed output of the symmetriser network would lead to a Gaussian distribution with mean stat_dict["mean"]["..."] and std stat_dict["std"]["..."].
    """
    def __init__(self, rep_feats, between_feats, suffix="", n_periodicity=None, improper=False, n_att=2, n_heads=8, dense_layers=2, dropout=0., layer_norm=True, symmetriser_feats=None, attention_hidden_feats=None, stat_dict=None, positional_encoding=True, gated:bool=False, wrong_symmetry:bool=False):
        super().__init__()

        if improper:
            assert constants.IMPROPER_CENTRAL_IDX in [1,2] # make this more general later on

        self.gated = gated

        if n_periodicity is None:
            n_periodicity = constants.N_PERIODICITY_PROPER
            if improper:
                n_periodicity = constants.N_PERIODICITY_IMPROPER

        # make n_periodicity being saved in the state dict: 
        self.register_buffer("n_periodicity", torch.tensor(n_periodicity).long())

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

        self.wrong_symmetry = wrong_symmetry

        self.improper = improper


        # single layer dense nn to project the representation to the between_feats dimension
        # each interaction type has its own projector
        rep_projected_feats = between_feats if not positional_encoding else between_feats-1
        self.rep_projector = RepProjector(dim_tupel=4, in_feats=rep_feats, improper=improper, out_feats=rep_projected_feats)


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

            if self.wrong_symmetry:
                assert constants.IMPROPER_CENTRAL_IDX == 2

                # use all permutations that leave the central atom fixed (we still store the three independent improper torsions in the graph and will thus sum 3 times over the same energy contribution. since this is just for ablation, we dont need to optimize this.)
                perms = torch.tensor([[0,1,2,3],[3,1,2,0],[1,3,2,0],[0,3,2,1],[3,0,2,1],[1,0,2,3]], dtype=torch.int32)
                positional_encoding = torch.tensor([[0],[0],[1],[0]], dtype=torch.float32)

        n_out_feats = n_periodicity if not gated else 2*n_periodicity

        self.torsion_model = SymmetrisedTransformer(n_feats=rep_projected_feats, n_heads=n_heads, hidden_feats=attention_hidden_feats, n_layers=n_att, out_feats=n_out_feats, permutations=perms, layer_norm=layer_norm, dropout=dropout, symmetriser_layers=dense_layers, symmetriser_hidden_feats=symmetriser_feats, positional_encoding=copy.deepcopy(positional_encoding))


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

            if self.gated:
                # multiply a gate value to the output of the model. This value is between 0 and 1 and makes it easier to learn the zero value
                # to obtain the correct statistics, we 

                # shape of coeffs: n_torsions, 2*n_periodicity
                gate_values = torch.sigmoid(coeffs[:,self.n_periodicity:])
                coeffs = coeffs[:,:self.n_periodicity]

            coeffs = coeffs*self.k_std + self.k_mean

            if self.gated:
                coeffs = coeffs * gate_values

        g.nodes[level].data["k"+self.suffix] = coeffs

        return g