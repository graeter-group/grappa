
import torch

from grappa.models.final_layer import ToPositive, ToRange


def get_default_statistics():
    DEFAULT_STATISTICS = {
    'mean':
        {'n2_k': torch.Tensor([763.2819]), 'n2_eq': torch.Tensor([1.2353]), 'n3_k': torch.Tensor([105.6576]), 'n3_eq': torch.Tensor([1.9750]), 'n4_k': torch.Tensor([ 1.5617e-01, -5.8312e-01,  7.0820e-02, -6.3840e-04,  4.7139e-04, -4.1655e-04]), 'n4_improper_k': torch.Tensor([ 0.0000, -2.3933,  0.0000,  0.0000,  0.0000,  0.0000])},
    'std':
        {'n2_k': torch.Tensor([161.2278]), 'n2_eq': torch.Tensor([0.1953]), 'n3_k': torch.Tensor([26.5965]), 'n3_eq': torch.Tensor([0.0917]), 'n4_k': torch.Tensor([0.4977, 1.2465, 0.1466, 0.0192, 0.0075, 0.0066]), 'n4_improper_k': torch.Tensor([0.0000, 4.0571, 0.0000, 0.0000, 0.0000, 0.0000])}}
    return DEFAULT_STATISTICS



class skippedLinear(torch.nn.Module):
    """
    Implementing a single layer (linear + activation) with skip connection. The skip connection is implemented by adding the input to the output (after the activation).
    The skip is only performed if the output dimension is a multiple of the input dimension.
    Layer norm is performed at the beginning of the layer.
    """
    def __init__(self, in_feats, out_feats=None, activation=torch.nn.ELU(), layer_norm=True):
        super().__init__()
        # bias = not layer_norm
        bias = True

        if out_feats is None:
            out_feats = in_feats
            
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.out_feat_factor = int(out_feats/in_feats)

        # only do a skip connection if the output features is a multiple of the input features
        self.do_skip = (out_feats/in_feats).is_integer()

        self.linear = torch.nn.Linear(in_feats, out_feats, bias=bias)
        self.activation = activation


        self.do_layer_norm = layer_norm

        if layer_norm:
            self.layer_norm = torch.nn.LayerNorm(normalized_shape=(in_feats,))

    def forward(self, h):
        if self.do_layer_norm:
            h = self.layer_norm(h)
        if self.do_skip:
            h_skip = h
        h = self.linear(h)
        h = self.activation(h)
        if self.do_skip:
            h_skip = h_skip.repeat_interleave(repeats=self.out_feat_factor, dim=-1)
            h = h + h_skip
        return h



# class GatedSkipLinear(torch.nn.Module):
#     """
#     Layer with skip that can be used if the output has half the dimension than the input.
#     DOESNT WORK, TAKES TOO MUCH MEMORY
#     """
#     def __init__(self, in_feats, activation=torch.nn.ELU(), layer_norm=True):
#         super().__init__()

#         out_features = in_feats/2
#         assert out_features.is_integer()
#         out_features = int(out_features)

#         self.linear = torch.nn.Linear(in_feats, out_features)
#         self.score_layer = torch.nn.Linear(in_feats, in_feats)

#         self.activation = activation

#         if layer_norm:
#             self.layer_norm = torch.nn.LayerNorm(normalized_shape=(in_feats,))

#         self.do_layer_norm = layer_norm


#     def forward(self, x):

#         if self.do_layer_norm:
#             x = self.layer_norm(x)

#         x_skip = x
#         x_score_pred = x

#         x = self.activation(self.linear(x))
#         scores = self.score_layer(x_score_pred)

#         # reshape the input and scores to group every two elements
#         x_skip = x_skip.view(x_skip.shape[0],-1, 2)
#         scores = scores.view(scores.shape[0],-1, 2)

#         # select the element with the higher score
#         max_scores_indices = torch.argmax(scores, dim=-1)
#         x_skip_new = x_skip[:,:,max_scores_indices]
#         scores = scores[:,:,max_scores_indices]


#         # also multiply the output with a sigmoid of the score
#         x = x * torch.sigmoid(scores)

#         return x + x_skip_new



class DenseBlock(torch.nn.Module):
    """
    Implements a stack of linear layers with skip connections and constant width.
    """
    def __init__(self, feats, depth:int=1) -> None:
        super().__init__()
        self.block = torch.nn.ModuleList([
            skippedLinear(feats, feats) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.block:
            x = layer(x)
        return x


# class DenseShrinkingBlock(torch.nn.Module):
#     """
#     DOESNT WORK, TAKES TOO MUCH MEMORY
#     Implements a stack of linear layers where the feature dimension is halved at the beginning and then each second layer.
#     That is, the output dimension is input_dim * 2**(-ceil(depth/2))
#     """
#     def __init__(self, feats, depth=1, ) -> None:
#         super().__init__()
#         self.block = torch.nn.ModuleList()
#         for i in range(depth):
#             if i % 2 == 0:
#                 self.block.append(GatedSkipLinear(in_feats=feats))
#                 feats = feats/2
#                 if not feats.is_integer():
#                     raise ValueError(f"The number of features must be a multiple of 2. Occured at layer {i} with {feats} features.")
#                 feats = int(feats)
#             else:
#                 self.block.append(skippedLinear(in_feats=feats, out_feats=feats))

#         self.out_feats = feats

#     def forward(self, x):
#         for layer in self.block:
#             x = layer(x)
#         return x


class WriteBondParameters(torch.nn.Module):
    """
    Layer that takes as input the output of the representation layer and writes the torsion parameters into the graph enforcing the permutation symmetry by a symmetrizer network \psi:
    symmetric_feature = \psi(xi,xj) + \psi(xj,xi)
    out = \phi(symmetric_feature)

    The default mean and std deviation of the dataset can be overwritten by handing over a stat_dict with stat_dict['mean'/'std'][level_name] = value
    """
    def __init__(self, rep_feats, between_feats, suffix="", stat_dict=None, legacy=True, depth=4):
        super().__init__()

        assert not stat_dict is None
        k_mean=stat_dict["mean"]["n2_k"].item()
        k_std=stat_dict["std"]["n2_k"].item()
        eq_mean=stat_dict["mean"]["n2_eq"].item()
        eq_std=stat_dict["std"]["n2_eq"].item()

        self.suffix = suffix

        if legacy:
            self.symmetrizer = torch.nn.Sequential(
                skippedLinear(2*rep_feats, between_feats),
                DenseBlock(feats=between_feats, depth=depth-1)
            )
            

            self.bond_nn = torch.nn.Sequential(
                DenseBlock(feats=between_feats, depth=2),
                torch.nn.Linear(between_feats, 2),
            )

        else:

            pass
        
        self.to_k = ToPositive(mean=k_mean, std=k_std, min_=0)
        self.to_eq = ToPositive(mean=eq_mean, std=eq_std)


    def forward(self, g):
        # bonds:
        pairs = g.nodes["n2"].data["idxs"]
        try:
            # this has the shape num_pairs, 2, rep_dim
            inputs = g.nodes["n1"].data["h"][pairs]
        except IndexError as err:
            err.message += f"\nIt might be that g.nodes['n2'].data['idxs'] has the wrong datatype. It should be a long, byte or bool but is {pairs.dtype}"

#-----------------#
        feat_1 = inputs[:,0,:]
        feat_2 = inputs[:,1,:]
        inputs = self.symmetrizer(torch.cat((feat_1, feat_2), dim=-1)) + self.symmetrizer(torch.cat((feat_2, feat_1), dim=-1))
        # create input features for the bond_nn that are symmetric wrt permutation of the two atoms
        # inputs now has shape num_pairs, between_feat and is a representation of the bond not the atom tuple (which has an ordering)

        # due to the symmetrization, the input vector is redundant, every entry appears twice.
        # we keep this for now due to compatibility with espaloma functionality, but it would be more efficient to remove the redundancy (every energy contribution is calculated twice, now really giving the same value.)

        coeffs = self.bond_nn(inputs)

        coeffs[:,0] = self.to_eq(coeffs[:,0])
        coeffs[:,1] = self.to_k(coeffs[:,1])

        g.nodes["n2"].data["eq"+self.suffix] = coeffs[:,0].unsqueeze(dim=-1)
        g.nodes["n2"].data["k"+self.suffix] = coeffs[:,1].unsqueeze(dim=-1)
        return g
    
#%%
class WriteAngleParameters(torch.nn.Module):
    """
    Layer that takes as input the output of the representation layer and writes the torsion parameters into the graph enforcing the permutation symmetry by a symmetrizer network \psi:
    symmetric_feature = \psi(xi,xj,xk) + \psi(xk,xj,xi)
    out = \phi(symmetric_feature)

    The default mean and std deviation of the dataset can be overwritten by handing over a stat_dict with stat_dict['mean'/'std'][level_name] = value
    """

    def __init__(self, rep_feats, between_feats, suffix="", stat_dict=None, legacy=True, depth=4):
        super().__init__()

        assert not stat_dict is None
        k_mean=stat_dict["mean"]["n3_k"].item()
        k_std=stat_dict["std"]["n3_k"].item()
        eq_mean=stat_dict["mean"]["n3_eq"].item()
        eq_std=stat_dict["std"]["n3_eq"].item()

        self.suffix = suffix

        if legacy:

            self.symmetrizer = torch.nn.Sequential(
                skippedLinear(3*rep_feats, between_feats),
                DenseBlock(feats=between_feats, depth=depth-1)
            )
            

            self.angle_nn = torch.nn.Sequential(
                DenseBlock(feats=between_feats, depth=2),
                torch.nn.Linear(between_feats, 2),
            )

        else:

            pass


        self.to_k = ToPositive(mean=k_mean, std=k_std, min_=0)
        self.to_eq = ToRange(max_=torch.pi, std=eq_std)


    def forward(self, g):
        # bonds:
        # however, is it maybe better to do the sum in the arguments directly? or is this not so unique then?
        pairs = g.nodes["n3"].data["idxs"]
        try:
            inputs = g.nodes["n1"].data["h"][pairs]
        except IndexError as err:
            err.message += f"\nIt might be that g.nodes['n3'].data['idxs'] has the wrong datatype. It should be a long, byte or bool but is {pairs.dtype}"

#---------------
        feat_1 = inputs[:,0,:]
        feat_2 = inputs[:,1,:]
        feat_3 = inputs[:,2,:]
        inputs = self.symmetrizer(torch.cat((feat_1, feat_2, feat_3), dim=-1)) + self.symmetrizer(torch.cat((feat_3, feat_2, feat_1), dim=-1))
        # create input features for the bond_nn that are symmetric wrt permutation of the two atoms
        # inputs now has shape num_pairs, between_feat and is a representation of the bond not the atom tuple (which has an ordering)

        # due to the symmetrization, the input vector is redundant, every entry appears twice.
        # we keep this for now due to compatibility with espaloma functionality, but it would be more efficient to remove the redundancy (every energy contribution is calculated twice, now really giving the same value.)

        coeffs = self.angle_nn(inputs)

        coeffs[:,0] = self.to_eq(coeffs[:,0])
        coeffs[:,1] = self.to_k(coeffs[:,1])

        g.nodes["n3"].data["eq"+self.suffix] = coeffs[:,0].unsqueeze(dim=-1)
        g.nodes["n3"].data["k"+self.suffix] = coeffs[:,1].unsqueeze(dim=-1)
        return g

