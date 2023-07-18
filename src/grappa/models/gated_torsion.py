#%%
import torch
from .readout import DenseBlock, skippedLinear


#NOTE: Include a std deviation parameter here too and decide for the C infty version.

class GatedTorsion(torch.nn.Module):
    """
    Multiply with a final binary softmax (ie sigmoid) layer, allowing the network to be more accurate around zero.
    This is still experimental. 

    GatedTorsion layer that takes as input the output of the representation layer and writes the torsion parameters into the graph enforcing the permutation symmetry by a symmetrizer network \psi:
    symmetric_feature = \sum_symmetric_permutations \psi(xi,xj,xk,xl)
    out = \phi(symmetric_feature) * sigmoid(\chi(symmetric_feature))

    \phi is a dense neural network and \chi is a classifier network, predicting a gate score of "how nonzero" the torsion parameter should be.
    """
    def __init__(self, rep_feats, between_feats, suffix="", n_periodicity=None, magnitude=0.001, turn_on_at_p=0.1, improper=False, dead=False, hardness=1, legacy=True, depth=4):

        if n_periodicity is None:
            n_periodicity = 6
        if improper:
            n_periodicity = 3

        super().__init__()
        self.suffix = suffix

        if legacy:
            self.symmetrizer = torch.nn.Sequential(
                skippedLinear(4*rep_feats, between_feats),
                DenseBlock(feats=between_feats, depth=depth-1)
            )
            

            self.torsion_nn = torch.nn.Sequential(
                DenseBlock(feats=between_feats, depth=2),
                torch.nn.Linear(between_feats, n_periodicity),
            )

        else:

            pass


        self.classification_nn = torch.nn.Sequential(
            skippedLinear(between_feats, between_feats),
            torch.nn.Linear(between_feats, n_periodicity),
        )

        self.wrong_symmetry = False

        self.magnitude = magnitude
        self.turn_on_at_p = turn_on_at_p
        self.improper = improper
        self.dead = dead
        self.hardness = hardness

    def forward(self, g):
        # bonds:
        # every index pair appears twice, with permuted end points therefore 0.5 factor in energy calculation and sum is done automatically
        level = "n4"
        if self.improper:
            level += "_improper"
        
        if not level in g.ntypes:
            return g

        pairs = g.nodes[level].data["idxs"]
        inputs = g.nodes["n1"].data["h"][pairs]

#---------------
        feat_1 = inputs[:,0,:]
        feat_2 = inputs[:,1,:]
        feat_3 = inputs[:,2,:]
        feat_4 = inputs[:,3,:]
        
        if self.improper and self.wrong_symmetry:
            inputs = self.symmetrizer(torch.cat((feat_1, feat_2, feat_3, feat_4), dim=-1)) + self.symmetrizer(torch.cat((feat_4, feat_2, feat_1, feat_3), dim=-1)) + self.symmetrizer(torch.cat((feat_3, feat_2, feat_1, feat_4), dim=-1))
        else:
            inputs = self.symmetrizer(torch.cat((feat_1, feat_2, feat_3, feat_4), dim=-1)) + self.symmetrizer(torch.cat((feat_4, feat_3, feat_2, feat_1), dim=-1))


        # create input features for the bond_nn that are symmetric wrt permutation of the two atoms
        # inputs now has shape num_pairs, between_feat and is a representation of the bond not the atom tuple (which has an ordering)

        # due to the symmetrization, the input vector is redundant, every entry appears twice.
        # we keep this for now due to compatibility with espaloma functionality, but it would be more efficient to remove the redundancy (every energy contribution is calculated twice, now really giving the same value.)

        coeffs = self.torsion_nn(inputs)
        classification_score = self.classification_nn(inputs)

        importance_probability = torch.sigmoid(classification_score)

        if self.dead:
            # gate value: zero if p is smaller than or equal to the turn on probablity, rises continuosly, and is 1 if p is 1
            gate_value = 1./(1.-self.turn_on_at_p) * torch.nn.functional.relu(importance_probability - self.turn_on_at_p)
        else:
            gate_value = importance_probability ** self.hardness

        g.nodes[level].data["score"+self.suffix] = classification_score
        g.nodes[level].data["k"+self.suffix] = coeffs*self.magnitude * gate_value
        return g