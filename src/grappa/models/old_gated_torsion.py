#%%


import torch

from grappa.models.old_readout import ResidualDenseLayer


"""
Get a prediction by multiplying a final linear layer with a final softmax layer, allowing the network to get more accurate around zero.
This is still very experimental. 
"""
#NOTE: Include a std deviation parameter here too and decide for the C infty version.

class GatedTorsion(torch.nn.Module):
    def __init__(self, rep_feats, between_feats, suffix="", n_periodicity=6, magnitude=0.001, turn_on_at_p=0.1, improper=False, dead=False, hardness=1):
        super().__init__()
        self.suffix = suffix

        self.shared_nn = torch.nn.Sequential(
            torch.nn.Linear(4*rep_feats, between_feats),
            torch.nn.ELU(),
        )

        self.classification_nn = torch.nn.Sequential(
            ResidualDenseLayer(torch.nn.ELU(), torch.nn.Linear, between_feats, between_feats),
            torch.nn.Linear(between_feats, between_feats),
            torch.nn.ELU(),
            torch.nn.Linear(between_feats, n_periodicity),
        )

        self.torsion_nn = torch.nn.Sequential(
            #ResidualDenseLayer(torch.nn.ELU(), torch.nn.Linear, between_feats, between_feats),
            #ResidualDenseLayer(torch.nn.ELU(), torch.nn.Linear, between_feats, between_feats),
            ResidualDenseLayer(torch.nn.ELU(), torch.nn.Linear, between_feats, between_feats),
            torch.nn.Linear(between_feats, between_feats),
            torch.nn.ELU(),
            torch.nn.Linear(between_feats, n_periodicity),
        )
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
        pairs = g.nodes[level].data["idxs"]
        inputs = g.nodes["n1"].data["h"][pairs]

#---------------
        inputs = torch.cat((inputs[:,0,:], inputs[:,1,:], inputs[:,2,:], inputs[:,3,:]), dim=-1)
        # inputs now has shape num_pairs, rep_dim*2
        inputs = self.shared_nn(inputs)
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