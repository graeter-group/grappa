
import torch

from grappa.models.final_layer import ToPositive, ToRange


def get_default_statistics():
    DEFAULT_STATISTICS = {
    'mean':
        {'n2_k': torch.Tensor([763.2819]), 'n2_eq': torch.Tensor([1.2353]), 'n3_k': torch.Tensor([105.6576]), 'n3_eq': torch.Tensor([1.9750]), 'n4_k': torch.Tensor([ 1.5617e-01, -5.8312e-01,  7.0820e-02, -6.3840e-04,  4.7139e-04, -4.1655e-04]), 'n4_improper_k': torch.Tensor([ 0.0000, -2.3933,  0.0000,  0.0000,  0.0000,  0.0000])},
    'std':
        {'n2_k': torch.Tensor([161.2278]), 'n2_eq': torch.Tensor([0.1953]), 'n3_k': torch.Tensor([26.5965]), 'n3_eq': torch.Tensor([0.0917]), 'n4_k': torch.Tensor([0.4977, 1.2465, 0.1466, 0.0192, 0.0075, 0.0066]), 'n4_improper_k': torch.Tensor([0.0000, 4.0571, 0.0000, 0.0000, 0.0000, 0.0000])}}
    return DEFAULT_STATISTICS




class ResidualDenseLayer(torch.nn.Module):
    def __init__(self, activation, module_fct, *module_args):
        super().__init__()
        self.module_skip = module_fct(*module_args)
        self.module1 = module_fct(*module_args)
        self.module2 = module_fct(*module_args)
        self.activation = activation


    def forward(self, h):
        h_skip = self.module_skip(h)
        h = self.module1(h)
        h = self.activation(h)
        h = self.module2(h)
        return self.activation(h_skip+h)


"""
Use a NN with normalized output to predict bond parameters. pooling is trivial as of now. The mean and std deviation of the dataset can be overwritten by handing over a dict with key tree mean/std -> level_name
"""
class WriteBondParameters(torch.nn.Module):
    def __init__(self, rep_feats, between_feats, suffix="", stat_dict=None):
        super().__init__()

        assert not stat_dict is None
        k_mean=stat_dict["mean"]["n2_k"].item()
        k_std=stat_dict["std"]["n2_k"].item()
        eq_mean=stat_dict["mean"]["n2_eq"].item()
        eq_std=stat_dict["std"]["n2_eq"].item()
        
        
        self.suffix = suffix

        self.bond_nn = torch.nn.Sequential(
            torch.nn.Linear(2*rep_feats, between_feats),
            torch.nn.ELU(),
            #ResidualDenseLayer(torch.nn.ELU(), torch.nn.Linear, between_feats, between_feats),
            ResidualDenseLayer(torch.nn.ELU(), torch.nn.Linear, between_feats, between_feats),
            torch.nn.Linear(between_feats, between_feats),
            torch.nn.ELU(),
            torch.nn.Linear(between_feats, 2),
        )
        self.to_k = ToPositive(mean=k_mean, std=k_std, min_=0)
        self.to_eq = ToPositive(mean=eq_mean, std=eq_std)


    def forward(self, g):
        # bonds:
        # every index pair appears twice, therefore 0.5 factor in energy calculation
        pairs = g.nodes["n2"].data["idxs"]
        try:
            inputs = g.nodes["n1"].data["h"][pairs]
        except IndexError as err:
            err.message += f"\nIt might be that g.nodes['n2'].data['idxs'] has the wrong datatype. It should be a long, byte or bool but is {pairs.dtype}"

#---------------
        inputs = torch.cat((inputs[:,0,:], inputs[:,1,:]), dim=-1)
        # inputs now has shape num_pairs, rep_dim*2
        coeffs = self.bond_nn(inputs)

        coeffs[:,0] = self.to_eq(coeffs[:,0])
        coeffs[:,1] = self.to_k(coeffs[:,1])

        g.nodes["n2"].data["eq"+self.suffix] = coeffs[:,0].unsqueeze(dim=-1)
        g.nodes["n2"].data["k"+self.suffix] = coeffs[:,1].unsqueeze(dim=-1)
        return g
    
#%%
class WriteAngleParameters(torch.nn.Module):
    def __init__(self, rep_feats, between_feats, suffix="", stat_dict=None):
        super().__init__()

        assert not stat_dict is None
        k_mean=stat_dict["mean"]["n3_k"].item()
        k_std=stat_dict["std"]["n3_k"].item()
        eq_mean=stat_dict["mean"]["n3_eq"].item()
        eq_std=stat_dict["std"]["n3_eq"].item()

        self.suffix = suffix

        self.angle_nn = torch.nn.Sequential(
            torch.nn.Linear(3*rep_feats, between_feats),
            torch.nn.ELU(),
            #ResidualDenseLayer(torch.nn.ELU(), torch.nn.Linear, between_feats, between_feats),
            ResidualDenseLayer(torch.nn.ELU(), torch.nn.Linear, between_feats, between_feats),
            torch.nn.Linear(between_feats, between_feats),
            torch.nn.ELU(),
            torch.nn.Linear(between_feats, 2),
        )
        self.to_k = ToPositive(mean=k_mean, std=k_std, min_=0)
        self.to_eq = ToRange(max_=torch.pi, std=eq_std)


    def forward(self, g):
        # bonds:
        # every index pair appears twice, with permuted end points therefore 0.5 factor in energy calculation and sum is done automatically
        # however, is it maybe better to do the sum in the arguments directly? or is this not so unique then?
        pairs = g.nodes["n3"].data["idxs"]
        try:
            inputs = g.nodes["n1"].data["h"][pairs]
        except IndexError as err:
            err.message += f"\nIt might be that g.nodes['n3'].data['idxs'] has the wrong datatype. It should be a long, byte or bool but is {pairs.dtype}"

#---------------
        inputs = torch.cat((inputs[:,0,:], inputs[:,1,:], inputs[:,2,:]), dim=-1)
        # inputs now has shape num_pairs, rep_dim*2
        coeffs = self.angle_nn(inputs)

        coeffs[:,0] = self.to_eq(coeffs[:,0])
        coeffs[:,1] = self.to_k(coeffs[:,1])

        g.nodes["n3"].data["eq"+self.suffix] = coeffs[:,0].unsqueeze(dim=-1)
        g.nodes["n3"].data["k"+self.suffix] = coeffs[:,1].unsqueeze(dim=-1)
        return g


class WriteTorsionParameters(torch.nn.Module):
    def __init__(self, rep_feats, between_feats, suffix="", n_periodicity=6, magnitude=0.001):
        super().__init__()
        self.suffix = suffix

        self.torsion_nn = torch.nn.Sequential(
            torch.nn.Linear(4*rep_feats, between_feats),
            torch.nn.ELU(),
            #ResidualDenseLayer(torch.nn.ELU(), torch.nn.Linear, between_feats, between_feats),
            #ResidualDenseLayer(torch.nn.ELU(), torch.nn.Linear, between_feats, between_feats),
            ResidualDenseLayer(torch.nn.ELU(), torch.nn.Linear, between_feats, between_feats),
            torch.nn.Linear(between_feats, between_feats),
            torch.nn.ELU(),
            torch.nn.Linear(between_feats, n_periodicity),
        )
        self.magnitude = magnitude


    def forward(self, g):
        # bonds:
        # every index pair appears twice, with permuted end points therefore 0.5 factor in energy calculation and sum is done automatically
        # however, is it maybe better to do the sum in the arguments directly? or is this not so unique then?
        pairs = g.nodes["n4"].data["idxs"]
        try:
            inputs = g.nodes["n1"].data["h"][pairs]
        except IndexError as err:
            err.message += f"\nIt might be that g.nodes['n4'].data['idxs'] has the wrong datatype. It should be a long, byte or bool but is {pairs.dtype}"

#---------------
        inputs = torch.cat((inputs[:,0,:], inputs[:,1,:], inputs[:,2,:], inputs[:,3,:]), dim=-1)
        # inputs now has shape num_pairs, rep_dim*2
        coeffs = self.torsion_nn(inputs)

        g.nodes["n4"].data["k"+self.suffix] = coeffs*self.magnitude
        return g

class WriteImproperParameters(WriteTorsionParameters):
    def forward(self, g):
        # bonds:
        # every index pair appears twice, with permuted end points therefore 0.5 factor in energy calculation and sum is done automatically
        # however, is it maybe better to do the sum in the arguments directly? or is this not so unique then?
        pairs = g.nodes["n4_improper"].data[".item()idxs"]
        try:
            inputs = g.nodes["n1"].data["h"][pairs]
        except IndexError as err:
            err.message += f"\nIt might be that g.nodes['n4_improper'].data['idxs'] has the wrong datatype. It should be a long, byte or bool but is {pairs.dtype}"

    #---------------
        inputs = torch.cat((inputs[:,0,:], inputs[:,1,:], inputs[:,2,:], inputs[:,3,:]), dim=-1)
        # inputs now has shape num_pairs, rep_dim*2
        coeffs = self.torsion_nn(inputs)

        g.nodes["n4_improper"].data["k"+self.suffix] = coeffs*self.magnitude
        return g