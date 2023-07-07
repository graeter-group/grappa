# #%%

# import torch
# import espaloma as esp
# import dgl


# class ToPositive(torch.nn.Module):
#     def __init__(self, mean, std, min_=0):
#         super().__init__()
#         self.mean_over_std = mean/std
#         self.std = std
#         self.min_ = min_

#     def forward(self, x):
#         # for min=0, implements m+x*s for m+x*s > s, s*exp(m/s+x-1) else  (linear with mean and std until one std over zero)
#         return self.std * (torch.nn.functional.elu(self.mean_over_std+x-1)+1) + self.min_
    
# # maps values to a range (0, max) with mean max/2 and given approx std
# class ToRange(torch.nn.Module):
#     def __init__(self, max_, std):
#         super().__init__()
#         self.std_over_max = std/max_
#         self.max = max_

#     def forward(self, x):
#         # 
#         return self.max * torch.sigmoid(self.std_over_max*x)
    
# class ResidualGraphLayer(torch.nn.Module):
#     def __init__(self, n_feat, activation, module_fct, *module_args, self_interaction=True, conv_skip=False):
#         super().__init__()
#         self.conv_skip = conv_skip
#         if conv_skip:
#             self.module_skip = module_fct(*module_args)
#         else:
#             self.module_skip = torch.nn.Sequential(
#             torch.nn.Linear(n_feat,n_feat),
#             torch.nn.ELU(),
#             torch.nn.Linear(n_feat,n_feat),
#         )

#         self.module1 = module_fct(*module_args)
#         self.module2 = module_fct(*module_args)
#         self.activation = activation

#         if self_interaction:
#             self.self_interaction = torch.nn.Sequential(
#                 torch.nn.Linear(n_feat,n_feat),
#                 torch.nn.ELU(),
#                 torch.nn.Linear(n_feat,n_feat),
#             )
#         else:
#             self.self_interaction = torch.nn.ELU()

#     def forward(self, g, h):
#         if self.conv_skip:
#             h_skip = self.module_skip(g,h)
#         else:
#             h_skip = self.module_skip(h)
#         h = self.module1(g,h)
#         h = self.self_interaction(h)
#         h = self.module2(g,h)
#         return self.activation(h_skip+h)
    
# class ResidualDenseLayer(torch.nn.Module):
#     def __init__(self, activation, module_fct, *module_args):
#         super().__init__()
#         self.module_skip = module_fct(*module_args)
#         self.module1 = module_fct(*module_args)
#         self.module2 = module_fct(*module_args)
#         self.activation = activation


#     def forward(self, h):
#         h_skip = self.module_skip(h)
#         h = self.module1(h)
#         h = self.activation(h)
#         h = self.module2(h)
#         return self.activation(h_skip+h)



# class Representation(torch.nn.Module):
#     def __init__(self, in_feats, h_feats, out_feats, elements_only=False, n_residuals=3, n_conv=2):
#         super().__init__()
#         self.elements_only = elements_only
#         if elements_only:
#             in_feats = 16
#         # self.conv1 = dgl.nn.GraphConv(in_feats, h_feats)
#         # self.conv2 = dgl.nn.GraphConv(h_feats, h_feats)
#         # self.conv3 = dgl.nn.GraphConv(h_feats, h_feats)
#         # self.conv4 = dgl.nn.GraphConv(h_feats, out_feats)

#         self.pre_dense = torch.nn.Sequential(
#             torch.nn.Linear(in_feats, h_feats),
#         )

#         self.post_dense = torch.nn.Sequential(
#             torch.nn.Linear(h_feats, out_feats),
#         )

#         self.blocks = torch.nn.ModuleList([
#             ResidualGraphLayer(h_feats, torch.nn.ELU(), dgl.nn.pytorch.conv.SAGEConv, h_feats, h_feats, "mean")
#                 for i in range(n_residuals)
#             ]
#             +
#             [
#             dgl.nn.pytorch.conv.SAGEConv(h_feats, h_feats, aggregator_type="mean") for i in range(n_conv)
#             ])
#         # NOTE no nonlinearities there?


#     def forward(self, g, in_feature=None):
#         if in_feature is None:
#             in_feature = g.nodes["n1"].data["h0"]
#         if self.elements_only:
#             in_feature = in_feature[:,:16]

#         g_ = dgl.to_homo(g.edge_type_subgraph(["n1_neighbors_n1"]))

#         h = self.pre_dense(in_feature)

#         for block in self.blocks:
#             h = block(g_,h)
            
#         h = self.post_dense(h)
#         g.nodes["n1"].data["h"] = h
#         return g

# class WriteBondParameters(torch.nn.Module):
#     def __init__(self, rep_feats, between_feats, suffix="", k_mean=0.3, k_std=0.1, eq_mean=2.5, eq_std=1., k_min=0.1):
#         super().__init__()
#         self.suffix = suffix

#         self.bond_nn = torch.nn.Sequential(
#             torch.nn.Linear(2*rep_feats, between_feats),
#             torch.nn.ELU(),
#             #ResidualDenseLayer(torch.nn.ELU(), torch.nn.Linear, between_feats, between_feats),
#             ResidualDenseLayer(torch.nn.ELU(), torch.nn.Linear, between_feats, between_feats),
#             torch.nn.Linear(between_feats, between_feats),
#             torch.nn.ELU(),
#             torch.nn.Linear(between_feats, 2),
#         )
#         self.to_k = ToPositive(mean=k_mean, std=k_std, min_=k_min)
#         self.to_eq = ToPositive(mean=eq_mean, std=eq_std)


#     def forward(self, g):
#         # bonds:
#         # every index pair appears twice, therefore 0.5 factor in energy calculation
#         pairs = g.nodes["n2"].data["idxs"]
#         inputs = g.nodes["n1"].data["h"][pairs]

# #---------------
#         inputs = torch.cat((inputs[:,0,:], inputs[:,1,:]), dim=-1)
#         # inputs now has shape num_pairs, rep_dim*2
#         coeffs = self.bond_nn(inputs)

#         coeffs[:,0] = self.to_eq(coeffs[:,0])
#         coeffs[:,1] = self.to_k(coeffs[:,1])

#         g.nodes["n2"].data["eq"+self.suffix] = coeffs[:,0].unsqueeze(dim=-1)
#         g.nodes["n2"].data["k"+self.suffix] = coeffs[:,1].unsqueeze(dim=-1)
#         return g
    
# #%%
# class WriteAngleParameters(torch.nn.Module):
#     def __init__(self, rep_feats, between_feats, suffix="", k_mean=0.16, k_std=0.03, eq_std=torch.pi*0.3, k_min=0.1):
#         super().__init__()
#         self.suffix = suffix

#         self.angle_nn = torch.nn.Sequential(
#             torch.nn.Linear(3*rep_feats, between_feats),
#             torch.nn.ELU(),
#             #ResidualDenseLayer(torch.nn.ELU(), torch.nn.Linear, between_feats, between_feats),
#             ResidualDenseLayer(torch.nn.ELU(), torch.nn.Linear, between_feats, between_feats),
#             torch.nn.Linear(between_feats, between_feats),
#             torch.nn.ELU(),
#             torch.nn.Linear(between_feats, 2),
#         )
#         self.to_k = ToPositive(mean=k_mean, std=k_std, min_=k_min)
#         self.to_eq = ToRange(max_=torch.pi, std=eq_std)


#     def forward(self, g):
#         # bonds:
#         # every index pair appears twice, with permuted end points therefore 0.5 factor in energy calculation and sum is done automatically
#         # however, is it maybe better to do the sum in the arguments directly? or is this not so unique then?
#         pairs = g.nodes["n3"].data["idxs"]
#         inputs = g.nodes["n1"].data["h"][pairs]

# #---------------
#         inputs = torch.cat((inputs[:,0,:], inputs[:,1,:], inputs[:,2,:]), dim=-1)
#         # inputs now has shape num_pairs, rep_dim*2
#         coeffs = self.angle_nn(inputs)

#         coeffs[:,0] = self.to_eq(coeffs[:,0])
#         coeffs[:,1] = self.to_k(coeffs[:,1])

#         g.nodes["n3"].data["eq"+self.suffix] = coeffs[:,0].unsqueeze(dim=-1)
#         g.nodes["n3"].data["k"+self.suffix] = coeffs[:,1].unsqueeze(dim=-1)
#         return g


# class WriteTorsionParameters(torch.nn.Module):
#     def __init__(self, rep_feats, between_feats, suffix="", n_periodicity=6, magnitude=0.001):
#         super().__init__()
#         self.suffix = suffix

#         self.torsion_nn = torch.nn.Sequential(
#             torch.nn.Linear(4*rep_feats, between_feats),
#             torch.nn.ELU(),
#             #ResidualDenseLayer(torch.nn.ELU(), torch.nn.Linear, between_feats, between_feats),
#             #ResidualDenseLayer(torch.nn.ELU(), torch.nn.Linear, between_feats, between_feats),
#             ResidualDenseLayer(torch.nn.ELU(), torch.nn.Linear, between_feats, between_feats),
#             torch.nn.Linear(between_feats, between_feats),
#             torch.nn.ELU(),
#             torch.nn.Linear(between_feats, n_periodicity),
#         )
#         self.magnitude = magnitude


#     def forward(self, g):
#         # bonds:
#         # every index pair appears twice, with permuted end points therefore 0.5 factor in energy calculation and sum is done automatically
#         # however, is it maybe better to do the sum in the arguments directly? or is this not so unique then?
#         pairs = g.nodes["n4"].data["idxs"]
#         inputs = g.nodes["n1"].data["h"][pairs]

# #---------------
#         inputs = torch.cat((inputs[:,0,:], inputs[:,1,:], inputs[:,2,:], inputs[:,3,:]), dim=-1)
#         # inputs now has shape num_pairs, rep_dim*2
#         coeffs = self.torsion_nn(inputs)

#         g.nodes["n4"].data["k"+self.suffix] = coeffs*self.magnitude
#         return g

# class WriteImproperParameters(WriteTorsionParameters):
#     def forward(self, g):
#         # bonds:
#         # every index pair appears twice, with permuted end points therefore 0.5 factor in energy calculation and sum is done automatically
#         # however, is it maybe better to do the sum in the arguments directly? or is this not so unique then?
#         pairs = g.nodes["n4_improper"].data["idxs"]
#         inputs = g.nodes["n1"].data["h"][pairs]

#     #---------------
#         inputs = torch.cat((inputs[:,0,:], inputs[:,1,:], inputs[:,2,:], inputs[:,3,:]), dim=-1)
#         # inputs now has shape num_pairs, rep_dim*2
#         coeffs = self.torsion_nn(inputs)

#         g.nodes["n4_improper"].data["k"+self.suffix] = coeffs*self.magnitude
#         return g


# #%%
# # no factor 1/2 included!
# # does not support batching:
# # shape of k: tuple x periodicity
# # shape of angle: tuple x confs
# # phases are all zero
# # k[some_tuple] must be ordered with increasing periodicity, starting at 1
# # angle must be in radians
# # if offset if False, implements \sum_n k_n cos(n*phi) (i.e. without offset)
# def torsion_energy(k, angle, offset=True):
#     max_periodicity = k.shape[1]
#     # n_tuples = k.shape[0]
#     # n_batches = angle.shape[0]
#     # periodicities = torch.tensor(range(1,max_periodicity+1), dtype=torch.float32).repeat(n_batches, n_tuples, 1)

#     # bring all in the shape   tuple x periodicity x conf
#     periodicity = torch.tensor(range(1,max_periodicity+1), device=k.device).unsqueeze(dim=0).unsqueeze(dim=-1)
#     angle = angle.unsqueeze(dim=1)
#     k = k.unsqueeze(dim=-1)

#     if not offset:
#         energy = k*torch.cos(periodicity*angle)
#     else:
#         energy = torch.abs(k) + k*torch.cos(periodicity*angle)
#     # sum over all dims except the conf
#     energy = energy.sum(dim=0).sum(dim=0)
#     return energy


# # does not support batching
# # 1/2 is included!
# # shape of k, eq: tuple x 1
# # shape of distances: tuple x confs
# def harmonic_energy(k, eq, distances):
#     energy = k*torch.square(distances-eq)
#     # sum over all dims except the tuple dim
#     energy = energy.sum(dim=0)
#     return 0.5*energy

# class WriteEnergy(torch.nn.Module):
#     def __init__(self, terms=["n2", "n3", "n4", "n4_improper"], suffix="", offset_torsion=True):
#         super().__init__()
#         self.offset_torsion = offset_torsion
#         self.suffix = suffix
#         self.terms = terms
#         self.geom = esp.mm.geometry.GeometryInGraph()

#     def forward(self, g):
#         if not "xyz" in g.nodes["n1"].data.keys():
#             return g
        
#         g = self.geom(g)
        
#         energy = 0

#         for term in self.terms:
#             contrib = WriteEnergy.get_energy_contribution(g, term=term, suffix=self.suffix, offset_torsion=self.offset_torsion)
#             contrib = contrib.unsqueeze(dim=0) #artifact of formulation for batching
#             energy += contrib
#             with torch.no_grad():
#                 g.nodes["g"].data["u_"+term+self.suffix] = contrib

#         g.nodes["g"].data["u"+self.suffix] = energy
#         return g
    
#     @staticmethod
#     def get_energy_contribution(g, term, suffix, offset_torsion=True):
#         k = g.nodes[term].data["k"+suffix]
#         dof_data = g.nodes[term].data["x"]

#         if term in ["n2", "n3"]:
#             eq = g.nodes[term].data["eq"+suffix]
#             # divide by two to compensate double counting of invariant terms
#             return harmonic_energy(k=k, eq=eq, distances=dof_data)/2.
        


#         en = torsion_energy(k=k,angle=dof_data, offset=offset_torsion)
#         if term == "n4":
#             # divide by two to compensate double counting of invariant terms
#             return en/2.
#         if term == "n4_improper":
#             # divide by three to compensate triple counting of invariant terms
#             return en/3.
