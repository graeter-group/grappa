import torch
 
from grappa.models.geometry import GeometryInGraph

# no factor 1/2 included!
# does not support batching:
# shape of k: tuple x periodicity
# shape of angle: tuple x confs
# phases are all zero
# k[some_tuple] must be ordered with increasing periodicity, starting at 1
# angle must be in radians
# if offset if False, implements \sum_n k_n cos(n*phi) (i.e. without offset)
def torsion_energy(k, angle, offset=True):
    max_periodicity = k.shape[1]
    # n_tuples = k.shape[0]
    # n_batches = angle.shape[0]
    # periodicities = torch.tensor(range(1,max_periodicity+1), dtype=torch.float32).repeat(n_batches, n_tuples, 1)

    # bring all in the shape   tuple x periodicity x conf
    periodicity = torch.tensor(range(1,max_periodicity+1), device=k.device).unsqueeze(dim=0).unsqueeze(dim=-1)
    angle = angle.unsqueeze(dim=1)
    k = k.unsqueeze(dim=-1)

    if not offset:
        energy = k*torch.cos(periodicity*angle)
    else:
        energy = torch.abs(k) + k*torch.cos(periodicity*angle)
    # sum over all dims except the conf
    energy = energy.sum(dim=0).sum(dim=0)
    return energy


# does not support batching
# 1/2 is included!
# shape of k, eq: tuple x 1
# shape of distances: tuple x confs
def harmonic_energy(k, eq, distances):
    energy = k*torch.square(distances-eq)
    # sum over all dims except the tuple dim
    energy = energy.sum(dim=0)
    return 0.5*energy

class WriteEnergy(torch.nn.Module):
    def __init__(self, terms=["n2", "n3", "n4", "n4_improper"], suffix="", offset_torsion=True):
        super().__init__()
        self.offset_torsion = offset_torsion
        self.suffix = suffix
        self.terms = terms
        self.geom = GeometryInGraph()

    def forward(self, g):
        if not "xyz" in g.nodes["n1"].data.keys():
            return g
        
        g = self.geom(g)
        
        energy = 0

        for term in self.terms:
            contrib = WriteEnergy.get_energy_contribution(g, term=term, suffix=self.suffix, offset_torsion=self.offset_torsion)
            contrib = contrib.unsqueeze(dim=0) #artefact of formulation for batching
            energy += contrib
            with torch.no_grad():
                g.nodes["g"].data["u_"+term+self.suffix] = contrib

        g.nodes["g"].data["u"+self.suffix] = energy
        return g
    
    @staticmethod
    def get_energy_contribution(g, term, suffix, offset_torsion=True):
        k = g.nodes[term].data["k"+suffix]
        dof_data = g.nodes[term].data["x"]

        if term in ["n2", "n3"]:
            eq = g.nodes[term].data["eq"+suffix]
            # divide by two to compensate double counting of invariant terms
            return harmonic_energy(k=k, eq=eq, distances=dof_data)/2.
        


        en = torsion_energy(k=k,angle=dof_data, offset=offset_torsion)
        if term == "n4":
            # divide by two to compensate double counting of invariant terms
            return en/2.
        if term == "n4_improper":
            # divide by three to compensate triple counting of invariant terms
            return en/3.
