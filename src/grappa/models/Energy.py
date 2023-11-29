import torch
import dgl
from grappa.utils.dgl_utils import grad_available
 
from grappa.models.InternalCoordinates import InternalCoordinates

def torsion_energy(k, angle, offset=True):
    """
    returns a tensor of shape tuples x confs containing the energy contributions of each torsion angle individually.

    implements
    sum_n k_n cos(n*phi) (+ |k_n| if offset=True)

    shape of k: tuples x periodicity
    shape of angle: tuples x confs
    phases are all zero
    k[some_tuple] must be ordered with increasing periodicity, starting at 1
    angle must be in radians
    if offset if False, implements \sum_n k_n cos(n*phi) (i.e. without offset)
    """
    max_periodicity = k.shape[1]
    # n_tuples = k.shape[0]
    # n_batches = angle.shape[0]
    # periodicities = torch.tensor(range(1,max_periodicity+1), dtype=torch.float32).repeat(n_batches, n_tuples, 1)

    # bring all in the shape   tuple x periodicity x conf
    periodicity = torch.tensor(range(1,max_periodicity+1), device=k.device, dtype=torch.float32).unsqueeze(dim=0).unsqueeze(dim=-1)
    angle = angle.unsqueeze(dim=1)
    k = k.unsqueeze(dim=-1)

    if not offset:
        energy = k*torch.cos(periodicity*angle)
    else:
        energy = torch.abs(k) + k*torch.cos(periodicity*angle)

    # sum over the periodicity dim
    energy = energy.sum(dim=1)

    # energy is now of shape tuple x conf. to obtain the total energy, one can sum over all tuples
    return energy


def harmonic_energy(k, eq, distances):
    """
    returns a tensor of shape tuples x confs containing the energy contributions of each tuple (bond/angle) individually.
    implements
    0.5 * k * (distances - eq)^2

    shape of k, eq: n_tuples
    shape of distances: n_tuples x confs
    """
    if len(k.shape) != 1:
        raise ValueError(f"k must be a 1d tensor, but has shape {k.shape}")
    energy = k.unsqueeze(dim=-1)*torch.square(distances-eq.unsqueeze(dim=-1))
    return 0.5 * energy


def pool_energy(g, energies, term, suffix):
    """
    Given a tensor of energy contributions of shape tuples x confs, returns a the energy pooled over the tuple dimension. this operation recognizes batched graphs.
    returns a tensor of shape n_batches x confs
    """
    if not energies.shape[0] == g.num_nodes(term):
        raise ValueError(f"shape of energies {energies.shape} does not match number of nodes {g.num_nodes(term)}")
    
    g.nodes[term].data['unpooled_energy'+suffix] = energies

    pooled_energies = dgl.readout_nodes(g, op='sum', ntype=term, feat='unpooled_energy'+suffix)
    
    return pooled_energies


class Energy(torch.nn.Module):
    """
    Class that writes the bonded energy of molecule conformations into a graph. First, torsional angles, angles and distances are calculated, then their energy contributions are added and written under g.nodes["g"].data["energy"+write_suffix] and g.nodes["g"].data["energy_"+term+write_suffix] for each term. If gradients is True, the gradients of the total energy w.r.t. the xyz coordinates are calculated and written under g.nodes["n1"].data["gradient"+write_suffix].
    If the internal coordiantes are not already written in the graph, calculates them using the InternalCoordinates module.
    """
    def __init__(self, terms:list=["n2", "n3", "n4", "n4_improper"], suffix:str="", offset_torsion:bool=False, write_suffix=None, gradients:bool=True):
        """
        terms: list of terms to be considered. must be a subset of ["n2", "n3", "n4", "n4_improper"]
        suffix: suffix of the parameters stored in the graph.
        offset_torsion: whether to include the constant offset term (that makes the contribution positive) in the torsion energy calculation
        write_suffix: suffix of the energy and gradient attributes written to the graph. if None, write_suffix is set to suffix.
        gradients: whether to calculate gradients of the total energy w.r.t. the xyz coordinates. This cannot be done if the context does not allow autograd (e.g. if this function is called while torch.no_grad() is active).
        """
        super().__init__()

        if not isinstance(terms, list):
            raise ValueError("terms must be a list")
        self.offset_torsion = offset_torsion
        self.suffix = suffix
        self.write_suffix = write_suffix if not write_suffix is None else suffix
        self.terms = terms
        self.gradients = gradients
        self.geom = InternalCoordinates()


    def forward(self, g):
        """
        First, torsional angles, bonds angles and distances are calculated, then their energy contributions are added and written under g.nodes["g"].data["energy"+write_suffix] and g.nodes["g"].data["energy_"+term+write_suffix] for each term.
        Also stores the individual contributions of shape  at g.nodes[term].data["energy"+write_suffix] for each term.
        """
        grad_enabled = torch.is_grad_enabled()
        if not grad_enabled and self.gradients:
            torch.set_grad_enabled(True)

        if not "xyz" in g.nodes["n1"].data.keys():
            raise ValueError("xyz coordinates must be stored in g.nodes['n1'].data['xyz']")
        
        if self.gradients:
            with torch.enable_grad():
                g.nodes["n1"].data["xyz"].requires_grad = True

        # calculate internal coordinates if they are not already in the graph
        g = self.geom(g)

        num_confs = g.nodes['n1'].data["xyz"].shape[1]
        num_batch = g.num_nodes("g")

        energy = torch.zeros((num_batch, num_confs), device=g.nodes['n1'].data["xyz"].device)

        for term in self.terms:
            if not term in g.ntypes:
                raise ValueError(f"term {term} not in g.ntypes")
            
            # get the energy contribution of this term in shape (num_batch, num_confs)

            contrib, tuple_energies = Energy.get_energy_contribution(g, term=term, suffix=self.suffix, offset_torsion=self.offset_torsion)
            if not contrib is None:
                energy += contrib
                g.nodes["g"].data["energy_"+term+self.write_suffix] = contrib.detach()
                g.nodes[term].data["energy"+self.write_suffix] = tuple_energies

        g.nodes["g"].data["energy"+self.write_suffix] = energy
        
        if self.gradients and grad_available():
            # calculate gradients
            grad = torch.autograd.grad(energy.sum(), g.nodes["n1"].data["xyz"], retain_graph=True, create_graph=True, allow_unused=True)[0]
            g.nodes["n1"].data["gradient"+self.write_suffix] = grad

        if self.gradients:
            torch.set_grad_enabled(grad_enabled)

        return g
    
    @staticmethod
    def get_energy_contribution(g, term, suffix, offset_torsion=True):
        """
        Returns:
        en, energies
        where en is the total energy contribution from this term and energies is a tensor of shape (num_tuples, num_confs) containing the energy contribution of each tuple individually.
        """
        if term not in g.ntypes:
            return None, None
        if "k"+suffix not in g.nodes[term].data.keys():
            raise RuntimeError(f"{term} has no k{suffix} attribute")

        k = g.nodes[term].data["k"+suffix]
        dof_data = g.nodes[term].data["x"]

        if term in ["n2", "n3"]:
            eq = g.nodes[term].data["eq"+suffix]
            energies = harmonic_energy(k=k, eq=eq, distances=dof_data)
            en = pool_energy(g=g, energies=energies, term=term, suffix=suffix)
  
        if term in ["n4", "n4_improper"]:
            energies = torsion_energy(k=k,angle=dof_data, offset=offset_torsion)
            en = pool_energy(g=g, energies=energies, term=term, suffix=suffix)

        return en, energies
