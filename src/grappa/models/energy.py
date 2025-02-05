from typing import Optional
import torch
from torch import Tensor
import dgl
from dgl import DGLGraph
from grappa.utils.dgl_utils import grad_available
 
from grappa.models.internal_coordinates import InternalCoordinates
import copy

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

def get_partial_params(g: DGLGraph, term: str, param: str, grappa_suffix: str="", trad_suffix: str="_ref", min_grappa_atoms: Optional[dict]=None) -> Tensor:
    """
    Retrieves the partial parameters for a given term from the graph.

    Partial parameterization means that only interactions containing at least a spezified number of grappa atoms are parameterized with grappa parameters and that all other interactions are parameterized with traditional parameters.

    args:
        g (DGLGraph): The graph containing the data.
        term (str): The term for which parameters are retrieved.
        param (str): The parameter name.
        grappa_suffix (str): The suffix for the grappa parameter.
        trad_suffix (str): The suffix for the traditional parameter.
        min_grappa_atoms (Optional[dict]): A dictionary containing the minimum number of grappa atoms for each interaction to be parameterized with grappa. If None, the default value of one grappa atom is used for each interaction.
    """

    if param + grappa_suffix not in g.nodes[term].data.keys():
        raise RuntimeError(f"{term} has no {param}{grappa_suffix} attribute")
    
    if param + trad_suffix not in g.nodes[term].data.keys():
        raise RuntimeError(f"{term} has no {param}{trad_suffix} attribute")
    
    if "num_grappa_atoms" not in g.nodes[term].data.keys():
        raise RuntimeError(f"{term} has no num_grappa_atoms attribute")

    if min_grappa_atoms is None:
        min_grappa_atoms = {"n2": 1, "n3": 1, "n4": 1, "n4_improper": 1}

    is_grappa_interaction = g.nodes[term].data["num_grappa_atoms"] >= min_grappa_atoms[term]
    
    # For propers and impropers, we need to adjust the traditional parameters to match the periodicity of the grappa parameters.
    # This is necessary because the grappa parameters may have different periodicities, and we need to ensure that the traditional
    # parameters are aligned with the same periodicity to correctly compute the energy contributions.
    if "n4" in term:
        _, n_periodicity = g.nodes[term].data[param + grappa_suffix].shape
        is_grappa_interaction = is_grappa_interaction.repeat(n_periodicity,1).permute((1, 0))
        return is_grappa_interaction * g.nodes[term].data[param + grappa_suffix] + (~is_grappa_interaction) * g.nodes[term].data[param + trad_suffix][:,:n_periodicity] 

    return is_grappa_interaction * g.nodes[term].data[param + grappa_suffix] + (~is_grappa_interaction) * g.nodes[term].data[param + trad_suffix] 


def get_params(g: DGLGraph, term:str, param:str, suffix:str, partial_param: bool=False, min_grappa_atoms: Optional[dict]=None) -> Tensor:
    """
    Retrieves the parameters for a given term from the graph.

    retrieved.
    Args:
        g (DGLGraph): The graph containing the data.
        term (str): The term for which parameters are retrieved.
        param (str): The parameter name.
        suffix (str): The suffix for the parameter.
        partial_param (bool): Whether to use partial parameterization. If True, the arg suffix is ignored. Defaults to False.
        min_grappa_atoms (Optional[dict]): A dictionary containing the minimum number of grappa atoms for each interaction to be parameterized with grappa. If None, the default value of one grappa atom is used for each interaction. Only used if partial_param is True.

    Returns:
        Tensor: The retrieved parameters.
    """
    if partial_param:
        return get_partial_params(g, term, param, min_grappa_atoms=min_grappa_atoms)
    else:
        if param+suffix not in g.nodes[term].data.keys():
            raise RuntimeError(f"{term} has no {param}{suffix} attribute")

        return g.nodes[term].data[param+suffix]


class Energy(torch.nn.Module):
    def __init__(self, terms:list=["bond", "angle", "torsion", "improper"], suffix:str="", offset_torsion:bool=False, write_suffix=None, gradients:bool=True, gradient_contributions:bool=False, partial_param:bool=False, min_grappa_atoms: Optional[dict]=None):
        """
        Module that writes the energy of molecular conformations into a dgl graph. First, internal coordinates such as torsional angles, angles and distances are calculated, then their energy contributions are added and stored at g.nodes["g"].data["energy"] and g.nodes["g"].data["energy_"+term] for each term. The gradients of the total energy w.r.t. the xyz coordinates are calculated and stored at g.nodes["n1"].data["gradient"].
        
        ----------
        Args:
        ----------
        terms: list of terms to be considered. must be a subset of ["bond", "angle", "torsion", "improper"]
        suffix: suffix of the parameters stored in the graph.
        offset_torsion: whether to include the constant offset term (that makes the contribution positive) in the torsion energy calculation
        write_suffix: suffix of the energy and gradient attributes written to the graph. if None, write_suffix is set to suffix.
        gradients: whether to calculate gradients of the total energy w.r.t. the xyz coordinates. This cannot be done if the context does not allow autograd (e.g. if this function is called while torch.no_grad() is active).
        partial_param: whether to use partial parameterization. default is False.
        min_grappa_atoms: A dictionary containing the minimum number of grappa atoms for each interaction to be parameterized with grappa. If None, the default value of one grappa atom is used for each interaction. Only used if partial_param is True.
        """
        super().__init__()

        if not isinstance(terms, list):
            raise ValueError("terms must be a list")
        self.offset_torsion = offset_torsion
        self.suffix = suffix
        self.write_suffix = write_suffix if not write_suffix is None else suffix
        self.gradients = gradients
        self.gradient_contributions = gradient_contributions
        self.geom = InternalCoordinates()
        self.partial_param = partial_param
        self.TERM_TO_LEVEL = {
            "bond": "n2",
            "angle": "n3",
            "proper": "n4",
            "torsion": "n4", # also allow "torsion" for "proper" for backwards compatibility
            "improper": "n4_improper"
        }
        self.LEVEL_TO_TERM = {v: k for k, v in self.TERM_TO_LEVEL.items()}
        self.LEVEL_TO_TERM["n4"] = "proper"
        self.terms = [self.TERM_TO_LEVEL[t] for t in terms]
        if isinstance(min_grappa_atoms, dict):
            min_grappa_atoms = {self.TERM_TO_LEVEL[k]: v for k, v in min_grappa_atoms.items()}
        self.min_grappa_atoms = min_grappa_atoms

    def forward(self, g):
        """
        First, torsional angles, bonds angles and distances are calculated, then their energy contributions are added and written under g.nodes["g"].data["energy"+write_suffix] and g.nodes["g"].data["energy_"+term+write_suffix] for each term.
        Also stores the individual contributions of shape  at g.nodes[term].data["energy"+write_suffix] for each term.
        """
        if self.gradient_contributions and not self.gradients:
            raise ValueError("Gradient contributions cannot be calculated if gradients are not enabled.")

        grad_enabled = copy.deepcopy(torch.is_grad_enabled())
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
            termname = self.LEVEL_TO_TERM[term]
            if not term in g.ntypes:
                raise ValueError(f"term {term} not in g.ntypes")

            # get the energy contribution of this term in shape (num_batch, num_confs)

            contrib, tuple_energies = Energy.get_energy_contribution(g, term=term, suffix=self.suffix, offset_torsion=self.offset_torsion, partial_param=self.partial_param, min_grappa_atoms=self.min_grappa_atoms)
            if not contrib is None:
                if self.gradient_contributions and grad_available():
                    # condition under which we can calculate the gradient:
                    if contrib.shape[0] > 0 and not torch.all(contrib==0):
                        grad = torch.autograd.grad(contrib.sum(), g.nodes["n1"].data["xyz"], retain_graph=True, create_graph=True, allow_unused=True)[0]
                    else:
                        grad = torch.zeros_like(g.nodes["n1"].data["xyz"])
                    g.nodes["n1"].data["gradient_"+self.write_suffix+termname] = grad
                
                energy += contrib
                g.nodes["g"].data["energy_"+self.write_suffix+termname] = contrib.detach()
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
    def get_energy_contribution(g, term, suffix, offset_torsion=True, partial_param:bool=False, min_grappa_atoms: Optional[dict]=None):
        """
        Returns:
        en, energies
        where en is the total energy contribution from this term and energies is a tensor of shape (num_tuples, num_confs) containing the energy contribution of each tuple individually.
        """
        if term not in g.ntypes:
            return None, None

        k = get_params(g, term, "k", suffix, partial_param=partial_param, min_grappa_atoms=min_grappa_atoms)
        dof_data = g.nodes[term].data["x"]

        if term in ["n2", "n3"]:
            eq = get_params(g, term, "eq", suffix, partial_param=partial_param, min_grappa_atoms=min_grappa_atoms)
            energies = harmonic_energy(k=k, eq=eq, distances=dof_data)
            en = pool_energy(g=g, energies=energies, term=term, suffix=suffix)
  
        if term in ["n4", "n4_improper"]:
            energies = torsion_energy(k=k,angle=dof_data, offset=offset_torsion)
            en = pool_energy(g=g, energies=energies, term=term, suffix=suffix)

        return en, energies
