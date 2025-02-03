"""
test whether the openmm wrapper works as expected by implementing the identity and comparing the results.
"""
import pytest

# only run if openmm is installed:
pytest.importorskip("openmm")

@pytest.mark.slow
def test_openmm_wrapper():

    from grappa.wrappers.openmm_wrapper import OpenmmGrappa
    from openmm.app import PDBFile, ForceField
    from openmm.unit import angstrom, kilocalorie_per_mole
    from grappa.utils.openmm_utils import get_energies
    import numpy as np
    import copy
    from grappa.data import Molecule, Parameters
    import torch
    from grappa.constants import BONDED_CONTRIBUTIONS
    from pathlib import Path

    thisdir = Path(__file__).parent

    #####################
    pdb = PDBFile(str(thisdir/'../examples/usage/T4.pdb'))
    topology = pdb.topology
    ff = ForceField('amber99sbildn.xml', 'tip3p.xml')
    system = ff.createSystem(pdb.topology)
    original_system = copy.deepcopy(system)
    #####################

    # create a graph that stores reference parameters:
    mol = Molecule.from_openmm_system(openmm_system=system, openmm_topology=topology)
    params = Parameters.from_openmm_system(openmm_system=system, mol=mol)
    g = mol.to_dgl()
    g = params.write_to_dgl(g)


    class identity(torch.nn.Module):
        """
        Model that simply writes the parameters with suffix from some graph in the given graph.
        """
        def forward(self, graph):
            suffix = '_ref'
            for lvl, param in BONDED_CONTRIBUTIONS:
                graph.nodes[lvl].data[param] = g.nodes[lvl].data[param+suffix]
            return graph

    model = identity()
    model.field_of_view = 5

    # build a grappa model that handles the ML pipeline
    grappa = OpenmmGrappa(model, device='cpu')

    # write grappa parameters to the system:
    system = grappa.parametrize_system(system, topology)
    positions = pdb.positions.value_in_unit(angstrom)
    positions = np.array([positions])

    en, grads = get_energies(system, positions)
    orig_en, orig_grads = get_energies(original_system, positions)

    assert np.allclose(grads, orig_grads, atol=1e-3)
    # cannot assert energy proximity, because only energy differences between conformations are meaningful