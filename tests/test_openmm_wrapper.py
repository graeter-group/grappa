"""
test whether the openmm wrapper works as expected by implementing the identity and comparing the results.
"""
import pytest

# only run if openmm is installed:
pytest.importorskip("openmm", reason="OpenMM not available")

@pytest.mark.slow
def test_openmm_wrapper_identity():

    from grappa.wrappers.openmm_wrapper import OpenmmGrappa
    from openmm.app import PDBFile, ForceField, Modeller
    from openmm.unit import angstrom, kilocalorie_per_mole
    from grappa.utils.openmm_utils import get_energies
    import numpy as np
    import copy
    from grappa.utils.openmm_utils import get_subtopology, OPENMM_ION_RESIDUES, OPENMM_WATER_RESIDUES
    
    from grappa.data import Molecule, Parameters
    import torch
    from grappa.constants import BONDED_CONTRIBUTIONS
    from pathlib import Path

    thisdir = Path(__file__).parent

    #####################
    pdbfile = PDBFile(str(thisdir/'testfiles/T4.pdb'))
    classical_ff = ForceField('amber99sbildn.xml', 'tip3p.xml')
    
    # Solvate and prepare the system
    modeller = Modeller(pdbfile.topology, pdbfile.positions)
    # modeller.deleteWater()
    modeller.addHydrogens(classical_ff)
    # modeller.addSolvent(classical_ff, model="tip3p", padding=1.0 * unit.nanometers, neutralize=True)
    topology = modeller.getTopology()
    positions_ = modeller.getPositions()

    system = classical_ff.createSystem(topology)
    original_system = copy.deepcopy(system)
    #####################

    # create a graph that stores reference parameters:
    sub_topology = get_subtopology(topology, exclude_residues=OPENMM_WATER_RESIDUES+OPENMM_ION_RESIDUES)
    mol = Molecule.from_openmm_system(openmm_system=system, openmm_topology=sub_topology)
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
    positions = positions_.value_in_unit(angstrom)
    positions = np.array([positions])

    en, grads = get_energies(system, positions)
    orig_en, orig_grads = get_energies(original_system, positions)

    assert np.allclose(grads, orig_grads, atol=1e-3)
    # cannot assert energy proximity, because only energy differences between conformations are meaningful


def openmm_wrapper_pdb(pdb_path:str):
    
    from openmm.app import ForceField, PDBFile, Modeller
    from openmm import unit
    from grappa import OpenmmGrappa
    from grappa.utils.openmm_utils import get_energies
    from grappa.constants import get_grappa_units_in_openmm
    import numpy as np

    # Load PDB and topology (this is a test system with two ubiquitins and a few water molecules)
    pdbfile = PDBFile(pdb_path)
    topology = pdbfile.topology

    # Load classical force field
    classical_ff = ForceField("amber99sbildn.xml", "tip3p.xml")

    # Solvate and prepare the system
    # we do not solvate because it gets unnecessarily slow. we just keep a few water molecules from the raw pdb instead.
    modeller = Modeller(topology, pdbfile.positions)
    # modeller.deleteWater()
    modeller.addHydrogens(classical_ff)
    # modeller.addSolvent(classical_ff, model="tip3p", padding=1.0 * unit.nanometers, neutralize=True)
    topology = modeller.getTopology()
    positions = modeller.getPositions()

    # Create classical and Grappa systems
    orig_system = classical_ff.createSystem(topology)
    grappa_ff = OpenmmGrappa.from_tag("grappa-1.4.1-light")
    system = grappa_ff.parametrize_system(orig_system, topology, plot_dir=None)

    # Compute gradients
    DISTANCE_UNIT = get_grappa_units_in_openmm()["LENGTH"]
    positions = np.array([positions.value_in_unit(DISTANCE_UNIT)])
    orig_energy, orig_grads = get_energies(orig_system, positions)
    grappa_energy, grappa_grads = get_energies(system, positions)

    # Compute RMSE between Amber and Grappa gradients
    rmse = np.sqrt(np.mean((orig_grads - grappa_grads) ** 2))
    
    # Assert RMSE < 10
    assert rmse < 10, f"Gradient cRMSE between amber99 and grappa-light too high: {rmse}"
    
    # Assert max force deviation is < 50 kcal/mol/A:
    assert np.max(np.abs(orig_grads - grappa_grads)) < 50, f"Max force deviation between amber99 and grappa-light too high: {np.max(np.abs(orig_grads - grappa_grads))}"


@pytest.mark.slow
def test_openmm_wrapper_monomer():
    """Test whether OpenMM wrapper works by comparing Grappa and Amber gradients."""
    from pathlib import Path

    thisdir = Path(__file__).parent
    pdb_path = str(thisdir/'testfiles/T4.pdb')
    openmm_wrapper_pdb(pdb_path)


@pytest.mark.slow
def test_openmm_wrapper_multimer():
    """Test whether OpenMM wrapper works by comparing Grappa and Amber gradients."""
    from pathlib import Path

    thisdir = Path(__file__).parent
    pdb_path = str(thisdir/'testfiles/two_ubqs.pdb')
    openmm_wrapper_pdb(pdb_path)