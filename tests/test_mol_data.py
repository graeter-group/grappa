import numpy as np
import pytest

from grappa.data import Dataset, Molecule, MolData
from grappa.utils import graph_utils


# QM energies of a bromine-containing molecule are of the order of -2e6 kcal/mol, where float32 values are 0.25 kcal/mol apart.
ABSOLUTE_ENERGY = -2153976.0
RELATIVE_ENERGIES = np.array([0.37016205, -6.58217908, -2.75611439, 4.06010352, -0.11934217, 1.01234567])


def get_bromomethane_moldata(energy, nonbonded_energy):
    # CH3Br, atom 2 is the bromine
    molecule = Molecule(
        atoms=[1, 2, 3, 4, 5],
        bonds=[(1, 2), (1, 3), (1, 4), (1, 5)],
        impropers=[],
        atomic_numbers=[6, 35, 1, 1, 1],
        partial_charges=[-0.1, -0.2, 0.1, 0.1, 0.1],
    )
    rng = np.random.default_rng(0)
    n_confs = len(energy)
    xyz = rng.normal(size=(n_confs, 5, 3))
    gradient = rng.normal(size=(n_confs, 5, 3))
    return MolData(
        molecule=molecule,
        xyz=xyz,
        energy=energy,
        gradient=gradient,
        mol_id='CBr',
        ff_energy={'reference_ff': {'nonbonded': nonbonded_energy}},
        ff_gradient={'reference_ff': {'nonbonded': 0.1 * gradient}},
    )


def test_float32_rounding_is_relevant():
    # guard: the test energies are not representable in float32 to better than 0.01 kcal/mol
    energy = ABSOLUTE_ENERGY + RELATIVE_ENERGIES
    assert np.abs(energy.astype(np.float32).astype(np.float64) - energy).max() > 0.01


def test_to_dgl_keeps_relative_energies():
    energy = ABSOLUTE_ENERGY + RELATIVE_ENERGIES
    nonbonded_energy = -80.0 + 0.5 * RELATIVE_ENERGIES
    g = get_bromomethane_moldata(energy, nonbonded_energy).to_dgl()

    energy_qm = g.nodes['g'].data['energy_qm'].double().numpy().flatten()
    assert np.allclose(energy_qm, energy - energy.mean(), atol=1e-5, rtol=0)

    energy_nonbonded = g.nodes['g'].data['energy_reference_ff_nonbonded'].double().numpy().flatten()
    assert np.allclose(energy_nonbonded, nonbonded_energy - nonbonded_energy.mean(), atol=1e-5, rtol=0)


def test_reference_energies_keep_relative_energies():
    energy = ABSOLUTE_ENERGY + RELATIVE_ENERGIES
    nonbonded_energy = -80.0 + 0.5 * RELATIVE_ENERGIES
    dataset = Dataset.from_moldata([get_bromomethane_moldata(energy, nonbonded_energy)], subdataset='test')
    dataset.create_reference(ref_terms=['nonbonded'])
    g, _ = dataset[0]

    reference_energy = energy - nonbonded_energy
    energy_ref = graph_utils.get_energies(g, suffix='_ref').double().numpy()
    assert np.allclose(energy_ref, reference_energy - reference_energy.mean(), atol=1e-5, rtol=0)


def test_center_energy():
    energy = np.array([ABSOLUTE_ENERGY + 1.0, np.nan, ABSOLUTE_ENERGY - 1.0, ABSOLUTE_ENERGY + 3.0])
    centered = MolData.center_energy(energy)
    assert centered.dtype == np.float64
    assert np.isnan(centered[1])
    assert np.allclose(centered[[0, 2, 3]], [0.0, -2.0, 2.0], atol=1e-9, rtol=0)

    all_nan = np.full(3, np.nan)
    assert np.all(np.isnan(MolData.center_energy(all_nan)))
