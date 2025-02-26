from dgl import DGLGraph
import pytest

from grappa.data import Dataset
from grappa.utils.tuple_indices import get_neighbor_dict
from grappa.utils.graph_utils import get_neighbor_atomic_numbers_dict, get_connected_atoms, get_methly_carbon_atom, get_ace_carbonly_carbon_atom, get_nterminal_alpha_carbon_atom, get_nterminal_carbonly_carbon_atom, get_second_amide_nitrogen_atom, get_nterminal_side_chain_atoms, get_nterminal_atoms_from_dipeptide, get_cterminal_atoms_from_dipeptide, are_connected

@pytest.fixture
def aa_graph() -> DGLGraph:
    d = Dataset.from_tag('dipeptides-300K-amber99')
    graph, subdata = d[0]
    return graph

@pytest.fixture
def neighbor_dict(aa_graph) -> dict:
    return get_neighbor_dict(aa_graph.nodes["n2"].data["idxs"].tolist())

@pytest.fixture
def neighbor_atomic_numbers_dict(aa_graph, neighbor_dict) -> dict:
    return get_neighbor_atomic_numbers_dict(aa_graph, neighbor_dict)

def test_get_connected_atoms_allowed(aa_graph, neighbor_dict):
    connected_atoms = get_connected_atoms(0, neighbor_dict, [])
    connected_atoms.sort()
    assert connected_atoms == [*range(aa_graph.num_nodes("n1"))]

def test_get_connected_atoms_forbidden(neighbor_dict):
    connected_atoms = get_connected_atoms(0, neighbor_dict, [6])
    connected_atoms.sort()
    assert connected_atoms == [0, 1, 2, 3, 4, 5]

def test_get_methly_carbon_atom(aa_graph, neighbor_atomic_numbers_dict):
    assert get_methly_carbon_atom(aa_graph, neighbor_atomic_numbers_dict) == [2, 41]

def test_get_ace_carbonly_carbon_atom(aa_graph, neighbor_dict, neighbor_atomic_numbers_dict):
    assert get_ace_carbonly_carbon_atom(aa_graph, neighbor_dict, neighbor_atomic_numbers_dict) == 0

def test_get_nterminal_alpha_carbon_atom(aa_graph, neighbor_dict, neighbor_atomic_numbers_dict):
    assert get_nterminal_alpha_carbon_atom(aa_graph, neighbor_dict, neighbor_atomic_numbers_dict) == 7

def test_get_nterminal_carbonly_carbon_atom(aa_graph, neighbor_dict, neighbor_atomic_numbers_dict):
    assert get_nterminal_carbonly_carbon_atom(aa_graph, neighbor_dict, neighbor_atomic_numbers_dict) == 8

def test_get_second_amide_nitrogen_atom(aa_graph, neighbor_dict, neighbor_atomic_numbers_dict):
    assert get_second_amide_nitrogen_atom(aa_graph, neighbor_dict, neighbor_atomic_numbers_dict) == 23

def test_get_nterminal_resdiue_atoms(aa_graph):
    nterminal_residue_atoms = get_nterminal_side_chain_atoms(aa_graph)
    nterminal_residue_atoms.sort()
    assert nterminal_residue_atoms == [10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22]

def test_get_nterminal_atoms(aa_graph):
    nterminal_atoms = get_nterminal_atoms_from_dipeptide(aa_graph)
    nterminal_atoms.sort()
    assert nterminal_atoms == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

def test_get_cterminal_atoms(aa_graph):
    cterminal_atoms = get_cterminal_atoms_from_dipeptide(aa_graph)
    cterminal_atoms.sort()
    assert cterminal_atoms == [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]

def test_are_connected(neighbor_dict):
    assert are_connected([0, 1, 2, 3, 4, 5], neighbor_dict) == True
    assert are_connected([0, 1, 2, 3, 4, 5, 45], neighbor_dict) == False