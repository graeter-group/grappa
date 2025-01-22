import pytest  # add to requirements.txt
from dgl import DGLGraph
import torch
import random

from grappa.data import Dataset, AnnotateGrappaAtomsNInteractions
from grappa.data.transforms import get_khop_connected_atoms, one_hot_to_idx, idx_to_one_hot

class TestAnnotateGrappaAtomsNInteractions:
    @pytest.fixture
    def aa_graph(self) -> DGLGraph:
        d = Dataset.from_tag('dipeptides-300K-amber99')
        graph, subdata = d[0]
        return graph

    def test_get_khop_connected_atoms(self, aa_graph):
        one_hop_connected_atoms = get_khop_connected_atoms(aa_graph, khop=1, start_atom=2)
        assert torch.equal(one_hop_connected_atoms, torch.tensor([0, 2, 3, 4, 5]))
        two_hop_connected_atoms = get_khop_connected_atoms(aa_graph, khop=2, start_atom=2)
        assert torch.equal(two_hop_connected_atoms, torch.tensor([0, 1, 2, 3, 4, 5, 6]))
        three_hop_connected_atoms = get_khop_connected_atoms(aa_graph, khop=3, start_atom=2)
        assert torch.equal(three_hop_connected_atoms, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 16]))

    def test_one_hot_to_idx(self):
        one_hot = torch.Tensor([0, 1, 0, 0, 1])
        idx = one_hot_to_idx(one_hot)
        assert torch.equal(idx, torch.tensor([1, 4]))
    
    def test_idx_to_one_hot(self):
        idx = torch.tensor([1, 4])
        one_hot = idx_to_one_hot(idx, 5)
        assert torch.equal(one_hot, torch.tensor([0, 1, 0, 0, 1]))

    def test_annotate_grappa_atoms_n_interactions_deterministic(self, aa_graph):
       graph = AnnotateGrappaAtomsNInteractions(deterministic=True)(aa_graph)
       # grappa atoms = tensor([0,1,2,3,4,5,6])
       assert torch.equal(graph.nodes["n1"].data["grappa_atom"], torch.tensor([1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=torch.float32))
       assert torch.equal(graph.nodes["n2"].data["num_grappa_atoms"], torch.tensor([2,2,2,2,2,2,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=torch.int32))
       assert torch.equal(graph.nodes["n3"].data["num_grappa_atoms"], torch.tensor([3,3,3,2,2,3,3,3,3,3,3,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=torch.int32))
       assert torch.equal(graph.nodes["n4"].data["num_grappa_atoms"], torch.tensor([4,4,4,4,4,4,2,3,3,1,2,0,2,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,3,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=torch.int32))
       assert torch.equal(graph.nodes["n4_improper"].data["num_grappa_atoms"], torch.tensor([4,4,4,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=torch.int32))

    def test_annotate_grappa_atoms_n_interactions_all_grappa(self, aa_graph):
        random.seed(0)
        graph = AnnotateGrappaAtomsNInteractions(deterministic=False)(aa_graph)
        assert torch.equal(graph.nodes["n1"].data["grappa_atom"], torch.ones((46,), dtype=torch.float32))
        assert torch.equal(graph.nodes["n2"].data["num_grappa_atoms"], 2*torch.ones((47,), dtype=torch.int32))
        assert torch.equal(graph.nodes["n3"].data["num_grappa_atoms"], 3*torch.ones((80,), dtype=torch.int32))
        assert torch.equal(graph.nodes["n4"].data["num_grappa_atoms"], 4*torch.ones((110,), dtype=torch.int32))
        assert torch.equal(graph.nodes["n4_improper"].data["num_grappa_atoms"], 4*torch.ones((42,), dtype=torch.int32))
    
    def test_annotate_grappa_atoms_n_interactions_random(self, aa_graph):
        random.seed(1) # khop=1 and start_atom=16
        graph = AnnotateGrappaAtomsNInteractions(deterministic=False)(aa_graph)
        # grappa atoms = tensor([6, 16]) 
        assert torch.equal(graph.nodes["n1"].data["grappa_atom"], torch.tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.float32))
        assert torch.equal(graph.nodes["n2"].data["num_grappa_atoms"], torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32))
        assert torch.equal(graph.nodes["n3"].data["num_grappa_atoms"], torch.tensor([0, 0, 0, 1, 2, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32))
        assert torch.equal(graph.nodes["n4"].data["num_grappa_atoms"], torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 2, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32))
        assert torch.equal(graph.nodes["n4_improper"].data["num_grappa_atoms"], torch.tensor([1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32))
        