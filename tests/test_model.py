import pytest
import torch
import shutil
import pandas as pd

from grappa.models import GrappaModel, Energy
from grappa.models.internal_coordinates import InternalCoordinates
from grappa.data import Dataset
from grappa.utils.model_loading_utils import model_from_tag, get_path_from_tag, get_model_dir, store_with_comment


def test_model():
    ds = Dataset.from_tag('small_example')

    model = GrappaModel()
    
    graph = ds[0][0]

    assert 'k' not in graph.nodes['n2'].data.keys()
    assert 'eq' not in graph.nodes['n2'].data.keys()

    updated_graph = model(graph)

    assert 'k' in updated_graph.nodes['n2'].data.keys()
    assert 'eq' in updated_graph.nodes['n2'].data.keys()



def perform_backward_passes(device):
    ds = Dataset.from_tag('small_example')

    model = GrappaModel(gnn_attentional_layers=2, graph_node_features=32, gnn_width=32,
                        symmetric_transformer_width=32)

    energy_module = Energy()
    coordinates_module = InternalCoordinates()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    
    # for each device, perform two backward passes:
    for graph in (ds[0][0], ds[1][0]):

        graph = graph.to(device)
        model.to(device)
        energy_module.to(device)
        coordinates_module.to(device)

        updated_graph = model(graph)

        updated_graph = coordinates_module(updated_graph)
        updated_graph = energy_module(updated_graph)

        energy_qm = updated_graph.nodes['g'].data['energy_qm'] - updated_graph.nodes['g'].data['energy_reference_ff_nonbonded']
        energy_qm -= energy_qm.mean(dim=1, keepdim=True)
        energy_grappa = updated_graph.nodes['g'].data['energy']

        assert energy_qm.shape == energy_grappa.shape
        energy_mse = torch.mean((energy_qm - energy_grappa)**2)

        force_qm = - (updated_graph.nodes['n1'].data['gradient_qm'] - updated_graph.nodes['n1'].data['gradient_reference_ff_nonbonded'])
        force_grappa = updated_graph.nodes['n1'].data['gradient']

        force_mse = torch.mean((force_qm - force_grappa)**2)

        loss = energy_mse + force_mse

        # assert there are no nans in the loss:
        assert not torch.any(torch.isnan(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def test_backward_pass():
    perform_backward_passes('cpu')


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda available")
def test_backward_pass_cuda():
    perform_backward_passes('cuda')


def test_downloaded_model():
    ds = Dataset.from_tag('small_example')
    ds.create_reference()

    modeltag = 'grappa-1.4.1-light'

    # remove the model if present to test re-downloading it:
    # remove tag entry:
    csv_path = get_model_dir() / 'models.csv'
    if csv_path.exists():
        model_csv = pd.read_csv(get_model_dir() / 'models.csv', comment='#', dtype=str)
        if modeltag in model_csv['tag'].values:
            model_csv = model_csv[model_csv['tag'] != modeltag]
            store_with_comment(model_csv, get_model_dir() / 'models.csv')
    modelpath = get_model_dir()/modeltag
    if modelpath.exists():
        shutil.rmtree(modelpath.parent)

    # load the model:
    model = model_from_tag(modeltag)

    model = torch.nn.Sequential(model, Energy()).eval()
    device = 'cpu'
    mol = ds[0][0]
    mol = mol.to(device)
    model.to(device)
    mol = model(mol)

    # assert that the force crmse is below 10 kcal/mol/angstroem
    qm_minus_nonbonded_grad = mol.nodes['n1'].data['gradient_ref']
    grappa_grad = mol.nodes['n1'].data['gradient']
    crmse = torch.mean((qm_minus_nonbonded_grad - grappa_grad)**2)**0.5

    if crmse > 10:
        raise ValueError(f"Internal Error. Force crmse is too high: {crmse}, should be below 10 kcal/mol/angstroem!")