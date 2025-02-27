import pytest
import pandas as pd
import shutil
import numpy as np

from grappa.training.evaluator import Evaluator
from grappa.data import Dataset
from grappa.data import GraphDataLoader
from grappa.utils.data_utils import get_data_path

def test_download():
    # first, delete the dataset if it exists:
    dspath = get_data_path()/'datasets'/'small_example'
    if dspath.exists():
        shutil.rmtree(dspath)
    if (get_data_path()/'dataset_tags.csv').exists():
        dataset_csv = pd.read_csv(get_data_path()/'dataset_tags.csv')
        if 'small_example' in dataset_csv['tag'].values:
            dataset_csv = dataset_csv[dataset_csv['tag']!='small_example']
            dataset_csv.to_csv(get_data_path()/'dataset_tags.csv', index=False)

    dataset = Dataset.from_tag('small_example')
    assert dataset is not None
    assert len(dataset) > 1

def test_dataloader():
    dataset = Dataset.from_tag('small_example')
    dataloader = GraphDataLoader(dataset, batch_size=2, shuffle=True)
    for batch,_ in dataloader:
        assert batch is not None
        assert 'xyz' in batch.nodes['n1'].data.keys()


def test_unbatching():
    ds = Dataset.from_tag('small_example')

    metric_dicts = []
    for batchsize in [1,10]:
        loader = GraphDataLoader(ds, batch_size=batchsize, shuffle=True, conf_strategy='max')

        evaluator = Evaluator(suffix='_gaff-2.11_total', suffix_ref='_qm')
        
        for batch, dsnames in loader:
            evaluator.step(batch, dsnames)

        d = evaluator.pool()
        metric_dicts.append(d)

    dsname = list(metric_dicts[0].keys())[0]
    assert np.allclose(np.array(list(metric_dicts[0][dsname].values())), np.array(list(metric_dicts[1][dsname].values())))
