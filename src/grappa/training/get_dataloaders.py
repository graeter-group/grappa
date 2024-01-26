from grappa.data import Dataset, GraphDataLoader
from pathlib import Path
from typing import List, Dict, Tuple, Union
import torch
import json
from grappa.utils.dataset_utils import get_path_from_tag


def get_dataloaders(datasets:List[Union[Path, str, Dataset]], conf_strategy:Union[str, int]=100, train_batch_size:int=1, val_batch_size:int=1, test_batch_size:int=1, train_loader_workers:int=1, val_loader_workers:int=1, test_loader_workers:int=1, seed:int=0, pin_memory:bool=True, splitpath:Path=None, partition:Union[Tuple[float,float,float], Tuple[Tuple[float,float,float],Dict[str, Tuple[float, float, float]]]]=(0.8,0.1,0.1), pure_train_datasets:List[Union[Path, str, Dataset]]=[], pure_val_datasets:List[Union[Path, str, Dataset]]=[], pure_test_datasets:List[Union[Path, str, Dataset]]=[], subsample_train:Dict[str, int]={}, subsample_val:Dict[str, int]={}, subsample_test:Dict[str, int]={}, weights:Dict[str,str]={}, balance_factor:float=0., classical_needed:bool=False, in_feat_names:List[str]=None, save_splits:Union[str,Path]=None, val_conf_strategy:int=200)->Tuple[GraphDataLoader, GraphDataLoader, GraphDataLoader]:
    """
    This function returns train, validation, and test dataloaders for a given list of datasets.

    Args:
        datasets (List[Path]): List of paths to the datasets.
        conf_strategy (str, optional): Strategy for sampling conformations when batching. Defaults to 'mean'.
        train_batch_size (int, optional): Batch size for the training dataloader. Defaults to 1.
        val_batch_size (int, optional): Batch size for the validation dataloader. Defaults to 1.
        test_batch_size (int, optional): Batch size for the test dataloader. Defaults to 1.
        train_loader_workers (int, optional): Number of worker processes for the training dataloader. Defaults to 1.
        val_loader_workers (int, optional): Number of worker processes for the validation dataloader. Defaults to 2.
        test_loader_workers (int, optional): Number of worker processes for the test dataloader. Defaults to 2.
        pin_memory (bool, optional): Whether to pin memory for the dataloaders. Defaults to True.
        splitpath (Path, optional): Path to the split file. If provided, the function will load the split from this file. If not, it will generate a new split. Defaults to None.
        partition (Union[Tuple[float,float,float], Dict[str, Tuple[float, float, float]]], optional): Partition of the dataset into train, validation, and test. Can be a tuple of three floats or a dictionary with 'train', 'val', and 'test' keys. Defaults to (0.8,0.1,0.1).
        pure_..._datasets: list of paths to datasets that are only for one specific set type, independent on which mol_ids occur. this can be used to be independent of the splitting by mol_ids. in the case of the espaloma benchmark, this is used to have the same molecules in the test and train set (where training is on rna-diverse-conformations and testing on rna-trinucleotide-conformations)
        subsample_... : dictionary of dsname and a float between 0 and 1 specifying the subsampling factor (dsname is stem of the dspaths. subsampling is applied after splitting)
        weights (Dict[str,str], optional): Dictionary mapping subdataset names to weights. If a subdataset name is not in the dictionary, it is assigned a weight of 1.0. If a subdataset has e.g. weight factor 2, it is sampled 2 times more often per epoch than it would be with factor 1. The total number of molecules sampled per epoch is unaffected, i.e. some molecules will not be sampled at all if the weight of other datasets is > 1. Defaults to {}.
        balance_factor (float, optional): parameter between 0 and 1 that balances sampling of the train datasets: 0 means that the molecules are sampled uniformly across all datasets, 1 means that the probabilities are re-weighted such that the sampled number of molecules per epoch is the same for all datasets. The weights assigned in 'weights' are multiplied by the weight factor obtained from balancing. Defaults to 0.
        save_splits (Union[str,Path], optional): Path to save the split file as json. If None, the split is not saved. Defaults to None.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: A tuple containing the train, validation, and test dataloaders.
    """

    paths = []
    # Get the dataset
    for i, dataset in enumerate(datasets):
        if isinstance(dataset, str):
            try:
                # if it is a valid tag, intialize from tag
                dataset = get_path_from_tag(tag=dataset)
            except ValueError:
                # assume that it is a path
                dataset = Path(dataset)
        if isinstance(dataset, Path):
            assert dataset.exists(), f"Dataset path {dataset} does not exist."
            datasets[i] = str(dataset)
            paths.append(dataset)

        elif not isinstance(dataset, Dataset):
            raise ValueError(f"Dataset must be a Path or Dataset, but got {type(dataset)}")
        
    if not len(paths) == len(set(paths)):
        raise ValueError(f"Duplicate paths in dataset list:\n{paths}")

    dataset = Dataset()
    for ds in datasets:
        if isinstance(ds, Dataset):
            dataset += ds
        elif isinstance(ds, Path) or isinstance(ds, str):
            print(f"Loading dataset from {ds}...")
            dataset += Dataset.load(ds)
                
        else:
            raise ValueError(f"Unknown type for dataset: {type(ds)}")

    # Remove uncommon features for enabling batching
    dataset.remove_uncommon_features()
    keep_feats = None
    if not in_feat_names is None:
        keep_feats = ['gradient_ref'] + in_feat_names
        if classical_needed:
            keep_feats += ['gradient_classical']

        dataset.clean(keep_feats=keep_feats)

    # Get the split ids
    # load if path in config, otherwise generate. For now, always generate it.
    if splitpath is not None:
        split_ids = json.load(open(splitpath, 'r'))
        print(f'Using split ids from {splitpath}')
        split_ids = dataset.calc_split_ids(partition=partition, seed=seed, existing_split=split_ids)
    else:
        split_ids = dataset.calc_split_ids(partition=partition, seed=seed)

    if not save_splits is None:
        if isinstance(save_splits, str):
            save_splits = Path(save_splits)
        assert isinstance(save_splits, Path)
        save_splits.parent.mkdir(parents=True, exist_ok=True)
        with open(save_splits, 'w') as f:
            json.dump(split_ids, f, indent=4)

    tr, vl, te = dataset.split(*split_ids.values())

    # Add pure datasets
    #########################################################
    for ds in pure_train_datasets:
        if isinstance(ds, Dataset):
            tr += ds
        elif isinstance(ds, Path) or isinstance(ds, str):
            try:
                # if it is a valid tag, intialize from tag
                ds = get_path_from_tag(tag=ds)
            except ValueError:
                # assume that it is a path
                ds = Path(ds)
            if ds in paths:
                raise ValueError(f"Pure train dataset {ds} already in datasets list.")
            print(f"Loading dataset from {ds}...")
            tr += Dataset.load(ds)
        else:
            raise ValueError(f"Unknown type for dataset: {type(ds)}")
    
    for ds in pure_val_datasets:
        if isinstance(ds, Dataset):
            vl += ds
        elif isinstance(ds, Path) or isinstance(ds, str):
            try:
                # if it is a valid tag, intialize from tag
                ds = get_path_from_tag(tag=ds)
            except ValueError:
                # assume that it is a path
                ds = Path(ds)
            if ds in paths:
                raise ValueError(f"Pure train dataset {ds} already in datasets list.")
            print(f"Loading dataset from {ds}...")
            vl += Dataset.load(ds)
        else:
            raise ValueError(f"Unknown type for dataset: {type(ds)}")
        
    for ds in pure_test_datasets:
        if isinstance(ds, Dataset):
            te += ds
        elif isinstance(ds, Path) or isinstance(ds, str):
            try:
                # if it is a valid tag, intialize from tag
                ds = get_path_from_tag(tag=ds)
            except ValueError:
                # assume that it is a path
                ds = Path(ds)
            if ds in paths:
                raise ValueError(f"Pure train dataset {ds} already in datasets list.")
            print(f"Loading dataset from {ds}...")
            te += Dataset.load(ds)
        else:
            raise ValueError(f"Unknown type for dataset: {type(ds)}")
        
    for ds, subsampling_dict in zip([tr, vl, te], [subsample_train, subsample_val, subsample_test]):
        for dsname, subsampling_factor in subsampling_dict.items():
            ds.subsample(subsampling_factor, dsname)
        

    if len(pure_train_datasets) > 0:
        tr.remove_uncommon_features()
        tr.clean(keep_feats=keep_feats)
    if len(pure_val_datasets) > 0:
        vl.remove_uncommon_features()
        vl.clean(keep_feats=keep_feats)
    if len(pure_test_datasets) > 0:
        te.remove_uncommon_features()
    #########################################################

    #########################################################
    REPLACE_HYBRIDIZATION = False
    if REPLACE_HYBRIDIZATION:
        # load a Hybridization predictor:
        from grappa.models.graph_attention import GrappaGNN
        class GraphModel(torch.nn.Module):
            def __init__(self, n_classes=6):
                super().__init__()
                
                self.gnn = GrappaGNN(in_feat_name=['partial_charge', 'atomic_number', 'degree', 'ring_encoding'], out_feats=32, n_conv=0, n_att=1)
                self.lin_1 = torch.nn.Linear(32, 32)
                self.lin_out = torch.nn.Linear(32, n_classes)

            def forward(self, g):
                # Forward pass through the model
                g = self.gnn(g)
                scores = g.nodes['n1'].data['h']
                scores = torch.nn.functional.elu(scores)
                scores = self.lin_1(scores)
                scores = torch.nn.functional.elu(scores)
                scores = self.lin_out(scores)
                return scores

        # load the state dict:
        checkpoint_path = '/hits/fast/mbm/seutelf/grappa/tests/hybridization_feature/checkpoints/best-model-v3.ckpt'
        state_dict = torch.load(checkpoint_path)['state_dict']
        # remove the prefix:
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        # load the model:
        model = GraphModel()
        model.load_state_dict(state_dict)

        print('Loaded Hybridization predictor from:', checkpoint_path)

        model = torch.nn.Sequential(
            GraphModel(),
            torch.nn.Softmax(dim=1)
        )

        model.eval()

        model = model.to('cpu')

        import copy

        # now overwrite the sp_hybridization feature with the predicted one:
        def overwrite_sp_hybridization(ds):
            print('Overwriting sp_hybridization feature with predicted one...')
            for i, _ in enumerate(ds):
                print(i, end='\r')
                with torch.no_grad():
                    new_hybrid = model(ds[i][0]).detach().clone().cpu()
                    new_hybrid = copy.deepcopy(new_hybrid)
                ds[i][0].nodes['n1'].data['sp_hybridization'] = new_hybrid
            print()
            return ds

        print('Overwriting sp_hybridization feature with predicted one...')
        tr = overwrite_sp_hybridization(tr)
        vl = overwrite_sp_hybridization(vl)
        te = overwrite_sp_hybridization(te)

    #########################################################

    # Get the dataloaders
    train_loader = GraphDataLoader(tr, batch_size=train_batch_size, shuffle=True, num_workers=train_loader_workers, pin_memory=pin_memory, conf_strategy=conf_strategy, weights=weights, balance_factor=balance_factor, drop_last=True)
    val_loader = GraphDataLoader(vl, batch_size=val_batch_size, shuffle=False, num_workers=val_loader_workers, pin_memory=pin_memory, conf_strategy=val_conf_strategy, drop_last=False)
    test_loader = GraphDataLoader(te, batch_size=test_batch_size, shuffle=False, num_workers=test_loader_workers, pin_memory=pin_memory, conf_strategy='max', drop_last=False)

    return train_loader, val_loader, test_loader