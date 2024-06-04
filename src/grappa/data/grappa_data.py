import pytorch_lightning as pl
from pathlib import Path
from typing import List, Dict, Tuple, Union
import torch
import json
import copy
import logging
from grappa.data import Dataset, GraphDataLoader
from grappa.utils.dataset_utils import get_path_from_tag
from tqdm import tqdm

class GrappaData(pl.LightningDataModule):
    def __init__(self,
                 datasets: List[Union[Path, str, Dataset]],
                 pure_train_datasets: List[Union[Path, str, Dataset]] = [],
                 pure_val_datasets: List[Union[Path, str, Dataset]] = [], pure_test_datasets: List[Union[Path, str, Dataset]] = [],
                 splitpath: Path = None,
                 partition: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 train_batch_size: int = 32,
                 val_batch_size: int = 32,
                 test_batch_size: int = 1,
                 ref_terms: List[str] = ['nonbonded'],
                 train_loader_workers: int = 1,
                 val_loader_workers: int = 1,
                 test_loader_workers: int = 1,
                 conf_strategy: Union[str, int] = 32,
                 seed: int = 0,
                 pin_memory: bool = True,
                 tr_subsampling_factor: float = None,
                 weights: Dict[str, float] = {},
                 balance_factor: float = 0.,
                 in_feat_names: List[str] = None,
                 save_splits: Union[str, Path] = None,
                 val_conf_strategy: int = 200,
                 split_ids: Dict[str, List[str]] = None,
                 keep_features: bool = False
                ):
        """
        This class handles the preparation of train, validation, and test dataloaders for a given list of datasets.

        Args:
            datasets (List[Union[Path, str, Dataset]]): List of datasets, provided as paths, strings, or Dataset objects.
            splitpath (Path, optional): Path to the split file. If provided, the function will load the split from this file. If not, it will generate a new split. Defaults to None.
            partition (Tuple[float, float, float], optional): Partition of the dataset into train, validation, and test splits. Defaults to (0.8, 0.1, 0.1).
            conf_strategy (Union[str, int], optional): Strategy for sampling conformations when batching. Defaults to 32.
            train_batch_size (int, optional): Batch size for the training dataloader. Defaults to 1.
            val_batch_size (int, optional): Batch size for the validation dataloader. Defaults to 1.
            test_batch_size (int, optional): Batch size for the test dataloader. Defaults to 1.
            ref_terms (List[str], optional): Terms for which to use the reference contributions. These will be subtracted from the QM energies and gradients, the result is stored in energy_ref / gradient_ref. Defaults to ['nonbonded'].
            train_loader_workers (int, optional): Number of worker processes for the training dataloader. Defaults to 1.
            val_loader_workers (int, optional): Number of worker processes for the validation dataloader. Defaults to 1.
            test_loader_workers (int, optional): Number of worker processes for the test dataloader. Defaults to 1.
            seed (int, optional): Random seed for reproducibility. Defaults to 0.
            pin_memory (bool, optional): Whether to pin memory for the dataloaders. Defaults to True.
            pure_train_datasets (List[Union[Path, str, Dataset]], optional): List of datasets specifically for training. Defaults to [].
            pure_val_datasets (List[Union[Path, str, Dataset]], optional): List of datasets specifically for validation. Defaults to [].
            pure_test_datasets (List[Union[Path, str, Dataset]], optional): List of datasets specifically for testing. Defaults to [].
            tr_subsampling_factor (float, optional): Subsampling factor for the training dataset. Defaults to None.
            weights (Dict[str, float], optional): Dictionary mapping subdataset names to weights. Defaults to {}.
            balance_factor (float, optional): Balances sampling of the train datasets between 0 and 1. Defaults to 0.
            in_feat_names (List[str], optional): List of feature names to keep. Defaults to None.
            save_splits (Union[str, Path], optional): Path to save the split file as JSON. Defaults to None.
            val_conf_strategy (int, optional): Strategy for sampling conformations for the validation dataloader. Defaults to 200.
            split_ids (Dict[str, List[str]], optional): Dictionary containing the split IDs. Defaults to None.
            keep_features (bool, optional): Whether to keep features during processing. Defaults to False.
            """
        super().__init__()
        self.datasets = datasets
        self.conf_strategy = conf_strategy
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.ref_terms = ref_terms
        self.train_loader_workers = train_loader_workers
        self.val_loader_workers = val_loader_workers
        self.test_loader_workers = test_loader_workers
        self.seed = seed
        self.pin_memory = pin_memory
        self.splitpath = splitpath
        self.partition = partition
        self.pure_train_datasets = pure_train_datasets
        self.pure_val_datasets = pure_val_datasets
        self.pure_test_datasets = pure_test_datasets
        self.tr_subsampling_factor = tr_subsampling_factor
        self.weights = weights
        self.balance_factor = balance_factor
        self.in_feat_names = in_feat_names
        self.save_splits = save_splits
        self.val_conf_strategy = val_conf_strategy
        self.split_ids = split_ids
        self.keep_features = keep_features

        self.is_set_up = False

    def setup(self, stage: str = None):
        if self.is_set_up:
            return
        
        ds_paths = GrappaData._tags_to_paths([e for e in self.datasets if not isinstance(e, Dataset)])
        pure_train_paths = GrappaData._tags_to_paths([e for e in self.pure_train_datasets if not isinstance(e, Dataset)])
        pure_val_paths = GrappaData._tags_to_paths([e for e in self.pure_val_datasets if not isinstance(e, Dataset)])
        pure_test_paths = GrappaData._tags_to_paths([e for e in self.pure_test_datasets if not isinstance(e, Dataset)])

        GrappaData._check_ds_disjoint(ds_paths, pure_train_paths, pure_val_paths, pure_test_paths)

        # Load and add to the datasets
        ################################################
        dataset = [ds for ds in self.datasets if isinstance(ds, Dataset)]
        if len(ds_paths) > 0:
            # Log such that each dspath gets its own line:
            logging.info('Loading datasets from:\n'+"\n".join([str(p) for p in ds_paths]))
            dataset += [Dataset.load(p) for p in tqdm(ds_paths, desc='Loading datasets')]

        pure_train_sets = [ds for ds in self.pure_train_datasets if isinstance(ds, Dataset)]
        if len(pure_train_sets) > 0 or len(pure_train_paths) > 0:
            logging.info('Loading pure train datasets from:\n'+"\n".join([str(p) for p in pure_train_paths]))
            pure_train_sets += [Dataset.load(p) for p in tqdm(pure_train_paths, desc='Loading pure train datasets')]

        pure_val_sets = [ds for ds in self.pure_val_datasets if isinstance(ds, Dataset)]
        if len(pure_val_sets) > 0 or len(pure_val_paths) > 0:
            logging.info('Loading pure val datasets from:\n'+"\n".join([str(p) for p in pure_val_paths]))
            pure_val_sets += [Dataset.load(p) for p in tqdm(pure_val_paths, desc='Loading pure val datasets')]

        pure_test_sets = [ds for ds in self.pure_test_datasets if isinstance(ds, Dataset)]
        if len(pure_test_sets) > 0 or len(pure_test_paths) > 0:
            logging.info('Loading pure test datasets from:\n'+"\n".join([str(p) for p in pure_test_paths]))
            pure_test_sets += [Dataset.load(p) for p in tqdm(pure_test_paths, desc='Loading pure test datasets')]

        ################################################
        # transform the dataset to a single dataset:
        dataset = Dataset.concatenate(*dataset)
    

        # Get the split ids
        if self.split_ids is not None:
            assert isinstance(self.split_ids, dict), f"split_ids must be a dictionary, but got {type(self.split_ids)}"
            assert self.split_ids.keys() == {'train', 'val', 'test'}, f"split_ids must have keys 'train', 'val', and 'test', but got {self.split_ids.keys()}"

        # try to find canonical positions for the split file
        if self.splitpath is not None:
            if isinstance(self.splitpath, str):
                self.splitpath = Path(self.splitpath)
            if not self.splitpath.exists():
                self.splitpath = get_path_from_tag(tag=self.splitpath)/'split.json'
            assert self.splitpath.exists(), f"Split file {self.splitpath} does not exist."
            self.split_ids = json.load(open(self.splitpath, 'r'))
            logging.info(f'Using split ids from {self.splitpath}')

        # starting from the passed split ids or path, add more ids that are not included yet
        self.split_ids = dataset.calc_split_ids(partition=self.partition, seed=self.seed, existing_split=self.split_ids)

        # save the split ids:
        if self.save_splits is not None:
            if isinstance(self.save_splits, str):
                self.save_splits = Path(self.save_splits)
            assert isinstance(self.save_splits, Path), f"save_splits must be a Path or str, but got {type(self.save_splits)}"
            self.save_splits.parent.mkdir(parents=True, exist_ok=True)
            with open(self.save_splits, 'w') as f:
                json.dump(self.split_ids, f, indent=4)
            logging.info(f'Saved split ids to {self.save_splits}')

        self.tr, self.vl, self.te = dataset.split(*self.split_ids.values())
    
        self.tr = Dataset.concatenate(self.tr, *pure_train_sets)
        self.vl = Dataset.concatenate(self.vl, *pure_val_sets)
        self.te = Dataset.concatenate(self.te, *pure_test_sets)

        if self.tr_subsampling_factor is not None:
            if self.tr_subsampling_factor == 0.:
                logging.warning("Subsampling factor is 0, training set will be empty.")
            self.tr = self.tr.subsampled(self.tr_subsampling_factor, seed=self.seed)

        # Remove uncommon features for enabling batching
        self.tr = self._format_dataset(self.tr, ref_terms=self.ref_terms)
        self.vl = self._format_dataset(self.vl, ref_terms=self.ref_terms)
        self.te = self._format_dataset(self.te, ref_terms=self.ref_terms)

        logging.info("Loaded data:\n" + f"Train mols: {len(self.tr)}, Validation mols: {len(self.vl)}, Test mols: {len(self.te)}"+"\n")

        self.is_set_up = True

    def _format_dataset(self, ds:Dataset, ref_terms=['nonbonded']):
        '''
        Removes uncommon features and all n1 features except those needed as input for the model.
        '''
        if len(ds) == 0:
            return ds
        ds.create_reference(ref_terms=ref_terms)
        if not self.keep_features:
            ds.remove_uncommon_features()
            keep_feats = None
            if not self.in_feat_names is None:
                keep_feats = ['gradient_ref'] + self.in_feat_names
                ds.clean(keep_feats=keep_feats)
        return ds

    @staticmethod
    def _tags_to_paths(tags: List[Union[str, Path]]) -> List[Path]:
        paths = []
        for tag in tags:
            if isinstance(tag, str):
                try:
                    paths.append(get_path_from_tag(tag))
                except ValueError:
                    if not Path(tag).exists():
                        raise ValueError(f"Dataset path {tag} does not exist.")
                    paths.append(Path(tag))
            elif isinstance(tag, Path):
                paths.append(tag)
            else:
                raise ValueError(f"Dataset must be a Path or tag, but got {type(tag)}")
            
        if not len(paths) == len(set(paths)):
            raise ValueError(f"Duplicate paths in dataset list:\n{paths}\ntags:\n{tags}")
        return paths

    @staticmethod
    def _check_ds_disjoint(ds_paths, pure_train_paths, pure_val_paths, pure_test_paths):
        # print warning if these lists are not disjoint:
        for i in range(3):
            for j in range(i):
                ds_paths_1 = [ds_paths, pure_train_paths, pure_val_paths, pure_test_paths][i]
                ds_paths_2 = [ds_paths, pure_train_paths, pure_val_paths, pure_test_paths][j]
                if len(set(ds_paths_1).intersection(set(ds_paths_2))) > 0:
                    print(f"Warning: Dataset paths {ds_paths_1} and {ds_paths_2} are not disjoint: {set(ds_paths_1).intersection(set(ds_paths_2))}")


    def train_dataloader(self):
        return GraphDataLoader(self.tr, batch_size=self.train_batch_size, shuffle=True, num_workers=self.train_loader_workers, pin_memory=self.pin_memory, conf_strategy=self.conf_strategy, weights=self.weights, balance_factor=self.balance_factor, drop_last=True)

    def val_dataloader(self):
        return GraphDataLoader(self.vl, batch_size=self.val_batch_size, shuffle=False, num_workers=self.val_loader_workers, pin_memory=self.pin_memory, conf_strategy=self.val_conf_strategy, drop_last=False)

    def test_dataloader(self):
        return GraphDataLoader(self.te, batch_size=self.test_batch_size, shuffle=False, num_workers=self.test_loader_workers, pin_memory=self.pin_memory, conf_strategy='max', drop_last=False)
