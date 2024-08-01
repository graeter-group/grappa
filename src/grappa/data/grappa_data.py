import pytorch_lightning as pl
from pathlib import Path
from typing import List, Dict, Tuple, Union
import torch
import json
import copy
import logging
from grappa.data.dataset import Dataset
from grappa.data.graph_data_loader import GraphDataLoader
from grappa.utils.data_utils import get_moldata_path
from tqdm import tqdm

class GrappaData(pl.LightningDataModule):
    def __init__(self,
                 datasets: List[str],
                 pure_train_datasets: List[str] = [],
                 pure_val_datasets: List[str] = [],
                 pure_test_datasets: List[str] = [],
                 splitpath: Path = None,
                 partition: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 train_batch_size: int = 32,
                 val_batch_size: int = 32,
                 test_batch_size: int = 1,
                 ref_terms: List[str] = ['nonbonded'],
                 train_loader_workers: int = 4,
                 val_loader_workers: int = 4,
                 test_loader_workers: int = 4,
                 conf_strategy: Union[str, int] = 32,
                 ff_lookup: Dict[str, str] = {},
                 seed: int = 0,
                 pin_memory: bool = True,
                 tr_subsampling_factor: float = None,
                 tr_max_confs:int = None,
                 weights: Dict[str, float] = {},
                 balance_factor: float = 0.,
                 in_feat_names: List[str] = None,
                 save_splits: Union[str, Path] = None,
                 val_conf_strategy: int = 200,
                 split_ids: Dict[str, List[str]] = None,
                 keep_features: bool = False,
                ):
        """
        This class handles the preparation of train, validation, and test dataloaders for a given list of datasets.

        Args:
            datasets (List[Union[Path, str, Dataset]]): List of dataset tags (defined in the dataset_tags.csv file or located at datasets/tag)
            splitpath (Path, optional): Path to the split file. If provided, the function will load the split from this file. If not, it will generate a new split. Defaults to None.
            partition (Tuple[float, float, float], optional): Partition of the dataset into train, validation, and test splits. Defaults to (0.8, 0.1, 0.1).
            conf_strategy (Union[str, int], optional): Strategy for sampling conformations when batching. Defaults to 32.
            ff_lookup (Dict[str,str]): dictionary mapping dataset names to force field names. If a dataset name is not present in the lookup, the force field is determined automatically by selecting the force field that provides all necessary terms, or, if this is not unique, by selecting the force field 'reference_ff' if present.
            train_batch_size (int, optional): Batch size for the training dataloader. Defaults to 1.
            val_batch_size (int, optional): Batch size for the validation dataloader. Defaults to 1.
            test_batch_size (int, optional): Batch size for the test dataloader. Defaults to 1.
            ref_terms (List[str], optional): Terms for which to use the reference contributions. These will be subtracted from the QM energies and gradients, the result is stored in energy_ref / gradient_ref. Defaults to ['nonbonded'].
            train_loader_workers (int, optional): Number of worker processes for the training dataloader. Defaults to 1.
            val_loader_workers (int, optional): Number of worker processes for the validation dataloader. Defaults to 1.
            test_loader_workers (int, optional): Number of worker processes for the test dataloader. Defaults to 1.
            seed (int, optional): Random seed for reproducibility. Defaults to 0.
            pin_memory (bool, optional): Whether to pin memory for the dataloaders. Defaults to True.
            pure_train_datasets (List[str], optional): List of dataset tags to be used as pure training datasets without using the mol_is for splitting.
            pure_val_datasets (List[str], optional): List of dataset tags to be used as pure validation datasets without using the mol_is for splitting.
            pure_test_datasets (List[str], optional): List of dataset tags to be used as pure test datasets without using the mol_is for splitting.
            tr_subsampling_factor (float, optional): Subsampling factor for the training dataset. Defaults to None.
            tr_max_confs (int, optional): Maximum number of conformations to keep in the training set. Defaults to None.
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
        self.tr_max_confs = tr_max_confs
        self.weights = weights
        self.balance_factor = balance_factor
        self.in_feat_names = in_feat_names
        self.save_splits = save_splits
        self.val_conf_strategy = val_conf_strategy
        self.split_ids = split_ids
        self.keep_features = keep_features
        self.ff_lookup = ff_lookup

        self.train_cleanup = True # set this to manually to False if you want to keep the reference ff data in the training set (e.g. for evaluating the reference data on the training set)
        self.num_test_mols = None # number of molecules in the test set, we might need to keep track of this for the evaluation

        self.is_set_up = False

    def setup(self, stage: str = None):
        if self.is_set_up:
            return

        GrappaData._check_ds_disjoint(self.datasets, self.pure_train_datasets, self.pure_val_datasets, self.pure_test_datasets)

        # Load and add to the datasets
        ################################################
        dataset = []
        if len(self.datasets) > 0:
            # Log such that each dspath gets its own line:
            logging.info('Loading datasets from:\n'+"\n".join([str(p) for p in self.datasets]))
            dataset += [Dataset.from_tag(p) for p in tqdm(self.datasets, desc='Loading datasets')]

        pure_train_sets = [ds for ds in self.pure_train_datasets if isinstance(ds, Dataset)]
        if len(pure_train_sets) > 0 or len(self.pure_train_datasets) > 0:
            logging.info('Loading pure train datasets from:\n'+"\n".join([str(p) for p in self.pure_train_datasets]))
            pure_train_sets += [Dataset.from_tag(p) for p in tqdm(self.pure_train_datasets, desc='Loading pure train datasets')]

        pure_val_sets = [ds for ds in self.pure_val_datasets if isinstance(ds, Dataset)]
        if len(pure_val_sets) > 0 or len(self.pure_val_datasets) > 0:
            logging.info('Loading pure val datasets from:\n'+"\n".join([str(p) for p in self.pure_val_datasets]))
            pure_val_sets += [Dataset.from_tag(p) for p in tqdm(self.pure_val_datasets, desc='Loading pure val datasets')]

        pure_test_sets = [ds for ds in self.pure_test_datasets if isinstance(ds, Dataset)]
        if len(pure_test_sets) > 0 or len(self.pure_test_datasets) > 0:
            logging.info('Loading pure test datasets from:\n'+"\n".join([str(p) for p in self.pure_test_datasets]))
            pure_test_sets += [Dataset.from_tag(p) for p in tqdm(self.pure_test_datasets, desc='Loading pure test datasets')]

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
                # assume its a tag
                self.splitpath = get_moldata_path(tag=self.splitpath)/'split.json'
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

        # since all confs are returned in the test dataloader, batches could become too large for the GPU memory. Therefore, we restrict the number of conformations to val_conf_strategy and split individual molecules into multiple entries if necessary
        self.te.apply_max_confs(confs=self.val_conf_strategy)
        self.num_test_mols = self.te.num_mols

        if self.tr_subsampling_factor is not None:
            if self.tr_subsampling_factor == 0.:
                logging.warning("Subsampling factor is 0, training set will be empty.")
            self.tr = self.tr.subsampled(self.tr_subsampling_factor, seed=self.seed)

        if self.tr_max_confs is not None:
            if self.tr_max_confs == 0:
                logging.warning("Maximum number of conformations is 0, training set will be empty.")
            self.tr = self.tr.remove_confs(int(self.tr_max_confs), seed=self.seed)

        # write reference data as energy_ref = energy_qm - sum(energy_ref_terms) / gradient_ref = ...
        self.tr.create_reference(ref_terms=self.ref_terms, ff_lookup=copy.deepcopy(self.ff_lookup), cleanup=self.train_cleanup)
        self.vl.create_reference(ref_terms=self.ref_terms, ff_lookup=copy.deepcopy(self.ff_lookup), cleanup=self.train_cleanup)
        self.te.create_reference(ref_terms=self.ref_terms, ff_lookup=copy.deepcopy(self.ff_lookup), cleanup=False)

        # Remove uncommon features for enabling batching
        self.tr.remove_uncommon_features()
        self.vl.remove_uncommon_features()

        logging.info("Loaded data:\n" + f"Train mols: {len(self.tr)}, Validation mols: {len(self.vl)}, Test mols: {len(self.te)}"+"\n")

        self.is_set_up = True

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
        return GraphDataLoader(self.tr, batch_size=self.train_batch_size, shuffle=True, num_workers=self.train_loader_workers, pin_memory=self.pin_memory, conf_strategy=self.conf_strategy, weights=self.weights, balance_factor=self.balance_factor, drop_last=len(self.tr) > self.train_batch_size)

    def val_dataloader(self):
        return GraphDataLoader(self.vl, batch_size=self.val_batch_size, shuffle=False, num_workers=self.val_loader_workers, pin_memory=self.pin_memory, conf_strategy=self.val_conf_strategy, drop_last=False)

    def test_dataloader(self):
        # since the conf_strategy is 'max', the test loader will always load all conformations
        # thus, the batchsize should be 1, otherwise, could throw an error if two molecules form the same batch have different number of conformations
        return GraphDataLoader(self.te, batch_size=self.test_batch_size, shuffle=False, num_workers=self.test_loader_workers, pin_memory=self.pin_memory, conf_strategy='max', drop_last=False)
