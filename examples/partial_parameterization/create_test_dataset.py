# %%
"""
Create the test dataset for the following dataset tags based on the espaloma split:
- spice-dipeptide-amber99-random{grappa_ratio}-grappa
- spice-dipeptide-amber99-connected{grappa_ratio}-grappa
- spice-dipeptide-amber99-all-grappa
- spice-dipeptide-amber99-sidechain-grappa

The test dataset is saved as dgl graphs under the tag f"{dataset_tag}-test". The original dataset is deleted after creating the test dataset for all dataset tags except for 'spice-dipeptide-amber99-sidechain-grappa', as the dataset 'spice-dipeptide-amber99-sidechain-grappa' is used to train grappa partial. 
"""

import json
from pathlib import Path
import shutil

from grappa.data import Dataset, clear_tag
from grappa.utils.data_utils import get_data_path, get_moldata_path

def create_test_dataset_from_split(dataset_tag: str, split_path: str="espaloma_split", delete_original_dataset: bool=False) -> None:
    """
    Create the test dataset for a dataset tag based on the split path. 

    The test dataset is saved as dgl graphs under the tag f"{dataset_tag}-test".

    Args:
        dataset_tag (str): The dataset tag.
        split_path (str): The path to the split.json file or the split tag.
        delete_original_dataset (bool): Whether to delete the original dataset. Default is False.
    """
    if isinstance(split_path, str):
        split_path = Path(split_path)
    else:
        raise TypeError(f"split_path should be a string, got {type(split_path)}.")
    if not split_path.exists():
        # Get the split path from the split tag
        split_path = get_moldata_path(tag=split_path)/'split.json'
        assert split_path.exists(), f"Split file {split_path} does not exist."

    dataset = Dataset.from_tag(dataset_tag)
    # Load the split ids from the split file
    split_ids = json.load(open(split_path, 'r'))
    # Save the dataset as dgl graphs
    *_, test_dataset = dataset.split(*split_ids.values())

    out_tag = f"{dataset_tag}-test"
    # Save the dataset as dgl graphs
    test_dataset.save(Path(get_data_path(), "dgl_datasets", out_tag))
    # Save the tag and dataset to the dataset_tags.csv file
    get_moldata_path(out_tag, data_dir=Path(get_data_path(), "dgl_datasets"))
    if delete_original_dataset:
        shutil.rmtree(Path(get_data_path(), "dgl_datasets", dataset_tag))
        clear_tag(dataset_tag)

# %%
for grappa_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    create_test_dataset_from_split(f"spice-dipeptide-amber99-random{grappa_ratio}-grappa", delete_original_dataset=True)
    create_test_dataset_from_split(f"spice-dipeptide-amber99-connected{grappa_ratio}-grappa", delete_original_dataset=True)

create_test_dataset_from_split("spice-dipeptide-amber99-all-grappa", delete_original_dataset=True)
create_test_dataset_from_split("spice-dipeptide-amber99-sidechain-grappa", delete_original_dataset=False)

