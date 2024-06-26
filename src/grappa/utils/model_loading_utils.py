import torch
from pathlib import Path
from typing import Union, Dict
from grappa.models import GrappaModel
import yaml
import csv
import pandas as pd
from grappa.utils.data_utils import get_repo_dir, download_zipped_dir
from grappa.training.utils import get_model_from_path
import logging

def model_from_path(path:Union[str,Path])->GrappaModel:
    """
    Loads a model from a path to a checkpoint. The parent dir of the checkpoint must contain a config.yaml file that includes the model config.
    """
    return get_model_from_path(path, device='cpu')

def model_from_tag(tag:str='latest')->GrappaModel:
    """
    Loads a model from a tag. With each release, the mapping tag to url of model weights is updated such that models returned by this function are always at a version that works in the respective release.
    Possible tags are defined in src/tags.csv.
    """
    csv_path = Path(__file__).parent.parent.parent / 'tags.csv'

    df = pd.read_csv(csv_path, comment='#')
    tags = df['tag'].values
    if tag not in tags:
        
        # if the tag is ...-n.x, try to find n.x.m with m maximal, n,m,x: int
        new_tag = find_tag(tag, tags)
        if new_tag is not None:
            tag = new_tag
            assert tag in tags, f"Internal error: tag {tag} not found in tags.csv"
        else:
            raise ValueError(f"Tag {tag} not found in tags.csv")
    
    # tag must be unique:
    idxs = tags==tag
    if sum(idxs)!=1:
        raise ValueError(f"Tag {tag} is not unique in tags.csv: {sum(idxs)} entries found.")
    idx = idxs.argmax()

    path = df['path'].iloc[idx]

    if not Path(path).is_absolute():
        path = get_repo_dir()/Path(path)
    if not Path(path).exists():
        logging.info(f"Path {path} does not exist. Downloading the model weights.")
        url = df['url'].iloc[idx]
        download_zipped_dir(url=url, target_dir=path.parent)

    return model_from_path(path=path)


def find_tag(tag, tags):
    """
    Finds the tag in tags that corresponds to the same model as the input tag.
    if the tag is ...-n.x, try to find n.x.m with m maximal, n,m,x: int
    """
    tag_suffix = tag.split('-')[-1]
    if len(tag_suffix.split('.'))==2:
        if tag_suffix[-1].isdigit() and tag_suffix[-2].isdigit():
            n = int(tag_suffix.split('.')[0])
            x = int(tag_suffix.split('.')[1])
            # find pattern ...-n.x.m in tags:
            cond = lambda t: tag in t and len(t.split('-')[-1].split('.'))==3 and t.split('-')[-1].split('.')[0]==str(n) and t.split('-')[-1].split('.')[1]==str(x) and t.split('-')[-1].split('.')[2].isdigit()
            idxs = [i for i, t in enumerate(tags) if cond(t)]
            if len(idxs)>0:
                # find the max m:
                max_m = -2
                for i in idxs:
                    m = int(tags[i].split('-')[-1].split('.')[-1])
                    if m>max_m:
                        max_m = m
                        idx = i
                return tags[idx]
    return None