import torch
from pathlib import Path
from typing import Union, Dict
from grappa.models import GrappaModel
import yaml
import csv
import pandas as pd
from grappa.utils.data_utils import get_repo_dir, download_zipped_dir
from grappa.utils.training_utils import get_model_from_path
import logging

def get_model_dir():
    return get_repo_dir() / 'models'

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
    csv_path = get_repo_dir() / 'models' / 'models.csv'
    published_csv_path = get_repo_dir() / 'models' / 'published_models.csv'
    COMMENT="# Defines a map from model tag to local checkpoint path or url to zipped checkpoint and config file.\n# The checkpoint path is absolute or relative to the root directory of the project. A corresponding config.yaml is required to be present in the same directory."

    if not csv_path.exists():
        empty_df = pd.DataFrame(columns=['tag', 'path', 'description'])
        store_with_comment(empty_df, csv_path, COMMENT)

    df = pd.read_csv(csv_path, comment='#', dtype=str)
    tags = df['tag'].values
    url_df = pd.read_csv(published_csv_path, dtype=str)
    url_tags = url_df['tag'].values

    all_tags = list(tags) + list(url_tags)

    # try to find a later version of the model if the tag is not found:
    if tag not in all_tags:
        # if the tag is ...-n.x, try to find n.x.m with m maximal, n,m,x: int
        new_tag = find_tag(tag, all_tags)
        if new_tag is not None:
            logging.info(f"Tag {tag} not found. Using {new_tag} instead.")
            tag = new_tag
            assert tag in all_tags, f"Internal error: tag {tag} not found"
        else:
            raise ValueError(f"Tag {tag} not found, available tags: {all_tags}")

    if tag in tags:
        # tag must be unique:
        idxs = tags==tag
        if sum(idxs)!=1:
            raise ValueError(f"Tag {tag} is not unique in tags.csv: {sum(idxs)} entries found.")
        idx = idxs.argmax()

        path = df['path'].iloc[idx]

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model {tag} not found at {path}.")

        if not path.is_absolute():
            path = get_repo_dir()/path

        if not str(path).endswith('.ckpt'):
            path = get_ckpt(path)

    else:
        url = url_df[url_df['tag']==tag]['url'].values[0]
        description = url_df[url_df['tag']==tag]['description'].values[0]
        path = get_model_dir() / tag
        logging.info(f"Downloading model {tag} from {url}")
        download_zipped_dir(url=url, target_dir=path)
        # find a .ckpt file in the directory:
        get_ckpt(path)
        df = pd.concat([df, pd.DataFrame([{'tag': tag, 'path': str(path), 'description': description}])], ignore_index=True)
        store_with_comment(df, csv_path, COMMENT)

    return model_from_path(path=path)


def find_tag(tag, tags):
    """
    Finds the tag in tags that corresponds to the same model as the input tag.
    if the tag is ...-n.x, try to find n.x.m with m maximal, n,x,m: int
    """
    tag_suffix = tag.split('-')[-1]
    if len(tag_suffix.split('.'))==2:
        if tag_suffix.split('.')[-1].isdigit() and tag_suffix.split('.')[-2].isdigit():
            n = int(tag_suffix.split('.')[0])
            x = int(tag_suffix.split('.')[1])
            # find pattern ...-n.x.m in tags:
            cond = lambda t: tag in t and len(t.split('-')[-1].split('.'))==3 and t.split('-')[-1].split('.')[0]==str(n) and t.split('-')[-1].split('.')[1]==str(x) and t.split('-')[-1].split('.')[2].isdigit()
            idxs = [i for i, t in enumerate(tags) if cond(t)]
            if len(idxs)>0:
                # find the max m:
                max_m = -1
                for i in idxs:
                    m = int(tags[i].split('-')[-1].split('.')[-1])
                    if m>max_m:
                        max_m = m
                        idx = i
                return tags[idx]
    return None

def store_with_comment(df, path, comment):
    with open(path, 'w') as f:
        f.write(comment + '\n')
        df.to_csv(f, index=False)

def get_ckpt(path):
    ckpt_files = list(path.rglob('*.ckpt'))
    if len(ckpt_files)==0:
        raise FileNotFoundError(f"No .ckpt file found in {path}")
    elif len(ckpt_files)>1:
        raise FileNotFoundError(f"Multiple .ckpt files found in {path}: {ckpt_files}")
    path = ckpt_files[0]
    return path