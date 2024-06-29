from pathlib import Path
import argparse
import shutil
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
import os
from typing import List
from grappa.utils import get_data_path


MODELPATH = Path(__file__).parent.parent.parent.parent/'models'


def grappa_export():

    parser = argparse.ArgumentParser(description='Copies a checkpoint, a config.yaml file and all .json and .txt files in children of that dir to grappa/models/modelname/. Adds the modelname and path to grappa/src.tags.csv')
    parser.add_argument('--modelname', '-n', type=str, help='Name of the model, e.g. grappa-1.0\nIf None, the wandb id is used.', required=True)
    parser.add_argument('--checkpoint_path', '-cp', type=str, help='Absolute path to the lightning checkpoint that should be exported. Has to be specified if id is not given.', required=True)
    parser.add_argument('--description', '-m', type=str, help='Description of the model.', required=False, default='')

    args = parser.parse_args()

    targetdir = MODELPATH/f'{args.modelname}'
    if targetdir.exists():
        logging.warning(f"Model {args.modelname} already exists in {targetdir}. Removing it...")
        shutil.rmtree(targetdir)

    targetdir.mkdir(exist_ok=True, parents=True)

    # copy files:
    ckpt_path = Path(args.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist.")

    config_path = ckpt_path.parent/'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config path {config_path} does not exist.")

    ckpt_target = targetdir/'checkpoint.ckpt'
    logging.info(f"Copying {ckpt_path} to {ckpt_target}...")
    shutil.copy(ckpt_path, ckpt_target)
    shutil.copy(config_path, targetdir/'config.yaml')

    # copy all json and txt files in children of ckpt_path (and the containing dirs):
    for f in list(ckpt_path.parent.rglob('*.json')) + list(ckpt_path.parent.rglob('*.txt')):
        target = targetdir/f.relative_to(ckpt_path.parent)
        target.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(f, target)

    # read csv:
    csv_path = MODELPATH.parent/'models.csv'
    if csv_path.exists():
        csv = pd.read_csv(csv_path, comment='#')
    else:
        csv = pd.DataFrame(columns=['tag', 'path', 'description'])

    # remove modeltag if present:
    if args.modelname in csv.tag.values:
        logging.warning(f"Model {args.modelname} already exists in {csv_path}. Removing it...")
    csv = csv[csv.tag != args.modelname]

    # append to csv:
    relpath = targetdir.relative_to(MODELPATH.parent)
    csv = csv._append({'tag': args.modelname, 'path': relpath, 'description': args.description}, ignore_index=True)

    COMMENT="# Defines a map from model tag to local checkpoint path or url to zipped checkpoint and config file.\n# The checkpoint path is absolute or relative to the root directory of the project. A corresponding config.yaml is required to be present in the same directory."
    
    # Write the csv file with a comment at the top:
    with open(csv_path, 'w') as f:
        f.write(COMMENT + '\n')
        csv.to_csv(f, index=False)
    logging.info(f"Model {args.modelname} exported to {targetdir} and added to {csv_path}.")


def _release_model(release_tag:str, modelname:str):
    """
    Uploads a model to grappas github repository. GitHub CLI needs to be installed (https://github.com/cli/cli/blob/trunk/docs/install_linux.md). The release must exist already (e.g. by gh release create).
    The model dir is zipped and uploaded to the release. Assumes that the model is already exported, i.e. in the grappa/models/modelname/ directory.
    """

    modeldir = MODELPATH/f'{modelname}'
    if not modeldir.exists():
        raise FileNotFoundError(f"Expected model dir {modeldir} does not exist.")

    # zip model dir:
    zippath = modeldir.with_suffix('.zip')
    shutil.make_archive(zippath, 'zip', modeldir)

    # upload to release:
    logging.info(f"Uploading {zippath} to release {release_tag}...")
    os.system(f"gh release upload {release_tag} {zippath.absolute()}")
    os.remove(zippath)

def release_model():
    parser = argparse.ArgumentParser(description='Uploads a model to a given release of grappa using github CLI. The release must exist on the server. and github CLI must be installed.')
    parser.add_argument('--release_tag', '-t', type=str, required=True, help='The tag of the release that the model should be uploaded to.')
    parser.add_argument('--modelname', '-m', type=str, required=True, help='The name of the model that should be uploaded.')

    args = parser.parse_args()

    _release_model(args.release_tag, args.modelname)


def _upload_datasets(release_tag:str, dstags:List[str]):
    """
    Uploads datasets to a release of grappa. GitHub CLI needs to be installed (https://github.com/cli/cli/blob/trunk/docs/install_linux.md). The release must exist already. The dataset dirs at data/datastes/dstag are zipped and uploaded to the release.
    """

    for dstag in dstags:
        datasetdir = get_data_path()/dstag
        if not datasetdir.exists():
            raise FileNotFoundError(f"Expected dataset dir {datasetdir} does not exist.")

        # zip dataset dir:
        zippath = datasetdir.with_suffix('.zip')
        shutil.make_archive(zippath, 'zip', datasetdir)

        # upload to release:
        logging.info(f"Uploading {zippath} to release {release_tag}...")
        os.system(f"gh release upload {release_tag} {zippath.absolute()}")
        os.remove(zippath)

def upload_datasets():
    parser = argparse.ArgumentParser(description='Uploads datasets to a given release of grappa using github CLI. The release must exist on the server. and github CLI must be installed.')
    parser.add_argument('--release_tag', '-t', type=str, required=True, help='The tag of the release that the datasets should be uploaded to.')
    parser.add_argument('--dstags', '-d', type=str, nargs='+', required=True, help='The names of the datasets that should be uploaded.')

    args = parser.parse_args()

    _upload_datasets(args.release_tag, args.dstags)