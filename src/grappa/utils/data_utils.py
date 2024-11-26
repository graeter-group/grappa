import zipfile
import os
import requests
from tqdm import tqdm
from pathlib import Path
from typing import Union
from tqdm import tqdm
import logging
import pandas as pd
import numpy as np
import shutil
import pkg_resources
import grappa

def get_data_csv_path()->Path:
    '''
    Returns the path of the csv file defining tag -> dataset mapping.
    '''
    return get_repo_dir() / "data" / "dataset_tags.csv"

def get_published_data_csv_path()->Path:
    '''
    Returns the path of the csv file defining tag -> dataset mapping.
    '''
    return get_src_dir() / "data" / "published_datasets.csv"

def is_installed_via_pypi()->bool:
    '''
    Returns True if the package is installed via pip, False otherwise.
    Assumes that the package is installed via pip if the package is in the working set and the path to the package contains 'site-packages' or 'dist-packages'. NOTE: Make this better.
    '''
    if not 'grappa-ff' in pkg_resources.working_set.by_key:
        return False
    package_path = os.path.abspath(grappa.__file__)
    if 'site-packages' in package_path or 'dist-packages' in package_path:
        return True
    return False

def get_src_dir() -> Path:
    '''
    Returns the path to the src/grappa directory of the repository. (not to src but to src/grappa !)
    '''
    if is_installed_via_pypi():
        return Path(__file__).resolve().parents[1]
    else:
        # If the package is not installed (development mode)
        return Path(__file__).resolve().parents[1]

def get_repo_dir() -> Path:
    '''
    Returns the path to the root of the repository.
    '''
    if is_installed_via_pypi():
        GRAPPA_DIR = Path().home() / '.grappa_cache'
        GRAPPA_DIR.mkdir(exist_ok=True)
        (GRAPPA_DIR/'models').mkdir(exist_ok=True)
        (GRAPPA_DIR/'data').mkdir(exist_ok=True)
        return GRAPPA_DIR
    else:
        # If the package is not installed (development mode)
        return Path(__file__).resolve().parents[3]

def get_data_path()->Path:
    '''
    Returns the default path where to look for datasets (grappa/data)
    '''
    return get_repo_dir() / "data"

def get_moldata_path(tag:str, data_dir:Union[Path,str]=get_data_path()/'datasets')->Path:
    '''
    Returns the path to a dataset given a tag defined in the csv file at get_data_csv_path(). If the tag is not in the csv file, looks at get_data_path() for a dir with that name and appends the tag and its location to the csv file. If the dataset is not at the corresponding location but a url is specified in the csv file, it is downloaded. Available tags for download are:

    BENCHMARK ESPALOMA:
        - 'spice-des-monomers'
        - 'spice-pubchem'
        - 'gen2'
        - 'gen2-torsion'
        - 'spice-dipeptide'
        - 'protein-torsion'
        - 'pepconf-dlc'
        - 'rna-diverse'
        - 'rna-trinucleotide'

    PEPTIDE DATASET:
        - dipeptides-300K-openff-1.2.0
        - dipeptides-300K-amber99
        - dipeptides-300K-charmm36_nonb
        - dipeptides-1000K-openff-1.2.0
        - dipeptides-1000K-amber99
        - dipeptides-1000K-charmm36_nonb
        - uncapped-300K-openff-1.2.0
        - uncapped-300K-amber99
        - dipeptides-hyp-dop-300K-amber99

    RADICAL DATASET:
        - dipeptides-radical-300K
        - bondbreak-radical-peptides-300K

    SPLITFILE:
        'espaloma_split'
    '''

    tag = str(tag)

    # Load the csv file
    csv_path = get_data_csv_path()
    url_csv_path = get_published_data_csv_path()
    if not csv_path.exists():
        # create empty csv file:
        df = pd.DataFrame(columns=['tag', 'path', 'description'])
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    
    df = pd.read_csv(csv_path, dtype=str)
    url_df = pd.read_csv(url_csv_path, dtype=str)
    url = url_df[url_df['tag']==tag]['url'].values[0] if tag in url_df['tag'].values else None
    description = url_df[url_df['tag']==tag]['description'].values[0] if tag in url_df['tag'].values else None

    tags = df['tag'].values
    tag_present = tag in tags

    if tag_present:
        row = df[df['tag']==tag]
        if len(row) > 1:
            raise ValueError(f"Multiple entries for tag {tag} in the dataset_tags.csv file.")
        path = str(row['path'].values[0])
        # if path is specified:
        if not path in ['nan', '']:
            if Path(path).exists():
                found_path = Path(path)
            elif url is not None:
                logging.warning(f"Dataset {tag} not found at {path} as specified in the dataset_tags.csv file. Deleting the entry from the csv file and downloading again...")
                df = df[df['tag']!=tag]
                found_path = Path(data_dir)/tag
                df = pd.concat([df, pd.DataFrame([{'tag': tag, 'path': str(found_path), 'description': ''}])], ignore_index=True)
                Path(str(csv_path)).parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(csv_path, index=False)
            else:
                raise FileNotFoundError(f"Dataset {tag} not found at {path}.")
        # try to find data_dir/tag:
        elif (Path(data_dir)/tag).exists():
            logging.info(f"Dataset {tag} not in the dataset_tags.csv file but found at {Path(data_dir)/tag}. Appending to the csv file...")
            found_path = Path(data_dir)/tag

            df.loc[df['tag']==tag, 'path'] = str(found_path)
            Path(str(csv_path)).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
        else:
            raise FileNotFoundError(f"Dataset {tag} not found at {path}.")

    else:
        # try to find data_dir/tag:
        if (Path(data_dir)/tag).exists():
            logging.info(f"Dataset {tag} not in the dataset_tags.csv file but found at {Path(data_dir)/tag}. Appending to the csv file...")
            found_path = Path(data_dir)/tag
            df = pd.concat([df, pd.DataFrame([{'tag': tag, 'path': str(found_path), 'description': ''}])], ignore_index=True)
            Path(str(csv_path)).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
        # download the dataset from the url if specified:
        elif url is not None:
            logging.info(f"Dataset {tag} not found in dataset_tags.csv. Downloading from {url} to {data_dir/tag}...")
            found_path = load_dataset(url=url, data_dir=data_dir, dirname=tag)
            df = pd.concat([df, pd.DataFrame([{'tag': tag, 'path': str(found_path), 'description': description}])], ignore_index=True)
            # for backwards compatibility, remove the old dgl dataset if present so that it will be overwritten by the new one upon call of Dataset.from_tag:
            dgl_dir = get_data_path()/'dgl_datasets'/tag
            if dgl_dir.exists():
                logging.warning(f"Removing old dgl dataset at {dgl_dir}, which will be overwritten by the downloaded dataset.")
                shutil.rmtree(dgl_dir)
            Path(str(csv_path)).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
        else:
            raise FileNotFoundError(f"Dataset {tag} not found in the dataset_tags.csv file and not at {data_dir/tag}.")
    
    # if the path is not absolute, concat it with the root dir of grappa:
    if not found_path.is_absolute():
        found_path = get_repo_dir() / found_path

    return found_path


def load_dataset(url:str, data_dir:Path=get_data_path()/'datasets', dirname:str=None)->Path:
    """
    Downloads a zip dataset from a given URL if it's not already present in the local directory, 
    then extracts it.

    Parameters:
        url (str): The URL of the dataset to download.
        data_dir (str): The local directory to store and extract the dataset. Default is 'grappa/data/dgl_datasets'.

    Returns:
        str: Path to the directory where the dataset is extracted.
    """

    data_dir = Path(data_dir).absolute()

    # Extract dirname from URL
    if dirname is None:
        dirname = url.split('/')[-1].split('.')[0]
    dir_path = data_dir / dirname


    # Download the folder if it doesn't exist
    if not dir_path.exists():
        logging.info(f"Downloading {dirname} from:\n'{url}'")

        download_zipped_dir(url=url, target_dir=dir_path)

    return dir_path



def download_zipped_dir(url:str, target_dir:Path):
    # this is the path to the zip file that is deleted after extraction
    target_dir = Path(target_dir).absolute()

    target_dir.mkdir(parents=True, exist_ok=True)

    zip_path = target_dir.with_suffix('.zip')

    # Start the download
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure the request was successful

    # Get the total file size from headers
    total_size = int(response.headers.get('content-length', 0))

    # Initialize the progress bar
    with tqdm(total=total_size, unit='B', unit_scale=True) as t:
        with open(zip_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
                t.update(len(chunk))

    assert zip_path.exists(), f"Download failed, file not found at {zip_path}"

    # Unzip the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(str(target_dir))
    
    # delete the zip file
    try:
        os.remove(zip_path)
    except OSError as e:
        print(f"Error while deleting the zip file: {e}")

    logging.info(f"Downloaded to {target_dir}")
