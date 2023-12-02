import zipfile
import os
import requests
from tqdm import tqdm
from pathlib import Path
from typing import Union

def get_data_path()->Path:
    '''
    Returns the default path where to look for datasets.
    '''
    return Path(__file__).parents[3] / "data"


def get_path_from_tag(tag:str, data_dir:Union[Path,str]=get_data_path()/'dgl_datasets')->Path:
    '''
    Returns the path to a dataset given a tag. If the dataset is not at the corresponding location, it is downloaded. The tag is the filename of the dataset, available tags are:
    BENCHMARK ESPALOMA:
        - 'spice-dipeptide'
        - 'spice-des-monomers'
        - 'spice-pubchem'
        ...

    PEPTIDE DATASET:
        - 'tripeptide_amber99sbildn'
        - 'spice_dipeptide_amber99sbildn'
    '''

    URLS = {
        'tripeptides_amber99sbildn': 'https://github.com/LeifSeute/test_torchhub/releases/download/test_release/tripeptides_amber99sbildn.zip',
    }
    
    if not tag in URLS:
        raise ValueError(f"Tag {tag} not recognized. Available tags are {list(URLS.keys())}")
    
    return load_dataset(url=URLS[tag], data_dir=data_dir)



def load_dataset(url, data_dir=get_data_path()/'dgl_datasets'):
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

    # Create the directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)

    # Extract filename from URL
    filename = url.split('/')[-1].split('.')[0]
    dir_path = data_dir / filename


    # Download the file if it doesn't exist
    if not dir_path.exists():
        print(f"Downloading {filename} from:\n'{url}'")

        # this is the path to the zip file that is deleted after extraction
        zip_path = dir_path.with_suffix('.zip')

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

        # print(f"Downloaded {zip_path}")

        # Unzip the file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(str(data_dir))
            print(f"Stored dataset at:\n{dir_path}")
        
        # delete the zip file
        os.remove(zip_path)

    return dir_path