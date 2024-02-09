from grappa.utils.run_utils import load_yaml
from pathlib import Path
from grappa.models.deploy import model_from_config
import torch
from grappa.models.energy import Energy
import json
from grappa.utils.train_utils import remove_module_prefix
import os

def get_grappa_model(checkpoint_path):

    chkpt = torch.load(checkpoint_path)
    state_dict = chkpt['state_dict']

    config = Path(checkpoint_path).parent.parent.parent/'files'/'grappa_config.yaml'

    config = load_yaml(config)

    splitpath = Path(checkpoint_path).parent.parent.parent/'files'/'split.json'
    if not splitpath.exists():
        print(f"Warning: split_names.json not found in {splitpath}.")
        split_names = None
    else:
        with open(splitpath, 'r') as f:
            split_names = json.load(f)

    model = model_from_config(config['model_config'])

    full_model = torch.nn.Sequential(
        model,
        Energy(suffix=''),
        # Energy(suffix='_ref', write_suffix="_classical_ff")
    )

    state_dict = remove_module_prefix(state_dict)
    full_model.load_state_dict(state_dict)
    model = next(iter(full_model.children()))

    return model, config, split_names


def store_model_dict(checkpoint_path, modelname, modelpath=Path(__file__).parent.parent.parent.parent/'models', release_tag=None):
    """
    Stores a dictionary {state_dict: state_dict, config: config, split_names: split_names} at modelpath/modelname.pth
    The config entails a configuration of the training run that produced the checkpoint, the split names a list of identifiers for the train, validation and test molecules.
    """
    modelpath = Path(modelpath)
    model, config, split_names = get_grappa_model(checkpoint_path)
    model = model.eval()
    model = model.cpu()

    state_dict = model.state_dict()

    model_dict = {'state_dict': state_dict, 'config': config, 'split_names': split_names}

    if not modelpath.exists():
        modelpath.mkdir()

    if Path(modelpath/f'{modelname}.pth').exists():
        # ask the user for confirmation:
        print(f"Model {modelname} already exists at {modelpath/f'{modelname}.pth'}.")
        if input("Do you want to overwrite it? (y/n): ") != 'y':
            print("Aborting.")
            return
        else:
            print("Overwriting.")
            # remove the old model:
            os.remove(modelpath/f'{modelname}.pth')

    print(f"Saving model {modelname} to {modelpath/f'{modelname}.pth'}")

    torch.save(model_dict, modelpath/f'{modelname}.pth')


def release_model(release_tag:str, modelname:str, modelpath=Path(__file__).parent.parent.parent.parent/'models'):
    """
    Uploads a model to grappas github repository. GitHub CLI needs to be installed (https://github.com/cli/cli/blob/trunk/docs/install_linux.md). The release must exist already.
    """

    os.system(f"gh release upload {release_tag} {modelpath/f'{modelname}.pth'} --clobber")

