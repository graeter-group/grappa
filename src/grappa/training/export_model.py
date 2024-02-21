from grappa.utils.run_utils import load_yaml
from pathlib import Path
from grappa.models.deploy import model_from_config
import torch
from grappa.models.energy import Energy
import json
from grappa.utils.train_utils import remove_module_prefix
from grappa.training.resume_trainrun import get_dir_from_id
import os
import argparse


def grappa_export():

    parser = argparse.ArgumentParser(description='Export a model such that it can be loaded easily. Stores the model, toghether with a config dict and the dataset partition that is was trained on in a .pth file.\nThen, the model can be loaded via model = grappa.utils.loading_utils.model_from_dict(torch.load(modelpath/modelname.pth)). If executed from within a wandb run directory, the model is expected to be in the files/checkpoints/best-model.ckpt file.')
    parser.add_argument('--modelname', '-n', type=str, help='Name of the model, e.g. grappa-1.0\nIf None, the wandb id is used.', default=None)
    parser.add_argument('--checkpoint_path', '-cp', type=str, help='Absolute path to the lightning checkpoint that should be exported. Has to be specified if id is not given.', default=None)
    parser.add_argument('--id', '-i', type=str, help='The wandb id of the run that should be exported. Searches for the oldest best-model.ckpt file that belongs to that run. If you use this argument, the function has to be executed from the dir that contains the wandb folder. Has to be specified if checkpoint_path is not given.', default=None)
    parser.add_argument('--release_tag', type=str, default=None, help='If not None, uploads the model to a given release of grappa using github CLI. The release must exist on the server. and github CLI must be installed.') #

    MODELPATH = Path(__file__).parent.parent.parent.parent/'models'

    args = parser.parse_args()

    if args.checkpoint_path is not None and args.id is not None:
        raise ValueError("Either id or checkpoint_path has to be specified, not both.")

    if args.checkpoint_path is None:
        if args.id is None:
            checkpoint_path = Path.cwd() / 'files' / 'checkpoints' / 'best-model.ckpt'
        else:
            checkpoint_path = Path(get_dir_from_id(run_id=args.id, wandb_folder=Path.cwd()/'wandb')) / 'files/checkpoints/best-model.ckpt'
    else:
        checkpoint_path = Path(args.checkpoint_path)

    if args.modelname is None:
        modelname = checkpoint_path.parent.parent.parent.name.split('-')[-1]
    else:
        modelname = args.modelname

    model_dict = get_model_dict(checkpoint_path)

    store_model_dict(model_dict, modelname=modelname, release_tag=args.release_tag, modelpath=MODELPATH)




def get_grappa_model(checkpoint_path):

    chkpt = torch.load(checkpoint_path)
    state_dict = chkpt['state_dict']

    config = Path(checkpoint_path).parent.parent.parent/'files'/'grappa_config.yaml'

    config = load_yaml(config)

    splitpath = Path(checkpoint_path).parent.parent.parent/'files'/'split.json'
    if not splitpath.exists():
        if (Path(checkpoint_path).parent.parent.parent/'files/files'/'split.json').exists():
            splitpath = Path(checkpoint_path).parent.parent.parent/'files/files'/'split.json'
            with open(splitpath, 'r') as f:
                split_names = json.load(f)
        else:
            print(f"Warning: split_names.json not found in {splitpath}")
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


def get_model_dict(checkpoint_path):
    """
    Returns a dictionary {state_dict: state_dict, config: config, split_names: split_names} at modelpath/modelname.pth
    The config entails a configuration of the training run that produced the checkpoint, the split names a list of identifiers for the train, validation and test molecules.
    """
    model, config, split_names = get_grappa_model(checkpoint_path)
    model = model.eval()
    model = model.cpu()

    state_dict = model.state_dict()

    model_dict = {'state_dict': state_dict, 'config': config, 'split_names': split_names}

    return model_dict


def store_model_dict(model_dict, modelname, modelpath=Path(__file__).parent.parent.parent.parent/'models', release_tag=None):
    """
    Stores a dictionary {state_dict: state_dict, config: config, split_names: split_names} at modelpath/modelname.pth
    The config entails a configuration of the training run that produced the checkpoint, the split names a list of identifiers for the train, validation and test molecules.
    """
    modelpath = Path(modelpath)

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

    if not release_tag is None:
        release_model(release_tag, modelname, modelpath=modelpath)


def release_model(release_tag:str, modelname:str, modelpath=Path(__file__).parent.parent.parent.parent/'models'):
    """
    Uploads a model to grappas github repository. GitHub CLI needs to be installed (https://github.com/cli/cli/blob/trunk/docs/install_linux.md). The release must exist already.
    """

    modelfile = modelpath/f'{modelname}.pth'
    if not modelfile.exists():
        modelfile = modelpath/f'{modelname}'
        if not modelfile.exists():
            raise FileNotFoundError(f"Model {modelname} not found at {modelfile} or {modelfile}.pth")

    os.system(f"gh release upload {release_tag} {modelpath/f'{modelname}.pth'} --clobber")

def grappa_release():

    parser = argparse.ArgumentParser(description='Uploads a model to a given release of grappa using github CLI. The release must exist on the server. and github CLI must be installed.')
    parser.add_argument('--release_tag', '-t', type=str, required=True, help='The tag of the release that the model should be uploaded to.')
    parser.add_argument('--modelname', '-m', type=str, required=True, help='The name of the model that should be uploaded.')

    args = parser.parse_args()

    release_model(args.release_tag, args.modelname)