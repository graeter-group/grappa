from grappa.utils.run_utils import load_yaml
from pathlib import Path
from grappa.models.deploy import model_from_config
import torch
from grappa.models.energy import Energy
import json
from grappa.utils.train_utils import remove_module_prefix


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


def store_model_dict(checkpoint_path, modelname, modelpath=Path(__file__).parent.parent.parent.parent/'models'):
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
        raise FileExistsError(f"Model {modelname} already exists in {modelpath}")

    print(f"Saving model {modelname} to {modelpath/f'{modelname}.pth'}")

    torch.save(model_dict, modelpath/f'{modelname}.pth')
