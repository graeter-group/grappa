from grappa.models import Energy, GrappaModel

import torch

def remove_module_prefix(state_dict):
    """ Remove the 'model.' prefix in the beginning of the keys from the state dict keys """
    new_state_dict = {}
    for k,v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_model(model_config, checkpoint_path=None):

    full_model = GrappaModel.model_from_config(**model_config)

    # add energy calculation
    full_model = torch.nn.Sequential(
        full_model,
        Energy(suffix='', gradients=True),
    )

    if not checkpoint_path is None:
        state_dict = torch.load(checkpoint_path)['state_dict']

        state_dict = remove_module_prefix(state_dict)
        full_model.load_state_dict(state_dict)

    model = next(iter(full_model.children()))

    return model