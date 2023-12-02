#%%
from grappa.data import Dataset, GraphDataLoader
from grappa.utils.run_utils import get_data_path, load_yaml
import torch
from grappa.models.deploy import model_from_config
from grappa.models.Energy import Energy
from pathlib import Path

def package_model(checkpoint_path, config, modelname):

    modelpath = Path('/hits/fast/mbm/seutelf/grappa/models')
    
    chkpt = torch.load(checkpoint_path)
    state_dict = chkpt['state_dict']

    config = load_yaml(config/'files'/'grappa_config.yaml')

    def remove_module_prefix(state_dict):
        """ Remove the 'model.' prefix in the beginning of the keys from the state dict keys """
        new_state_dict = {}
        for k,v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    class ParamFixer(torch.nn.Module):
        def forward(self, g):
            g.nodes['n2'].data['k'] = g.nodes['n2'].data['k'][:,0]
            g.nodes['n2'].data['eq'] = g.nodes['n2'].data['eq'][:,0]
            g.nodes['n3'].data['k'] = g.nodes['n3'].data['k'][:,0]
            g.nodes['n3'].data['eq'] = g.nodes['n3'].data['eq'][:,0]
            return g

    model = model_from_config(config=config['model_config'])

    full_model = torch.nn.Sequential(
        model,
        ParamFixer(),
        Energy(suffix=''),
        Energy(suffix='_ref', write_suffix="_classical_ff")
    )

    state_dict = remove_module_prefix(state_dict)
    full_model.load_state_dict(state_dict) # this also loads the state_dict into model.
    # check:
    # model2 = next(iter(full_model.children()))
    # assert all([torch.all(v==model2.state_dict()[k]) for k,v in model.state_dict().items() ])
    model = model.eval()
    model = model.cpu()

    state_dict = model.state_dict()

    model_dict = {'state_dict': state_dict, 'config': config}


    torch.save(model_dict, modelpath/f'{modelname}.pth')

#%%
checkpoint_path = '/hits/fast/mbm/seutelf/grappa/tests/experiments/checkpoints/autumn-energy-16/model-epoch=499-avg/val/rmse_gradients=6.40.ckpt'

config = Path('/hits/fast/mbm/seutelf/grappa/tests/experiments/wandb/run-20231129_173644-tfbiylf2')

modelname = 'protein_test_11302023'

package_model(checkpoint_path, config, modelname)
# %%
checkpoint_path = '/hits/fast/mbm/seutelf/grappa/tests/checkpoints/whncielx/model-epoch=189-avg/val/rmse_gradients=9.04.ckpt'

config = Path('/hits/fast/mbm/seutelf/grappa/tests/wandb/run-20231129_184216-whncielx')

modelname = 'am1bc_test_11302023'

package_model(checkpoint_path, config, modelname)
# %%