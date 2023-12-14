#%%
from grappa.models.grappa import GrappaModel
from grappa.training.get_dataloaders import get_dataloaders
from grappa.training.evaluation import Evaluator, ExplicitEvaluator
from grappa.utils.dataset_utils import get_data_path
from grappa.utils.graph_utils import get_param_statistics
from grappa.models.energy import Energy
from grappa.utils.loading_utils import load_model
import torch
#%%

tr, vl,_ = get_dataloaders([str(get_data_path()/'dgl_datasets'/'Capped_AA_break_breakpoint')])
param_statistics = get_param_statistics(tr)
model = GrappaModel(param_statistics=param_statistics)
model = torch.nn.Sequential(model, Energy())
# %%

evaluator = Evaluator()
for g, dsnames in vl:
    g = model(g)
    evaluator.step(g, dsnames)

# %%
evaluator.pool()

# %%
checkpoint_path = '/hits/fast/mbm/seutelf/grappa/tests/wandb/run-20231214_015501-zih17qw0/files/checkpoints/last.ckpt'
# checkpoint_path = '/hits/fast/mbm/seutelf/grappa/tests/wandb/run-20231214_105318-71sc76cy/files/checkpoints/last.ckpt'
checkpoint_path='/hits/fast/mbm/seutelf/grappa/tests/wandb/run-20231214_030848-gd2oop8p/files/checkpoints/best-model.ckpt'

from grappa.utils.run_utils import load_yaml
from pathlib import Path
from grappa.models.deploy import model_from_config

def get_grappa_model(checkpoint_path):

    chkpt = torch.load(checkpoint_path)
    state_dict = chkpt['state_dict']

    config = Path(checkpoint_path).parent.parent.parent/'files'/'grappa_config.yaml'

    config = load_yaml(config)


    def remove_module_prefix(state_dict):
        """ Remove the 'model.' prefix in the beginning of the keys from the state dict keys """
        new_state_dict = {}
        for k,v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    model = model_from_config(config['model_config'])

    full_model = torch.nn.Sequential(
        model,
        Energy(suffix=''),
        Energy(suffix='_ref', write_suffix="_classical_ff")
    )

    state_dict = remove_module_prefix(state_dict)
    full_model.load_state_dict(state_dict)
    model = next(iter(full_model.children()))
    return model
model = get_grappa_model(checkpoint_path)
#%%

full_model = torch.nn.Sequential(
    model,
    Energy(suffix=''),
)
evaluator = ExplicitEvaluator(keep_data=True)
for g, dsnames in vl:
    g = full_model(g)
    evaluator.step(g, dsnames)

# %%
evaluator.pool()

# %%
import matplotlib.pyplot as plt

# plot gradient components
plt.figure(figsize=(10,10))
plt.scatter(evaluator.reference_gradients['Capped_AA_break_breakpoint'].flatten().numpy(), evaluator.gradients['Capped_AA_break_breakpoint'].flatten().numpy())
# %%
