#%%
from grappa.training.trainrun import do_trainrun
from grappa.utils.run_utils import get_rundir, get_data_path
from grappa.training.config import default_config

#%%
config = default_config(model_tag='small')
config['project'] = 'grappa_tests'

config['data_config']['datasets'] = [str(get_data_path()/'dgl_datasets'/'spice-des-monomers')]

# config['in_feat_name'] += ['sp_hybridization']

do_trainrun(config)