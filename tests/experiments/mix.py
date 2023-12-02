"""
Train on a mixture of protein, small molecule and amber99 protein data.
"""
#%%
from grappa.training.trainrun import do_trainrun
from grappa.utils.run_utils import get_rundir, get_data_path
from grappa.training.config import default_config

#%%
config = default_config(model_tag='small')
config['model_config']['n_conv'] = 3
config['model_config']['n_att'] = 2
config['model_config']['rep_feats'] = 256
config['model_config']['n_att_readout'] = 4


# config['data_config']['datasets'] = [
#     str(get_data_path()/'dgl_datasets'/'spice-des-monomers'),
#     str(get_data_path()/'dgl_datasets'/'spice-dipeptide'),
# ]

# NOTE CHECK WHETHER DATASETS ONLY ONCE IN LIST

# config['data_config']['datasets'] += [
config['data_config']['datasets'] = [
    str(get_data_path()/'peptides'/'dgl_datasets'/'spice'), # these are amber99 parametrised
    # str(get_data_path()/'dgl_datasets'/'spice-des-monomers'),
    # str(get_data_path()/'dgl_datasets'/'spice-dipeptide'),
    # str(get_data_path()/'dgl_datasets'/'spice-pubchem'),
    # str(get_data_path()/'dgl_datasets'/'pepconf-dlc'),
    # str(get_data_path()/'dgl_datasets'/'protein-torsion'),
]
config['data_config']['train_batch_size'] = 10

# config['model_config']['in_feat_name'] += ['sp_hybridization']
# config['model_config']['in_feat_name'] = ['atomic_number', 'partial_charge']

# config['lit_model_config']['log_classical'] = True
# config['lit_model_config']['log_params'] = True
config['lit_model_config']['start_qm_epochs'] = 5
config['lit_model_config']['classical_epochs'] = 5
config['lit_model_config']['energy_weight'] = 50
config['lit_model_config']['gradient_weight'] = 1

config['trainer_config']['max_epochs'] = 10000


model = do_trainrun(config, project='tests')