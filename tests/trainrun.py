#%%
from grappa.training.trainrun import do_trainrun
from grappa.utils.dataset_utils import get_data_path
from grappa.training.config import default_config

#%%
config = default_config(model_tag='med', benchmark=True)
config['model_config']['n_conv'] = 3
config['model_config']['n_att'] = 2
config['model_config']['rep_feats'] = 256
config['model_config']['n_att_readout'] = 4


# config['data_config']['datasets'] = [
#     str(get_data_path()/'dgl_datasets'/'spice-des-monomers'),
#     str(get_data_path()/'dgl_datasets'/'spice-dipeptide'),
# ]


# config['data_config']['datasets'] += [
# # config['data_config']['datasets'] = [
#     str(get_data_path()/'dgl_datasets'/'rna-nucleoside'),
#     str(get_data_path()/'dgl_datasets'/'rna-diverse'),
#     str(get_data_path()/'dgl_datasets'/'rna-trinucleotide'),
# ]

config['data_config']['partition'][1].update({
    # 'spice-des-monomers': (0., 1., 0.),
    # 'pepconf-dlc': (0., 1., 0.),
})

config['data_config']['pure_test_datasets'] = []
config['data_config']['pure_val_datasets'] = [str(get_data_path()/"dgl_datasets"/'rna-trinucleotide')]

config['data_config']['train_batch_size'] = 10

# config['model_config']['in_feat_name'] += ['sp_hybridization']

# config['lit_model_config']['log_classical'] = True
# config['lit_model_config']['log_params'] = True
config['lit_model_config']['start_qm_epochs'] = 5
config['lit_model_config']['start_qm_epochs'] = 0
config['lit_model_config']['classical_epochs'] = 5
config['lit_model_config']['energy_weight'] = 20

# config['trainer_config']['max_epochs'] = 150


do_trainrun(config, project='tests')