"""
"""
#%%
from grappa.training.trainrun import do_trainrun
from grappa.training.get_dataloaders import get_dataloaders
from grappa.utils.dataset_utils import get_data_path
from grappa.training.config import default_config

#%%
config = default_config(model_tag='small')


# config['data_config']['datasets'] = [
#     str(get_data_path()/'dgl_datasets'/'spice-des-monomers'),
#     str(get_data_path()/'dgl_datasets'/'spice-dipeptide'),
#     str(get_data_path()/'dgl_datasets'/'gen2'),
#     str(get_data_path()/'dgl_datasets'/'spice_dipeptide_amber99sbildn'),
#     str(get_data_path()/'dgl_datasets'/'rna-diverse'),
# ]

config['data_config']['datasets'] = [
# config['data_config']['datasets'] = [
    str(get_data_path()/'dgl_datasets'/'spice-des-monomers'),
    str(get_data_path()/'dgl_datasets'/'spice-dipeptide'),
    # str(get_data_path()/'dgl_datasets'/'spice_dipeptide_amber99sbildn'),
    # str(get_data_path()/'dgl_datasets'/'tripeptides_amber99sbildn'),
    # str(get_data_path()/'dgl_datasets'/'spice-pubchem'),
    # str(get_data_path()/'dgl_datasets'/'pepconf-dlc'),
    str(get_data_path()/'dgl_datasets'/'protein-torsion'),
]



config['data_config']['pure_val_datasets'] = []
config['data_config']['pure_train_datasets'] = []

config['data_config']['train_batch_size'] = 8
config['data_config']['val_batch_size'] = 1

config['data_config']['weights'] = {
    # 'spice-des-monomers': 5,
    # 'pepconf-dlc': 2.,
    # 'protein-torsion': 5.,
    # 'rna-diverse': 10.,
}

config['data_config']['balance_factor'] = 0.5

config['lit_model_config']['start_qm_epochs'] = 0
config['lit_model_config']['energy_weight'] = 1
config['lit_model_config']['gradient_weight'] = 0.5
config['lit_model_config']['param_weight'] = 1e-4
config['lit_model_config']['time_limit'] = 25

config['trainer_config']['max_epochs'] = 10000
config['test_model'] = True

config["model_config"]["learnable_statistics"] = True

config['model_config']['gated_torsion'] = True

trafo_width=128
symmetriser_width=128
dropout=0.1
n_heads=4

config['model_config']['parameter_dropout'] = dropout
config['model_config']['gnn_dropout_attention'] = dropout
config['model_config']['gnn_dropout_conv'] = dropout
config['model_config']['gnn_dropout_final'] = dropout
config['model_config']['gnn_dropout_initial'] = dropout


config['model_config']['bond_transformer_width'] = trafo_width
config['model_config']['angle_transformer_width'] = trafo_width
config['model_config']['proper_transformer_width'] = trafo_width
config['model_config']['improper_transformer_width'] = trafo_width

config['model_config']['bond_symmetriser_width'] = symmetriser_width
config['model_config']['angle_symmetriser_width'] = symmetriser_width
config['model_config']['proper_symmetriser_width'] = symmetriser_width
config['model_config']['improper_symmetriser_width'] = symmetriser_width

# config['model_config']['in_feat_name'] += ['is_radical']

config['trainer_config']['name'] = 'learnable_statistics'
#%%
# model = do_trainrun(config, project='test_new_models')

#%%
tr, vl, te = get_dataloaders(**config['data_config'])
# %%
import torch

xyz = []
dsnames = []
for g, dsname in tr.to('cuda'):
    xyz.append(g.nodes['n1'].data['xyz'].flatten())
    dsnames += dsname

xyz = torch.cat(xyz, dim=0)
print(xyz.mean())

# print counts of dsnames
print({dsname: dsnames.count(dsname) for dsname in set(dsnames)})
# %%
