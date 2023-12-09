"""
Train on a mixture of protein, small molecule and amber99 protein data.
"""
#%%
from grappa.training.trainrun import do_trainrun
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

config['data_config']['datasets'] += [
# config['data_config']['datasets'] = [
    # str(get_data_path()/'dgl_datasets'/'spice-des-monomers'),
    # str(get_data_path()/'dgl_datasets'/'spice-dipeptide'),
    str(get_data_path()/'dgl_datasets'/'spice_dipeptide_amber99sbildn'),
    str(get_data_path()/'dgl_datasets'/'tripeptides_amber99sbildn'),
    # str(get_data_path()/'dgl_datasets'/'spice-pubchem'),
    # str(get_data_path()/'dgl_datasets'/'pepconf-dlc'),
    # str(get_data_path()/'dgl_datasets'/'protein-torsion'),
]

config['data_config']['train_batch_size'] = 10
config['data_config']['val_batch_size'] = 10

config['data_config']['partition'] = [[0.8, 0.1, 0.1], {
    # 'pepconf-dlc': (0., 1., 0.),
    # 'spice-des-monomers': (0., 1., 0.),
    # 'gen2': (0., 1., 0.),
    # 'gen2-torsion': (0., 1., 0.),
    'protein-torsion': (0., 1., 0.),
    'tripeptides_amber99sbildn': (0., 1., 0.),
    'rna-nucleoside': (1., 0., 0.),
}]

# config['data_config']['partition'] = [0.5, 0.5, 0.0]

config['lit_model_config']['start_qm_epochs'] = 1
config['lit_model_config']['energy_weight'] = 1
config['lit_model_config']['gradient_weight'] = 1e-1
config['lit_model_config']['add_restarts'] = []

config['trainer_config']['max_epochs'] = 500
config['test_model'] = True

config['model_config']['gated_torsion'] = True

trafo_width=128
symmetriser_width=128
dropout=0.1
n_heads=4

config['model_config']['parameter_dropout'] = dropout
config['model_config']['gnn_dropout_attention'] = dropout

config['model_config']['bond_transformer_width'] = trafo_width
config['model_config']['angle_transformer_width'] = trafo_width
config['model_config']['proper_transformer_width'] = trafo_width
config['model_config']['improper_transformer_width'] = trafo_width

config['model_config']['bond_symmetriser_width'] = symmetriser_width
config['model_config']['angle_symmetriser_width'] = symmetriser_width
config['model_config']['proper_symmetriser_width'] = symmetriser_width
config['model_config']['improper_symmetriser_width'] = symmetriser_width



model = do_trainrun(config, project='test_new_model')