#%%
from grappa.training.trainrun import do_trainrun
from grappa.utils.dataset_utils import get_data_path
from grappa.training.config import default_config

#%%
config = default_config(model_tag='small')


config['data_config']['datasets'] = [
    str(get_data_path()/'dgl_datasets'/'spice-des-monomers'),
    str(get_data_path()/'dgl_datasets'/'spice-dipeptide'),
    # str(get_data_path()/'dgl_datasets'/'spice_dipeptide_amber99sbildn'),
]

config['data_config']['pure_train_datasets'] = []

config['data_config']['train_batch_size'] = 16
config['data_config']['val_batch_size'] = 16
config['data_config']['conf_strategy'] = 100

config['lit_model_config']['start_qm_epochs'] = 1
config['lit_model_config']['energy_weight'] = 1
config['lit_model_config']['gradient_weight'] = 0.5
config['lit_model_config']['param_weight'] = 1e-4
config['lit_model_config']['tuplewise_weight'] = 1e-4
config['lit_model_config']['add_restarts'] = []

config['trainer_config']['max_epochs'] = 500
config['test_model'] = False

config["model_config"]["learnable_statistics"] = True

config['model_config']['gated_torsion'] = True

trafo_width=256
symmetriser_width=128
dropout=0.1
n_heads=8

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
#%%

model = do_trainrun(config, project='test_resumption')
# %%
