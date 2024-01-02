#%%
import yaml
from grappa.training.resume_trainrun import get_dir_from_id
from grappa.training.get_dataloaders import get_dataloaders
from grappa.utils.dataset_utils import get_data_path
from grappa.utils.train_utils import get_model

#%%


datasets = [
        str(get_data_path()/"dgl_datasets"/dsname) for dsname in
        [
            "rna-diverse",
            'spice-des-monomers',
        ]
    ]

tr, val, te = get_dataloaders(datasets=datasets)
# %%
run_id = 'p1e1ojyd'
project = 'hpo_grappa_final'

wandb_folder = '/hits/fast/mbm/seutelf/grappa/experiments/hyperparameter_optimization/wandb'

run_dir = get_dir_from_id(wandb_folder=wandb_folder, run_id=run_id)

model_config = yaml.load(open(run_dir / 'files/grappa_config.yaml', 'r'), Loader=yaml.FullLoader)['model_config']

ckpt_dir = run_dir / 'files/checkpoints/last.ckpt'

model = get_model(model_config=model_config, checkpoint_path=None).to('cuda')
# %%
model.eval()
# model.train()
for i, (g, dsnames) in enumerate(val):
    g = g.to('cuda')
    print(f'num conf: {g.nodes["g"].data["energy_ref"].shape[1]}')
    g = model(g)
    print(f'n2_k_nan: {g.nodes["n2"].data["k"].isnan().any()}')
    # print(f'n2_eq_nan: {g.nodes["n2"].data["eq"].isnan().any()}')
    
    # print(f'n3_k_nan: {g.nodes["n3"].data["k"].isnan().any()}')
    # print(f'n3_eq_nan: {g.nodes["n3"].data["eq"].isnan().any()}')

    # print(f'n4_feat_nan: {g.nodes["n4"].data["k"].isnan().any()}')

    print()

    if i > -1:
        break
# %%

# childs = list(model.children())

# bond_writer = childs[1].bond_writer
# # %%
# bond_writer.eval()
# g = bond_writer(g)
# print(f'n2_k_nan: {g.nodes["n2"].data["k"].isnan().any()}')

# # %%
# for i, submodel in enumerate(bond_writer.children()):
#     pass