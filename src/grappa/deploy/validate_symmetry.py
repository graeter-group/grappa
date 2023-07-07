#%%
import numpy as np
import torch
from grappa.run import run_utils
# from grappa.deploy.deploy import model_from_version
from grappa.models.get_models import get_full_model

def validate_bond_symmetry(g, suffix=""):
    """
    Returns True if the bond features are symmetric wrt permutation of the two atoms.
    Is not implemented efficiently, should only be used for debugging.
    """
    pairs = g.nodes["n2"].data[f"idxs{suffix}"]
    k = g.nodes["n2"].data[f"k{suffix}"]
    eq = g.nodes["n2"].data[f"eq{suffix}"]
    
    flag_symmetric = True
    for n,(i,j) in enumerate(pairs):
        # find other pair:
        n2 = torch.where(torch.all(pairs == torch.tensor([j,i]), dim=-1))[0]
        if not (torch.allclose(k[n], k[n2]) and torch.allclose(eq[n], eq[n2])):
            flag_symmetric = False
            break
    return flag_symmetric
#%%
if __name__=="__main__":
    from grappa.models import espaloma_default
    [ds], _ = run_utils.get_data(["/hits/fast/mbm/seutelf/data/datasets/PDBDatasets/spice/amber99sbildn_60_dgl.bin"], n_graphs=1)
    model_new = get_full_model(old=False)
    model_old = get_full_model(old=True)
    model_esp = espaloma_default.get_model()
    #%%
    for g in ds:
        g.nodes["n1"].data["h0"] = torch.randn((g.num_nodes("n1"), 117))
        g = model_esp(g)
        print("esp: ", validate_bond_symmetry(g))
        g = model_old(g)
        print("old: ", validate_bond_symmetry(g))
        g = model_new(g)
        print("new: ", validate_bond_symmetry(g))
# %%
