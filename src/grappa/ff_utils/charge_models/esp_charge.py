#%%
import warnings
from openmm import unit

# supress openff warning:
import logging
logging.getLogger("openff").setLevel(logging.ERROR)
from openff.toolkit.topology import Molecule

# only import esploma_charge in the function to make the package independent of it
def get_espaloma_charge_model():
    from espaloma_charge.openff_wrapper import EspalomaChargeToolkitWrapper
    toolkit_registry = EspalomaChargeToolkitWrapper()

    def get_charges(top:Molecule, radical_indices=None):
        top.assign_partial_charges('espaloma-am1bcc', toolkit_registry=toolkit_registry)
        # convert to openmm quantity
        with warnings.catch_warnings(): # catch a warning given by espaloma charge because the openff molecule has a conformation
            warnings.simplefilter("ignore")
            c_esp = [c.magnitude for c in top.partial_charges]
        return [unit.Quantity(c, unit.elementary_charge) for c in c_esp]
    
    return get_charges

#%%
if __name__ == "__main__":

    esp_model = get_espaloma_charge_model()
    #%%
    from PDBData.PDBDataset import PDBDataset
    dspath = "/hits/fast/mbm/seutelf/data/datasets/PDBDatasets/spice_amber99sbildn"
    ds = PDBDataset.load_npz(dspath)
    ds.mols = ds.mols[:10]
    import copy
    graphs = copy.deepcopy(ds.to_dgl())
    #%%
    ds.parametrize(get_charges=esp_model, suffix="_esp", openff_charge_flag=True)
    esp_graphs = ds.to_dgl()
    #%%
    for i in range(len(ds)):
        print(esp_graphs[i].nodes["n1"].data["q_esp"][10:15, 0])
        print(graphs[i].nodes["n1"].data["q_amber99sbildn"][10:15, 0])
        print()
    #%%
    #%%
    # OLD TESTS
    from openff.toolkit.topology import Molecule
    from espaloma_charge.openff_wrapper import EspalomaChargeToolkitWrapper
    toolkit_registry = EspalomaChargeToolkitWrapper()

    #%%
    from PDBData.PDBDataset import PDBDataset
    dspath = "/hits/fast/mbm/seutelf/data/datasets/PDBDatasets/spice_amber99sbildn"
    # %%
    ds = PDBDataset.load_npz(dspath)
    # %%
    mol = ds[0].to_openff()
    # %%
    mol.assign_partial_charges('espaloma-am1bcc', toolkit_registry=toolkit_registry)
    c_esp = [c.magnitude for c in mol.partial_charges]
    # %%
    g = ds[0].to_dgl()
    c_amb = g.nodes["n1"].data["q_amber99sbildn"][:,0].tolist()

    # %%
    print(*[f"{a} {b}" for a, b in zip(c_esp, c_amb)], sep="\n")
    # %%
