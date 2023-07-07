#%%

from pathlib import Path

import dgl
from openmm.app import ForceField
import numpy as np
from grappa.run import run_utils
from grappa.deploy import deploy
import shutil
import os

def make_dicts(
        storepath="/hits/fast/mbm/share/seutelf/parameters",
        modelname="trained_on_spice",
        dspath="/hits/fast/mbm/seutelf/data/datasets/PDBDatasets/AA_opt_nat",
        ds_name="amber99sbildn_60",
        version="/hits/fast/mbm/seutelf/grappa/mains/runs/rad/versions/0",
        collagen=True):
    """
    Hacky way to get a dictionary of parameters from a trained model for a given dataset. In the future, we provide a method where you can simply give a pdb instead of a PDBDataset. Therefore, it will also be independetn of the PDBData package.
    """

    from PDBData.PDBDataset import PDBDataset
    from PDBData.create_graph import offmol_indices
    from PDBData.charge_models.charge_models import model_from_dict


    storename = Path(Path(dspath).stem)
    target_path = Path(storepath)/storename/Path(modelname)
    os.makedirs(target_path, exist_ok=True)

    model_name = "best_model.pt"
    model = deploy.model_from_version(version, model_name=model_name)
    config_path = Path(version)/Path("config.yaml")
    model_path = Path(version)/Path(model_name)

    shutil.copy(config_path, target_path/Path("config.yaml"))
    shutil.copy(model_path, target_path/Path(model_name))


    pdb_ds = PDBDataset.load_npz(Path(dspath)/Path(ds_name))

    pdb_ds.parametrize(forcefield=ForceField("amber99sbildn.xml"), calc_energies=False, allow_radicals=True, get_charges=model_from_dict("heavy"), charge_suffix="_amber99sbildn", collagen=collagen)

    ds = pdb_ds.to_dgl()


    # assume that the indices in dgl are repeated and that their entries are all the same
    # change this in own model with symmetry functions
    # also make this much less hard-coded

    def hard_coded_get_parameters(level:str, g, mol, suffix="_amber99sbildn"):

        lvl = int(level[1])
        if "improper" in level:
            lvl = 5
        methods = [None, offmol_indices.atom_indices, offmol_indices.bond_indices, offmol_indices.angle_indices, offmol_indices.proper_torsion_indices, offmol_indices.improper_torsion_indices]
        off_indices = methods[lvl](mol.to_openff_graph())


        dgl_indices = g.nodes[level].data["idxs"][:len(off_indices)].numpy()
        if level == "n1":
            dgl_indices = dgl_indices[:, 0]

        assert np.all(dgl_indices == off_indices), f"{off_indices}\n!=\n{dgl_indices}, shapes: off {off_indices.shape}, dgl {dgl_indices.shape}"

        out_dict = {"idxs":dgl_indices}
        if level == "n1":
            out_dict.update({
                "q": g.nodes[level].data[f"q{suffix}"][:len(off_indices)].numpy()[:,0],
                "sigma": g.nodes[level].data[f"sigma{suffix}"][:len(off_indices)].numpy()[:,0],
                "epsilon": g.nodes[level].data[f"epsilon{suffix}"][:len(off_indices)].numpy()[:,0]})
        
        elif level == "n2":
            out_dict.update({
                "k": g.nodes[level].data[f"k{suffix}"][:len(off_indices)].numpy()[:,0],
                "eq": g.nodes[level].data[f"eq{suffix}"][:len(off_indices)].numpy()[:,0]})
        elif level == "n3":
            out_dict.update({
                "k": g.nodes[level].data[f"k{suffix}"][:len(off_indices)].numpy()[:,0],
                "eq": g.nodes[level].data[f"eq{suffix}"][:len(off_indices)].numpy()[:,0]})
        elif level == "n4":
            out_dict.update({
                "ks": g.nodes[level].data[f"k{suffix}"][:len(off_indices)].numpy()})
        elif level == "n4_improper":
            out_dict.update({
                "ks": g.nodes[level].data[f"k{suffix}"][:len(off_indices)].numpy()})

        return out_dict


    for i,g in enumerate(ds):
        g = model(g)
        mol = pdb_ds[i]
        path = target_path/Path("data")/Path(f"{mol.name}")
        os.makedirs(path, exist_ok=True)
        
        params = {}

        interaction_type = ["atom", "bond", "angle", "proper", "improper"]
        for i, l in enumerate(["n1", "n2", "n3", "n4", "n4_improper"]):
            p = hard_coded_get_parameters(l, g, mol)
            itype = interaction_type[i]
            for ptype in p.keys():
                params[f"{itype}_{ptype}"] = p[ptype]

        np.savez(path/Path("parameters.npz"), **params)

        mol.write_pdb(path/Path("pep.pdb"))