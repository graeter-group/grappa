#%%
from openmm.app import PDBFile

import dgl
import h5py

import numpy as np
import tempfile
import os.path
import torch
import openmm.app
from typing import Union
from pathlib import Path
from typing import List, Tuple, Dict, Union, Callable
import matplotlib.pyplot as plt

import copy



def get_pdbfile(hdf5:str, idx:int):
    with h5py.File(hdf5, "r") as f:
        for i, name in enumerate(f.keys()):
            if i==idx:
                pdb = f[name]["pdb"]
                with tempfile.TemporaryDirectory() as tmp:
                    pdbpath = os.path.join(tmp, 'pep.pdb')
                    with open(pdbpath, "w") as pdb_file:
                        pdb_file.writelines([line.decode('UTF-8') for line in pdb])
                    return PDBFile(pdbpath)


#%%


def rename_cap_Hs(topology:openmm.app.topology.Topology)->openmm.app.topology.Topology:
    # rename the Hs of the cap atoms if necessary: (we adopt the convention to name them HH31, HH32, HH33 for backwards compatibility):
    for res in topology.residues():
        if res.name in ["ACE", "NME"]:
            for atom in res.atoms():
                if atom.name == "H3":
                    atom.name = "HH31"
                elif atom.name == "H2":
                    atom.name = "HH32"
                elif atom.name == "H1":
                    atom.name = "HH33"
    return topology



def eval_params(param_dict, param_dict_ref, plotpath:Union[str, Path], fontsize=16, ff_name="Forcefield", ref_name="Amber99sbildn", collagen=False, fontname="DejaVu Sans", figsize=6):
    """
    Creates a plot in which parameters predicted by the given forcefield are compared with those in the dataset.
    Improper torsion parameters cannot be compared as we have 3 improper terms with independent parameters.
    """

    if not plotpath is None:
        plotpath = Path(plotpath)
        plotpath.mkdir(parents=True, exist_ok=True)

    if not fontname is None:
        import matplotlib as mpl
        mpl.rc('font',family=fontname)
    
    bond_eqs = []
    bond_eqs_ref = []

    bond_ks = []
    bond_ks_ref = []

    angle_eqs = []
    angle_eqs_ref = []

    angle_ks = []
    angle_ks_ref = []

    torsion_ks = []
    torsion_ks_ref = []


    bond_eqs.append(param_dict["bond_eq"].flatten())
    bond_eqs_ref.append(param_dict_ref["bond_eq"].flatten())

    bond_ks.append(param_dict["bond_k"].flatten())
    bond_ks_ref.append(param_dict_ref["bond_k"].flatten())

    angle_eqs.append(param_dict["angle_eq"].flatten())
    angle_eqs_ref.append(param_dict_ref["angle_eq"].flatten())

    angle_ks.append(param_dict["angle_k"].flatten())
    angle_ks_ref.append(param_dict_ref["angle_k"].flatten())

    ks = param_dict["proper_ks"]
    phases = param_dict["proper_phases"]
    ks = np.where(phases==0, ks, -ks)
    torsion_ks.append(ks[:,:4].flatten())
    ks_ref = param_dict_ref["proper_ks"]
    phases_ref = param_dict_ref["proper_phases"]
    ks_ref = np.where(phases_ref==0, ks_ref, -ks_ref)
    torsion_ks_ref.append(ks_ref[:,:4].flatten())


    bond_eqs = np.concatenate(bond_eqs, axis=0)
    bond_eqs_ref = np.concatenate(bond_eqs_ref, axis=0)

    bond_ks = np.concatenate(bond_ks, axis=0)
    bond_ks_ref = np.concatenate(bond_ks_ref, axis=0)

    angle_eqs = np.concatenate(angle_eqs, axis=0)
    angle_eqs_ref = np.concatenate(angle_eqs_ref, axis=0)

    angle_ks = np.concatenate(angle_ks, axis=0)
    angle_ks_ref = np.concatenate(angle_ks_ref, axis=0)

    torsion_ks = np.concatenate(torsion_ks, axis=0)
    torsion_ks_ref = np.concatenate(torsion_ks_ref, axis=0)



    def get_global_min_max(data1, data2):
        return min(min(data1), min(data2)), max(max(data1), max(data2))

    # Bond-related plots
    fig1, ax1 = plt.subplots(1,2, figsize=(2*figsize, figsize))
    min_val, max_val = get_global_min_max(bond_eqs_ref, bond_eqs)
    ax1[0].scatter(bond_eqs_ref, bond_eqs, s=6, alpha=0.2)
    ax1[0].plot([min_val, max_val], [min_val, max_val], color="black", linewidth=0.8)
    ax1[0].set_title("Bond Equilibrium Distances [Å]", fontsize=fontsize, fontname=fontname)
    ax1[0].set_xlabel(f"{ref_name}", fontsize=fontsize, fontname=fontname)
    ax1[0].set_ylabel(f"{ff_name}", fontsize=fontsize, fontname=fontname)
    ax1[0].tick_params(axis='both', which='major', labelsize=fontsize-2)

    min_val, max_val = get_global_min_max(bond_ks_ref, bond_ks)
    ax1[1].scatter(bond_ks_ref, bond_ks, s=6, alpha=0.2)
    ax1[1].plot([min_val, max_val], [min_val, max_val], color="black", linewidth=0.8)
    ax1[1].set_title("Bond Force Constants [kcal/mol/Å²]", fontsize=fontsize, fontname=fontname)
    ax1[1].set_xlabel(f"{ref_name}", fontsize=fontsize, fontname=fontname)
    ax1[1].set_ylabel(f"{ff_name}", fontsize=fontsize, fontname=fontname)
    ax1[1].tick_params(axis='both', which='major', labelsize=fontsize-2)

    # Angle-related plots
    fig2, ax2 = plt.subplots(1,2, figsize=(2*figsize, figsize))
    min_val, max_val = get_global_min_max(angle_eqs_ref, angle_eqs)
    ax2[0].scatter(angle_eqs_ref, angle_eqs, s=6, alpha=0.2)
    ax2[0].plot([min_val, max_val], [min_val, max_val], color="black", linewidth=0.8)
    ax2[0].set_title("Angle Equilibrium Angles [Deg]", fontsize=fontsize, fontname=fontname)
    ax2[0].set_xlabel(f"{ref_name}", fontsize=fontsize, fontname=fontname)
    ax2[0].set_ylabel(f"{ff_name}", fontsize=fontsize, fontname=fontname)
    ax2[0].tick_params(axis='both', which='major', labelsize=fontsize-2)

    min_val, max_val = get_global_min_max(angle_ks_ref, angle_ks)
    ax2[1].scatter(angle_ks_ref, angle_ks, s=6, alpha=0.2)
    ax2[1].plot([min_val, max_val], [min_val, max_val], color="black", linewidth=0.8)
    ax2[1].set_title("Angle Force Constants [kcal/mol/Deg²]", fontsize=fontsize, fontname=fontname)
    ax2[1].set_xlabel(f"{ref_name}", fontsize=fontsize, fontname=fontname)
    ax2[1].set_ylabel(f"{ff_name}", fontsize=fontsize, fontname=fontname)
    ax2[1].tick_params(axis='both', which='major', labelsize=fontsize-2)

    # Torsion-related plots
    fig3, ax3 = plt.subplots(1,1, figsize=(figsize*1.1, figsize))
    min_val, max_val = get_global_min_max(torsion_ks_ref, torsion_ks)
    ax3.scatter(torsion_ks_ref, torsion_ks, s=6, alpha=0.2)
    ax3.plot([min_val, max_val], [min_val, max_val], color="black", linewidth=0.8)
    ax3.set_title("Torsion Coefficients [kcal/mol]", fontsize=fontsize, fontname=fontname)
    ax3.set_xlabel(f"{ref_name}", fontsize=fontsize, fontname=fontname)
    ax3.set_ylabel(f"{ff_name}", fontsize=fontsize, fontname=fontname)
    ax3.tick_params(axis='both', which='major', labelsize=fontsize-2)

    # Save and close the plots
    plotpath = Path(plotpath)
    fig1.tight_layout()
    fig1.savefig(plotpath / "bond_params.png", dpi=500)
    plt.close(fig1)

    fig2.tight_layout()
    fig2.savefig(plotpath / "angle_params.png", dpi=500)
    plt.close(fig2)

    fig3.tight_layout()
    fig3.savefig(plotpath / "torsion_params.png", dpi=500)
    plt.close(fig3)
# %%
