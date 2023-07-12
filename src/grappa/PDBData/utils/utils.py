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