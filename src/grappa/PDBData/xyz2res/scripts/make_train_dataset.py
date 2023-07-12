

#%%
import torch
import dgl
from openmm.app import PDBFile
from openmm.unit import angstrom
import numpy as np
from pathlib import Path
import random
import json
import os

from grappa.PDBData.xyz2res.constants import MAX_ELEMENT
from grappa.units import RESIDUES



def make_train_set(name:str='pdbs', path:str=str(Path("data/pdbs")), collagen=True):
    import ase
    from ase.geometry.analysis import Analysis
    """
    Run with arguments:
        '-p', '--path'
            default="data/pdbs"
            type=str
            help="The path from which files with .pdb ending are loaded (recursively)."

        '-n', '--name'
        default='pdbs'
        type=str
        help="Name of the train set"

    Generate dgl graphs with CA atom type one-hot encoded from pdb files.
    Do this by converting the pdb file to ase molecule to check whether ase gets the same connectivity as the pdb file.
    """
    class BondMismatch(Exception):
        pass

    POS_UNIT = angstrom

    graphs = []
    labels = [] # store unique (random) integers to store them in the graph
    name_dict={} # map integers to names

    if collagen:
        from grappa.ff_utils.classical_ff.collagen_utility import add_bonds

    # %%
    counter = 0
    counter
    errors = []
    err_types = []
    err_res = dict.fromkeys(RESIDUES)
    for k in err_res.keys():
        err_res[k] = 0

    for filename in Path(path).rglob('*.pdb'):
        print(counter, end="\r")
        counter += 1
        try:
            mol = PDBFile(str(filename))
            if collagen:
                mol.topology = add_bonds(mol.topology)
            atom_numbers = []
            residues = []
            res_indices = []
            atom_names = []
            res_names = []
            for a in mol.topology.atoms():
                atom_numbers.append(a.element.atomic_number)
                residues.append(RESIDUES.index(a.residue.name))
                res_indices.append(a.residue.index)
                atom_names.append(a.name)
                res_names.append(a.residue.name)

            res_names = set(res_names)

            bonds = []
            for b in mol.topology.bonds():
                a0, a1 = b[0].index, b[1].index
                if a0 < a1:
                    bonds.append((a0,a1))
                else:
                    bonds.append((a1,a0))

            n = mol.topology.getNumAtoms()
            pos = mol.positions
            pos = pos.value_in_unit(POS_UNIT)
            pos = np.array([np.array(v) for v in pos])

            ase_mol = ase.Atoms(f"N{n}")
            ase_mol.set_positions(pos)
            ase_mol.set_atomic_numbers(atom_numbers)

            ana = Analysis(ase_mol)
            connectivity = ana.nl[0].get_connectivity_matrix()

            node_pairs = [(n1,n2) for (n1,n2) in connectivity.keys() if n1!=n2]

            # check the bonds.
    
            if set(bonds) != set(node_pairs):
                ona = (set(bonds) - set(node_pairs))
                ano = (set(node_pairs) - set(bonds))
                atoms = [a.name for a in mol.topology.atoms()]

                errstr = f"Encountered dissimilar bonds at {str(filename)}.\nIn openmm but not ASE: {ona}, atoms {[(atoms[i], atoms[j]) for (i,j) in ona]}.\nIn ASE but not openmm: {ano}, atoms {[(atoms[i], atoms[j]) for (i,j) in ano]}. (The indices are starting at zero not at one!)"

                raise BondMismatch(errstr)


            node_tensors = [ torch.tensor([node_pair[i] for node_pair in node_pairs], dtype=torch.int32) for i in [0,1] ]

            g = dgl.graph(tuple(node_tensors))
            
            g.ndata["atomic_number"] = torch.nn.functional.one_hot(torch.tensor(atom_numbers), num_classes=MAX_ELEMENT)*1.
            g.ndata["residue"] = torch.tensor(residues)
            g.ndata["res_index"] = torch.tensor(res_indices)

            g.ndata["c_alpha"] = torch.tensor([name == "CA" for name in atom_names])

            g = dgl.add_reverse_edges(g)

            # find a unique integer to label the graph
            h = counter
            while h in name_dict.keys():
                h = random.randint(-2**30, 2**30)
            
            graphs.append(g)
            labels.append(h)
            name_dict[h] = str(filename)
            
        
        except BondMismatch as e:
            errors.append(str(filename))
            err_types.append(type(e))
            for res in set([RESIDUES[i] for i in residues]):
                err_res[res] += 1
            # raise e
            continue

        except Exception as e:
            raise e

    #%%
    if len(errors) > 0:
        print("errors at: ")
        print(errors)
        print()
        print(err_types)
        print()
        print("errors occured n times for ")
        print(err_res)
    print(f"successful for {len(graphs)} out of {counter} molecules.")
    #%%

    os.makedirs("data", exist_ok=True)

    dspath = str(Path('data')/Path(name+"_dgl"+'.bin'))
    dgl.data.utils.save_graphs(dspath, graphs, {"name_hash": torch.tensor(labels, dtype=torch.int32)})

    with open(str(Path('data')/Path(name+"_names"+'.json')), "w") as f:
        json.dump(name_dict, f)

    # %%

if __name__=="__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', default=str(Path("data/pdbs")), type=str, help="The path from which files with .pdb ending are loaded (recursively).")
    parser.add_argument('-n', '--name', default='pdbs', type=str, help="Name of the train set")

    args = parser.parse_args()

    path = args.path

    name = args.name

    make_train_set(name=name, path=path)
    
    # generate the pep-1 dataset for the residue hashes:
    print("generating pep-1 dataset for residue hashes")
    make_train_set(path="data/pdbs/pep1", name="pep1")