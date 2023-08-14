#%%
import numpy as np
import dgl
import torch
import tempfile
import os.path
from .PDBMolecule import PDBMolecule
from ..ff_utils.create_graph import find_radical
from pathlib import Path
from typing import Union, List, Tuple, Dict, Any, Optional
import h5py
from grappa.units import DISTANCE_UNIT, ENERGY_UNIT, FORCE_UNIT
from openmm.unit import bohr, Quantity, hartree, mole
from openmm.unit import unit
import random

from grappa.ff import ForceField as GrappaFF
from openmm.app import ForceField

from grappa.ff_utils.classical_ff.collagen_utility import get_mod_amber99sbildn

import json
import copy
import matplotlib.pyplot as plt

#%%

class PDBDataset:
    """
    Handles the generation of dgl graphs from PDBMolecules. Stores PDBMolecules in a list.
    Uses an hdf5 file containing the pdbfiles as string and xyz, elements, energies and gradients as numpy arrays to store tha dataset on a hard drive.
    """
    def __init__(self, mols:List[PDBMolecule]=[], info=True)->None:
        self.mols:List[PDBMolecule] = copy.deepcopy(mols)
        self.info:str = info # flag whether to print information on certain operations.
        # self.subsets:Dict[str, np.ndarray] = {"full":np.arange(len(mols))} # dictionary of subsets of the dataset. maps name of the subset to array of indices. does not support nested subsets currently.

    def __len__(self)->int:
        return len(self.mols)

    def __getitem__(self, idx:Union[int, list, np.ndarray, slice]):
        if isinstance(idx, int):
            return self.mols[idx]

        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self.mols))
            return PDBDataset(self.mols[start:stop:step], info=self.info)

        if isinstance(idx, list) or isinstance(idx, np.ndarray):
            return PDBDataset([self.mols[i] for i in idx], info=self.info)

    
    def __setitem__(self, idx:int, mol:PDBMolecule)->None:
        self.mols[idx] = mol
    
    def __delitem__(self, idx:int)->None:
        del self.mols[idx]

    def __iter__(self)->PDBMolecule:
        return iter(self.mols)

    def __add__(self, other:"PDBDataset")->"PDBDataset":
        return PDBDataset(mols=self.mols + other.mols, info=self.info and other.info)
        

    def append(self, mol:PDBMolecule)->None:
        """
        Appends a PDBMolecule to the dataset.
        """
        self.mols.append(mol)
    

    def to_dgl(self, idxs:List[int]=None, split:List[float]=None, seed:int=0, forcefield:ForceField=None, allow_radicals=False, collagen=False)->List[dgl.DGLGraph]:
        """
        Converts a list of indices to a list of dgl graphs.
        If the Molecule is already parametrised, forcefield, allow_radicals, collagen and get_charges are ignored.
        """
        if idxs is None:
            idxs = list(range(len(self)))

        if self.info:
            n_idxs = len(idxs) if isinstance(idxs, list) else ""
            print(f"converting PDBDataset to {n_idxs} dgl graphs...")

        def _get_graph(i, idx):
            if self.info:
                print(f"converting {i+1}/{len(idxs)}", end="\r")
            return self.mols[idx].to_dgl(classical_ff=forcefield, allow_radicals=allow_radicals, collagen=collagen)
        
        if self.info:
            print("\n")
        
        glist = [_get_graph(i, idx) for i, idx in enumerate(idxs)]

        if split is None:
            return glist
        else:
            return dgl.data.utils.split_dataset(glist, split, shuffle=True, random_state=seed)
        

    def split(self, split:Tuple[float, float, float]=(0.8, 0.1, 0.1), seed:int=0, existing_split_names:Tuple[List[str], List[str], List[str]]=None)->Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[List[str], List[str], List[str]]]:
        """
        Order: Train, Validation, Test
        Function that splits the dataset ensuring that no molecules with same name are different splits. Returns three arrays of indices for train, validation and test set. If existing_split_names is given, the split is done such that molecules with name appearing in any split of existing_split_names are in the same split. The returned names contain the names in existing_split_names for the respective split.
        """
        if self.info:
            print("splitting dataset...")
        names = [mol.name for mol in self.mols]

        assert all([name is not None for name in names]), "not all molecules have a name"


        # remove duplicates with numpy:
        unique_names = np.unique(names)

        np.random.seed(seed)
        np.random.shuffle(unique_names)
        
        n = len(unique_names)
        if len(names) == n and existing_split_names is None:
            # no duplicates, splitting is easy.
            n_train = int(n * split[0])
            n_val = int(n * split[1])
            
            train_names = unique_names[:n_train]
            val_names = unique_names[n_train:n_train+n_val]
            test_names = unique_names[n_train+n_val:]
        else:
            if existing_split_names is not None:
                train_names = list(set(names).intersection(set(existing_split_names[0])))
                val_names = list(set(names).intersection(set(existing_split_names[1])))
                test_names = list(set(names).intersection(set(existing_split_names[2])))
                remaining_names = list(set(unique_names) - set(train_names + val_names + test_names))
                
                n_train = int(len(remaining_names) * split[0])
                n_val = int(len(remaining_names) * split[1])
                
                train_names += remaining_names[:n_train]
                val_names += remaining_names[n_train:n_train+n_val]
                test_names += remaining_names[n_train+n_val:]
            else:
                n_train = int(n * split[0])
                n_val = int(n * split[1])
                
                train_names = unique_names[:n_train]
                val_names = unique_names[n_train:n_train+n_val]
                test_names = unique_names[n_train+n_val:]
                

        train_names = [str(n) for n in train_names]
        val_names = [str(n) for n in val_names]
        test_names = [str(n) for n in test_names]

        train_indices = [i for i, name in enumerate(names) if name in train_names]
        val_indices = [i for i, name in enumerate(names) if name in val_names]
        test_indices = [i for i, name in enumerate(names) if name in test_names]

        # add existing split names to the names obtained from this split
        if existing_split_names is not None:
            train_names = list(set(train_names).union(set(existing_split_names[0])))
            val_names = list(set(val_names).union(set(existing_split_names[1])))
            test_names = list(set(test_names).union(set(existing_split_names[2])))
        
        
        return (np.array(train_indices), np.array(val_indices), np.array(test_indices)), (train_names, val_names, test_names)

    
    def save(path:Union[str, Path]):
        """
        Saves the dataset to an hdf5 file.
        """
        raise NotImplementedError

    def save_npz(self, path:Union[str, Path], overwrite:bool=False)->None:
        """
        Save the dataset to npz files.
        """
        num_confs = sum(map(len, self.mols))

        if self.info:
            print(f"saving PDBDataset of {len(self)} mols and {num_confs} confs to npz files...")
        if os.path.exists(str(path)):
            if not overwrite:
                raise FileExistsError(f"path {str(path)} already exists, set overwrite=True to overwrite it.")
        if len(self) == 0:
            raise ValueError("dataset is empty.")
        
        os.makedirs(str(path), exist_ok=True)

        for id, mol in enumerate(self.mols):
            mol.compress(str(Path(path)/Path(str(id)+".npz")))
    
    @classmethod
    def load(cls, path:Union[str, Path]):
        """
        Loads a dataset from an hdf5 file.
        """
        pass

    @classmethod
    def load_npz(cls, path:Union[str, Path], keep_order=True, n_max=None, info=True):
        """
        Loads a dataset from npz files.
        """
        obj = cls()
        obj.info = info
        # load:
        if not keep_order:
            paths = Path(path).rglob('*.npz')
        else:
            paths = sorted([p for p in Path(path).rglob('*.npz')])

        for i, npz in enumerate(paths):
            if n_max is not None:
                if i >= n_max:
                    break
            mol = PDBMolecule.load(Path(npz))
            obj.append(mol)

        if len(obj.mols) == 0:
            raise ValueError(f"no npz files found in path {str(path)}")

        return obj

    def save_dgl(self, path:Union[str, Path], idxs:List[int]=None, overwrite:bool=False, forcefield:ForceField=None, allow_radicals=False, collagen=False, get_charges=None)->None:
        """
        Saves the dgl graphs that belong to the dataset.
        If the Molecule is already parametrised, forcefield, allow_radicals, collagen and get_charges are ignored.
        """
        if len(self) == 0:
            raise ValueError("cannot save empty dataset")

        if idxs is None:
            idxs = list(range(len(self)))

        if os.path.exists(str(path)):
            if not overwrite:
                raise FileExistsError(f"path {str(path)} already exists, set overwrite=True to overwrite it.")
        
        os.makedirs(str(Path(path).parent), exist_ok=True)

        dgl.save_graphs(path, self.to_dgl(idxs, forcefield=forcefield, allow_radicals=allow_radicals, collagen=collagen))

        seqpath = str(Path(path).parent/Path(path).stem) + "_seq.json"
        with open (seqpath, "w") as f:
            json.dump([self.mols[i].sequence for i in idxs], f, indent=4)


    
    def parametrize(self, forcefield:Union[ForceField,str]=get_mod_amber99sbildn(), get_charges=None, allow_radicals=False, collagen=False, skip_errs=False)->None:
        """
        Parametrizes the dataset with a forcefield.
        Writes the following entries to the graph:
        ...
        get_charges: a function that takes a topology and returns a list of charges as openmm Quantities in the order of the atoms in topology.
        if not openffmol is None, get_charge can also take an openffmolecule instead.

        If the forcefield is a string it is interpreted as the name of an openff small molecule forcefield.
        """

        skipped = []
        if self.info:
            print("parametrizing PDBDataset...")
        for i, mol in enumerate(self.mols):
            if self.info:
                print(f"parametrizing {i+1}/{len(self.mols)}", end="\r")
            try:
                mol.parametrize(forcefield=forcefield, get_charges=get_charges, allow_radicals=allow_radicals, collagen=collagen)
            except Exception as e:
                if not skip_errs:
                    raise type(e)(str(e) + f" in molecule {mol.sequence}")
                else:
                    skipped.append(i)

        if self.info:
            print()
            if len(skipped) > 0:
                print(f"skipped {len(skipped)} molecules because of errors")
                self.mols[:] = [mol for i, mol in enumerate(self.mols) if not i in skipped]

    
    def filter_validity(self, forcefield:ForceField=get_mod_amber99sbildn(), sigmas:Tuple[float,float]=(1.,1.), quickload:bool=True)->None:
        """
        Checks if the stored energies and gradients are consistent with the forcefield. Removes the entry from the dataset if the energies and gradients are not within var_param[0] and var_param[1] standard deviations of the stored energies/forces. If sigmas are 1, this corresponds to demanding that the forcefield data is better than simply always guessing the mean.
        """
        if self.info:
            print("filtering valid mols of PDBDataset by comparing with class ff...")
        keep = []
        removed = 0
        kept = 0
        for i, mol in enumerate(self.mols):
            valid = mol.conf_check(forcefield=forcefield, sigmas=sigmas, quickload=quickload)
            keep.append(valid)
            removed += int(not valid)
            kept += int(valid)
            if self.info:
                print(f"filtering {i+1}/{len(self.mols)}, kept {kept}, removed {removed}", end="\r")
        if self.info:
            print()

        # use slicing to modify the list inplace:
        self.mols[:] = [mol for i, mol in enumerate(self.mols) if keep[i]]
        

        
    def filter_confs(self, max_energy:float=65., max_force:float=200, reference=False)->None:
        """
        Filters out conformations with energies or forces that are over 60 kcal/mol away from the minimum of the dataset (not the actual minimum). Remove molecules is less than 2 conformations are left. Apply this before parametrizing or re-apply the parametrization after filtering. Units are kcal/mol and kcal/mol/angstrom.
        """

        removed_confs = 0
        total_confs = 0
        keep = []
        for i, mol in enumerate(self.mols):
            before = len(mol)
            more_than2left = mol.filter_confs(max_energy=max_energy, max_force=max_force, reference=reference)
            keep.append(more_than2left)
            after = len(mol)
            removed_confs += before - after
            total_confs += after

        # use slicing to modify the list inplace:
        self.mols[:] = [mol for i, mol in enumerate(self.mols) if keep[i]]
        if self.info:
            print(f"Removed {removed_confs} confs due to filtering high energy confs. {total_confs} are left.")
            if len(keep)-sum(keep) > 0:
                print("Removed {len(keep)-sum(keep)} mols")


    @classmethod
    def from_hdf5(
        cls,
        path: Union[str,Path],
        element_key: str = "atomic_numbers",
        energy_key: str = "dft_total_energy",
        xyz_key: str = "conformations",
        grad_key: str = "dft_total_gradient",
        hdf5_distance: unit = None,
        hdf5_energy: unit = None,
        hdf5_force: unit = None,
        n_max:int=None,
        skip_errs:bool=True,
        info:bool=True,
        randomize:bool=False,
        with_smiles:bool=False,
        seed:int=0,):
        """
        Generates a dataset from an hdf5 file.
        """

        PARTICLE = mole.create_unit(6.02214076e23 ** -1, "particle", "particle")
        HARTREE_PER_PARTICLE = hartree / PARTICLE
        SPICE_DISTANCE = bohr
        SPICE_ENERGY = HARTREE_PER_PARTICLE
        SPICE_FORCE = SPICE_ENERGY / SPICE_DISTANCE

        if hdf5_distance is None:
            hdf5_distance = SPICE_DISTANCE
        if hdf5_energy is None:
            hdf5_energy = SPICE_ENERGY
        if hdf5_force is None:
            hdf5_force = SPICE_FORCE

        obj = cls()
        obj.info = info
        counter = 0
        failed_counter = 0
        if info:
            print("loading dataset from hdf5 file...") 
        with h5py.File(path, "r") as f:
            it = f.keys()
            if randomize:
                import random
                # set random seed
                random.seed(seed)
                it = list(it)
                random.shuffle(it)
            for name in it:
                if not n_max is None:
                    if len(obj) >= n_max:
                        break
                try:
                    elements = f[name][element_key]
                    energies = f[name][energy_key]
                    xyz = f[name][xyz_key]
                    grads = f[name][grad_key]
                    elements = np.array(elements, dtype=np.int64)
                    xyz = Quantity(np.array(xyz), hdf5_distance).value_in_unit(DISTANCE_UNIT)
                    grads = Quantity(np.array(grads), hdf5_force).value_in_unit(FORCE_UNIT)
                    energies = Quantity(np.array(energies) - np.array(energies).mean(axis=-1), hdf5_energy).value_in_unit(ENERGY_UNIT)

                    smiles = None
                    if with_smiles:
                        smiles = f[name]["smiles"][0]

                    mol = PDBMolecule.from_xyz(elements=elements, xyz=xyz, energies=energies, gradients=grads, smiles=smiles)
                    mol.name = name.upper()
                    obj.append(mol)
                    counter += 1
                except KeyboardInterrupt:
                    raise
                except:
                    failed_counter += 1
                    if not skip_errs:
                        raise
                if info:
                    print(f"stored {counter}, failed for {failed_counter}, storing {str(name)[:8]} ...", end="\r")

        if info:
            print()
        return obj


    @classmethod
    def from_spice(cls, path: Union[str,Path], info:bool=True, n_max:int=None, randomize:bool=False, skip_errs:bool=True, with_smiles:bool=False, seed:int=0):
        """
        Generates a dataset from an hdf5 file with spice unit convention.
        """
        PARTICLE = mole.create_unit(6.02214076e23 ** -1, "particle", "particle")
        HARTREE_PER_PARTICLE = hartree / PARTICLE
        SPICE_DISTANCE = bohr
        SPICE_ENERGY = HARTREE_PER_PARTICLE
        SPICE_FORCE = SPICE_ENERGY / SPICE_DISTANCE

        obj = cls.from_hdf5(
            path,
            element_key="atomic_numbers",
            energy_key="dft_total_energy",
            xyz_key="conformations",
            grad_key="dft_total_gradient",
            hdf5_distance=SPICE_DISTANCE,
            hdf5_energy=SPICE_ENERGY,
            hdf5_force=SPICE_FORCE,
            info=info,
            n_max=n_max,
            randomize=randomize,
            skip_errs=skip_errs,
            with_smiles=with_smiles,
            seed=seed,
        )

        if not with_smiles:
            obj.remove_names_spice()
            
        return obj

    
    def remove_names(self, patterns:List[str], upper=False)->None:
        """
        Removes the molecules with names where one of the patterns occurs.
        If upper is True, this is case-insensitive.
        """

        keep = []

        for i, mol in enumerate(self.mols):
            valid = True
            if not mol.name is None:
                for pattern in patterns:
                    name = mol.name
                    if upper:
                        pattern = pattern.upper()
                        name = name.upper()

                    if pattern in name:
                        valid = False
                        break

            keep.append(valid)

        # use slicing to modify the list inplace:
        self.mols[:] = [mol for i, mol in enumerate(self.mols) if keep[i]]
        print(f"removed {len(keep)-sum(keep)} mols during remove_names.")
        

    def remove_names_spice(self)->None:

        # the names of all residues that occur:
        resnames = list(set([r for mol in self.mols for r in mol.name.split("-")]))
        resnames = [r.upper() for r in resnames]

        import json
        with open(Path(__file__).parent/ Path("xyz2res/scripts/hashed_residues.json"), "r") as f:
            # hard code the added since it isnt in the json file but can still be mapped, see xyz2res
            ALLOWED_RESIDUES = set(list(json.load(f).values()) + ["ILE", "LEU"])


        # delete hid and hip since we map everything to hie:
        ALLOWED_RESIDUES = ALLOWED_RESIDUES - set(["HID", "HIP"])

        # subtract the allowed residues:
        remove_res = list(set(resnames) - set(ALLOWED_RESIDUES))
        if self.info:
            print(f"filtering for known residues...\nfound residues:{resnames}\nallowed are {ALLOWED_RESIDUES},\nremoving residues:{remove_res}")
        self.remove_names(remove_res)

        names = []
        doubles = []
        for mol in self.mols:
            seq_from_name = mol.name.upper()
            seq_from_name = "ACE-"+seq_from_name+"-NME"
            if seq_from_name != mol.sequence:
                if self.info:
                    print(f"WARNING: seq from name {seq_from_name} != seq from xyz2res {mol.sequence}")
            name = mol.name
            if name in names:
                doubles.append(name)
            else:
                names.append(name)
        if len(doubles) > 0:
            if self.info:
                print(f"WARNING: found double names {doubles}")


    @classmethod
    def from_pdbs(cls, path: Union[str,Path], energy_name="energies.npy", force_name="forces.npy", xyz_name="positions.npy", is_force:bool=True, info:bool=True, n_max:int=None, skip_errs:bool=False, allow_incomplete:bool=False):
        """
        Generates a dataset from a folder with pdb files and .npy files containing energies and forces.
        """

        mols = []

        # loop over all pdb files (also nested in subfolders):
        for pdbpath in Path(path).rglob('*.pdb'):
            # get the parent of the pdb folder. if it contains only the one pdbfile and a file named energy_name, we assume it is a valid folder:
            if len(list(pdbpath.parent.glob('*.pdb'))) == 1 and (pdbpath.parent / energy_name).is_file():
                try:
                    energies = np.load(pdbpath.parent / energy_name) # load the energies.npy file
                    gradients = np.load(pdbpath.parent / force_name) # load the forces.npy file
                    xyz = np.load(pdbpath.parent / xyz_name) # load the positions.npy file

                    if xyz.shape[1] != gradients.shape[1]:
                        raise ValueError(f"number of atoms per mol {xyz.shape[1]} != number of force vectors per mol {gradients.shape[1]}")
                    if xyz.shape[2] != 3:
                        raise ValueError(f"number of dimensions per mol {xyz.shape[2]} != 3")
                    if gradients.shape[2] != 3:
                        raise ValueError(f"number of dimensions per force {gradients.shape[2]} != 3")
                    if gradients.shape[0] != energies.shape[0]:
                        raise ValueError(f"number of energies {energies.shape[0]} != number of forces {gradients.shape[0]}")

                    if is_force:
                        gradients = -gradients
                    
                    if len(energies) != len(xyz):
                        if allow_incomplete:
                            xyz = xyz[:len(energies)]
                        else:
                            raise ValueError(f"number of energies {len(energies)} != number of xyz {len(xyz)}")

                    mol = PDBMolecule.from_pdb(pdbpath=pdbpath, xyz=xyz, gradients=gradients, energies=energies)
                    mols.append(mol)
                    
                    if not n_max is None:
                        if len(mols) >= n_max:
                            break
                
                except:
                    if not skip_errs:
                        raise
                    
        if info:
            print(f"loaded {len(mols)} molecules with {sum([len(m) for m in mols])} confs from {path}")

        return cls(mols, info=info)
        


    def energy_hist(self, filename=None, show=True):
        energies = np.array([])
        for m in self.mols:
            e = m.energies
            e -= e.min()
            energies = np.concatenate([energies, e], axis=0)
        plt.hist(energies, bins=100)
        plt.xlabel("QM Energies")
        plt.ylabel("Count")
        plt.title("QM Energies")
        plt.yscale("log")
        if not filename is None:
            plt.savefig(filename)
        if show:
            plt.show()



    def grad_hist(self, filename=None, show=True):
        grads = np.array([])
        for m in self.mols:
            e = m.gradients
            grads = np.concatenate([grads, e], axis=0)
        plt.hist(grads, bins=100)
        plt.xlabel("QM Gradients")
        plt.ylabel("Count")
        plt.title("QM Gradients")
        plt.yscale("log")
        if not filename is None:
            plt.savefig(filename)
        if show:
            plt.show()


    def compare_with_espaloma(self, tag:str="latest"):

        def lambd(i, mol):
            try:
                if self.info:
                    print(f"applying espaloma to {i}...", end="\r")
                mol.compare_with_espaloma(tag=tag)
            except:
                if self.info:
                    print("\nFailed to compare with espaloma")
            return mol

        self.mols[:] = [lambd(i, mol) for i, mol in enumerate(self.mols) ]


    def calc_ff_data(self, forcefield, suffix="", collagen:bool=False, allow_radicals:bool=False, remove_errs=False):
        """
        Calculates the energies and forces for the given forcefield and stores them in the molecules.
        """

        def _write_data(mol, idx, forcefield, collagen, suffix):
            # determine whether smiles or not:
            if len(mol.pdb) == 1:
                raise ValueError(f"Cannot calculate energies and forces for molecules given by a smiles string: Will have to use openff, this takes too long. Smiles is {mol.pdb[0]}")
            try:
                if self.info:
                    print(f"calculating energies and forces for {idx+1}/{len(self.mols)}...", end="\r")

                if allow_radicals and not isinstance(forcefield, GrappaFF):
                    forcefield = find_radical.add_radical_residues(forcefield, topology=mol.to_openmm(collagen=collagen).topology)

                energies, grads = mol.get_ff_data(forcefield, collagen=collagen)

                mol.graph_data["g"][f"u{suffix}"] = energies
                mol.graph_data["n1"][f"grad{suffix}"] = grads.transpose(1,0,2)
            except:
                if remove_errs:
                    return None
            
            return mol
        

        self.mols[:] = [_write_data(mol, idx, forcefield, collagen, suffix) for idx, mol in enumerate(self.mols) if not _write_data(mol, idx, forcefield, collagen, suffix) is None]



    def get_eval_data(self, suffix, by_element, by_residue, radicals:bool=False):


        def se(a, b):
            return np.sum((a-b)**2)

        def rmse(a, b):
            return np.sqrt(np.mean((a-b)**2))

        def mae(a, b):
            return np.mean(np.abs(a-b))

        qm_energies = []
        ff_energies = []

        qm_grads = []
        ff_grads = []

        e_rmse_per_mol = []
        grad_rmse_per_mol = []

        se_per_element = {}
        se_per_residue = {}

        rmse_per_element = {}
        rmse_per_residue = {}

        n_forces_res = {}
        n_forces_elem = {}

        e_key = f"u{suffix}"
        f_key = f"grad{suffix}"
        for mol in self.mols:
            if "g" in mol.graph_data.keys():
                if e_key in mol.graph_data["g"].keys():

                    en_ff = mol.graph_data["g"][e_key].flatten()
                    en_mol = mol.energies

                    ff_energies.append(en_ff-en_ff.mean())
                    qm_energies.append(en_mol-en_mol.mean())
                    
                    e_rmse_per_mol.append(rmse(en_ff-en_ff.mean(), en_mol-en_mol.mean()))

            if "n1" in mol.graph_data.keys():
                if f_key in mol.graph_data["n1"].keys():

                    # shape: atoms*confs*3
                    grad_qm = mol.gradients.transpose(1,0,2)
                    grad_ff = mol.graph_data["n1"][f_key]

                    grad_rmse_per_mol.append(rmse(grad_ff, grad_qm))

                    ff_grads.append(grad_ff.flatten())
                    qm_grads.append(grad_qm.flatten())

                    # now per element:
                    radical_indices = []
                    if radicals:
                        radical_indices,_,_ = find_radical.get_radicals(topology=mol.to_openmm(collagen=True).topology)
                    if by_element:
                        for i, element in enumerate(mol.elements):
                            if i in radical_indices:
                                element = -1

                            if not element in se_per_element.keys():
                                se_per_element[element] = 0
                                n_forces_elem[element] = 0
                            se_per_element[element] += se(grad_ff[i], grad_qm[i])
                            n_forces_elem[element] += np.prod(grad_ff[i].shape)
                        

                    # now per residue:
                    if by_residue:
                        if not mol.residues is None:
                            for i, res in enumerate(mol.residues):
                                if not res in se_per_residue.keys():
                                    se_per_residue[res] = 0
                                    n_forces_res[res] = 0
                                se_per_residue[res] += se(grad_ff[i], grad_qm[i])
                                n_forces_res[res] += np.prod(grad_ff[i].shape)


        # done with the loop, now process:

        if by_element:

            from grappa.run.eval_utils import ELEMENTS
            # sort the elements by their number:
            for element in sorted(se_per_element.keys()):
                elem_str = ELEMENTS[element]
                rmse_per_element[elem_str] = np.sqrt(se_per_element[element]/n_forces_elem[element])

        if by_residue:

            for res in se_per_residue.keys():
                rmse_per_residue[res] = np.sqrt(se_per_residue[res]/n_forces_res[res])

        del se_per_element
        del se_per_residue


        ff_energies = np.concatenate(ff_energies, axis=0)
        qm_energies = np.concatenate(qm_energies, axis=0)

        ff_grads = np.concatenate(ff_grads, axis=0).flatten()
        qm_grads = np.concatenate(qm_grads, axis=0).flatten()

        eval_data = {"energy_rmse":rmse(ff_energies, qm_energies), "energy_mae":mae(ff_energies, qm_energies), "grad_rmse":rmse(ff_grads, qm_grads), "grad_mae":mae(ff_grads, qm_grads)}

        rmse_data = {}
        if by_element:
            rmse_data["rmse_per_element"] = rmse_per_element
        if by_residue:
            rmse_data["rmse_per_residue"] = rmse_per_residue
        
        # convert to float:
        for key, value in eval_data.items():
            eval_data[key] = float(value)
        
        for key, value in rmse_data.items():
            for key2, value2 in value.items():
                rmse_data[key][key2] = float(value2)


        return eval_data, rmse_data, e_rmse_per_mol, grad_rmse_per_mol, ff_energies, qm_energies, ff_grads, qm_grads, rmse_per_element, rmse_per_residue




    def evaluate(self, suffix:str="", plotpath:Union[str, Path]=None, by_element=True, by_residue=True, scatter=True, name="Forcefield", compare_suffix=None, fontsize=16, compare_name="reference", radicals:bool=False, refname="QM")->Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculates the RMSE and MAE of the data stored with suffix and returns a dictionary containing these. If a plotpath is given, also creates a scatterplot of the energies and forces, a histogram for the grad rmse per molecule and a histogram for the rmse per element/residue.
        """

        def rmse(a,b):
            return np.sqrt(np.mean((a-b)**2))

        eval_data, rmse_data, e_rmse_per_mol, grad_rmse_per_mol, ff_energies, qm_energies, ff_grads, qm_grads, rmse_per_element, rmse_per_residue = self.get_eval_data(suffix=suffix, by_element=by_element, by_residue=by_residue, radicals=radicals)

        if not compare_suffix is None:
            _,_ , e_rmse_per_mol_compare, grad_rmse_per_mol_compare, _,_,_,_, rmse_per_element_compare, rmse_per_residue_compare = self.get_eval_data(suffix=compare_suffix, by_element=by_element, by_residue=by_residue, radicals=radicals)

        if not plotpath is None:
            plotpath = Path(plotpath)
            if not plotpath.exists():
                plotpath.mkdir(parents=True)
            if scatter:

                # now create the plots:
                ff_title = name

                fig, ax = plt.subplots(1,2, figsize=(10,5))

                if len(qm_energies)>1e3:
                    ax[0].scatter(qm_energies, ff_energies, s=1, alpha=0.4)
                else:
                    ax[0].scatter(qm_energies, ff_energies)
                ax[0].plot(qm_energies, qm_energies, color="black", linewidth=0.8)
                ax[0].set_title("Energies [kcal/mol]]", fontsize=fontsize)
                ax[0].set_xlabel(f"{refname} Energies", fontsize=fontsize)
                ax[0].set_ylabel(f"{ff_title} Energies", fontsize=fontsize)
                ax[0].tick_params(axis='both', which='major', labelsize=fontsize-2)

                # now the gradients:
                if len(qm_grads)>1e3:
                    ax[1].scatter(qm_grads, ff_grads, s=0.5, alpha=0.2)
                else:
                    ax[1].scatter(qm_grads, ff_grads, s=1, alpha=0.4)
                ax[1].plot(qm_grads, qm_grads, color="black", linewidth=0.8)
                ax[1].set_title("Forces [kcal/mol/Å]", fontsize=fontsize)
                ax[1].set_xlabel(f"{refname} Forces", fontsize=fontsize)
                ax[1].set_ylabel(f"{ff_title} Gradients", fontsize=fontsize)
                ax[1].tick_params(axis='both', which='major', labelsize=fontsize-2)


                ax[0].text(0.05, 0.95, f"RMSE: {eval_data['energy_rmse']:.2f} kcal/mol", transform=ax[0].transAxes, fontsize=fontsize-2, verticalalignment='top')


                ax[1].text(0.05, 0.95, f"RMSE: {rmse(ff_grads, qm_grads):.2f} kcal/mol/Å", transform=ax[1].transAxes, fontsize=fontsize-2, verticalalignment='top')


                plt.tight_layout()
                        
                # save the figure:
                plt.savefig(plotpath / Path("scatter.png"), dpi=300)
                plt.close()

            # now plot the rmse per molecule histogram:
            fig, ax = plt.subplots(1,2, figsize=(10,5))
            ax[0].hist(e_rmse_per_mol, bins=10)
            ax[0].set_xlabel("RMSE [kcal/mol]", fontsize=fontsize)
            ax[0].set_ylabel("Count", fontsize=fontsize)
            ax[0].set_title("Energy RMSE per Molecule", fontsize=fontsize)
            ax[0].tick_params(axis='both', which='major', labelsize=fontsize-2)

            
            ax[1].hist(grad_rmse_per_mol, bins=10)
            ax[1].set_xlabel("RMSE [kcal/mol/Å]", fontsize=fontsize)
            ax[1].set_ylabel("Count", fontsize=fontsize)
            ax[1].set_title("Force RMSE per Molecule", fontsize=fontsize)
            ax[1].tick_params(axis='both', which='major', labelsize=fontsize-2)
            plt.tight_layout()
            plt.savefig(plotpath / Path("rmse_per_mol.png"), dpi=300)
            plt.close()


            if by_element or by_residue:
                if compare_suffix is None:
                    def make_elem_plot(ax):
                        # create a barplot:
                        ax.bar(rmse_per_element.keys(), rmse_per_element.values())
                        ax.set_ylabel("RMSE [kcal/mol/Å]", fontsize=fontsize)
                        ax.set_xlabel("Element", fontsize=fontsize)
                        ax.set_title("Force RMSE per Element", fontsize=fontsize)
                        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
                        return ax
                    
                    def make_res_plot(ax):
                        # sort residues to ensure 'b' and 'z' appear at the very right side
                        residues = sorted(rmse_per_residue.keys(), key=lambda k: k.lower() in ['ace', 'nme'])
                        values = [rmse_per_residue[res] for res in residues]
                        
                        # create a barplot:
                        ax.bar(residues, values)
                        ax.set_ylabel("RMSE [kcal/mol/Å]", fontsize=fontsize)
                        ax.set_xlabel("Residue", fontsize=fontsize)
                        ax.set_title("Force RMSE per Residue", fontsize=fontsize)
                        ax.tick_params(axis='both', which='major', labelsize=fontsize-2, rotation=45)
                        return ax
                    
                    if by_element:
                        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                        ax = make_elem_plot(ax)
                        plt.tight_layout()
                        plt.savefig(plotpath / Path("rmse_per_element.png"), dpi=300)
                        plt.close()

                    if by_residue:
                        fig, ax = plt.subplots(1, 1, figsize=(10, 5))  # Adjusted width for residue plot
                        ax = make_res_plot(ax)
                        plt.tight_layout()
                        plt.savefig(plotpath / Path("rmse_per_residue.png"), dpi=300)
                        plt.close()

                else:
                    assert not compare_name is None

                    def make_elem_compare_plot(ax):
                        # create a grouped barplot:
                        barWidth = 0.35  # the width of the bars
                        r1 = np.arange(len(rmse_per_element))
                        r2 = [x + barWidth for x in r1]
                        
                        ax.bar(r1, rmse_per_element.values(), width=barWidth, label=name)
                        ax.bar(r2, rmse_per_element_compare.values(), width=barWidth, label=compare_name)
                        
                        ax.set_ylabel("RMSE [kcal/mol/Å]", fontsize=fontsize)
                        ax.set_xlabel("Element", fontsize=fontsize)
                        ax.set_title("Force RMSE per Element", fontsize=fontsize)
                        ax.set_xticks([r + barWidth / 2 for r in range(len(rmse_per_element))], rmse_per_element.keys())
                        ax.legend()
                        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
                        return ax

                    def make_res_compare_plot(ax):
                        residues = sorted(rmse_per_residue.keys(), key=lambda k: k.lower() in ['ace', 'nme'])
                        values = [rmse_per_residue[res] for res in residues]
                        values_compare = [rmse_per_residue_compare[res] for res in residues]

                        # create a grouped barplot:
                        barWidth = 0.35  # the width of the bars
                        r1 = np.arange(len(residues))
                        r2 = [x + barWidth for x in r1]
                        
                        ax.bar(r1, values, width=barWidth, label=name)
                        ax.bar(r2, values_compare, width=barWidth, label=compare_name)
                        
                        ax.set_ylabel("RMSE [kcal/mol/Å]", fontsize=fontsize)
                        ax.set_xlabel("Residue", fontsize=fontsize)
                        ax.set_title("Force RMSE per Residue", fontsize=fontsize)
                        ax.set_xticks([r + barWidth / 2 for r in range(len(residues))], residues)
                        ax.legend()
                        ax.tick_params(axis='both', which='major', labelsize=fontsize-2, rotation=45)
                        return ax
                    
                    if by_element:
                        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                        ax = make_elem_compare_plot(ax)
                        plt.tight_layout()
                        plt.savefig(plotpath / Path("rmse_per_element.png"), dpi=300)
                        plt.close()

                    if by_residue:
                        fig, ax = plt.subplots(1, 1, figsize=(10, 5))  # Adjusted width for residue plot
                        ax = make_res_compare_plot(ax)
                        plt.tight_layout()
                        plt.savefig(plotpath / Path("rmse_per_residue.png"), dpi=300)
                        plt.close()


            en_std = np.std(qm_energies)
            grad_std = np.std(qm_grads)

            eval_data["energy_std"] = float(en_std)
            eval_data["grad_std"] = float(grad_std)

            # save the dicts with json:
            with open(plotpath / Path("eval_data.json"), "w") as f:
                json.dump(eval_data, f, indent=4)
            with open(plotpath / Path("rmse_data.json"), "w") as f:
                json.dump(rmse_data, f, indent=4)

        return eval_data, rmse_data
        


    def residue_hist(self, plotpath:Union[str, Path], fontsize=16):
        """
        Creates a histogram of the residues that occur in the dataset excluding ACE and NME.
        """

        from collections import Counter

        # Gather all residues
        all_residues = []
        for mol in self.mols:
            if mol.sequence is not None:
                all_residues.extend(mol.sequence.split('-'))


        # Exclude ACE and NME residues
        all_residues = [res for res in all_residues if res not in ['ACE', 'NME']]
        
        # Count each residue's occurrences
        residue_counts = Counter(all_residues)

        # Sorting residues for better visualization
        residues_sorted = sorted(residue_counts.keys())
        counts_sorted = [residue_counts[res] for res in residues_sorted]

        # Plot the bar chart
        plt.figure(figsize=(10, 5))
        plt.bar(residues_sorted, counts_sorted, edgecolor='black')
        plt.title('Histogram of Residues (Excluding ACE and NME)', fontsize=fontsize)
        plt.xlabel('Residue Name', fontsize=fontsize)
        plt.ylabel('Count', fontsize=fontsize)
        plt.xticks(residues_sorted, rotation=45, fontsize=fontsize-2)
        plt.yticks(fontsize=fontsize-2)
        plt.tight_layout()

        # Save the plot to the specified path
        plt.savefig(Path(plotpath)/Path("residue_hist.png"), dpi=300)

        # Close the figure to free up memory
        plt.close()





    @staticmethod
    def create_splitted_datasets(datasets:List["PDBDataset"], split:Tuple[float, float, float], seed:int=0)->Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]], Tuple[List[str], List[str], List[str]]]:
        """
        Order is: Train, Validation, Test
        Function that splits the datasets acording to the given split ratio. If name_split is given, the split will be done such that molecules with name appearing in any split of existing_split_names are in the same split.
        Returns a list of the indices of the split datasets of shape (n_dataset, 3, n_molecules_in_split)
        To do this, the function first splits dataset with the most molecules, uses the names obtained there for the split of the second most and so on.
        """
        # first get the number of molecules in each dataset:
        n_mols = [len(dataset) for dataset in datasets]
        # now get the order of the datasets such that len(datasets[i]) <= len(datasets[i+1])
        # this is because we want to split the smaller datasets first. Otherwise, small datasets might end up entirely in the train set.
        order = np.argsort(n_mols)[::-1]

        split_indices = [[]]*len(datasets)
        
        assert len(split_indices) == len(datasets)

        split_names = None
        for i in order:
            dataset = datasets[i]
            split_idxs, split_names = dataset.split(split, seed=seed, existing_split_names=split_names)
            split_indices[i] = split_idxs

        assert type(split_names[0]==list)
        assert type(split_names[0][0]) == str, f"Split names should be a tuple of lists of strings, but are {type(split_names)}[{type(split_names[0])}][{type(split_names[0][0])}]"

        return split_indices, split_names

    @staticmethod
    def get_splitted_datasets(datasets:List["PDBDataset"], split_names:Tuple[List[str], List[str], List[str]], split=[0.8, 0.1, 0.1], seed=0)->Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]], Tuple[List[str], List[str], List[str]]]:
        """
        Order is: Train, Validation, Test
        Split the dataset according to the given split_names.
        """
        split_indices = [tuple([])]*len(datasets)
        split_names = copy.deepcopy(split_names)
        for i in range(len(datasets)):
            dataset = datasets[i]
            split_idxs, split_names = dataset.split(seed=seed, split=split, existing_split_names=split_names)
            split_indices[i] = split_idxs

        return split_indices, split_names

    @staticmethod
    def get_dgl_splits(datasets:List["PDBDataset"], split_indices:List[Tuple[np.ndarray, np.ndarray, np.ndarray]])->List[Tuple[List[dgl.DGLGraph], List[dgl.DGLGraph], List[dgl.DGLGraph]]]:
        """
        Order is: Train, Validation, Test
        Returns [(train_dgl_list, val_dgl_list, test_dgl_list) for ds in datasets]
        """
        dgl_splits = [tuple([])]*len(datasets)
        for i in range(len(datasets)):
            dataset = datasets[i]
            split_idxs = split_indices[i]

            dgl_splits[i] = tuple(dataset.to_dgl(split_idxs[loader_type]) for loader_type in range(3))

        return dgl_splits


    def split_by_names(self, split_names:Union[str, Path, Tuple[List[str], List[str], List[str]]])->Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]], Tuple[List[str], List[str], List[str]]]:
        """
        Order is: Train, Validation, Test
        Split the dataset according to the given split_names. All names that are not occuring at all are put in the test set.
        """
        if isinstance(split_names, (str, Path)):
            with open(split_names, "r") as f:
                split_names = json.load(f)

        splitter = SplittedDataset.create_with_names(datasets=[self], split_names=split_names, nographs=True)
        ds_tr, ds_vl, ds_te = (splitter.get_splitted_datasets(datasets=[self], ds_type=ds_type)[0] for ds_type in ["tr", "vl", "te"])

        return ds_tr, ds_vl, ds_te



from dgl.dataloading import GraphDataLoader
import json


class SplittedDataset:
    """
    Class that holds train, val and test sets for several sub-datasets to be able to differentiate between them.
    """
    def __init__(self) -> None:

        split_indices:List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

        split_names:Tuple[List[str], List[str], List[str]] = None

        dgl_splits:List[Tuple[List[dgl.DGLGraph], List[dgl.DGLGraph], List[dgl.DGLGraph]]] = None




    @classmethod
    def create(cls, datasets:List[PDBDataset], split:Tuple[float, float, float]=(0.8,0.1,0.1), seed:int=0):
        """
        Generates the split_indices and split_names for the given datasets.
        """

        self = cls()

        # generate the split indices:
        self.split_indices, self.split_names = PDBDataset.create_splitted_datasets(datasets, split, seed)

        # write the dgl datasets:
        self.make_dgl(datasets=datasets)

        return self


    def get_full_loaders(self, shuffle:bool=True, max_train_mols:int=None)->Tuple[GraphDataLoader, GraphDataLoader, GraphDataLoader]:
        """
        Returns the shuffled loaders for the full datasets.
        """
        if self.dgl_splits is None:
            raise ValueError("No dgl splits have been created yet.")
        
        tr_ds = [g for subset in self.dgl_splits for g in subset[0]]
        if not max_train_mols is None:
            import random
            random.seed(0)
            random.shuffle(tr_ds)
            tr_ds[:] = tr_ds[:max_train_mols]

        self.train_mols = len(tr_ds)

        val_ds = [g for subset in self.dgl_splits for g in subset[1]]
        test_ds = [g for subset in self.dgl_splits for g in subset[2]]

        tr_loader = GraphDataLoader(tr_ds, shuffle=shuffle)
        val_loader = GraphDataLoader(val_ds, shuffle=shuffle)
        test_loader = GraphDataLoader(test_ds, shuffle=shuffle)

        return tr_loader, val_loader, test_loader



    def get_loaders(self, index:int)->Tuple[GraphDataLoader, GraphDataLoader, GraphDataLoader]:
        """
        Returns the shuffled loaders for the given subdataset.
        """
        if self.dgl_splits is None:
            raise ValueError("No dgl splits have been created yet.")
        
        tr_ds = self.dgl_splits[index][0]
        val_ds = self.dgl_splits[index][1]
        test_ds = self.dgl_splits[index][2]

        tr_loader = GraphDataLoader(tr_ds, shuffle=True)
        val_loader = GraphDataLoader(val_ds, shuffle=True)
        test_loader = GraphDataLoader(test_ds, shuffle=True)

        return tr_loader, val_loader, test_loader


    def get_splitted_datasets(self, datasets:List[PDBDataset], ds_type:str="te")->List[PDBDataset]:
        """
        Returns the train datasets obtained from the internally stored split_indices.
        ds_type must be string in tr,vl,te
        """
        if self.split_indices is None:
            raise ValueError("No split indices have been created yet.")
        
        if len(self.split_indices) != len(datasets):
            raise ValueError("The number of datasets does not match the number of split indices.")

        if ds_type == "tr":
            j = 0
        elif ds_type == "vl":
            j = 1
        elif ds_type == "te":
            j = 2
        else:
            raise ValueError("ds_type must be string in tr,vl,te")

        return [datasets[i][self.split_indices[i][j]] for i in range(len(datasets))]



    def get_splitted_datasets_from_indices(self, datasets:List[PDBDataset], ds_type:str="te")->List[PDBDataset]:
        """
        Returns the train datasets obtained from the internally stored split_indices.
        ds_type must be string in tr,vl,te
        """
        if self.split_indices is None:
            raise ValueError("No split indices have been created yet.")
        
        if len(self.split_indices) != len(datasets):
            raise ValueError("The number of datasets does not match the number of split indices.")

        if ds_type == "tr":
            j = 0
        elif ds_type == "vl":
            j = 1
        elif ds_type == "te":
            j = 2
        else:
            raise ValueError("ds_type must be string in tr,vl,te")

        return [datasets[i][self.split_indices[i][j]] for i in range(len(datasets))]


    def save(self, filename):
        """
        Saves the split_indices and split_names for the given datasets in a npz/json file.
        """
        with open(filename + ".json", "w") as f:
            json.dump(list(self.split_names), f, indent=4)

        # save the split indices:
        out = {}
        for i in range(len(self.split_indices)):
            out["train_idxs_" + str(i)] = self.split_indices[i][0]
            out["val_idxs_" + str(i)] = self.split_indices[i][1]
            out["test_idxs_" + str(i)] = self.split_indices[i][2]

        np.savez(filename + ".npz", **out)



    @classmethod
    def load(cls, filename, datasets:List[PDBDataset]):
        """
        Loads the split_indices and split_names for the given datasets from a npz file.
        """
        self = cls()
        # load the split indices and split names:
        with open(filename + ".json", "r") as f:
            self.split_names = tuple(json.load(f))

        # load the split indices:
        split_idxs = np.load(filename + ".npz")

        assert len(split_idxs.keys()) == 3*len(datasets), f"The number of datasets does not match the number of split indices: {len(split_idxs.keys())} != {3*len(datasets)}"

        self.split_indices = [tuple([])]*len(datasets)

        for i in range(len(datasets)):
            self.split_indices[i] = (split_idxs["train_idxs_" + str(i)], split_idxs["val_idxs_" + str(i)], split_idxs["test_idxs_" + str(i)])

        # write the dgl datasets:
        self.make_dgl(datasets)

        return self


    @classmethod
    def create_with_names(cls, datasets:List[PDBDataset], split_names:Tuple[List[str], List[str], List[str]], split=[0.8, 0.1, 0.1], seed=0, nographs=False):
        """
        Generates the split_indices and split_names for the given datasets, contrained by the names in split_names.
        """

        self = cls()

        # generate the split indices:
        self.split_indices, self.split_names = PDBDataset.get_splitted_datasets(datasets, split_names, split=split, seed=seed)

        # write the dgl datasets:
        if not nographs:
            self.make_dgl(datasets)

        return self


    @classmethod
    def load_from_names(cls, filename:str, datasets:List[PDBDataset], split=[0.8, 0.1, 0.1], seed=0, nographs=False):
        """
        Generates the split_indices and split_names for the given datasets, contrained by the names in split_names stored at filename.
        """

        # load the split indices and split names:
        with open(filename + ".json", "r") as f:
            split_names = tuple(json.load(f))
        
        return SplittedDataset.create_with_names(datasets, split_names, split=split, seed=seed, nographs=nographs)



    def make_dgl(self, datasets:List[PDBDataset]):
        """
        Creates the dgl splits from the split_indices.
        """
        if self.split_indices is None:
            raise ValueError("No split indices have been created yet.")
        
        self.dgl_splits = PDBDataset.get_dgl_splits(datasets, self.split_indices)