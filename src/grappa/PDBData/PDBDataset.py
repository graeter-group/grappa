#%%
import numpy as np
import dgl
import torch
import tempfile
import os.path
from .PDBMolecule import PDBMolecule
from pathlib import Path
from typing import Union, List, Tuple, Dict, Any, Optional
import h5py
from grappa.units import DISTANCE_UNIT, ENERGY_UNIT, FORCE_UNIT
from openmm.unit import bohr, Quantity, hartree, mole
from openmm.unit import unit
from openmm.app import ForceField
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
        self.mols = copy.deepcopy(mols)
        self.info = info # flag whether to print information on certain operations.

    def __len__(self)->int:
        return len(self.mols)

    def __getitem__(self, idx:int)->dgl.graph:
        return self.mols[idx]
    
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
    
    def to_dgl(self, idxs:List[int]=None, split:List[float]=None, seed:int=0, forcefield:ForceField=ForceField('amber99sbildn.xml'), allow_radicals=False, collagen=False)->List[dgl.DGLGraph]:
        """
        Converts a list of indices to a list of dgl graphs.
        """
        if idxs is None:
            idxs = list(range(len(self)))

        if self.info:
            print(f"converting PDBDataset to {len(idxs)} dgl graphs...")

        def _get_graph(i, idx):
            if self.info:
                print(f"converting {i+1}/{len(idxs)}", end="\r")
            return self.mols[idx].to_dgl(graph_data=True, classical_ff=forcefield, allow_radicals=allow_radicals, collagen=collagen)
        
        if self.info:
            print("\n")
        
        glist = [_get_graph(i, idx) for i, idx in enumerate(idxs)]

        if split is None:
            return glist
        else:
            return dgl.data.utils.split_dataset(glist, split, shuffle=True, random_state=seed)
        
    
    def save(path:Union[str, Path]):
        """
        Saves the dataset to an hdf5 file.
        """
        pass

    def save_npz(self, path:Union[str, Path], overwrite:bool=False)->None:
        """
        Save the dataset to npz files.
        """
        if self.info:
            print(f"saving PDBDataset of length {len(self)} to npz files...")
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
    def load_npz(cls, path:Union[str, Path], keep_order=False, n_max=None):
        """
        Loads a dataset from npz files.
        """
        obj = cls()
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

    def save_dgl(self, path:Union[str, Path], idxs:List[int]=None, overwrite:bool=False, forcefield:ForceField=ForceField('amber99sbildn.xml'), allow_radicals=False, collagen=False)->None:
        """
        Saves the dgl graphs that belong to the dataset.
        """
        if len(self) == 0:
            raise ValueError("cannot save empty dataset")

        if os.path.exists(str(path)):
            if not overwrite:
                raise FileExistsError(f"path {str(path)} already exists, set overwrite=True to overwrite it.")
        
        os.makedirs(str(Path(path).parent), exist_ok=True)

        dgl.save_graphs(path, self.to_dgl(idxs, forcefield=forcefield, allow_radicals=allow_radicals, collagen=collagen))

        if idxs is None:
            idxs = list(range(len(self)))

        seqpath = str(Path(path).parent/Path(path).stem) + "_seq.json"
        with open (seqpath, "w") as f:
            json.dump([self.mols[i].sequence for i in idxs], f)


    
    def parametrize(self, forcefield:ForceField=ForceField('amber99sbildn.xml'), get_charges=None, allow_radicals=False, collagen=False)->None:
        """
        Parametrizes the dataset with a forcefield.
        Writes the following entries to the graph:
        ...
        get_charges: a function that takes a topology and returns a list of charges as openmm Quantities in the order of the atoms in topology.
        if not openffmol is None, get_charge can also take an openffmolecule instead.
        """

        if self.info:
            print("parametrizing PDBDataset...")
        for i, mol in enumerate(self.mols):
            if self.info:
                print(f"parametrizing {i+1}/{len(self.mols)}", end="\r")
            try:
                mol.parametrize(forcefield=forcefield, get_charges=get_charges, allow_radicals=allow_radicals, collagen=collagen)
            except Exception as e:
                raise type(e)(str(e) + f" in molecule {mol.sequence}")
        if self.info:
            print()

    
    def filter_validity(self, forcefield:ForceField=ForceField("amber99sbildn.xml"), sigmas:Tuple[float,float]=(1.,1.))->None:
        """
        Checks if the stored energies and gradients are consistent with the forcefield. Removes the entry from the dataset if the energies and gradients are not within var_param[0] and var_param[1] standard deviations of the stored energies/forces. If sigmas are 1, this corresponds to demanding that the forcefield data is better than simply always guessing the mean.
        """
        if self.info:
            print("filtering valid mols of PDBDataset by comparing with class ff...")
        keep = []
        removed = 0
        kept = 0
        for i, mol in enumerate(self.mols):
            valid = mol.conf_check(forcefield=forcefield, sigmas=sigmas)
            keep.append(valid)
            removed += int(not valid)
            kept += int(valid)
            if self.info:
                print(f"filtering {i+1}/{len(self.mols)}, kept {kept}, removed {removed}", end="\r")
        if self.info:
            print()

        # use slicing to modify the list inplace:
        self.mols[:] = [mol for i, mol in enumerate(self.mols) if keep[i]]
        

        
    def filter_confs(self, max_energy:float=60., max_force:float=200, reference=False)->None:
        """
        Filters out conformations with energies or forces that are over 60 kcal/mol away from the minimum of the dataset (not the actual minimum). Remove molecules is less than 2 conformations are left. Apply this before parametrizing or re-apply the parametrization after filtering. Units are kcal/mol and kcal/mol/angstrom.
        """

        keep = []
        for i, mol in enumerate(self.mols):
            more_than2left = mol.filter_confs(max_energy=max_energy, max_force=max_force, reference=reference)
            keep.append(more_than2left)

        # use slicing to modify the list inplace:
        self.mols[:] = [mol for i, mol in enumerate(self.mols) if keep[i]]
        if self.info:
            print(f"removed {len(keep)-sum(keep)} mols after filtering out high energy confs")


    @classmethod
    def from_hdf5(
        cls,
        path: Union[str,Path],
        element_key: str = "atomic_numbers",
        energy_key: str = "dft_total_energy",
        xyz_key: str = "conformations",
        grad_key: str = "dft_total_gradient",
        hdf5_distance: unit = DISTANCE_UNIT,
        hdf5_energy: unit = ENERGY_UNIT,
        hdf5_force: unit = FORCE_UNIT,
        n_max:int=None,
        skip_errs:bool=True,
        info:bool=True,
        randomize:bool=False,):
        """
        Generates a dataset from an hdf5 file.
        """
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
                it = list(it)
                random.shuffle(it)
            for name in it:
                if not n_max is None:
                    if len(obj) > n_max:
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

                    mol = PDBMolecule.from_xyz(elements=elements, xyz=xyz, energies=energies, gradients=grads)
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
    def from_spice(cls, path: Union[str,Path], info:bool=True, n_max:int=None, randomize:bool=False, skip_errs:bool=True):
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
        )

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


# %%
