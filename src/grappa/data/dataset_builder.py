from dataclasses import dataclass, field
from typing import Union, Mapping
from pathlib import Path
import numpy as np
from tqdm import tqdm

from ase.io import read
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry.analysis import Analysis
from ase.units import kcal, mol, Angstrom

from grappa.data.parameters import Parameters
from grappa.data import Molecule, MolData, clear_tag
from grappa.utils.openmm_utils import get_nonbonded_contribution
from grappa.utils.system_utils import openmm_system_from_gmx_top, openmm_system_from_dict
from grappa.utils.graph_utils import get_isomorphic_permutation, get_isomorphisms

def match_molecules(molecules: list[Molecule], verbose = False) -> dict[int,list[int]]:
    """Match relative to first Molecule in molecules
    """

    permutations = {0: list(range(len(molecules[0].atoms)))}
    if len(molecules) == 1:
        return permutations

    graphs = [mol.to_dgl() for mol in molecules]

    isomorphisms = get_isomorphisms([graphs[0]],graphs,silent=True)
    matched_idxs = [idxs[1] for idxs in list(isomorphisms)]
    if len(matched_idxs) < len(molecules):
        print(f"Couldn't match all graphs to first graph, only {matched_idxs}!")
        # except RuntimeError as e:
        #     print('Skipping Molecule matching!')
        #     raise e
    if verbose:
        print(isomorphisms)

    for isomorphism in list(isomorphisms):
        [idx1,idx2] = isomorphism
        permutation = get_isomorphic_permutation(graphs[idx1],graphs[idx2])
        permutations[idx2] = permutation
    if verbose:
        print(permutations)
    return permutations

def moldata_from_qm_dict(qm_dict: Mapping, mol_id: str) -> MolData:
    """Qm_dict should be a dict like object. Must contain keys xyz, energy, atomic_numbers and either force or gradient
    """
    assert all([x in qm_dict for x in ['xyz','energy','atomic_numbers']]), f"The QM dictionary must contain xyz, energy and atomic_numbers but is missing keys: {list(filename_dict.keys())}"
    assert ('force' in qm_dict) != ('gradient' in qm_dict), f"The QM dictionary must contain either 'force' or 'gradient', but not both: {list(filename_dict.keys())}"

    valid_idxs = np.isfinite(qm_dict['energy'])
    valid_idxs = np.where(valid_idxs)[0]
    energy = qm_dict['energy'][valid_idxs]
    xyz =  qm_dict['xyz'][valid_idxs]
    if 'force' in qm_dict:
        gradient = -qm_dict['force'][valid_idxs]
    else:
        gradient = qm_dict['gradient'][valid_idxs]

    # use ase to get bonds from positions
    atoms = Atoms(positions=qm_dict['xyz'][-1],numbers=qm_dict['atomic_numbers'])
    mol = Molecule.from_ase(atoms)

    mol_data = MolData(molecule=mol,xyz=qm_dict['xyz'],energy=qm_dict['energy'],gradient=qm_dict['gradient'],mol_id=mol_id)
    return mol_data

@dataclass
class DatasetBuilder:
    """The DatasetBuilder converts QM data and nonbonded information (e.g. from a GROMACS topology file) to a grappa dataset.

    Attributes
    ----------
    entries
        Dictionary with mol-ids as keys and MolData values. The MolData object contains all information to build a dataset entry
    complete_entries
        Set of mol-ids with both QM data and nonbonded information. It is not validated when value is returned.

    Example
    ----------
    >>> db = DatasetBuilder.from_QM_ase("example-data-in")
    >>> db.add_nonbonded_from_gmx_top("example-data-in", add_pdb=True)
    >>> db.remove_bonded_parameters()
    >>> db.filter_bad_nonbonded()
    >>> db.write_to_dir("example-data-out", overwrite=True, delete_dgl=True)
    """
    entries: dict[str,MolData] = field(default_factory=dict)
    complete_entries: set[str] = field(default_factory=set)

    @classmethod
    def from_moldata(cls, moldata_dir: Union[str,Path], mol_id_from_file_stem:bool = False):
        moldata_dir = Path(moldata_dir)
        entries = {}
        complete_entries = set()
        npzs = list(moldata_dir.glob('*npz'))
        for npz in sorted(npzs):
            print(npz.name)
            moldata = MolData.load(npz)
            if mol_id_from_file_stem:
                moldata.mol_id = npz.stem

            entries.update({moldata.mol_id:moldata})
            complete_entries.add(moldata.mol_id)
        return cls(entries=entries,complete_entries=complete_entries)

    @classmethod
    def from_QM_dicts(cls, qm_data_dir: Union[str,Path], verbose:bool = False):
        """ Expects nested QM data dir. One molecule per directory. One array per input type. 
        Arrays of the same type should have the same filename. Assuming units to be default grappa units
        
        Attributes
        ----------
        qm_data_dir 
            Directory with subdirectories containing a single npz file with QM data. The subdirectory names will be used as mol_ids
        """
        qm_data_dir = Path(qm_data_dir)
        entries = {}
        subdirs =  list(qm_data_dir.iterdir())
        for subdir in sorted(subdirs):
            mol_id = subdir.name 
            print(mol_id)

            npz_files = list(subdir.glob('*.npz'))
            if len(npz_files) > 1:
                print(f"Multiple npz files in {subdir.name}, taking {npz_files[0]}.")
            elif len(npz_files) == 0:
                print(f"No npz file found in {subdir}. Skipping!")
                continue

            npz = np.load(npz_files[0])
            mol_data = moldata_from_qm_dict(npz, mol_id)
            if len(mol_data.energy) < 3:
                print(f"Too few conformations: {len(mol_data.energy)}. Skipping entry!")
                continue
            entries[mol_id] = mol_data

        return cls(entries=entries)

    @classmethod
    def from_QM_arrays(cls, qm_data_dir: Union[str,Path], filename_dict:Union[dict[str,str],None]= None, verbose:bool = False):
        """ Expects nested QM data dir. One molecule per directory. One array per input type. 
        Arrays of the same type should have the same filename. Assuming units to be default grappa units
        
        Attributes
        ----------
        qm_data_dir 
            Directory with subdirectories containing QM data arrays for every dataset entry
        filename_dict
            Dictionary of QM data array files. Must contain keys xyz, energy, atomic_numbers and either force or gradient

        """
        if filename_dict is None:
            filename_dict = {'energy': 'psi4_energies.npy',
                             'force': 'psi4_forces.npy',
                             'xyz': 'positions.npy',
                             'atomic_numbers': 'atomic_numbers.npy'}
        assert all([x in filename_dict for x in ['xyz','energy','atomic_numbers']]), f"The filename dictionary must contain xyz, energy and atomic_numbers but is missing keys: {list(filename_dict.keys())}"
        assert ('force' in filename_dict) != ('gradient' in filename_dict), f"The filename dictionary must contain either 'force' or 'gradient', but not both: {list(filename_dict.keys())}"

        qm_data_dir = Path(qm_data_dir)
        entries = {}
        subdirs =  list(qm_data_dir.iterdir())
        for subdir in sorted(subdirs):
            mol_id = subdir.name 
            print(mol_id)
            if not all([(subdir / fn).exists() for fn in filename_dict.values()]):
                print(f"Couldn't find all filenames in subdirectory. Skipping!")
                continue
            
            qm_dict = {}
            for k,v in filename_dict.items():
                qm_dict[k] = np.load(subdir / v)
            if 'force' in qm_dict:
                qm_dict['gradient'] = - qm_dict.pop('force')
            print(qm_dict)
            mol_data = moldata_from_qm_dict(qm_dict, mol_id)
            entries[mol_id] = mol_data
        return cls(entries=entries)

    @classmethod
    def from_QM_ase(cls, qm_data_dir: Union[str,Path], ase_index_str: str = ':', verbose:bool = False):
        """ Expects nested QM data dir. One molecule per directory."""
        qm_data_dir = Path(qm_data_dir)
        entries = {}
        subdirs =  list(qm_data_dir.iterdir())

        for subdir in sorted(subdirs):
            mol_id = subdir.name 
            print(mol_id)
            QM_calculations = []
            gaussian_files = list(subdir.glob(f"*.log")) + list(subdir.glob('*.out'))

            # create geometries: list[list[Atoms]]
            for file in gaussian_files:
                QM_calculations.append(read(file,index=ase_index_str))
            
            # create molecules
            molecules = []
            for conformations in QM_calculations:
                molecules.append(Molecule.from_ase(conformations[-1]))  #taking [-1] could be better than [0] for optimizations, could check for conf with min energy 
                        
            # different QM files could have different atom order, matching this 
            # the ase atoms object is not a good container for changing energies and forces, maybe do this when the 
            # values are more accessible    
            if len(molecules) > 1:  
                print("Matching atom order in QM files")     
                permutations = match_molecules(molecules,verbose=verbose)     
            else:
                permutations = {0:list(range(QM_calculations[0][0].get_positions().shape[0]))}                      

            # merge conformations
            QM_data = {'xyz':[],'energy':[],'gradient':[]}
            for idx,permutation in permutations.items():
                xyz = []
                energy = []
                gradient = []
                for conformation in QM_calculations[idx]:
                    try:
                        xyz_conf = conformation.get_positions()[permutation]
                        energy_conf = conformation.get_potential_energy() * mol / kcal # conversion to grappa units [kcal/mol]
                        force_conf = conformation.get_forces()[permutation] * mol / kcal *Angstrom # conversion to grappa units [kcal/mol A]
                        # append after to only add to list if all three properties exist
                        xyz.append(xyz_conf)
                        energy.append(energy_conf)
                        gradient.append(-force_conf)# - to convert from force to gradient
                    except PropertyNotImplementedError as e:
                        print(f"Caught the exception: {e}")
                QM_data['xyz'].extend(np.asarray(xyz))
                QM_data['energy'].extend(np.asarray(energy))
                QM_data['gradient'].extend(np.asarray(gradient)) 

            if len(QM_data['energy']) == 0:
                print(f"No QM data available for {mol_id}")
                continue

            # convert to array
            for k in QM_data.keys():
                QM_data[k] = np.asarray(QM_data[k]).squeeze()

            # create MolData list
            mol_data = MolData(molecule=molecules[0],xyz=QM_data['xyz'],energy=QM_data['energy'],gradient=QM_data['gradient'],mol_id=mol_id)
            entries[mol_id] = mol_data

        return cls(entries=entries)

    def add_nonbonded_from_gmx_top(self, top_data_dir: Union[str,Path], add_pdb:bool=False):
        """Replaces molecule of entry with gmx top molecule and permutates moldata xyz and forces
        """
        top_data_dir = Path(top_data_dir)
        if add_pdb:
            import io
            from openmm.app import PDBFile
        subdirs =  list(top_data_dir.iterdir())
        for subdir in sorted(subdirs):
            mol_id = subdir.name 
            print(mol_id)     
            if not mol_id in self.entries.keys():   
                print(f"Entry {mol_id} not in DatasetBuilder entries. Skipping!")
                continue
            # get top file
            try:
                top_file = sorted(list(subdir.glob(f"*.top")))[0]
            except IndexError as e:
                print(f"No GROMACS topology file in {subdir}. Skipping!")
                continue

            print(f"Parsing first found topology file: {top_file}.")
            system, topology = openmm_system_from_gmx_top(top_file)
            # create molecule and get permutation
            mol = Molecule.from_openmm_system(system,topology)

            # reorder atoms of QM data if atomic number list is different
            if mol.atomic_numbers != self.entries[mol_id].molecule.atomic_numbers:
                print(f"Atomic numbers of QM data and GROMACS topology doesn't match! Trying to match by graph.")
                permutations = match_molecules([mol,self.entries[mol_id].molecule])
                if len(permutations) != 2:
                    print(f"Couldn't match QM-derived Molecule to gmx top Molecule for {mol_id}.Skipping!")
                    continue
                # replace data
                print(permutations[1])
                print(mol.atomic_numbers)
                print(self.entries[mol_id].molecule.atomic_numbers)
                print([self.entries[mol_id].molecule.atomic_numbers[jj] for jj in permutations[1]])
                permutation = permutations[1]
                self.entries[mol_id].molecule = mol
                self.entries[mol_id].xyz = self.entries[mol_id].xyz[:,permutation]
                self.entries[mol_id].gradient = self.entries[mol_id].gradient[:,permutation]
            else:
                self.entries[mol_id].molecule = mol

            if add_pdb:
                buffer = io.StringIO()
                PDBFile.writeFile(topology,positions=self.entries[mol_id].xyz[-1,:,:],file=buffer)
                self.entries[mol_id].pdb = buffer.getvalue()
            # add nonbonded energy
            # energy, force = get_nonbonded_contribution(system,self.entries[mol_id].xyz)
            self.entries[mol_id].add_ff_data(system,xyz=self.entries[mol_id].xyz)
            self.entries[mol_id]._validate()
            self.complete_entries.add(mol_id)
    
    def remove_bonded_parameters(self):
        """Remove bonded parameters in MolData.classical_parameters and removes bonded energy/force contributions in MolData.ff_energy/force
        """
        for mol_id,moldata in self.entries.items():
            nan_prms = Parameters.get_nan_params(moldata.molecule)
            moldata.classical_parameter_dict = {'reference_ff': nan_prms}
            for contribution in ['bond','angle','proper','improper']:
                for ff_name, ff_dict in moldata.ff_energy.items():
                    ff_dict.pop(contribution,None)
                for ff_name, ff_dict in moldata.ff_gradient.items():
                    ff_dict.pop(contribution,None)

    def filter_bad_nonbonded(self):
        """ Remove conformations where the nonbonded contribution to energy and gradient is much higher than the QM energy and gradient
        """
            #filter out bad energies
        count_remove = {'entries':0,'conformations':0}
        rmv_entries = []
        for mol_id, entry in self.entries.items():
            if mol_id not in self.complete_entries:
                continue
            print(mol_id)
            energy_mm = entry.ff_energy['reference_ff']['nonbonded'] - np.mean(entry.ff_energy['reference_ff']['nonbonded'])
            energy_qm = entry.energy - np.mean(entry.energy)

            gradient_norm_mm = np.average(np.linalg.norm(entry.ff_gradient['reference_ff']['nonbonded'],axis=2),axis=1)
            gradient_norm_qm = np.average(np.linalg.norm(entry.gradient,axis=2),axis=1)

            bad_idxs = []
            for j in range(len(energy_qm)):
                if (energy_mm[j] / energy_qm[j] > 2 and energy_mm[j] > 10 ) or (gradient_norm_mm[j]/gradient_norm_qm[j] > 2 and gradient_norm_mm[j] > 10):
                    print(f"bad MM energy for {mol_id} conformation {j}")
                    print(energy_mm[j],energy_qm[j])
                    print(gradient_norm_mm[j],gradient_norm_qm[j])
                    bad_idxs.append(j)
            if len(bad_idxs) > 0:
                print(f" Removing conformations {bad_idxs}")
                print(len(entry.energy))
                try:
                    entry.delete_states(bad_idxs)
                    count_remove['conformations'] += len(bad_idxs)
                except ValueError as e:
                    print(e)
                    print(f"Removing dataset entry {mol_id}")
                    rmv_entries.append(mol_id)
                print(len(entry.energy))

        print(f"Removing entries: {rmv_entries}")
        for rmv_entry in rmv_entries:
            self.entries.pop(rmv_entry,None)
            self.complete_entries.remove(rmv_entry)
        count_remove['entries'] = len(rmv_entries)
        print(f"Removed {count_remove['entries']} molecule entries and {count_remove['conformations']} conformations.")

    def write_to_dir(self, dataset_dir: Union[str,Path], overwrite:bool=False, delete_dgl:bool=False):
        """ """
        dataset_dir = Path(dataset_dir)
        dataset_dir.mkdir(parents=True,exist_ok=True)
        npzs_existing = list(dataset_dir.glob('*npz'))
        if len(npzs_existing) > 0:
            print(f"{len(npzs_existing)} npz files already in output directory!")
            if not overwrite:
                print(f"Not writing dataset because npz files are already in directory!")
                return
            else:
                print('Removing .npz files in dataset directory')
                for file in dataset_dir.glob('*.npz'):
                    file.unlink()
        mol_ids = self.entries.keys()
        output_entries = []
        for entry_idx in self.complete_entries:
            if entry_idx in mol_ids:
                output_entries.append(entry_idx)
        print(f"Writing {len(output_entries)} complete entries out of {len(self.entries)} total entries in the DatasetBuilder.")
        for output_entry_idx in output_entries:
            self.entries[output_entry_idx].save(dataset_dir / f"{output_entry_idx}.npz")
        if delete_dgl:
            print(f"Clearing dgl dataset with tag: {dataset_dir.name}")
            clear_tag(dataset_dir.name)

    def write_pdb(self, pdb_dir: Union[str,Path]):
        """Write all pdbs of DataBuilder.entries to a pdb directory
        """
        pdb_dir = Path(pdb_dir)
        pdb_dir.mkdir(parents=True,exist_ok=True)
        for entry in self.entries.values():
            if entry.pdb is not None:
                with open(pdb_dir / f"{entry.mol_id}.pdb",'w') as f:
                    f.write(entry.pdb)



    def _validate(self):
        """ """
        raise NotImplementedError


# %%
