from dataclasses import dataclass, field
from typing import Union
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

#%%

@dataclass
class DatasetBuilder:
    entries: dict[str,MolData] = field(default_factory=dict)
    complete_entries: set[str] = field(default_factory=set)

    @classmethod
    def from_moldata(cls, moldata_dir: Path):
        entries = {}
        complete_entries = set()
        npzs = list(moldata_dir.glob('*npz'))
        for npz in sorted(npzs):
            print(npz.name)
            moldata = MolData.load(npz)
            entries.update({moldata.mol_id:moldata})
            complete_entries.add(moldata.mol_id)
        return cls(entries=entries,complete_entries=complete_entries)

    @classmethod
    def from_QM_arrays(cls, qm_data_dir: Path, verbose:bool = False):
        """ Expects nested QM data dir. One molecule per directory. One array per input type. Assuming units to be default grappa units
        """
        entries = {}
        subdirs =  list(qm_data_dir.iterdir())
        for subdir in sorted(subdirs):
            mol_id = subdir.name 
            print(mol_id)
            energy = np.load(subdir/"psi4_energies.npy")
            gradient = -np.load(subdir/"psi4_forces.npy")

            xyz = np.load(subdir/"positions.npy")
            atomic_numbers = np.load(subdir/"atomic_numbers.npy") # not needed

            valid_idxs = np.isfinite(energy)
            valid_idxs = np.where(valid_idxs)[0]
            energy = energy[valid_idxs]
            gradient = gradient[valid_idxs]
            xyz = xyz[valid_idxs]

            # use ase to get bonds from positions
            atoms = Atoms(positions=xyz[-1],numbers=atomic_numbers)
            mol = Molecule.from_ase(atoms)

            mol_data = MolData(molecule=mol,xyz=xyz,energy=energy,gradient=gradient,mol_id=mol_id)
            entries[mol_id] = mol_data
        return cls(entries=entries)

    @classmethod
    def from_QM_ase(cls, qm_data_dir: Path, parse_all_configurations: bool = True, verbose:bool = False):
        """ Expects nested QM data dir. One molecule per directory."""
        entries = {}
        subdirs =  list(qm_data_dir.iterdir())
        index_str = ':' if parse_all_configurations else None   # could extend this to the pass index-str itself

        for subdir in sorted(subdirs):
            mol_id = subdir.name 
            print(mol_id)
            QM_calculations = []
            gaussian_files = list(subdir.glob(f"*.log")) + list(subdir.glob('*.out'))

            # create geometries: list[list[Atoms]]
            for file in gaussian_files:
                QM_calculations.append(read(file,index=index_str))
            
            # create molecules
            molecules = []
            for conformations in QM_calculations:
                molecules.append(Molecule.from_ase(conformations[-1]))  #taking [-1] could be better than [0] for optimizations
                        
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

    def add_nonbonded_from_gmx_top(self, top_data_dir: Path, add_pdb:bool=False):
        """Replaces molecule of entry with gmx top molecule and permutates moldata xyz and forces
        """
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
