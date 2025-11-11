from dataclasses import dataclass, field
from typing import Union, Mapping, Optional
from pathlib import Path
import numpy as np

from ase.io import read
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.units import kcal, mol, Angstrom

from grappa.data.parameters import Parameters
from grappa.data import Molecule, MolData, clear_tag
from grappa.utils.openmm_utils import get_nonbonded_contribution
from grappa.utils.system_utils import openmm_system_from_gmx_top, openmm_system_from_dict
from grappa.utils.graph_utils import get_isomorphic_permutation, get_isomorphisms
import importlib.util
from openmm import System
from openmm.app import Topology, ForceField, PDBFile
import logging
import warnings

def match_molecules(molecules: list[Molecule], verbose = False, _topology_matching=False) -> dict[int,list[int]]:
    """
    Each conformation is represented by an own Molecule instance, but possibly with different atom order. We check whether the corresponding graphs are all isomorphic, that is whether ase recognized the bonds correctly for all conformations.
    We match all molceular graphs against the last Molecule in molecules. Then we calculate for each conformation the permutation of the atoms such that the atomic numbers match the last molecule. (We pick the last as it is probably the most stable one if the input is from an optimization trajectory.)
    Returns a dictionary of {idx: permutation} such that molecules[idx] with reordered atoms corresponds to the first molecule. This means that e.g. for atomic numbers: molecules[idx].atomic_number[permutation[idx]] = molecules[0].atomic_number

    _topology_matching: Internal flag that determines the warnings printed
    """

    permutations = {}
    if len(molecules) == 1:
        return permutations
    
    # first check whether number of atoms and bonds is the same for all molecules, print warning otherwise (more informative than the isomorphism warning)
    if not _topology_matching:
        num_atoms, num_bonds = len(molecules[0].atoms), len(molecules[0].bonds)
        for idx, mol in enumerate(molecules):
            if len(mol.atoms) != num_atoms:
                logging.warning(f"Number of atoms in molecular graph {idx} (num atoms:{len(mol.atoms)}) is not the same as in molecular graph 0 (num atoms:{num_atoms})!\nThe following atoms are present in 0 but not in {idx}: {set(molecules[0].atoms) - set(mol.atoms)}\nThe following atoms are present in {idx} but not in 0: {set(mol.atoms) - set(molecules[-1].atoms)}")

            if len(mol.bonds) != num_bonds:
                logging.warning(f"Number of bonds in molecular graph {idx} (num bonds:{len(mol.bonds)}) is not the same as in molecular graph 0 (num bonds:{num_bonds})! This is most likely due to the state being too far away from equilibrium such that ASE does not recognize a bond where it should be.\nThe following bonds are present in 0 but not in {idx}: {set(molecules[0].bonds) - set(mol.bonds)}\nThe following bonds are present in {idx} but not in 0: {set(mol.bonds) - set(molecules[0].bonds)}")
    else:
        assert len(molecules) == 2, f"Internal error, bug! _topology_matching should only be used for two molecules. Got {len(molecules)}"
        atoms_topology = molecules[0].atoms
        atoms_ase = molecules[1].atoms

        bonds_topology = molecules[0].bonds
        bonds_ase = molecules[1].bonds

        if len(atoms_topology) != len(atoms_ase):
            logging.warning(f"Number of atoms in molecule derived from the gromacs topology ({len(atoms_topology)}) is not the same as in the molecule constructed from ASE ({len(atoms_ase)})!\nThe following atoms are present in the topology but not in the ASE molecule: {set(atoms_topology) - set(atoms_ase)}\nThe following atoms are present in the ASE molecule but not in the topology: {set(atoms_ase) - set(atoms_topology)}")
        if len(bonds_topology) != len(bonds_ase):
            logging.warning(f"Number of bonds in molecule derived from the gromacs topology ({len(bonds_topology)}) is not the same as in the molecule constructed from ASE ({len(bonds_ase)})!\nThe following bonds are present in the topology but not in the ASE molecule: {set(bonds_topology) - set(bonds_ase)}\nThe following bonds are present in the ASE molecule but not in the topology: {set(bonds_ase) - set(bonds_topology)}")

    # convert to dgl graphs
    graphs = [mol.to_dgl() for mol in molecules]

    isomorphisms = get_isomorphisms([graphs[0]], graphs, silent=True)
    matched_idxs = [idxs[1] for idxs in list(isomorphisms)]
    if len(matched_idxs) < len(molecules):
        if not _topology_matching:
            logging.info(f"Couldn't match all molecular graphs of the {len(molecules)} states to the last graph, only {matched_idxs}!")
    if verbose:
        logging.info(isomorphisms)

    for isomorphism in list(isomorphisms):
        idx1, idx2 = isomorphism
        assert idx1 == 0, f'Internal error, bug! idx1 should be 0 as we only have a lost with one graph above!'
        permutation = get_isomorphic_permutation(graphs[0], graphs[idx2])
        permutations[idx2] = permutation

    if verbose:
        logging.info(permutations)
    return permutations

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
    >>> data_dir = Path("example-data-in")
    >>> db = DatasetBuilder()
    >>> for filename in data_dir.glob("*.out"):
    >>>     db.entry_from_qm_output_file(mol_id=filename.stem,qm_output_files=[filename])
    >>> for filename in data_dir.glob("*.top")
    >>>     db.add_nonbonded_from_gmx_top(mol_id=filename.stem,top_file=filename, add_pdb=True)
    >>> db.remove_bonded_parameters()
    >>> db.filter_bad_nonbonded()
    >>> db.write_to_dir("example-data-out", overwrite=True, delete_dgl=True)
    """
    entries: dict[str,MolData] = field(default_factory=dict)
    complete_entries: set[str] = field(default_factory=set)

    skip_impropers: bool = False # can be used for debugging. skips the reading of impropers from the openmm system.

    @staticmethod
    def _get_ref_from_gmx_topology(top_file: Union[str,Path]):
        """Add a GROMACS topology file as reference for connectivity checking during entry addition.
        """
        top_path = Path(top_file)
        system, topology = openmm_system_from_gmx_top(top_path)
        _reference_topology_molecule = Molecule.from_openmm_topology(topology, partial_charges=[0.0 for _ in range(len(list(topology.atoms())))], impropers=[])
        return _reference_topology_molecule

    @staticmethod
    def _get_ref_from_openmm_topology(topology: Topology):
        """Add an OpenMM system and topology as reference for connectivity checking during entry addition.
        """
        _reference_topology_molecule = Molecule.from_openmm_topology(topology, partial_charges=[0.0 for _ in range(len(list(topology.atoms())))], impropers=[])
        return _reference_topology_molecule

    @ staticmethod
    def _get_ref_topology(reference_topology: Optional[Union[str,Path,Topology]]):
        """Get reference topology molecule from either a gromacs topology file or an openmm Topology object.
        """
        try:
            _reference_topology_molecule = None
            if reference_topology is not None:
                if isinstance(reference_topology, (str,Path)):
                    _reference_topology_molecule = DatasetBuilder._get_ref_from_gmx_topology(reference_topology)
                elif isinstance(reference_topology, Topology):
                    _reference_topology_molecule = DatasetBuilder._get_ref_from_openmm_topology(reference_topology)
                else:
                    raise ValueError(f"reference_topology must be either a path to a GROMACS topology file or an OpenMM Topology object. Got {type(reference_topology)}")
        except Exception as e:
            raise ValueError(f"Could not read reference topology from {reference_topology}. Set the reference_topology to None if you don't want to use a reference topology for connectivity checking.") from e
        return _reference_topology_molecule

    ### Dataset creation ###
    @classmethod
    def from_moldata(cls, moldata_dir: Union[str,Path], mol_id_from_file_stem:bool = False):
        """Create DatasetBuilder object from a directory with MolData files. Can read previously written datasets for manipulation or extension.
        """
        moldata_dir = Path(moldata_dir)
        entries = {}
        complete_entries = set()
        npzs = list(moldata_dir.glob('*npz'))
        for npz in sorted(npzs):
            moldata = MolData.load(npz)
            if mol_id_from_file_stem:
                moldata.mol_id = npz.stem

            entries.update({moldata.mol_id:moldata})
            complete_entries.add(moldata.mol_id)
        return cls(entries=entries,complete_entries=complete_entries)

    def entry_from_qm_dict_file(self, mol_id:str, filename: Union[str,Path], reference_topology: Optional[Union[str,Path,Topology]] = None) -> None:
        """ Creates a DatasetBuilder entry from specified QM data array files. Assuming units to be default grappa units.
        
        Parameters
        ----------
        mol_id 
            The identifier for the molecule entry.
        filename
            File name of the QM data dictionary. Must include keys
            'xyz', 'energy', 'atomic_numbers', and either 'force' or 'gradient'.
        reference_topology
            Optional reference topology for connectivity checking during entry addition. Can be a path to a GROMACS topology file or an OpenMM Topology object.

        Returns
        -------
        None
            This method updates the internal `entries` dictionary with the new entry.
        """

        ## load qm dict and create entry
        qm_dict = np.load(filename)
        self._entry_from_qm_dict(qm_dict=qm_dict, mol_id=mol_id, mol=None, overwrite=False, validate=True, reference_topology=None)

    def entry_from_qm_array_files(self, mol_id: str, filename_dict:dict[str,str]) -> None:
        """
        Creates a DatasetBuilder entry from specified QM data array files. Assuming units to be default grappa units.

        Parameters
        ----------
        mol_id
            The identifier for the molecule entry.
        filename_dict 
            A dictionary mapping the keys 'energy', 'force'/'gradient', 'xyz', and 'atomic_numbers' to filenames. 

        Returns
        -------
        None
            This method updates the internal `entries` dictionary with the new entry.
        """

        ## create QM dict
        qm_dict = {}
        for k,v in filename_dict.items():
            qm_dict[k] = np.load(v)

        ## add entry
        self._entry_from_qm_dict(mol_id=mol_id, qm_dict=qm_dict, mol=None, overwrite=False, validate=True)

    def entry_from_qm_output_file(self, mol_id:str, qm_output_files:list[Union[Path,str]],  ase_index_str: str = ':', reference_topology: Optional[Union[str,Path,Topology]] = None) -> None:
        """
            Using ASE, processes QM output files to extract geometries, energies, and gradients. From this, a DatasetBuilder entry is created.
            Since ASE uses eV and Angstrom as internal units, the energies and gradients are converted to the Grappa units kcal/mol and A, respectively.

            Parameters
            ----------
            mol_id 
                Unique identifier for the molecule.
            qm_output_files : list of Union[Path, str]
                List of paths or strings representing the QM output files to be processed.
            ase_index_str
                The ASE index string to specify the atomic index to be used when reading the QM output files.
            reference_topology
                Optional reference topology for connectivity checking during entry addition. Can be a path to a GROMACS topology file or an OpenMM Topology object.

            Returns
            -------
            None
                This method updates the internal `entries` dictionary with the new entry.
        """

        ## read geometries
        QM_calculations: list[list[Atoms]] = [] # list where each entry is a set of conformations of the given molecule (in case there are several qm output files)
        for qm_output_file in qm_output_files:
            QM_calculations.append(read(qm_output_file,index=ase_index_str))
        
        ## create Grappa Molecules for each qm file.
        # the order of atoms can be different for each qm file, but the same within the qm file.
        molecules = []
        for idx, conformations in enumerate(QM_calculations):
            # create a mol to get the atom order:
            mol = Molecule.from_ase(conformations[-1], verbose=True)
            # NOTE: taking [-1] could be better than [0] for optimizations, could also check for conf with min energy
            self._connectivity_check(mol, reference_topology, mol_id, idx, molecules)
            molecules.append(mol)
                    
        ## different QM files could have different atom order, matching this  
        if len(molecules) > 1:  
            logging.info("Matching atom order in QM files")     
            permutations = match_molecules(molecules, verbose=False)     
        else:
            permutations = {0:list(range(QM_calculations[0][0].get_positions().shape[0]))}                      

        ## merge conformations
        qm_dict = {'xyz':[],'energy':[],'gradient':[]}
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
                    logging.warning(f"Caught the exception: {e}")
            qm_dict['xyz'].extend(np.asarray(xyz))
            qm_dict['energy'].extend(np.asarray(energy))
            qm_dict['gradient'].extend(np.asarray(gradient)) 

        if len(qm_dict['energy']) == 0:
            logging.warning(f"No QM data available for {mol_id}")
            return

        ## convert to array
        for k in qm_dict.keys():
            qm_dict[k] = np.asarray(qm_dict[k]).squeeze()

        ## add entry
        self._entry_from_qm_dict(mol_id=mol_id, qm_dict=qm_dict, mol=molecules[0], overwrite=False, validate=False)

    def _entry_from_qm_dict(self, mol_id: str, qm_dict: Mapping,  mol: Union[Molecule,None] = None, overwrite:bool = False, validate:bool = True, reference_topology: Optional[Union[str,Path,Topology]] = None) -> None:
        """
        Creates or updates an entry in the DatasetBuilder from the provided quantum mechanical data dictionary.

        Parameters
        ----------
        mol_id 
            The identifier for the molecule entry.
        qm_dict
            A dictionary-like object containing the quantum mechanical data. Must include keys 'xyz', 
            'energy', 'atomic_numbers', and either 'force' or 'gradient'.
        mol
            Grappa Molecule. If None, creates it via ase Atoms from positions and atomic numbers.
        overwrite 
            If True, will overwrite an existing entry with the same mol_id.
        validate
            If True, validates the qm_dict before processing.
        reference_topology
            Optional reference topology for connectivity checking during entry addition. Can be a path to a GROMACS topology file or an OpenMM Topology object.

        Returns
        -------
        None
            This method updates the internal `entries` dictionary with the new entry.
        """

        ## check whether entry exists ##
        if mol_id in self.entries.keys():
            if overwrite is False:
                logging.warning(f"Entry {mol_id} already exists in DatasetBuilder. Will not create a new entry!")
                return
            else:
                logging.info(f"Entry {mol_id} already exists in DatasetBuilder. Overwriting it with new entry!")

        ## skip invalid entry ##
        if validate:
            if not self._validate_qm_dict(qm_dict,mol_id):
                return
            
        ## create MolData object ##
        valid_idxs = np.isfinite(qm_dict['energy'])
        valid_idxs = np.where(valid_idxs)[0]

        energy = qm_dict['energy'][valid_idxs]
        xyz =  qm_dict['xyz'][valid_idxs]
        if 'force' in qm_dict:
            gradient = -qm_dict['force'][valid_idxs]
        else:
            gradient = qm_dict['gradient'][valid_idxs]

        if mol is None:
            # use ase to get bonds from positions
            atoms = Atoms(positions=xyz[-1],numbers=qm_dict['atomic_numbers'])
            mol = Molecule.from_ase(atoms, verbose=True)
        
        # perform the check whether the bonds match the reference topology
        if reference_topology is not None:
            self._compare_with_reference_topology(mol=mol, reference_mol=DatasetBuilder._get_ref_topology(reference_topology), context=mol_id)

        mol_data = MolData(molecule=mol,xyz=xyz,energy=energy,gradient=gradient,mol_id=mol_id)

        self.entries[mol_id] = mol_data
        return
    
    def _connectivity_check(self, mol: Molecule, reference_topology: Optional[Union[str,Path,Topology]], mol_id: str, idx: int, molecules: list) -> None:
            # check that the set of bonds fits:
            perform_connectivity_check = True
            if reference_topology is not None:
                reference_mol = DatasetBuilder._get_ref_topology(reference_topology)
            elif idx > 0:
                reference_mol = molecules[0]
            else:
                perform_connectivity_check = False
            if perform_connectivity_check:
                self._compare_with_reference_topology(mol=mol, reference_mol=reference_mol, context=f"{mol_id}/qm_file_{idx}")

    def _validate_qm_dict(self, qm_dict:Mapping, mold_id: str) -> bool:
        """Validate that QM dict has the required keys and enough conformations (>3) to be included in a dataset.
        """

        valid = True
        # check qm dict keys
        if not all([x in qm_dict for x in ['xyz','energy','atomic_numbers']]):
            logging.warning(f"The QM dictionary for '{mold_id}' must contain xyz, energy and atomic_numbers but only has following keys: {list(qm_dict.keys())}. Skipping entry!")
            valid = False
        if ('force' in qm_dict) == ('gradient' in qm_dict):
            logging.warning(f"The QM dictionary must contain either 'force' or 'gradient', but not both or none: {list(qm_dict.keys())}. Skipping entry!")
            valid = False
        # check number of valid conformations
        valid_idxs = np.isfinite(qm_dict['energy'])
        valid_idxs = np.where(valid_idxs)[0]
        if len(valid_idxs) < 3:
            logging.warning(f"Too few conformations: {len(valid_idxs)} < 3. Skipping entry!")
            valid = False
        return valid

    def _compare_with_reference_topology(self, mol: Molecule, reference_mol:Molecule, context: str) -> None:
        """
        Compare the ASE-derived molecule bonds against the optional reference topology.
        """
        if mol is None or reference_mol is None:
            return

        topo_atoms = len(reference_mol.atoms)
        mol_atoms = len(mol.atoms)
        if topo_atoms != mol_atoms:
            raise ValueError(
                f"Atom count mismatch between topology ({topo_atoms}) and ASE molecule ({mol_atoms}) for {context}."
            )

        def _normalized_bonds(bonds):
            normalized = set()
            for bond in bonds:
                a, b = bond
                normalized.add((a, b) if a <= b else (b, a))
            return normalized

        ref_bonds = _normalized_bonds(reference_mol.bonds)
        mol_bonds = _normalized_bonds(mol.bonds)
        if ref_bonds != mol_bonds:
            missing = ref_bonds - mol_bonds
            additional = mol_bonds - ref_bonds
            logging.warning(
                f"Bond mismatch for {context}: missing {missing if missing else 'none'}, "
                f"additional {additional if additional else 'none'}."
            )

    def add_nonbonded_from_gmx_top(self, mol_id:str, top_file: Union[str,Path], add_pdb:bool=False):
        """
        Add nonbonded data from a GROMACS topology to entry. Replaces the molecule of an existing entry with a GROMACS topology molecule and permutes molecular data (xyz and forces).
        Optionally, generates a PDB representation of the molecule and adds nonbonded energy.

        Parameters
        ----------
        mol_id 
            Unique identifier for the molecule entry.
        top_file 
            Path to the GROMACS topology file used to replace the molecule.
        add_pdb 
            If True, a PDB file will be generated from the GROMACS topology and added to the entry.

        Returns
        -------
        None
            This method modifies the entry in place and does not return a value.
        """
        if add_pdb:
            import io
            from openmm.app import PDBFile
        
        if not mol_id in self.entries.keys():   
            logging.warning(f"Entry {mol_id} not in DatasetBuilder entries. Skipping!")
            return

        ## create molecule
        system, topology = openmm_system_from_gmx_top(top_file)

        if add_pdb:
            buffer = io.StringIO()
            PDBFile.writeFile(topology,positions=self.entries[mol_id].xyz[-1,:,:],file=buffer)
            self.entries[mol_id].pdb = buffer.getvalue()

        self.add_nonbonded(mol_id,system,topology)


    def add_nonbonded(self, mol_id:str, system: System, topology: Topology):
        """	
        Add nonbonded data from an OpenMM system to entry. Replaces the molecule of an existing entry with an OpenMM topology and permutes molecular data (xyz and forces) such that the atomic numbers match the topology if necessary. The id in the topology must correspond to the index in the system, which is zero-based. You might need to shift your topology ids to start at zero.

        Parameters
        ----------
        mol_id 
            Unique identifier for the molecule entry.
        system
            OpenMM system object containing force field parameters to calculate nonbonded energy, forces and improper positions.
        topology
            OpenMM topology object used to replace the molecule.

        Returns
        -------
        None
            This method modifies the entry in place and does not return a value.
        """
        if not mol_id in self.entries.keys():   
            logging.warning(f"Entry {mol_id} not in DatasetBuilder entries. Skipping!")
            return
        
        ## create molecule
        mol = Molecule.from_openmm_system(system, topology, skip_impropers=self.skip_impropers, verbose=True)

        # if the atomic numbers match but the number of bonds is different, assume the order is fine and ASE merely didnt identify the right bonds. print a warning then.
        if mol.atomic_numbers == self.entries[mol_id].molecule.atomic_numbers and len(mol.bonds) != len(self.entries[mol_id].molecule.bonds):
            bonds_ase = self.entries[mol_id].molecule.bonds
            bonds_top = mol.bonds
            logging.warning(f"Number of bonds in molecule derived from the gromacs topology ({len(bonds_top)}) is not the same as in the molecule constructed from ASE ({len(bonds_ase)})!\nThe following bonds are present in the ASE molecule but not in the topology: {set(bonds_ase) - set(bonds_top)}\nThe following bonds are present in the topology but not in the ASE molecule: {set(bonds_top) - set(bonds_ase)}\nIgnoring this for now and picking the topology from the gromacs force field. But this indicates either broken states (atoms that are too close/far) or topologies that are not read properly. Check whether the gromacs-topology-bonds metioned above are what you intend.")

        # it might be that the atoms are permuted, then resolve this by checking for isomorphism and calculating an isomorphic permutation:
        elif mol.atomic_numbers != self.entries[mol_id].molecule.atomic_numbers or mol.bonds != self.entries[mol_id].molecule.bonds:
            logging.info(f"Atomic numbers of QM data and force field topology doesn't match! Matching by graph isomorphism.")
            permutations = match_molecules([mol,self.entries[mol_id].molecule], _topology_matching=True)
            if len(permutations) != 2:
                # logging.warning(f"Couldn't match QM-derived Molecule to gmx top Molecule for {mol_id}.Skipping!")
                raise ValueError(f"Couldn't match QM-derived Molecule (via ASE) to molecular graph derived from the gromacs topology for {mol_id}. There is something wrong with the topology or bonds are not recognized by ASE! See warnings above.")
                # return
            # replace data
            permutation = permutations[1]
            self.entries[mol_id].molecule = mol
            self.entries[mol_id].xyz = self.entries[mol_id].xyz[:,permutation]
            self.entries[mol_id].gradient = self.entries[mol_id].gradient[:,permutation]
        else:
            self.entries[mol_id].molecule = mol

        ## add nonbonded energy
        # energy, force = get_nonbonded_contribution(system,self.entries[mol_id].xyz)
        self.entries[mol_id].add_ff_data(system,xyz=self.entries[mol_id].xyz)
        self.entries[mol_id]._validate()

        self.complete_entries.add(mol_id)


    def add_nonbonded_from_pdb(self, mol_id:str, pdb_file:Path, forcefield:ForceField=ForceField('amber99sbildn.xml')):
        """
        """
        if not mol_id in self.entries.keys():   
            logging.warning(f"Entry {mol_id} not in DatasetBuilder entries. Skipping!")
            return

        ## create system from pdb:
        topology = PDBFile(str(pdb_file)).getTopology()
        system = forcefield.createSystem(topology)

        self.add_nonbonded(mol_id,system,topology)

        if pdb_file is not None:
            with open(pdb_file,'r') as f:
                self.entries[mol_id].pdb = f.read()
                assert isinstance(self.entries[mol_id].pdb, str)


    ### Dataset Manipulation ###
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
            energy_mm = entry.ff_energy['reference_ff']['nonbonded'] - np.mean(entry.ff_energy['reference_ff']['nonbonded'])
            energy_qm = entry.energy - np.mean(entry.energy)

            gradient_norm_mm = np.average(np.linalg.norm(entry.ff_gradient['reference_ff']['nonbonded'],axis=2),axis=1)
            gradient_norm_qm = np.average(np.linalg.norm(entry.gradient,axis=2),axis=1)

            EPS = 1e-6

            bad_idxs = []
            for j in range(len(energy_qm)):
                if (np.abs(energy_mm[j]) / np.abs((energy_qm[j]+EPS)) > 2 and energy_mm[j] > 10 ) or (np.abs(gradient_norm_mm[j])/(np.abs(gradient_norm_qm[j])+EPS) > 2 and gradient_norm_mm[j] > 10):
                    logging.info(f"bad MM energy for {mol_id} conformation {j}")
                    bad_idxs.append(j)
            if len(bad_idxs) > 0:
                logging.info(f" Removing conformations {bad_idxs}")
                try:
                    entry.delete_states(bad_idxs)
                    count_remove['conformations'] += len(bad_idxs)
                except ValueError as e:
                    logging.warning(e)
                    logging.info(f"Removing dataset entry {mol_id}")
                    rmv_entries.append(mol_id)

        logging.info(f"Removing entries: {rmv_entries}")
        for rmv_entry in rmv_entries:
            self.entries.pop(rmv_entry,None)
            self.complete_entries.remove(rmv_entry)
        count_remove['entries'] = len(rmv_entries)
        logging.info(f"Removed {count_remove['entries']} molecule entries and {count_remove['conformations']} conformations.")


    ### Dataset Writing ###
    def write_to_dir(self, dataset_dir: Union[str,Path], overwrite:bool=False, delete_dgl:bool=False):
        """ """
        dataset_dir = Path(dataset_dir)
        dataset_dir.mkdir(parents=True,exist_ok=True)
        npzs_existing = list(dataset_dir.glob('*npz'))
        if len(npzs_existing) > 0:
            logging.info(f"{len(npzs_existing)} npz files already in output directory!")
            if not overwrite:
                logging.warning(f"Not writing dataset because npz files are already in directory!")
                return
            else:
                logging.info('Removing .npz files in dataset directory')
                for file in dataset_dir.glob('*.npz'):
                    file.unlink()
        mol_ids = self.entries.keys()
        output_entries = []
        for entry_idx in self.complete_entries:
            if entry_idx in mol_ids:
                output_entries.append(entry_idx)
        logging.info(f"Writing {len(output_entries)} complete entries out of {len(self.entries)} total entries in the DatasetBuilder.")
        for output_entry_idx in output_entries:
            self.entries[output_entry_idx].save(dataset_dir / f"{output_entry_idx}.npz")
        if delete_dgl:
            logging.info(f"Clearing dgl dataset with tag: {dataset_dir.name}")
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
