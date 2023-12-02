"""
Contains the grappa input dataclass 'MolData', which is an extension of the dataclass 'Molecule' that contains conformational data and characterizations like smiles string or PDB file.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
from grappa.data import Molecule, Parameters
from grappa.utils import openff_utils, openmm_utils
import torch
from dgl import DGLGraph
from grappa import constants

import pkgutil


@dataclass
class MolData():
    """
    Dataclass for entries in datasets on which grappa can be trained.
    """
    molecule: Molecule
    classical_parameters: Parameters # these are used for regularisation and to estimate the statistics of the reference energy and gradient

    # conformational data:
    xyz: np.ndarray
    energy: np.ndarray
    gradient: np.ndarray

    # reference values (centered bonded energy, bonded gradient)
    energy_ref: np.ndarray
    gradient_ref: np.ndarray

    mol_id: str

    # mol_id is either a smiles string or a sequence string that is also stored
    sequence: Optional[str] = None
    smiles: Optional[str] = None

    # improper contributions of the reference forcefield:
    improper_energy_ref: Optional[np.ndarray] = None
    improper_gradient_ref: Optional[np.ndarray] = None
    
    # additional characterizations:
    mapped_smiles: Optional[str] = None
    pdb: Optional[str] = None

    # nonbonded contributions:
    ff_nonbonded_energy: Dict[str, np.ndarray] = None
    ff_nonbonded_gradient: Dict[str, np.ndarray] = None

    # classical forcefield energies: dictionaries mapping a force field name to an array of energies
    ff_energy: Dict[str, np.ndarray] = None
    ff_gradient: Dict[str, np.ndarray] = None


    def _validate(self):
        for k,v in self.ff_energy.items():
            assert v.shape == self.energy.shape, f"Shape of ff_energy {k} does not match energy: {v.shape} vs {self.energy.shape}"
        for k,v in self.ff_gradient.items():
            assert v.shape == self.gradient.shape, f"Shape of ff_gradient {k} does not match gradient: {v.shape} vs {self.gradient.shape}"


    def  __post_init__(self):

        # setting {} by default is not possible in dataclasses, thus do this here:
        if self.ff_energy is None:
            self.ff_energy = dict()
        if self.ff_gradient is None:
            self.ff_gradient = dict()
        if self.ff_nonbonded_energy is None:
            self.ff_nonbonded_energy = dict()
        if self.ff_nonbonded_gradient is None:
            self.ff_nonbonded_gradient = dict()

        if not "qm" in self.ff_energy.keys():
            self.ff_energy["qm"] = self.energy
        if not "qm" in self.ff_gradient.keys():
            self.ff_gradient["qm"] = self.gradient

        self._validate()
    
    
    def to_dgl(self, max_element=constants.MAX_ELEMENT, exclude_feats:list[str]=[])->DGLGraph:
        """
        Converts the molecule to a dgl graph with node features. The elements are one-hot encoded.
        Also creates entries 'xyz', 'energy_ref' and 'gradient_ref' in the global node type g and in the atom node type n1, which contain the reference energy and gradient, respectively. The shapes are different than in the class attributes, namely (1, n_confs) and (n_atoms, n_confs, 3) respectively. (This is done because feature tensors must have len == num_nodes)

        The dgl graph has the following node types:
            - g: global
            - n1: atoms
            - n2: bonds
            - n3: angles
            - n4: propers
            - n4_improper: impropers
        The node type n1 carries the feature 'ids', which are the identifiers in self.atoms. The other interaction levels (n{>1}) carry the idxs (not ids) of the atoms as ordered in self.atoms as feature 'idxs'. These are not the identifiers but must be translated back to the identifiers using ids = self.atoms[idxs] after the forward pass.
        This also stores classical parameters except for improper torsions.
        """
        g = self.molecule.to_dgl(max_element=max_element, exclude_feats=exclude_feats)
        
        # write reference energy and gradient in the shape (1, n_confs) and (n_atoms, n_confs, 3) respectively
        g.nodes['g'].data['energy_ref'] = torch.tensor(self.energy_ref.reshape(1, -1), dtype=torch.float32)
        
        g.nodes['g'].data['energy_ref'] -= g.nodes['g'].data['energy_ref'].mean(dim=1)

        g.nodes['n1'].data['gradient_ref'] = torch.tensor(self.gradient_ref.transpose(1, 0, 2), dtype=torch.float32)

        if not self.improper_energy_ref is None:
            g.nodes['g'].data['improper_energy_ref'] = torch.tensor(self.improper_energy_ref.reshape(1, -1), dtype=torch.float32)
            g.nodes['g'].data['improper_energy_ref'] -= g.nodes['g'].data['improper_energy_ref'].mean(dim=1)

        if not self.improper_gradient_ref is None:
            g.nodes['n1'].data['improper_gradient_ref'] = torch.tensor(self.improper_gradient_ref.transpose(1, 0, 2), dtype=torch.float32)

        for k, v in self.ff_energy.items():
            g.nodes['g'].data[f'energy_{k}'] = torch.tensor(v.reshape(1, -1), dtype=torch.float32)
        
        for k, v in self.ff_gradient.items():
            g.nodes['n1'].data[f'gradient_{k}'] = torch.tensor(v.transpose(1, 0, 2), dtype=torch.float32)

        # write positions in shape (n_atoms, n_confs, 3)
        g.nodes['n1'].data['xyz'] = torch.tensor(self.xyz.transpose(1, 0, 2), dtype=torch.float32)

        g = self.classical_parameters.write_to_dgl(g=g)

        return g
    

    def to_dict(self):
        """
        Save the molecule as a dictionary of arrays.
        """
        array_dict = dict()
        array_dict['xyz'] = self.xyz
        array_dict['energy'] = self.energy
        array_dict['gradient'] = self.gradient
        array_dict['energy_ref'] = self.energy_ref
        array_dict['gradient_ref'] = self.gradient_ref
        array_dict['mol_id'] = self.mol_id

        if not self.mapped_smiles is None:
            array_dict['mapped_smiles'] = self.mapped_smiles
        if not self.pdb is None:
            array_dict['pdb'] = self.pdb
        if not self.smiles is None:
            array_dict['smiles'] = self.smiles
        if not self.sequence is None:
            array_dict['sequence'] = self.sequence

        if not self.improper_energy_ref is None:
            array_dict['improper_energy_ref'] = self.improper_energy_ref
        if not self.improper_gradient_ref is None:
            array_dict['improper_gradient_ref'] = self.improper_gradient_ref

        moldict = self.molecule.to_dict()
        assert set(moldict.keys()).isdisjoint(array_dict.keys()), "Molecule and MolData have overlapping keys."
        array_dict.update(moldict)

        # remove bond, angle, proper, improper since these are stored in the molecule
        paramdict = {
            k: v for k, v in self.classical_parameters.to_dict().items() if k not in ['atoms', 'bonds', 'angles', 'propers', 'impropers']
        }

        if not set(paramdict.keys()).isdisjoint(array_dict.keys()):
            raise ValueError(f"Parameter keys and array keys overlap: {set(paramdict.keys()).intersection(array_dict.keys())}")

        array_dict.update(paramdict)

        # add force field energies and gradients
        for v, k in self.ff_energy.items():
            if f'energy_{v}' in array_dict.keys():
                raise ValueError(f"Duplicate key: energy_{v}")
            array_dict[f'energy_{v}'] = k
        for v, k in self.ff_gradient.items():
            if f'gradient_{v}' in array_dict.keys():
                raise ValueError(f"Duplicate key: gradient_{v}")
            array_dict[f'gradient_{v}'] = k
        for v, k in self.ff_nonbonded_energy.items():
            if f'nonbonded_energy_{v}' in array_dict.keys():
                raise ValueError(f"Duplicate key: nonbonded_energy_{v}")
            array_dict[f'nonbonded_energy_{v}'] = k
        for v, k in self.ff_nonbonded_gradient.items():
            if f'nonbonded_gradient_{v}' in array_dict.keys():
                raise ValueError(f"Duplicate key: nonbonded_gradient_{v}")
            array_dict[f'nonbonded_gradient_{v}'] = k

        return array_dict
    

    @classmethod
    def from_dict(cls, array_dict:Dict):
        """
        Create a Molecule from a dictionary of arrays.
        """
        xyz = array_dict.get('xyz')
        energy = array_dict.get('energy')
        gradient = array_dict.get('gradient')
        energy_ref = array_dict.get('energy_ref')
        gradient_ref = array_dict.get('gradient_ref')
        mol_id = array_dict.get('mol_id')
        if isinstance(mol_id, np.ndarray):
            mol_id = str(mol_id)

        mapped_smiles = array_dict.get('mapped_smiles', None)
        if isinstance(mapped_smiles, np.ndarray):
            mapped_smiles = str(mapped_smiles)

        pdb = array_dict.get('pdb', None)
        if isinstance(pdb, np.ndarray):
            pdb = str(pdb)

        smiles = array_dict.get('smiles', None)
        if isinstance(smiles, np.ndarray):
            smiles = str(smiles)

        sequence = array_dict.get('sequence', None)
        if isinstance(sequence, np.ndarray):
            sequence = str(sequence)


        improper_energy_ref = array_dict.get('improper_energy_ref', None)
        improper_gradient_ref = array_dict.get('improper_gradient_ref', None)

        param_keys = ['bond_k', 'bond_eq', 'angle_k', 'angle_eq', 'proper_ks', 'proper_phases', 'improper_ks', 'improper_phases']

        tuple_keys = ['atoms', 'bonds', 'angles', 'propers', 'impropers']

        exclude_molecule_keys = ['xyz', 'mol_id', 'pdb', 'mapped_smiles', 'smiles', 'sequence'] + param_keys

        # Reconstruct the molecule from the dictionary. for this, we need to filter out the keys that are not part of the molecule. We can assume that all keys are disjoint since we check this during saving.
        molecule_dict = {k: v for k, v in array_dict.items() if not k in exclude_molecule_keys and not 'energy' in k and not 'gradient' in k}

        molecule = Molecule.from_dict(molecule_dict)

        # Reconstruct the parameters, excluding keys that are part of the molecule
        param_dict = {k: array_dict[k] for k in array_dict if k in param_keys or k in tuple_keys}
        classical_parameters = Parameters.from_dict(param_dict)

        # Extract force field energies and gradients
        ff_energy = {k.split('_', 1)[1]: v for k, v in array_dict.items() if k.startswith('energy_') and not k=='energy_ref'}
        ff_gradient = {k.split('_', 1)[1]: v for k, v in array_dict.items() if k.startswith('gradient_') and not k == 'gradient_ref'}
        ff_nonbonded_energy = {k.split('_', 2)[2]: v for k, v in array_dict.items() if k.startswith('nonbonded_energy_')}
        ff_nonbonded_gradient = {k.split('_', 2)[2]: v for k, v in array_dict.items()if k.startswith('nonbonded_gradient_')}

        # Initialize a new MolData object
        return cls(
            xyz=xyz,
            energy=energy,
            gradient=gradient,
            energy_ref=energy_ref,
            gradient_ref=gradient_ref,
            mol_id=mol_id,
            molecule=molecule,
            classical_parameters=classical_parameters,
            ff_energy=ff_energy,
            ff_gradient=ff_gradient,
            ff_nonbonded_energy=ff_nonbonded_energy,
            ff_nonbonded_gradient=ff_nonbonded_gradient,
            improper_energy_ref=improper_energy_ref,
            improper_gradient_ref=improper_gradient_ref,
            mapped_smiles=mapped_smiles,
            pdb=pdb,
            smiles=smiles,
            sequence=sequence,
        )

    

    def save(self, path:str):
        """
        Save the molecule to a npz file.
        """
        np.savez(path, **self.to_dict())

    @classmethod
    def load(cls, path:str):
        """
        Load the molecule from a npz file.
        """
        array_dict = np.load(path)
        return cls.from_dict(array_dict)


    @classmethod
    def from_data_dict(cls, data_dict:Dict[str, Union[np.ndarray, str]], forcefield='openff-1.2.0.offxml', partial_charge_key:str='partial_charges'):
        """
        Create a MolData object from a dictionary containing a mapped_smiles string or pdb and arrays of conformations, energies and gradients, but not necessarily the interaction tuples and classical parameters.
        The forcefield is used to obtain the interaction tuples and classical parameters. If a smiles string is used, the forcefield refers to an openff forcefield. If a pdb file is used, the forcefield refers to an openmm forcefield.
        The following dictionray items are required:
            - Either: mapped_smiles: (str) or pdb: (str)
            - Either: smiles: (str) or sequence: (str)
            - xyz: (n_confs, n_atoms, 3)
            - energy: (n_confs,)
            - gradient: (n_confs, n_atoms, 3)
        """
        mapped_smiles = None
        if 'mapped_smiles' in data_dict.keys():
            mapped_smiles = data_dict['mapped_smiles']
            if not isinstance(mapped_smiles, str):
                mapped_smiles = mapped_smiles[0]
        pdb = None
        if 'pdb' in data_dict.keys():
            pdb = data_dict['pdb']
            if not isinstance(pdb, str):
                pdb = pdb[0]
        assert mapped_smiles is not None or pdb is not None, "Either a smiles string or a pdb file must be provided."
        assert not (mapped_smiles is not None and pdb is not None), "Either a smiles string or a pdb file must be provided, not both."

        smiles = data_dict.get('smiles', None)
        sequence = data_dict.get('sequence', None)

        mol_id = data_dict.get('smiles', data_dict.get('sequence', None))
        if mol_id is None:
            raise ValueError("Either a smiles string or a sequence string must be provided as key 'smiles' or 'sequence' in the data dictionary.")
        if isinstance(mol_id, np.ndarray):
            mol_id = mol_id[0]

        xyz = data_dict['xyz']
        energy = data_dict['energy_qm']
        gradient = data_dict['gradient_qm']
        partial_charges = data_dict.get(partial_charge_key, None)
        energy_ref = data_dict.get('energy_ref', None)
        gradient_ref = data_dict.get('gradient_ref', None)
        

        if mapped_smiles is not None:
            self = cls.from_smiles(mapped_smiles=mapped_smiles, xyz=xyz, energy=energy, gradient=gradient, forcefield=forcefield, partial_charges=partial_charges, energy_ref=energy_ref, gradient_ref=gradient_ref, mol_id=mol_id, forcefield_type='openff', smiles=smiles)
        else:
            raise NotImplementedError("pdb files are not supported yet.")

        self.sequence = sequence

        # Extract force field energies and gradients
        self.ff_energy.update({k.split('_', 1)[1]: v for k, v in data_dict.items() if k.startswith('energy_') and not k == 'energy_ref'})
        self.ff_gradient.update({k.split('_', 1)[1]: v for k, v in data_dict.items() if k.startswith('gradient_') and not k == 'gradient_ref'})
        self.ff_nonbonded_energy.update({k.split('_', 2)[2]: v for k, v in data_dict.items() if k.startswith('nonbonded_energy_')})
        self.ff_nonbonded_gradient.update({k.split('_', 2)[2]: v for k, v in data_dict.items()if k.startswith('nonbonded_gradient_')})

        return self


    @classmethod
    def from_openmm_system(cls, openmm_system, openmm_topology, xyz, energy, gradient, mol_id:str, partial_charges=None, energy_ref=None, gradient_ref=None, mapped_smiles=None, pdb=None, ff_name:str=None, sequence:str=None, smiles:str=None):
        """
        Use an openmm system to obtain classical parameters and interaction tuples.
        If partial charges is None, the charges are obtained from the openmm system.
        If energy_ref and gradient_ref are None, they are also calculated from the openmm system.
        mapped_smiles and pdb have no effect on the system, are optional and only required for reproducibility.
        If the improper parameters are incompatible with the openmm system, the improper torsion parameters are all set to zero.
        """
        import openmm
        mol = Molecule.from_openmm_system(openmm_system=openmm_system, openmm_topology=openmm_topology, partial_charges=partial_charges)
        params = Parameters.from_openmm_system(openmm_system, mol=mol, allow_skip_improper=True)

        self = cls(molecule=mol, classical_parameters=params, xyz=xyz, energy=energy, gradient=gradient, energy_ref=energy_ref, gradient_ref=gradient_ref, mapped_smiles=mapped_smiles, pdb=pdb, mol_id=mol_id)

        if not partial_charges is None:
            # set the partial charges in the openmm system
            openmm_system = openmm_utils.set_partial_charges(system=openmm_system, partial_charges=partial_charges)

        # calculate the reference energy and gradient from the openmm system to (maybe?) include this in the training objective
        total_ref_energy, total_ref_gradient = openmm_utils.get_energies(openmm_system=openmm_system, xyz=xyz)
        total_ref_gradient = -total_ref_gradient # the reference gradient is the negative of the force

        if ff_name is None:
            ff_name = 'reference_ff'

        self.ff_energy[ff_name] = total_ref_energy
        self.ff_gradient[ff_name] = total_ref_gradient
        

        if self.energy_ref is None:
            # calculate reference energy and gradient from the openmm system using the partial charges provided

            # create a deep copy of the system:
            system2 = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(openmm_system))

            # remnove all but the nonbonded forces in this copy:
            self.energy_ref, self.gradient_ref = openmm_utils.get_energies(openmm_system=system2, xyz=xyz)
            self.gradient_ref = -self.gradient_ref # the reference gradient is the negative of the force

        # calculate the contribution from improper torsions in the system:
        # remove all forces but periodic torsions
        openmm_system = openmm_utils.remove_forces_from_system(openmm_system, keep=['PeriodicTorsionForce'])

        # get a list of sets of improper torsion tuples:
        improper_set = {tuple(sorted(t)) for t in self.molecule.impropers}

        # set all ks to zero that are not impropers:
        for force in openmm_system.getForces():
            if not isinstance(force, openmm.PeriodicTorsionForce):
                raise NotImplementedError(f"Removed all but PeriodicTorsionForce, but found a different force: {force.__class__.__name__}")
            for i in range(force.getNumTorsions()):
                atom1, atom2, atom3, atom4, periodicity, phase, k = force.getTorsionParameters(i)
                if not tuple(sorted((self.molecule.atoms[atom1], self.molecule.atoms[atom2], self.molecule.atoms[atom3], self.molecule.atoms[atom4]))) in improper_set:
                    force.setTorsionParameters(i, atom1, atom2, atom3, atom4, periodicity, phase, 0)


        # get energy and gradient. these are now only sourced from improper torsions.
        self.improper_energy_ref, self.improper_gradient_ref = openmm_utils.get_energies(openmm_system=openmm_system, xyz=xyz)
        self.improper_gradient_ref = -self.improper_gradient_ref # the reference gradient is the negative of the force

        return self
    

    @classmethod
    def from_smiles(cls, mapped_smiles, xyz, energy, gradient, partial_charges=None, energy_ref=None, gradient_ref=None, forcefield='openff_unconstrained-1.2.0.offxml', mol_id=None, forcefield_type='openff'):
        """
        Create a Molecule from a mapped smiles string and an openff forcefield. The openff_forcefield is used to initialize the interaction tuples, classical parameters and, if partial_charges is None, to obtain the partial charges.
        The forcefield_type can be either openff, openmm or openmmforcefields.
        """
        if forcefield_type == 'openff':
            system, topology, openff_mol = openff_utils.get_openmm_system(mapped_smiles, openff_forcefield=forcefield, partial_charges=partial_charges)
        
        elif forcefield_type == 'openmm':
            raise NotImplementedError("This does not work for openff molecules. The residues are needed.")

            from openmm.app import ForceField
            from openff.toolkit import Molecule as OFFMolecule

            ff = ForceField(forcefield)
            openff_mol = OFFMolecule.from_mapped_smiles(mapped_smiles, allow_undefined_stereo=True)
            topology = openff_mol.to_topology().to_openmm()
            system = ff.createSystem(topology)

        elif forcefield_type == 'openmmforcefields':
            raise NotImplementedError("openmmforcefields is not supported yet.")
        else:
            raise ValueError(f"forcefield_type must be either openff, openmm or openmmforcefields, not {forcefield_type}")

        smiles = openff_mol.to_smiles(mapped=False)

        if mol_id is None:
            mol_id = smiles
        

        self = cls.from_openmm_system(openmm_system=system, openmm_topology=topology, xyz=xyz, energy=energy, gradient=gradient, partial_charges=partial_charges, energy_ref=energy_ref, gradient_ref=gradient_ref, mapped_smiles=mapped_smiles, mol_id=mol_id, smiles=smiles)

        self.molecule.add_features(['ring_encoding', "sp_hybridization", "is_aromatic"], openff_mol=openff_mol)

        return self