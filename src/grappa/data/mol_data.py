"""
Contains the grappa input dataclass 'MolData', which is an extension of the dataclass 'Molecule' that contains conformational data and characterizations like smiles string or PDB file.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
from grappa.data import Molecule, Parameters
import torch
from dgl import DGLGraph
from grappa import constants
from grappa import units as grappa_units
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory


@dataclass(init=False)
class MolData():
    """
    Dataclass for entries in datasets on which grappa can be trained. Contains a set of states of a molecule, qm energies and reference energies (qm minus nonbonded energy of some classical forcefield). Can be stored as npz files. A list of MolData objects is considered to be a 'grappa dataset'.
    Assumes the following shapes:
        - energy: (n_confs,)
        - xyz: (n_confs, n_atoms, 3)
        - gradient: (n_confs, n_atoms, 3)
    """
    molecule: Molecule # (also stores the partial charges)

    # conformational data:
    xyz: np.ndarray

    mol_id: str

    classical_parameter_dict: Dict[str, Parameters] = None # these are used for regularisation and to estimate the statistics of the reference energy and gradient

    # mol_id is either a smiles string or a sequence string that is also stored
    sequence: Optional[str] = None
    smiles: Optional[str] = None
    
    # additional characterizations:
    mapped_smiles: Optional[str] = None
    pdb: Optional[str] = None # pdb file as string. openmm topology can be recovered via openmm_utils.get_topology_from_pdb

    # classical forcefield energy contributions: dictionaries mapping a force field name to an array of bonded energies
    # this will be used in grappa.data.Dataset
    ff_energy: Dict[str, Dict[str, np.ndarray]] = None
    ff_gradient: Dict[str, Dict[str, np.ndarray]] = None

    add_data: Dict[str, np.ndarray] = None # additional data that can be anything

    def __init__(self,
                molecule:Molecule,
                xyz:np.ndarray,
                energy:np.ndarray,
                gradient:np.ndarray,
                mol_id:str,
                classical_parameters:Union[Parameters, Dict[str, Parameters]]=None,
                ff_energy:Dict[str, Dict[str, np.ndarray]]=None,
                ff_gradient:Dict[str, Dict[str, np.ndarray]]=None,
                smiles:str=None,
                sequence:str=None,
                mapped_smiles:str=None,
                pdb:str=None,
                add_data:Dict[str, np.ndarray]={}
                ):
        self.molecule = molecule
        self.xyz = xyz
        self.mol_id = mol_id
        self.classical_parameter_dict = classical_parameters if not isinstance(classical_parameters, Parameters) else {'reference_ff': classical_parameters}
        self.smiles = smiles
        self.sequence = sequence
        self.pdb = pdb
        self.mapped_smiles = mapped_smiles

        if ff_energy is None:
            ff_energy = dict()
        if ff_gradient is None:
            ff_gradient = dict()

        self.ff_energy = ff_energy
        self.ff_gradient = ff_gradient

        self.ff_energy['qm'] = {'total': energy}
        self.ff_gradient['qm'] = {'total': gradient}

        self.molecule.add_features(['ring_encoding', 'degree'])

        self.add_data = add_data

        self.__post_init__()


    # .energy and .gradient are the qm energies and gradients
    @property
    def energy(self):
        return self.ff_energy['qm']['total']
    
    @property
    def gradient(self):
        return self.ff_gradient['qm']['total']
    
    @property
    def classical_parameters(self):
        ff_keys = list(self.classical_parameter_dict.keys())
        return self.classical_parameter_dict[ff_keys[0]] if len(ff_keys) == 1 else self.classical_parameter_dict['reference_ff']
    
    @energy.setter
    def energy(self, value):
        self.ff_energy['qm']['total'] = value

    @gradient.setter
    def gradient(self, value):
        self.ff_gradient['qm']['total'] = value

    @classical_parameters.setter
    def classical_parameters(self, value):
        self.classical_parameter_dict = {'reference_ff': value}


    def _validate(self):
        # if not self.energy.shape[0] > 0:
        #     raise ValueError(f"Energy must have at least one entry, but has shape {self.energy.shape}")
        
        for k,v in self.ff_energy.items():
            for kk,vv in v.items():
                assert len(vv) == len(self.energy), f"Length of ff_energy {k} does not match energy: {len(vv)} vs {len(self.energy)}"
        for k,v in self.ff_gradient.items():
            for kk,vv in v.items():
                if not self.gradient is None:
                    assert vv.shape == self.gradient.shape, f"Shape of ff_gradient {k} does not match gradient: {vv.shape} vs {self.gradient.shape}"

        
        if self.mol_id is None or self.mol_id == 'None':
            raise Warning(f"mol_id is not provided. For training on different molecules, this is necessary.")

        # check shapes:
        if len(self.energy.shape) != 1:
            raise ValueError(f"Energy must have shape (n_confs,) but has shape {self.energy.shape}")
        if len(self.gradient.shape) != 3:
            raise ValueError(f"Gradient must have shape (n_confs, n_atoms, 3) but has shape {self.gradient.shape}")
        if self.xyz.shape[0] == 0:
            raise ValueError(f"xyz must have at least one conformation, but has shape {self.xyz.shape}")
        if self.xyz.shape[2] != 3:
            raise ValueError(f"xyz must have shape (n_confs, n_atoms, 3) but has shape {self.xyz.shape}")
        if self.xyz.shape[1] == 0:
            raise ValueError(f"xyz must have at least one atom, but has shape {self.xyz.shape}")
        if self.xyz.shape != self.gradient.shape:
            raise ValueError(f"Shape of xyz {self.xyz.shape} does not match gradient {self.gradient.shape}")
        if self.xyz.shape[0] != self.energy.shape[0]:
            raise ValueError(f"Shape of xyz {self.xyz.shape} does not match energy {self.energy.shape}")


    def  __post_init__(self):

        # setting {} by default is not possible in dataclasses, thus do this here:
        if self.ff_energy is None:
            self.ff_energy = dict()
        if self.ff_gradient is None:
            self.ff_gradient = dict()

        if not "qm" in self.ff_energy.keys():
            self.ff_energy["qm"] = {"total": self.energy}

        if not "qm" in self.ff_gradient.keys():
            self.ff_gradient["qm"] = {"total": self.gradient}

        if self.classical_parameters is None:
            # create parameters that are all nan but in the correct shape:
            self.classical_parameters = Parameters.get_nan_params(mol=self.molecule)

        self.mol_id = str(self.mol_id)

        self._validate()


    def to_dgl(self, max_element=constants.MAX_ELEMENT, exclude_feats:List[str]=[])->DGLGraph:
        """
        Converts the molecule to a dgl graph with node features. The elements are one-hot encoded.
        Also creates entries 'xyz', 'energy_qm' and 'gradient_qm' in the global node type g and in the atom node type n1 respectively. The shapes are different than in the class attributes, namely (1, n_confs) and (n_atoms, n_confs, 3) respectively. (This is done because feature tensors must have len == num_nodes)

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
        
        # write positions in shape (n_atoms, n_confs, 3)
        g.nodes['n1'].data['xyz'] = torch.tensor(self.xyz.transpose(1, 0, 2), dtype=torch.float32)

        # write all stored energy and gradient contributions in the shape (1, n_confs) and (n_atoms, n_confs, 3) respectively
        for ff_name, ff_dict in self.ff_energy.items():
            for k, v in ff_dict.items():
                if ff_name == "qm":
                    assert k == "total", "Only total qm energy is supported."
                    g.nodes['g'].data[f'energy_{ff_name}'] = torch.tensor(v.reshape(1, -1), dtype=torch.float32)
                else:
                    g.nodes['g'].data[f'energy_{ff_name}_{k}'] = torch.tensor(v.reshape(1, -1), dtype=torch.float32)

        for ff_name, ff_dict in self.ff_gradient.items():
            for k, v in ff_dict.items():
                if ff_name == "qm":
                    g.nodes['n1'].data[f'gradient_{ff_name}'] = torch.tensor(v.transpose(1, 0, 2), dtype=torch.float32)
                else:
                    g.nodes['n1'].data[f'gradient_{ff_name}_{k}'] = torch.tensor(v.transpose(1, 0, 2), dtype=torch.float32)

        g = self.classical_parameters.write_to_dgl(g=g)

        return g
    

    def to_dict(self):
        """
        Save the molecule as a dictionary of arrays.
        """
        array_dict = dict()
        array_dict['xyz'] = self.xyz
        array_dict['mol_id'] = np.array(str(self.mol_id))

        if not self.mapped_smiles is None:
            array_dict['mapped_smiles'] = np.array(str(self.mapped_smiles))
        if not self.pdb is None:
            array_dict['pdb'] = np.array(str(self.pdb))
        if not self.smiles is None:
            array_dict['smiles'] = np.array(str(self.smiles))
        if not self.sequence is None:
            array_dict['sequence'] = np.array(str(self.sequence))

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
            for kk, vv in k.items():
                if f'energy_{v}_{kk}' in array_dict.keys():
                    raise ValueError(f"Key {f'energy_{v}_{kk}'} already exists in array_dict.")
                array_dict[f'energy_{v}_{kk}'] = vv

        for v, k in self.ff_gradient.items():
            for kk, vv in k.items():
                if f'gradient_{v}_{kk}' in array_dict.keys():
                    raise ValueError(f"Key {f'gradient_{v}_{kk}'} already exists in array_dict.")
                array_dict[f'gradient_{v}_{kk}'] = vv

        array_dict.update({f'add_data_{k}': v for k, v in self.add_data.items()})

        return array_dict
    

    @classmethod
    def from_dict(cls, array_dict:Dict):
        """
        Create a Molecule from a dictionary of arrays.
        """
        xyz = array_dict['xyz']
        mol_id = array_dict['mol_id']
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
        ff_names_energy = set(['_'.join(k.split('_')[1:-1]) for k in array_dict.keys() if k.startswith('energy_')])
        ff_names_gradient = set(['_'.join(k.split('_')[1:-1]) for k in array_dict.keys() if k.startswith('gradient_')])

        ff_energy = dict()
        ff_gradient = dict()
        for ff_name in ff_names_energy:
            ff_energy[ff_name] = {k.split('_')[-1]: v for k, v in array_dict.items() if k.startswith(f'energy_{ff_name}_')}
        for ff_name in ff_names_gradient:
            ff_gradient[ff_name] = {k.split('_')[-1]: v for k, v in array_dict.items() if k.startswith(f'gradient_{ff_name}_')}

        add_data = {'_'.join(k.split('_')[2:]): v for k, v in array_dict.items() if k.startswith('add_data_')}

        # Initialize a new MolData object
        return cls(
            xyz=xyz,
            energy=ff_energy['qm']['total'],
            gradient=ff_gradient['qm']['total'],
            mol_id=mol_id,
            molecule=molecule,
            classical_parameters=classical_parameters,
            ff_energy=ff_energy,
            ff_gradient=ff_gradient,
            mapped_smiles=mapped_smiles,
            pdb=pdb,
            smiles=smiles,
            sequence=sequence,
            add_data=add_data
        )

    

    def save(self, path:Union[Path,str]):
        """
        Save the molecule to a npz file.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        d = self.to_dict()
        np.savez(path, **d)

    @classmethod
    def load(cls, path:str):
        """
        Load the molecule from a npz file.
        """
        array_dict = np.load(path)
        return cls.from_dict(array_dict)

    @classmethod
    def from_data_dict(cls, data_dict:Dict[str, Union[np.ndarray, str]], forcefield='openff-1.2.0.offxml', partial_charge_key:str='partial_charges', allow_nan_params:bool=False, charge_model:str='None'):
        """
        Constructor for loading from espaloma datasets. Assumes that energy_ref = energy_qm - energy_nonbonded.
        Create a MolData object from a dictionary containing a mapped_smiles string or pdb and arrays of conformations, energies and gradients, but not necessarily the interaction tuples and classical parameters.
        The forcefield is used to obtain the interaction tuples and classical parameters. If a smiles string is used, the forcefield refers to an openff forcefield. If a pdb file is used, the forcefield refers to an openmm forcefield.
        The following dictionray items are required:
            - Either: mapped_smiles: (str) or pdb: (str)
            - Either: smiles: (str) or sequence: (str)
            - xyz: (n_confs, n_atoms, 3)
            - energy: (n_confs,)
            - gradient: (n_confs, n_atoms, 3)
        ----------
        Parameters:
            - data_dict: Dict[str, Union[np.ndarray, str]]
            - forcefield: str
            - partial_charge_key: str
            - allow_nan_params: bool: If True, the parameters are set to nans if they cannot be obtained from the forcefield. If False, an error is raised.
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

        mol_id = data_dict.get('mol_id', data_dict.get('smiles', data_dict.get('sequence', None)))
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
            self = cls.from_smiles(mapped_smiles=mapped_smiles, xyz=xyz, energy=energy, gradient=gradient, forcefield=forcefield, partial_charges=partial_charges, mol_id=mol_id, forcefield_type='openff', smiles=smiles, allow_nan_params=allow_nan_params, charge_model=charge_model)
        else:
            raise NotImplementedError("mapped smiles is missing. construction from pdb files is not supported yet.")

        self.sequence = sequence

        # Extract force field energies and gradients
        ff_energies = {k.split('_', 1)[1]: v for k, v in data_dict.items() if k.startswith('energy_') and k != 'energy_ref'}
        ff_gradients = {k.split('_', 1)[1]: v for k, v in data_dict.items() if k.startswith('gradient_') and k != 'gradient_ref'}
        ff_nonbonded_energy = {k.split('_', 2)[2]: v for k, v in data_dict.items() if k.startswith('nonbonded_energy_')}
        ff_nonbonded_gradient = {k.split('_', 2)[2]: v for k, v in data_dict.items() if k.startswith('nonbonded_gradient_')}

        # assume that the nonbonded energy of the reference ff is the difference between espalomas _qm and _ref contributions:
        self.ff_energy['reference_ff'] = {'nonbonded': energy - energy_ref}
        self.ff_gradient['reference_ff'] = {'nonbonded': gradient - gradient_ref}

        for ff_name in ff_energies.keys():
            if ff_name not in self.ff_energy.keys():
                self.ff_energy[ff_name] = {"total": ff_energies[ff_name]}
                if ff_name in ff_nonbonded_energy.keys():
                    self.ff_energy[ff_name]["nonbonded"] = ff_nonbonded_energy[ff_name]

        for ff_name in ff_gradients.keys():
            if ff_name not in self.ff_gradient.keys():
                self.ff_gradient[ff_name] = {"total": ff_gradients[ff_name]}
                if ff_name in ff_nonbonded_gradient.keys():
                    self.ff_gradient[ff_name]["nonbonded"] = ff_nonbonded_gradient[ff_name]

        return self


    @classmethod
    def from_openmm_system(cls, openmm_system, openmm_topology, xyz, energy, gradient, mol_id:str, partial_charges=None, mapped_smiles=None, pdb=None, ff_name:str=None, sequence:str=None, smiles:str=None, allow_nan_params:bool=True,charge_model:str="None", skip_ff:bool=False):
        """
        Use an openmm system to obtain classical contributions, classical parameters and the interaction tuples. Calculates the contributions:
            - total: total energy and gradient
            - nonbonded: nonbonded energy and gradient
            - bond: bond energy and gradient
            - angle: angle energy and gradient
            - proper: proper torsion energy and gradient
            - improper: improper torsion energy and gradient (Only works if the impropers are given as PeriodicTorsionForce in the openmm system)

        If partial charges is None, the charges are obtained from the openmm system.
        mapped_smiles and pdb have no effect on the system, are optional and only required for reproducibility.
        If the improper parameters are incompatible with the openmm system, the improper torsion parameters are all set to zero.
        ----------
        Parameters:
            - openmm_system: openmm.System - the openmm system used for energy and gradient calculations
            - openmm_topology: openmm.Topology - the openmm topology that was used for creating the system
            - xyz: (n_confs, n_atoms, 3) - coordinates of the molecule
            - energy: (n_confs,) - qm energies
            - gradient: (n_confs, n_atoms, 3) - qm gradients
            - mol_id: str - a unique identifier for the molecule
            - partial_charges: (n_atoms,) - partial charges, if None, the charges are obtained from the openmm system
            - mapped_smiles: str - a mapped smiles string that is not used but only stored in the dataset
            - pdb: str - a pdb file as string that is not used but only stored in the dataset
            - ff_name: str - the name of the forcefield that was used to obtain the parameters. If None, the name is set to 'reference_ff'
            - sequence: str - a sequence string that is not used but only stored in the dataset
            - smiles: str - a smiles string that is not used but only stored in the dataset
            - allow_nan_params: bool - If True, the grappa.data.Parameters are set to nans if they cannot be obtained from the forcefield. If False, an error is raised.
        """


        mol = Molecule.from_openmm_system(openmm_system=openmm_system, openmm_topology=openmm_topology, partial_charges=partial_charges, mapped_smiles=mapped_smiles, charge_model=charge_model)

        if not skip_ff:
            try:        
                params = Parameters.from_openmm_system(openmm_system, mol=mol, allow_skip_improper=True)
            except Exception as e:
                if allow_nan_params:
                    params = Parameters.get_nan_params(mol=mol)
                else:
                    tb = traceback.format_exc()
                    raise ValueError(f"Could not obtain parameters from openmm system: {e}\n{tb}. Consider setting allow_nan_params=True, then the parameters for this molecule will be set to nans and ignored during training.")
                
        else:
            params = Parameters.get_nan_params(mol=mol)

        self = cls(molecule=mol, classical_parameters=params, xyz=xyz, energy=energy, gradient=gradient, mapped_smiles=mapped_smiles, pdb=pdb, mol_id=mol_id, sequence=sequence, smiles=smiles)

        if not skip_ff:
            self.add_ff_data(openmm_system=openmm_system, ff_name=ff_name, partial_charges=partial_charges, xyz=xyz)

        return self



    def add_ff_data(self, openmm_system, ff_name=None, partial_charges=None, xyz=None):
        """
        Add the classical forcefield contributions to the MolData object. The contributions are stored in the ff_energy and ff_gradient dictionaries.
        If partial charges is None, the charges are obtained from the openmm system.
        ----------
        Parameters:
            - openmm_system: openmm.System
            - ff_name: str
            - partial_charges: (n_atoms,)
            - xyz: (n_confs, n_atoms, 3)
        """
        from grappa.utils import openmm_utils

        if ff_name is None:
            ff_name = 'reference_ff'

        self.ff_energy[ff_name] = dict()
        self.ff_gradient[ff_name] = dict()

        if not partial_charges is None:
            # set the partial charges in the openmm system
            openmm_system = openmm_utils.set_partial_charges(system=openmm_system, partial_charges=partial_charges)

        # remove the cmmotionremover force if it exists:
        openmm_system = openmm_utils.remove_forces_from_system(openmm_system, 'CMMotionRemover')

        # calculate the reference-forcefield's energy and gradient from the openmm system
        total_ff_energy, total_ff_gradient = openmm_utils.get_energies(openmm_system=openmm_system, xyz=xyz)
        total_ff_gradient = -total_ff_gradient # the reference gradient is the negative of the force
        self.ff_energy[ff_name]['total'] = total_ff_energy
        self.ff_gradient[ff_name]['total'] = total_ff_gradient
        
        nonbonded_energy, nonbonded_gradient = openmm_utils.get_nonbonded_contribution(openmm_system, xyz)
        self.ff_energy[ff_name]['nonbonded'] = nonbonded_energy
        self.ff_gradient[ff_name]['nonbonded'] = nonbonded_gradient

        # store the other contributions as well, for learning only certain contributions.

        improper_energy_ff, improper_gradient_ff = openmm_utils.get_improper_contribution(openmm_system, xyz, molecule=self.molecule)
        self.ff_energy[ff_name]['improper'] = improper_energy_ff
        self.ff_gradient[ff_name]['improper'] = improper_gradient_ff

        bond_energy_ff, bond_gradient_ff = openmm_utils.get_bond_contribution(openmm_system, xyz)
        self.ff_energy[ff_name]['bond'] = bond_energy_ff
        self.ff_gradient[ff_name]['bond'] = bond_gradient_ff

        angle_energy_ff, angle_gradient_ff = openmm_utils.get_angle_contribution(openmm_system, xyz)
        self.ff_energy[ff_name]['angle'] = angle_energy_ff
        self.ff_gradient[ff_name]['angle'] = angle_gradient_ff

        torsion_energy_ff, torsion_gradient_ff = openmm_utils.get_torsion_contribution(openmm_system, xyz)
        self.ff_energy[ff_name]['proper'] = torsion_energy_ff - improper_energy_ff
        self.ff_gradient[ff_name]['proper'] = torsion_gradient_ff - improper_gradient_ff
    

    @classmethod
    def from_smiles(cls, mapped_smiles, xyz, energy, gradient, partial_charges=None, forcefield='openff_unconstrained-1.2.0.offxml', mol_id=None, forcefield_type='openff', smiles=None, allow_nan_params:bool=False, charge_model:str='None', ff_name:str=None):
        """
        Create a Molecule from a mapped smiles string and an openff forcefield. The openff_forcefield is used to initialize the interaction tuples, classical parameters and, if partial_charges is None, to obtain the partial charges.
        The forcefield_type can be either openff, openmm or openmmforcefields.
        ----------
        Parameters:
            - mapped_smiles: str
            - xyz: (n_confs, n_atoms, 3)
            - energy: (n_confs,)
            - gradient: (n_confs, n_atoms, 3)
            - partial_charges: (n_atoms,)
            - forcefield: str
            - mol_id: str
            - forcefield_type: str
            - smiles: str
            - allow_nan_params: bool: If True, the parameters are set to nans if they cannot be obtained from the forcefield. If False, an error is raised.
        
        """
        from grappa.utils import openff_utils, openmm_utils
        if forcefield_type == 'openff':
            system, topology, openff_mol = openff_utils.get_openmm_system(mapped_smiles, openff_forcefield=forcefield, partial_charges=partial_charges)
        
        elif forcefield_type == 'openmm':
            raise NotImplementedError("This does not work for openff molecules at the moment. The residues are needed!")

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
        
        if not smiles is None:
            smiles = openff_mol.to_smiles(mapped=False)

        if mol_id is None:
            mol_id = smiles
        

        self = cls.from_openmm_system(openmm_system=system, openmm_topology=topology, xyz=xyz, energy=energy, gradient=gradient, partial_charges=partial_charges, mapped_smiles=mapped_smiles, mol_id=mol_id, smiles=smiles, allow_nan_params=allow_nan_params, charge_model=charge_model)

        return self


    def __repr__(self):
        return self.__str__()

    def __str__(self):
        n_confs = len(self.energy)
        mol_id = self.mol_id
        molecule_str = str(self.molecule)
        forcefields = ', '.join(self.ff_energy.keys()) if self.ff_energy else 'None'

        return f"<{self.__class__.__name__} (\nn_confs: {n_confs},\nmol_id: {mol_id},\nmolecule: {molecule_str},\nforcefields: {forcefields}\n)>"


    def calc_energies_openmm(self, openmm_forcefield, forcefield_name:str, partial_charges:np.ndarray=None):
        """
        Calculate the energies and gradients of the molecule using an openmm forcefield.
        If partial_charges is None, the charges are obtained from the openmm forcefield.
        """
        assert self.pdb is not None, "MolData.pdb must be provided to calculate energies with openmm."
        from grappa.utils import openmm_utils

        openmm_top = openmm_utils.topology_from_pdb(self.pdb)
        openmm_sys = openmm_forcefield.createSystem(topology=openmm_top)

        if partial_charges is not None:
            openmm_sys = openmm_utils.set_partial_charges(openmm_sys, partial_charges)

        self.write_energies(openmm_system=openmm_sys, forcefield_name=forcefield_name)



    def write_energies(self, openmm_system, forcefield_name:str):
        """
        Write the energies and forces of the molecule to the ff_energy and ff_gradient dicts.
        Assumes that the openmm_system has the correct order of atoms.
        """
        import openmm.unit as unit
        from grappa.utils import openmm_utils

        xyz = unit.Quantity(self.xyz, grappa_units.DISTANCE_UNIT).value_in_unit(unit.angstrom)

        # get the energies and forces from openmm
        total_energy, total_gradient = openmm_utils.get_energies(openmm_system=openmm_system, xyz=xyz)
        total_gradient = -total_gradient

        self.ff_energy[forcefield_name] = unit.Quantity(total_energy, unit.kilocalorie_per_mole).value_in_unit(grappa_units.ENERGY_UNIT)
        self.ff_gradient[forcefield_name] = unit.Quantity(total_gradient, unit.kilocalorie_per_mole/unit.angstrom).value_in_unit(grappa_units.FORCE_UNIT)


    def delete_states(self, delete_idxs:Union[List[int], np.ndarray]):
        """
        Delete states from the object. Raises an error if no states are left.
        """
        if isinstance(delete_idxs, list):
            delete_idxs = np.array(delete_idxs)
        if len(delete_idxs) == 0:
            return self
        if len(delete_idxs) == len(self.energy):
            raise ValueError("Cannot delete all states. At least one state must remain.")
        
        for ff_name, v in self.ff_energy.items():
            for contrib_name, vv in v.items():
                self.ff_energy[ff_name][contrib_name] = np.delete(vv, delete_idxs, axis=0)

        self.xyz = np.delete(self.xyz, delete_idxs, axis=0)
        for ff_name, v in self.ff_gradient.items():
            for contrib_name, vv in v.items():
                self.ff_gradient[ff_name][contrib_name] = np.delete(vv, delete_idxs, axis=0)

        return self

    def add_bonded_ff_information(self, ff_name:str, ff_type:str='openff', ff:str=None, allow_nan_params:bool=False):
        """
        Calculate the energies and gradients using a forcefield and add them to the MolData.ff_energy and MolData.ff_gradient dicts.
        The partial charges remain unchanged.
        ff_name: str - the name under which the entry is stored in the ff_energy and ff_gradient dicts
        ff_type: str - the type of the forcefield. Can be either 'openmm' or 'openff'
        ff: str - the forcefield file used for creating the system.
        allow_nan_params: bool - If True, the grappa.data.Parameters are set to nans if they cannot be obtained from the forcefield. If False, an error is raised.
        """
        # current partial charges:
        partial_charges = self.molecule.partial_charges

        if ff_type == 'openmm':
            raise NotImplementedError("This is not supported yet.")
            # from grappa.utils import openmm_utils
            # openmm_top = openmm_utils.topology_from_pdb(self.pdb)
            # openmm_sys = openmm_utils.create_system_from_pdb(pdb=self.pdb, ff=ff)
        elif ff_type == 'openff':
            from grappa.utils import openff_utils
            openmm_system, topology, openff_mol = openff_utils.get_openmm_system(mapped_smiles=self.mapped_smiles, openff_forcefield=ff, partial_charges=partial_charges)
        else:
            raise ValueError(f"ff_type must be either 'openmm' or 'openff', not {ff_type}")


        # add parameter information:
        try:        
            params = Parameters.from_openmm_system(openmm_system, mol=self.molecule, allow_skip_improper=True)
        except Exception as e:
            if allow_nan_params:
                params = Parameters.get_nan_params(mol=self.molecule)
            else:
                tb = traceback.format_exc()
                raise ValueError(f"Could not obtain parameters from openmm system: {e}\n{tb}. Consider setting allow_nan_params=True, then the parameters for this molecule will be set to nans and ignored during training.")

        self.classical_parameter_dict[ff_name] = params

        # add ff data:
        self.add_ff_data(openmm_system=openmm_system, ff_name=ff_name, partial_charges=partial_charges, xyz=self.xyz)


    @classmethod
    def from_pdb(cls, pdb:Union[str,list], mol_id:str, forcefield:str='amber99sbildn.xml', skip_ff:bool=False):
        """
        Creates a mock MolData object from a pdb file. The energy and gradient are set to nan.
        pdb: str - the contents of the pdb file, separated by newlines or a list of strings describing the lines of the pdb file.
        """

        from openmm.app import PDBFile, ForceField

        ff_name = forcefield.strip('.xml')

        if isinstance(pdb, list):
            assert all(isinstance(p, str) for p in pdb), "All pdb lines must be strings."
            pdbstring = '\n'.join(pdb)

        else:
            assert isinstance(pdb, str), "pdb must be a string or a list of strings."
            pdbstring = pdb

        with TemporaryDirectory() as tmpdirname:
            pdb_file = Path(tmpdirname) / f"{mol_id}.pdb"
            with open(pdb_file, 'w') as f:
                f.write(pdbstring)
                f.flush()

            pdb_file = PDBFile(str(pdb_file))

        ff = ForceField(forcefield)
        system = ff.createSystem(pdb_file.topology)

        moldata = MolData.from_openmm_system(openmm_system=system, openmm_topology=pdb_file.topology, xyz=np.zeros((1,)+np.array(pdb_file.positions).shape), energy=np.array((0,)), gradient=np.zeros((1,)+np.array(pdb_file.positions).shape), mol_id=mol_id, pdb=pdbstring, skip_ff=skip_ff)

        return moldata