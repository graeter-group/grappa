#%%
from collections import Counter
import numpy as np
import copy
from typing import List, Tuple, Dict, Union, Callable
from pathlib import Path
import os
import logging

from . import match_utils
from .representation import AtomList

from .match_utils import read_rtp


def match_mol(mol: list, AAs_reference: dict, seq: list, log:bool=True):
    '''
    Returns a list describing a permutation for the atoms in the mol to match them with those in the AAs_reference.

        Parameters:
            mol (list): List of AtomLists, describing a molecule by subgroups
            AAs_reference (dict): A reference to identify residues and their constituents, usually given by a forcefield

        Returns:
            permutation (list of int)
    '''

    if log:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
        logging.info(f"--- Matching molecule with sequence {seq} to reference ---")

    mol_ref = _create_molref(AAs_reference, seq)

    if log:
        logging.info(f"created reference: {mol_ref}")

    atom_order = []

    for i, res in enumerate(seq):
        res_atom_order = match_residue(i=i, res=res, mol_ref=mol_ref, mol=mol, log=log)
        atom_order.extend(res_atom_order)

    return atom_order


def seq_from_filename(filename:Path, AAs_reference:dict=None, cap:bool=True):
    '''
    Returns a list of strings describing a sequence found in a filename.
    Only works for single amino acids with caps, assumes that non-radicals contain '_nat'.
    '''    
    components = filename.stem.split(sep='_')
    seq = []

    if not any([components[1][:3] in ['Ace','Nme']]):
        seq.append(components[0].upper() if components[1] == 'nat' else components[0].upper() + '_R')
    elif len(components[0]) == 3:
        if 'Ace' in components[1]:
            return ['ACE_R',components[0].upper(),'NME']
        else:
            return ['ACE',components[0].upper(),'NME_R']

    if cap:
        if len(components[0]) == 3:
            seq.insert(0,'ACE')
            seq.append('NME')
        elif not AAs_reference is None:
            if components[0][1:].upper() in AAs_reference.keys():
                if components[0].startswith('N'):
                    seq.append('NME')
                elif components[0].startswith('C'):
                    seq.insert(0,'ACE')
                else:
                    raise ValueError(f"Invalid filename {filename} for sequence conversion!")
        else:
            raise ValueError(f"Invalid filename {filename} for sequence conversion!")
    return seq

def get_radref(filename:Union[str,Path], rtp_path:Union[str,Path], cap:bool=True):
    """
    Creates a reference dict from an rtp and updates it for a radical with position inferred from the filename and the sequence of the molecule. If the molecule is not a radical, only returns the reference from rtp path and the sequence.
    """
    filename = Path(filename)
    AAs_reference = read_rtp(rtp_path)
    seq = seq_from_filename(filename=filename, AAs_reference=AAs_reference, cap=cap)
    if not match_utils.is_radical(filename):
        return AAs_reference, seq
    
    # infer radical atom from filename
    heavy_name = match_utils.radname_from_log(filename)
    # infer AA from filename
    rad_AAs = []
    for aa in seq:
        if aa.endswith('_R'):
            rad_AAs.append(aa)

    if len(rad_AAs) > 1:
        raise ValueError(f"More than one radical residue in sequence {seq}. This is not supported yet.")
    
    rad_AA = rad_AAs[0]
    rad_AA = rad_AA[:-2]

    radAA_atom_names = [atom[0] for atom in AAs_reference[rad_AA]['atoms']]
    if not heavy_name in radAA_atom_names:
        raise ValueError(f"Could not find radical atom {heavy_name} in residue {rad_AA},\nheavy name prediction is false, filename stem is: {filename.stem},\nff reference is {radAA_atom_names}")

    rad_reference = generate_radical_reference(AAs_reference=AAs_reference, AA=rad_AA, heavy_name=heavy_name, log=False)

    return rad_reference, seq

def generate_radical_reference(AAs_reference: dict, AA: str, heavy_name: str, log:bool=True):
    '''
    generates a dict with reference for one radical residue given by AA and heavy)name.
    Assumes a hydrogen was abstracted from the radical atom, carrying the name 'heavy_name'
    '''
    if log:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
        logging.info(f"--- Generating radical reference entry for residue {AA} with a {heavy_name} radical ---")
    ## search for hydrogen attached to radical
    rmv_H = None
    for bond in AAs_reference[AA]['bonds'][::-1]:
        bond_atoms = bond
        if heavy_name in bond_atoms:
            heavy_partner = bond_atoms[0] if heavy_name == bond_atoms[1] else bond_atoms[1]
            # assumes that it does not matter which hydrogen was abstracted
            if heavy_partner.startswith('H'):
                rmv_H = heavy_partner
                break

    if rmv_H is None:
        raise ValueError(f"Could not find hydrogen attached to radical {heavy_name} in residue {AA}!")

    AAs_reference[AA + '_R'] = copy.deepcopy(AAs_reference[AA])
    for atom in AAs_reference[AA + '_R']['atoms']:
        if rmv_H == atom[0]:
            rmv_atom = atom
            break
    for bond in AAs_reference[AA + '_R']['bonds']:
        if rmv_H in bond:
            rmv_bond = bond
            break
    AAs_reference[AA + '_R']['atoms'].remove(rmv_atom)
    AAs_reference[AA + '_R']['bonds'].remove(rmv_bond)
    return AAs_reference


def read_g09(file: Path, sequence: str, AAs_reference: dict, trajectory_in=None, log:bool=True):
    '''
    Returns a mol, i.e. a list of atom lists and an ASE trajectory.
    This assumes that the atoms are ordered by residue in the order of the given sequence!
    '''
    from ase.io import read
    if trajectory_in is None:
        if log:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
            logging.info(f"--- Starting to read coordinates from file {file} with sequence {sequence} ---")
        trajectory = read(file,index=':')
    else:
        trajectory = trajectory_in

    ## get formula and separation of atoms into residues ##
    formula = trajectory[-1].get_chemical_formula(mode='all')

    AA_delim_idxs = []              # index of first atom in new residue   
    pos = 0
    # this assignment of AA_delim_idx assumes the atoms to be ordered by residues
    for AA in sequence:
        AA_len = len(AAs_reference[AA]['atoms'])
        AA_delim_idxs.append([pos,pos+AA_len])
        pos += AA_len
    assert pos == len(formula) , f"Mismatch in number of atoms ({pos} vs {len(formula)}) assumed from input sequence {sequence}:{pos} and {formula}:{len(formula)}"

    ## split molecule into residues and get list of bonds. reconstruct atomnames using list ##
    bonds = match_utils.bond_majority_vote(trajectory)

    res_AtomLists = []
    for _,AA_delim_idx in enumerate(AA_delim_idxs):

        AA_atoms = [[idx,formula[idx]] for idx in list(range(*AA_delim_idx))]
        
        AA_partners = [list(np.extract((np.array(bond) < AA_delim_idx[1]) & (AA_delim_idx[0] <= np.array(bond)),bond)) for bond in bonds[AA_delim_idx[0]:AA_delim_idx[1]]]
        
        AA_bonds = [[i+AA_delim_idx[0],partner] for i, partners in enumerate(AA_partners) for partner in partners]

        res_AtomLists.append(AtomList(AA_atoms,AA_bonds))

    return res_AtomLists, trajectory


################
# MATCH ONE RESIDUE
def match_residue(i:int, res:str, mol_ref, mol, log:bool=False):
    if log:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    ## get mol_ref from function here
        
    AL = mol[i]
    AL_ref = mol_ref[i]

    ## create mapping scheme for sorting the atoms correctly ##
    c1 = sorted(AL.get_neighbor_elements(0))
    c2 = sorted(AL_ref.get_neighbor_elements(0))
    assert  c1 == c2 , f"Elements for residue {i} ({c1}) don't map to FF reference ({c2}! Can't continue the mapping" 
    # assume both graphs are isomorphic after this assert

    mapdict = {}
    # maps from the indices of the atoms in the residue to the indices of the atoms in the reference
    # #try simple matching by neighbor elements
    for order in range(4):
        n1 = AL.get_neighbor_elements(order)
        n2 = AL_ref.get_neighbor_elements(order)

        occurences = Counter(n1)
        if log:
            logging.debug(order)
            logging.debug(occurences)
            logging.debug(mapdict)

        for key,val in occurences.items():
            if val == 1:
                if AL.atoms[n1.index(key)].idx not in mapdict.keys():
                    if log:
                        logging.debug(AL.atoms[n1.index(key)].idx,mapdict)
                    mapdict[AL.atoms[n1.index(key)].idx] = AL_ref.atoms[n2.index(key)].idx
    AA_remainder = []
    AA_refremainder = []
    rmapdict = dict(map(reversed, mapdict.items()))
    if len(mapdict.keys()) < len(AL):
        AA_remainder = sorted(list(set(AL.idxs)-set(mapdict.keys())))
        AA_refremainder = sorted(list(set(AL_ref.idxs)-set(rmapdict.keys())))
        if log:
            logging.debug(f"{AA_remainder}/{AA_refremainder} remain unmatched after neighbor comparison!")
    if log:
        logging.debug(mapdict)

    # try matching of indistinguishable atoms bound to the same atom that has a mapping under the condition that all neighbouring atoms match too (i.e. matching depth == 1, maybe change this: NOTE).
    # do this as long as this has an effect (i.e. the number of unmatched gets smaller)
    n_unmatched = len(AA_remainder)
    first_run = True
    while (n_unmatched > 0):
        if (not first_run) and len(AA_remainder)==n_unmatched:
            break
        first_run = False
        n_unmatched = len(AA_remainder)
        # the atom indices that are already matched:
        AA_match = list(mapdict.keys())
        for matched in AA_match:

            # get lists of Atoms that are neighbors and ref-neighbors and that are not matched already
            neighbors = [n for n in AL.by_idx(matched).neighbors if n.idx in AA_remainder]
            refneighbors = [n for n in AL_ref.by_idx(mapdict[matched]).neighbors if n.idx in AA_refremainder]

            # now sort those lists such that the elements match, then we can identify the atoms by each other:
            neighbors.sort(key=lambda a: a.element)
            refneighbors.sort(key=lambda a: a.element)
            assert [a.element for a in neighbors] == [a.element for a in refneighbors]

            # if the list has the same length, match list entries, else throw an error
            if len(neighbors) == len(refneighbors):
                for j in range(len(neighbors)):
                    n = neighbors[j]
                    r = refneighbors[j]
                    # get lists of matched neighbors
                    n_neighb = [k for k in n.neighbors if not k.idx in AA_remainder]
                    r_neighb = [k for k in r.neighbors if not k.idx in AA_refremainder]
                    
                    # if the matched neighbors agree, match the atoms
                    if len(n_neighb) == len(r_neighb):
                        if set([mapdict[k.idx] for k in n_neighb]) == set([k.idx for k in r_neighb]):
                            n_id = n.idx
                            r_id = r.idx
                            
                            mapdict[n_id] = r_id
                            rmapdict[r_id] = n_id
                            AA_remainder.remove(n_id)
                            AA_refremainder.remove(r_id)

                            if log:
                                logging.debug(f"match {n_id}:{r_id}")
            else:
                raise ValueError(f"Number of neighbors for {matched} ({len(neighbors)}) and {mapdict[matched]} ({len(refneighbors)}) don't match! Can't continue the mapping")

    if log:
        if len(AA_remainder):
            logging.info(f"{AA_remainder}/{AA_refremainder} in {res} remain unmatched; checking for indistinguishable atoms")
        else:
            logging.info(f"All atoms in {res} were successfully matched:{mapdict}!")
        


    if len(AA_remainder):
        raise ValueError(f"{AA_remainder}/{AA_refremainder} in {res} remain unmatched, cannot complete matching")

    if log:
        logging.debug(mol_ref[i].idxs)
        logging.debug(rmapdict)

    res_atom_order = [rmapdict[atom] for atom in mol_ref[i].idxs]

    return res_atom_order
##################


def write_trjtopdb(outfile: Path, trajectory, atom_order: list, seq: list, AAs_reference: dict, log:bool=True):
    '''
    For all states in the trajectory, a PDB file containing the energy in eV is generated.
    '''
    from ase.calculators.calculator import PropertyNotImplementedError

    if log:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
        logging.info(f"--- Writing Trajectory of molecule with sequence {seq} to PDBs ---")
    outfile.parent.mkdir(exist_ok=True)

    ## write single pdb per opt step## 
    for i,step in enumerate(trajectory):
        writefile = outfile.parent / f"{outfile.stem}_{i}.pdb"
        with open(writefile,'w') as f:
            f.write(f"COMPND     step {i}\n")
            f.write(f"AUTHOR     generated by {__file__}\n")
            try:
                f.write(f"REMARK   0\nREMARK   0 energy: {step.get_total_energy()} eV\n")
            except PropertyNotImplementedError:
                f.write(f"REMARK   0\nREMARK   0 energy: nan eV\n")
            AA_pos = 0
            res_len = len(AAs_reference[seq[AA_pos]]['atoms'])
            for j,k in enumerate(atom_order):
                if j >= res_len:
                    AA_pos += 1
                    res_len += len(AAs_reference[seq[AA_pos]]['atoms'])
                atomname = AAs_reference[seq[AA_pos]]['atoms'][j-res_len][0]
                f.write('{0}  {1:>5d} {2:^4s} {3:<3s} {4:1s}{5:>4d}    {6:8.3f}{7:8.3f}{8:8.3f}{9:6.2f}{10:6.2f}          {11:>2s}\n'.format('ATOM',j,atomname ,seq[AA_pos].split(sep="_")[0],'A',AA_pos+1,*step.positions[k],1.00,0.00,atomname[0]))
            f.write("END")
    return


## general utils ##
'''
Construct a molecule, i.e. List of AtomLists, from what the force field expecs the residues to look like. 
'''
def _create_molref(AAs_reference: dict, sequence: list):
    mol_ref = []
    for res in sequence:
        AA_refatoms = [[atom[0],atom[0][0]] for atom in AAs_reference[res]['atoms']]
        AA_refbonds = [bond for bond in AAs_reference[res]['bonds'] if not any(idx.startswith(('-','+')) for idx in bond)]
        AA_refAtomList = AtomList(AA_refatoms,AA_refbonds)
        mol_ref.append(AA_refAtomList)
    return mol_ref

#%%


# testing:
if __name__ == "__main__":
    DATASET_PATH = "xyz2res/scripts/data/pdbs/pep1/F"
    from pathlib import Path

    from openmm.app import PDBFile
    from openmm.unit import angstrom
    import numpy as np
    import tempfile
    from .match_utils import read_rtp
    import ase
    from ase.geometry.analysis import Analysis
    rtp_path = Path(__file__).parent.parent/Path("classical_ff/amber99sb-star-ildnp.ff/aminoacids.rtp")
    #%%

    for p in Path(DATASET_PATH).rglob("*.pdb"):
        # make ase molecule:
        pdb = PDBFile(str(p))

        residues = []

        xyz = pdb.positions
        xyz = xyz.value_in_unit(angstrom)
        xyz = np.array([[np.array(v) for v in xyz]])

        elements = []
        for a in pdb.topology.atoms():
            elements.append(a.element.atomic_number)
            if not a.residue.name in residues:
                residues.append(a.residue.name)


        elements = np.array(elements)
        sequence = residues

        n = xyz[0].shape[0]
        pos = xyz[0]
        ase_mol = ase.Atoms(f"N{n}")
        ase_mol.set_positions(pos)
        ase_mol.set_atomic_numbers(elements)

        # get AtomLists-molecule
        AAs_reference = read_rtp(rtp_path)
        mol, _ = read_g09(None, sequence, AAs_reference, trajectory_in=[ase_mol], log=False)

        # call match_mol
        atom_order = match_mol(mol, AAs_reference, sequence, log=True)

# %%
