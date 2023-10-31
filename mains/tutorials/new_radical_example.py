#%%
# imports:
from pathlib import Path

from kimmdy.parsing import read_top
from kimmdy.topology.topology import Topology
from grappa_interface import generate_input

from grappa.ff import ForceField
from grappa.constants import TopologyDict, ParamDict
from grappa.io import Molecule
#%%
top_path = Path(__file__).parents[0] / "input_data" / "Ala_out.top"
top = Topology(read_top(top_path))
#%%
curr_input = generate_input(top)
#%%    
at_map = top.ff.atomtypes
mol = Molecule.create_empty()
mol.additional_features["is_radical"] = []

for atom in top.atoms.values():
    mol.atoms.append(int(atom.nr))
    mol.atomic_nrs.append(int(at_map[atom.type].at_num))
    mol.partial_charges.append(float(at_map[atom.type].charge))
    mol.epsilons.append(float(at_map[atom.type].epsilon))
    mol.sigmas.append(float(at_map[atom.type].sigma))
    mol.additional_features["is_radical"].append(atom.is_radical)

mol.bonds = [[int(bond.ai),int(bond.aj)] for bond in top.bonds.values()]
mol.angles = [[int(angle.ai),int(angle.aj),int(angle.ak)] for angle in top.angles.values()]
mol.propers = [[int(proper.ai),int(proper.aj),int(proper.ak),int(proper.al)] for proper in top.proper_dihedrals.values()]
mol.impropers = [[int(improper.ai),int(improper.aj),int(improper.ak),int(improper.al)] for improper in top.improper_dihedrals.values()]

# ring membership can be inferred from atom type, I think
#%%
mol.to_json('input_data/Ala_in.json')
#%%
mol2 = Molecule.from_json('input_data/Ala_in.json')

#%%
# investigate the output:

print("dict keys and mean values:\n")
print(*[(k, v.mean()) for (k,v) in zip(params.keys(), params.values())], sep="\n")

print("\n\nsome equililibrium angles:")
print(params["angle_eq"][:5])
print("\n\nsome charges:")
print(params["atom_q"][:5])
print()

#%%
# get information on the units by printing the forcefield:
print(ff)
# %%
# load data describing a radical:
# import json
# with open('input_data/GrAPPa_input_HAT.json', "r") as f:
#     data: TopologyDict = json.load(f)

# # initialize the forcefield from a tag:
# ff = ForceField.from_tag("radical_latest")

# # %%
# # predict parameters
# params: ParamDict = ff.params_from_topology_dict(data)