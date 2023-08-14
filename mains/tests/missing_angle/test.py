#%%
import openmm
from openmm.app import ForceField
from grappa.ff_utils.SysWriter import SysWriter
import json
#%%
with open("in.json", "r") as f:
    in_dict = json.load(f)

#%%
writer = SysWriter.from_dict(in_dict, allow_radicals=True, classical_ff=ForceField("amber99sbildn.xml"))
# %%
sys = writer.sys
# %%
for f in sys.getForces():
    if isinstance(f, openmm.HarmonicAngleForce):
        print(f.getNumAngles())

#%%
forcefield = ForceField("amber99sbildn.xml")
topology = writer.top
# %%
import copy
from grappa.ff_utils.create_graph.read_pdb import one_atom_replace_h23_to_h12
from grappa.ff_utils.create_graph import read_pdb

def generate_unmatched_templates(topology:topology, forcefield:ForceField):
    """
    Removes the entries LYN and CYM from the forcefield.
    This function extends generateTemplatesForUnmatchedResidues from openmm:
    In the forcefield there are some templated that match the radical, i.e. the uncharged version of a positively charged residue. we add these to the templates to the unmatched ones with the corrected name. we can do this because our ff only applies to the standard versions of amino acids, any deviation is assumed to be caused by being a radical.
    Not guaranteed to work properly for forcefields other than amber99sbildn.
    """
    [templates, residues] = forcefield.generateTemplatesForUnmatchedResidues(topology)
    
    corrections = generate_unmatched_templates.corrections

    try:
        matches = copy.deepcopy(forcefield.getMatchingTemplates(topology))
    except ValueError:
        # this happens when there are no matched residues
        matches = None
    
    if not matches is None:
        for (residue, match) in zip(topology.residues(), matches):
            if match.name in corrections.keys():
                original_res = corrections[match.name]
                # fake a generated fail-template:
                match.name = original_res
                for i,_ in enumerate(match.atoms):
                    # rename some Hs according to the PDBMolecule convention:
                    match.atoms[i].name = one_atom_replace_h23_to_h12(match.atoms[i].name, resname=original_res)
                    
                    # delete the types so that they are regenerated later:
                    match.atoms[i].type = None

                templates.append(match)
                residues.append(residue)

    return templates, residues
# these are empirical for the amber99sbildn forcefield
generate_unmatched_templates.corrections = {"LYN":"LYS", "CYM":"CYS"}
#%%
[templates, residues] = generate_unmatched_templates(topology=topology, forcefield=forcefield)

for t_idx, template in enumerate(templates):
    resname = template.name
    if resname == "HIS":
        resname = "HIE"
    ref_template = forcefield._templates[resname]
    # the atom names stored in the template of the residue
    ref_names = [a.name for a in ref_template.atoms]
    ref_names = [one_atom_replace_h23_to_h12(n, resname=resname) for n in ref_names]

    # check whether all atoms can be matched:
    atom_names = [read_pdb.one_atom_replace_h23_to_h12(a.name, resname=resname) for a in template.atoms]
    diff1 = set(atom_names) - set(ref_names)
    diff2 = set(ref_names) - set(atom_names)
    if len(diff2) > 2 or len(diff1) > 0:
        raise ValueError(f"Template {template.name} does not match reference template:\nIn pdb, not in reference: {diff1}\nIn reference, not in pdb:{diff2},\nallowed is at most one Hydrogen atom that is not in the pdb and no atom that is not in the reference.")
        
    # iterate over the present atoms:
    for atom in template.atoms:
        name = read_pdb.one_atom_replace_h23_to_h12(atom.name, resname=resname)

        # find the atom with that name in the reference template:
        try:
            ref_idx = ref_names.index(name)
        except ValueError:
            print(f"Atom {name} not found in reference template {template.name}: {ref_names}")
            raise
            # set the atom types to be the same:
        atom.type = ref_template.atoms[ref_idx].type

    # create a new template
    template.name = template.name +f"_rad_{t_idx}"
    forcefield.registerResidueTemplate(template)
# %%
for t in templates:
    print(*dir(t), sep="\n")
# %%


# %%
for i in top.atoms():
    print(i.name, i.id, i.index, i.residue.index)
# %%

# %%
from grappa.ff_utils.create_graph.utils import openmm2rdkit_graph
from grappa.ff_utils.create_graph.tuple_indices import get_indices
# %%
rdmol = openmm2rdkit_graph(topology)
# %%
idxs = get_indices(rdmol)
# %%
len(idxs["n3"])
# %%
