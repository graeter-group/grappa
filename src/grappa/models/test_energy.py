# if __name__== "__main__":
#     #%%
#     # test the parametrization:
#     from openmm.app import PDBFile
#     import espaloma as esp
#     from openff.toolkit import Molecule
#     import dgl
#     import os
#     import sys

#     module_path = "/hits/fast/mbm/seutelf/software"
#     if module_path not in sys.path:
#         sys.path.append(module_path)

#     from data_generation import parametrize

#     import models




#     DATASET_PATH = "/hits/fast/mbm/seutelf/data/datasets/pep2/small/AA"
#     glist, _ = dgl.load_graphs(os.path.join(DATASET_PATH, "300K.dgl"))
#     xyz_graph = glist[0]

#     pdb_path = "/hits/fast/mbm/seutelf/data/raw_pdb_files/pep2/AA/pep.pdb"
#     pdb = PDBFile(pdb_path)
#     openff_mol = Molecule.from_polymer_pdb(str(pdb_path))
#     g = esp.Graph(openff_mol).heterograph
#     g.nodes["n1"].data["xyz"] = xyz_graph.nodes["n1"].data["xyz"][:,:5]
#     g = parametrize.parametrize_amber(g, pdb.topology, suffix="_amber99sb")

#     #%%
    
#     en_writer = models.WriteEnergy(suffix="_amber99sb", offset_torsion=True)
#     g = en_writer(g)

#     term_dic = {"n2":"bond", "n3": "angle", "n4": "torsion", "n4_improper":"n4_improper"}

#     for term in en_writer.terms:
#         print(term)
#         contrib = g.nodes["g"].data["u_"+term+"_amber99sb"][0]
#         ref = g.nodes["g"].data["u_"+term_dic[term]+"_amber99sb"][0]
#         print([round(en,ndigits=5) for en in contrib.tolist()])
#         print([round(en,ndigits=5) for en in ref.tolist()])
#         print()
    
#     contrib = g.nodes["g"].data["u_amber99sb"][0]
#     ref = g.nodes["g"].data["u_bonded_amber99sb"][0]
#     print([round(en,ndigits=5) for en in contrib.tolist()])
#     print([round(en,ndigits=5) for en in ref.tolist()])

#     # %%

