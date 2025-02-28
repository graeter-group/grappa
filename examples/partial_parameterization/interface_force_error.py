# %%
"""
Analysis of the cRMSE force and grappa contribution for the interface atoms in the 'spice-dipeptide-amber99-sidechain-grappa-test' dataset.

Please generate the 'spice-dipeptide-amber99-sidechain-grappa-test' dataset by running the script 'create_spice_dipetide_partial.py' and subsequently 'create_test_dataset.py' beforehand:).
"""
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import seaborn as sns
from dgl import DGLGraph

from grappa.models import Energy
from grappa.models.internal_coordinates import InternalCoordinates
from grappa.utils.model_loading_utils import model_from_tag, model_from_path
from grappa.data import Dataset, MolData
from grappa.utils.graph_utils import is_hydrogen_atom, is_carbon_atom, is_nitrogen_atom, is_oxygen_atom


def indices_to_mask(indices: list | set, length: int) -> np.ndarray:
    """
    Convert a list of indices to a binary mask.

    Args:
        indices (list | set): The indices to convert.
        length (int): Number of indices.
    """
    if isinstance(indices, set):
        indices = list(indices)
    bool_array = np.zeros(length, dtype=bool)
    bool_array[indices] = True
    return bool_array


def componentwise_rmse_force(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """
    Calculate the compentwise RMSE (cRMSE).
    Args:
        y_true (np.ndarray): The true forces.
        y_pred (np.ndarray): The predicted forces.
        mask (np.ndarray): Binary mask for .
    """
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))

def get_grappa_contribution_for_atom(graph: DGLGraph, atom: int, min_grappa_atoms: dict={"n2": 1, "n3": 1, "n4": 1, "n4_improper": 1}) -> dict:
    """
    Compute the grappa contribution to the interactions for a given atom.

    Args:
        graph (DGLGraph): The graph object.
        atom (int): The atom index to calculate the grappa contribution for.
        min_grappa_atoms (dict): The minimum number of grappa atoms per interaction. Default is {"n2": 1, "n3": 1, "n4": 1, "n4_improper": 1}.
    """
    contribution = {}
    term_to_interaction = {"n2": "bonds", "n3": "angles", "n4": "propers", "n4_improper": "impropers"}
    for term in ["n2", "n3", "n4", "n4_improper"]:
        interaction = term_to_interaction[term]
        interactions = graph.nodes[term].data["idxs"].detach().numpy()
        is_grappa_interaction = graph.nodes[term].data["num_grappa_atoms"].detach().numpy() >= min_grappa_atoms[term]
        interactions_with_atom = np.where(interactions == atom)[0]
        contribution[f"grappa_{interaction}"] = is_grappa_interaction[interactions_with_atom].sum()
        contribution[f"total_{interaction}"] = len(interactions_with_atom)
        contribution[f"trad_{interaction}"] = contribution[f"total_{interaction}"] - contribution[f"grappa_{interaction}"]
        # Check if the atom is involved in any interactions of the interaction type 
        if contribution[f"total_{interaction}"]:
            contribution[f"grappa_ratio_{interaction}"] = contribution[f"grappa_{interaction}"] / contribution[f"total_{interaction}"]
        else: # Some atoms do not have impropers
            contribution[f"grappa_ratio_{interaction}"] = None

    contribution["grappa_total"] = sum([contribution[f"grappa_{interaction}"] for interaction in ["bonds", "angles", "propers", "impropers"]])
    contribution["total"] = sum([contribution[f"total_{interaction}"] for interaction in ["bonds", "angles", "propers", "impropers"]])
    contribution["trad_total"] = contribution["total"] - contribution["grappa_total"]
    contribution["grappa_ratio"] = contribution["grappa_total"] / contribution["total"]
    return contribution

def get_force_crmse_and_grappa_contribution_of_sidechain_interface(ckpt_path: str) -> Tuple[dict, dict, dict, list]:
    """
    Compute the cRMSE force and grappa contributuion for the interface atoms in the 'spice-dipeptide-amber99-sidechain-grappa-test' dataset.

    Interface atoms have mixed parameters from grappa and amber. The cRMSE force is calculated for the partial parameteriation, the full amber parameterization and the full grappa parameterization. The 'spice-dipeptide-amber99-sidechain-grappa-test' needs to be generated prior by running the script 'create_spice_dipetide_partial.py' and subsequently 'create_test_dataset.py'. Please note that the three proline residues of the dataset are excluded from the computation of the cRMSE force and grappa contribution for the inidividual interface atoms.
    
    Args:
        ckpt_path (str): The path to the checkpoint file.
    """
    min_grappa_atoms: dict={"bond": 1, "angle": 1, "proper": 1, "improper": 1} # No other combinations are supported right now
    grappa_model = model_from_path(ckpt_path)
    ds = Dataset.from_tag('spice-dipeptide-amber99-sidechain-grappa-test')
    energy_module_partial = Energy(partial_param=True, min_grappa_atoms=min_grappa_atoms)
    energy_modul_full = Energy(partial_param=False)
    rmse_partial_dict, rmse_amber_dict, rmse_grappa_dict = defaultdict(list), defaultdict(list), defaultdict(list)
    contribution = []
    for graph in tqdm(ds.graphs):
        atom_masks = {}
        graph = grappa_model(graph)
        graph_partial = energy_module_partial(graph.clone())
        graph_grappa = energy_modul_full(graph.clone())

        # Get grappa and qm forces:
        forces_grappa_partial_bonded = -graph_partial.nodes['n1'].data['gradient']
        forces_grappa_partial = forces_grappa_partial_bonded - graph_partial.nodes['n1'].data['gradient_amber99sbildn_nonbonded']
        forces_grappa_partial = forces_grappa_partial.detach().numpy()
        forces_qm = -graph_partial.nodes['n1'].data['gradient_qm'].detach().numpy()
        forces_amber = -graph_partial.nodes['n1'].data['gradient_amber99sbildn_total'].detach().numpy()
        forces_grappa_full_bonded = -graph_grappa.nodes['n1'].data['gradient'].detach().numpy()
        forces_grappa_full = forces_grappa_full_bonded - graph_grappa.nodes['n1'].data['gradient_amber99sbildn_nonbonded'].detach().numpy()
       

        # Get grappa, interface and amber atoms:
        atom_masks["grappa"] = graph_partial.nodes["n1"].data["grappa_atom"].detach().numpy().astype(bool)
        is_grappa_bond = graph_partial.nodes["n2"].data["num_grappa_atoms"].detach().numpy() >= min_grappa_atoms["bond"]
        is_grappa_angle = graph_partial.nodes["n3"].data["num_grappa_atoms"].detach().numpy() >= min_grappa_atoms["angle"]
        is_grappa_proper = graph_partial.nodes["n4"].data["num_grappa_atoms"].detach().numpy() >= min_grappa_atoms["proper"]
        is_grappa_improper = graph_partial.nodes["n4_improper"].data["num_grappa_atoms"].detach().numpy() >= min_grappa_atoms["improper"]

        bonds = graph_partial.nodes["n2"].data["idxs"].detach().numpy()
        angles = graph_partial.nodes["n3"].data["idxs"].detach().numpy()
        propers = graph_partial.nodes["n4"].data["idxs"].detach().numpy()
        impropers = graph_partial.nodes["n4_improper"].data["idxs"].detach().numpy()

        total_atoms = graph_partial.num_nodes("n1")

        grappa_bond_atoms_idxs = set(bonds[is_grappa_bond].flatten())
        is_grappa_bond_atom = indices_to_mask(grappa_bond_atoms_idxs, total_atoms)
        grappa_angle_atoms_idxs = set(angles[is_grappa_angle].flatten())
        is_grappa_angle_atom = indices_to_mask(grappa_angle_atoms_idxs, total_atoms)
        grappa_proper_atoms_idxs = set(propers[is_grappa_proper].flatten())
        is_grappa_proper_atom = indices_to_mask(grappa_proper_atoms_idxs, total_atoms)
        grappa_improper_atoms_idxs = set(impropers[is_grappa_improper].flatten())
        is_grappa_improper_atom = indices_to_mask(grappa_improper_atoms_idxs, total_atoms)

        # Grappa dominant only: ~ is_grappa_atom & is_grappa_proper_atom
        atom_masks["interface"] = (~atom_masks["grappa"]) & (is_grappa_bond_atom | is_grappa_angle_atom | is_grappa_proper_atom |  is_grappa_improper_atom)
        atom_masks["amber"] = ~(atom_masks["grappa"] |  atom_masks["interface"])

        # Get individual interface atoms for all amino acids except Pro:
        if sum(atom_masks["interface"]) == 8:
            atom_masks["CA"] = is_grappa_bond_atom & ~atom_masks["grappa"]
            assert sum(atom_masks["CA"]) == 1, f"{sum(atom_masks['CA'])} interface CA atoms found."

            angel_reached_atoms = np.where(is_grappa_angle_atom & ~is_grappa_bond_atom)[0]
            assert len(angel_reached_atoms) == 3, f"{len(angel_reached_atoms)} interface angle atoms found." 
            for atom in angel_reached_atoms:
                if is_hydrogen_atom(graph_partial, atom):
                    atom_masks["HA"] = indices_to_mask({atom}, total_atoms)
                elif is_nitrogen_atom(graph_partial, atom):
                    atom_masks["N1"] = indices_to_mask({atom}, total_atoms)
                elif is_carbon_atom(graph_partial, atom):
                    atom_masks["C1"] = indices_to_mask({atom}, total_atoms)
                else:
                    raise ValueError(f"Unexpected atom type found for interface angles: {graph_partial.nodes['n1'].data['atomic_numbers'][atom]}")
            proper_reached_atoms = np.where(is_grappa_proper_atom & ~is_grappa_angle_atom)[0]
            assert len(proper_reached_atoms) == 4, f"{len(proper_reached_atoms)} interface proper atoms found."    
            for atom in proper_reached_atoms:
                if is_hydrogen_atom(graph_partial, atom):
                    atom_masks["H1"] = indices_to_mask({atom}, total_atoms)
                elif is_carbon_atom(graph_partial, atom):
                    atom_masks["C0"] = indices_to_mask({atom}, total_atoms)
                elif is_nitrogen_atom(graph_partial, atom):
                    atom_masks["N2"] = indices_to_mask({atom}, total_atoms)
                elif is_oxygen_atom(graph_partial, atom):
                    atom_masks["O1"] = indices_to_mask({atom}, total_atoms)
                else:
                    raise ValueError(f"Unexpected atom type found for interface propers: {graph_partial.nodes['n1'].data['atomic_numbers'][atom]}")
            for name in ["C0", "N1", "H1", "CA", "HA", "C1", "O1", "N2"]:
                atom = np.where(atom_masks[name])[0]
                # grappa_ratio[name].append(get_grappa_contribution_for_atom(graph_partial, atom))
                c = get_grappa_contribution_for_atom(graph_partial, atom)
                c["atom"] = name
                contribution.append(c)


        # Get root mean sequared error of forces for the whole molecule:
        for name, mask in atom_masks.items():
            rmse_partial_dict[name].append(componentwise_rmse_force(forces_qm, forces_grappa_partial, mask))
            rmse_amber_dict[name].append(componentwise_rmse_force(forces_qm, forces_amber, mask))
            rmse_grappa_dict[name].append(componentwise_rmse_force(forces_qm, forces_grappa_full, mask))
            
    return rmse_partial_dict, rmse_amber_dict, rmse_grappa_dict, contribution


# %%
# Comppute the cRMSE force for the partial parameterization, the full amber parameterization and the full grappa parameterization for the vanilla grappa 1.40 model for the 'spice-dipeptide-amber99-sidechain-grappa-test' dataset. Grappa contributions are calculated for the interface atoms in the dataset, too.
partial, amber, grappa, contribution = get_force_crmse_and_grappa_contribution_of_sidechain_interface("../../models/grappa-1.4.0/checkpoint.ckpt")


# %%
# Plot the cRMSE force for the partial parameterization against the full amber parameterization.
# Amber atoms and Grappa atoms are parameterized by the Amber ff99SB-ILDN force field and the Grappa 1.40 force field, respectively. The interface atoms are parameterized by a mix of both force fields.
aspect = 1.0

plt.figure(figsize=plt.figaspect(aspect)).subplots_adjust(
    left=0.2, right=0.9, top=0.9, bottom=0.2
)
plt.scatter(partial["grappa"], amber["grappa"], label="Grappa atoms", color="#dc7ec0")
plt.scatter(partial["interface"], amber["interface"], label="Interface atoms", color="#927bc8") # "#956cb4"
plt.scatter(partial["amber"], amber["amber"], label="Amber atoms", color="#4878d0")
plt.plot([0, 20], [0, 20], color='black', linestyle='--')
plt.xlabel("Grappa 1.40 - partial params", size=12)
plt.ylabel("Amber ff99SB-ILDN - full params", size=12)

plt.xlim(*(0, 20))
plt.ylim(*(0, 20))

x_lim = plt.gca().get_xlim()
xtick_pos = np.linspace(*x_lim, 5)
xtick_label = [str(int(pos)) for pos in xtick_pos]  # Float to int conversion !!!
plt.xticks(ticks=xtick_pos, labels=xtick_label, size=12)
plt.yticks(ticks=xtick_pos, labels=xtick_label, size=12)

plt.legend(fontsize=12, title_fontsize=14)

plt.title("cRMSE force [kcal/mol/Å]", size=12)


# %%
# Plot the cRMSE force for the partial parameterization against the full grappa parameterization.
aspect = 1.0

plt.figure(figsize=plt.figaspect(aspect)).subplots_adjust(
    left=0.2, right=0.9, top=0.9, bottom=0.2
)
plt.scatter(partial["grappa"], grappa["grappa"], label="Grappa atoms", color="#dc7ec0")
plt.scatter(partial["interface"], grappa["interface"], label="Interface atoms", color="#927bc8") # "#956cb4"
plt.scatter(partial["amber"], grappa["amber"], label="Amber atoms", color="#4878d0")
plt.plot([0, 20], [0, 20], color='black', linestyle='--')
plt.xlabel("Grappa 1.40 - partial params", size=12)
plt.ylabel("Grappa 1.40 - full params", size=12)

plt.xlim(*(0, 20))
plt.ylim(*(0, 20))

x_lim = plt.gca().get_xlim()
xtick_pos = np.linspace(*x_lim, 5)
xtick_label = [str(int(pos)) for pos in xtick_pos]  # Float to int conversion !!!
plt.xticks(ticks=xtick_pos, labels=xtick_label, size=12)
plt.yticks(ticks=xtick_pos, labels=xtick_label, size=12)

plt.legend(fontsize=12, title_fontsize=14)
plt.title("cRMSE force [kcal/mol/Å]", size=12)


# %%
# Now, let's put the cRMSE force for the individual interface atoms into a dataframe and plot them in a boxplot.
df_partial = pd.DataFrame({k: partial[k] for k in ["C0", "N1", "H1", "CA", "HA", "C1", "O1", "N2"]}).melt(var_name="Atom", value_name="RMSE force")
df_amber = pd.DataFrame({k: amber[k] for k in ["C0", "N1", "H1", "CA", "HA", "C1", "O1", "N2"]}).melt(var_name="Atom", value_name="RMSE force")
df_grappa = pd.DataFrame({k: grappa[k] for k in ["C0", "N1", "H1", "CA", "HA", "C1", "O1", "N2"]}).melt(var_name="Atom", value_name="RMSE force")

df_partial["Model"] = "Grappa 1.40 - partial params"
df_amber["Model"] = "Amber ff99SB-ILDN - full params"
df_grappa["Model"] = "Grappa 1.40 - full params"
df = pd.concat([df_partial, df_amber, df_grappa])

# %%
aspect = 0.75

plt.figure(figsize=plt.figaspect(aspect)).subplots_adjust(
    left=0.2, right=0.9, top=0.9, bottom=0.2
)
sns.boxplot(data=df, x="Atom", y="RMSE force", hue="Model", palette=["#ee854a", "#82c6e2","#d65f5f",])

plt.xlabel("Interface atoms", size=12)
plt.ylabel("cRMSE force [kcal/mol/Å]", size=12)

plt.xticks(size=12)
plt.yticks(size=12)
plt.legend(title=None, fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.title("cRMSE force [kcal/mol/Å]", size=12)

# %%
# Next, let's plot ratio of Grapapa bonds, angles, propers, and impropers for the individual interface atoms in a barplot.

# Construct a appropriate dataframe for the barplot
df_contribution = pd.DataFrame(contribution)
df_interaction_ratios = df_contribution[["grappa_ratio_bonds", "grappa_ratio_angles", "grappa_ratio_propers", "grappa_ratio_impropers", "atom"]].melt(id_vars="atom", var_name="Interaction", value_name="Grappa ratio")
df_interaction_ratios["Interaction"] = df_interaction_ratios["Interaction"].str.replace("grappa_ratio_", "")
df_interaction_ratios["Interaction"] = df_interaction_ratios["Interaction"].str.capitalize()

aspect = 0.75
plt.figure(figsize=plt.figaspect(aspect)).subplots_adjust(
    left=0.2, right=0.9, top=0.9, bottom=0.2
)

ax = sns.barplot(df_interaction_ratios, x="atom", y="Grappa ratio", hue="Interaction", palette=sns.color_palette("gray")[::-1], edgecolor="black", linewidth=2, err_kws={"color": "black", "linewidth": 2}, capsize= 0.2, errorbar="sd")
plt.xlabel("Interface atoms", size=12)
plt.ylabel("Ratio of Grappa interactions", size=12)
plt.legend(title=None, fontsize=12, title_fontsize=14)
plt.ylim(*(0, 1))
plt.xticks(size=12)
plt.yticks(size=12)
plt.show()

# %%
# Last, let's plot ratio of grappa interactions for the individual interface atoms in a barplot.
sns.barplot(df_contribution, x="atom", y="grappa_ratio", color="#927bc8", edgecolor="black", linewidth=2, err_kws={"color": "black", "linewidth": 2}, capsize= 0.2, errorbar="sd")
plt.xlabel("Interface atoms", size=12)
plt.ylabel("Ratio of Grappa interactions", size=12)
plt.ylim(*(0, 1))
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend(title=None, fontsize=12, title_fontsize=14, frameon=False)
plt.show()

























# %%
