#%%
# from PDBData.xyz2res.constants import RESIDUES
RESIDUES = ['ACE', 'NME', 'CYS', 'ASP', 'SER', 'GLN', 'LYS', 'ILE', 'PRO', 'THR', 'PHE', 'ASN', 'GLY', 'HIS', 'LEU', 'ARG', 'TRP', 'ALA', 'VAL', 'GLU', 'TYR', 'MET', "HIE", "HID", "HIP", "HYP", "DOP"]

from grappa.constants import ONELETTER

from pathlib import Path

MAX_ELEMENT = 26

# from grappa.constants import MAX_ELEMENT

ELEMENTS = {
    -1: "Radical",
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I"
}



import torch
from grappa.training.utilities import get_grad
from grappa.training.grappa_training import GrappaTrain
from grappa.run import run_utils
from grappa.models import get_models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
#%%

def evaluate(loaders, loader_names, model, device="cpu", plot=False, plot_folder=None, metrics=True, on_forces=True, verbose=True, ref_ff="ref", rmse_plots=True):
    """
    Evaluates the model on the given loaders.
    """
    
    if verbose:
        print("Evaluating model...")
    model.eval()
    eval_data = {}
    rmse_data = {}
    if metrics:
        if verbose:
            print("  Calculating metrics...")
        for name in loader_names:
            eval_data[name] = {}
            rmse_data[name] = {}
        if len(loaders) != len(loader_names):
            raise ValueError(f"number of loaders and loader_names must match, but are {len(loaders)} and {len(loader_names)}")

        for loader, name in zip(loaders, loader_names):
            if len(loader) == 0:
                if verbose:
                    print(f"    Skipping {name} because it is empty")
                continue
            if verbose:
                print(f"    Evaluating {name}...")

            en_total_ae = 0
            en_total_se = 0
            g_total_ae = 0
            g_total_se = 0
            ff_g_total_ae = 0
            ff_g_total_se = 0
            ff_en_total_ae = 0
            ff_en_total_se = 0


            res_nums = {res_id:0 for res_id,_ in enumerate(RESIDUES)}
            atom_nums = {el:0 for el in range(MAX_ELEMENT)}
            radical_num = 0

            res_se = {res_id:0 for res_id,_ in enumerate(RESIDUES)}
            atom_se = {el:0 for el in range(MAX_ELEMENT)}
            radical_se = 0

            ff_res_se = {res_id:0 for res_id,_ in enumerate(RESIDUES)}
            ff_atom_se = {el:0 for el in range(MAX_ELEMENT)}
            ff_radical_se = 0

            num_energies = 0
            num_grad_components = 0
            on_f = on_forces and "grad_ref" in next(iter(loader)).nodes["n1"].data.keys()
            for i, g in enumerate(loader):

                grads, model, g = get_grad(model=model, batch=g, device=device)
                energies = (g.nodes["g"].data["u"] + g.nodes["g"].data["u_nonbonded_ref"]).flatten()
                en_ref = g.nodes["g"].data["u_qm"].flatten()
                
                grad_ff = None
                if not ref_ff is None:
                    if on_f:
                        grad_ff = g.nodes["n1"].data[f"grad_total_{ref_ff}"]

                    en_ff = g.nodes["g"].data[f"u_total_{ref_ff}"].flatten()


                if on_f:
                    grad_ref = g.nodes["n1"].data["grad_ref"]

                with torch.no_grad():
                    # subtract means, only the difference is physical
                    energies -= energies.mean(dim=-1)
                    en_ref -= en_ref.mean(dim=-1)
                    if not ref_ff is None:
                        en_ref_ff = en_ff - en_ff.mean(dim=-1)

                    # add to total error
                    en_total_ae += torch.abs(energies - en_ref).sum()
                    en_total_se += torch.square(energies - en_ref).sum()
                    if not ref_ff is None:
                        ff_en_total_ae += torch.abs(en_ref_ff-en_ref).sum()
                        ff_en_total_se += torch.square(en_ref_ff-en_ref).sum()

                    if on_f:

                        residues = torch.argmax(g.nodes["n1"].data["residue"], dim=1)
                        for res_id, _ in enumerate(RESIDUES):
                            res_indices = torch.argwhere(residues == res_id)[:,0]
                            res_se[res_id] += torch.square(grads[res_indices] - grad_ref[res_indices]).sum()
                            if not grad_ff is None:
                                ff_res_se[res_id] += torch.square(grad_ff[res_indices] - grad_ref[res_indices]).sum()

                            num_components = len(grads[res_indices].flatten())
                            res_nums[res_id] += num_components


                        onehot_radicals = g.nodes["n1"].data["is_radical"].int()
                        radical_indices = torch.argwhere(onehot_radicals == 1)[:,0]
                        radical_se += torch.square(grads[radical_indices] - grad_ref[radical_indices]).sum()
                        if not grad_ff is None:
                            ff_radical_se += torch.square(grad_ff[radical_indices] - grad_ref[radical_indices]).sum()

                        num_components = len(grads[radical_indices].flatten())
                        radical_num += num_components


                        atom_numbers = torch.argmax(g.nodes["n1"].data["atomic_number"], dim=1)

                        for atom_number in range(MAX_ELEMENT):
                            atom_mask = (atom_numbers == atom_number)*(onehot_radicals == 0)
                            atom_ids = torch.argwhere(atom_mask)[:,0]
                            atom_se[atom_number] += torch.square(grads[atom_ids] - grad_ref[atom_ids]).sum()
                            if not grad_ff is None:
                                ff_atom_se[atom_number] += torch.square(grad_ff[atom_ids] - grad_ref[atom_ids]).sum()

                            num_components = len(grads[atom_ids].flatten())
                            atom_nums[atom_number] += num_components


                        grads = grads.flatten()
                        grad_ref = grad_ref.flatten()
                        g_total_ae += torch.abs(grads - grad_ref).sum()
                        g_total_se += torch.square(grads - grad_ref).sum()
                       
                        if not grad_ff is None:
                            grad_ff = grad_ff.flatten()
                            ff_g_total_ae += torch.abs(grad_ff - grad_ref).sum()
                            ff_g_total_se += torch.square(grad_ff - grad_ref).sum()


                        num_grad_components += len(grads)

                    num_energies += len(energies)

                # loader loop end


            with torch.no_grad():
                if on_f and num_grad_components != 0: # hacky but should work since either all or none are None

                    rmse_data[name]["radical_grad_rmse"] = torch.sqrt(radical_se/radical_num).item() if radical_num > 0 else float("nan")
                    
                    rmse_data[name][f"res_grad_rmse"] = {res: torch.sqrt(res_se[res_id]/res_nums[res_id]).item() if res_nums[res_id] > 0 else float("nan") for res_id, res in enumerate(RESIDUES)}
                    rmse_data[name][f"atom_grad_rmse"] = {atom_number: torch.sqrt(atom_se[atom_number]/atom_nums[atom_number]).item() if atom_nums[atom_number] > 0 else float("nan") for atom_number in range(MAX_ELEMENT)}


                    eval_data[name]["grad_mae"] = (g_total_ae/num_grad_components).item()
                    eval_data[name]["grad_rmse"] = torch.sqrt(g_total_se/num_grad_components).item()


                if num_energies != 0:
                    eval_data[name]["energy_mae"] = (en_total_ae/num_energies).item()
                    eval_data[name]["energy_rmse"] = torch.sqrt(en_total_se/num_energies).item()


                    if not ref_ff is None:
                        rmse_data[name][f"{ref_ff}_radical_grad_rmse"] = torch.sqrt(ff_radical_se/radical_num).item() if radical_num > 0 else float("nan")
                        rmse_data[name][f"{ref_ff}_res_grad_rmse"] = {res: torch.sqrt(ff_res_se[res_id]/res_nums[res_id]).item() if res_nums[res_id] > 0 else float("nan") for res_id, res in enumerate(RESIDUES)}
                        rmse_data[name][f"{ref_ff}_atom_grad_rmse"] = {atom_number: torch.sqrt(ff_atom_se[atom_number]/atom_nums[atom_number]).item() if atom_nums[atom_number] > 0 else float("nan") for atom_number in range(MAX_ELEMENT)}

                        eval_data[name][f"{ref_ff}_grad_mae"] = (ff_g_total_ae/num_grad_components).item()
                        eval_data[name][f"{ref_ff}_grad_rmse"] = torch.sqrt(ff_g_total_se/num_grad_components).item()

                if num_energies > 0 and not ref_ff is None:
                    eval_data[name][f"{ref_ff}_energy_mae"] = (ff_en_total_ae/num_energies).item()
                    eval_data[name][f"{ref_ff}_energy_rmse"] = torch.sqrt(ff_en_total_se/num_energies).item()


    if rmse_plots and on_forces:
        # plot gradients rmse differentiating between residue types, and atom types
        if verbose:
            print("  Doing rmse plots...")
        if plot_folder is None:
            plot_folder = os.getcwd()
        if plot:
            subfolder = os.path.join(plot_folder, "rmse_plots")
        else:
            subfolder = plot_folder

        os.makedirs(subfolder, exist_ok=True)
        # do a histogram plot for the different residues:


        def compare_residue_lvl(loader_name, ref_ff=None):
            residue_data_ref = rmse_data[loader_name][f"{ref_ff}_res_grad_rmse"]
            residue_data = rmse_data[loader_name][f"res_grad_rmse"]
            total_rmse_ref = eval_data[loader_name][f"{ref_ff}_grad_rmse"]
            total_rmse = eval_data[loader_name][f"grad_rmse"]

            # Convert to single letter code:
            residue_data_ref = {ONELETTER[residue]: rmse for residue, rmse in residue_data_ref.items() if not np.isnan(rmse)}
            residue_data = {ONELETTER[residue]: rmse for residue, rmse in residue_data.items() if not np.isnan(rmse)}

            # Ensure same order and entries for both data sets
            residues = list(set(residue_data.keys()).union(set(residue_data_ref.keys())))
            residues.sort()  # optional, for better visualization
            ref_rmse = [residue_data_ref.get(residue, np.nan) for residue in residues]
            result_rmse = [residue_data.get(residue, np.nan) for residue in residues]

            # Create bar plot with grouped bars
            width = 0.35

            # Add total rmse to end of lists
            residues.append('Total')
            result_rmse.append(total_rmse)
            ref_rmse.append(total_rmse_ref)

            x = np.arange(len(residues))

            fig, ax = plt.subplots(figsize=(10,6))
            ax.bar(x - width/2, result_rmse, width, label="model")
            ax.bar(x + width/2, ref_rmse, width, label=ref_ff)

            ax.set_xlabel('Residue')
            ax.set_ylabel('Gradient RMSE [kcal/mol/Å]')
            ax.set_title('Gradient RMSE for Different Residues')
            ax.set_xticks(x)
            ax.set_xticklabels(residues)
            ax.legend()

            plot_loc = os.path.join(subfolder, f"comparision_{loader_name}_{ref_ff}_residue_rmse.png")

            os.makedirs(str(Path(plot_loc).parent), exist_ok=True)
            plt.savefig(plot_loc)

            plt.close()


        
        def plot_residues(loader_name, ref_ff=None):

            prefix = ""
            if not ref_ff is None:
                prefix = f"{ref_ff}_"
            residue_data = rmse_data[loader_name][f"{prefix}res_grad_rmse"]
            total_rmse = eval_data[loader_name][f"{prefix}grad_rmse"]

            # Convert dictionary to pandas DataFrame
            df = pd.DataFrame(list(residue_data.items()), columns=['Residue', 'Gradient RMSE'])
            
            # convert to single letter code:
            df["Residue"] = df["Residue"].apply(lambda x: ONELETTER[x])

            # add total rmse, naming it "total":
            df.loc[len(df.index)] = ['Total', total_rmse]

            # Drop NaN values
            df = df.dropna()

            # Adjust font sizes
            plt.rcParams.update({'font.size': 14})

            # Plot
            plt.figure(figsize=(10,6))  # optional, to adjust figure size
            plt.bar(df['Residue'], df['Gradient RMSE'], color='skyblue')
            plt.xlabel('Residue')
            plt.ylabel('Gradient RMSE [kcal/mol/Å]')
            plt.title('Gradient RMSE for Different Residues')
            if not ref_ff is None:
                plt.title(f'Gradient RMSE for Different Residues ({ref_ff})')
            # plt.show() # uncomment for debugging

            # os.makedirs(subfolder, exist_ok=True)
            # if not ref_ff is None:
            #     plt.savefig(os.path.join(subfolder, f"{loader_name}_{ref_ff}_residue_rmse.png"))
            # else:
            #     plt.savefig(os.path.join(subfolder, f"{loader_name}_residue_rmse.png"))

            if not ref_ff is None:
                plot_loc = os.path.join(subfolder, f"{loader_name}_{ref_ff}_residue_rmse.png")
            else:
                plot_loc = os.path.join(subfolder, f"{loader_name}_residue_rmse.png")

            os.makedirs(str(Path(plot_loc).parent), exist_ok=True)
            plt.savefig(plot_loc)

            plt.close()

            # do the next plot without the ref_ff
            if not ref_ff is None:
                compare_residue_lvl(loader_name=loader_name, ref_ff=ref_ff)
                return plot_residues(loader_name=loader_name, ref_ff=None)


        def compare_atom_level(loader_name, ref_ff=None):
            atom_data_ref = rmse_data[loader_name][f"{ref_ff}_atom_grad_rmse"]
            atom_data = rmse_data[loader_name][f"atom_grad_rmse"]
            total_rmse_ref = eval_data[loader_name][f"{ref_ff}_grad_rmse"]
            total_rmse = eval_data[loader_name][f"grad_rmse"]
            radical_rmse_ref = rmse_data[loader_name][f"{ref_ff}_radical_grad_rmse"]
            radical_rmse = rmse_data[loader_name][f"radical_grad_rmse"]

            # Convert atom number to element and filter for nans:
            atom_data_ref = {ELEMENTS[atom_number]: rmse for atom_number, rmse in atom_data_ref.items() if atom_number in ELEMENTS.keys() and not np.isnan(rmse)}
            atom_data = {ELEMENTS[atom_number]: rmse for atom_number, rmse in atom_data.items() if atom_number in ELEMENTS.keys() and not np.isnan(rmse)}

            # Ensure same order and entries for both data sets
            elements = list(set(atom_data.keys()).union(set(atom_data_ref.keys())))
            ref_rmse = [atom_data_ref.get(element, np.nan) for element in elements]
            result_rmse = [atom_data.get(element, np.nan) for element in elements]

            # Create bar plot with grouped bars
            width = 0.35

            # Add total rmse and radical rmse to end of lists
            elements.extend(['Radical', 'Total'])
            result_rmse.extend([radical_rmse, total_rmse])
            ref_rmse.extend([radical_rmse_ref, total_rmse_ref])

            x = np.arange(len(elements))

            fig, ax = plt.subplots(figsize=(10,6))
            ax.bar(x - width/2, result_rmse, width, label="model")
            ax.bar(x + width/2, ref_rmse, width, label=ref_ff)

            ax.set_xlabel('Element')
            ax.set_ylabel('Gradient RMSE [kcal/mol/Å]')
            ax.set_title('Gradient RMSE for Different Elements')
            ax.set_xticks(x)
            ax.set_xticklabels(elements)
            ax.legend()

            plot_loc = os.path.join(subfolder, f"comparision_{loader_name}_{ref_ff}_element_rmse.png")
            os.makedirs(str(Path(plot_loc).parent), exist_ok=True)
            plt.savefig(plot_loc)
            plt.close()

        
        def plot_atom_numbers(loader_name, ref_ff=None):
            prefix = ""
            if not ref_ff is None:
                prefix = f"{ref_ff}_"

            atom_data=rmse_data[name][f"{prefix}atom_grad_rmse"]
            radical_rmse=rmse_data[name][f"{prefix}radical_grad_rmse"]
            total_rmse=eval_data[name][f"{prefix}grad_rmse"]


            #first convert atom number to element and filter for nans:
            atom_data_ = {ELEMENTS[atom_number]: rmse for atom_number, rmse in atom_data.items() if atom_number in ELEMENTS.keys() and not np.isnan(rmse)}

            # Convert dictionary to pandas DataFrame
            df = pd.DataFrame(list(atom_data_.items()), columns=['Element', 'Gradient RMSE'])

            # add radical rmse, naming it "radical":
            df.loc[len(df.index)] = ['Radical', radical_rmse]
            # add total rmse, naming it "total":
            df.loc[len(df.index)] = ['Total', total_rmse]
            
            # Drop NaN values
            df = df.dropna()

            # Adjust font sizes
            plt.rcParams.update({'font.size': 14})

            # Plot
            plt.figure(figsize=(10,6))
            plt.bar(df['Element'], df['Gradient RMSE'], color='skyblue')
            plt.xlabel('Element')
            plt.ylabel('Gradient RMSE [kcal/mol/Å]')
            if ref_ff is not None:
                plt.title(f'Gradient RMSE for Different Elements ({ref_ff})')
            else:
                plt.title('Gradient RMSE for Different Elements')

            if ref_ff is not None:
                plot_loc = os.path.join(subfolder, f"{loader_name}_{ref_ff}_element_rmse.png")
            else:
                plot_loc = os.path.join(subfolder, f"{loader_name}_element_rmse.png")
            os.makedirs(str(Path(plot_loc).parent), exist_ok=True)
            plt.savefig(plot_loc)
            plt.close()

            if not ref_ff is None:
                compare_atom_level(loader_name=loader_name, ref_ff=ref_ff)
                return plot_atom_numbers(loader_name=loader_name, ref_ff=None)
        

        for loader, name in zip(loaders, loader_names):
            if len(loader) == 0:
                continue
            plot_residues(loader_name=name, ref_ff=ref_ff)
            plot_atom_numbers(loader_name=name, ref_ff=ref_ff)



    if plot:
        if verbose:
            print("  Plotting...")
        GrappaTrain.compare_all(model=model, loaders=loaders, dataset_names=loader_names, device=device, energies=["ref", "reference_ff"], forcefield_name=ref_ff, folder_path=plot_folder, grads=on_forces, verbose=verbose)


    return {"eval_data":eval_data, "rmse_data": rmse_data}














#%%

if __name__=="__main__":
    import dgl
    n_graphs = 3
    ds_paths = "/hits/fast/mbm/seutelf/data/datasets/PDBDatasets/AA_opt_rad/heavy_60_dgl.bin"
    ds = dgl.data.utils.load_graphs(ds_paths)[0]
    ds = ds[:n_graphs]
    loader = dgl.dataloading.GraphDataLoader(ds)
    #%%
    # build a model from a config file:
    
    from grappa.models.deploy import model_from_version
    version = "/hits/fast/mbm/seutelf/grappa/mains/runs/rad/versions/8_rad_col_eric"
    
    model = model_from_version(version=version, device="cuda")

    #%%
    d = evaluate([loader], ["test"], model=model, device="cuda", plot=False, plot_folder=None, metrics=True, on_forces=True, verbose=True, ref_ff="amber99sbildn", rmse_plots=True)
    # %%
    d["rmse_data"]["test"]["res_grad_rmse"]
    #%%
    d["rmse_data"]["test"]["atom_grad_rmse"]
    #%%
    d["rmse_data"]["test"]["radical_grad_rmse"]
    #%%
    d["eval_data"]["test"]["grad_rmse"]
    #%%
    d["rmse_data"]["test"]["amber99sbildn_radical_grad_rmse"]
    #%%
    d["eval_data"]["test"]["amber99sbildn_grad_rmse"]

# %%
