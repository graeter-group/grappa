#%%
# PARSE ESPALOMA REPORT SUMMARY
with open("report_summary.csv", "r") as f:
    data = f.read()

# Parse the data
parsed_data = {}
lines = data.split('\n')
for i in range(0, len(lines)):
    if "(te)" in lines[i]:
        dataset = lines[i].split()[0][1:]
        energy = float(lines[i+2].split()[1].split('_')[0][1:])
        force = float(lines[i+3].split()[1].split('_')[0][1:])
        parsed_data[dataset] = {'rmse_energies': energy, 'crmse_gradients': force}

# Save the data
import json
import numpy as np

with open("espaloma_test_results.json", "w") as f:
    json.dump(parsed_data, f, indent=4)

espaloma_results = parsed_data

# %%
# gaff results:
    
gaff_results_espaloma = {
    "gen2": {
        "rmse_energies": 2.29,
        "crmse_gradients": 10.51
    },
    "pepconf-dlc": {
        "rmse_energies": 3.53,
        "crmse_gradients": 8.07
    },
    "gen2-torsion": {
        "rmse_energies": 2.53,
        "crmse_gradients": 10.5
    },
    "protein-torsion": {
        "rmse_energies": 3.53,
        "crmse_gradients": 8.07
    },
    "spice-pubchem": {
        "rmse_energies": 4.39,
        "crmse_gradients": 14.02
    },
    "spice-dipeptide": {
        "rmse_energies": 4.24,
        "crmse_gradients": 11.90
    },
    "spice-des-monomers": {
        "rmse_energies": 1.88,
        "crmse_gradients": 9.46
    },
    "rna-diverse": {
        "rmse_energies": 5.65,
        "crmse_gradients": 17.19
    },
    "rna-trinucleotide": {
        "rmse_energies": 5.79,
        "crmse_gradients": 17.15
    }
}

ff14sb_results_espaloma = {
    'spice-dipeptide': {
        'rmse_energies': 4.36,
        'crmse_gradients': 11.57
    },
}

rna_of3_results_espaloma = {
    'rna-diverse': {
        'rmse_energies': 6.06,
        'crmse_gradients': 19.38
    },
    'rna-trinucleotide': {
        'rmse_energies': 5.94,
        'crmse_gradients': 19.82
    }
}

with open("gaff_test_results.json", "w") as f:
    json.dump(gaff_results_espaloma, f, indent=4)

with open("ff14sb_test_results.json", "w") as f:
    json.dump(ff14sb_results_espaloma, f, indent=4)

with open("rna_of3_test_results.json", "w") as f:
    json.dump(rna_of3_results_espaloma, f, indent=4)

# %%
    
with open("results_grappa_hybrid.json", "r") as f:
    grappa_results = json.load(f)

ds_order = [
    'spice-pubchem',
    'spice-des-monomers',
    'gen2',
    'gen2-torsion',
    'spice-dipeptide',
    'pepconf-dlc',
    'protein-torsion',
    'rna-diverse',
    'rna-trinucleotide',
]

# now make a large dictionary in this order:

table = [
    [ds, grappa_results[ds]['n_mols'], grappa_results[ds]['n_confs'], {
        'Grappa': [grappa_results[ds]['rmse_energies'], grappa_results[ds]['crmse_gradients']],
        'Espaloma': [espaloma_results[ds]['rmse_energies'], espaloma_results[ds]['crmse_gradients']],
        'Gaff-2.11': [gaff_results_espaloma[ds]['rmse_energies'], gaff_results_espaloma[ds]['crmse_gradients']],
        'ff14SB': [ff14sb_results_espaloma[ds]['rmse_energies'], ff14sb_results_espaloma[ds]['crmse_gradients']] if ds in ff14sb_results_espaloma else [np.nan, np.nan],
        'RNA.OL3': [rna_of3_results_espaloma[ds]['rmse_energies'], rna_of3_results_espaloma[ds]['crmse_gradients']] if ds in rna_of3_results_espaloma else [np.nan, np.nan],
        'std':[grappa_results[ds]['std_energies'], grappa_results[ds]['std_gradients']/np.sqrt(3)],
    }] for ds in ds_order
]

with open("table_dicts.json", "w") as f:
    json.dump(table, f, indent=4)
# %%

