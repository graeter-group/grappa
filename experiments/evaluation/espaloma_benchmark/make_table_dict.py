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
    'pepconf-dlc': {
        'rmse_energies': 3.59,
        'crmse_gradients': 9.13
    },
    # 'protein-torsion': {
    #     'rmse_energies': 3.59,
    #     'crmse_gradients': 9.13
    # },
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

# NOTE: CHECK THAT THE TOTAL DATASETS HAVE THIS SIZE
# ds_size_espaloma = {
#     'spice-pubchem': []
# }

with open("gaff_test_results.json", "w") as f:
    json.dump(gaff_results_espaloma, f, indent=4)

with open("ff14sb_test_results.json", "w") as f:
    json.dump(ff14sb_results_espaloma, f, indent=4)

with open("rna_of3_test_results.json", "w") as f:
    json.dump(rna_of3_results_espaloma, f, indent=4)

# %%
    
with open("results.json", "r") as f:
    grappa_results = json.load(f)

# assure consistency with espaloma results:
def check_deviation(ds, ff_type, grappa_result, espaloma_result, metric):
    grappa_metric = grappa_result['test'][ds][ff_type][metric]['mean']
    grappa_error = grappa_result['test'][ds][ff_type][metric]['std']
    espaloma_metric = espaloma_result[ds][metric]
    if not np.isclose(grappa_metric, espaloma_metric, atol=grappa_error):
        print(f'Encountered deviation from espaloma for {ds} with {ff_type} on {metric}: GRAPPA - {grappa_metric} +- {grappa_error}, Espaloma - {espaloma_metric}')

for ds in grappa_results['test'].keys():
    # GAFF-2.11 comparison
    if ds in gaff_results_espaloma.keys():
        check_deviation(ds, 'gaff-2.11', grappa_results, gaff_results_espaloma, 'rmse_energies')
        check_deviation(ds, 'gaff-2.11', grappa_results, gaff_results_espaloma, 'crmse_gradients')
    # Amber14 comparison
    if ds in ff14sb_results_espaloma.keys():
        check_deviation(ds, 'amber14', grappa_results, ff14sb_results_espaloma, 'rmse_energies')
        check_deviation(ds, 'amber14', grappa_results, ff14sb_results_espaloma, 'crmse_gradients')
    # RNA OF3 (Amber14) comparison
    if ds in rna_of3_results_espaloma.keys():
        check_deviation(ds, 'amber14', grappa_results, rna_of3_results_espaloma, 'rmse_energies')
        check_deviation(ds, 'amber14', grappa_results, rna_of3_results_espaloma, 'crmse_gradients')


#%%
# sort: small_mol - peptide - rna

boltzmann = [
    'spice-pubchem',
    'spice-des-monomers',
    'spice-dipeptide',
    'rna-diverse',
    'rna-trinucleotide',
]

opts = [
    'gen2',
    'pepconf-dlc',
]

scans = [
    'gen2-torsion',
    'protein-torsion',
]

# now make a large dictionary in the order above:
for ds_order, name in [(boltzmann, 'boltzmann'), (opts, 'opts'), (scans, 'scans')]:
    table = [
        [
            ds, 
            grappa_results[ds]['n_mols'], 
            grappa_results[ds]['n_confs'],
            {
                ff: [
                    results[ds]['rmse_energies']['mean'],
                    results[ds]['rmse_energies']['std'],
                    results[ds]['crmse_gradients']['mean'],
                    results[ds]['crmse_gradients']['std']
                ] if ds in results.keys() else [np.nan, np.nan, np.nan, np.nan]
                for ff, results in [
                    ('Grappa', grappa_results),
                    ('Espaloma', espaloma_results),
                    ('RNA.OL3', rna_of3_results_espaloma),
                    ('ff14SB', ff14sb_results_espaloma)
                ]
            }
        ] for ds in ds_order
    ]

    with open(f"table_{name}.json", "w") as f:
        json.dump(table, f, indent=4)
    
with open("readme.txt", 'w') as f:
    f.write('The json files store lists for every dataset:\n[dsname, n_mols, n_confs, "forcefield": [rmse_energies-mean, rmse_energies-std, crmse_gradients-mean, crmse_gradients-mean]]\nUnits are kcal/mol and Angstrom. crmse is the componentwise-rmse, which is smaller by a factor of sqrt(3) than the actual force-vector rmse.')
# %%

