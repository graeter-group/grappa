#%%
# PARSE ESPALOMA REPORT SUMMARY
with open("report_summary.csv", "r") as f:
    data = f.read()

# Parse the data
parsed_data = {}
lines = data.split('\n')
for i in range(len(lines)):
    line_parts = lines[i].split()
    # Check if line has enough parts and contains dataset category
    if len(line_parts) > 1:
        if "(te)" in line_parts[1]:

            dataset = line_parts[0].strip('>').strip('<')

            energy_values = lines[i+2].split()[1].replace('$', '').replace('{', '').replace('}', '').split('_')
            force_values = lines[i+3].split()[1].replace('$', '').replace('{', '').replace('}', '').split('_')
            
            energy_mean = float(energy_values[0])
            energy_lower = float(energy_values[1].split('^')[0])
            energy_upper = float(energy_values[1].split('^')[1])
            force_mean = float(force_values[0])
            force_lower = float(force_values[1].split('^')[0])
            force_upper = float(force_values[1].split('^')[1])

            
            # Assuming std is approximated by the difference between the mean and lower bound (or upper bound)
            energy_std = (energy_upper - energy_lower) / 2
            force_std = (force_upper - force_lower) / 2
            
            parsed_data[dataset] = {
                'rmse_energies': {'mean': energy_mean, 'std': energy_std},
                'crmse_gradients': {'mean': force_mean, 'std': force_std}
            }


# Save the data
import json
import numpy as np

with open("espaloma_test_results.json", "w") as f:
    json.dump(parsed_data, f, indent=4)

espaloma_results = parsed_data

print(json.dumps(espaloma_results, indent=4))

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

with open("ds_size.json", "r") as f:
    ds_size = json.load(f)


# now make a large dictionary in the order above:
for ds_order, name in [(boltzmann, 'boltzmann'), (opts, 'opts'), (scans, 'scans')]:
    table = [
        [
            ds, 
            grappa_results['test'][ds]['n_mols'],
            grappa_results['test'][ds]['n_confs'],
            grappa_results['test'][ds]['std_energies']['mean'],
            grappa_results['test'][ds]['std_gradients']['mean']/np.sqrt(3), # rescale to component-wise std deviation
            grappa_results['test'][ds]['std_energies']['std'],
            grappa_results['test'][ds]['std_gradients']['std']/np.sqrt(3), # rescale to component-wise std deviation
            {
                ff: [
                    results['rmse_energies']['mean'],
                    results['rmse_energies']['std'],
                    results['crmse_gradients']['mean'],
                    results['crmse_gradients']['std']
                ] if not results is None else [np.nan, np.nan, np.nan, np.nan]
                for ff, results in [
                    ('Grappa', grappa_results['test'][ds]['grappa']),
                    ('Espaloma', espaloma_results[ds] if ds in espaloma_results.keys() else None),
                    ('Gaff-2.11', grappa_results['test'][ds]['gaff-2.11']),
                    ('RNA.OL3', grappa_results['test'][ds]['amber14'] if ds in ['rna-diverse', 'rna-trinucleotide'] else None),
                    ('ff14SB', grappa_results['test'][ds]['amber14'] if 'amber14' in grappa_results['test'][ds].keys() else None),
                ]
            }
        ] for ds in ds_order
    ]

    with open(f"table_{name}.json", "w") as f:
        json.dump(table, f, indent=4)
    
with open("readme.txt", 'w') as f:
    f.write('The json files store lists for every dataset:\n[dsname, n_mols, n_confs, std_energies, std_forces, std_energies_std, std_forces_std, "forcefield": [rmse_energies-mean, rmse_energies-std, crmse_gradients-mean, crmse_gradients-mean]]\nUnits are kcal/mol and Angstrom. crmse is the componentwise-rmse, which is smaller by a factor of sqrt(3) than the actual force-vector rmse.')
# %%

