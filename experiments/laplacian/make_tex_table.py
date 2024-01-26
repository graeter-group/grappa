#%%

ONLY_BOLTZMANN = True
WITH_STD = False

import math

def generate_tex_table(data, precision=2):

    forcefields = ['Grappa', 'Espaloma', 'Gaff-2.11', 'ff14SB', 'RNA.OL3']
    if WITH_STD:
        forcefields.append('std')

    # remove entries that are not in forcefields
    data = [[entry[0], entry[1], entry[2], {ff: entry[3][ff] for ff in forcefields if ff in entry[3]}] for entry in data]

    def format_value(value):
        if value is None or math.isnan(value):
            return ''
        else:
            return f'{{:.{precision}f}}'.format(value)

    def make_bold_if_best(values):
        valid_values = [v for v in values if v is not None and not math.isnan(v)]
        if valid_values:
            min_value = min(valid_values)
            return [f'\\textbf{{{format_value(v)}}}' if v == min_value else format_value(v) for v in values]
        else:
            return [format_value(v) for v in values]

    # LaTeX header
    tex_content = r'''\documentclass[varwidth]{standalone}
\usepackage{array}
\usepackage{multirow}
\usepackage{adjustbox}

\begin{document}

\begin{adjustbox}{width=\textwidth,center}
\centering

\begin{tabular}{l c l c c c c c c}
\hline
\hline
'''

    # tex_content += f'Dataset & Mols &  & Grappa & Espaloma & Gaff-2.11 & ff14SB & RNA.OL3 & std \\'

    tex_content += rf'''Dataset & Mols & & {" & ".join(ff for ff in forcefields)} \\
\hline
'''

    # Process each dataset
    for entry in data:
        dataset, mols, _, metrics = entry

        # Extract metrics for each force field and ensure correct format
        metrics_values = {ff: metrics[ff] if ff in metrics else [None, None] for ff in forcefields}

        energy_values = make_bold_if_best([metrics_values[ff][0] for ff in metrics_values])
        force_values = make_bold_if_best([metrics_values[ff][1] for ff in metrics_values])

        tex_content += rf'''\multirow{{2}}{{*}}{{{dataset}}} & \multirow{{2}}{{*}}{{{mols}}} & \textit{{Energy}} & {' & '.join(energy_values)} \\
                             &                    & \textit{{Force}}  & {' & '.join(force_values)} \\
\hline
'''

    # LaTeX footer
    tex_content += r'''\hline
\hline
\end{tabular}

\end{adjustbox}

\vspace{5pt}
\small
Energy and Force-component RMSE for Boltzmann-sampled conformations of unseen test molecules in [kcal/mol] and [kcal/mol/\AA{}] respectively.


\end{document}
'''
    return tex_content


#%%

import json
from pathlib import Path

with open("table_dicts.json", "r") as f:
    data= json.load(f)


if ONLY_BOLTZMANN:
    DSNAMES = {
        'spice-pubchem': 'SPICE-Pubchem',
        'spice-des-monomers': 'SPICE-DES-Monomers',
        'spice-dipeptide': 'SPICE-Dipeptide',
        'rna-diverse': 'RNA-Diverse',
        'rna-trinucleotide': 'RNA-Trinucleotide',
    }

else:
    DSNAMES = {
        'spice-pubchem': 'SPICE-Pubchem',
        'spice-des-monomers': 'SPICE-DES-Monomers',
        'gen2': 'Gen2-Opt',
        'gen2-torsion': 'Gen2-Torsion',
        'spice-dipeptide': 'SPICE-Dipeptide',
        'pepconf-dlc': 'Pepconf-Opt',
        'protein-torsion': 'Protein-Torsion',
        'rna-diverse': 'RNA-Diverse',
        'rna-trinucleotide': 'RNA-Trinucleotide',
    }

# rename dataset and remove entries that are not in DSNAMES
remove = []
for i, entry in enumerate(data):
    if entry[0] in DSNAMES:
        entry[0] = DSNAMES[entry[0]]
    else:
        remove.append(i)

for i in reversed(remove):
    data.pop(i)

# Generate TeX file content with default precision
tex_file_content = generate_tex_table(data, precision=1)


with open(Path(__file__).parent/'table.tex', 'w') as f:
    f.write(tex_file_content)
# %%
