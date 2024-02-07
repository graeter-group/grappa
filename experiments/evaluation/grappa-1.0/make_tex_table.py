#%%
import math

def generate_tex_table(boltzmann, scan, opt, precision=2, caption=False):
    forcefields = ['Grappa', 'Espaloma', 'Gaff-2.11', 'ff14SB']
    section_titles = ['BOLTZMANN SAMPLED', 'TORSION SCAN', 'OPTIMIZATION']
    sections = [boltzmann, scan, opt]

    def format_value(value):
        if value is None or math.isnan(value if value is not None else 0):
            return ''  # Return empty string for undefined or NaN values
        else:
            return f'{value:.{precision}f}'

    def make_bold_if_best(values):
        valid_values = [v if v is not None and not math.isnan(v if v is not None else 0) else float('inf') for v in values]
        if valid_values and not all(v == float('inf') for v in valid_values):  # Check if not all values are 'inf'
            min_value = min(valid_values)
            return [f'\\textbf{{{format_value(v)}}}' if format_value(v) == format_value(min_value) and v != float('inf') else format_value(v) for v in values]
        else:
            return ['' for _ in values]  # Return list of empty strings if all values are invalid

    tex_content = r'''\documentclass[varwidth]{standalone}
\usepackage{adjustbox}
\usepackage{array}
\usepackage{multirow}
\usepackage{caption} % Include the caption package

% Define variables for vertical spacing
\newcommand{\widthbetweentype}{7pt}

\begin{document}

\begin{adjustbox}{width=\textwidth,center}
\centering

\renewcommand{\arraystretch}{1.0} % Adjust the factor as needed for more or less space

\begin{tabular}{l c c l c c c c c}

\hline
\hline
Test Dataset & Mols & Confs & & Grappa & Espaloma & Gaff-2.11 & ff14SB/RNA.OL3 \\
\hline
'''

    # Iterate through each section (Boltzmann, Torsion Scan, Optimization)
    for section_index, data in enumerate(sections):
        tex_content += f'\multicolumn{{8}}{{l}}{{\\vspace{{\\widthbetweentype}}}} \\\\[-1em]\n'
        tex_content += f'\multicolumn{{8}}{{l}}{{\\small{{{section_titles[section_index]}}}}} \\\\'
        tex_content += '\\hline\n'
        for entry in data:
            dataset, mols, confs, metrics = entry
            # Process metrics for each forcefield
            energy_values = [metrics[ff][0] if ff in metrics else None for ff in forcefields]
            force_values = [metrics[ff][2] if ff in metrics else None for ff in forcefields]

            energy_errs = [metrics[ff][1] if ff in metrics else None for ff in forcefields]
            force_errs = [metrics[ff][3] if ff in metrics else None for ff in forcefields]

            bold_energy_values = make_bold_if_best(energy_values)
            bold_force_values = make_bold_if_best(force_values)

            # Add dataset rows to the LaTeX content
            tex_content += f"\\multirow{{2}}{{*}}{{{dataset}}} & \\multirow{{2}}{{*}}{{{mols}}} & \\multirow{{2}}{{*}}{{{confs}}} & \\textit{{Energy}} & " + ' & '.join(bold_energy_values) + "\\\\\n"
            tex_content += f"                                   &                       &                         & \\textit{{Force}}  & " + ' & '.join(bold_force_values) + "\\\\\n\\hline\n"

    tex_content += r'''\hline
\hline
\end{tabular}

\end{adjustbox}
'''
    if caption:
        tex_content += r'''
\vspace{5pt}
\small
Energy and force-component RMSE for unseen test molecules in [kcal/mol] and [kcal/mol/\AA{}] respectively.
'''
    tex_content += r'''
\end{document}
'''


    return tex_content


#%%

import json
from pathlib import Path

with open("table_boltzmann.json", "r") as f:
    boltzmann = json.load(f)

with open("table_scans.json", "r") as f:
    scans = json.load(f)

with open("table_opts.json", "r") as f:
    opts = json.load(f)


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
    'spice-dipeptide_amber99sbildn': 'SPICE-Dipeptide-Amber99',
    'pepconf-dlc_amber99sbildn': 'Pepconf-Opt-Amber99',
    'protein-torsion_amber99sbildn': 'Protein-Torsion-Amber99',
    'tripeptides_amber99sbildn': 'Tripeptides-Amber99',
    'dipeptide_rad': 'Dipeptide-Radical',
}



# rename dataset
for data in [boltzmann, scans, opts]:
    for i, entry in enumerate(data):
        entry[0] = DSNAMES[entry[0]]

#%%

# Generate TeX file content with default precision
tex_file_content = generate_tex_table(boltzmann, scans, opts, precision=1, caption=False)


with open(Path(__file__).parent/'table.tex', 'w') as f:
    f.write(tex_file_content)
# %%
