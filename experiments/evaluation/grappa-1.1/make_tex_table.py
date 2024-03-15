#%%
import math

WITH_ERR = False
LEAVE_OUT_GAFF = True

def generate_tex_table(boltzmann, scan, opt, precision=2, caption=False, with_err=False, leave_out_gaff=False):
    forcefields = ['Grappa-AM1-BCC', 'Grappa-ff99SB', 'Espaloma', 'Gaff-2.11', 'ff14SB']
    if leave_out_gaff:
        forcefields = ['Grappa-AM1-BCC', 'Grappa-ff99SB', 'Espaloma', 'ff14SB']
    section_titles = ['BOLTZMANN SAMPLED', 'TORSION SCAN', 'OPTIMIZATION']
    sections = [boltzmann, scan, opt]


    def format_value(value:float)->str:
        if value is None or math.isnan(value):
            return ''  # Return empty string for undefined or NaN values
        else:
            formatted_value = f'{value:.{precision}f}'
            # Check if formatted value is zero when non-zero value is expected
            if float(formatted_value) == 0.0 and value != 0:
                # Increase precision until a non-zero formatted value is obtained or a maximum precision is reached
                max_precision = 10  # Set a reasonable limit to prevent infinite loops
                for extra_precision in range(precision + 1, max_precision):
                    formatted_value = f'{value:.{extra_precision}f}'
                    if float(formatted_value) != 0.0:
                        break
            return formatted_value

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
'''
    if leave_out_gaff:
        tex_content += r'''\begin{tabular}{l c c l c c c c c c}
\hline
\hline
\multirow{2}{*}{Dataset} & \multirow{2}{*}{Test Mols} & \multirow{2}{*}{Confs} & & Grappa & Grappa & \multirow{2}{*}{Espaloma} & ff14SB, & Mean\\
& & & & AM1-BCC & ff99SB & & RNA.OL3 & Predictor\\
\hline
'''
    else:
        tex_content += r'''\begin{tabular}{l c c l c c c c c c c}
\hline
\hline
\multirow{2}{*}{Dataset} & \multirow{2}{*}{Test Mols} & \multirow{2}{*}{Confs} & & Grappa & Grappa & \multirow{2}{*}{Espaloma} & \multirow{2}{*}{Gaff-2.11} & ff14SB, & Mean\\
& & & & AM1-BCC & ff99SB & & & RNA.OL3 & Predictor\\
\hline
'''


    # Iterate through each section (Boltzmann, Torsion Scan, Optimization)
    for section_index, data in enumerate(sections):
        tex_content += f'\multicolumn{{8}}{{l}}{{\\vspace{{\\widthbetweentype}}}} \\\\[-1em]\n'
        tex_content += f'\multicolumn{{8}}{{l}}{{\\small{{{section_titles[section_index]}}}}} \\\\'
        tex_content += '\\hline\n'
        for entry in data:
            dataset, mols, confs, std_energies, std_forces, std_energies_err, std_forces_err, metrics = entry
            # Process metrics for each forcefield
            energy_values = [metrics[ff][0] if ff in metrics else None for ff in forcefields]
            force_values = [metrics[ff][2] if ff in metrics else None for ff in forcefields]

            energy_errs = [metrics[ff][1] if ff in metrics else None for ff in forcefields]
            force_errs = [metrics[ff][3] if ff in metrics else None for ff in forcefields]

            energy_values.append(std_energies)
            force_values.append(std_forces)

            energy_errs.append(std_energies_err)
            force_errs.append(std_forces_err)

            bold_energy_values = make_bold_if_best(energy_values)
            bold_force_values = make_bold_if_best(force_values)

            energy_errs = [format_value(v) for v in energy_errs]
            force_errs = [format_value(v) for v in force_errs]

            if with_err:
                bold_energy_values = [f'{v} $\pm$ {e}' if v!='' else '' for v, e in zip(bold_energy_values, energy_errs)]
                bold_force_values = [f'{v} $\pm$ {e}' if v!='' else '' for v, e in zip(bold_force_values, force_errs)]

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
    'dipeptide_rad': 'Radical-Dipeptide',
    'hyp-dop': 'HYP-DOP-Dipeptide',
}



# rename dataset
for data in [boltzmann, scans, opts]:
    for i, entry in enumerate(data):
        entry[0] = DSNAMES[entry[0]]

#%%

# Generate TeX file content with default precision
tex_file_content = generate_tex_table(boltzmann, scans, opts, precision=1, caption=False, with_err=WITH_ERR, leave_out_gaff=LEAVE_OUT_GAFF)


with open(Path(__file__).parent/'table.tex', 'w') as f:
    f.write(tex_file_content)
# %%
