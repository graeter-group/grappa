#%%
from pathlib import Path
import pandas as pd 

csvpath = Path(__file__).parent.parent/'data'/'published_datasets.csv'

if not csvpath.exists():
    df = pd.DataFrame(columns=['tag', 'url', 'description'])
else:
    df = pd.read_csv(str(csvpath))
# %%
base_url = "https://github.com/graeter-group/grappa/releases/download/v.1.2.0/"

tags = [
    "dipeptides-300K-amber99", 
    "dipeptides-1000K-amber99", 
    "dipeptides-300K-openff-1.2.0", 
    "dipeptides-1000K-openff-1.2.0", 
    "dipeptides-300K-charmm36_nonb", 
    "dipeptides-1000K-charmm36_nonb",
    "uncapped-300K-amber99",
    "dipeptides-hyp-dop-300K-amber99", 
    "dipeptides-radical-300K", 
    "bondbreak-radical-peptides-300K",
    "uncapped-300K-openff-1.2.0",
    "spice-pubchem", 
    "rna-nucleoside", 
    "gen2", 
    "spice-des-monomers", 
    "spice-dipeptide", 
    "rna-diverse", 
    "gen2-torsion", 
    "pepconf-dlc", 
    "protein-torsion", 
    "rna-trinucleotide"
    "espaloma_split",
]

for tag in tags:
    df = df._append({'tag': tag, 'url': f"{base_url}{tag}.zip", 'description': ''}, ignore_index=True)

all_tags = df['tag'].tolist()
assert len(all_tags) == len(set(all_tags)), "Duplicate tags found"

df.to_csv(str(csvpath), index=False)
# %%
