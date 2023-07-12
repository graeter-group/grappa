"""
Generate pdb files for training to recognize C alphas.
Requires the pepgen package.
"""
if __name__ == "__main__":

    from pathlib import Path
    from pepgen.dataset import generate
    # generate all (capped) 1 and 2 peptides, 20 5-peptides and 5 100-peptides
    for (n_max, l) in [
            (int(1e4), 1), 
            (int(1e4), 2), 
            (100, 5), 
            (5, 100),
        ]:
        generate(n_max=n_max, length=l, outpath=str(Path(f"data/pdbs/pep{l}")))