#%%
from grappa.PDBData.xyz2res.constants import RESIDUES

store = True

from grappa.PDBData.PDBDataset import PDBDataset
from grappa.PDBData.PDBMolecule import PDBMolecule
from openmm.unit import kilocalorie_per_mole, angstrom
from pathlib import Path
from grappa.ff_utils.charge_models.charge_models import model_from_dict
import copy
import os
import shutil
import json

from pathlib import Path
dpath = "/hits/fast/mbm/share/datasets_Eric"
storepath = "/hits/fast/mbm/seutelf/data/datasets/PDBDatasets"
#%%

overwrite = True
n_max = None

MAX_ENERGY = 200
MAX_FORCE = 400


for pathname in [
                # "AA_opt_nat",
                "AA_scan_nat",
                # "AA_opt_rad",
                # "AA_scan_rad",
                ]:
    print(f"Starting {pathname}...")
    counter = 0
    expected_fails = {}
    unexpected_fails = {}
    bond_check_fails = {}

    ds = PDBDataset([])

    if overwrite:
        if os.path.exists(str(Path(storepath)/Path(pathname)/Path("base"))):
            shutil.rmtree(str(Path(storepath)/Path(pathname))/Path("base"))

    for p in (Path(dpath)/Path(pathname)).rglob("*.log"):
        ###########################################################
        # INIT
        try:
            mol = PDBMolecule.from_gaussian_log_rad(p, e_unit=kilocalorie_per_mole*23.0609, dist_unit=angstrom, force_unit=kilocalorie_per_mole*23.0609/angstrom)
            mol.name = p.stem
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if not any([skip_res in str(p) for skip_res in ["Glu", "Asp", "Ile", "Trp", "NmeCH3", "Nme", "Hie", "AceCH3", "Nala"]]):
                # unexpected fail!
                unexpected_fails[str(p)] = str(e)
            else:
                # expected fail
                expected_fails[str(p)] = str(e)
            continue

        # filter out extreme conformations with energies above 200 kcal/mol away from the minimum
        mol.filter_confs(max_energy=MAX_ENERGY, max_force=MAX_FORCE)

        ###########################################################
        # BOND CHECK
        try:
            mol.bond_check()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            bond_check_fails[str(p)] = str(e)
            continue
        ###########################################################
        # PARAMETRIZE CHECK
        try:
            mol_copy = copy.deepcopy(mol)
            mol_copy.parametrize(allow_radicals=True, get_charges=model_from_dict(tag="heavy"), collagen=True)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Failed to parametrize {p}:\n  e: {e}")
            print()
            continue

        if mol.energies is None:
            print(f"No energies present for {p}.")
            print()
            continue

        if mol.gradients is None:
            print(f"No energies present for {p}.")
            print()
            continue

        ds.append(mol)
        counter += 1
        if not n_max is None:
            if counter > n_max:
                break

    print(f"Finished {pathname} with {counter} molecules.")
    print()
    print(f"expected fails:\n{json.dumps(expected_fails, indent=4)}\n")
    print(f"bond check fails:\n{json.dumps(bond_check_fails, indent=4)}\n")
    print(f"unexpected fails:\n{json.dumps(unexpected_fails, indent=4)}\n\n")
    print()
    print(f"  {len(expected_fails)} expected fails.")
    print(f"  {len(unexpected_fails)} unexpected fails.")
    print(f"  {len(bond_check_fails)} bond check fails.")

    residue_count = {res:0 for res in RESIDUES}
    # count how often which residue appeared:
    for mol in ds:
        for res in RESIDUES:
            if res in mol.name.upper():
                residue_count[res] += 1

    # visualize the residue counts in the console using # signs normalized to 20 for the most common residue
    max_count = max(residue_count.values())
    print("Residue counts:")
    for res, count in residue_count.items():
        print(f"  {res}: {'#'*int(count/(max_count+1)*20)} {count} ")

    print()

    if store:
        ds.save_npz(Path(storepath)/Path(pathname)/Path("base"), overwrite=True)
        print(f"Stored {len(ds)} molecules for {pathname}.")
# %%

