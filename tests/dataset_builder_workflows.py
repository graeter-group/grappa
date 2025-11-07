"""
Defines two example workflows for building datasets using GROMACS and OpenMM
"""

import os
import shlex
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from grappa.data import MolData
from grappa.data.dataset_builder import DatasetBuilder

MAX_GROMACS_MOLECULES = 2
TRIPEPTIDE_ZIP_URL = "https://github.com/graeter-group/grappa/releases/download/v.1.4.1/tripeptide_example_data.zip"

_CAPPING_RENAMES = {
    ("NME", "C"): "CH3",
    ("NME", "H1"): "HH31",
    ("NME", "H2"): "HH32",
    ("NME", "H3"): "HH33",
}


def _download_tripeptide_example(download_dir: Path) -> Path:
    """
    Download and extract the example dataset into download_dir, returning the unpacked root.
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    archive_path = download_dir / "tripeptide_example_data.zip"

    if not archive_path.exists():
        with urlopen(TRIPEPTIDE_ZIP_URL) as response, open(archive_path, "wb") as file_handle:
            shutil.copyfileobj(response, file_handle)

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(download_dir)

    dataset_root = download_dir / "tripeptide_example_data"
    if not dataset_root.exists():
        raise FileNotFoundError(f"Expected '{dataset_root}' after extraction.")
    return dataset_root


def _run_gromacs_topology_build(gmx_executable: str, work_dir: Path) -> Path:
    """
    Reproduce the example command sequence (pdb2gmx + editconf) inside work_dir.
    """
    work_dir = Path(work_dir)

    # select amber99sb-ildn forcefield (6), and TIP3P water model (1)
    forcefield_selection_input = r"6\n1\n"
    
    cwd = shlex.quote(str(work_dir))
    gmx = shlex.quote(gmx_executable)

    pdb2gmx_cmd = (
        f"cd {cwd} && "
        f"printf '{forcefield_selection_input}' | "
        f"{gmx} pdb2gmx -f pep.pdb -o pep.gro -p pep.top -ignh"
    )
    if os.system(pdb2gmx_cmd) != 0:
        raise RuntimeError(f"gmx pdb2gmx failed for {work_dir.name}: {pdb2gmx_cmd}")

    editconf_cmd = f"cd {cwd} && {gmx} editconf -f pep.gro -o pep.gro -c -d 1.0"
    if os.system(editconf_cmd) != 0:
        raise RuntimeError(f"gmx editconf failed for {work_dir.name}: {editconf_cmd}")

    top_file = work_dir / "pep.top"
    if not top_file.exists():
        raise FileNotFoundError(f"GROMACS topology not written at {top_file}")
    return top_file


def _write_gromacs_ready_pdb(source: Path, target: Path) -> None:
    """
    Copy `source` PDB into `target`, renaming atoms that GROMACS expects
    for capped residues (ACE/NME) so pdb2gmx can match RTP entries.
    """
    target.parent.mkdir(parents=True, exist_ok=True)

    def _rename_line(line: str) -> str:
        if not line.startswith(("ATOM", "HETATM")):
            return line
        residue = line[17:20].strip()
        atom = line[12:16].strip()
        replacement = _CAPPING_RENAMES.get((residue, atom))
        if replacement is None:
            return line
        return f"{line[:12]}{replacement:>4}{line[16:]}"

    with open(source, "r", encoding="utf-8") as handle_in, open(target, "w", encoding="utf-8") as handle_out:
        for line in handle_in:
            handle_out.write(_rename_line(line))


def _prepare_example_inputs(
    dataset_root: Path,
    tmp_root: Path,
    gmx_executable: str,
    *,
    max_molecules: int = MAX_GROMACS_MOLECULES,
) -> tuple[Path, Path, list[str]]:
    """
    Constructing example-QM-input and example-gmx-input directories and copying files there.
    """
    qm_dir = tmp_root / "example-QM-input"
    md_dir = tmp_root / "example-gmx-input"
    qm_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    processed: list[str] = []
    molecule_dirs = sorted(path for path in dataset_root.iterdir() if (path / "qm_data.npz").exists())
    for mol_dir in molecule_dirs[:max_molecules]:
        mol_id = mol_dir.name

        # Copy QM data
        qm_target = qm_dir / mol_id
        qm_target.mkdir(parents=True, exist_ok=True)
        shutil.copy(mol_dir / "qm_data.npz", qm_target / "qm_data.npz")

        # Prepare MD / topology inputs
        md_target = md_dir / mol_id
        md_target.mkdir(parents=True, exist_ok=True)
        _write_gromacs_ready_pdb(mol_dir / "pep.pdb", md_target / "pep.pdb")
        _run_gromacs_topology_build(gmx_executable, md_target)
        processed.append(mol_id)

    return qm_dir, md_dir, processed


def example_dataset_builder_gromacs(output_dir: Path):
    """
    Download a dataset of npz files, construct gromacs .top files using pdb2gmx with the amber99sbildn force field. Then construct a dataset builder from it, and make sure that it is consistent.
    """
    output_dir = Path(output_dir)

    gmx_executable = shutil.which("gmx")
    assert gmx_executable is not None, "gmx executable not found."

    download_dir = output_dir / "download"
    dataset_root = _download_tripeptide_example(download_dir)

    qm_dir, md_dir, mol_ids = _prepare_example_inputs(
        dataset_root,
        output_dir,
        gmx_executable,
        max_molecules=MAX_GROMACS_MOLECULES,
    )

    assert mol_ids, "example preparation failed to select any molecules."
    builder = DatasetBuilder()

    # add QM data to the dataset builder from npz files
    for qm_entry in sorted(qm_dir.iterdir()):
        builder.entry_from_qm_dict_file(mol_id=qm_entry.name, filename=qm_entry / "qm_data.npz")

    # obtain nonbonded parameters from GROMACS topology files
    for top_entry in sorted(md_dir.iterdir()):
        if top_entry.suffix == ".ff":
            continue
        builder.add_nonbonded_from_gmx_top(mol_id=top_entry.name, top_file=top_entry / "pep.top", add_pdb=True)

    # builder is initialized, now just verify contents
    expected_ids = set(mol_ids)
    assert builder.complete_entries == expected_ids
    assert all(builder.entries[mol_id].pdb for mol_id in expected_ids)

    builder.remove_bonded_parameters()
    builder.filter_bad_nonbonded()

    output_dir = output_dir / "example-dataset-grappa-format"
    builder.write_to_dir(output_dir, overwrite=True, delete_dgl=True)

    moldata_files = sorted(output_dir.glob("*.npz"))
    assert len(moldata_files) == len(expected_ids)

    roundtrip = DatasetBuilder.from_moldata(output_dir)
    assert set(roundtrip.entries.keys()) == expected_ids

    sample = MolData.load(str(moldata_files[0]))
    assert sample.ff_energy["reference_ff"]["nonbonded"].shape[0] == sample.energy.shape[0]
    assert np.isfinite(sample.energy).all()


def example_dataset_builder_openmm(output_dir):
    """
    Download the dataset, build QM entries, add nonbonded parameters predicted by the OpenMM amber99sbildn forcefield,
    and ensure serialized MolData files pass validation.
    """
    download_dir = output_dir / "download"
    dataset_root = _download_tripeptide_example(download_dir)

    builder = DatasetBuilder()

    qm_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    assert qm_dirs, "example archive did not contain any molecule directories."

    for qm_dir in qm_dirs:
        builder.entry_from_qm_dict_file(mol_id=qm_dir.name, filename=qm_dir / "qm_data.npz")
        builder.add_nonbonded_from_pdb(mol_id=qm_dir.name, pdb_file=qm_dir / "pep.pdb")

    assert builder.entries, "DatasetBuilder did not register any QM entries."
    assert builder.complete_entries == set(builder.entries.keys()), "Nonbonded augmentation missing for some entries."

    builder.remove_bonded_parameters()
    builder.filter_bad_nonbonded()

    output_dir = output_dir / "example-moldata"
    builder.write_to_dir(output_dir, overwrite=True, delete_dgl=False)

    moldata_files = sorted(output_dir.glob("*.npz"))
    assert moldata_files, "DatasetBuilder did not write any MolData archives."
    assert len(moldata_files) == len(builder.complete_entries)

    roundtrip = DatasetBuilder.from_moldata(output_dir)
    assert set(roundtrip.entries.keys()) == builder.complete_entries

    sample = MolData.load(str(moldata_files[0]))
    assert sample.xyz.shape == sample.gradient.shape
    assert sample.energy.shape[0] == sample.xyz.shape[0]
    assert "reference_ff" in sample.ff_energy
    assert "nonbonded" in sample.ff_energy["reference_ff"]
    assert sample.ff_energy["reference_ff"]["nonbonded"].shape[0] == sample.energy.shape[0]
    assert np.isfinite(sample.energy).all()


if __name__ == "__main__":
    output_dir = Path("./tmp/").resolve().absolute()
    print(f"Running GROMACS and openmm example dataset builders in {output_dir}")

    print(f"Running OpenMM example in {output_dir / 'openmm'}...")
    example_dataset_builder_openmm(output_dir / "openmm")
    print(f"OpenMM example completed successfully. Output in {output_dir / 'openmm'}.")

    if shutil.which("gmx") is not None:
        print(f"Running GROMACS example in {output_dir / 'gromacs'}...")
        example_dataset_builder_gromacs(output_dir / "gromacs")
        print(f"GROMACS example completed successfully. Output in {output_dir / 'gromacs'}.")
    else:
        print("Skipping GROMACS example, gmx executable not found.")