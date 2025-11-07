from __future__ import annotations

import shutil
import pytest

from dataset_builder_workflows import example_dataset_builder_gromacs, example_dataset_builder_openmm


@pytest.mark.dataset_builder
def test_dataset_builder_tutorial_dataset(tmp_path):
    """
    Integration test: Download the dataset, build QM entries, add OpenMM nonbonded parameters,
    and ensure serialized MolData files pass validation.
    """
    pytest.importorskip("openmm", reason="DatasetBuilder requires OpenMM to obtain nonbonded data.")

    example_dataset_builder_openmm(tmp_path / "openmm")

@pytest.mark.dataset_builder
def test_dataset_builder_gromacs_topology(tmp_path):
    """
    Test the dataset builder workflow with gromacs topology files for nonbonded parameters.
    """
    pytest.importorskip("openmm", reason="DatasetBuilder requires OpenMM to obtain nonbonded data.")
    gmx_executable = shutil.which("gmx")
    if gmx_executable is None:
        pytest.skip("gmx executable not found; skipping GROMACS topology integration test.")

    example_dataset_builder_gromacs(tmp_path)
