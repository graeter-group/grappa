from __future__ import annotations

import shutil
import pytest



@pytest.mark.dataset_builder
def test_dataset_builder_tutorial_dataset(tmp_path):
    """
    Integration test: Download the dataset, build QM entries, add OpenMM nonbonded parameters,
    and ensure serialized MolData files pass validation.
    """
    pytest.importorskip("openmm", reason="DatasetBuilder requires OpenMM to obtain nonbonded data.")

    from dataset_builder_workflows import example_dataset_builder_openmm
    example_dataset_builder_openmm(tmp_path / "openmm", enforce_topology=False)


@pytest.mark.dataset_builder
def test_dataset_builder_tutorial_dataset_enforced_topology(tmp_path):
    """
    Same as tutorial dataset test, but force bonds to come from the provided topology.
    """
    pytest.importorskip("openmm", reason="DatasetBuilder requires OpenMM to obtain nonbonded data.")

    from dataset_builder_workflows import example_dataset_builder_openmm
    example_dataset_builder_openmm(tmp_path / "openmm_enforced", enforce_topology=True)

@pytest.mark.dataset_builder
def test_dataset_builder_gromacs_topology(tmp_path):
    """
    Test the dataset builder workflow with gromacs topology files for nonbonded parameters.
    """
    pytest.importorskip("openmm", reason="DatasetBuilder requires OpenMM to obtain nonbonded data.")
    gmx_executable = shutil.which("gmx")
    if gmx_executable is None:
        pytest.skip("gmx executable not found; skipping GROMACS topology integration test.")

    from dataset_builder_workflows import example_dataset_builder_gromacs

    example_dataset_builder_gromacs(tmp_path / "gromacs_relaxed", enforce_topology=False)


@pytest.mark.dataset_builder
def test_dataset_builder_gromacs_topology_enforced(tmp_path):
    """
    Test the gromacs topology workflow enforcing topology-derived bonds.
    """
    pytest.importorskip("openmm", reason="DatasetBuilder requires OpenMM to obtain nonbonded data.")
    gmx_executable = shutil.which("gmx")
    if gmx_executable is None:
        pytest.skip("gmx executable not found; skipping GROMACS topology integration test.")

    from dataset_builder_workflows import example_dataset_builder_gromacs

    example_dataset_builder_gromacs(tmp_path / "gromacs_enforced", enforce_topology=True)
