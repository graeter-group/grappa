import shutil
import pytest

def is_gmx_available():
    return shutil.which("gmx") is not None

def gmx_wrapper_pdb(pdb_path:str):
    from pathlib import Path
    import importlib.util
    import shutil
    import os
    import numpy as np

    if importlib.util.find_spec("gmx-top4py") is None:
        pytest.skip("gmx-top4py not installed; skipping Gromacs wrapper test")

    gmx_dir = Path(__file__).parent / "gmx_temp_files"
    gmx_dir.mkdir(exist_ok=True, parents=True)

    # copy the pdbfile to the gmx_dir:
    shutil.copy(pdb_path, gmx_dir)
    file_name = Path(pdb_path).name
    os.chdir(f"{str(gmx_dir.absolute())}")

    # parameterize the system with amber99sbildn:
    # (the 6 1 flags are to select the traditional forcefield and water model)
    os.system(f'printf "6\n1\n "|gmx pdb2gmx -f {file_name} -o {file_name}.gro -p {file_name}.top -ignh')

    if not os.path.exists(f"{file_name}.top"):
        raise FileNotFoundError(f"{file_name}.top not found. Parametrization failed.")

    # Then, run grappa_gmx to create a new topology file using the grappa model
    ############################################
    os.system(f'grappa_gmx -f {file_name}.top -o {file_name}_grappa.top -t grappa-1.4.1-light')
    ############################################

    if not os.path.exists(f"{file_name}_grappa.top"):
        raise FileNotFoundError(f"{file_name}_grappa.top not found")


    forces_amber = get_forces_gmx(f"{file_name}.gro", f"{file_name}.top")
    forces_grappa = get_forces_gmx(f"{file_name}.gro", f"{file_name}_grappa.top")

    import matplotlib.pyplot as plt
    plt.scatter(forces_amber, forces_grappa)
    plt.savefig("forces.png")

    assert np.sqrt(np.mean((forces_amber - forces_grappa) ** 2)) < 12, f"Gradient cRMSE between amber99 and grappa-light larger than 12 kcal/mol/A: {np.sqrt(np.mean((forces_amber - forces_grappa) ** 2))}"
    assert np.max(np.abs(forces_amber - forces_grappa)) < 80, f"Max force deviation between amber99 and grappa-light larger than 80 kcal/mol/A: {np.max(np.abs(forces_amber - forces_grappa))}"

    # remove all temporary files ending with '#'
    for f in gmx_dir.glob("*#"):
        f.unlink()


def get_forces_gmx(grofile:str, topfile:str):
    """
    Returns the forces of the entire system in the gro file with MM parameters from the top file in kcal/mol/A.
    """
    import os
    import numpy as np

    rerun_mdp = """\
    ; Rerun settings
    integrator  = md        
    nsteps      = 0        
    nstlist     = 1         
    cutoff-scheme = Verlet
    ns_type     = grid
    pbc         = xyz
    coulombtype = PME
    rcoulomb    = 1.0
    rvdw        = 1.0

    ; Ensure trajectory and force outputs are written
    nstxout     = 1         ; Output coordinates every step
    nstfout     = 1         ; Output forces every step
    """
    with open("rerun.mdp", "w") as f:
        f.write(rerun_mdp)


    # turn off warning that arises due to net charge (we just evaluate energies and forces once)
    os.system(f'gmx grompp -f rerun.mdp -c {grofile} -p {topfile} -o rerun.tpr -maxwarn 2')
    os.system(f'gmx mdrun -s rerun.tpr -rerun {grofile} -deffnm rerun')

    if not os.path.exists("rerun.edr"):
        raise FileNotFoundError("rerun.edr not found")
    
    if not os.path.exists("rerun.trr"):
        raise FileNotFoundError("rerun.trr not found")

    os.system("echo '10 0' | gmx energy -f rerun.edr -o energy.xvg")

    if not os.path.exists("energy.xvg"):
        raise FileNotFoundError("energy.xvg not found")

    os.system("echo '0' | gmx traj -s rerun.tpr -f rerun.trr -of forces.xvg")

    forces = np.loadtxt("forces.xvg", comments=['@', '#'])

    # remove all files generated:
    os.system("rm -f rerun.*")
    os.system("rm -f energy.xvg")
    os.system("rm -f forces.xvg")

    # convert forces to kcal/mol/A:
    # from kj/mol to kcal/mol:
    forces = forces / 4.184
    # from nm to A:
    forces = forces / 10.

    return forces

# skip if gmx is not available:
@pytest.mark.skipif(not is_gmx_available(), reason="Gromacs not available")
@pytest.mark.slow
def test_gmx_wrapper_monomer():
    """Test gmx wrapper by comparing Grappa and Amber gradients for a monomer."""
    from pathlib import Path
    thisdir = Path(__file__).parent
    pdb_path = str(thisdir/'testfiles/T4.pdb')
    gmx_wrapper_pdb(pdb_path)

# skip if gmx is not available:
@pytest.mark.skipif(not is_gmx_available(), reason="Gromacs not available")
@pytest.mark.slow
def test_gmx_wrapper_multimer():
    """Test gmx wrapper by comparing Grappa and Amber gradients for a multimer."""
    from pathlib import Path
    thisdir = Path(__file__).parent
    pdb_path = str(thisdir/'testfiles/two_ubqs.pdb')
    gmx_wrapper_pdb(pdb_path)