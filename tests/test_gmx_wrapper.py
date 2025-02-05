import shutil
import pytest

def is_gmx_available():
    return shutil.which("gmx") is not None



# skip if gmx is not available:
@pytest.mark.skipif(not is_gmx_available(), reason="Gromacs not available")
@pytest.mark.slow
def test_gmx_wrapper():
    from pathlib import Path
    import shutil
    import os
    import numpy as np

    gmx_dir = Path(__file__).parent / "gmx_temp_files"
    gmx_dir.mkdir(exist_ok=True, parents=True)

    pdbfile = Path(__file__).parent / "../examples/usage/T4.pdb"
    # copy the pdbfile to the gmx_dir:
    shutil.copy(pdbfile, gmx_dir)
    
    os.chdir(f"{str(gmx_dir.absolute())}")

    # parameterize the system with amber99sbildn:
    # (the 6 1 flags are to select the traditional forcefield and water model)
    os.system('printf "6\n1\n "|gmx pdb2gmx -f T4.pdb -o T4.gro -p T4.top -ignh')

    if not os.path.exists("T4.top"):
        raise FileNotFoundError("T4.top not found. Parametrization failed.")

    # Then, run grappa_gmx to create a new topology file using the grappa model
    ############################################
    os.system('grappa_gmx -f T4.top -o T4_grappa.top -t grappa-1.4.1-light')
    ############################################

    if not os.path.exists("T4_grappa.top"):
        raise FileNotFoundError("T4_grappa.top not found")


    forces_amber = get_forces_gmx("T4.gro", "T4.top")
    forces_grappa = get_forces_gmx("T4.gro", "T4_grappa.top")
    
    import matplotlib.pyplot as plt
    plt.scatter(forces_amber, forces_grappa)
    plt.savefig("forces.png")

    assert np.sqrt(np.mean((forces_amber - forces_grappa) ** 2)) < 12, f"Gradient cRMSE between amber99 and grappa-light larger than 12 kcal/mol/A: {np.sqrt(np.mean((forces_amber - forces_grappa) ** 2))}"
    assert np.max(np.abs(forces_amber - forces_grappa)) < 80, f"Max force deviation between amber99 and grappa-light larger than 80 kcal/mol/A: {np.max(np.abs(forces_amber - forces_grappa))}"

    # remove all files generated:
    os.chdir("..")
    shutil.rmtree(gmx_dir)


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