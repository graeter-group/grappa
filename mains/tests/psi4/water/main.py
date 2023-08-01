from ase import Atoms
from ase.calculators.psi4 import Psi4
from ase.md.verlet import VelocityVerlet
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.io.trajectory import Trajectory
from ase.md import MDLogger
import time

start = time.time()
# Define water molecule
water = Atoms('H2O', positions=[(0, 0, 0), (0, 0.9572, 0), (0.9261, -0.2399, 0)])

# Set calculator
memory="16GB"
num_threads=8
calc = Psi4(atoms=water, method='HF', basis='6-31G', memory=memory, num_threads=num_threads)

water.set_calculator(calc)

# Set the momenta corresponding to T=300K
MaxwellBoltzmannDistribution(water, 300 * units.kB)

# We want to run MD with constant energy using the VelocityVerlet algorithm.
dyn = VelocityVerlet(water, dt=0.5 * units.fs)  # 0.5 fs time step.

traj = Trajectory('water_md.traj', 'w', water)
logger = MDLogger(dyn, water, '-', header=True, stress=False,
          peratom=False, mode="w")
dyn.attach(logger, interval=10)
dyn.attach(traj.write, interval=10)

# Now run the dynamics
dyn.run(50)

end = time.time()
print("Time elapsed: ", end - start)