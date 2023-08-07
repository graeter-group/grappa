from openmm import unit


import os
import datetime

class Logger:
    def __init__(self, folder, print_to_screen=True):
        self.folder = str(folder)
        os.makedirs(self.folder, exist_ok=True)
        self.logfile = os.path.join(self.folder, 'log.txt')
        self.print_to_screen = print_to_screen

    def __call__(self, message:str=""):
        if self.print_to_screen:
            print(message)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.logfile, 'a') as f:
            f.write(f'[{timestamp}] {message}\n')



class CustomReporter(object):
    def __init__(self, step_interval):
        self.step_interval = step_interval
        self.potential_energies = []
        self.temperatures = []
        self.steps = []
        self.step = 0


    def describeNextReport(self, simulation):
        # We want a report at step intervals specified during initialization
        steps = self.step_interval
        # We don't need to halt simulation
        halt = False
        # We want both potential energy and temperature
        need_potential_energy = True
        need_temperature = True
        return (steps, need_potential_energy, False, False, need_temperature, halt)

    def report(self, simulation, state):
        # Get potential energy and convert it to kcal/mol
        potential_energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        self.potential_energies.append(potential_energy)

        # Get kinetic energy and convert it to kcal/mol
        kinetic_energy = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)

        # convert to kilojoules:
        kinetic_energy = unit.Quantity(kinetic_energy, unit.kilojoules / (6.02214076*1e23))

        # Compute temperature in Kelvin from equipartition theorem
        ndf = 3 * simulation.system.getNumParticles() - 3
        temperature = 2 * kinetic_energy / (unit.BOLTZMANN_CONSTANT_kB * ndf)
        self.temperatures.append(temperature.value_in_unit(unit.kelvin))

        # Get simulation step number
        self.step += self.step_interval
        self.steps.append(self.step)


    def plot(self, filename, sampling_steps=None, potential_energies=None, fontsize=16):
        import matplotlib.pyplot as plt
        import numpy as np

        steps = np.array(self.steps)

        potential_energies = np.array(potential_energies)
        temperatures = np.array(self.temperatures)

        if sampling_steps is not None:
            sampling_steps = np.array(sampling_steps)

        fig, ax = plt.subplots(2, 1, figsize=(10,10))


        ax1 = ax[0]

        ax1.plot(steps, temperatures, label="Temperature")
        ax1.set_ylabel("T [K]", fontsize=fontsize)
        ax1.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        ax1.tick_params(axis='both', which='major', labelsize=fontsize)

        # set title for both axes:
        ax1.set_title("Temperature During Sampling", fontsize=fontsize)

        if sampling_steps is not None:
            # draw vertical lines at sampling steps and give them the label "sampled":
            for i, sampling_step in enumerate(sampling_steps):
                if i == 0:
                    ax1.axvline(sampling_step, color="black", linestyle="--", label="Sampling")
                else:
                    ax1.axvline(sampling_step, color="black", linestyle="--")

        ax1.legend(fontsize=fontsize, loc="best")

        hist_ax = ax[1]

        potential_energies -= np.min(potential_energies)
        hist_ax.hist(potential_energies, bins=10, color="blue")
        hist_ax.set_xlabel("E [kcal/mol]", fontsize=fontsize)
        hist_ax.set_ylabel("count", fontsize=fontsize)
        hist_ax.tick_params(axis='both', which='major', labelsize=fontsize)
        hist_ax.set_title("Potential Energies of Sampled States", fontsize=fontsize)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()



from tqdm import tqdm
from openmm.app import StateDataReporter
from sys import stdout

import io

class ProgressReporter(StateDataReporter):
    def __init__(self, reportInterval, total_steps, **kwargs):
        self.rep_buf = io.StringIO()
        super().__init__(self.rep_buf, reportInterval, **kwargs)  # Redirect output to None
        self.buffer = io.StringIO()
        self.pbar = tqdm(total=total_steps, ncols=70, file=self.buffer)

    def report(self, simulation, state):
        super().report(simulation, state)
        self.pbar.update(self._reportInterval)
        print(self.buffer.getvalue(), end="\r")
        self.buffer.truncate(0)

    def __del__(self):
        print()
        self.pbar.close()