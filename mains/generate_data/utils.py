from openmm import unit

class CustomReporter(object):
    def __init__(self, step_interval=1):
        self.step_interval = step_interval
        self.potential_energies = []
        self.temperatures = []
        self.steps = []

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
        
        # Get temperature and convert it to Kelvin
        temperature = state.getTemperature().value_in_unit(unit.kelvin)
        self.temperatures.append(temperature)

        # Get simulation step number
        step = state.getStep()
        self.steps.append(step)


    def plot(self, filename, sampling_steps=None, fontsize=16):
        import matplotlib.pyplot as plt
        import numpy as np

        steps = np.array(self.steps)
        potential_energies = np.array(self.potential_energies)
        temperatures = np.array(self.temperatures)

        if sampling_steps is not None:
            sampling_steps = np.array(sampling_steps)

        fig, ax = plt.subplots(2, 1, figsize=(10,10))

        plt.title("Potential energy and temperature during sampling simulation")

        ax1 = ax[0]
        ax1.plot(steps, potential_energies, color="blue")
        ax1.set_xlabel("step", fontsize=fontsize)
        ax1.set_ylabel("potential energy [kcal/mol]", fontsize=fontsize)
        ax1.tick_params(axis='both', which='major', labelsize=fontsize)

        ax2 = ax1.twinx()
        ax2.plot(steps, temperatures, color="red")
        ax2.set_ylabel("temperature [K]", fontsize=fontsize)
        ax2.tick_params(axis='both', which='major', labelsize=fontsize)

        if sampling_steps is not None:
            # draw vertical lines at sampling steps and give them the label "sampled":
            for sampling_step in sampling_steps:
                ax1.axvline(sampling_step, color="black", linestyle="--", label="sampled")
                ax2.axvline(sampling_step, color="black", linestyle="--", label="sampled")

        ax1.legend(fontsize=fontsize)

        hist_ax = ax[1]

        potential_energies -= np.min(potential_energies)
        hist_ax.hist(potential_energies, bins=10, color="blue")
        hist_ax.set_xlabel("rel. potential energy [kcal/mol]", fontsize=fontsize)
        hist_ax.set_ylabel("count", fontsize=fontsize)
        hist_ax.tick_params(axis='both', which='major', labelsize=fontsize)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
