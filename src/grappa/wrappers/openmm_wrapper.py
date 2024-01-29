from grappa.grappa import Grappa
from grappa.utils.openmm_utils import write_to_system
from grappa.data import Molecule


class openmm_Grappa(Grappa):
    """
    Model wrapper class. Wraps a trained model and provides the interface to predict bonded parameters for a certain molecule in openmm, where, given a topology and a system, it will write bonded parameters to the system.
    """
    def parametrize_system(self, system, topology):
        """
        Predicts parameters for the system and writes them to the system.
        system: openmm.System
        topology: openmm.Topology

        TODO: add option to specify sub-topologies that are to be parametrized
        """
        # convert openmm_topology (and system due to partial charges and impropers) to a Molecule
        molecule = Molecule.from_openmm_system(openmm_system=system, openmm_topology=topology)

        # predict parameters
        parameters = self.predict(molecule)

        # write parameters to system
        system = write_to_system(system, parameters)

        return system