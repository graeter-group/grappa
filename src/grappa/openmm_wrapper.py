from grappa.grappa import Grappa
from grappa import constants
from grappa.utils.openmm_utils import write_to_system
from grappa.data import Molecule


class openmm_Grappa:
    """
    Model wrapper class. Wraps a trained model and provides the interface to predict bonded parameters for a certain molecule in openmm, where, given a topology and a system, it will write bonded parameters to the system.
    """
    def __init__(self, model, max_element=constants.MAX_ELEMENT) -> None:
        self.grappa_model = Grappa(model, max_element=max_element)

    def parametrize_system(self, system, topology):
        """
        Predicts parameters for the system and writes them to the system.
        system: openmm.System
        topology: openmm.Topology
        """
        # convert openmm_topology to a Molecule
        molecule = Molecule.from_openmm(topology)

        # predict parameters
        parameters = self.grappa_model.predict(molecule)

        # write parameters to system
        system = write_to_system(system, parameters)

        return system