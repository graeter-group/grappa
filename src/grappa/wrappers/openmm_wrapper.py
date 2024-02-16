from grappa.grappa import Grappa
from grappa.utils.openmm_utils import write_to_system
from grappa.data import Molecule


class OpenmmGrappa(Grappa):
    """
    Model wrapper class. Wraps a trained model and provides the interface to predict bonded parameters for a certain molecule in openmm, where, given a topology and a system, it will write bonded parameters to the system.
    The system must already have partial charges assigned. It is necessary to specify the charge model used to assign the charges, as the bonded parameters depend on that model. Possible values are
        - 'classical': the charges are assigned using a classical force field. For grappa-1.0, this is only possible for peptides and proteins, where classical refers to the charges from the amber99sbildn force field.
        - 'am1BCC': the charges are assigned using the am1bcc method. These charges need to be used for rna and small molecules in grappa-1.0.
    """
    def parametrize_system(self, system, topology, charge_model:str='classical'):
        """
        Predicts parameters for the system and writes them to the system.
        system: openmm.System
        topology: openmm.Topology

        TODO: add option to specify sub-topologies that are to be parametrized. (do not parametrize water, ions, etc.)
        """
        # convert openmm_topology (and system due to partial charges and impropers) to a Molecule
        molecule = Molecule.from_openmm_system(openmm_system=system, openmm_topology=topology, charge_model=charge_model)

        # predict parameters
        parameters = super().predict(molecule)

        # write parameters to system
        system = write_to_system(system, parameters)

        return system
    

    # overwrite the original predict function to throw an error:
    def predict(self, molecule):
        raise NotImplementedError('This method is not available for OpenmmGrappa. Use parametrize_system instead.')