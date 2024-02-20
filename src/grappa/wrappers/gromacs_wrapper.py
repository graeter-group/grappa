from grappa.constants import MAX_ELEMENT
from grappa.grappa import Grappa
from grappa.models.grappa import GrappaModel
from grappa.utils.openmm_utils import write_to_system
from grappa.data import Molecule
import pkgutil

if __name__ == "__main__":

    class GromacsGrappa(Grappa):
        """
    
        
        It is necessary to specify the charge model used to assign the charges, as the bonded parameters depend on that model. Possible values are
            - 'classical': the charges are assigned using a classical force field. For grappa-1.0, this is only possible for peptides and proteins, where classical refers to the charges from the amber99sbildn force field.
            - 'am1BCC': the charges are assigned using the am1bcc method. These charges need to be used for rna and small molecules in grappa-1.0.
        
        """

        def __init__(*args, **kwargs):
            pkgutil.find_spec('kimmdy')
            super().__init__(*args, **kwargs)

        def parametrize_system(self, system, topology, charge_model:str='classical'):
            """
            Predicts parameters for the system and writes them to the system.
            system: openmm.System
            topology: openmm.Topology

            TODO: add option to specify sub-topologies that are to be parametrized. (do not parametrize water, ions, etc.)
            """
            # convert openmm_topology (and system due to partial charges and impropers) to a Molecule
            molecule = None
