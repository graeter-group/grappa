from grappa.grappa import Grappa
from grappa.data import Molecule, Parameters
from grappa import constants
from typing import List
from grappa.utils.torch_utils import to_numpy
from pathlib import Path
from typing import Union
import copy
import importlib.util
from grappa.utils.openmm_utils import OPENMM_WATER_RESIDUES, OPENMM_ION_RESIDUES
if importlib.util.find_spec("openmm") is not None:
    from grappa.utils.openmm_utils import get_subtopology
    import openmm
    from grappa.utils.openmm_utils import write_to_system


class OpenmmGrappa(Grappa):
    """
    Model wrapper class. Wraps a trained model and provides the interface to predict bonded parameters for a certain molecule in openmm, where, given a topology and a system, it will write bonded parameters to the system.
    The system must already have partial charges assigned.
    """
    def __init__(self, *args, **kwargs):
        assert importlib.util.find_spec("openmm") is not None, "OpenmmGrappa requires the openmm package to be installed."
        return super().__init__(*args, **kwargs)

    @classmethod
    def from_tag(cls, tag:str='latest', max_element=constants.MAX_ELEMENT, device:str='cpu'):
        """
        Loads a pretrained model from a tag. Currently, possible tags are 'grappa-1.3' and 'latest'
        """
        return super().from_tag(tag, max_element, device)
    
    def parametrize_system(self, system:"openmm.System", topology:"openmm.app.Topology", exclude_residues:List[str]=OPENMM_WATER_RESIDUES+OPENMM_ION_RESIDUES, plot_dir:Union[Path,str]=None):
        """
        Predicts parameters for the system and writes them to the system.
        system: openmm.System
        topology: openmm.app.Topology
        """
        # convert openmm_topology (and system due to partial charges and impropers) to a Molecule

        # create a sub topology excluding certain residue names (e.g. water, ions)
        # the atom.id of the atoms in this sub topology will be the same as the atom index in the original topology, i.e. as in the system.
        sub_topology = get_subtopology(topology, exclude_residues=exclude_residues)

        molecule = Molecule.from_openmm_system(openmm_system=system, openmm_topology=sub_topology)

        try:
            reference_parameters = copy.deepcopy(Parameters.from_openmm_system(openmm_system=system, mol=molecule, allow_skip_improper=True))
        except:
            reference_parameters = None

        # predict parameters
        parameters = super().predict(molecule)

        if plot_dir is not None:
            plot_dir = Path(plot_dir)

            
            if not reference_parameters is None:
                parameters.plot(filename=plot_dir/'grappa_parameters.png', compare_parameters=reference_parameters, name="Grappa", compare_name="Reference")
                parameters.compare_with(reference_parameters, filename=plot_dir/'parameter_comparison.png', xlabel="Grappa", ylabel="Reference")
            else:
                parameters.plot(filename=plot_dir/'grappa_parameters.png')

        # write parameters to system
        system = write_to_system(system, parameters)

        return system
    

    # overwrite the original predict function to throw an error:
    def predict(self, molecule):
        raise NotImplementedError('This method is not available for OpenmmGrappa. Use parametrize_system instead.')
