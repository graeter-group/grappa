from grappa.grappa import Grappa
from grappa.utils.openmm_utils import write_to_system
from grappa.data import Molecule, Parameters
from grappa import constants
from typing import List
from grappa.utils.openmm_utils import OPENMM_ION_RESIDUES, OPENMM_WATER_RESIDUES, get_subtopology

class OpenmmGrappa(Grappa):
    """
    Model wrapper class. Wraps a trained model and provides the interface to predict bonded parameters for a certain molecule in openmm, where, given a topology and a system, it will write bonded parameters to the system.
    The system must already have partial charges assigned. It is necessary to specify the charge model used to assign the charges, as the bonded parameters depend on that model. Possible values are
        - 'classical': the charges are assigned using a classical force field. For grappa-1.0, this is only possible for peptides and proteins, where classical refers to the charges from the amber99sbildn force field.
        - 'am1BCC': the charges are assigned using the am1bcc method. These charges need to be used for rna and small molecules in grappa-1.0.
    """
    @classmethod
    def from_tag(cls, tag:str='latest', max_element=constants.MAX_ELEMENT, device:str='cpu'):
        """
        Loads a pretrained model from a tag. Currently, possible tags are 'grappa-1.0', 'grappa-1.1' and 'latest'
        """
        return super().from_tag(tag, max_element, device)
    
    def parametrize_system(self, system, topology, charge_model:str='classical', exclude_residues:List[str]=OPENMM_WATER_RESIDUES+OPENMM_ION_RESIDUES, plot_dir:str=None):
        """
        Predicts parameters for the system and writes them to the system.
        system: openmm.System
        topology: openmm.Topology
        charge_model: str
            The charge model used to assign the charges. Possible values
                - 'classical': the charges are assigned using a classical force field. For grappa-1.0, this is only possible for peptides and proteins, where classical refers to the charges from the amber99sbildn force field.
                - 'am1BCC': the charges are assigned using the am1bcc method. These charges need to be used for rna and small molecules in grappa-1.0.

        TODO: add option to specify sub-topologies that are to be parametrized. (do not parametrize water, ions, etc.)
        """
        # convert openmm_topology (and system due to partial charges and impropers) to a Molecule

        # create a sub topology excluding certain residue names (e.g. water, ions)
        # the atom.id of the atoms in this sub topology will be the same as the atom index in the original topology, i.e. as in the system.
        sub_topology = get_subtopology(topology, exclude_residues=exclude_residues)

        molecule = Molecule.from_openmm_system(openmm_system=system, openmm_topology=sub_topology, charge_model=charge_model)

        try:
            reference_parameters = Parameters.from_openmm_system(openmm_system=system, mol=molecule, allow_skip_improper=True)
        except:
            reference_parameters = None

        # predict parameters
        parameters = super().predict(molecule)

        if plot_dir is not None:
            parameters.plot(filename=plot_dir+'/grappa_parameters.png')
            
            if not reference_parameters is None:
                parameters.plot(filename=plot_dir+'/reference_parameters.png', compare_parameters=reference_parameters, name="Grappa", compare_name="Amber ff99SB-ILDN")
                parameters.compare_with(reference_parameters, filename=plot_dir+'/parameter_comparison.png', xlabel="Grappa", ylabel="Amber ff99SB-ILDN")

        # write parameters to system
        system = write_to_system(system, parameters)

        return system
    

    # overwrite the original predict function to throw an error:
    def predict(self, molecule):
        raise NotImplementedError('This method is not available for OpenmmGrappa. Use parametrize_system instead.')
