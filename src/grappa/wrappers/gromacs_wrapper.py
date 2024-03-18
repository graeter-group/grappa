from grappa.grappa import Grappa
import pkgutil
from typing import Union
from pathlib import Path
import argparse
import importlib
import time

class GromacsGrappa(Grappa):
    """
    Wrapper for the grappa model to be used with gromacs. This class is a subclass of the grappa model and adds a method to write the parameters to a gromacs system.

    Example Usage:
    ```python
    from grappa.wrappers.gromacs_wrapper import GromacsGrappa
    grappa = GromacsGrappa()
    grappa.parametrize('path/to/topology.top')
    ```
    Then, a file 'path/to/topology_grappa.top' will be created, which contains the grappa-predicted parameters for the topology.
    
    The topology file needs to contain 

    It is necessary to specify the charge model used to assign the charges, as the bonded parameters depend on that model. Possible values are
        - 'classical': the charges are assigned using a classical force field. For grappa-1.1, this is only possible for peptides and proteins, where classical refers to the charges from the amber99sbildn force field.
        - 'am1BCC': the charges are assigned using the am1bcc method. These charges need to be used for rna and small molecules in grappa-1.1.
    
    """
    def parametrize(self, top_path:Union[str, Path], top_outpath:Union[str, Path]=None, charge_model:str='classical'):
        """
        Creates a .top file with the grappa-predicted parameters for the topology

        Args:
            top_path (Union[str, Path]): 'path/to/topology.top' The path to the topology file, parametrised by a classical force field (nonbonded parameters and improper torsion idxs are needed)
            top_outpath (Union[str, Path], optional): Defaults to 'path/to/topology_grappa.top'. The path to the output file.
            charge_model (str, optional): Defaults to 'classical'. The charge model used to assign the charges. Possible values
                - 'classical': the charges are assigned using a classical force field. For grappa-1.1, this is only possible for peptides and proteins, where classical refers to the charges from the amber99sbildn force field.
                - 'am1BCC': the charges are assigned using the am1bcc method. These charges need to be used for rna and small molecules in grappa-1.1.
        """
        assert importlib.util.find_spec('kimmdy') is not None, "kimmdy must be installed to use the GromacsGrappa class."
        
        if not top_outpath:
            top_outpath = Path(top_path).with_stem(Path(top_path).stem + "_grappa")

        # import this only when the function is called to make grappas dependency on kimmdy optional
        from kimmdy.topology.topology import Topology
        from kimmdy.parsing import read_top, write_top

        from grappa.utils.kimmdy_utils import KimmdyGrappaParameterizer

        # load the topology
        top_path = Path(top_path)
        topology = Topology(read_top(Path(top_path)))

        # call grappa model to write the parameters to the topology
        topology.parametrizer = KimmdyGrappaParameterizer(grappa_instance=self, charge_model=charge_model,grappa_tag='latest')
        topology.needs_parameterization = True
        
        ## write top file
        write_top(topology.to_dict(), top_outpath)
        
        return


def main_(top_path:Union[str,Path], top_outpath:Union[str,Path]=None, modeltag:str='grappa-1.1', charge_model:str='classical', device:str='cpu'):
    print('execute',time.time())
    grappa = GromacsGrappa.from_tag(modeltag, device=device)
    grappa.parametrize(top_path, top_outpath, charge_model=charge_model)
    return

def main():
    parser = argparse.ArgumentParser(description='Parametrize a topology with grappa')
    parser.add_argument('--top_path', '-f', type=str, required=True, help='path/to/topology.top: The path to the topology file, parametrised by a classical force field. The topology should not contain water or ions, as grappa does not predict parameters for these.')
    parser.add_argument('--top_outpath', '-o', type=str, default=None, help='path to the topology file written by grappa that can then be used as usual .top file in gromacs. Defaults to top_path with _grappa appended, i.e. path/to/topology_grappa.top')
    parser.add_argument('--modeltag', '-t', type=str, default='grappa-1.1', help='tag of the grappa model to use')
    parser.add_argument('--charge_model', '-c', type=str, default='classical', help='The charge model used to assign the partial charges. Possible values: classical, am1BCC')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='The device to use for grappas inference forward pass. Defaults to cpu.')
    args = parser.parse_args()
    return main_(args.top_path, top_outpath=args.top_outpath, modeltag=args.modeltag, charge_model=args.charge_model, device=args.device)
    