from grappa.grappa import Grappa
from typing import Union
from pathlib import Path
import sys
import argparse
import importlib
import logging
import warnings
from grappa.constants import Deprecated

if sys.version_info > (3, 10):
    from importlib_metadata import version
else:
    from importlib.metadata import version

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
    """
    def __init__(self, *args, **kwargs):
        assert importlib.util.find_spec('kimmdy') is not None, "kimmdy must be installed to use the GromacsGrappa class."
        return super().__init__(*args, **kwargs)

    def parametrize(self, top_path:Union[str, Path], top_outpath:Union[str, Path]=None, include_list:list = [], exclude_list:list = [], plot_parameters:bool=False, charge_model:Deprecated=None):
        """
        Creates a .top file with the grappa-predicted parameters for the topology

        Args:
            top_path (Union[str, Path]): 'path/to/topology.top' The path to the topology file, parametrised by a classical force field (nonbonded parameters and improper torsion idxs are needed)
            top_outpath (Union[str, Path], optional): Defaults to 'path/to/topology_grappa.top'. The path to the output file.
            include list: Include certain GROMACS topology molecules in `Reactive` molecule.
            exclude_list: Exclude certain GROMACS topology molecules in `Reactive` molecule.
            plot_parameters (bool, optional): Defaults to False. If True, a plot of the parameters is created and saved in the same directory as the output file.
        """
        assert importlib.util.find_spec('kimmdy') is not None, "kimmdy must be installed to use the GromacsGrappa class."

        if not charge_model is None:
            warnings.warn("The charge_model argument for GromacsGrappa.parametrize is deprecated, has no effect and will be removed in the future.", DeprecationWarning)
        
        if not top_outpath:
            top_outpath = Path(top_path).with_stem(Path(top_path).stem + "_grappa")

        plot_path = Path(Path(top_outpath).stem + "_parameters.png") if plot_parameters else None

        # import this only when the function is called to make grappas dependency on kimmdy optional
        from kimmdy.topology.topology import Topology
        from kimmdy.topology.utils import get_is_reactive_predicate_f
        from kimmdy.parsing import read_top, write_top

        from grappa.utils.kimmdy_utils import KimmdyGrappaParameterizer

        # load the topology
        top_path = Path(top_path)

        kimmdy_version = version("kimmdy")
        logging.info(f"kimmdy version {kimmdy_version}")
        if list(map(int,version("kimmdy").split('.'))) > [6,6,0]:
            topology = Topology(read_top(Path(top_path)),radicals='',is_reactive_predicate_f=get_is_reactive_predicate_f(include=include_list, exclude=exclude_list))   #radicals='' means kimmdy won't search for radicals
        else:
            logging.info(f"version number below '6.6.0', ignoring explicit includes and excludes!")
            topology = Topology(read_top(Path(top_path)),radicals='')

        # call grappa model to write the parameters to the topology
        topology.parametrizer = KimmdyGrappaParameterizer(grappa_instance=self, plot_path=plot_path)
        topology.needs_parameterization = True
        
        ## write top file
        logging.info(f"Writing topology with grappa parameters to {top_outpath}")
        write_top(topology.to_dict(), top_outpath)
        
        return


def main_(top_path:Union[str,Path], top_outpath:Union[str,Path]=None, modeltag:str='latest', device:str='cpu', include_list:list=[], exclude_list:list=[], plot_parameters:bool=False, modelpath:Union[str,Path]=None):
    if not modelpath is None:
        grappa = GromacsGrappa.from_ckpt(modelpath, device=device)
    else:
        grappa = GromacsGrappa.from_tag(modeltag, device=device)
    grappa.parametrize(top_path, top_outpath, include_list=include_list, exclude_list=exclude_list, plot_parameters=plot_parameters)
    return

def main():
    parser = argparse.ArgumentParser(description='Parametrize a topology with grappa')
    parser.add_argument('--top_path', '-f', type=str, required=True, help='path/to/topology.top: The path to the topology file, parametrised by a classical force field. The topology should not contain water or ions, as grappa does not predict parameters for these.')
    parser.add_argument('--top_outpath', '-o', type=str, default=None, help='path to the topology file written by grappa that can then be used as usual .top file in gromacs. Defaults to top_path with _grappa appended, i.e. path/to/topology_grappa.top')
    parser.add_argument('--modeltag', '-t', type=str, default='grappa-1.3', help='tag of the grappa model to use')
    parser.add_argument('--modelpath', '-ckpt', type=str, default=None, help='Path to the grappa model checkpoint. Overwrites the modeltag argument.')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='The device to use for grappas inference forward pass. Defaults to cpu.')
    parser.add_argument("--include","-w",nargs='+',default=[],help="Include certain GROMACS topology molecules in `Reactive` molecule.")
    parser.add_argument("--exclude","-b",nargs='+',default=[],help="Exclude certain GROMACS topology molecules in `Reactive` molecule.")    
    parser.add_argument('--plot_parameters', '-p', action='store_true', help='If set, a plot of the MM parameters is created and saved in the same directory as the output file.')
    args = parser.parse_args()

    return main_(args.top_path, top_outpath=args.top_outpath, modeltag=args.modeltag, device=args.device, include_list=args.include, exclude_list=args.exclude, plot_parameters=args.plot_parameters, modelpath=args.modelpath)
    