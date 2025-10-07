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
        assert importlib.util.find_spec('gmx_top4py') is not None, "gmx-top4py must be installed to use the GromacsGrappa class."
        return super().__init__(*args, **kwargs)

    def parametrize(self, top_path:Union[str, Path], top_outpath:Union[str, Path]=None, select_list:list = [], deselect_list:list = [], plot_parameters:bool=False, charge_model:Deprecated=None):
        """
        Creates a .top file with the grappa-predicted parameters for the topology

        Args:
            top_path (Union[str, Path]): 'path/to/topology.top' The path to the topology file, parametrised by a classical force field (nonbonded parameters and improper torsion idxs are needed)
            top_outpath (Union[str, Path], optional): Defaults to 'path/to/topology_grappa.top'. The path to the output file.
            select_list: Select certain GROMACS topology moleculetypes for parameterization with grappa.
            deselect_list: Deselect certain GROMACS topology moleculetypes for parameterization with grappa.
            plot_parameters (bool, optional): Defaults to False. If True, a plot of the parameters is created and saved in the same directory as the output file.
        """
        assert importlib.util.find_spec('gmx_top4py') is not None, "gmx-top4py must be installed to use the GromacsGrappa class."

        if not charge_model is None:
            warnings.warn("The charge_model argument for GromacsGrappa.parametrize is deprecated, has no effect and will be removed in the future.", DeprecationWarning)
        
        if not top_outpath:
            top_outpath = Path(top_path).with_stem(Path(top_path).stem + "_grappa")

        plot_path = Path(Path(top_outpath).stem + "_parameters.png") if plot_parameters else None

        # import this only when the function is called to make grappas dependency on gmx-top4py optional
        from gmx_top4py.topology.topology import Topology
        from gmx_top4py.topology.utils import get_is_selected_moleculetype_f
        from gmx_top4py.parsing import read_top, write_top

        from grappa.utils.kimmdy_utils import GrappaParameterizer

        # load the topology
        top_path = Path(top_path)

        gmx_top4py_version = version("gmx_top4py")
        logging.info(f"gmx-top4py version {gmx_top4py_version}")
        topology = Topology(read_top(Path(top_path)),radicals='',is_selected_moleculetype_f=get_is_selected_moleculetype_f(include=select_list, exclude=deselect_list))   #radicals='' means gmx-top4py won't search for radicals


        # call grappa model to write the parameters to the topology
        topology.parametrizer = GrappaParameterizer(grappa_instance=self, plot_path=plot_path)
        topology.needs_parameterization = True
        
        ## write top file
        logging.info(f"Writing topology with grappa parameters to {top_outpath}")
        write_top(topology.to_dict(), top_outpath)
        
        return


def main_(top_path:Union[str,Path], top_outpath:Union[str,Path]=None, modeltag:str='latest', device:str='cpu', select_list:list=[], deselect_list:list=[], plot_parameters:bool=False, modelpath:Union[str,Path]=None):
    if not modelpath is None:
        grappa = GromacsGrappa.from_ckpt(modelpath, device=device)
    else:
        grappa = GromacsGrappa.from_tag(modeltag, device=device)
    grappa.parametrize(top_path, top_outpath, include_list=select_list, exclude_list=deselect_list, plot_parameters=plot_parameters)
    return

def main():
    parser = argparse.ArgumentParser(description='Parametrize a topology with grappa')
    parser.add_argument('--top_path', '-f', type=str, required=True, help='path/to/topology.top: The path to the topology file, parametrised by a classical force field. The topology should not contain water or ions, as grappa does not predict parameters for these.')
    parser.add_argument('--top_outpath', '-o', type=str, default=None, help='path to the topology file written by grappa that can then be used as usual .top file in gromacs. Defaults to top_path with _grappa appended, i.e. path/to/topology_grappa.top')
    parser.add_argument('--modeltag', '-t', type=str, default='grappa-1.3', help='tag of the grappa model to use')
    parser.add_argument('--modelpath', '-ckpt', type=str, default=None, help='Path to the grappa model checkpoint. Overwrites the modeltag argument.')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='The device to use for grappas inference forward pass. Defaults to cpu.')
    parser.add_argument("--include", nargs='+',default=[],help=argparse.SUPPRESS)  # deprecated, use --select instead
    parser.add_argument("--exclude", nargs='+',default=[],help=argparse.SUPPRESS)  # deprecated, use --deselect instead
    parser.add_argument("--select", "-w",nargs='+',default=[],help="Select certain GROMACS topology moleculetypes for parameterization with grappa. Per default all moleculetypes except solvent and inorganic ions are selected.")
    parser.add_argument("--deselect","-b",nargs='+',default=[],help="Deselect certain GROMACS topology moleculetypes for parameterization with grappa.")
    parser.add_argument('--plot_parameters', '-p', action='store_true', help='If set, a plot of the MM parameters is created and saved in the same directory as the output file.')
    args = parser.parse_args()

    if args.include or args.exclude:
        warnings.warn("The commandline argument '--include' for grappa_gmx is deprecated and will be removed in the future. Use '--select' instead.", DeprecationWarning)
        args.select += args.include
    if args.exclude:
        warnings.warn("The commandline argument '--exclude' for grappa_gmx is deprecated and will be removed in the future. Use '--deselect' instead.", DeprecationWarning)
        args.deselect += args.exclude

    return main_(args.top_path, top_outpath=args.top_outpath, modeltag=args.modeltag, device=args.device, select_list=args.select, deselect_list=args.deselect, plot_parameters=args.plot_parameters, modelpath=args.modelpath)
    