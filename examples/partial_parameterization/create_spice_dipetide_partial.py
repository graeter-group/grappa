# %%
"""
Create serval datasets based on the 'spice-dipeptide-amber99' with different atoms annotated as grappa atoms.

Please note that the new datasets are only save as DGLGraph objects.

The following datasets are created for a ratio of grappa atoms e {0.1, 0.2, ..., 0.9}:
- spice-dipeptide-amber99-random{ratio}-grappa: Grappa atoms are randomly selected.
- spice-dipeptide-amber99-connected{ratio}-grappa: Grappa atoms are selected based on their index in the molecular graph. Selected atoms are checked for connectivity.

Moreover, the following datasets are created:
- spice-dipeptide-amber99-all-grappa: All atoms are annotated as grappa atoms.
- spice-dipeptide-amber99-sidechain-grappa: Only the side chain atoms of the N-terminal amino acid are annotated as grappa atoms.
"""
from typing import Callable
from pathlib import Path
from dgl import DGLGraph
import torch
from tqdm import tqdm

from grappa.data import Dataset, MolData
from grappa.data.transforms import idx_to_one_hot, annotate_num_grappa_atoms_in_interaction
from grappa.utils.graph_utils import get_nterminal_side_chain_atoms, get_ratio_of_atoms 
from grappa.utils.data_utils import get_data_path, get_moldata_path


def annotate_grappa_atoms_from_fn(graph: DGLGraph, fn: Callable, *fn_args, **fn_kwargs) -> DGLGraph:
    """
    Annotate the atoms obtained from the function as grappa atoms.

    The number of grappa atoms per interaction is also annotated to the graph.
    
    Args:
        graph (DGLGraph): The DGLGraph object representing the molecular graph.
        fn (Callable): The function to get the atoms to annotate as grappa atoms. The function should take the graph as input and return a list of atom indices.
        *fn_args: Additional positional arguments to pass to fn.
        **fn_kwargs: Additional keyword arguments to pass to fn.
    """

    grappa_atoms_ids = torch.tensor(fn(graph, *fn_args, **fn_kwargs))
    grappa_atoms = idx_to_one_hot(grappa_atoms_ids, graph.num_nodes("n1"))
    graph.nodes["n1"].data["grappa_atom"] = grappa_atoms
    graph = annotate_num_grappa_atoms_in_interaction(graph, grappa_atom_ids=grappa_atoms_ids)
    return graph

def create_peptide_dataset_with_grappa_atoms_from_fn(in_tag: str, fn: Callable, out_tag: str, *fn_args, **fn_kwargs) -> None:
    """
    Create a new dataset with the atoms obtained from the function as grappa atoms.

    The number of grappa atoms per interaction is also annotated to the graph based on the grapap atoms.

    Args:
        in_tag (str): The input dataset tag.
        fn (Callable): The function to get the atoms to annotate as grappa atoms. The function should take the graph as input and return a list of atom indices.
        out_tag (str): The output dataset tag.
        *fn_args: Additional positional arguments to pass to fn.
        **fn_kwargs: Additional keyword arguments to pass to fn.
    """
    dataset_dir = get_moldata_path(in_tag)
    graphs, mol_ids = [], []
    for file in tqdm(dataset_dir.glob("*.npz")):
        moldata = MolData.load(file)
        graph = moldata.to_dgl()
        graph = annotate_grappa_atoms_from_fn(graph, fn, *fn_args, **fn_kwargs)
        graphs.append(graph)
        mol_ids.append(moldata.mol_id)

    # Save the dataset as dgl graphs
    Dataset(graphs, mol_ids, subdataset=out_tag).save(Path(get_data_path(), "dgl_datasets", out_tag))
    # Save the tag and dataset to the dataset_tags.csv file
    get_moldata_path(out_tag, data_dir=Path(get_data_path(), "dgl_datasets"))

# %%
for grappa_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    create_peptide_dataset_with_grappa_atoms_from_fn("spice-dipeptide-amber99", get_ratio_of_atoms, f"spice-dipeptide-amber99-random{grappa_ratio}-grappa", ratio=grappa_ratio, random_sampling=True)
    create_peptide_dataset_with_grappa_atoms_from_fn("spice-dipeptide-amber99", get_ratio_of_atoms, f"spice-dipeptide-amber99-connected{grappa_ratio}-grappa", ratio=grappa_ratio, random_sampling=False)

create_peptide_dataset_with_grappa_atoms_from_fn("spice-dipeptide-amber99", get_ratio_of_atoms, "spice-dipeptide-amber99-all-grappa", ratio=1.0)
create_peptide_dataset_with_grappa_atoms_from_fn("spice-dipeptide-amber99", get_nterminal_side_chain_atoms, "spice-dipeptide-amber99-sidechain-grappa")
