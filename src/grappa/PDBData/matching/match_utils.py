from pathlib import Path
from itertools import takewhile
from typing import Union, List, Tuple


## utils (from kimmdy) ##
def read_rtp(path: Path) -> dict:
    # TODO: make this more elegant and performant
    with open(path, "r") as f:
        sections = _get_sections(f, "\n")
        d = {}
        for i, s in enumerate(sections):
            # skip empty sections
            if s == [""]:
                continue
            name, content = _extract_section_name(s)
            content = [c.split() for c in content if len(c.split()) > 0]
            if not name:
                name = f"BLOCK {i}"
            d[name] = _create_subsections(content)
            # d[name] = content

        return d

def _extract_section_name(ls):
    """takes a list of lines and return a tuple
    with the name and the lines minus the
    line that contained the name.
    Returns the empty string if no name was found.
    """
    for i, l in enumerate(ls):
        if l and l[0] != ";" and "[" in l:
            name = l.strip("[] \n")
            ls.pop(i)
            return (name, ls)
    else:
        return ("", ls)

def _is_not_comment(c: str) -> bool:
    return c != ";"

def _get_sections(seq, section_marker):
    data = [""]
    for line in seq:
        line = "".join(takewhile(_is_not_comment, line))
        if line.strip(" ").startswith(section_marker):
            if data:
                # first element will be empty
                # because newlines mark sections
                data.pop(0)
                # only yield section if non-empty
                if len(data) > 0:
                    yield data
                data = [""]
        data.append(line.strip("\n"))
    if data:
        yield data

def _create_subsections(ls):
    d = {}
    subsection_name = "other"
    for i, l in enumerate(ls):
        if l[0] == "[":
            subsection_name = l[1]
        else:
            if subsection_name not in d:
                d[subsection_name] = []
            d[subsection_name].append(l)

    return d
##


def radname_from_log(path: Union[Path,str]) -> str:
    """
    Returns the name of the radical atom from the log file if this is named properly. only works for single amino acids.
    """

    path = Path(path)
    name = path.stem
    radname = name.split("_")[1]
    return radname


def is_radical(filename: Union[Path,str]) -> bool:
    """
    Returns True if the filename is a radical amino acid.
    """
    filename = Path(filename)
    name = filename.stem
    if "nat" in name:
        return False
    else:
        return True
    
def list_to_tuple(List):
        if len(List) == 0:
            return ()
        else:
            return (tuple(List[0]),) + list_to_tuple(List[1:])
    
def tuple_to_list(tup):
    if len(tup) == 0:
        return []
    else:
        return [list(tup[0]),] + tuple_to_list(tup[1:])
        

def bond_majority_vote(trajectory) -> List:
    
    from ase.geometry.analysis import Analysis

    # do a majority vote on the bonds:
    bondvotes = {}
    for idx, state in enumerate(trajectory):
        ana = Analysis(state)
        [bonds] = ana.unique_bonds
        bonds = list_to_tuple(bonds)
        if not bonds in bondvotes.keys():
            bondvotes[bonds] = 0
        bondvotes[bonds] += 1
    bonds = max(bondvotes, key=bondvotes.get)
    bonds = tuple_to_list(bonds)
    return bonds