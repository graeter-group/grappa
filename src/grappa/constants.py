IMPROPER_CENTRAL_IDX = 2 # the index of the central atom in an improper torsion as used by grappa

MAX_ELEMENT = 53 # cover Iodine

N_PERIODICITY_PROPER = 6
N_PERIODICITY_IMPROPER = 6

BONDED_CONTRIBUTIONS = [("n2","k"), ("n2","eq"), ("n3","k"), ("n3","eq"), ("n4","k"), ("n4_improper","k")]

RESIDUES = ['ACE', 'NME', 'CYS', 'ASP', 'SER', 'GLN', 'LYS', 'ILE', 'PRO', 'THR', 'PHE', 'ASN', 'GLY', 'HIS', 'LEU', 'ARG', 'TRP', 'ALA', 'VAL', 'GLU', 'TYR', 'MET', "HIE", "HID", "HIP", "DOP", "HYP"]

ONELETTER = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "HIE": "H",
    "HIP": "H",
    "HID": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "HYP": "O",
    "DOP": "J",
    "ACE": "B",
    "NME": "Z"
}