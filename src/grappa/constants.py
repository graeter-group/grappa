IMPROPER_CENTRAL_IDX = 2 # the index of the central atom in an improper torsion as used by grappa

MAX_ELEMENT = 53 # cover Iodine

# NOTE: check effect of these. (grappa models have this parameter too)
N_PERIODICITY_PROPER = 6
N_PERIODICITY_IMPROPER = 6

CHARGE_MODELS = ['am1BCC', 'classical'] # classical: amber99ffsbildn-charges for peptides. The idea is that the tag 'classical' can refer to different charge models for different types of molecules (which grappa can usually distinguish). e.g. one could train for rna on charges from some classical rna forcefield without introducing an additional tag.

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


# creation: see grappa/tests/masses.py
ATOMIC_MASSES = {
    1: 1.008,
    2: 4.002,
    3: 6.94,
    4: 9.012,
    5: 10.81,
    6: 12.011,
    7: 14.007,
    8: 15.999,
    9: 18.998,
    10: 20.1797,
    11: 22.989,
    12: 24.305,
    13: 26.981,
    14: 28.085,
    15: 30.973,
    16: 32.06,
    17: 35.45,
    18: 39.95,
    19: 39.0983,
    20: 40.078,
    21: 44.955,
    22: 47.867,
    23: 50.9415,
    24: 51.9961,
    25: 54.938,
    26: 55.845,
    27: 58.933,
    28: 58.6934,
    29: 63.546,
    30: 65.38,
    31: 69.723,
    32: 72.63,
    33: 74.921,
    34: 78.971,
    35: 79.904,
    36: 83.798,
    37: 85.4678,
    38: 87.62,
    39: 88.905,
    40: 91.224,
    41: 92.906,
    42: 95.95,
    43: 97.0,
    44: 101.07,
    45: 102.905,
    46: 106.42,
    47: 107.8682,
    48: 112.414,
    49: 114.818,
    50: 118.71,
    51: 121.76,
    52: 127.6,
    53: 126.904
}