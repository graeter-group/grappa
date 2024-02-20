'''
Defines things that are constant across the whole package.
'''

from grappa import units


class GrappaUnits:
    LENGTH = units.angstrom
    ANGLE = units.radian
    ENERGY = units.kilocalorie_per_mol

    BOND_K = ENERGY / (LENGTH ** 2)
    BOND_EQ = LENGTH
    ANGLE_K = ENERGY / (ANGLE ** 2)
    ANGLE_EQ = ANGLE
    TORSION_K = ENERGY
    TORSION_PHASE = ANGLE

def get_grappa_units_in_openmm():
    from openmm.unit import angstrom, kilocalorie_per_mole, radian
    return {
        'LENGTH': angstrom,
        'ANGLE': radian,
        'ENERGY': kilocalorie_per_mole,
        'BOND_K': kilocalorie_per_mole / (angstrom ** 2),
        'BOND_EQ': angstrom,
        'ANGLE_K': kilocalorie_per_mole / (radian ** 2),
        'ANGLE_EQ': radian,
        'TORSION_K': kilocalorie_per_mole,
        'TORSION_PHASE': radian
    }
    


IMPROPER_CENTRAL_IDX = 2 # the index of the central atom in an improper torsion as used by grappa

MAX_ELEMENT = 53 # cover Iodine

# the periodicities for dataset creation. Grappa models have this as hyperparameter too, which must be smaller or equal than the values below.
N_PERIODICITY_PROPER = 6
N_PERIODICITY_IMPROPER = 6

CHARGE_MODELS = ['am1BCC', 'classical'] # classical: amber99ffsbildn-charges for peptides. The idea is that the tag 'classical' can refer to different charge models for different types of molecules (which grappa can usually distinguish). e.g. one could train for rna on charges from some classical rna forcefield without introducing an additional tag.

BONDED_CONTRIBUTIONS = [("n2","k"), ("n2","eq"), ("n3","k"), ("n3","eq"), ("n4","k"), ("n4_improper","k")]




# creation: see tests/misc/masses.py
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