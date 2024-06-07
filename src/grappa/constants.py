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

MAX_NUM_CHARGE_MODELS = 10 # the dimension used for one-hot encoding of the charge model. this is larger than the number of charge models to allow for additional charge models in the future while retining backwards compatibility with datasets and model weights that reflect fewer charge models.

CHARGE_MODELS = ['am1BCC', 'amber99', 'charmm'] # amber99: amber99ffsbildn-charges for peptides. The idea is that the tag 'amber99' can refer to different charge models for different types of molecules (which grappa can usually distinguish). e.g. one could train for rna on charges from some classical rna forcefield without introducing an additional tag.

BONDED_CONTRIBUTIONS = [("n2","k"), ("n2","eq"), ("n3","k"), ("n3","eq"), ("n4","k"), ("n4_improper","k")]
