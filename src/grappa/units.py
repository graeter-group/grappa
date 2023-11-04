"""
Units used internally in grappa.
"""

from openmm import unit as openmm_unit

from grappa.constants import RESIDUES

DISTANCE_UNIT = openmm_unit.angstrom
ENERGY_UNIT = openmm_unit.kilocalorie_per_mole
FORCE_UNIT = ENERGY_UNIT / DISTANCE_UNIT
ANGLE_UNIT = openmm_unit.radian
CHARGE_UNIT = openmm_unit.elementary_charge
MASS_UNIT = openmm_unit.dalton

BOND_K_UNIT = ENERGY_UNIT / (DISTANCE_UNIT ** 2)
BOND_EQ_UNIT = DISTANCE_UNIT
ANGLE_K_UNIT = ENERGY_UNIT / (ANGLE_UNIT ** 2)
ANGLE_EQ_UNIT = ANGLE_UNIT
TORSION_K_UNIT = ENERGY_UNIT
TORSION_PHASE_UNIT = ANGLE_UNIT


OPENMM_LENGTH_UNIT = openmm_unit.nanometer
OPENMM_ANGLE_UNIT = openmm_unit.radian
OPENMM_ENERGY_UNIT = openmm_unit.kilojoule_per_mole

OPENMM_BOND_EQ_UNIT = OPENMM_LENGTH_UNIT
OPENMM_ANGLE_EQ_UNIT = OPENMM_ANGLE_UNIT
OPENMM_TORSION_K_UNIT = OPENMM_ENERGY_UNIT
OPENMM_TORSION_PHASE_UNIT = OPENMM_ANGLE_UNIT
OPENMM_BOND_K_UNIT = OPENMM_ENERGY_UNIT / (OPENMM_LENGTH_UNIT**2)
OPENMM_ANGLE_K_UNIT = OPENMM_ENERGY_UNIT / (OPENMM_ANGLE_UNIT**2)
OPENMM_CHARGE_UNIT = openmm_unit.elementary_charge


def convert(value, from_unit, to_unit):
    """
    Convert a value from one unit to another.

    Parameters:
    value (float): The value to convert.
    from_unit (openmm.unit.Unit): The unit of the value.
    to_unit (openmm.unit.Unit): The unit to convert the value to.

    Returns:
    float: The converted value.
    """
    return openmm_unit.Quantity(value, from_unit).value_in_unit(to_unit)