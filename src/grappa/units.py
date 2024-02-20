"""
Units used internally in grappa.
"""

from openmm import unit as openmm_unit


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


class Unit:
    def __init__(self, time_dimension=0, length_dimension=0, mass_dimension=0,
                 current_dimension=0, temperature_dimension=0, amount_dimension=0,
                 luminous_intensity_dimension=0, in_si_units=1, name=''):
        self.time_dimension = time_dimension
        self.length_dimension = length_dimension
        self.mass_dimension = mass_dimension
        self.current_dimension = current_dimension
        self.temperature_dimension = temperature_dimension
        self.amount_dimension = amount_dimension
        self.luminous_intensity_dimension = luminous_intensity_dimension
        self.in_si_units = in_si_units
        self.name = name

    def __mul__(self, other):
        result = self.__copy__()
        result *= other
        return result
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        result = self.__copy__()
        result /= other
        return result
    
    def __rtruediv__(self, other):
        result = (self.__copy__())**-1
        result *= other
        return result
    
    def __imul__(self, other):
        if isinstance(other, Unit):
            self.time_dimension += other.time_dimension
            self.length_dimension += other.length_dimension
            self.mass_dimension += other.mass_dimension
            self.current_dimension += other.current_dimension
            self.temperature_dimension += other.temperature_dimension
            self.amount_dimension += other.amount_dimension
            self.luminous_intensity_dimension += other.luminous_intensity_dimension
            self.in_si_units *= other.in_si_units
            self.name = f"({self.name}*{other.name})" if self.name and other.name else self.name or other.name
        else:  # Assuming other is a scalar
            self.in_si_units *= other
        return self

    def __itruediv__(self, other):
        if isinstance(other, Unit):
            self.time_dimension -= other.time_dimension
            self.length_dimension -= other.length_dimension
            self.mass_dimension -= other.mass_dimension
            self.current_dimension -= other.current_dimension
            self.temperature_dimension -= other.temperature_dimension
            self.amount_dimension -= other.amount_dimension
            self.luminous_intensity_dimension -= other.luminous_intensity_dimension
            self.in_si_units /= other.in_si_units
            self.name = f"({self.name}/{other.name})" if self.name and other.name else self.name
        else:  # Assuming other is a scalar
            self.in_si_units /= other
        return self

    def __copy__(self):
        return Unit(self.time_dimension, self.length_dimension, self.mass_dimension,
                    self.current_dimension, self.temperature_dimension, self.amount_dimension,
                    self.luminous_intensity_dimension, self.in_si_units, self.name)

    # define powers of units:
    def __pow__(self, power):
        result = self.__copy__()
        result.time_dimension *= power
        result.length_dimension *= power
        result.mass_dimension *= power
        result.current_dimension *= power
        result.temperature_dimension *= power
        result.amount_dimension *= power
        result.luminous_intensity_dimension *= power
        result.in_si_units = self.in_si_units**power
        result.name = f"({self.name}^{power})" if self.name else None
        return result


    def __repr__(self):
        if self.name is not None:
            return self.name
        else:
            name = f"{self.in_si_units}"
            if self.time_dimension != 0:
                name += f" s^{self.time_dimension}"
            if self.length_dimension != 0:
                name += f" m^{self.length_dimension}"
            if self.mass_dimension != 0:
                name += f" kg^{self.mass_dimension}"
            if self.current_dimension != 0:
                name += f" A^{self.current_dimension}"
            if self.temperature_dimension != 0:
                name += f" K^{self.temperature_dimension}"
            if self.amount_dimension != 0:
                name += f" mol^{self.amount_dimension}"
            if self.luminous_intensity_dimension != 0:
                name += f" cd^{self.luminous_intensity_dimension}"
            return name


    def set_name(self, name):
        """
        Sets the name and returns the unit. This is useful for defining derived units in one line, e.g.:
        calorie = (4.184 * joule).set_name('calorie')
        """
        self.name = name
        return self


    def dimensionality(self):
        """
        Returns a tuple of dimensionalits of the unit for comparison purposes.
        """
        return (float(dim) for dim in (self.time_dimension, self.length_dimension, self.mass_dimension,
                                        self.current_dimension, self.temperature_dimension, self.amount_dimension,
                                        self.luminous_intensity_dimension))
    

    def to_openmm(self):
        """
        Returns the corresponding openmm unit. This function is not optimized, it imports openmm, to make the openmm dependency optional.
        Thus, convert your units to the corresponding system beforehand and do all of the operations in one unit system.
        """
        from openmm import unit as openmm_unit
        openmm_kg = openmm_unit.gram * 1e3
        openmm_s = openmm_unit.second
        openmm_ampere = openmm_unit.ampere
        openmm_kelvin = openmm_unit.kelvin
        openmm_mole = openmm_unit.mole
        openmm_candela = openmm_unit.candela

        # we can simply construct the unit from dimensionality and the in_si_units factor, which defines it relatively to the SI unit system.
        this_openmm_unit = (openmm_s**self.time_dimension) * (openmm_kg**self.mass_dimension) * (openmm_ampere**self.current_dimension) * (openmm_kelvin**self.temperature_dimension) * (openmm_mole**self.amount_dimension) * (openmm_candela**self.luminous_intensity_dimension) * self.in_si_units

        return this_openmm_unit



kg = Unit(mass_dimension=1, name='kg')
m = Unit(length_dimension=1, name='m')
s = Unit(time_dimension=1, name='s')
ampere = Unit(current_dimension=1, name='A')
kelvin = Unit(temperature_dimension=1, name='K')
mol = Unit(amount_dimension=1, name='mol')
cd = Unit(luminous_intensity_dimension=1, name='cd')

rad = Unit(name='rad') # define radian as being 'in SI units'

second = s
kilogram = kg
meter = m
mole = mol
candela = cd
radians = rad

# derived SI units
degree = (3.14159265358979323846/180 * rad).set_name('degree')
joule = (kg * m**2 / s**2).set_name('Joule')
newton = (kg * m / s**2).set_name('Newton')
pascal = (kg / (m * s**2)).set_name('Pascal')
watt = (joule/s).set_name('Watt')
coulomb = (ampere * s).set_name('Coulomb')
volt = (watt/ampere).set_name('Volt')
farad = (coulomb/volt).set_name('Farad')
ohm = (volt/ampere).set_name('Ohm')
siemens = (ampere/volt).set_name('Siemens')
weber = (volt*s).set_name('Weber')
tesla = (weber/m/m).set_name('Tesla')
henry = (weber/ampere).set_name('Henry')
hertz = (1/s).set_name('Hertz')
lux = (cd/m/m).set_name('Lux')

# non-SI units
calorie = (4.184 * joule).set_name('Calorie')
electronvolt = (1.602176634e-19 * joule).set_name('Electronvolt')

kilocalorie = (1000 * calorie).set_name('Kilocalorie')
kilojoule = (1000 * joule).set_name('Kilojoule')

kcal = kilocalorie
kj = kilojoule

# here, kcal/mol is a unit of energy, not of 'energy per substance'. It is simply a scaled version of kcal: The amount of energy that corresponds to 1kcal when present in every particle of a mole of particles. (so it is 1/Avogadro * kcal)
# If you want to use kcal/mol as a unit of energy per substance, just use the actual expression kcal/mol.
AVOGADRO_CONSTANT = 6.02214076e23
kcal_per_mole = (kilocalorie / AVOGADRO_CONSTANT).set_name('kcal_per_mole')
kj_per_mole = (kilojoule / AVOGADRO_CONSTANT).set_name('kj/mol_per_mole')

kcal_per_mol = kcal_per_mole
kilocalories_per_mol = kcal_per_mole
kilocalories_per_mole = kcal_per_mole
kj_per_mol = kj_per_mole
kilojoules_per_mol = kj_per_mole
kilojoules_per_mole = kj_per_mole


nanometer = (1e-9 * m).set_name('Nanometer')
angstrom = (1e-10 * m).set_name('Angstrom')

elementary_charge = (1.602176634e-19 * coulomb).set_name('Elementary_charge')




class Quantity:
    """
    A class to represent a quantity with a unit. Inspired by openmm.unit.Quantity. (We don't need the full functionality of openmm.unit.Quantity and want to be independent of openmm.)
    Key functionality:
    - Arithmetic operations with other quantities and with scalars: Overload of addition, subtraction, multiplication and division operators.
    - Conversion to other units: quantity.in_unit(some_unit_with_fitting_dimensionality)

    When two quantities are added or subtracted, the dimensionality of the units must be the same. The result will have the same unit as the left-hand side of the input, i.e. 1*m + 20*cm = 1.2*m.
    """
    def __init__(self, value, unit):
        """
        Initialize a Quantity with a value and a unit.
        :param value: The numerical value of the quantity.
        :param unit: The unit of the quantity, an instance of the Unit class.
        """
        self.value = value
        self.unit = unit

    def __add__(self, other):
        """
        Add two quantities. They must have the same dimensionality.
        """
        if not isinstance(other, Quantity):
            raise ValueError("Can only add Quantity to Quantity.")
        if self.unit.dimensionality() != other.unit.dimensionality():
            raise ValueError("Dimensionality of units must match for addition.")
        # Convert other's value to self's unit before adding
        converted_value = other.in_unit(self.unit)
        return Quantity(self.value + converted_value, self.unit)

    def __sub__(self, other):
        """
        Subtract two quantities. They must have the same dimensionality.
        """
        if not isinstance(other, Quantity):
            raise ValueError("Can only subtract Quantity from Quantity.")
        if self.unit.dimensionality() != other.unit.dimensionality():
            raise ValueError("Dimensionality of units must match for subtraction.")
        # Convert other's value to self's unit before subtracting
        converted_value = other.in_unit(self.unit)
        return Quantity(self.value - converted_value, self.unit)

    def __mul__(self, other):
        """
        Multiply a quantity by another quantity or a scalar.
        """
        if isinstance(other, Quantity):
            new_unit = self.unit * other.unit
            new_value = self.value * other.value
        else:
            new_unit = self.unit
            new_value = self.value * other
        return Quantity(new_value, new_unit)
    
    def __rmul__(self, other):
        """
        Enable multiplication where other is on the left-hand side of the operator.
        """
        # The operation is commutative, so we can just return the result of __mul__
        return self.__mul__(other)


    def __truediv__(self, other):
        """
        Divide a quantity by another quantity or a scalar.
        """
        if isinstance(other, Quantity):
            new_unit = self.unit / other.unit
            new_value = self.value / other.value
        else:
            new_unit = self.unit
            new_value = self.value / other
        return Quantity(new_value, new_unit)
    
    def __rtruediv__(self, other):
        """
        Enable division where other is on the left-hand side of the operator.
        """
        if isinstance(other, Quantity):
            new_unit = other.unit / self.unit
            new_value = other.value / self.value
        else:
            new_unit = self.unit**-1
            new_value = other / self.value
        return Quantity(new_value, new_unit)

    def in_unit(self, other_unit):
        """
        Convert the quantity to a given unit. The dimensionality must match.
        """
        if self.unit.dimensionality() != other_unit.dimensionality():
            raise ValueError("Dimensionality of units must match for conversion.")
        converted_value = self.value * self.unit.in_si_units / other_unit.in_si_units
        return Quantity(converted_value, other_unit)

    def __repr__(self):
        return f"{self.value} {self.unit}"

    def dimensionality(self):
        """
        Returns the dimensionality of the unit of the quantity for comparison purposes.
        """
        return self.unit.dimensionality()