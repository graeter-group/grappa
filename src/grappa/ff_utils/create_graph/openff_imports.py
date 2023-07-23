
def get_openff_imports():
    from openmmforcefields.generators import SystemGenerator
    from openff.toolkit.topology import Molecule

    return SystemGenerator, Molecule