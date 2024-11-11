from pathlib import Path
from typing import List
import pytest
from ase import Atoms
import numpy as np
from ase_extras.tip4p_2005 import TIP4P_2005
from ase.calculators.lj import LennardJones
from ase_extras.subsystem_calculator import SubsystemCalculator

from ase.io import read


def get_system_subset(atoms: Atoms, keep_species: List[str]) -> Atoms:
    """Create a system subset of the original Atoms object

    Args:
        atoms (Atoms): Original Atoms object
        keep_species (List[str]): indices to of species to keep

    Returns:
        Atoms: Atoms object subset
    """
    subset_atoms = atoms.copy()
    delete_indices = []

    for i, atom in enumerate(subset_atoms):
        if atom.symbol in keep_species:
            continue
        else:
            delete_indices.append(i)
    # Delete the unwanted species
    del subset_atoms[[i for i in delete_indices]]
    return subset_atoms


def test_ase_extras_calculator():
    test_dir = Path(__file__).resolve().parent
    xyz_infile = test_dir / "../resources/na_wat_system.xyz"
    system = read(xyz_infile)
    masks = []
    calculators = []
    mask_o_na = [atom.symbol in ["O", "Na"] for atom in system]
    o_na_subset = system[mask_o_na]

    calc_o_na = LennardJones(sigma=1.0, epsilon=1.5, rc=10)

    masks.append(mask_o_na)
    calculators.append(calc_o_na)

    # Water
    mask_water = [atom.symbol in ["O", "H"] for atom in system]
    water_subset = system[mask_water]
    calc_water = TIP4P_2005(rc=6.0)
    masks.append(mask_water)
    calculators.append(calc_water)
    # Use subsystem calculator
    subsys_calc = SubsystemCalculator(masks=masks, calculators=calculators)
    subsys_calc.calculate(system)
    forces_calc = subsys_calc.results["forces"]
    energy_calc = subsys_calc.results["energy"]

    # Check
    energy = 0.0
    calc_o_na.calculate(o_na_subset)
    energy += calc_o_na.results["energy"]
    forces_o_na = calc_o_na.results["forces"]

    forces = np.zeros(shape=(len(system), 3))
    forces[mask_o_na] = forces_o_na

    calc_water.calculate(water_subset)
    forces[mask_water] += calc_water.results["forces"]
    energy += calc_water.results["energy"]

    assert np.all(forces_calc == forces)
    assert energy_calc == energy

    system.calc = subsys_calc
    system_energy = system.get_potential_energy()
    assert system_energy == energy
    assert np.all(forces_calc == system.get_forces())


if __name__ == "__main__":
    test_ase_extras_calculator()
