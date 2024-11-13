import pytest
from ase.io import read
from ase import units
import numpy as np
from pathlib import Path
from ase.data import atomic_masses
from ase.calculators.lammpslib import LAMMPSlib

# Na, Cl and water parameters taken from Madrid-19 force-field
# 10.1063/1.5121392
# Sigma in Angstrom from Table III
sigma_na_na = 2.21737
sigma_na_cl = 3.00512
sigma_o_na = 2.60838
sigma_o_cl = 4.23867

# epsilon parameters (originally in kJ/mol) in the paper (from Table IV)
# To convert to ASE units (eV), multiply by the factor units.kJ/units.mol
# In LAMMPS, use units metal to be consistent with ASE
epsilon_na_na = 1.472356 * units.kJ / units.mol
epsilon_na_cl = 1.438894 * units.kJ / units.mol
epsilon_o_na = 0.793388 * units.kJ / units.mol
epsilon_o_cl = 0.061983 * units.kJ / units.mol


def lennard_jones_energy(r_ij, sigma_ij, epsilon_ij):
    r_dimless = sigma_ij / r_ij
    return 4 * epsilon_ij * (r_dimless**12 - r_dimless**6)


def test_lammps_lib():
    """
    Test the LAMMPSlib calculator in ASE
    """
    test_dir = Path(__file__).resolve().parent
    xyz_infile = test_dir / "../resources/lj_test.xyz"
    # Read in the input configuration with just one Na and O
    atoms = read(xyz_infile)
    atoms.pbc = [True, True, True]
    # Parameters
    lj_cutoff = 6.0
    o_atomic_mass = atomic_masses[8]
    na_atomic_mass = atomic_masses[11]

    # ---------
    atom_types = {
        "O": 1,
        "Na": 2,
    }  # LAMMPS needs atom types and can't use atomic numbers or symbols

    # list of strings of LAMMPS commands. You need to supply enough to define the potential to be used e.g. [“pair_style eam/alloy”, “pair_coeff * * potentials/NiAlH_jea.eam.alloy Ni Al”]
    cmds = [
        f"pair_style lj/cut {lj_cutoff}",
        "pair_coeff 1 1 0.0 0.0",
        "pair_coeff 2 2 0.0 0.0",
        f"pair_coeff 1 2 {epsilon_o_na} {sigma_o_na}",
    ]

    amendments = [f"mass 1 {o_atomic_mass}", f"mass 2 {na_atomic_mass}"]
    lammps_calc = LAMMPSlib(lmpcmds=cmds, atom_types=atom_types, amendments=amendments)
    # ---------
    atoms.calc = lammps_calc
    energy_lmp = atoms.get_potential_energy()

    # Check atom IDs in LAMMPS?
    nlocal = atoms.calc.lmp.extract_global("nlocal")
    ids = atoms.calc.lmp.extract_atom("id")
    x = atoms.calc.lmp.extract_atom("x")
    expected_ids = [1, 2]

    assert nlocal == len(atoms)

    for i in range(nlocal):
        assert ids[i] == expected_ids[i]
        # Check the position
        for k in range(3):
            assert x[i][k] == atoms[i].position[k]

    # ---------
    # Check with hand-written LJ potential
    r_ij = atoms.get_distance(0, 1, mic=False)
    energy_ref = lennard_jones_energy(r_ij, sigma_o_na, epsilon_o_na)
    assert np.isclose(energy_lmp, energy_ref, atol=1e-6)


if __name__ == "__main__":
    test_lammps_lib()
