import ase
import pytest
from ase.io import read
from ase import Atoms, units
import numpy as np
from pathlib import Path
from ase.data import atomic_masses
# from ase.calculators.lammpslib import LAMMPSlib
from ase_extras.mylammps import LAMMPSlib
from ase.build import molecule
from ase.md.langevin import Langevin

# TIP4P/2005 water parameters
sigma_o_o = 3.1589 # In Angstroms 
epsilon_o_o = 0.1852 * units.kcal / units.mol # Originally in kcal/mol, converted to ASE units 

def add_bond_angle_commands(atoms:Atoms, amendments:list[str], bond_type:int, angle_type:int):
    """Add LAMMPS commands to create bonds and angles, given an atoms object with the ordering OHH for the water molecules

    Args:
        amendments (list[str]): list of strings with LAMMPS commands 
        bond_type (int): Bond type 
        angle_type (int): Angle type 
    """
    for atom in atoms:
        if atom.symbol=='O':
            o_id = atom.index + 1
            h1_id = o_id + 1
            h2_id = o_id + 2 
            # The IDs start from 1, not 0
            # Add bonds 
            amendments.append(f"create_bonds single/bond {bond_type} {o_id} {h1_id}")
            amendments.append(f"create_bonds single/bond {bond_type} {o_id} {h2_id}")
            # Add the angles 
            amendments.append(f"create_bonds single/angle {angle_type} {o_id} {h1_id} {h2_id} special no")

def get_bond_lengths(atoms):
    # just three atoms
    oh1_length = atoms.get_distance(0, 1)
    oh2_length = atoms.get_distance(0, 2)
    hh_length = atoms.get_distance(1, 2)
    return oh1_length, oh2_length, hh_length

def test_lammps_water():
    """
    Test that you can constrain a rigid water molecule using the LAMMPSlib calculator and ASE
    """
    test_dir = Path(__file__).resolve().parent
    xyz_infile = test_dir / "../resources/single_water.xyz"
    water = read(xyz_infile)

    # tip4p_constraints = []
    # for atom in water:
    #     if atom.symbol == "O":
    #         tip4p_constraints.append([atom.index, atom.index + 1])
    #         tip4p_constraints.append([atom.index, atom.index + 2])
    #         tip4p_constraints.append([atom.index + 1, atom.index + 2])
    # rattle_constraints = ase.constraints.FixBondLengths(tip4p_constraints)
    # water.set_constraint(rattle_constraints)

    # Parameters
    cutoff = 6.0
    o_atomic_mass = atomic_masses[8]
    h_atomic_mass = atomic_masses[1]
    o_charge = -1.1128
    h_charge = 0.5564
    oh_bond_type = 1
    ohh_angle_type = 1
    qm_distance = 0.1546
    temp_kT = 25.7e-3  # in eV at room temperature
    temperature = temp_kT / units.kB

    # ---------
    # Things for ASE - LAMMPS calculator 
    atom_types = {
        "O": 1,
        "H": 2,
    }  # LAMMPS needs atom types and can't use atomic numbers or symbols

    atomic_type_masses = {"O": o_atomic_mass, "H": h_atomic_mass}

    # list of strings of LAMMPS commands. You need to supply enough to define the potential to be used e.g. [“pair_style eam/alloy”, “pair_coeff * * potentials/NiAlH_jea.eam.alloy Ni Al”]
    cmds = [
        f"pair_style  lj/cut/tip4p/cut {atom_types['O']} {atom_types['H']} {oh_bond_type} {ohh_angle_type} {qm_distance} {cutoff}",
        "bond_style harmonic",
        "angle_style harmonic",
        f"pair_coeff {atom_types['O']} {atom_types['O']} {epsilon_o_o} {sigma_o_o}",
        f"pair_coeff {atom_types['H']} {atom_types['H']} 0.0 2.0",
        f"pair_coeff {atom_types['O']} {atom_types['H']} 0.0 2.0",
    ]

    lammps_header = ["units metal", "atom_style full", "atom_modify map array sort 0 0"]

    # extra list of strings of LAMMPS commands to be run post initialization. (Use: Initialization amendments) e.g.
    # [“mass 1 58.6934”]
    amendments = [f"mass {atom_types['O']} {o_atomic_mass}", 
                  f"mass {atom_types['H']} {h_atomic_mass}",
                  f"set type {atom_types['O']} charge {o_charge}",
                  f"set type {atom_types['H']} charge {h_charge}",
                  f"fix constraint all shake 1e-6 20 0 b {oh_bond_type} a {ohh_angle_type} t {atom_types['O']} {atom_types['H']}",
                  f"bond_coeff {oh_bond_type} 1000000 0.9572",
                  f"angle_coeff {ohh_angle_type} 1000000 104.52",
                  f"fix 2 all nvt temp {temperature} {temperature} 100 tchain 10"
                  ]

    # Add bonds and angles 
    bonds_angles = []
    add_bond_angle_commands(water, bonds_angles, oh_bond_type, ohh_angle_type)

    # Create the LAMMPS calculator 
    lammps_calc = LAMMPSlib(lmpcmds = cmds, atom_types= atom_types, atomic_type_masses = atomic_type_masses,lammps_header = lammps_header, amendments= amendments, n_bond_types=1, n_angle_types=1, bonds_per_atom = 2,
        angles_per_atom=1, bond_angle_creation=bonds_angles)
    # ---------

    timestep = 1.0 * units.fs

    water.calc = lammps_calc
    # energy_lmp = water.get_potential_energy()
    water.calc.calculate(water, properties=["energy, forces"], system_changes=["positions"])
    energy_lmp2 = water.calc.results["energy"]
    energies = water.calc.results["energies"]
    breakpoint()

    # Check the old bond lengths 
    oh1_init, oh2_init, hh_init = get_bond_lengths(water)

    # Check the number of bonds 
    n_bonds, all_bonds = water.calc.lmp.gather_bonds()
    n_angles, all_angles = water.calc.lmp.gather_angles()

    assert n_angles == 1
    assert n_bonds == 2

    dyn = Langevin(
        water,
        timestep=timestep,
        temperature_K=temp_kT / units.kB,
        friction=1,
        fixcm=False,
    )
    
    dyn.run(1)

    # Get the final bond lengths after running the simulation
    oh1_final, oh2_final, hh_final = get_bond_lengths(water)

    assert oh1_init == oh1_final
    assert oh2_init == oh2_final
    assert hh_init == hh_final
    


if __name__ == "__main__":
    test_lammps_water()
