import argparse
import json
from pathlib import Path
from ase import Atoms, units
import ase
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.calculators.plumed import Plumed
from ase.md.langevin import Langevin
from ase.data import atomic_masses
from ase_extras.mylammps import LAMMPSlib

# Inspired by tutorial PLUMED Masterclass 21.3

# Na, Cl and water parameters taken from Madrid-19 force-field
# 10.1063/1.5121392
# Sigma in Angstrom from Table III
sigma_na_na = 2.21737
sigma_na_cl = 3.00512
sigma_o_na = 2.60838
sigma_o_cl = 4.23867
sigma_cl_cl = 4.69906

# epsilon parameters (originally in kJ/mol) in the paper (from Table IV)
# To convert to ASE units (eV), multiply by the factor units.kJ/units.mol
epsilon_na_na = 1.472356 * units.kJ / units.mol
epsilon_na_cl = 1.438894 * units.kJ / units.mol
epsilon_o_na = 0.793388 * units.kJ / units.mol
epsilon_o_cl = 0.061983 * units.kJ / units.mol
epsilon_cl_cl = 0.076923 * units.kJ / units.mol

# TIP4P/2005 water parameters
sigma_o_o = 3.1589 # In Angstroms 
epsilon_o_o = 0.1852 * units.kcal / units.mol # Originally in kcal/mol, converted to ASE units 

def add_bond_angle_commands(
    atoms: Atoms, amendments: list[str], bond_type: int, angle_type: int
):
    """Add LAMMPS commands to create bonds and angles, given an atoms object with the ordering OHH for the water molecules

    Args:
        amendments (list[str]): list of strings with LAMMPS commands
        bond_type (int): Bond type
        angle_type (int): Angle type
    """
    for atom in atoms:
        if atom.symbol == "O":
            o_id = atom.index + 1
            h1_id = o_id + 1
            h2_id = o_id + 2
            # The IDs start from 1, not 0
            # Add bonds
            amendments.append(f"create_bonds single/bond {bond_type} {o_id} {h1_id}")
            amendments.append(f"create_bonds single/bond {bond_type} {o_id} {h2_id}")
            # Add the angles
            amendments.append(
                f"create_bonds single/angle {angle_type} {h1_id} {o_id} {h2_id} special no"
            )

def main(
    in_xyz_file: Path,
    max_cutoff: float,
    ion_distance: float,
    n_steps: int,
    colvar_file: Path,
    traj_file: Path,
    metadata_file: Path,
):
    # Read the entire system
    system = read(in_xyz_file)

    tip4p_constraints = []
    for atom in system:
        if atom.symbol == "O":
            tip4p_constraints.append([atom.index, atom.index + 1])
            tip4p_constraints.append([atom.index, atom.index + 2])
            tip4p_constraints.append([atom.index + 1, atom.index + 2])
    rattle_constraints = ase.constraints.FixBondLengths(tip4p_constraints)
    system.set_constraint(rattle_constraints)

    # Create the LAMMPS calculator object
    # -----------------------------------------------------
    # Parameters
    cutoff = 6.0
    o_atomic_mass = atomic_masses[8]
    h_atomic_mass = atomic_masses[1]
    na_atomic_mass = atomic_masses[11]
    cl_atomic_mass = atomic_masses[17]
    o_charge = -1.1128
    h_charge = 0.5564
    na_charge = 0.85
    cl_charge = -0.85
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
        "Na": 3,
        "Cl": 4,
    }  # LAMMPS needs atom types and can't use atomic numbers or symbols

    atomic_type_masses = {
        "O": o_atomic_mass,
        "H": h_atomic_mass,
        "Na": na_atomic_mass,
        "Cl": cl_atomic_mass,
    }

    # list of strings of LAMMPS commands. You need to supply enough to define the potential to be used e.g. [“pair_style eam/alloy”, “pair_coeff * * potentials/NiAlH_jea.eam.alloy Ni Al”]
    cmds = [
        f"pair_style  lj/cut/tip4p/cut {atom_types['O']} {atom_types['H']} {oh_bond_type} {ohh_angle_type} {qm_distance} {cutoff}",
        "bond_style harmonic",
        "angle_style harmonic",
        f"pair_coeff {atom_types['O']} {atom_types['O']} {epsilon_o_o} {sigma_o_o}",
        f"pair_coeff {atom_types['H']} {atom_types['H']} 0.0 2.0",
        f"pair_coeff {atom_types['O']} {atom_types['H']} 0.0 2.0",
        f"pair_coeff {atom_types['O']} {atom_types['Na']} {epsilon_o_na} {sigma_o_na}",
        f"pair_coeff {atom_types['Na']} {atom_types['H']} 0.0 2.0",
        f"pair_coeff {atom_types['Na']} {atom_types['Na']} {epsilon_na_na} {sigma_na_na}",
        f"pair_coeff {atom_types['Cl']} {atom_types['Cl']} {epsilon_cl_cl} {sigma_cl_cl}",
        f"pair_coeff {atom_types['O']} {atom_types['Cl']} {epsilon_o_cl} {sigma_o_cl}",
        f"pair_coeff {atom_types['H']} {atom_types['Cl']} 0.0 2.0",
        f"pair_coeff {atom_types['Na']} {atom_types['Cl']} {epsilon_na_cl} {sigma_na_cl}",
    ]

    lammps_header = ["units metal", "atom_style full", "atom_modify map array sort 0 0"]

    # extra list of strings of LAMMPS commands to be run post initialization. (Use: Initialization amendments) e.g.
    # [“mass 1 58.6934”]
    amendments = [
        f"mass {atom_types['O']} {o_atomic_mass}",
        f"mass {atom_types['H']} {h_atomic_mass}",
        f"mass {atom_types['Na']} {na_atomic_mass}",
        f"mass {atom_types['Cl']} {cl_atomic_mass}",
        f"set type {atom_types['O']} charge {o_charge}",
        f"set type {atom_types['H']} charge {h_charge}",
        f"set type {atom_types['Na']} charge {na_charge}",
        f"set type {atom_types['Cl']} charge {cl_charge}",
        f"fix constraint all shake 1e-6 20 0 b {oh_bond_type} a {ohh_angle_type} t {atom_types['O']} {atom_types['H']}",
        f"bond_coeff {oh_bond_type} 1000000 0.9572",
        f"angle_coeff {ohh_angle_type} 1000000 104.52",
    ]

    # Add bonds and angles
    bonds_angles = []
    add_bond_angle_commands(system, bonds_angles, oh_bond_type, ohh_angle_type)

    # Create the LAMMPS calculator
    lammps_calc = LAMMPSlib(
        lmpcmds=cmds,
        atom_types=atom_types,
        atomic_type_masses=atomic_type_masses,
        lammps_header=lammps_header,
        amendments=amendments,
        n_bond_types=1,
        n_angle_types=1,
        bonds_per_atom=2,
        angles_per_atom=1,
        bond_angle_creation=bonds_angles,
    )
    # -----------------------------------------------------
    timestep = 1.0 * units.fs
    
    # setup = [
    #     f"UNITS LENGTH=A TIME=fs ENERGY=eV",
    #     "d1: DISTANCE ATOMS=1,2",
    #     f"restraint: RESTRAINT ARG=d1 AT={ion_distance} KAPPA=150.0",
    #     f"PRINT ARG=d1,restraint.bias FILE={colvar_file} STRIDE=100",
    # ]

    # atoms.calc = Plumed(
    #     calc=lammps_calc, input=setup, timestep=timestep, atoms=atoms, kT=temp_kT
    # )

    system.calc = lammps_calc

    dyn = Langevin(
        system,
        timestep=timestep,
        temperature_K=temp_kT / units.kB,
        friction=1,
        fixcm=False,
    )
    # dyn.run(500) # equilibration
    # traj = Trajectory(traj_file, "w", atoms)
    # dyn.attach(traj.write, interval=100)
    dyn.run(1)  # 200000 in tutorial

    # Write out metadata into a JSON file
    with open(metadata_file, "w") as f:
        f.write(
            json.dumps(
                dict(
                    distance=ion_distance,
                    max_cutoff=max_cutoff,
                )
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Umbrella sampling with PLUMED and ASE."
    )
    parser.add_argument(
        "--in_xyz_file", type=Path, required=True, help="Path to input .xyz file"
    )
    parser.add_argument(
        "--max_cutoff",
        type=float,
        required=True,
        help="Cutoff for LJ and TIP4P/2005 interactions; must be less than or equal to half the box size",
    )
    parser.add_argument(
        "--ion_distance", type=float, required=True, help="Ion-ion distance"
    )
    parser.add_argument(
        "--n_steps", type=int, required=True, help="Number of (biased) MD steps"
    )
    parser.add_argument(
        "--colvar_file", type=Path, required=True, help="Unique COLVAR file output path"
    )
    parser.add_argument(
        "--traj_file", type=Path, required=True, help="Path to output trajectory file"
    )
    parser.add_argument(
        "--metadata_file", type=Path, required=True, help="Path to output metadata file"
    )

    args = parser.parse_args()

    main(
        args.in_xyz_file,
        args.max_cutoff,
        args.ion_distance,
        args.n_steps,
        args.colvar_file,
        args.traj_file,
        args.metadata_file,
    )
