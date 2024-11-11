import argparse
import json
from pathlib import Path
from ase import Atoms, units
import ase
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.calculators.plumed import Plumed
from ase.calculators.lj import LennardJones
from ase_extras.tip4p_2005 import TIP4P_2005
from ase_extras.subsystem_calculator import SubsystemCalculator
from ase.md.langevin import Langevin

# Inspired by tutorial PLUMED Masterclass 21.3

# Na, Cl and water parameters taken from Madrid-19 force-field
# 10.1063/1.5121392
# Sigma in Angstrom from Table III
sigma_na_na = 2.21737
sigma_na_cl = 3.00512
sigma_o_na = 2.60838
sigma_o_cl = 4.23867

# epsilon parameters (originally in kJ/mol) in the paper (from Table IV)
# To convert to ASE units (eV), multiply by the factor units.kJ/units.mol
epsilon_na_na = 1.472356 * units.kJ / units.mol
epsilon_na_cl = 1.438894 * units.kJ / units.mol
epsilon_o_na = 0.793388 * units.kJ / units.mol
epsilon_o_cl = 0.061983 * units.kJ / units.mol


def create_subset_calculator(atoms: Atoms, max_cutoff: float):
    """Creates the subset calculator, with TIP4P/2005 for water, Lennard Jones parameters for other interactions. Hard-coded for now

    Args:
        atoms (Atoms): Total system atoms
        max_cutoff(float): Cutoff for interactions to prevent interactions with periodic images

    Returns:
        ase.Calculator: returned calculator object
    """
    masks = []
    calculators = []
    
    # water interactions
    mask_water = [atom.symbol in ["O", "H"] for atom in atoms]
    calc_water = TIP4P_2005(rc=max_cutoff)
    masks.append(mask_water)
    calculators.append(calc_water)
    # # Na-Na interactions
    # mask_na_na = [atom.symbol == "Na" for atom in atoms]
    # calc_na_na = LennardJones(sigma=sigma_na_na, epsilon=epsilon_na_na, rc=max_cutoff)
    # masks.append(mask_na_na)
    # calculators.append(calc_na_na)
    # Na-Cl interactions
    mask_na_cl = [atom.symbol in ["Na", "Cl"] for atom in atoms]
    calc_na_cl = LennardJones(sigma=sigma_na_cl, epsilon=epsilon_na_cl, rc=max_cutoff)
    masks.append(mask_na_cl)
    calculators.append(calc_na_cl)
    # O-Na interactions
    mask_o_na = [atom.symbol in ["O", "Na"] for atom in atoms]
    calc_o_na = LennardJones(sigma=sigma_o_na, epsilon=epsilon_o_na, rc=max_cutoff)
    masks.append(mask_o_na)
    calculators.append(calc_o_na)
    # O-Cl interactions
    mask_o_cl = [atom.symbol in ["O", "Cl"] for atom in atoms]
    calc_o_cl = LennardJones(sigma=sigma_o_cl, epsilon=epsilon_o_cl, rc=max_cutoff)
    masks.append(mask_o_cl)
    calculators.append(calc_o_cl)

    calc_subset = SubsystemCalculator(masks=masks, calculators=calculators)
    return calc_subset


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
    atoms = read(in_xyz_file)

    # Get the calculator for the system object
    calc_subset = create_subset_calculator(atoms, max_cutoff)

    timestep = 1.0 * units.fs
    temp_kT = 25.7e-3  # in eV at room temperature

    setup = [
        f"UNITS LENGTH=A TIME=fs ENERGY=eV",
        "d1: DISTANCE ATOMS=1,2",
        f"restraint: RESTRAINT ARG=d1 AT={ion_distance} KAPPA=150.0",
        f"PRINT ARG=d1,restraint.bias FILE={colvar_file} STRIDE=100",
    ]

    atoms.calc = Plumed(
        calc=calc_subset, input=setup, timestep=timestep, atoms=atoms, kT=temp_kT
    )

    dyn = Langevin(
        atoms,
        timestep=timestep,
        temperature_K=temp_kT / units.kB,
        friction=1,
        fixcm=False,
    )
    # dyn.run(500) # equilibration
    traj = Trajectory(traj_file, "w", atoms)
    dyn.attach(traj.write, interval=100)
    dyn.run(n_steps)  # 200000 in tutorial

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
