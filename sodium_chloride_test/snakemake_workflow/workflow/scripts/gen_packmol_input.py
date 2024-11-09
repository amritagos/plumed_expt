import argparse
from pathlib import Path
from typing import List

import numpy as np


def generate_packmol_input(
    cation_file: Path,
    cl_file: Path,
    tip4p_file: Path,
    system_file: Path,
    cation_position: np.ndarray,
    anion_position: np.ndarray,
    n_waters: int,
    box_dims: np.ndarray,
    output_path: Path,
):

    # Format the output text based on inputs
    packmol_content = f"""# All atoms from different molecules will be at least 2.0 Angstroms apart
# from each other at the solution.

tolerance 2.0

# The files are in the simple Molden-xyz format

filetype xyz

# The name of the output file.

output {system_file}

structure {cation_file}
  number 1
  center
  fixed {cation_position[0]} {cation_position[1]} {cation_position[2]} 0. 0. 0.
  inside box 0. 0. 0. {box_dims[0]} {box_dims[1]} {box_dims[2]}
end structure

structure {cl_file}
  number 1
  fixed {anion_position[0]} {anion_position[1]} {anion_position[2]} 0. 0. 0.
  inside box 0. 0. 0. {box_dims[0]} {box_dims[1]} {box_dims[2]}
end structure

structure {tip4p_file}
  number {n_waters}
  inside box 0. 0. 0. {box_dims[0]} {box_dims[1]} {box_dims[2]}
end structure
"""
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to output file
    with output_path.open("w") as f:
        f.write(packmol_content)


def main(
    cation_file: Path,
    anion_file: Path,
    water_file: Path,
    system_file: Path,
    ion_distance: float,
    n_wat: int,
    output_path: Path,
    box_dims: List[float],
):
    cation_position = 0.5 * np.array(box_dims)
    # Place the anion at ion_distance away in the x dimension
    anion_position = cation_position + np.array([ion_distance, 0.0, 0.0])
    generate_packmol_input(
        cation_file,
        anion_file,
        water_file,
        system_file,
        cation_position,
        anion_position,
        n_wat,
        box_dims,
        output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Packmol input file for a system."
    )
    parser.add_argument(
        "--cation_file", type=Path, required=True, help="Path to cation .xyz file"
    )
    parser.add_argument(
        "--anion_file", type=Path, required=True, help="Path to anion .xyz file"
    )
    parser.add_argument(
        "--water_file", type=Path, required=True, help="Path to TIP4P/2005 .xyz file"
    )
    parser.add_argument(
        "--system_file",
        type=Path,
        required=True,
        help="Path to the packmol output system .xyz file",
    )
    parser.add_argument(
        "--ion_distance",
        type=float,
        required=True,
        help="Distance between cation and anion. Cation is placed in the center",
    )
    parser.add_argument(
        "--n_wat",
        type=int,
        required=True,
        help="Number of water molecules in the box",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to Packmol input file that will be created",
    )
    parser.add_argument(
        "--box_dims", type=float, nargs=3, required=True, help="Box dimensions (x y z)"
    )

    args = parser.parse_args()

    main(
        args.cation_file,
        args.anion_file,
        args.water_file,
        args.system_file,
        args.ion_distance,
        args.n_wat,
        args.output_path,
        args.box_dims,
    )
