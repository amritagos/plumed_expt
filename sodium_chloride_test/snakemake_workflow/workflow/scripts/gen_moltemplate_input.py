import argparse
from pathlib import Path
from typing import List

import numpy as np


def generate_moltemplate_input(
    cation_file: Path,
    anion_file: Path,
    water_file: Path,
    n_cations: int,
    n_anions: int,
    n_waters: int,
    box_dims: np.ndarray,
    output_path: Path,
):

    # Format the output text based on inputs
    moltemplate_input_content = f"""import \"{str(water_file.resolve())}\"  # <- This defines the TIP4P/2005 water molecule.
import \"{str(cation_file.resolve())}\"
import \"{str(anion_file.resolve())}\"

fe = new FeIon[{n_cations}] 
cl = new ClIon[{n_anions}]

wat  = new TIP4P_2005 [{n_waters}]

# Periodic boundary conditions:
write_once("Data Boundary") {{
   0.0  {box_dims[0]}  xlo xhi
   0.0  {box_dims[1]}  ylo yhi
   0.0  {box_dims[2]}  zlo zhi
}}
"""
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to output file
    with output_path.open("w") as f:
        f.write(moltemplate_input_content)


def main(
    cation_file: Path,
    anion_file: Path,
    water_file: Path,
    n_cations: int,
    n_anions: int,
    n_wat: int,
    box_dims: List[float],
    output_path: Path,
):

    generate_moltemplate_input(
        cation_file,
        anion_file,
        water_file,
        n_cations,
        n_anions,
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
        "--n_cations",
        type=int,
        required=True,
        help="Number of cations in the box",
    )
    parser.add_argument(
        "--n_anions",
        type=int,
        required=True,
        help="Number of anions in the box",
    )
    parser.add_argument(
        "--n_wat",
        type=int,
        required=True,
        help="Number of water molecules in the box",
    )
    parser.add_argument(
        "--box_dims", type=float, nargs=3, required=True, help="Box dimensions (x y z)"
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to Packmol input file that will be created",
    )

    args = parser.parse_args()

    main(
        args.cation_file,
        args.anion_file,
        args.water_file,
        args.n_cations,
        args.n_anions,
        args.n_wat,
        args.box_dims,
        args.output_path,
    )
