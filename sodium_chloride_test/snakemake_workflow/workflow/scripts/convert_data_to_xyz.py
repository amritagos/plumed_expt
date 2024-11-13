import argparse
from pathlib import Path
from typing import List
from ase_extras.ase_read_write_data import read_lammps_data
from ase.io import write

import numpy as np


def main(
    data_file: Path,
    output_xyz: Path,
):
    o_atomic_num = 8
    h_atomic_num = 1
    na_atomic_num = 11
    cl_atomic_num = 17
    Z_type_dict = {1: o_atomic_num, 2: h_atomic_num, 3: na_atomic_num, 4: cl_atomic_num}
    # Read in the LAMMPS data file
    with open(data_file, "r") as f:
        atoms = read_lammps_data(f, Z_of_type=Z_type_dict, atom_style="full")

    # Write out the XYZ file
    write(output_xyz, atoms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LAMMPS data files to extended XYZ files."
    )
    parser.add_argument(
        "--data_file", type=Path, required=True, help="Path to the data file to convert"
    )
    parser.add_argument(
        "--output_xyz",
        type=Path,
        required=True,
        help="Path to XYZ file that will be created",
    )

    args = parser.parse_args()

    main(
        args.data_file,
        args.output_xyz,
    )
