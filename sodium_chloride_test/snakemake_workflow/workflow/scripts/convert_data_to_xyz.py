import argparse
from pathlib import Path
from typing import List
from ase_read_write_data import read_lammps_data
from ase.io import write

import numpy as np


def main(
    data_file: Path,
    cation_type_data: int,
    cation_type_ase: int,
    anion_type_data: int,
    anion_type_ase: int,
    o_type_data: int,
    o_type_ase: int,
    h_type_data: int,
    h_type_ase: int,
    output_xyz: Path,
):

    # Read in the LAMMPS data file
    with open(data_file, "r") as f:
        atoms = read_lammps_data(f, atom_style="full")

    # Reassign atom types
    for atom in atoms:
        if atom.number == cation_type_data:
            atom.number = cation_type_ase
        elif atom.number == anion_type_data:
            atom.number = anion_type_ase
        elif atom.number == o_type_data:
            atom.number = o_type_ase
        elif atom.number == h_type_data:
            atom.number = h_type_ase

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
        "--cation_type_data",
        type=int,
        required=True,
        help="Atom number of the cation in the LAMMPS data file",
    )
    parser.add_argument(
        "--cation_type_ase",
        type=int,
        required=True,
        help="Atom number of cation, to be written to XYZ (ASEs)",
    )
    parser.add_argument(
        "--anion_type_data",
        type=int,
        required=True,
        help="Atom number of the anion in the LAMMPS data file",
    )
    parser.add_argument(
        "--anion_type_ase",
        type=int,
        required=True,
        help="Atom number of anion, to be written to XYZ (ASEs)",
    )
    parser.add_argument(
        "--o_type_data",
        type=int,
        required=True,
        help="Atom number of O in the LAMMPS data file",
    )
    parser.add_argument(
        "--o_type_ase",
        type=int,
        required=True,
        help="Atom number of O, to be written to XYZ (ASEs)",
    )
    parser.add_argument(
        "--h_type_data",
        type=int,
        required=True,
        help="Atom number of H in the LAMMPS data file",
    )
    parser.add_argument(
        "--h_type_ase",
        type=int,
        required=True,
        help="Atom number of H, to be written to XYZ (ASEs)",
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
        args.cation_type_data,
        args.cation_type_ase,
        args.anion_type_data,
        args.anion_type_ase,
        args.o_type_data,
        args.o_type_ase,
        args.h_type_data,
        args.h_type_ase,
        args.output_xyz,
    )
