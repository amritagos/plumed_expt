import ase
import pytest
from ase.io import read
from ase import Atoms, units
import numpy as np
from pathlib import Path
from ase.io.lammpsdata import read_lammps_data, write_lammps_data


def set_bond_array(atoms: Atoms, bondtype1: int, bondtype2: int):
    bonds_in = []

    # Set the bonds
    for atom in atoms:
        if atom.symbol == "O":
            o_id = atom.index + 1
            h1_id = o_id + 1
            h2_id = o_id + 2
            # type atom1 atom2
            bonds_in.append([bondtype1, o_id, h1_id])  # OH1
            bonds_in.append([bondtype1, o_id, h2_id])  # OH2
            bonds_in.append([bondtype2, h1_id, h2_id])  # H1H2

    bonds = [""] * len(atoms) if len(bonds_in) > 0 else None

    if bonds is not None:
        for atom_type, at1, at2 in bonds_in:
            i_a1 = at1 - 1
            i_a2 = at2 - 1
            if len(bonds[i_a1]) > 0:
                bonds[i_a1] += ","
            bonds[i_a1] += f"{i_a2:d}({atom_type:d})"
        for i, bond in enumerate(bonds):
            if len(bond) == 0:
                bonds[i] = "_"
        atoms.arrays["bonds"] = np.array(bonds)


def set_charges(atoms, charge_dict):
    charges = []
    for atom in atoms:
        q = charge_dict[atom.symbol]
        charges.append(q)
    atoms.set_initial_charges(charges)


def test_lammps_water():
    """
    Test that you can constrain a rigid water molecule using the LAMMPSlib calculator and ASE
    """
    test_dir = Path(__file__).resolve().parent
    xyz_infile = test_dir / "../resources/single_water.xyz"
    water = read(xyz_infile)
    outdata = test_dir / "water_out.data"
    o_charge = -1.1128
    h_charge = 0.5564

    set_bond_array(water, 1, 2)

    set_charges(water, {"O": o_charge, "H": h_charge})

    bonds_out = water.arrays.get("bonds")
    charges_out = water.get_initial_charges()

    # Write out the bonds
    with open(outdata, "w") as f:
        write_lammps_data(
            f,
            water,
            specorder=["O", "H"],
            reduce_cell=False,
            force_skew=False,
            prismobj=None,
            write_image_flags=False,
            masses=True,
            velocities=False,
            units="metal",
            bonds=True,
            atom_style="full",
        )
    # Read in the file again
    Z_of_type = {1: 8, 2: 1}
    with open(outdata, "r") as f:
        water_from_file = read_lammps_data(
            f,
            Z_of_type=Z_of_type,
            sort_by_id=True,
            read_image_flags=True,
            units="metal",
            atom_style="full",
        )
    bonds_in = water_from_file.arrays.get("bonds")
    charges_in = water_from_file.get_initial_charges()

    for bondi, bondj in zip(bonds_in, bonds_out):
        assert bondi == bondj

    for chargei, chargej in zip(charges_in, charges_out):
        assert chargei == chargej

    outdata.unlink()


if __name__ == "__main__":
    test_lammps_water()
