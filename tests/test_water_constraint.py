import numpy as np
import pytest
from ase_extras.rigid_water_constraint import MyWaterConstraint
from ase.build import molecule


def get_bond_lengths(atoms):
    # just three atoms
    oh1_length = atoms.get_distance(0, 1)
    oh2_length = atoms.get_distance(0, 2)
    hh_length = atoms.get_distance(1, 2)
    return oh1_length, oh2_length, hh_length


def test_constraint():
    # Create a water molecule
    atoms = molecule("H2O")
    a_idx = 0
    h1_pos = atoms[1].position
    h2_pos = atoms[2].position

    oh1_len_desired, oh2_len_desired, hh_len_desired = get_bond_lengths(atoms)

    constraint = MyWaterConstraint(a_idx, h1_pos, h2_pos)

    newforces = np.zeros(shape=(len(atoms), 3))
    newforces[0] += np.array([3.0, 4.5, 2.0])
    newpositions = atoms.get_positions()
    newpositions[0] += np.array([1.6, 0.5, 0.7])

    constraint.adjust_positions(atoms, newpositions)
    atoms.set_positions(newpositions)

    oh1_len_adj, oh2_len_adj, hh_len_adj = get_bond_lengths(atoms)

    print(f"{oh1_len_desired}")
    print(f"{oh2_len_desired}")
    print(f"{hh_len_desired}")

    print(f"{oh1_len_adj}")
    print(f"{oh2_len_adj}")
    print(f"{hh_len_adj}")

    assert np.isclose(oh1_len_adj, oh1_len_desired, atol=1e-5)
    assert np.isclose(oh2_len_adj, oh2_len_desired, atol=1e-5)
    assert np.isclose(hh_len_adj, hh_len_desired, atol=1e-5)


if __name__ == "__main__":
    test_constraint()
