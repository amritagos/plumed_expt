import numpy as np


def orthogonalize(vector1, vector2):
    vector2 = vector2 / np.linalg.norm(vector2)
    return vector1 - np.dot(vector1, vector2) * vector2


class MyWaterConstraint:
    """Constrain bonds and angles in water molecules. Only works if H positions are not moved"""

    def __init__(self, a: int, h1_pos, h2_pos):
        self.a = a  # index of O atom
        self.h1_pos = h1_pos
        self.h2_pos = h2_pos

    def adjust_positions(self, atoms, newpositions):
        h_center = 0.5 * (self.h1_pos + self.h2_pos)
        new_o_pos = newpositions[self.a]  # which must be moved
        old_o_pos = atoms[self.a].position
        bond_length = np.linalg.norm(old_o_pos - self.h1_pos)
        hh_distance = np.linalg.norm(self.h1_pos - self.h2_pos)
        # line from orthogonalized new_o_pos to center
        new_dir = new_o_pos - h_center
        new_dir = orthogonalize(new_dir, self.h1_pos - self.h2_pos)
        new_dir = new_dir / np.linalg.norm(new_dir)  # unit vector
        distance_from_center = np.sqrt(bond_length**2 - (0.5 * hh_distance) ** 2)
        newpositions[self.a] = h_center + distance_from_center * new_dir

    def adjust_forces(self, atoms, forces):
        dir = np.cross(
            self.h1_pos - atoms[self.a].position, self.h2_pos - atoms[self.a].position
        )
        dir = dir / np.linalg.norm(dir)
        forces[self.a] = dir * np.dot(forces[self.a], dir)
