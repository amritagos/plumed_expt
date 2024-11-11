import ase
from ase.calculators.calculator import BaseCalculator, Calculator, all_changes
from ase_extras.rigid_water_constraint import MyWaterConstraint
import numpy as np


class SubsystemCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        masks,
        calculators: list[Calculator],
        restart=None,
        ignore_bad_restart_file=BaseCalculator._deprecated,
        label=None,
        atoms=None,
        directory=".",
        **kwargs
    ):
        super().__init__(
            restart, ignore_bad_restart_file, label, atoms, directory, **kwargs
        )
        self.masks = masks
        self.calculators = calculators
        self.n_total_atoms = len(masks[0])
        for mask in masks:
            assert len(mask) == self.n_total_atoms

    def calculate(
        self, atoms, properties=["energy", "forces"], system_changes=all_changes
    ):
        Calculator.calculate(self, atoms, properties, system_changes)
        forces = np.zeros(shape=(self.n_total_atoms, 3))
        energy = 0.0

        for mask, calc in zip(self.masks, self.calculators):
            system_indices = np.array(range(0, len(atoms), 1))
            subsystem = atoms[mask]
            subsystem_system_indices = system_indices[mask]
            # Set RATTLE constraints on O-H bonds
            # OHH ordering
            tip4p_constraints = []
            lj_wat_constraints = (
                []
            )  # Constraints that must be applied for Lennard-Jones interactions with O
            for atom in subsystem:
                if atom.symbol == "O":
                    if atom.index + 2 < len(subsystem):
                        if (
                            subsystem[atom.index + 1].symbol == "H"
                            and subsystem[atom.index + 2].symbol == "H"
                        ):
                            tip4p_constraints.append([atom.index, atom.index + 1])
                            tip4p_constraints.append([atom.index, atom.index + 2])
                            tip4p_constraints.append([atom.index + 1, atom.index + 2])
                    else:
                        # For Lennard-Jones interactions
                        a_index = atom.index  # Index in subsystem
                        h1_pos = atoms.positions[
                            subsystem_system_indices[a_index] + 1
                        ]  # from the original atoms object
                        h2_pos = atoms.positions[subsystem_system_indices[a_index] + 2]
                        c = MyWaterConstraint(
                            a_index, h1_pos, h2_pos
                        )  # constraint on the O
                        lj_wat_constraints.append(c)

            if len(tip4p_constraints) > 0:
                rattle_constraints = ase.constraints.FixBondLengths(tip4p_constraints)
                subsystem.set_constraint(rattle_constraints)
            if len(lj_wat_constraints) > 0:
                for constraint in lj_wat_constraints:
                    subsystem.set_constraint(constraint)
            calc.calculate(subsystem)
            energy += calc.results["energy"]
            forces[mask] += calc.results["forces"]

        self.results["energy"] = energy
        self.results["forces"] = forces
