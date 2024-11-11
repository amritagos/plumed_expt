import ase
from ase.calculators.calculator import BaseCalculator, Calculator, all_changes
import numpy as np

class SubsystemCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, masks, calculators:list[Calculator], restart=None, ignore_bad_restart_file=BaseCalculator._deprecated, label=None, atoms=None, directory='.',**kwargs):
        super().__init__(restart, ignore_bad_restart_file, label, atoms, directory, **kwargs)
        self.masks = masks 
        self.calculators = calculators
        self.n_total_atoms = len(masks[0])
        for mask in masks:
            assert len(mask) == self.n_total_atoms
        
    def calculate(self, atoms, properties=['energy', 'forces'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        forces = np.zeros(shape=(self.n_total_atoms, 3))
        energy = 0.0

        for mask, calc in zip(self.masks, self.calculators):
            subsystem = atoms[mask]
            # Set RATTLE constraints on O-H bonds
            # OHH ordering 
            constraints = []
            for atom in subsystem:
                if atom.symbol=='O':
                    if subsystem[atom.index+1].symbol=='H' and subsystem[atom.index+2].symbol=='H':
                        constraints.append([atom.index, atom.index+1])
                        constraints.append([atom.index, atom.index+2])
                        constraints.append([atom.index+1, atom.index+2])
            if len(constraints)>0:
                rattle_constraints = ase.constraints.FixBondLengths(constraints) 
                subsystem.set_constraint(rattle_constraints)
            calc.calculate(subsystem)
            energy += calc.results["energy"]
            forces[mask] += calc.results["forces"]
        
        self.results["energy"] = energy
        self.results["forces"] = forces