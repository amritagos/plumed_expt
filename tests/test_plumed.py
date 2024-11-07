import pytest
from ase.calculators.lj import LennardJones
from ase.calculators.plumed import Plumed
from ase.constraints import FixedPlane
from ase.md.langevin import Langevin
from ase.io import read
from ase import units
import plumed
import numpy as np
from pathlib import Path


def test_colvar_output():
    """Starting from a planar LJ cluster, runs Molecular Dynamics simulation with ASE
    and PLUMED. Tests that the colvar file created by PLUMED has the correct calculated collective variables
    Test adapted from the ASE-PLUMED tutorial (https://github.com/Sucerquia/ASE-PLUMED_tutorial)
    """
    test_dir = Path(__file__).resolve().parent
    xyz_infile = test_dir / "../resources/isomerLJ.xyz"
    out_colvarfile = test_dir / "COLVAR"
    # Read in the input configuration
    atoms = read(xyz_infile)

    # ---------
    timestep = 0.005
    ps = 1000 * units.fs  #  To convert to ASE units from ps, divide by this value
    # setup replaces the plumed.dat file
    setup = [
        f"UNITS LENGTH=A TIME={1/ps} ENERGY={units.mol/units.kJ}",
        "c1: COORDINATIONNUMBER SPECIES=1-7 MOMENTS=2-3"
        + " SWITCH={RATIONAL R_0=1.5 NN=8 MM=16}",
        f"PRINT ARG=c1.* STRIDE=100 FILE={str(out_colvarfile)}",
        "FLUSH STRIDE=1000",
    ]
    print(setup)
    # Constraint to keep the system in a plane
    cons = [FixedPlane(i, [0, 0, 1]) for i in range(7)]
    atoms.set_constraint(cons)
    atoms.set_masses([1, 1, 1, 1, 1, 1, 1])

    atoms.calc = Plumed(
        calc=LennardJones(rc=2.5, r0=3.0),
        input=setup,
        timestep=timestep,
        atoms=atoms,
        kT=0.1,
    )

    dyn = Langevin(
        atoms, timestep, temperature_K=0.1 / units.kB, friction=1, fixcm=False
    )
    dyn.run(1)
    # ---------
    colvar = plumed.read_as_pandas(str(out_colvarfile))
    assert np.all(np.array(colvar["c1.moment-3"][:1]) == np.array([1.335796]))
    assert np.all(np.array(colvar["c1.moment-2"][:1]) == np.array([0.757954]))
    # Delete the COLVAR file
    out_colvarfile.unlink()
    # ------------
    # Not needed, but this is how you can interact with the plumed.Plumed object
    plumed_version = np.zeros(1, dtype=np.intc)
    atoms.calc.plumed.cmd("getApiVersion", plumed_version)
    print(f"{plumed_version=}")


if __name__ == "__main__":
    test_colvar_output()
