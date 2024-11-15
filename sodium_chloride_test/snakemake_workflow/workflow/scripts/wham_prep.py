import argparse
from pathlib import Path
import numpy as np
import plumed


def wham(
    bias,
    *,
    frame_weight=None,
    traj_weight=None,
    T: float = 1.0,
    maxiter: int = 1000,
    threshold: float = 1e-40,
    verbose: bool = False,
):
    """Weighted histogram analysis method

    Args:
        bias (_type_): Bias with shape (nframes, ntraj). Each trajectory should have the same number of frames
        frame_weight (_type_, optional): Weight for each frame. Defaults to None.
        traj_weight (_type_, optional): Weight for each trajectory. Defaults to None.
        T (float, optional): This is k_B*T where k_B is the Boltzmann constant and T is the temperature. Defaults to 1.0.
        maxiter (int, optional): Maximum number of iterations. Defaults to 1000.
        threshold (float, optional): Convergence criterion. Defaults to 1e-40.
        verbose (bool, optional): Prints out some things if True. Defaults to False.

    Returns:
        _type_: _description_
    """

    nframes = bias.shape[0]
    ntraj = bias.shape[1]

    # default values
    if frame_weight is None:
        frame_weight = np.ones(nframes)
    if traj_weight is None:
        traj_weight = np.ones(ntraj)

    assert len(traj_weight) == ntraj
    assert len(frame_weight) == nframes

    # divide by T once for all
    shifted_bias = bias / T
    # track shifts
    shifts0 = np.min(shifted_bias, axis=0)
    shifted_bias -= shifts0[np.newaxis, :]
    shifts1 = np.min(shifted_bias, axis=1)
    shifted_bias -= shifts1[:, np.newaxis]

    # do exponentials only once
    expv = np.exp(-shifted_bias)

    Z = np.ones(ntraj)

    Zold = Z

    if verbose:
        print("WHAM: start\n")
    for nit in range(maxiter):
        # find unnormalized weights
        weight = 1.0 / np.matmul(expv, traj_weight / Z) * frame_weight
        # update partition functions
        Z = np.matmul(weight, expv)
        # normalize the partition functions
        Z /= np.sum(Z * traj_weight)
        # monitor change in partition functions
        eps = np.sum(np.log(Z / Zold) ** 2)
        Zold = Z
        if verbose:
            print(f"WHAM: iteration {nit} eps {eps}\n")
        if eps < threshold:
            break
    nfev = nit
    logW = np.log(weight) + shifts1

    if verbose:
        print("WHAM: end")

    return {"logW": logW, "logZ": np.log(Z) - shifts0, "nit": nit, "eps": eps}


def main(
    colvar_in_files: list[Path],
    kbT: float,
    temperature: float,
    grid_min: float,
    grid_max: float,
    grid_bin: int,
    out_colvar: Path,
    fes_filename: Path,
    fes_log_filename: Path,
    out_plumed: Path,
):
    data = []  # Contains the CVs and bias values

    for p in colvar_in_files:
        data.append(plumed.read_as_pandas(str(p)))

    # Now deduce the number of windows, total frames
    n_windows = len(data)
    total_frames = len(data[0])

    bias = np.zeros(shape=(total_frames, n_windows))
    for i in range(n_windows):
        bias[:, i] = data[i]["restraint.bias"][-len(bias) :]

    w = wham(bias=bias, T=kbT)
    # Produce file for PLUMED
    colvar = data[0]
    colvar["logweights"] = w["logW"]
    plumed.write_pandas(colvar, str(out_colvar))

    # Write the PLUMED input file as well
    with open(out_plumed, "w") as f:
        print(
            f"""UNITS LENGTH=A TIME=fs ENERGY=eV
d1: READ FILE={out_colvar} VALUES=d1 IGNORE_TIME
bb: READ FILE={out_colvar} VALUES=restraint.bias 
lw: READ FILE={out_colvar} VALUES=logweights 

# use the command below to compute the histogram of phi
# we use a smooth kernel to produce a nicer graph here
hhd1: HISTOGRAM ARG=d1 GRID_MIN={grid_min} GRID_MAX={grid_max} GRID_BIN={grid_bin} BANDWIDTH=0.05
ffd1: CONVERT_TO_FES GRID=hhd1 TEMP={temperature}
DUMPGRID GRID=ffd1 FILE={fes_filename}

# we use a smooth kernel to produce a nicer graph here
hhd1r: HISTOGRAM ARG=d1 GRID_MIN={grid_min} GRID_MAX={grid_max} GRID_BIN={grid_bin} BANDWIDTH=0.05 LOGWEIGHTS=lw
ffd1r: CONVERT_TO_FES GRID=hhd1r TEMP={temperature} 
DUMPGRID GRID=ffd1r FILE={fes_log_filename}

""",
            file=f,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WHAM analysis and preparation of PLUMED file."
    )
    parser.add_argument("--colvar_in_files", type=Path, nargs="+")

    parser.add_argument("--kbT", type=float, help="Boltzmann constant * temperature")

    parser.add_argument("--T", type=float, help="Temperature in Kelvin")

    parser.add_argument(
        "--grid_min", type=float, help="Lower bounds for the grid in PLUMED"
    )
    parser.add_argument(
        "--grid_max", type=float, help="Upper bounds for the grid in PLUMED"
    )
    parser.add_argument("--grid_bin", type=int, help="Number of bins for the bin")

    parser.add_argument("--out_colvar", type=Path)

    parser.add_argument("--fes_filename", type=Path)

    parser.add_argument("--fes_log_filename", type=Path)

    parser.add_argument("--out_plumed", type=Path)

    args = parser.parse_args()

    main(
        args.colvar_in_files,
        args.kbT,
        args.T,
        args.grid_min,
        args.grid_max,
        args.grid_bin,
        args.out_colvar,
        args.fes_filename,
        args.fes_log_filename,
        args.out_plumed,
    )
