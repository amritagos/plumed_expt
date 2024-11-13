from pathlib import Path
import sys
import json
from typing import List
import pandas as pd
from ase.io import read, write


def main(
    traj_paths: List[Path],
    output_traj: Path,
):

    all_trajs = []  # Will contain concatenated trajectory

    for p in traj_paths:
        all_trajs += read(p, index=":")

    write(output_traj, all_trajs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--traj_paths", type=Path, nargs="+")
    parser.add_argument("--output_traj", type=Path)

    args = parser.parse_args()

    main(args.traj_paths, args.output_traj)
