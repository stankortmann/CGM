"""Merge omega part files into a single omega_final.hdf5 file."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def merge_omega_parts(directory: Path, output_name: str = "omega_final.hdf5") -> Path:
    part_files = sorted(
        path for path in directory.glob("omega_part*.hdf5") if path.name != output_name
    )

    if not part_files:
        raise FileNotFoundError(f"No omega_part*.hdf5 files found in {directory}")

    combined = {}
    redshift = None
    box_size_mpc = None

    for path in part_files:
        with h5py.File(path, "r") as handle:
            if redshift is None:
                redshift = handle.attrs["redshift"]
            if box_size_mpc is None:
                box_size_mpc = handle.attrs["box_size_Mpc"]

            omega_group = handle["omega_parameters"]
            for ion, dataset in omega_group.items():
                combined[ion] = combined.get(ion, 0.0) + float(np.asarray(dataset[()]))

    output_path = directory / output_name
    with h5py.File(output_path, "w") as handle:
        handle.attrs["redshift"] = redshift
        handle.attrs["box_size_Mpc"] = box_size_mpc
        omega_group = handle.create_group("omega_parameters")
        for ion in sorted(combined):
            omega_group.create_dataset(ion, data=combined[ion])

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge omega part files into omega_final.hdf5")
    parser.add_argument("--dir", required=True, help="Directory containing omega_part*.hdf5 files")
    args = parser.parse_args()

    directory = Path(args.dir).expanduser().resolve()
    output_path = merge_omega_parts(directory)
    print(f"Wrote merged omega file to {output_path}")


if __name__ == "__main__":
    main()