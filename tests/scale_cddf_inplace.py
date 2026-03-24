#!/usr/bin/env python3
"""Scale element and ion CDDF arrays in single_cd HDF5 files in place."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


# Put your 17 files here. If no CLI files are passed, this list is used.
DEFAULT_INPUT_FILES = [
    "/cosma8/data/do012/dc-kort1/CGM/L100/L100_m6/Thermal/z_0.000/hdf5_data/oxygen/slice_0.hdf5",
    "/cosma8/data/do012/dc-kort1/CGM/L100/L100_m6/Thermal/z_0.000/hdf5_data/oxygen/slice_1.hdf5",
    "/cosma8/data/do012/dc-kort1/CGM/L100/L100_m6/Thermal/z_0.000/hdf5_data/oxygen/slice_2.hdf5",
    "/cosma8/data/do012/dc-kort1/CGM/L100/L100_m6/Thermal/z_0.000/hdf5_data/oxygen/slice_3.hdf5",
    "/cosma8/data/do012/dc-kort1/CGM/L100/L100_m6/Thermal/z_0.000/hdf5_data/oxygen/slice_4.hdf5",
    "/cosma8/data/do012/dc-kort1/CGM/L100/L100_m6/Thermal/z_0.000/hdf5_data/oxygen/slice_5.hdf5",
    "/cosma8/data/do012/dc-kort1/CGM/L100/L100_m6/Thermal/z_0.000/hdf5_data/oxygen/slice_6.hdf5",
    "/cosma8/data/do012/dc-kort1/CGM/L100/L100_m6/Thermal/z_0.000/hdf5_data/oxygen/slice_7.hdf5",
    "/cosma8/data/do012/dc-kort1/CGM/L100/L100_m6/Thermal/z_0.000/hdf5_data/oxygen/slice_8.hdf5",
    "/cosma8/data/do012/dc-kort1/CGM/L100/L100_m6/Thermal/z_0.000/hdf5_data/oxygen/slice_9.hdf5",
    "/cosma8/data/do012/dc-kort1/CGM/L100/L100_m6/Thermal/z_0.000/hdf5_data/oxygen/slice_10.hdf5",
    "/cosma8/data/do012/dc-kort1/CGM/L100/L100_m6/Thermal/z_0.000/hdf5_data/oxygen/slice_11.hdf5",
    "/cosma8/data/do012/dc-kort1/CGM/L100/L100_m6/Thermal/z_0.000/hdf5_data/oxygen/slice_12.hdf5",
    "/cosma8/data/do012/dc-kort1/CGM/L100/L100_m6/Thermal/z_0.000/hdf5_data/oxygen/slice_13.hdf5",
    "/cosma8/data/do012/dc-kort1/CGM/L100/L100_m6/Thermal/z_0.000/hdf5_data/oxygen/slice_14.hdf5",
    "/cosma8/data/do012/dc-kort1/CGM/L100/L100_m6/Thermal/z_0.000/hdf5_data/oxygen/slice_15.hdf5",
    "/cosma8/data/do012/dc-kort1/CGM/L100/L100_m6/Thermal/z_0.000/hdf5_data/oxygen/total.hdf5"

    # "/path/to/single_cd_02.hdf5",
    # "/path/to/single_cd_03.hdf5",
]


def _groups_from_cfg(file_handle: h5py.File) -> list[str]:
    """Return target groups (element + ions) from serialized cfg if present."""
    if "cfg" not in file_handle.attrs:
        return []

    cfg = json.loads(file_handle.attrs["cfg"])
    chemistry = cfg.get("chemistry", {})

    groups: list[str] = []
    element = chemistry.get("element")
    if isinstance(element, str):
        groups.append(element)

    ions = chemistry.get("ion", [])
    if isinstance(ions, list):
        groups.extend([ion for ion in ions if isinstance(ion, str)])

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(groups))


def _scale_group_cddf(file_handle: h5py.File, group_name: str, factor: float) -> bool:
    """Scale <group>/cddf by factor. Return True if dataset was changed."""
    if group_name not in file_handle:
        return False

    group = file_handle[group_name]
    if "cddf" not in group:
        return False

    dataset = group["cddf"]
    if not np.issubdtype(dataset.dtype, np.floating):
        raise TypeError(
            f"Dataset '{group_name}/cddf' has non-floating dtype {dataset.dtype}; "
            "refusing in-place scaling to avoid precision loss."
        )

    dataset[...] = dataset[...] / factor
    return True


def scale_file(path: Path, factor: float, fallback_scan: bool = True) -> tuple[int, list[str]]:
    """Scale all relevant cddf datasets in one HDF5 file.

    Returns:
        count of modified datasets, and list of modified group names.
    """
    changed: list[str] = []

    with h5py.File(path, "r+") as handle:
        groups = _groups_from_cfg(handle)

        for group_name in groups:
            if _scale_group_cddf(handle, group_name, factor):
                changed.append(group_name)

        # Optional fallback: if cfg is missing/partial, still update top-level groups with cddf.
        if fallback_scan:
            for group_name, obj in handle.items():
                if not isinstance(obj, h5py.Group):
                    continue
                if group_name in changed:
                    continue
                if "cddf" in obj and _scale_group_cddf(handle, group_name, factor):
                    changed.append(group_name)

    return len(changed), changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Divide element/ion CDDF datasets by a factor in-place for one or more HDF5 files."
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Path(s) to HDF5 files to update. If omitted, DEFAULT_INPUT_FILES is used.",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=16.0,
        help="Divisor for CDDF datasets (default: 16).",
    )
    parser.add_argument(
        "--no-fallback-scan",
        action="store_true",
        help="Only update groups listed in cfg.chemistry (element + ions).",
    )
    parser.add_argument(
        "--print-file-list",
        action="store_true",
        help="Print the resolved file list before processing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.factor == 0:
        raise ValueError("--factor must be non-zero")

    files_to_process = args.files if args.files else [Path(p) for p in DEFAULT_INPUT_FILES]

    if not files_to_process:
        raise ValueError(
            "No input files provided. Add paths to DEFAULT_INPUT_FILES or pass files on the command line."
        )

    if args.print_file_list:
        print("Input files:")
        for idx, file_path in enumerate(files_to_process, start=1):
            print(f"  {idx:02d}. {file_path}")

    total_files = 0
    total_datasets = 0

    for file_path in files_to_process:
        total_files += 1

        if not file_path.exists():
            print(f"[SKIP] {file_path} (does not exist)")
            continue

        try:
            count, groups = scale_file(
                file_path,
                factor=args.factor,
                fallback_scan=not args.no_fallback_scan,
            )
            total_datasets += count
            if count == 0:
                print(f"[OK]   {file_path} (no cddf datasets found)")
            else:
                groups_str = ", ".join(groups)
                print(f"[OK]   {file_path} (updated {count} groups: {groups_str})")
        except Exception as err:  # pragma: no cover - defensive CLI error path
            print(f"[ERR]  {file_path} ({err})")

    print(f"Done. Processed {total_files} files; updated {total_datasets} cddf datasets.")


if __name__ == "__main__":
    main()
