#!/usr/bin/env python3
"""Build stacked.hdf5 by averaging element/ion CDDFs across slice_*.hdf5 files."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import h5py
import numpy as np


SLICE_RE = re.compile(r"^slice_(\d+)\.hdf5$")


def find_slice_files(input_dir: Path) -> list[Path]:
    """Return slice files sorted by their numeric index."""
    matches: list[tuple[int, Path]] = []
    for path in input_dir.glob("slice_*.hdf5"):
        match = SLICE_RE.match(path.name)
        if not match:
            continue
        matches.append((int(match.group(1)), path))

    matches.sort(key=lambda item: item[0])
    return [path for _, path in matches]


def groups_from_cfg(handle: h5py.File) -> list[str]:
    """Read element + ion group names from cfg attribute, if present."""
    if "cfg" not in handle.attrs:
        return []

    cfg = json.loads(handle.attrs["cfg"])
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


def fallback_groups(handle: h5py.File) -> list[str]:
    """Fallback: all top-level groups containing a cddf dataset."""
    groups: list[str] = []
    for group_name, obj in handle.items():
        if isinstance(obj, h5py.Group) and "cddf" in obj:
            groups.append(group_name)
    return groups


def resolve_groups(reference_file: Path) -> list[str]:
    """Determine which groups should be stacked."""
    with h5py.File(reference_file, "r") as handle:
        groups = groups_from_cfg(handle)
        if groups:
            return groups
        return fallback_groups(handle)


def average_cddf(slice_files: list[Path], groups: list[str]) -> dict[str, np.ndarray]:
    """Compute average cddf for each target group over all slice files."""
    sums: dict[str, np.ndarray] = {}
    dtypes: dict[str, np.dtype] = {}

    for file_index, file_path in enumerate(slice_files):
        with h5py.File(file_path, "r") as handle:
            for group_name in groups:
                if group_name not in handle:
                    raise KeyError(f"Missing group '{group_name}' in {file_path}")
                group = handle[group_name]
                if "cddf" not in group:
                    raise KeyError(f"Missing dataset '{group_name}/cddf' in {file_path}")

                cddf = np.asarray(group["cddf"][:])

                if file_index == 0:
                    sums[group_name] = np.zeros_like(cddf, dtype=np.float64)
                    dtypes[group_name] = group["cddf"].dtype
                elif cddf.shape != sums[group_name].shape:
                    raise ValueError(
                        f"Shape mismatch for '{group_name}/cddf' in {file_path}: "
                        f"expected {sums[group_name].shape}, got {cddf.shape}"
                    )

                sums[group_name] += cddf.astype(np.float64)

    n_slices = float(len(slice_files))
    averages: dict[str, np.ndarray] = {}
    for group_name, cddf_sum in sums.items():
        avg = cddf_sum / n_slices
        averages[group_name] = avg.astype(dtypes[group_name], copy=False)

    return averages


def write_stacked_file(
    template_slice: Path,
    output_file: Path,
    averages: dict[str, np.ndarray],
    n_slices: int,
) -> None:
    """Copy template file, then replace cddf datasets with stacked averages."""
    shutil.copy2(template_slice, output_file)

    with h5py.File(output_file, "r+") as handle:
        for group_name, avg in averages.items():
            if group_name not in handle or "cddf" not in handle[group_name]:
                raise KeyError(
                    f"Output template missing '{group_name}/cddf' in {output_file}"
                )
            handle[group_name]["cddf"][...] = avg

        handle.attrs["n_slices_stacked"] = int(n_slices)
        handle.attrs["stacked_from_last_slice"] = template_slice.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Average element/ion cddf datasets across slice_*.hdf5 in a directory, "
            "then write stacked.hdf5."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        help="Directory containing slice_0.hdf5 ... slice_N.hdf5 files.",
    )
    parser.add_argument(
        "--output-name",
        default="stacked.hdf5",
        help="Output filename placed inside input_dir (default: stacked.hdf5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    slice_files = find_slice_files(input_dir)
    if not slice_files:
        raise FileNotFoundError(
            f"No files matching slice_<index>.hdf5 were found in: {input_dir}"
        )

    groups = resolve_groups(slice_files[0])
    if not groups:
        raise RuntimeError(
            "Could not determine element/ion groups to stack (no cfg chemistry and no cddf groups found)."
        )

    averages = average_cddf(slice_files, groups)

    template_slice = slice_files[-1]
    output_file = input_dir / args.output_name

    write_stacked_file(
        template_slice=template_slice,
        output_file=output_file,
        averages=averages,
        n_slices=len(slice_files),
    )

    print(f"Stacked {len(slice_files)} slices into {output_file}")
    print(f"Template copied from last slice: {template_slice}")
    print(f"Updated cddf groups: {', '.join(groups)}")


if __name__ == "__main__":
    main()
