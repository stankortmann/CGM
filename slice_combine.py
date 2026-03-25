#!/usr/bin/env python3
"""Merge slice HDF5 files by summing 2D column densities and recomputing CDDFs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import unyt as u
from mpi4py import MPI
from swiftsimio.objects import cosmo_array

from spec_analysis import density_profiles
from spec_analysis import unpack_data
from spec_analysis.save_data import projection_saver
from spec_analysis.unpack_data import single_cd


SLICE_RE = re.compile(r"^slice_(\d+)\.hdf5$")


def find_slice_files(input_dir: Path) -> list[tuple[int, Path]]:
    """Return (slice_index, path) sorted by numeric index."""
    matches: list[tuple[int, Path]] = []
    for path in input_dir.glob("slice_*.hdf5"):
        m = SLICE_RE.match(path.name)
        if m is None:
            continue
        matches.append((int(m.group(1)), path))

    matches.sort(key=lambda x: x[0])
    return matches


def group_names_from_cfg(handle: h5py.File) -> list[str]:
    """Read element and ions from cfg attribute."""
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

    # preserve order and remove duplicates
    return list(dict.fromkeys(groups))


def fallback_groups(handle: h5py.File) -> list[str]:
    """Fallback to all top-level groups containing column_density and cddf."""
    groups: list[str] = []
    for name, obj in handle.items():
        if not isinstance(obj, h5py.Group):
            continue
        if "column_density" in obj and "cddf" in obj:
            groups.append(name)
    return groups


def initialize_cddf_engine(cfg_ns):
    """Initialize column_density_2d_swift exactly like in box_mpi_multiple."""
    data_unpacker = unpack_data.unwrapper(cfg_ns)
    snapshot = data_unpacker.load_snapshot()
    cd_2d = density_profiles.column_density_2d_swift(
        cfg=cfg_ns,
        data_unpacker=data_unpacker,
        snapshot=snapshot,
        element=cfg_ns.chemistry.element,
        mpi=True,
    )
    return cd_2d, float(data_unpacker.scale_factor)


def merge_job_output_name(first_idx: int, last_idx: int) -> str:
    """Filename format requested by user, using 1-based slice labels."""
    return f"merged_slices_{first_idx + 1}_{last_idx + 1}.hdf5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge consecutive slice_*.hdf5 blocks, sum 2D column densities, "
            "and recompute CDDF using column_density_2d_swift logic."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing slice_*.hdf5 files.",
    )
    parser.add_argument(
        "--merge_factor",
        type=int,
        required=True,
        help="Number of consecutive slices merged into one output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    merge_factor = int(args.merge_factor)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if merge_factor < 2:
        if rank == 0:
            raise ValueError("merge_factor must be >= 2")
        return

    if not input_dir.is_dir():
        if rank == 0:
            raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
        return

    indexed_slices = find_slice_files(input_dir)
    if not indexed_slices:
        if rank == 0:
            raise FileNotFoundError(f"No slice_<index>.hdf5 files found in {input_dir}")
        return

    n_slices = len(indexed_slices)
    if n_slices % merge_factor != 0:
        if rank == 0:
            raise ValueError(
                f"Number of slices ({n_slices}) must be divisible by merge_factor ({merge_factor})."
            )
        return

    jobs: list[list[tuple[int, Path]]] = []
    for i in range(0, n_slices, merge_factor):
        jobs.append(indexed_slices[i : i + merge_factor])

    n_jobs = len(jobs)

    if size != n_jobs:
        if rank == 0:
            raise RuntimeError(
                "MPI size must equal number of merge jobs for one-core-per-merger mode: "
                f"need {n_jobs} ranks, got {size}."
            )
        return

    my_job = jobs[rank]
    first_idx = my_job[0][0]
    last_idx = my_job[-1][0]
    output_name = merge_job_output_name(first_idx, last_idx)
    output_path = input_dir / output_name

    template_path = my_job[0][1]

    template_slice = single_cd(str(template_path), load_cd=False)
    cfg_ns = template_slice.cfg

    groups = [template_slice.element_name] + [ion for ion in template_slice.ions.keys()]
    if not groups:
        with h5py.File(template_path, "r") as h0:
            groups = group_names_from_cfg(h0)
            if not groups:
                groups = fallback_groups(h0)
        if not groups:
            raise RuntimeError(f"Could not resolve element/ion groups from {template_path}")

    # Initialize class the same way as in box_mpi_multiple.
    cddf_engine, scale_factor = initialize_cddf_engine(cfg_ns)

    saver = projection_saver(cfg_ns, use_compression=True, dtype=np.float32, comm=MPI.COMM_SELF)

    summed_maps: dict[str, np.ndarray] = {}
    map_units: dict[str, str] = {}
    z_mins: list[float] = []
    z_maxs: list[float] = []
    z_unit = None

    for _, slice_path in my_job:
        slice_data = single_cd(str(slice_path), load_cd=True)

        with h5py.File(slice_path, "r") as h:
            zmin = float(h["z_min"][()])
            zmax = float(h["z_max"][()])
            z_mins.append(zmin)
            z_maxs.append(zmax)
            if z_unit is None:
                z_unit = h["z_min"].attrs.get("unit", "Mpc")

        for group in groups:
            if group == slice_data.element_name:
                if slice_data.element_cd is None:
                    raise KeyError(f"Missing {group}/column_density in {slice_path}")
                arr = np.asarray(slice_data.element_cd.value, dtype=np.float64)
                unit = str(slice_data.element_cd.units)
            else:
                if group not in slice_data.ions:
                    raise KeyError(f"Missing group '{group}' in {slice_path}")
                ion_cd = slice_data.ions[group]["column_density"]
                if ion_cd is None:
                    raise KeyError(f"Missing {group}/column_density in {slice_path}")
                arr = np.asarray(ion_cd.value, dtype=np.float64)
                unit = str(ion_cd.units)

            if group not in summed_maps:
                summed_maps[group] = np.array(arr, dtype=np.float64, copy=True)
                map_units[group] = unit
            else:
                if arr.shape != summed_maps[group].shape:
                    raise ValueError(
                        f"Shape mismatch for {group}/column_density in {slice_path}: "
                        f"expected {summed_maps[group].shape}, got {arr.shape}"
                    )
                summed_maps[group] += arr

    merged_z_min = min(z_mins)
    merged_z_max = max(z_maxs)

    z_unit_obj = u.Unit(str(z_unit))
    los_range = [
        cosmo_array(
            merged_z_min,
            z_unit_obj,
            comoving=False,
            scale_factor=scale_factor,
            scale_exponent=1,
        ),
        cosmo_array(
            merged_z_max,
            z_unit_obj,
            comoving=False,
            scale_factor=scale_factor,
            scale_exponent=1,
        ),
    ]

    element_name = cfg_ns.chemistry.element
    element_cd = cosmo_array(
        summed_maps[element_name],
        u.Unit(str(map_units[element_name])),
        comoving=False,
        scale_factor=scale_factor,
        scale_exponent=0,
    )

    ion_cd_map = {}
    for ion in cfg_ns.chemistry.ion:
        ion_cd_map[ion] = cosmo_array(
            summed_maps[ion],
            u.Unit(str(map_units[ion])),
            comoving=False,
            scale_factor=scale_factor,
            scale_exponent=0,
        )

    saver.save_projection_file(
        file_path=output_path,
        cd_2d_obj=cddf_engine,
        element_column_density=element_cd,
        ion_column_density_map=ion_cd_map,
        los_range_local=los_range,
    )

    with h5py.File(output_path, "r+") as out_h:
        # Update cfg.window.projection_slices to match merged slice count.
        if "cfg" in out_h.attrs:
            cfg_dict_out = json.loads(out_h.attrs["cfg"])
            if "window" in cfg_dict_out and isinstance(cfg_dict_out["window"], dict):
                cfg_dict_out["window"]["projection_slices"] = int(n_jobs)
                out_h.attrs["cfg"] = json.dumps(cfg_dict_out)

        out_h.attrs["merge_factor"] = int(merge_factor)
        out_h.attrs["merged_from_slices"] = json.dumps([p.name for _, p in my_job])

    comm.Barrier()

    if rank == 0:
        print(f"Merged {n_slices} input slices into {n_jobs} merged files in {input_dir}")

    print(
        f"Rank {rank}: wrote {output_path.name} from slices "
        f"{first_idx}..{last_idx} with z=[{merged_z_min}, {merged_z_max}]"
    )


if __name__ == "__main__":
    main()
