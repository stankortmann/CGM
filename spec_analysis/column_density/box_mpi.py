# cd_swift_hdf5.py

from swiftsimio import load
import numpy as np
import unyt as u
from pathlib import Path
import h5py
import json
from dataclasses import is_dataclass, asdict
from swiftsimio.objects import cosmo_array, cosmo_factor, cosmo_quantity
from mpi4py import MPI

# own modules
from spec_analysis import unpack_data
from spec_analysis import density_profiles
from spec_analysis import plot


comm = MPI.COMM_WORLD



def cfg_to_serializable(cfg):
    """
    Recursively convert a dataclass or dict to something JSON serializable.
    Handles cosmo_array, cosmo_quantity, and cosmo_factor from swiftsimio.
    """

    if is_dataclass(cfg):
        cfg = asdict(cfg)

    if isinstance(cfg, dict):
        return {k: cfg_to_serializable(v) for k, v in cfg.items()}

    elif isinstance(cfg, (list, tuple)):
        return [cfg_to_serializable(x) for x in cfg]

    elif isinstance(cfg, cosmo_array):
        arr = cfg.to_comoving()
        return {"value": arr.value.tolist(), "unit": str(arr.units)}

    elif isinstance(cfg, cosmo_quantity):
        q = cfg.to_comoving()
        return {"value": float(q), "unit": str(q.units)}

    elif isinstance(cfg, cosmo_factor):
        return float(cfg)

    else:
        return cfg


def run_box_column_density_parallel(cfg,comm):
    """
    Compute 2D column density and CDDFs for a simulation snapshot,
    using MPI tiling in x-y with overlap.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    # -------------------------------------------------
    # Unpack simulation
    # -------------------------------------------------

    data_unpacker = unpack_data.unwrapper(cfg)
    comoving_box_size = data_unpacker.box_size.to("Mpc")

    cfg.window.x = [x * comoving_box_size for x in cfg.window.x]
    cfg.window.y = [y * comoving_box_size for y in cfg.window.y]
    cfg.window.z = [z * comoving_box_size for z in cfg.window.z]

    proj_axis = {"x": cfg.window.x, "y": cfg.window.y, "z": cfg.window.z}
    proj_range = proj_axis[cfg.window.projection_axis]

    # -------------------------------------------------
    # MPI TILE GRID
    # -------------------------------------------------

    n_tile = int(np.sqrt(size))
    if n_tile * n_tile != size:
        if rank == 0:
            raise RuntimeError("MPI ranks must be a perfect square")

    ix = rank % n_tile
    iy = rank // n_tile

    x_min, x_max = cfg.window.x
    y_min, y_max = cfg.window.y
    z_min, z_max = cfg.window.z

    dx = (x_max - x_min) / n_tile
    dy = (y_max - y_min) / n_tile

    overlap = getattr(cfg.window, "tile_overlap", 0.001)*comoving_box_size

    tile_x = [
        x_min + ix * dx - overlap,
        x_min + (ix + 1) * dx + overlap
    ]

    tile_y = [
        y_min + iy * dy - overlap,
        y_min + (iy + 1) * dy + overlap
    ]

    tile_z = [z_min, z_max]

    region = [tile_x, tile_y, tile_z]

    # -------------------------------------------------
    # LOAD SNAPSHOT TILE
    # -------------------------------------------------

    snapshot = data_unpacker.load_snapshot(load_region=region)

    if rank == 0:
        print("Gas particles are loaded (MPI tiled)")

    # -------------------------------------------------
    # COLUMN DENSITY PROJECTION
    # -------------------------------------------------

    cd_2d = density_profiles.column_density_2d_swift(
        cfg=cfg,
        data_unpacker=data_unpacker,
        snapshot=snapshot,
        element=cfg.chemistry.element
    )

    local_element = cd_2d.element_column_density.to_physical()

    local_ions = {}
    for ion in cfg.chemistry.ion:
        local_ions[ion] = cd_2d.column_density_ion(ion=ion).to_physical()

    # -------------------------------------------------
    # REMOVE OVERLAP
    # -------------------------------------------------

    resolution = cfg.window.resolution
    tile_res = resolution // n_tile

    if overlap != 0:

        pixels_per_mpc = resolution / (x_max - x_min).value
        overlap_pixels = int(overlap.value * pixels_per_mpc)

        if overlap_pixels > 0:

            local_element = local_element[
                overlap_pixels:-overlap_pixels,
                overlap_pixels:-overlap_pixels
            ]

            for ion in cfg.chemistry.ion:

                local_ions[ion] = local_ions[ion][
                    overlap_pixels:-overlap_pixels,
                    overlap_pixels:-overlap_pixels
                ]

    # -------------------------------------------------
    # GATHER MPI RESULTS
    # -------------------------------------------------

    gathered_element = comm.gather(local_element, root=0)

    gathered_ions = {}
    for ion in cfg.chemistry.ion:
        gathered_ions[ion] = comm.gather(local_ions[ion], root=0)
    #all ranks return here, but only rank 0 has the full data in gathered_element and gathered_ions
    if rank != 0:
        return

    # -------------------------------------------------
    # STITCH ELEMENT MAP
    # -------------------------------------------------

    full_element = np.zeros((resolution, resolution))

    for r, tile in enumerate(gathered_element):

        ix = r % n_tile
        iy = r // n_tile

        full_element[
            ix * tile_res:(ix + 1) * tile_res,
            iy * tile_res:(iy + 1) * tile_res
        ] = tile

    # -------------------------------------------------
    # STITCH ION MAPS
    # -------------------------------------------------

    full_ions = {}

    for ion in cfg.chemistry.ion:

        full_map = np.zeros((resolution, resolution))

        for r, tile in enumerate(gathered_ions[ion]):

            ix = r % n_tile
            iy = r // n_tile

            full_map[
                ix * tile_res:(ix + 1) * tile_res,
                iy * tile_res:(iy + 1) * tile_res
            ] = tile

        full_ions[ion] = full_map



    print("Projection tiles stitched together")

    # -------------------------------------------------
    # ORIGINAL SAVING CODE
    # -------------------------------------------------

    hdf5_dir = Path(data_unpacker.output_directory) / "hdf5_data"
    hdf5_dir.mkdir(parents=True, exist_ok=True)

    hdf5_path = hdf5_dir / f"{cfg.chemistry.element}_column_density.hdf5"

    with h5py.File(hdf5_path, "w") as f:

        cfg_serializable = cfg_to_serializable(cfg)
        f.attrs['cfg'] = json.dumps(cfg_serializable)

        xedges = cd_2d.xedges.to_physical()
        ds_x = f.create_dataset("xedges", data=xedges.value)
        ds_x.attrs['unit'] = str(xedges.units)

        yedges = cd_2d.yedges.to_physical()
        ds_y = f.create_dataset("yedges", data=yedges.value)
        ds_y.attrs['unit'] = str(yedges.units)

        element_cddf, element_bin_centers, element_bin_width = cd_2d.column_density_distribution_function(
            ion=None,
            log_column_density_range=cfg.cddf.log_range,
            n_bins=cfg.cddf.bins,
            los_range=proj_range
        )

        grp_elem = f.create_group(f"{cfg.chemistry.element}")

        elem_cd = cd_2d.element_column_density
        ds_elem = grp_elem.create_dataset("column_density", data=elem_cd.value)
        ds_elem.attrs['unit'] = str(elem_cd.units)

        grp_elem.create_dataset("cddf", data=element_cddf)
        grp_elem.create_dataset("bin_centers", data=element_bin_centers)
        grp_elem.create_dataset("bin_width", data=element_bin_width)

        for ion in cfg.chemistry.ion:

            print("Calculating for ion", ion)

            n_ion_column_density = stitched_ions[ion]

            ion_cddf, ion_bin_centers, ion_bin_width = cd_2d.column_density_distribution_function(
                ion=ion,
                ion_column_density=n_ion_column_density,
                log_column_density_range=cfg.cddf.log_range,
                n_bins=cfg.cddf.bins,
                los_range=proj_range
            )

            grp_ion = f.create_group(f"{ion}")

            ds_ion = grp_ion.create_dataset("column_density", data=n_ion_column_density.value)
            ds_ion.attrs["unit"] = str(n_ion_column_density.units)

            grp_ion.create_dataset("cddf", data=ion_cddf)
            grp_ion.create_dataset("bin_centers", data=ion_bin_centers)
            grp_ion.create_dataset("bin_width", data=ion_bin_width)

    print("All data and cfg settings saved to", hdf5_path)