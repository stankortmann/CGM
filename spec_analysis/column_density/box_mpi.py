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


def run_box_column_density_parallel(cfg, comm):
    """
    Compute 2D column density and CDDFs using:
    - identical global projection grid on all ranks
    - tile-based masking
    - MPI SUM reduction (no stitching)
    """

    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # -------------------------------------------------
    # Unpack simulation
    # -------------------------------------------------

    data_unpacker = unpack_data.unwrapper(cfg)
    comoving_box_size = data_unpacker.box_size.to("Mpc")

    # Convert window to physical comoving coordinates
    cfg.window.x = [x * comoving_box_size for x in cfg.window.x]
    cfg.window.y = [y * comoving_box_size for y in cfg.window.y]
    cfg.window.z = [z * comoving_box_size for z in cfg.window.z]
    #projection_axis = cfg.window.projection_axis
    proj_axis = {"x": cfg.window.x, "y": cfg.window.y, "z": cfg.window.z}
    proj_range = proj_axis[cfg.window.projection_axis]
    
    x_min, x_max = cfg.window.x
    y_min, y_max = cfg.window.y
    z_min, z_max = cfg.window.z

    # -------------------------------------------------
    # MPI TILE GRID
    # -------------------------------------------------

    n_tile = int(np.sqrt(comm_size))
    if n_tile * n_tile != comm_size:
        if comm_rank == 0:
            raise RuntimeError("MPI ranks must be a perfect square")

    ix = comm_rank % n_tile
    iy = comm_rank // n_tile

    dx = (x_max - x_min) / n_tile
    dy = (y_max - y_min) / n_tile

    # Overlap ONLY for particle loading
    overlap = getattr(cfg.window, "tile_overlap", 0.01) * comoving_box_size

    tile_x = [x_min + ix * dx - overlap,
              x_min + (ix + 1) * dx + overlap]

    tile_y = [y_min + iy * dy - overlap,
              y_min + (iy + 1) * dy + overlap]

    tile_z = [z_min, z_max]

    region = [tile_x, tile_y, tile_z]

    # -------------------------------------------------
    # LOAD TILE WITH OVERLAP
    # -------------------------------------------------

    snapshot = data_unpacker.load_snapshot(load_region=region)

    if comm_rank == 0:
        print("Gas particles loaded (MPI tiled)")

    # -------------------------------------------------
    # COLUMN DENSITY PROJECTION (FULL GLOBAL GRID)
    # -------------------------------------------------

    cd_2d = density_profiles.column_density_2d_swift(
        cfg=cfg,
        data_unpacker=data_unpacker,
        snapshot=snapshot,
        element=cfg.chemistry.element,
        mpi=True
    )

    resolution = cfg.window.resolution

    # IMPORTANT:
    # Do NOT slice. Keep full grid.
    local_element = cd_2d.element_column_density.to_physical()

    local_ions = {}
    for ion in cfg.chemistry.ion:
        local_ions[ion] = cd_2d.column_density_ion(ion).to_physical()

    # -------------------------------------------------
    # TILE MASK (ZERO OUTSIDE CORE REGION)
    # -------------------------------------------------

    # global grid edges (identical on all ranks)
    x_edges = cd_2d.xedges.to_comoving()
    y_edges = cd_2d.yedges.to_comoving()

    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    ix_min = ix * nx // n_tile
    ix_max = (ix + 1) * nx // n_tile

    iy_min = iy * ny // n_tile
    iy_max = (iy + 1) * ny // n_tile

    # Column density arrays are indexed as (x, y).
    mask = np.zeros((nx, ny), dtype=bool)
    mask[ix_min:ix_max, iy_min:iy_max] = True
    print(f"Rank {comm_rank} indices of core region: ix [{ix_min}:{ix_max}], iy [{iy_min}:{iy_max}]")
    
    # Apply mask
    local_element[~mask] = 0.0

    for ion in cfg.chemistry.ion:
        local_ions[ion][~mask] = 0.0
    
    # -------------------------------------------------
    # MPI SUM REDUCTION (ALL RANKS HAVE FULL GRID, JUST SUMMING)
    # -------------------------------------------------

    full_element = comm.reduce(local_element.value, op=MPI.SUM, root=0)

    full_ions = {}
    for ion in cfg.chemistry.ion:
        full_ions[ion] = comm.reduce(local_ions[ion].value, op=MPI.SUM, root=0)

    if comm_rank != 0:
        return

    # Re-wrap into cosmo_array
    n_element_column_density = cosmo_array(
        full_element,
        units=local_element.units,
        comoving=local_element.comoving,
        cosmo_factor=local_element.cosmo_factor,
    )
    for ion in cfg.chemistry.ion:
        full_ions[ion] = cosmo_array(
            full_ions[ion],
            units=local_ions[ion].units,
            comoving=local_ions[ion].comoving,
            cosmo_factor=local_ions[ion].cosmo_factor,
        )

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
            column_density=n_element_column_density,
            log_column_density_range=cfg.cddf.log_range,
            n_bins=cfg.cddf.bins,
            los_range=proj_range
        )

        grp_elem = f.create_group(f"{cfg.chemistry.element}")

        
        ds_elem = grp_elem.create_dataset("column_density", data=n_element_column_density.value)
        ds_elem.attrs['unit'] = str(n_element_column_density.units)

        grp_elem.create_dataset("cddf", data=element_cddf)
        grp_elem.create_dataset("bin_centers", data=element_bin_centers)
        grp_elem.create_dataset("bin_width", data=element_bin_width)

        for ion in cfg.chemistry.ion:

            print("Calculating for ion", ion)

            n_ion_column_density = full_ions[ion]

            ion_cddf, ion_bin_centers, ion_bin_width = cd_2d.column_density_distribution_function(
                column_density=n_ion_column_density,
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