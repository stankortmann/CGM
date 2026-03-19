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

def save_projection_file(file_path, cd_2d_obj, element_column_density, ion_column_density_map, los_range_local):
    los_distance_local = (los_range_local[1] - los_range_local[0]).to("Mpc").to_physical()

    with h5py.File(file_path, "w") as f:
        f.attrs['cfg'] = json.dumps(cfg_serializable)

        ds_x = f.create_dataset("xedges", data=xedges_physical.value)
        ds_x.attrs['unit'] = str(xedges_physical.units)

        ds_y = f.create_dataset("yedges", data=yedges_physical.value)
        ds_y.attrs['unit'] = str(yedges_physical.units)

        proj_vals = np.array([los_range_local[0].to_physical().value, los_range_local[1].to_physical().value])
        ds_proj = f.create_dataset("proj_range", data=proj_vals)
        ds_proj.attrs['unit'] = str(los_range_local[0].to_physical().units)

        ds_los = f.create_dataset("los_distance", data=los_distance_local.value)
        ds_los.attrs['unit'] = str(los_distance_local.units)

        ds_zmin = f.create_dataset("z_min", data=los_range_local[0].to_physical().value)
        ds_zmin.attrs['unit'] = str(los_range_local[0].to_physical().units)

        ds_zmax = f.create_dataset("z_max", data=los_range_local[1].to_physical().value)
        ds_zmax.attrs['unit'] = str(los_range_local[1].to_physical().units)

        element_cddf, element_bin_centers, element_bin_width = cd_2d_obj.column_density_distribution_function(
            column_density=element_column_density,
            log_column_density_range=cfg.cddf.log_range,
            n_bins=cfg.cddf.bins,
            los_range=los_range_local
        )

        grp_elem = f.create_group(f"{cfg.chemistry.element}")
        ds_elem = grp_elem.create_dataset("column_density", data=element_column_density.value)
        ds_elem.attrs['unit'] = str(element_column_density.units)
        grp_elem.create_dataset("cddf", data=element_cddf)
        grp_elem.create_dataset("bin_centers", data=element_bin_centers)
        grp_elem.create_dataset("bin_width", data=element_bin_width)

        for ion in cfg.chemistry.ion:
            print("Calculating for ion", ion)
            n_ion_column_density = ion_column_density_map[ion]

            ion_cddf, ion_bin_centers, ion_bin_width = cd_2d_obj.column_density_distribution_function(
                column_density=n_ion_column_density,
                log_column_density_range=cfg.cddf.log_range,
                n_bins=cfg.cddf.bins,
                los_range=los_range_local
            )

            grp_ion = f.create_group(f"{ion}")
            ds_ion = grp_ion.create_dataset("column_density", data=n_ion_column_density.value)
            ds_ion.attrs["unit"] = str(n_ion_column_density.units)
            grp_ion.create_dataset("cddf", data=ion_cddf)
            grp_ion.create_dataset("bin_centers", data=ion_bin_centers)
            grp_ion.create_dataset("bin_width", data=ion_bin_width)



def run_box_column_density_parallel(cfg, comm):
    """
    Compute 2D column density and CDDFs using:
    - identical global projection grid on all ranks
    - tile-based masking
    - MPI SUM reduction (no stitching)
    """

    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    if cfg.window.projection_axis != "z":
        raise ValueError("box_mpi_multiple currently supports projection slicing only along z-axis")

    n_slices = int(cfg.window.projection_slices)
    if n_slices < 1:
        raise ValueError("cfg.window.projection_slices must be >= 1")

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
    if cfg.window.resolution % n_tile != 0:
        if comm_rank == 0:
            raise RuntimeError("cfg.window.resolution must be divisible by sqrt(MPI size) for tiled MPI")
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

    hdf5_dir = Path(data_unpacker.output_directory) / "hdf5_data"
    hdf5_dir.mkdir(parents=True, exist_ok=True)

    cfg_serializable = cfg_to_serializable(cfg)
    full_z_min = z_min
    full_z_max = z_max
    dz = (z_max - z_min) / n_slices

    total_element = None
    total_ions = None
    xedges_physical = None
    yedges_physical = None
    cd_2d_ref = None

    
    for i_slice in range(n_slices):
        slice_z_min = z_min + i_slice * dz
        slice_z_max = z_min + (i_slice + 1) * dz
        slice_proj_range = [slice_z_min, slice_z_max]

        
        region = [tile_x, tile_y, slice_proj_range]

        snapshot = data_unpacker.load_snapshot(load_region=region)

        if comm_rank == 0:
            print(f"Gas particles loaded (MPI tiled) for slice {i_slice}")

        cd_2d = density_profiles.column_density_2d_swift(
            cfg=cfg,
            data_unpacker=data_unpacker,
            snapshot=snapshot,
            element=cfg.chemistry.element,
            mpi=True
        )

        local_element = cd_2d.element_column_density.to_physical()

        local_ions = {}
        for ion in cfg.chemistry.ion:
            local_ions[ion] = cd_2d.column_density_ion(ion).to_physical()

        x_edges = cd_2d.xedges.to_comoving()
        y_edges = cd_2d.yedges.to_comoving()

        nx = len(x_edges) - 1
        ny = len(y_edges) - 1

        ix_min = ix * nx // n_tile
        ix_max = (ix + 1) * nx // n_tile

        iy_min = iy * ny // n_tile
        iy_max = (iy + 1) * ny // n_tile

        mask = np.zeros((nx, ny), dtype=bool)
        mask[ix_min:ix_max, iy_min:iy_max] = True
        print(f"Rank {comm_rank} indices of core region for slice {i_slice}: ix [{ix_min}:{ix_max}], iy [{iy_min}:{iy_max}]")

        local_element[~mask] = 0.0
        for ion in cfg.chemistry.ion:
            local_ions[ion][~mask] = 0.0

        full_element = comm.reduce(local_element.value, op=MPI.SUM, root=0)

        full_ions = {}
        for ion in cfg.chemistry.ion:
            full_ions[ion] = comm.reduce(local_ions[ion].value, op=MPI.SUM, root=0)

        if comm_rank != 0:
            continue

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

        if total_element is None:
            total_element = n_element_column_density.copy()
            total_ions = {ion: full_ions[ion].copy() for ion in cfg.chemistry.ion}
            xedges_physical = cd_2d.xedges.to_physical()
            yedges_physical = cd_2d.yedges.to_physical()
            cd_2d_ref = cd_2d
        else:
            total_element += n_element_column_density
            for ion in cfg.chemistry.ion:
                total_ions[ion] += full_ions[ion]

        slice_hdf5_path = hdf5_dir / f"slice_{i_slice}.hdf5"
        save_projection_file(
            file_path=slice_hdf5_path,
            cd_2d_obj=cd_2d,
            element_column_density=n_element_column_density,
            ion_column_density_map=full_ions,
            los_range_local=slice_proj_range,
        )

        print("All data and cfg settings saved to", slice_hdf5_path)

    if comm_rank != 0:
        return

    total_hdf5_path = hdf5_dir / "total.hdf5"
    save_projection_file(
        file_path=total_hdf5_path,
        cd_2d_obj=cd_2d_ref,
        element_column_density=total_element,
        ion_column_density_map=total_ions,
        los_range_local=[full_z_min, full_z_max],
    )

    print("All data and cfg settings saved to", total_hdf5_path)