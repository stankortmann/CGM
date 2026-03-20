from swiftsimio import load
import numpy as np
import unyt as u
from pathlib import Path
from swiftsimio.objects import cosmo_array
from mpi4py import MPI

# own modules
from spec_analysis import unpack_data
from spec_analysis import density_profiles
from spec_analysis import save_data


def _assemble_global_from_tiles(tile_maps, n_tile, tile_res, full_res):
    """Stitch rank-local (x, y) tile maps into one global map on root."""
    full_map = np.zeros((full_res, full_res), dtype=tile_maps[0].dtype)
    for rank_id, tile_map in enumerate(tile_maps):
        ix = rank_id % n_tile
        iy = rank_id // n_tile
        ix_min = ix * tile_res
        ix_max = (ix + 1) * tile_res
        iy_min = iy * tile_res
        iy_max = (iy + 1) * tile_res
        full_map[ix_min:ix_max, iy_min:iy_max] = tile_map
    return full_map



def run_slice_column_density_parallel(cfg, comm):
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

    global_resolution = int(cfg.window.resolution)
    tile_resolution = global_resolution // n_tile

    ix = comm_rank % n_tile
    iy = comm_rank // n_tile

    dx = (x_max - x_min) / n_tile
    dy = (y_max - y_min) / n_tile

    # Overlap ONLY for particle loading, take 1 cMpc as default
    overlap = getattr(cfg.window, "tile_overlap", 0.01) * comoving_box_size

    core_x = [x_min + ix * dx, x_min + (ix + 1) * dx]
    core_y = [y_min + iy * dy, y_min + (iy + 1) * dy]

    tile_x = [core_x[0] - overlap, core_x[1] + overlap]
    tile_y = [core_y[0] - overlap, core_y[1] + overlap]

    hdf5_dir = Path(data_unpacker.output_directory) / "hdf5_data"
    hdf5_dir.mkdir(parents=True, exist_ok=True)

    full_z_min = z_min
    full_z_max = z_max
    dz = (z_max - z_min) / n_slices

    total_element = None
    total_ions = None
    cd_2d_ref = None

    global_xedges = None
    global_yedges = None

    
    for i_slice in range(n_slices):
        slice_z_min = z_min + i_slice * dz
        slice_z_max = z_min + (i_slice + 1) * dz
        slice_proj_range = [slice_z_min, slice_z_max]

        
        region = [tile_x, tile_y, slice_proj_range]

        snapshot = data_unpacker.load_snapshot(load_region=region)

        if comm_rank == 0:
            print(f"Gas particles loaded (MPI tiled) for slice {i_slice}")

        original_x = cfg.window.x
        original_y = cfg.window.y
        original_resolution = cfg.window.resolution

        try:
            # Compute only the rank-local tile map to avoid full-grid memory on each rank.
            cfg.window.x = core_x
            cfg.window.y = core_y
            cfg.window.resolution = tile_resolution

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
        #get the configurations back to the original settings to avoid issues with next slice loading
        finally:
            cfg.window.x = original_x
            cfg.window.y = original_y
            cfg.window.resolution = original_resolution

        print(f"Rank {comm_rank} computed local tile for slice {i_slice}: ix {ix}, iy {iy}")

        gathered_element = comm.gather(np.ascontiguousarray(local_element.value), root=0)

        gathered_ions = {}
        for ion in cfg.chemistry.ion:
            gathered_ions[ion] = comm.gather(np.ascontiguousarray(local_ions[ion].value), root=0)

        if comm_rank != 0:
            continue

        full_element = _assemble_global_from_tiles(
            tile_maps=gathered_element,
            n_tile=n_tile,
            tile_res=tile_resolution,
            full_res=global_resolution,
        )

        full_ions = {}
        for ion in cfg.chemistry.ion:
            full_ions[ion] = _assemble_global_from_tiles(
                tile_maps=gathered_ions[ion],
                n_tile=n_tile,
                tile_res=tile_resolution,
                full_res=global_resolution,
            )

        if global_xedges is None:
            global_xedges = cosmo_array(
                np.linspace(x_min.value, x_max.value, global_resolution + 1),
                u.Mpc,
                comoving=True,
                scale_factor=snapshot.metadata.scale_factor,
                scale_exponent=1,
            )
            global_yedges = cosmo_array(
                np.linspace(y_min.value, y_max.value, global_resolution + 1),
                u.Mpc,
                comoving=True,
                scale_factor=snapshot.metadata.scale_factor,
                scale_exponent=1,
            )

            # Save full stitched maps with global edges metadata.
            cd_2d.__dict__["xedges"] = global_xedges
            cd_2d.__dict__["yedges"] = global_yedges
            
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
            cd_2d_ref = cd_2d
            cd_2d_ref.__dict__["xedges"] = global_xedges
            cd_2d_ref.__dict__["yedges"] = global_yedges
        #summing the column density maps across slices to get the total column density map for the full projection range
        else:
            total_element += n_element_column_density
            for ion in cfg.chemistry.ion:
                total_ions[ion] += full_ions[ion]

        slice_hdf5_path = hdf5_dir / f"slice_{i_slice}.hdf5"
        save_data.save_projection_file(
            file_path=slice_hdf5_path,
            cfg=cfg,
            cd_2d_obj=cd_2d,
            element_column_density=n_element_column_density,
            ion_column_density_map=full_ions,
            los_range_local=slice_proj_range,
            use_compression=True,
        )

        print("All data and cfg settings saved to", slice_hdf5_path)

    if comm_rank != 0:
        return

    total_hdf5_path = hdf5_dir / "total.hdf5"
    save_data.save_projection_file(
        file_path=total_hdf5_path,
        cfg=cfg,
        cd_2d_obj=cd_2d_ref,
        element_column_density=total_element,
        ion_column_density_map=total_ions,
        los_range_local=[full_z_min, full_z_max],
        use_compression=True,
    )

    print("All data and cfg settings saved to", total_hdf5_path)