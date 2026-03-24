from swiftsimio import load
import numpy as np
import unyt as u
from pathlib import Path
from swiftsimio.objects import cosmo_array
from mpi4py import MPI

# own modules
from spec_analysis import unpack_data
from spec_analysis import density_profiles
from spec_analysis.save_data import projection_saver



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

    hdf5_dir = Path(data_unpacker.output_directory) / "hdf5_data" / str(cfg.chemistry.element)
    hdf5_dir.mkdir(parents=True, exist_ok=True)

    # Create a single ProjectionSaver instance for all slices and final projection
    # This ensures both per-slice and final projection use consistent settings
    saver = projection_saver(cfg, use_compression=True, dtype=np.float32, comm=comm)

    full_z_min = z_min
    full_z_max = z_max
    dz = (z_max - z_min) / n_slices

    total_element = None
    total_ions = None
    cd_2d_ref = None

    

    
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

        slice_hdf5_path = hdf5_dir / f"slice_{i_slice}.hdf5"
        n_element_column_density, stitched_ions = saver.save_projection_file_tiled_mpi(
            file_path=slice_hdf5_path,
            cd_2d_obj=cd_2d,
            local_element_column_density=local_element,
            local_ion_column_density_map=local_ions,
            los_range_local=slice_proj_range,
            n_tile=n_tile,
            tile_resolution=tile_resolution,
            global_resolution=global_resolution,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            map_tag_base=1000 + 100 * i_slice,
        )

        if comm_rank != 0:
            continue

        if total_element is None:
            total_element = n_element_column_density
            total_ions = {ion: stitched_ions[ion] for ion in cfg.chemistry.ion}
            cd_2d_ref = cd_2d
            cd_2d_ref.__dict__["xedges"] = cosmo_array(
                np.linspace(x_min.value, x_max.value, global_resolution + 1),
                u.Mpc,
                comoving=True,
                scale_factor=snapshot.metadata.scale_factor,
                scale_exponent=1,
            )
            cd_2d_ref.__dict__["yedges"] = cosmo_array(
                np.linspace(y_min.value, y_max.value, global_resolution + 1),
                u.Mpc,
                comoving=True,
                scale_factor=snapshot.metadata.scale_factor,
                scale_exponent=1,
            )
        #summing the column density maps across slices to get the total column density map for the full projection range
        else:
            total_element += n_element_column_density
            for ion in cfg.chemistry.ion:
                total_ions[ion] += stitched_ions[ion]

        print("All data and cfg settings saved to", slice_hdf5_path)

    if comm_rank != 0:
        return

    total_hdf5_path = hdf5_dir / "total.hdf5"
    saver.save_projection_file(
        file_path=total_hdf5_path,
        cd_2d_obj=cd_2d_ref,
        element_column_density=total_element,
        ion_column_density_map=total_ions,
        los_range_local=[full_z_min, full_z_max],
    )

    print("All data and cfg settings saved to", total_hdf5_path)