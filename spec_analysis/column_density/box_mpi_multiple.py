# cd_swift_hdf5.py

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
            cd_2d_ref = cd_2d
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