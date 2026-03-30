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
from spec_analysis import plot
from spec_analysis import save_data
from spec_analysis import chemistry as chem


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


def _collect_tile_map_on_root(comm, local_tile, n_tile, tile_res, full_res, tag):
    """Collect one rank-local tile map per rank onto root without Python object gather."""
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_tile = np.ascontiguousarray(local_tile)

    if rank == 0:
        full_map = np.zeros((full_res, full_res), dtype=local_tile.dtype)

        # Place root tile.
        ix = rank % n_tile
        iy = rank // n_tile
        ix_min = ix * tile_res
        ix_max = (ix + 1) * tile_res
        iy_min = iy * tile_res
        iy_max = (iy + 1) * tile_res
        full_map[ix_min:ix_max, iy_min:iy_max] = local_tile

        # Receive and place all other tiles.
        for src in range(1, size):
            recv_tile = np.empty((tile_res, tile_res), dtype=local_tile.dtype)
            comm.Recv(recv_tile, source=src, tag=tag)

            ix = src % n_tile
            iy = src // n_tile
            ix_min = ix * tile_res
            ix_max = (ix + 1) * tile_res
            iy_min = iy * tile_res
            iy_max = (iy + 1) * tile_res
            full_map[ix_min:ix_max, iy_min:iy_max] = recv_tile

        return full_map

    comm.Send(local_tile, dest=0, tag=tag)
    return None


def run_box_column_density_parallel(cfg, comm):
    """
    Compute 2D column density using:
    - rank-local tile projections to minimize per-rank memory
    - MPI gather and stitch to assemble global map on root
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

    global_resolution = int(cfg.window.resolution)
    tile_resolution = global_resolution // n_tile

    ix = comm_rank % n_tile
    iy = comm_rank // n_tile

    dx = (x_max - x_min) / n_tile
    dy = (y_max - y_min) / n_tile

    # Overlap ONLY for particle loading
    overlap = getattr(cfg.window, "tile_overlap", 0.01) * comoving_box_size

    core_x = [x_min + ix * dx, x_min + (ix + 1) * dx]
    core_y = [y_min + iy * dy, y_min + (iy + 1) * dy]

    tile_x = [core_x[0] - overlap, core_x[1] + overlap]
    tile_y = [core_y[0] - overlap, core_y[1] + overlap]

    tile_z = [z_min, z_max]

    region = [tile_x, tile_y, tile_z]

    # -------------------------------------------------
    # LOAD TILE WITH OVERLAP
    # -------------------------------------------------

    snapshot = data_unpacker.load_snapshot(load_region=region)

    #loading chimes table once to avoid repeated loading in each slice
    chimes = chem.chimes(data_unpacker.chimes_table_path, 
                ions_to_cache=cfg.chemistry.ion)

    if comm_rank == 0:
        print("Gas particles loaded (MPI tiled)")

    # -------------------------------------------------
    # COLUMN DENSITY PROJECTION (LOCAL TILE ONLY)
    # -------------------------------------------------

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
            chimes=chimes,
            snapshot=snapshot,
            element=cfg.chemistry.element,
            mpi=True
        )

        local_element = cd_2d.element_column_density.to_physical()

        local_ions = {}
        for ion in cfg.chemistry.ion:
            local_ions[ion] = cd_2d.column_density_ion(ion).to_physical()
    finally:
        cfg.window.x = original_x
        cfg.window.y = original_y
        cfg.window.resolution = original_resolution

    if comm_rank == 0:
        print("Local tile histograms computed")

    # -------------------------------------------------
    # GATHER TILES AND ASSEMBLE ON ROOT
    # -------------------------------------------------

    full_element = _collect_tile_map_on_root(
        comm=comm,
        local_tile=local_element.value,
        n_tile=n_tile,
        tile_res=tile_resolution,
        full_res=global_resolution,
        tag=101,
    )

    full_ions = {}
    for ion_index, ion in enumerate(cfg.chemistry.ion):
        full_ions[ion] = _collect_tile_map_on_root(
            comm=comm,
            local_tile=local_ions[ion].value,
            n_tile=n_tile,
            tile_res=tile_resolution,
            full_res=global_resolution,
            tag=200 + ion_index,
        )

    if comm_rank != 0:
        return

    # Create global edges for the full assembled map
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

    # Inject global edges into cd_2d object for save_projection_file
    cd_2d.__dict__["xedges"] = global_xedges
    cd_2d.__dict__["yedges"] = global_yedges

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

    hdf5_dir = Path(data_unpacker.output_directory) / "hdf5_data" / str(cfg.chemistry.element)
    hdf5_dir.mkdir(parents=True, exist_ok=True)

    hdf5_path = hdf5_dir / f"{cfg.chemistry.element}_column_density.hdf5"

    save_data.save_projection_file(
        file_path=hdf5_path,
        cfg=cfg,
        cd_2d_obj=cd_2d,
        element_column_density=n_element_column_density,
        ion_column_density_map=full_ions,
        los_range_local=proj_range,
        use_compression=True,
    )

    print("All data and cfg settings saved to", hdf5_path)