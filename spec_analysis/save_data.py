import json
from dataclasses import asdict, is_dataclass

import h5py
import numpy as np
from mpi4py import MPI
from swiftsimio.objects import cosmo_array, cosmo_factor, cosmo_quantity


def create_dataset_compressed(group, name, data, dtype=np.float64):
    """Create a compressed HDF5 dataset."""
    arr = np.asarray(data)
    return group.create_dataset(
        name,
        data=arr.astype(dtype, copy=False),
        chunks=True,
        compression="gzip",
        compression_opts=6,
        shuffle=True,
    )


def cfg_to_serializable(cfg):
    """Recursively convert config objects to JSON-serializable dict."""
    if is_dataclass(cfg):
        cfg = asdict(cfg)

    if isinstance(cfg, dict):
        return {k: cfg_to_serializable(v) for k, v in cfg.items()}

    if isinstance(cfg, (list, tuple)):
        return [cfg_to_serializable(x) for x in cfg]

    if isinstance(cfg, cosmo_array):
        arr = cfg.to_comoving()
        return {"value": arr.value.tolist(), "unit": str(arr.units)}

    if isinstance(cfg, cosmo_quantity):
        q = cfg.to_comoving()
        return {"value": float(q), "unit": str(q.units)}

    if isinstance(cfg, cosmo_factor):
        return float(cfg)

    return cfg


class projection_saver:
    """Handles HDF5 I/O for 2D column density projections and CDDFs."""

    def __init__(self, cfg, use_compression=False, dtype=np.float64, comm=None):
        """
        Initialize the saver with config and output settings.

        Parameters
        ----------
        cfg : dataclass
            Configuration object with chemistry, cddf, and window settings.
        use_compression : bool, optional
            Whether to use gzip compression in HDF5 output.
        dtype : np.dtype, optional
            Output data type for HDF5 datasets.
        comm : mpi4py.MPI.Comm, optional
            MPI communicator for collective operations.
        """
        self.cfg = cfg
        self.use_compression = use_compression
        self.dtype = dtype
        self.comm = comm if comm is not None else MPI.COMM_SELF

    def _write_array(self, group, name, values):
        """Helper to write array with optional compression."""
        if self.use_compression:
            return create_dataset_compressed(group, name, values, dtype=self.dtype)
        return group.create_dataset(name, data=np.asarray(values).astype(self.dtype, copy=False))

    def _collect_tile_map_on_root(self, local_tile, n_tile, tile_res, full_res, tag):
        """Stitch rank-local tile map onto root via array Send/Recv."""
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()

        local_tile = np.ascontiguousarray(local_tile)

        if rank == 0:
            full_map = np.zeros((full_res, full_res), dtype=local_tile.dtype)

            ix = rank % n_tile
            iy = rank // n_tile
            ix_min = ix * tile_res
            ix_max = (ix + 1) * tile_res
            iy_min = iy * tile_res
            iy_max = (iy + 1) * tile_res
            full_map[ix_min:ix_max, iy_min:iy_max] = local_tile

            for src in range(1, size):
                recv_tile = np.empty((tile_res, tile_res), dtype=local_tile.dtype)
                self.comm.Recv(recv_tile, source=src, tag=tag)

                ix = src % n_tile
                iy = src // n_tile
                ix_min = ix * tile_res
                ix_max = (ix + 1) * tile_res
                iy_min = iy * tile_res
                iy_max = (iy + 1) * tile_res
                full_map[ix_min:ix_max, iy_min:iy_max] = recv_tile

            return full_map

        self.comm.Send(local_tile, dest=0, tag=tag)
        return None

    def _reduced_cddf_from_local_tile(
        self,
        cd_2d_obj,
        local_column_density,
        log_column_density_range,
        n_bins,
        los_range_local,
    ):
        """Compute rank-local CDDF, sum across ranks, and average."""
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()

        local_cddf, local_centers, local_bin_width = cd_2d_obj.column_density_distribution_function(
            column_density=local_column_density,
            log_column_density_range=log_column_density_range,
            n_bins=n_bins,
            los_range=los_range_local,
        )

        local_cddf = np.asarray(local_cddf, dtype=np.float64)
        total_cddf = np.zeros_like(local_cddf) if rank == 0 else None
        self.comm.Reduce(local_cddf, total_cddf, op=MPI.SUM, root=0)

        if rank != 0:
            return None

        # Average the CDDF across all tiles
        total_cddf /= size

        return total_cddf, local_centers, local_bin_width

    def _write_projection_hdf5(
        self,
        file_path,
        xedges_physical,
        yedges_physical,
        element_column_density,
        ion_column_density_map,
        los_range_local,
        element_cddf_tuple,
        ion_cddf_map,
    ):
        """Write full projection with CD maps and CDDFs to HDF5."""
        cfg_serializable = cfg_to_serializable(self.cfg)
        los_distance_local = (los_range_local[1] - los_range_local[0]).to("Mpc").to_physical()
        save_projection_map = self.cfg.data_output.save_projection

        with h5py.File(file_path, "w") as f:
            f.attrs["cfg"] = json.dumps(cfg_serializable)

            ds_x = self._write_array(f, "xedges", xedges_physical.value)
            ds_x.attrs["unit"] = str(xedges_physical.units)

            ds_y = self._write_array(f, "yedges", yedges_physical.value)
            ds_y.attrs["unit"] = str(yedges_physical.units)

            proj_vals = np.array(
                [los_range_local[0].to_physical().value, los_range_local[1].to_physical().value],
                dtype=self.dtype,
            )
            ds_proj = self._write_array(f, "proj_range", proj_vals)
            ds_proj.attrs["unit"] = str(los_range_local[0].to_physical().units)

            ds_los = f.create_dataset("los_distance", data=float(los_distance_local.value))
            ds_los.attrs["unit"] = str(los_distance_local.units)

            ds_zmin = f.create_dataset("z_min", data=float(los_range_local[0].to_physical().value))
            ds_zmin.attrs["unit"] = str(los_range_local[0].to_physical().units)

            ds_zmax = f.create_dataset("z_max", data=float(los_range_local[1].to_physical().value))
            ds_zmax.attrs["unit"] = str(los_range_local[1].to_physical().units)

            element_cddf, element_bin_centers, element_bin_width = element_cddf_tuple

            grp_elem = f.create_group(f"{self.cfg.chemistry.element}")
            if save_projection_map:
                ds_elem = self._write_array(grp_elem, "column_density", element_column_density.value)
                ds_elem.attrs["unit"] = str(element_column_density.units)
            self._write_array(grp_elem, "cddf", element_cddf)
            self._write_array(grp_elem, "bin_centers", element_bin_centers)
            grp_elem.create_dataset("bin_width", data=element_bin_width)

            for ion in self.cfg.chemistry.ion:
                n_ion_column_density = ion_column_density_map[ion]
                ion_cddf, ion_bin_centers, ion_bin_width = ion_cddf_map[ion]

                grp_ion = f.create_group(f"{ion}")
                if save_projection_map:
                    ds_ion = self._write_array(grp_ion, "column_density", n_ion_column_density.value)
                    ds_ion.attrs["unit"] = str(n_ion_column_density.units)
                self._write_array(grp_ion, "cddf", ion_cddf)
                self._write_array(grp_ion, "bin_centers", ion_bin_centers)
                grp_ion.create_dataset("bin_width", data=ion_bin_width)

    def save_projection_file(
        self,
        file_path,
        cd_2d_obj,
        element_column_density,
        ion_column_density_map,
        los_range_local,
    ):
        """Save single projection with full CD maps and CDDFs."""
        xedges_physical = cd_2d_obj.xedges.to_physical()
        yedges_physical = cd_2d_obj.yedges.to_physical()

        element_cddf_tuple = cd_2d_obj.column_density_distribution_function(
            column_density=element_column_density,
            log_column_density_range=self.cfg.cddf.log_range,
            n_bins=self.cfg.cddf.bins,
            los_range=los_range_local,
        )

        ion_cddf_map = {}
        for ion in self.cfg.chemistry.ion:
            n_ion_column_density = ion_column_density_map[ion]
            ion_cddf_map[ion] = cd_2d_obj.column_density_distribution_function(
                column_density=n_ion_column_density,
                log_column_density_range=self.cfg.cddf.log_range,
                n_bins=self.cfg.cddf.bins,
                los_range=los_range_local,
            )

        self._write_projection_hdf5(
            file_path=file_path,
            xedges_physical=xedges_physical,
            yedges_physical=yedges_physical,
            element_column_density=element_column_density,
            ion_column_density_map=ion_column_density_map,
            los_range_local=los_range_local,
            element_cddf_tuple=element_cddf_tuple,
            ion_cddf_map=ion_cddf_map,
        )

    def save_projection_file_tiled_mpi(
        self,
        file_path,
        cd_2d_obj,
        local_element_column_density,
        local_ion_column_density_map,
        los_range_local,
        n_tile,
        tile_resolution,
        global_resolution,
        x_min,
        x_max,
        y_min,
        y_max,
        map_tag_base=100,
    ):
        """Join tiled maps via MPI, reduce CDDFs, and save on root."""
        rank = self.comm.Get_rank()

        full_element = self._collect_tile_map_on_root(
            local_tile=local_element_column_density.to_physical().value,
            n_tile=n_tile,
            tile_res=tile_resolution,
            full_res=global_resolution,
            tag=map_tag_base,
        )

        full_ions = {}
        for ion_index, ion in enumerate(self.cfg.chemistry.ion):
            full_ions[ion] = self._collect_tile_map_on_root(
                local_tile=local_ion_column_density_map[ion].to_physical().value,
                n_tile=n_tile,
                tile_res=tile_resolution,
                full_res=global_resolution,
                tag=map_tag_base + 100 + ion_index,
            )

        element_cddf_tuple = self._reduced_cddf_from_local_tile(
            cd_2d_obj=cd_2d_obj,
            local_column_density=local_element_column_density,
            log_column_density_range=self.cfg.cddf.log_range,
            n_bins=self.cfg.cddf.bins,
            los_range_local=los_range_local,
        )

        ion_cddf_map = {}
        for ion in self.cfg.chemistry.ion:
            ion_cddf_map[ion] = self._reduced_cddf_from_local_tile(
                cd_2d_obj=cd_2d_obj,
                local_column_density=local_ion_column_density_map[ion],
                log_column_density_range=self.cfg.cddf.log_range,
                n_bins=self.cfg.cddf.bins,
                los_range_local=los_range_local,
            )

        if rank != 0:
            return None, None

        scale_factor = cd_2d_obj.snapshot.metadata.scale_factor
        global_xedges = cosmo_array(
            np.linspace(x_min.value, x_max.value, global_resolution + 1),
            x_min.units,
            comoving=True,
            scale_factor=scale_factor,
            scale_exponent=1,
        )
        global_yedges = cosmo_array(
            np.linspace(y_min.value, y_max.value, global_resolution + 1),
            y_min.units,
            comoving=True,
            scale_factor=scale_factor,
            scale_exponent=1,
        )

        n_element_column_density = cosmo_array(
            full_element,
            units=local_element_column_density.units,
            comoving=local_element_column_density.comoving,
            cosmo_factor=local_element_column_density.cosmo_factor,
        )

        stitched_ions = {}
        for ion in self.cfg.chemistry.ion:
            stitched_ions[ion] = cosmo_array(
                full_ions[ion],
                units=local_ion_column_density_map[ion].units,
                comoving=local_ion_column_density_map[ion].comoving,
                cosmo_factor=local_ion_column_density_map[ion].cosmo_factor,
            )

        self._write_projection_hdf5(
            file_path=file_path,
            xedges_physical=global_xedges.to_physical(),
            yedges_physical=global_yedges.to_physical(),
            element_column_density=n_element_column_density,
            ion_column_density_map=stitched_ions,
            los_range_local=los_range_local,
            element_cddf_tuple=element_cddf_tuple,
            ion_cddf_map=ion_cddf_map,
        )

        return n_element_column_density, stitched_ions


# Backward-compatible module-level functions that wrap the class
def save_projection_file(
    file_path,
    cfg,
    cd_2d_obj,
    element_column_density,
    ion_column_density_map,
    los_range_local,
    use_compression=False,
    dtype=np.float64,
    ):
    """Module-level wrapper for single projection save."""
    saver = projection_saver(cfg, use_compression=use_compression, dtype=dtype)
    saver.save_projection_file(
        file_path=file_path,
        cd_2d_obj=cd_2d_obj,
        element_column_density=element_column_density,
        ion_column_density_map=ion_column_density_map,
        los_range_local=los_range_local,
    )


def save_projection_file_tiled_mpi(
    file_path,
    cfg,
    cd_2d_obj,
    local_element_column_density,
    local_ion_column_density_map,
    los_range_local,
    comm,
    n_tile,
    tile_resolution,
    global_resolution,
    x_min,
    x_max,
    y_min,
    y_max,
    map_tag_base=100,
    use_compression=False,
    dtype=np.float64,
):
    """Module-level wrapper for MPI tiled save."""
    saver = projection_saver(cfg, use_compression=use_compression, dtype=dtype, comm=comm)
    return saver.save_projection_file_tiled_mpi(
        file_path=file_path,
        cd_2d_obj=cd_2d_obj,
        local_element_column_density=local_element_column_density,
        local_ion_column_density_map=local_ion_column_density_map,
        los_range_local=los_range_local,
        n_tile=n_tile,
        tile_resolution=tile_resolution,
        global_resolution=global_resolution,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        map_tag_base=map_tag_base,
    )