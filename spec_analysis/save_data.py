import json
from dataclasses import asdict, is_dataclass

import h5py
import numpy as np
from swiftsimio.objects import cosmo_array, cosmo_factor, cosmo_quantity


def create_dataset_compressed(group, name, data, dtype=np.float64):
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

    xedges_physical = cd_2d_obj.xedges.to_physical()

    yedges_physical = cd_2d_obj.yedges.to_physical()

    cfg_serializable = cfg_to_serializable(cfg)
    los_distance_local = (los_range_local[1] - los_range_local[0]).to("Mpc").to_physical()

    def write_array(group, name, values):
        if use_compression:
            return create_dataset_compressed(group, name, values, dtype=dtype)
        return group.create_dataset(name, data=np.asarray(values).astype(dtype, copy=False))

    with h5py.File(file_path, "w") as f:
        f.attrs["cfg"] = json.dumps(cfg_serializable)

        ds_x = write_array(f, "xedges", xedges_physical.value)
        ds_x.attrs["unit"] = str(xedges_physical.units)

        ds_y = write_array(f, "yedges", yedges_physical.value)
        ds_y.attrs["unit"] = str(yedges_physical.units)

        proj_vals = np.array(
            [los_range_local[0].to_physical().value, los_range_local[1].to_physical().value],
            dtype=dtype,
        )
        ds_proj = write_array(f, "proj_range", proj_vals)
        ds_proj.attrs["unit"] = str(los_range_local[0].to_physical().units)

        ds_los = f.create_dataset("los_distance", data=float(los_distance_local.value))
        ds_los.attrs["unit"] = str(los_distance_local.units)

        ds_zmin = f.create_dataset("z_min", data=float(los_range_local[0].to_physical().value))
        ds_zmin.attrs["unit"] = str(los_range_local[0].to_physical().units)

        ds_zmax = f.create_dataset("z_max", data=float(los_range_local[1].to_physical().value))
        ds_zmax.attrs["unit"] = str(los_range_local[1].to_physical().units)

        element_cddf, element_bin_centers, element_bin_width = cd_2d_obj.column_density_distribution_function(
            column_density=element_column_density,
            log_column_density_range=cfg.cddf.log_range,
            n_bins=cfg.cddf.bins,
            los_range=los_range_local,
        )

        grp_elem = f.create_group(f"{cfg.chemistry.element}")
        ds_elem = write_array(grp_elem, "column_density", element_column_density.value)
        ds_elem.attrs["unit"] = str(element_column_density.units)
        write_array(grp_elem, "cddf", element_cddf)
        write_array(grp_elem, "bin_centers", element_bin_centers)
        grp_elem.create_dataset("bin_width", data=element_bin_width)

        for ion in cfg.chemistry.ion:
            n_ion_column_density = ion_column_density_map[ion]

            ion_cddf, ion_bin_centers, ion_bin_width = cd_2d_obj.column_density_distribution_function(
                column_density=n_ion_column_density,
                log_column_density_range=cfg.cddf.log_range,
                n_bins=cfg.cddf.bins,
                los_range=los_range_local,
            )

            grp_ion = f.create_group(f"{ion}")
            ds_ion = write_array(grp_ion, "column_density", n_ion_column_density.value)
            ds_ion.attrs["unit"] = str(n_ion_column_density.units)
            write_array(grp_ion, "cddf", ion_cddf)
            write_array(grp_ion, "bin_centers", ion_bin_centers)
            grp_ion.create_dataset("bin_width", data=ion_bin_width)