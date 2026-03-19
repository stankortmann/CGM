# cd_swift_hdf5.py

from swiftsimio import load
import numpy as np
import unyt as u
from pathlib import Path
import h5py
import json
from dataclasses import is_dataclass, asdict
from swiftsimio.objects import cosmo_array, cosmo_factor, cosmo_quantity


# own modules
from spec_analysis import unpack_data
from spec_analysis import density_profiles
from spec_analysis import plot

# function
def create_dataset_compressed(group, name, data, dtype=np.float64):
    return group.create_dataset(
        name,
        data=data.astype(dtype), #maybe go to float32 to save space? but be careful with precision loss!
        chunks=True,
        compression="gzip",
        compression_opts=6,
        shuffle=True
    )

def cfg_to_serializable(cfg):
    """
    Recursively convert a dataclass or dict to something JSON serializable.
    Handles cosmo_array, cosmo_quantity, and cosmo_factor from swiftsimio.
    """
    if is_dataclass(cfg):
        cfg = asdict(cfg)  # convert dataclass to dict recursively

    if isinstance(cfg, dict):
        return {k: cfg_to_serializable(v) for k, v in cfg.items()}

    elif isinstance(cfg, (list, tuple)):
        return [cfg_to_serializable(x) for x in cfg]

    # --- SWIFTSIMIO COSMO TYPES ---
    elif isinstance(cfg, cosmo_array):
        # Convert to comoving values, then get value and units
        arr = cfg.to_comoving()  # swiftsimio method
        return {"value": arr.value.tolist(), "unit": str(arr.units)}

    elif isinstance(cfg, cosmo_quantity):
        # Single value with units
        q = cfg.to_comoving()
        return {"value": float(q), "unit": str(q.units)}

    elif isinstance(cfg, cosmo_factor):
        # dimensionless scaling factor
        return float(cfg)

    # --- Fallback ---
    else:
        return cfg

def run_box_column_density(cfg):
    """
    Compute 2D column density and CDDFs for a simulation snapshot,
    and save all results along with cfg settings to an HDF5 file.
    """
    # --- Unpack simulation ---
    data_unpacker = unpack_data.unwrapper(cfg)
    comoving_box_size = data_unpacker.box_size.to("Mpc")
    
    cfg.window.x = [x * comoving_box_size for x in cfg.window.x]
    cfg.window.y = [y * comoving_box_size for y in cfg.window.y]
    cfg.window.z = [z * comoving_box_size for z in cfg.window.z]

    proj_axis = {"x": cfg.window.x, "y": cfg.window.y, "z": cfg.window.z}
    proj_range = proj_axis[cfg.window.projection_axis]

    # --- Load snapshot ---
    region = [cfg.window.x, cfg.window.y, cfg.window.z]
    snapshot = data_unpacker.load_snapshot(load_region=region)
    print("Gas particles are loaded")

    # --- Column density calculation ---
    print("Calculating for element", cfg.chemistry.element)
    cd_2d = density_profiles.column_density_2d_swift(
        cfg=cfg,
        data_unpacker=data_unpacker,
        snapshot=snapshot,
        element=cfg.chemistry.element
    )

    # --- Prepare HDF5 file ---
    hdf5_dir = Path(data_unpacker.output_directory) / "hdf5_data"
    hdf5_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = hdf5_dir / f"{cfg.chemistry.element}_column_density.hdf5"
    
    with h5py.File(hdf5_path, "w") as f:

        # --- Save cfg as JSON attribute ---
        cfg_serializable = cfg_to_serializable(cfg)
        f.attrs['cfg'] = json.dumps(cfg_serializable)

        # --- Save xedges with units ---
        xedges = cd_2d.xedges.to_physical()
        ds_x = create_dataset_compressed(f, "xedges", xedges.value)
        ds_x.attrs['unit'] = str(xedges.units)

        # --- Save yedges with units ---
        yedges = cd_2d.yedges.to_physical()
        ds_y = create_dataset_compressed(f, "yedges", yedges.value)
        ds_y.attrs['unit'] = str(yedges.units)

        # --- Save CDDF for element ---
        element_cddf, element_bin_centers, element_bin_width = cd_2d.column_density_distribution_function(
            column_density=cd_2d.element_column_density,
            log_column_density_range=cfg.cddf.log_range,
            n_bins=cfg.cddf.bins,
            los_range=proj_range
        )

        # --- SAVING FOR THE ELEMENT ---
        grp_elem = f.create_group(f"{cfg.chemistry.element}")

        elem_cd = cd_2d.element_column_density.to_physical()
        ds_elem = create_dataset_compressed(grp_elem, "column_density", elem_cd.value)
        ds_elem.attrs['unit'] = str(elem_cd.units)

        create_dataset_compressed(grp_elem, "cddf", element_cddf)
        create_dataset_compressed(grp_elem, "bin_centers", element_bin_centers)
        grp_elem.create_dataset("bin_width", data=element_bin_width)

        # --- Save ions ---
        for ion in cfg.chemistry.ion:
            print("Calculating for ion", ion)

            n_ion_column_density = cd_2d.column_density_ion(ion=ion).to_physical()

            ion_cddf, ion_bin_centers, ion_bin_width = cd_2d.column_density_distribution_function(
                column_density=n_ion_column_density,
                log_column_density_range=cfg.cddf.log_range,
                n_bins=cfg.cddf.bins,
                los_range=proj_range
            )

            grp_ion = f.create_group(f"{ion}")

            ds_ion = create_dataset_compressed(grp_ion, "column_density", n_ion_column_density.value)
            ds_ion.attrs["unit"] = str(n_ion_column_density.units)

            create_dataset_compressed(grp_ion, "cddf", ion_cddf)
            create_dataset_compressed(grp_ion, "bin_centers", ion_bin_centers)
            grp_ion.create_dataset("bin_width", data=ion_bin_width)

    print("All data and cfg settings saved to", hdf5_path)