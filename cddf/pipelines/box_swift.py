# cd_swift_hdf5.py

from swiftsimio import load
import numpy as np
import unyt as u
from pathlib import Path


# own modules
from cddf import unpack_data
from cddf import density_profiles
from cddf import plot
from cddf import save_data

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
    
    elem_cd = cd_2d.element_column_density.to_physical()
    ion_column_density_map = {}
    for ion in cfg.chemistry.ion:
        print("Calculating for ion", ion)
        ion_column_density_map[ion] = cd_2d.column_density_ion(ion=ion).to_physical()

    save_data.save_projection_file(
        file_path=hdf5_path,
        cfg=cfg,
        cd_2d_obj=cd_2d,
        element_column_density=elem_cd,
        ion_column_density_map=ion_column_density_map,
        los_range_local=proj_range,
        use_compression=True,
    )

    print("All data and cfg settings saved to", hdf5_path)