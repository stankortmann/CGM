from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import unyt as u
from swiftsimio.objects import cosmo_quantity

import cddf.chemistry as chem

from cddf import density_profiles, unpack_data, galaxy_selection as gal_sel, save_data





def run_halo_column_density(cfg):
    """
    Compute 2D column densities and CDDFs for a single halo.

    Parameters
    ----------
    cfg : Config object
        Configuration containing simulation, chemistry, and window info.

    Returns
    -------
    None
    """

    # --- Unpack data ---
    data_unpacker = unpack_data.unwrapper(cfg)

    # Extend galaxy selection with proper units
    cfg.galaxy.extend = cosmo_quantity(
        cfg.galaxy.extend_value,
        u.Unit(cfg.galaxy.extend_unit),
        comoving=False,
        scale_factor=data_unpacker.scale_factor,
        scale_exponent=1
    )
    # Single-process execution (no MPI), with threading enabled inside project_gas.
    chimes = chem.chimes(data_unpacker.chimes_table_path, ions_to_cache=cfg.chemistry.ion)
     
    # Select single galaxy/halo
    single_galaxy = gal_sel.single_galaxy_swift_galaxy(cfg=cfg, data_unpacker=data_unpacker)
    print("Galaxy is selected")

    # Full line-of-sight range through the selected halo region.
    halo_proj_axis = {
        "x": single_galaxy.mask[0],
        "y": single_galaxy.mask[1],
        "z": single_galaxy.mask[2]
    }
    halo_proj_range = halo_proj_axis[cfg.window.projection_axis]

    cd_2d = density_profiles.column_density_2d_swift(
        cfg=cfg,
        data_unpacker=data_unpacker,
        chimes=chimes,
        element=cfg.chemistry.element,
        halo=single_galaxy
    )
    print("2D column density class is set up")

    n_element_column_density = cd_2d.element_column_density.to_physical()
    ion_column_density_map = {}

    # --- Transverse radial profiles ---
    r_max = cfg.galaxy.extend.to_physical()
    transverse_profiles = {}
    

    # Element transverse profile (from 2D projection map)
    r_centers, cd_profile = cd_2d.radial_column_density_profile(
        column_density_2d=n_element_column_density,
        
    )
    transverse_profiles[cfg.chemistry.element] = {
        "r_centers": r_centers,
        "column_density": cd_profile,
    }
    

    # Ions transverse profiles
    for ion in cfg.chemistry.ion:
        n_ion_column_density = cd_2d.column_density_ion(ion=ion).to_physical()
        ion_column_density_map[ion] = n_ion_column_density

        ion_r_centers, ion_cd_profile = cd_2d.radial_column_density_profile(
            column_density_2d=n_ion_column_density,
            
        )

        transverse_profiles[ion] = {
            "r_centers": ion_r_centers,
            "column_density": ion_cd_profile,
        }
        

    hdf5_dir = Path(data_unpacker.output_directory) / "hdf5_data" / str(cfg.chemistry.element)
    hdf5_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = hdf5_dir / f"halo_{int(np.asarray(single_galaxy.catalogue_id).item())}.hdf5"

    save_data.save_galaxy_projection_file(
        file_path=hdf5_path,
        cfg=cfg,
        cd_2d_obj=cd_2d,
        element_column_density=n_element_column_density,
        ion_column_density_map=ion_column_density_map,
        los_range_local=halo_proj_range,
        halo=single_galaxy,
        transverse_profiles=transverse_profiles,
        use_compression=True,
        dtype=np.float32,
    )

    print("All data and cfg settings saved to", hdf5_path)