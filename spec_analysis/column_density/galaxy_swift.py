from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import unyt as u
from swiftsimio.objects import cosmo_quantity

import spec_analysis.chemistry as chem

from spec_analysis import density_profiles, unpack_data, galaxy_selection as gal_sel, save_data


def _save_transverse_plot(r_centers, profile, name, output_dir):
    """Save radial average column density as function of transverse distance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{name}_transverse.png"

    x = r_centers.to("kpc").value
    y = profile.to("1/cm**2").value

    mask = np.isfinite(y) & (y > 0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x[mask], np.log10(y[mask]), lw=2)
    ax.set_xlabel("Transverse distance [kpc]")
    ax.set_ylabel(r"$\log_{10}(\langle N \rangle)\;[\mathrm{cm}^{-2}]$")
    ax.set_title(f"Transverse average column density: {name}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Finished", file_path)


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
        r_min=1 * u.kpc,
        r_max=r_max,
        n_bins=30,
        log_bins=False,
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
            r_min=1 * u.kpc,
            r_max=r_max,
            n_bins=30,
            log_bins=False,
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