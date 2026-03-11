# cd_swift.py

import numpy as np
import unyt as u
from spec_analysis import density_profiles
from spec_analysis import plot
from spec_analysis import unpack_data


def run_box_column_density(cfg):
    """
    Compute 2D column densities and CDDFs for elements and ions in a SWIFT snapshot.

    Parameters
    ----------
    cfg : Config object
        Configuration containing simulation, chemistry, and window info.

    Returns
    -------
    None
    """

    # --- Load snapshot ---
    data_unpacker = unpack_data.unwrapper(cfg)
    comoving_box_size = data_unpacker.box_size.to("Mpc")
    
    # Scale window to comoving box
    cfg.window.x = [x * comoving_box_size for x in cfg.window.x]
    cfg.window.y = [y * comoving_box_size for y in cfg.window.y]
    cfg.window.z = [z * comoving_box_size for z in cfg.window.z]

    region = [cfg.window.x, cfg.window.y, cfg.window.z]
    snapshot = data_unpacker.load_snapshot(load_region=region)
    gas_particles = snapshot.gas
    print("Gas particles are loaded")

    # --- Column density ---
    cd_2d = density_profiles.column_density_2d(
        cfg=cfg,
        data_unpacker=data_unpacker,
        length_unit="Mpc",
        gas_particles=gas_particles,
        element=cfg.chemistry.element
    )

    n_element_column_density = cd_2d.element_column_density
    print("Sum of element column density:", np.sum(n_element_column_density))

    # --- Plotter ---
    plotter = plot.column_density_plotter(
        x_edges=cd_2d.xedges,
        y_edges=cd_2d.yedges,
        length_unit="Mpc",
        data_unpacker=data_unpacker
    )

    # 2D element plot
    plotter.plot_xy(
        column_density_values=n_element_column_density.to("1/cm**2").value,
        element=cfg.chemistry.element,
        log_scale=True
    )

    # Projection range for CDDF
    proj_axis = {"x": cfg.window.x, "y": cfg.window.y, "z": cfg.window.z}
    proj_range = proj_axis[cfg.window.projection_axis]

    # CDDF for element
    element_cddf, element_bin_centers, element_bin_width = cd_2d.column_density_distribution_function(
        ion=None,
        log_column_density_range=None,
        n_bins=100,
        los_range=proj_range
    )

    plotter.plot_cddf_hist(
        cddf=element_cddf,
        bin_centers=element_bin_centers,
        bin_width=element_bin_width,
        element=cfg.chemistry.element,
        log_scale=True
    )

    # --- Ions ---
    for ion in cfg.chemistry.ion:
        n_ion_column_density = cd_2d.column_density_ion(ion=ion)

        # 2D plot
        plotter.plot_xy(
            column_density_values=n_ion_column_density.to("1/cm**2").value,
            ion=ion,
            log_scale=True
        )

        # CDDF
        ion_cddf, ion_bin_centers, ion_bin_width = cd_2d.column_density_distribution_function(
            ion=ion,
            log_column_density_range=None,
            n_bins=100,
            los_range=proj_range
        )

        plotter.plot_cddf_hist(
            cddf=ion_cddf,
            bin_centers=ion_bin_centers,
            bin_width=ion_bin_width,
            ion=ion,
            range_plot=None,
            log_scale=True
        )