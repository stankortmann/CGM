# cd_swift.py

from swiftsimio import load
import numpy as np
import unyt as u

# own modules
from spec_analysis import unpack_data
from spec_analysis import density_profiles
from spec_analysis import plot


def run_box_column_density(cfg):
    """
    Compute 2D column density and CDDFs for a simulation snapshot.
    
    Parameters
    ----------
    cfg : Config object
        The configuration object containing simulation, window, chemistry, etc.
    
    Returns
    -------
    None
    """
    
    # --- Unpack simulation ---
    data_unpacker = unpack_data.unwrapper(cfg)
    comoving_box_size = data_unpacker.box_size.to("Mpc")

    cfg.window.x = [x * comoving_box_size for x in cfg.window.x]
    cfg.window.y = [y * comoving_box_size for y in cfg.window.y]
    cfg.window.z = [z * comoving_box_size for z in cfg.window.z]

    # Projection axis length
    proj_axis = {"x": cfg.window.x, "y": cfg.window.y, "z": cfg.window.z}
    proj_range = proj_axis[cfg.window.projection_axis]

    # --- Load snapshot ---
    region = [cfg.window.x, cfg.window.y, cfg.window.z]
    snapshot = data_unpacker.load_snapshot(load_region=region)
    print("Gas particles are loaded")

    # --- Column density class ---
    cd_2d = density_profiles.column_density_2d_swift(
        cfg=cfg,
        data_unpacker=data_unpacker,
        snapshot=snapshot,
        element=cfg.chemistry.element
    )

    # --- Plot element column density ---
    n_element_column_density = cd_2d.element_column_density
    plotter = plot.column_density_plotter(
        x_edges=cd_2d.xedges,
        y_edges=cd_2d.yedges,
        length_unit="Mpc",
        data_unpacker=data_unpacker
    )

    plotter.plot_xy(
        column_density_values=n_element_column_density.to_physical().value,
        element=cfg.chemistry.element,
        log_scale=True
    )

    # --- CDDF for element ---
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

    # --- CDDFs for ions ---
    for ion in cfg.chemistry.ion:
        print("Calculating for ion", ion)
        n_ion_column_density = cd_2d.column_density_ion(ion=ion)

        # Plot 2D column density
        plotter.plot_xy(
            column_density_values=n_ion_column_density.to_physical().value,
            ion=ion,
            log_scale=True
        )

        # CDDF
        ion_cddf, ion_bin_centers, ion_bin_width = cd_2d.column_density_distribution_function(
            ion=ion,
            ion_column_density=n_ion_column_density,
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