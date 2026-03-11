# cd_swift.py

import unyt as u
from swiftsimio.objects import cosmo_array, cosmo_factor, cosmo_quantity
import numpy as np
import gc

from spec_analysis import density_profiles, plot, unpack_data, galaxy_selection as gal_sel


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

    # Select single galaxy/halo
    single_galaxy = gal_sel.single_galaxy_swift_galaxy(cfg=cfg, data_unpacker=data_unpacker)
    print("Galaxy is selected")

    # Projection axis range for CDDF
    halo_proj_axis = {
        "x": single_galaxy.mask[0],
        "y": single_galaxy.mask[1],
        "z": single_galaxy.mask[2]
    }
    halo_proj_range = halo_proj_axis[cfg.window.projection_axis]

    # --- 2D Column Density ---
    cd_2d = density_profiles.column_density_2d_swift(
        cfg=cfg,
        data_unpacker=data_unpacker,
        element=cfg.chemistry.element,
        halo=single_galaxy
    )
    print("2D column density class is set up")

    n_element_column_density = cd_2d.element_column_density

    # Plotter
    plotter = plot.column_density_plotter(
        x_edges=cd_2d.xedges,
        y_edges=cd_2d.yedges,
        length_unit="Mpc",
        data_unpacker=data_unpacker
    )

    # --- ELEMENT --- 2D
    plotter.plot_xy(
        column_density_values=n_element_column_density.to("1/cm**2").value,
        element=cfg.chemistry.element,
        log_scale=True
    )

    # --- ELEMENT CDDF ---
    element_cddf, element_bin_centers, element_bin_width = cd_2d.column_density_distribution_function(
        ion=None,
        log_column_density_range=None,
        n_bins=100,
        los_range=halo_proj_range
    )

    plotter.plot_cddf_hist(
        cddf=element_cddf,
        bin_centers=element_bin_centers,
        bin_width=element_bin_width,
        element=cfg.chemistry.element
    )

    # --- IONS ---
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
            ion_column_density=n_ion_column_density,
            log_column_density_range=None,
            n_bins=100,
            los_range=halo_proj_range
        )

        plotter.plot_cddf_hist(
            cddf=ion_cddf,
            bin_centers=ion_bin_centers,
            bin_width=ion_bin_width,
            ion=ion,
            range_plot=None
        )