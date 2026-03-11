from swiftsimio import load
from swiftsimio import SWIFTDataset
import swiftsimio as swift
from swiftsimio.visualisation.projection import project_gas
from swiftsimio.objects import cosmo_array, cosmo_factor, cosmo_quantity
import h5py 
import gc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import argparse
import yaml
import unyt as u
import pandas

#own modules
from spec_analysis import data_structure as ds
from spec_analysis import cosmology as cosmo
from spec_analysis import chemistry as chem
from spec_analysis import plot
from spec_analysis import unpack_data
from spec_analysis import density_profiles
from spec_analysis import galaxy_selection as gal_sel

if __name__ == "__main__":

      # --- Argument parser ---
    parser = argparse.ArgumentParser(description="Run CGM analysis")
    parser.add_argument(
        "--config",
        type=str,
        default="configurations/test.yaml",
        help="Path to the YAML configuration file"
    )
    args=parser.parse_args()
    # --- Load YAML config file ---
    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    # Create config object
    cfg = ds.Config(
        simulation=ds.Simulation(**cfg_dict['simulation']),
        data_output=ds.Data_output(**cfg_dict['data_output']),
        monitoring=ds.Monitoring(**cfg_dict['monitoring']),
        window=ds.Window(**cfg_dict['window']),
        chemistry=ds.Chemistry(**cfg_dict['chemistry']),
        galaxy=ds.Galaxy(**cfg_dict['galaxy'])
    )




#unpacking
data_unpacker = unpack_data.unwrapper(cfg)

#we now add an extra field to cfg.galaxy to get proper galaxy extend window
cfg.galaxy.extend= cosmo_quantity(cfg.galaxy.extend_value,
                                u.Unit(cfg.galaxy.extend_unit),
                                comoving=False,
                                scale_factor=data_unpacker.scale_factor,            
                                scale_exponent=1
                                )


single_galaxy=gal_sel.single_galaxy_new_new(cfg=cfg, data_unpacker=data_unpacker)
print("galaxy is selected")

#added the projection axis length for a single halo
halo_proj_axis={"x":single_galaxy.mask[0],
        "y":single_galaxy.mask[1],
        "z":single_galaxy.mask[2]}
halo_proj_range=halo_proj_axis[cfg.window.projection_axis]


# --- 2D COLUMN DENSITY ---  
cd_2d=density_profiles.column_density_2d_swift(
    cfg=cfg,
    filenames=data_unpacker,
    length_unit="Mpc",
    element=cfg.chemistry.element,
    halo=single_galaxy
)
print("Column density class is set up")
### --- ELEMENT --- ####
n_element_column_density=cd_2d.element_column_density

plotter = plot.column_density_plotter(x_edges=cd_2d.xedges, y_edges=cd_2d.yedges,
                                     length_unit="Mpc",
                                     data_unpacker=data_unpacker)

plotter.plot_xy(column_density_values=n_element_column_density.to("1/cm**2").value,
                element=cfg.chemistry.element,
                log_scale=True, 
                )


element_cddf,element_bin_centers,element_bin_width=cd_2d.column_density_distribution_function(ion=None,
                                                        log_column_density_range=None,
                                                        n_bins=100,
                                                        los_range=halo_proj_range
                                                        )

plotter.plot_cddf_hist(
                       cddf=element_cddf,
                       bin_centers=element_bin_centers,
                       bin_width=element_bin_width,
                       element=cfg.chemistry.element,
                       )
   



###----- IONS -----###
for ion in cfg.chemistry.ion:
    n_ion_column_density=cd_2d.column_density_ion(ion=ion)
    plotter.plot_xy(column_density_values=n_ion_column_density.to("1/cm**2").value,#it is already ensures that is is in the correct units
                    ion=ion,
                    log_scale=True, 
                    )
    

    ion_cddf,ion_bin_centers,ion_bin_width=cd_2d.column_density_distribution_function(
                                                            ion=ion,
                                                            ion_column_density=n_ion_column_density,
                                                            log_column_density_range=None, #if None it selects the complete range
                                                            n_bins=100,
                                                            los_range=halo_proj_range)

    plotter.plot_cddf_hist(
                        cddf=ion_cddf,
                        bin_centers=ion_bin_centers,
                        bin_width=ion_bin_width,
                        ion=ion,
                        range_plot=None, #range of the log bins
                        )
""" 
# --- TRANSVERSE DISTANCE COLUMN DENSITY --- 
cd_transverse=density_profiles.column_density_transverse(
    cfg=cfg,
    filenames=data_unpacker,
    comoving_box_size=comoving_box_size,
    length_unit="Mpc",
    gas_particles=gas_particles,
    element=cfg.chemistry.element,
    halo=single_galaxy
)
### --- ELEMENT --- ####


element_column_density=cd_transverse.radial_column_density_profile(
                                    ion=None, #so element is chosen
                                    r_max=100*u.kpc,
                                    n_bins=50,
                                    log_bins=False)

plotter.plot_transverse(
                column_density_values=element_column_density.to("1/cm**2").value,#it is already ensures that is is in the correct units
                r_centers=centers,
                r_widths=widths,
                element=cfg.chemistry.element,
                log_scale=False
                )
   



###----- IONS -----###
for ion in cfg.chemistry.ion:
    centers,widths,column_density=cd_transverse.radial_column_density_profile(
                                    ion=None, #so element is chosen
                                    r_max=100*u.kpc,
                                    n_bins=50,
                                    log_bins=False
                                    )
    plotter.plot_transverse(
                    column_density_values=column_density.to("1/cm**2").value,#it is already ensures that is is in the correct units
                    r_centers=centers,
                    r_widths=widths,
                    ion=ion,
                    log_scale=False
                    )
"""
exit()


