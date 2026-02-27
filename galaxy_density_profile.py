from swiftsimio import load
from swiftsimio import SWIFTDataset
import swiftsimio as swift
from swiftsimio.visualisation.projection import project_gas
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





data_unpacker = unpack_data.unwrapper(cfg)
comoving_box_size = data_unpacker.box_size.to("Mpc")

single_galaxy=gal_sel.single_galaxy(cfg=cfg,
                                halo_properties=data_unpacker.load_halo_properties(), 
                                gas_in_halo_properties=data_unpacker.load_gas_in_halo_properties())




#init the 2d column density class
cd_2d=density_profiles.column_density_2d(
    cfg=cfg,
    filenames=data_unpacker,
    gas_particles=gas_particles,
    element=cfg.chemistry.element
)
### --- ELEMENT HISTOGRAM ---
n_element_column_density=cd_2d.n_element_column_density

plotter = plot.column_density_plotter(x_edges=cd_2d.xedges, y_edges=cd_2d.yedges)

plotter.plot_xy(column_density_values=n_element_column_density, 
                column_density_unit=r"$n_{%s} [cm^{-2}]$" % cfg.chemistry.element,
                title="Column density of %s in x-y plane" % cfg.chemistry.element,
                log_scale=True, 
                output_path="test_colibre/column_density_%s.png" % cfg.chemistry.element)
print("Finished test_colibre/column_density_%s.png" % cfg.chemistry.element)

###----- ION HISTOGRAM -----###

n_ion_column_density=cd_2d.column_density_ion(ion=cfg.chemistry.ion)
plotter.plot_xy(column_density_values=n_ion_column_density.to("1/cm**2").value,#it is already ensures that is is in the correct units
                column_density_unit=r"$n_{%s} [cm^{-2}]$" % cfg.chemistry.ion,
                title="Column density of %s in x-y plane" % cfg.chemistry.ion,
                log_scale=True, 
                output_path="test_colibre/column_density_%s.png" % cfg.chemistry.ion)
print("Finished test_colibre/column_density_%s.png" % cfg.chemistry.ion)

ion_cddf,ion_log_bins=cd_2d.column_density_distribution_function(ion=cfg.chemistry.ion,
                                                        log_column_density_range=None, #if None it selects the complete range
                                                        n_bins=100,
                                                        normalize=True)

plotter.plot_cddf_hist(
                       cddf=ion_cddf,
                       log_bins=ion_log_bins,
                       ion=cfg.chemistry.ion,
                       element=None,
                       normalize=True,
                       range_plot=[-2,2], #range of the log bins
                       output_path="test_colibre/cddf_%s.png"% cfg.chemistry.ion
                       )
print("Finished test_colibre/cddf_%s.png" % cfg.chemistry.ion)

"""
element_cddf,element_log_bins=cd_2d.column_density_distribution_function(ion=cfg.chemistry.element,
                                                        log_column_density_range=None,
                                                        n_bins=100,
                                                        normalize=True)

plotter.plot_cddf_hist(
                       cddf=ion_cddf,
                       log_bins=ion_log_bins,
                       ion=None,
                       element=cfg.chemistry.element,
                       normalize=True,
                       output_path="test_colibre/cddf_%s.png"% cfg.chemistry.element
                       )
print("Finished test_colibre/cddf_%s.png" % cfg.chemistry.element)    
"""



