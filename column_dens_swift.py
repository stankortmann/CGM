from swiftsimio import load
from swiftsimio import SWIFTDataset
import swiftsimio as swift
from swiftsimio.visualisation.projection import project_gas
from swiftsimio.objects import cosmo_array, cosmo_factor
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

if __name__ == "__main__":

      # --- Argument parser ---
    parser = argparse.ArgumentParser(description="Run CGM analysis")
    parser.add_argument(
        "--config",
        type=str,
        default="configurations/test_box.yaml",
        help="Path to the YAML configuration file"
    )
    args=parser.parse_args()
    print(args)
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
cfg.window.x,cfg.window.y,cfg.window.z = [x*comoving_box_size for x in cfg.window.x], [y*comoving_box_size for y in cfg.window.y], [z*comoving_box_size for z in cfg.window.z]


dx=(cfg.window.x[1]-cfg.window.x[0])/cfg.window.resolution
dy=(cfg.window.y[1]-cfg.window.y[0])/cfg.window.resolution
dz=(cfg.window.z[1]-cfg.window.z[0])/cfg.window.resolution
#added the projection axis length for a full simulations slice
proj_axis={"x":cfg.window.x,
        "y":cfg.window.y,
        "z":cfg.window.z}
proj_range=proj_axis[cfg.window.projection_axis]



region = [
        cfg.window.x,
        cfg.window.y,
        cfg.window.z
    ]

snapshot = data_unpacker.load_snapshot(load_region=region)

gas_particles = snapshot.gas
print("Gas particles are loaded")
#init the 2d column density class with a certain element, after this we can 
#derive all the ions ass well
cd_2d=density_profiles.column_density_2d_swift(
    cfg=cfg,
    filenames=data_unpacker,
    snapshot=snapshot,
    length_unit="Mpc",
    element=cfg.chemistry.element
)
### --- ELEMENT --- ####
n_element_column_density=cd_2d.element_column_density

plotter = plot.column_density_plotter(x_edges=cd_2d.xedges.to_physical().value, 
                                    y_edges=cd_2d.yedges.to_physical().value,
                                    length_unit="Mpc",
                                    data_unpacker=data_unpacker)

plotter.plot_xy(column_density_values=n_element_column_density.to_physical().value,
                element=cfg.chemistry.element,
                log_scale=True, 
                )


element_cddf,element_bin_centers,element_bin_width=cd_2d.column_density_distribution_function(ion=None,
                                                        log_column_density_range=None,
                                                        n_bins=100,
                                                        los_range=proj_range)

plotter.plot_cddf_hist(
                       cddf=element_cddf,
                       bin_centers=element_bin_centers,
                       bin_width=element_bin_width,
                       element=cfg.chemistry.element,
                       log_scale=True
                       )
   



###----- IONS -----###
for ion in cfg.chemistry.ion:
    print("Calculating for ion",ion)
    n_ion_column_density=cd_2d.column_density_ion(ion=ion)
    plotter.plot_xy(column_density_values=n_ion_column_density.to_physical().value,#it is already ensures that is is in the correct units
                    ion=ion,
                    log_scale=True, 
                    )
    

    ion_cddf,ion_bin_centers,ion_bin_width=cd_2d.column_density_distribution_function(
                                                            ion=ion,
                                                            ion_column_density=n_ion_column_density, #avoids recomputation
                                                            log_column_density_range=None, #if None it selects the complete range
                                                            n_bins=100,
                                                            los_range=proj_range
                                                            )                                       
                                                            

    plotter.plot_cddf_hist(
                        cddf=ion_cddf,
                        bin_centers=ion_bin_centers,
                        bin_width=ion_bin_width,
                        ion=ion,
                        range_plot=None, #range of the log bins
                        log_scale=True
                        )
#exit the python environment
exit()







