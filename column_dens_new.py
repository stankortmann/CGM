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
from spec_analysis import column_density

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
        chemistry=ds.Chemistry(**cfg_dict['chemistry'])
    )





data_unpacker = unpack_data.unwrapper(cfg)
comoving_box_size = data_unpacker.box_size.to("Mpc")
cfg.window.x,cfg.window.y,cfg.window.z = [x*comoving_box_size for x in cfg.window.x], [y*comoving_box_size for y in cfg.window.y], [z*comoving_box_size for z in cfg.window.z]

dx=(cfg.window.x[1]-cfg.window.x[0])/cfg.window.resolution
dy=(cfg.window.y[1]-cfg.window.y[0])/cfg.window.resolution
dz=(cfg.window.z[1]-cfg.window.z[0])/cfg.window.resolution

region = [
        cfg.window.x,
        cfg.window.y,
        cfg.window.z
    ]

snapshot = data_unpacker.load_snapshot(load_region=region)

gas_particles = snapshot.gas

#init the 2d column density class
cd_2d=column_density.column_density_2d(
    cfg=cfg,
    filenames=data_unpacker,
    gas_particles=gas_particles
)
### --- ELEMENT HISTOGRAM ---
n_element_column_density=cd_2d.column_density_element(element=cfg.chemistry.element)

plotter = plot.column_density_plotter(x_edges=cd_2d.xedges, y_edges=cd_2d.yedges)

plotter.plot_xy(column_density_values=n_element_column_density, 
                column_density_unit=r"$n_{%s} [cm^{-2}]$" % cfg.chemistry.element,
                title="Column density of %s in x-y plane" % cfg.chemistry.element,
                log_scale=True, 
                output_path="test_colibre/column_density_%s.png" % cfg.chemistry.element)
print("Finished test_colibre/column_density_%s.png" % cfg.chemistry.element)

###----- ION HISTOGRAM -----###

n_ion_column_density=cd_2d.column_density_ion(ion=cfg.chemistry.ion)
plotter.plot_xy(column_density_values=n_ion_column_density, 
                column_density_unit=r"$n_{%s} [cm^{-2}]$" % cfg.chemistry.ion,
                title="Column density of %s in x-y plane" % cfg.chemistry.ion,
                log_scale=True, 
                output_path="test_colibre/column_density_%s.png" % cfg.chemistry.ion)
print("Finished test_colibre/column_density_%s.png" % cfg.chemistry.ion)



